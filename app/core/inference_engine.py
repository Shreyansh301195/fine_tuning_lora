"""
app/core/inference_engine.py
─────────────────────────────
Loads the fine-tuned Gemma + LoRA adapter and runs inference.
Supports both MLX and HuggingFace backends.
Uses lazy loading — model loads on first request.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

from loguru import logger

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Singleton model state
# ─────────────────────────────────────────────────────────────────────────────

_mlx_model = None
_mlx_tokenizer = None
_hf_model = None
_hf_tokenizer = None
_loaded_adapter: Optional[str] = None
_load_lock = threading.Lock()


def _find_latest_adapter() -> Optional[str]:
    """Return the most recently modified adapter directory."""
    adapter_dir = settings.adapter_dir
    if not adapter_dir.exists():
        return None
    candidates = sorted(
        [d for d in adapter_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


# ─────────────────────────────────────────────────────────────────────────────
# MLX Inference
# ─────────────────────────────────────────────────────────────────────────────

def _load_mlx_model(adapter_path: Optional[str]) -> Tuple[Any, Any]:
    global _mlx_model, _mlx_tokenizer, _loaded_adapter

    with _load_lock:
        if _mlx_model is not None and _loaded_adapter == adapter_path:
            return _mlx_model, _mlx_tokenizer

        from mlx_lm import load

        logger.info(f"Loading MLX model: {settings.base_model_id}")
        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            model, tokenizer = load(settings.base_model_id, adapter_path=adapter_path)
        else:
            logger.warning("No adapter found — running with base model only")
            model, tokenizer = load(settings.base_model_id)

        _mlx_model = model
        _mlx_tokenizer = tokenizer
        _loaded_adapter = adapter_path
        logger.success(f"MLX model loaded successfully")
        return model, tokenizer


def _infer_mlx(
    prompt: str,
    system_prompt: str,
    adapter_path: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> Tuple[str, int, float]:
    """Run MLX inference. Returns (response, tokens_generated, time_seconds)."""
    from mlx_lm import generate

    model, tokenizer = _load_mlx_model(adapter_path)

    # Format with Gemma chat template
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        verbose=False,
    )
    elapsed = time.time() - start

    # Strip the prompt from response if echoed back
    if response.startswith(formatted):
        response = response[len(formatted):]

    # Remove Gemma turn markers
    response = response.replace("<end_of_turn>", "").strip()

    tokens = len(tokenizer.encode(response))
    return response, tokens, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Inference (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _load_hf_model(adapter_path: Optional[str]) -> Tuple[Any, Any]:
    global _hf_model, _hf_tokenizer, _loaded_adapter

    with _load_lock:
        if _hf_model is not None and _loaded_adapter == adapter_path:
            return _hf_model, _hf_tokenizer

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading HF model: {settings.base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            settings.base_model_id, token=settings.hf_token or None
        )

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        base = AutoModelForCausalLM.from_pretrained(
            settings.base_model_id,
            torch_dtype=torch.float32,
            device_map=device,
            token=settings.hf_token or None,
        )

        if adapter_path and Path(adapter_path).exists():
            model = PeftModel.from_pretrained(base, adapter_path)
            logger.info(f"Loaded LoRA adapter from {adapter_path}")
        else:
            model = base
            logger.warning("Running base model (no adapter)")

        model.eval()
        _hf_model = model
        _hf_tokenizer = tokenizer
        _loaded_adapter = adapter_path
        return model, tokenizer


def _infer_hf(
    prompt: str,
    system_prompt: str,
    adapter_path: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> Tuple[str, int, float]:
    import torch

    model, tokenizer = _load_hf_model(adapter_path)
    messages = [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, len(new_tokens), elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(
    prompt: str,
    system_prompt: str = "You are a helpful UIDAI policy expert.",
    adapter_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    backend: str = "auto",
) -> Tuple[str, int, float]:
    """
    Generate a response from the fine-tuned model.
    Returns (response_text, tokens_generated, time_seconds).

    backend: "auto" uses settings.training_backend, or override with "mlx"/"hf"
    """
    if adapter_path is None:
        adapter_path = _find_latest_adapter()

    effective_backend = backend if backend != "auto" else settings.training_backend

    if effective_backend == "mlx":
        return _infer_mlx(
            prompt, system_prompt, adapter_path,
            max_tokens, temperature, top_p, repetition_penalty
        )
    else:
        return _infer_hf(
            prompt, system_prompt, adapter_path,
            max_tokens, temperature, top_p, repetition_penalty
        )


def unload_model():
    """Free GPU/memory by unloading the cached model."""
    global _mlx_model, _mlx_tokenizer, _hf_model, _hf_tokenizer, _loaded_adapter
    _mlx_model = None
    _mlx_tokenizer = None
    _hf_model = None
    _hf_tokenizer = None
    _loaded_adapter = None
    logger.info("Model unloaded from memory")


# Fix missing Any import
from typing import Any  # noqa: E402
