"""
app/api/routes/finetune.py
───────────────────────────
Fine-tuning management endpoints.

POST /api/v1/finetune/start         — Launch training job
GET  /api/v1/finetune/status        — Current training state (JSON)
GET  /api/v1/finetune/stream        — SSE real-time progress stream
POST /api/v1/finetune/stop          — Gracefully stop training
GET  /api/v1/finetune/artifacts     — List saved adapter checkpoints
GET  /api/v1/finetune/lora-explain  — Explain LoRA math for this config
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.lora_trainer import (
    get_training_state,
    request_stop,
    start_training,
)
from app.models.schemas import (
    FinetuneArtifactsResponse,
    FinetuneRequest,
    FinetuneStatusResponse,
    TrainingStatus as StatusEnum,
)

router = APIRouter(prefix="/finetune", tags=["Fine-Tuning"])

# SSE event queue — filled by training callbacks
_event_queue: asyncio.Queue = None


def _get_queue() -> asyncio.Queue:
    global _event_queue
    if _event_queue is None:
        _event_queue = asyncio.Queue(maxsize=500)
    return _event_queue


def _on_progress(event: dict):
    """Called by training thread to push events to SSE queue."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(_get_queue().put_nowait, json.dumps(event))
    except Exception as e:
        logger.debug(f"SSE queue push failed (no listener): {e}")


@router.post("/start", summary="Start LoRA fine-tuning job")
async def start_finetuning(req: FinetuneRequest) -> dict:
    """
    Launch a LoRA fine-tuning job in the background.
    
    LoRA injects small trainable matrices A and B into the target_modules.
    Only these adapters (≈0.6% of total params for Gemma-1B, r=8) are trained.
    Base model weights are completely frozen.
    
    Monitor progress via GET /finetune/stream (SSE) or GET /finetune/status.
    """
    state = get_training_state()
    if state.status == "running":
        raise HTTPException(status_code=409, detail="A training job is already running.")

    # Validate dataset exists
    if not settings.train_jsonl.exists():
        raise HTTPException(
            status_code=400,
            detail="No training data found. Run POST /ingest first."
        )

    run_name = req.run_name or f"gemma-lora-uidai-{int(time.time())}"

    config = {
        "base_model": req.base_model,
        "run_name": run_name,
        "backend": req.backend.value,
        "lora_r": req.lora_r,
        "lora_alpha": req.lora_alpha,
        "lora_dropout": req.lora_dropout,
        "target_modules": req.target_modules,
        "num_epochs": req.num_epochs,
        "batch_size": req.batch_size,
        "grad_accum_steps": req.grad_accum_steps,
        "learning_rate": req.learning_rate,
        "max_seq_len": req.max_seq_len,
        "warmup_steps": req.warmup_steps,
        "eval_steps": req.eval_steps,
        "save_steps": req.save_steps,
    }

    # Route to correct backend
    if req.backend.value == "mlx":
        start_training(config, on_progress=_on_progress)
    else:
        from app.core.hf_trainer import HFLoRATrainer
        import threading

        trainer = HFLoRATrainer(config)
        t = threading.Thread(
            target=trainer.train,
            kwargs={"on_progress": _on_progress},
            daemon=True,
            name="hf-lora-trainer",
        )
        t.start()

    return {
        "message": f"Training started — run_name: {run_name}",
        "run_name": run_name,
        "config": config,
        "trainable_param_estimate": _estimate_trainable_params(req),
        "stream_endpoint": "/api/v1/finetune/stream",
        "status_endpoint": "/api/v1/finetune/status",
    }


def _estimate_trainable_params(req: FinetuneRequest) -> dict:
    """
    Estimate trainable LoRA parameter count.
    
    For each target_module in a Gemma-1B layer:
      attention: q_proj(2048×2048), k_proj(256×2048), v_proj(256×2048), o_proj(2048×2048)
      FFN: gate_proj(8192×2048), up_proj(8192×2048), down_proj(2048×8192)
    
    LoRA params per matrix = r × (d + k)
    """
    # Approximate Gemma-1B dimensions
    module_dims = {
        "q_proj": (2048, 2048), "k_proj": (256, 2048),
        "v_proj": (256, 2048), "o_proj": (2048, 2048),
        "gate_proj": (8192, 2048), "up_proj": (8192, 2048), "down_proj": (2048, 8192),
    }
    num_layers = 18  # Gemma-1B has 18 transformer layers
    r = req.lora_r

    total_lora = 0
    for mod in req.target_modules:
        if mod in module_dims:
            d, k = module_dims[mod]
            total_lora += r * d + r * k  # A: r×k, B: d×r

    total_lora *= num_layers
    base_params = 1_000_000_000
    pct = 100 * total_lora / base_params

    return {
        "lora_trainable_params": f"{total_lora:,}",
        "base_model_params": f"{base_params:,}",
        "percentage_trained": f"{pct:.2f}%",
        "rank_r": r,
        "alpha": req.lora_alpha,
        "scale_factor": f"{req.lora_alpha / req.lora_r:.1f}×",
    }


@router.get("/status", response_model=FinetuneStatusResponse, summary="Get training status")
async def get_status() -> FinetuneStatusResponse:
    """Returns the current training state including latest loss and adapter path."""
    state = get_training_state()
    return FinetuneStatusResponse(
        status=StatusEnum(state.status),
        run_name=state.run_name or None,
        current_step=state.current_step,
        total_steps=state.total_steps,
        current_loss=state.current_loss,
        best_val_loss=state.best_val_loss,
        adapter_path=state.adapter_path,
        message=state.message,
    )


@router.get("/stream", summary="Stream training progress via SSE")
async def stream_training_progress(request) -> EventSourceResponse:
    """
    Server-Sent Events stream of training progress.
    
    Each event is a JSON object:
    {
      "event": "progress",
      "step": 10,
      "loss": 2.456,
      "val_loss": 2.567,    // only on eval steps
      "learning_rate": 3e-4,
      "tokens_per_sec": 1.23
    }
    
    Connect with:
      curl -N http://localhost:8000/api/v1/finetune/stream
      or EventSource in JavaScript
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        queue = _get_queue()
        while True:
            state = get_training_state()

            # Check if training finished
            if state.status in ("completed", "failed", "stopped"):
                yield json.dumps({
                    "event": "finished",
                    "status": state.status,
                    "message": state.message,
                    "best_val_loss": state.best_val_loss,
                    "adapter_path": state.adapter_path,
                })
                break

            # Drain available events
            try:
                event = await asyncio.wait_for(queue.get(), timeout=2.0)
                yield event
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield json.dumps({"event": "heartbeat", "step": state.current_step})

    return EventSourceResponse(event_generator())


@router.post("/stop", summary="Stop training gracefully")
async def stop_training() -> dict:
    """
    Signals the training loop to stop after the current step.
    Adapter weights up to the last saved checkpoint are preserved.
    """
    state = get_training_state()
    if state.status != "running":
        raise HTTPException(status_code=400, detail=f"No training is running (status: {state.status})")

    request_stop()
    return {
        "message": "Stop signal sent. Training will halt after current step.",
        "last_saved_step": state.current_step,
        "adapter_path": state.adapter_path,
    }


@router.get("/artifacts", response_model=FinetuneArtifactsResponse, summary="List saved adapters")
async def list_artifacts() -> FinetuneArtifactsResponse:
    """Lists all saved LoRA adapter checkpoints with their metadata."""
    adapter_dir = settings.adapter_dir
    if not adapter_dir.exists():
        return FinetuneArtifactsResponse(adapters=[])

    adapters = []
    for d in sorted(adapter_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        files = list(d.iterdir())
        adapters.append({
            "name": d.name,
            "path": str(d),
            "size_mb": round(sum(f.stat().st_size for f in files if f.is_file()) / 1e6, 2),
            "files": [f.name for f in files],
            "created_at": d.stat().st_mtime,
        })

    return FinetuneArtifactsResponse(adapters=adapters)


@router.get("/lora-explain", summary="Explain LoRA mechanics for current config")
async def explain_lora(lora_r: int = 8, lora_alpha: int = 16) -> dict:
    """
    Returns a detailed technical explanation of how LoRA works
    for the given r and alpha values, with Gemma-specific parameter counts.
    """
    scale = lora_alpha / lora_r
    # For Gemma-1B q_proj (2048×2048) as example
    d, k = 2048, 2048
    lora_params = lora_r * d + lora_r * k
    base_params = d * k
    compression = base_params / lora_params

    return {
        "lora_overview": {
            "core_idea": (
                "LoRA (Low-Rank Adaptation) keeps the pre-trained Gemma weights FROZEN. "
                "Instead of updating W₀ (the huge weight matrix), it trains two small "
                "matrices A and B such that ΔW = B·A, where rank(ΔW) = r << d,k."
            ),
            "forward_pass_formula": "output = W₀·x + (α/r) · B·(A·x)",
            "initialization": (
                "A is initialized with Gaussian random noise. "
                "B is initialized to ZERO — ensuring the adapter starts as an identity "
                "(no distortion at step 0)."
            ),
        },
        "your_config": {
            "rank_r": lora_r,
            "alpha": lora_alpha,
            "effective_scale": f"α/r = {scale:.2f}×",
            "interpretation": (
                f"With r={lora_r} and α={lora_alpha}, each LoRA update is scaled by "
                f"{scale:.1f}×. This means adapter contributions are "
                f"{'amplified' if scale > 1 else 'dampened'} relative to rank-1 updates."
            ),
        },
        "parameter_counts_example": {
            "layer": "q_proj (2048×2048)",
            "base_params": f"{base_params:,}",
            "lora_params": f"{lora_params:,}  (A: {lora_r}×{k} + B: {d}×{lora_r})",
            "compression_ratio": f"{compression:.0f}×  fewer parameters",
        },
        "which_weights_change": {
            "frozen": [
                "All base Gemma weights (embeddings, attention, FFN, layernorm)",
            ],
            "trained_only": [
                "lora_A matrices (shape: r × k) — initialized random",
                "lora_B matrices (shape: d × r) — initialized zero",
            ],
            "target_modules": [
                "q_proj, k_proj, v_proj, o_proj (attention)",
                "gate_proj, up_proj, down_proj (FFN / MLP)",
            ],
        },
        "loss_function": {
            "type": "Cross-Entropy Loss (next-token prediction)",
            "formula": "L = -Σ log P(token_t | token_1 ... token_{t-1})",
            "completion_masking": (
                "Loss is ONLY computed on the model response tokens. "
                "The user instruction tokens are masked (label = -100), "
                "preventing the model from overfitting to prompt patterns."
            ),
            "optimizer": "AdamW with cosine LR schedule",
        },
    }
