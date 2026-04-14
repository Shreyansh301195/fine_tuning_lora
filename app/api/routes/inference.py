"""
app/api/routes/inference.py
────────────────────────────
Inference endpoints for the fine-tuned Gemma + LoRA model.

POST /api/v1/inference          — Single-shot Q&A
POST /api/v1/inference/stream   — SSE streaming generation
GET  /api/v1/inference/compare  — Compare base vs fine-tuned model
POST /api/v1/inference/unload   — Free model from memory
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.inference_engine import (
    _find_latest_adapter,
    generate_response,
    unload_model,
)
from app.models.schemas import InferenceRequest, InferenceResponse

router = APIRouter(prefix="/inference", tags=["Inference"])


@router.post("/", response_model=InferenceResponse, summary="Query the fine-tuned model")
async def run_inference(req: InferenceRequest) -> InferenceResponse:
    """
    Send a question about UIDAI policy to the fine-tuned Gemma model
    and receive a response.
    
    The model uses:
    - Frozen base Gemma weights
    - LoRA adapter A and B matrices merged at inference time
    - Gemma chat template format for instruction-following
    
    If no adapter_path is specified, uses the most recently trained adapter.
    """
    adapter = req.adapter_path or _find_latest_adapter()

    try:
        response_text, tokens, elapsed = generate_response(
            prompt=req.prompt,
            system_prompt=req.system_prompt,
            adapter_path=adapter,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return InferenceResponse(
        prompt=req.prompt,
        response=response_text,
        model=settings.base_model_id,
        adapter_used=adapter,
        tokens_generated=tokens,
        generation_time_seconds=round(elapsed, 3),
    )


@router.post("/stream", summary="Stream generation tokens via SSE")
async def stream_inference(req: InferenceRequest) -> EventSourceResponse:
    """
    Streams generated tokens as Server-Sent Events.
    Each event contains a token chunk from the model.
    
    Connect with:
      const source = new EventSource('/api/v1/inference/stream', { method: 'POST' })
    or:
      curl -N -X POST http://localhost:8000/api/v1/inference/stream -d '{...}'
    """

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            # For MLX — use streaming generation
            if settings.training_backend == "mlx":
                from mlx_lm import load, stream_generate

                adapter = req.adapter_path or _find_latest_adapter()
                from app.core.inference_engine import _load_mlx_model
                model, tokenizer = _load_mlx_model(adapter)

                messages = [{"role": "user", "content": f"{req.system_prompt}\n\n{req.prompt}"}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                token_count = 0
                for token_text in stream_generate(
                    model, tokenizer, formatted,
                    max_tokens=req.max_tokens,
                    temp=req.temperature,
                    top_p=req.top_p,
                ):
                    yield json.dumps({"token": token_text, "index": token_count})
                    token_count += 1
                    await asyncio.sleep(0)  # yield control to event loop

            else:
                # HF fallback: generate full response then stream token by token
                response, tokens, elapsed = generate_response(
                    prompt=req.prompt,
                    system_prompt=req.system_prompt,
                    adapter_path=req.adapter_path or _find_latest_adapter(),
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    backend="hf",
                )
                words = response.split()
                for i, word in enumerate(words):
                    yield json.dumps({"token": word + " ", "index": i})
                    await asyncio.sleep(0.02)

            yield json.dumps({"event": "done"})

        except Exception as e:
            logger.exception("Streaming inference failed")
            yield json.dumps({"event": "error", "detail": str(e)})

    return EventSourceResponse(token_generator())


@router.get("/compare", summary="Compare base model vs fine-tuned model")
async def compare_models(
    prompt: str = "What are the privacy obligations of Requesting Entities under UIDAI regulations?"
) -> dict:
    """
    Runs the same prompt through:
    1. The base Gemma model (no adapter)
    2. The fine-tuned model (with LoRA adapter)
    
    Useful for qualitatively assessing the impact of fine-tuning.
    """
    # Base model (no adapter)
    try:
        base_response, base_tokens, base_time = generate_response(
            prompt=prompt,
            adapter_path=None,
            max_tokens=256,
            temperature=0.3,
        )
    except Exception as e:
        base_response = f"[Error: {e}]"
        base_tokens, base_time = 0, 0.0

    # Fine-tuned model
    adapter = _find_latest_adapter()
    try:
        ft_response, ft_tokens, ft_time = generate_response(
            prompt=prompt,
            adapter_path=adapter,
            max_tokens=256,
            temperature=0.3,
        )
    except Exception as e:
        ft_response = f"[Error: {e}]"
        ft_tokens, ft_time = 0, 0.0

    return {
        "prompt": prompt,
        "base_model": {
            "model": settings.base_model_id,
            "adapter": None,
            "response": base_response,
            "tokens_generated": base_tokens,
            "time_seconds": round(base_time, 3),
        },
        "fine_tuned_model": {
            "model": settings.base_model_id,
            "adapter": adapter,
            "response": ft_response,
            "tokens_generated": ft_tokens,
            "time_seconds": round(ft_time, 3),
        },
    }


@router.post("/unload", summary="Unload model from memory")
async def unload_model_endpoint() -> dict:
    """Free unified memory by unloading the cached model and tokenizer."""
    unload_model()
    return {"message": "Model unloaded from memory successfully"}
