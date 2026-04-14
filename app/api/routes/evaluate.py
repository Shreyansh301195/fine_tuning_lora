"""
app/api/routes/evaluate.py
───────────────────────────
Evaluation endpoints.

GET  /api/v1/evaluate             — Run full evaluation suite
GET  /api/v1/evaluate/history     — Return training/val loss history
GET  /api/v1/evaluate/metrics     — Get last computed metrics
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from app.core.evaluator import get_loss_history, run_evaluation
from app.core.inference_engine import generate_response, _find_latest_adapter
from app.models.schemas import EvalHistoryResponse, EvalMetrics, EvalRequest

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

_last_eval_result: Optional[Dict[str, Any]] = None
_eval_running: bool = False


def _make_generate_fn(adapter_path: Optional[str]):
    """Create a generate_fn closure bound to a specific adapter."""
    def generate_fn(question: str) -> str:
        try:
            response, _, _ = generate_response(
                prompt=question,
                adapter_path=adapter_path,
                max_tokens=256,
                temperature=0.0,   # deterministic for evaluation
            )
            return response
        except Exception as e:
            logger.warning(f"Generation failed during eval: {e}")
            return ""
    return generate_fn


@router.get("/", summary="Run full evaluation suite")
async def run_full_evaluation(
    background_tasks: BackgroundTasks,
    num_samples: int = 50,
    use_llm_judge: bool = True,
    async_mode: bool = False,
) -> dict:
    """
    Run the complete evaluation pipeline:
    
    **Classical NLP:**
    - Validation Loss (cross-entropy)
    - Perplexity (exp of val_loss)
    - BLEU-1, BLEU-4 (sacrebleu)
    - ROUGE-1, ROUGE-2, ROUGE-L
    
    **Semantic Metrics:**
    - Faithfulness Score — how factually consistent are answers with source docs (cosine sim)
    - Answer Relevance — does the answer address the question (cosine sim)
    
    **LLM-as-a-Judge:**
    - GPT-4o-mini rates each answer on accuracy, completeness, policy alignment, clarity (0-10)
    - Falls back to None if OPENAI_API_KEY not set
    
    **Ranking Metrics:**
    - nDCG@5, nDCG@10 — normalised discounted cumulative gain
    - MAP@5, MAP@10 — mean average precision
    
    Set `async_mode=true` to run in background and poll GET /evaluate/metrics.
    """
    global _eval_running, _last_eval_result

    if _eval_running:
        raise HTTPException(status_code=409, detail="Evaluation is already running")

    adapter_path = _find_latest_adapter()
    if adapter_path is None:
        raise HTTPException(
            status_code=400,
            detail="No adapter found. Train the model first via POST /finetune/start"
        )

    generate_fn = _make_generate_fn(adapter_path)

    if async_mode:
        def _run_eval():
            global _eval_running, _last_eval_result
            _eval_running = True
            try:
                result = run_evaluation(
                    generate_fn=generate_fn,
                    num_samples=num_samples,
                    use_llm_judge=use_llm_judge,
                )
                _last_eval_result = result
            finally:
                _eval_running = False

        background_tasks.add_task(_run_eval)
        return {"message": "Evaluation started in background", "poll_at": "/api/v1/evaluate/metrics"}

    # Synchronous mode
    _eval_running = True
    try:
        result = run_evaluation(
            generate_fn=generate_fn,
            num_samples=num_samples,
            use_llm_judge=use_llm_judge,
        )
        _last_eval_result = result
    finally:
        _eval_running = False

    return result


@router.get("/history", response_model=EvalHistoryResponse, summary="Training loss history")
async def get_eval_history() -> EvalHistoryResponse:
    """
    Returns the full training loss and validation loss history
    captured during the fine-tuning run. Use this to plot loss curves.
    """
    history = get_loss_history()
    return EvalHistoryResponse(
        steps=history["steps"],
        train_losses=history["train_losses"],
        val_losses=[v for v in history["val_losses"] if v is not None],
        perplexities=[p for p in history["perplexities"] if p is not None],
    )


@router.get("/metrics", summary="Last computed evaluation metrics")
async def get_last_metrics() -> dict:
    """
    Returns the most recently computed evaluation metrics,
    or a 404 if no evaluation has been run yet.
    """
    if _last_eval_result is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation has been run yet. Call GET /evaluate first."
        )
    return {"running": _eval_running, "results": _last_eval_result}
