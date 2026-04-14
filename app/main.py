"""
app/main.py
────────────
FastAPI application entrypoint.

Registers all routers, configures CORS, adds health check,
and handles application lifespan (startup/shutdown).

Run with:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    logger.info("=" * 60)
    logger.info("  Gemma LoRA Fine-Tuning API  —  Starting up")
    logger.info(f"  Base model  : {settings.base_model_id}")
    logger.info(f"  Backend     : {settings.training_backend.upper()}")
    logger.info(f"  LoRA r      : {settings.lora_r}  |  alpha: {settings.lora_alpha}")
    logger.info("=" * 60)

    # Create required data directories on startup
    settings.ensure_dirs()
    logger.info("Data directories initialised")

    # Warn if HF token is missing
    if not settings.hf_token:
        logger.warning(
            "HF_TOKEN not set — Gemma is a gated model. "
            "Set HF_TOKEN in .env and accept license at "
            "https://huggingface.co/google/gemma-3-1b-it"
        )

    yield

    # Shutdown
    from app.core.inference_engine import unload_model
    unload_model()
    logger.info("Model unloaded — Goodbye!")


# ── App instantiation ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Gemma LoRA Fine-Tuning API",
    description=(
        "🦾 **Fine-tune Gemma locally on Apple Silicon using LoRA** with UIDAI policy documents.\n\n"
        "### Workflow\n"
        "1. `POST /api/v1/ingest` — Download & process UIDAI PDFs into a JSONL dataset\n"
        "2. `POST /api/v1/finetune/start` — Launch LoRA training job (MLX native)\n"
        "3. `GET /api/v1/finetune/stream` — Stream real-time training loss via SSE\n"
        "4. `GET /api/v1/evaluate` — Run full evaluation suite\n"
        "5. `POST /api/v1/inference` — Query the fine-tuned model\n\n"
        "### LoRA at a Glance\n"
        "LoRA freezes all ~1B base Gemma weights and trains only tiny A·B adapter matrices "
        "(~6M params, ≈0.6%) inserted into attention and FFN layers. "
        "Call `GET /api/v1/finetune/lora-explain` for full technical details."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS ───────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ────────────────────────────────────────────────────────────────────

from app.api.routes.ingest import router as ingest_router
from app.api.routes.finetune import router as finetune_router
from app.api.routes.evaluate import router as evaluate_router
from app.api.routes.inference import router as inference_router

API_PREFIX = "/api/v1"

app.include_router(ingest_router, prefix=API_PREFIX)
app.include_router(finetune_router, prefix=API_PREFIX)
app.include_router(evaluate_router, prefix=API_PREFIX)
app.include_router(inference_router, prefix=API_PREFIX)


# ── Health & Root ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Returns service health and current configuration."""
    from app.core.lora_trainer import get_training_state
    state = get_training_state()
    return {
        "status": "healthy",
        "model": settings.base_model_id,
        "backend": settings.training_backend,
        "lora": {"r": settings.lora_r, "alpha": settings.lora_alpha, "dropout": settings.lora_dropout},
        "training_status": state.status,
        "hf_token_set": bool(settings.hf_token),
        "llm_judge": {
            "provider": "Ollama (local)",
            "model": settings.judge_model,
            "url": settings.ollama_base_url,
        },
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Gemma LoRA Fine-Tuning API</title>
      <style>
        body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0;
               display: flex; flex-direction: column; align-items: center; 
               justify-content: center; min-height: 100vh; margin: 0; }
        .card { background: #1e2130; border: 1px solid #2d3148; border-radius: 16px;
                padding: 40px 60px; text-align: center; max-width: 480px; }
        h1 { font-size: 1.8rem; margin-bottom: 8px; 
             background: linear-gradient(135deg, #a78bfa, #60a5fa);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        p { color: #94a3b8; margin-bottom: 24px; }
        a { display: inline-block; background: linear-gradient(135deg, #7c3aed, #2563eb);
            color: white; text-decoration: none; padding: 12px 28px; border-radius: 8px;
            font-weight: 600; margin: 6px; transition: opacity 0.2s; }
        a:hover { opacity: 0.85; }
        .badge { background: #0d9488; font-size: 0.75rem; padding: 3px 10px; 
                 border-radius: 20px; margin-left: 8px; vertical-align: middle; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>🦾 Gemma LoRA API</h1>
        <p>Fine-tune Gemma locally on Apple Silicon M2<br>with UIDAI policy documents</p>
        <a href="/docs">📖 Swagger Docs</a>
        <a href="/redoc">📚 ReDoc</a>
        <a href="/health">💚 Health</a>
        <hr style="border-color:#2d3148; margin: 24px 0;">
        <p style="font-size:0.85rem; color:#64748b;">
          Backend: <strong style="color:#a78bfa">MLX (Apple Silicon)</strong>
          &nbsp;|&nbsp; Model: <strong style="color:#60a5fa">gemma-3-1b-it</strong>
        </p>
      </div>
    </body>
    </html>
    """
