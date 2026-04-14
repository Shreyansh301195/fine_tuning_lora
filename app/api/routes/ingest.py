"""
app/api/routes/ingest.py
─────────────────────────
Endpoints for downloading and processing UIDAI policy documents.

POST /api/v1/ingest          — Trigger full ingestion pipeline
GET  /api/v1/ingest/status   — Check current dataset stats
GET  /api/v1/ingest/samples  — Preview a few training samples
DELETE /api/v1/ingest/reset  — Clear all downloaded data and datasets
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from app.core.config import settings
from app.core.document_processor import run_ingestion_pipeline
from app.models.schemas import IngestRequest, IngestStatusResponse

router = APIRouter(prefix="/ingest", tags=["Ingest"])

# Ingestion state
_ingest_state = {
    "status": "idle",          # idle | running | completed | failed
    "docs_downloaded": 0,
    "chunks_created": 0,
    "train_samples": 0,
    "valid_samples": 0,
    "message": "No ingestion run yet",
    "errors": [],
    "log": [],
}
_ingest_lock = threading.Lock()


def _progress_callback(msg: str):
    with _ingest_lock:
        _ingest_state["log"].append(msg)
        _ingest_state["message"] = msg


def _run_ingestion(req: IngestRequest):
    global _ingest_state
    with _ingest_lock:
        _ingest_state["status"] = "running"
        _ingest_state["log"] = []

    try:
        result = run_ingestion_pipeline(
            max_docs=req.max_docs,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            train_split=req.train_split,
            progress_callback=_progress_callback,
        )
        with _ingest_lock:
            _ingest_state.update(
                {
                    "status": "completed",
                    "docs_downloaded": result.docs_downloaded,
                    "chunks_created": result.chunks_created,
                    "train_samples": result.train_samples,
                    "valid_samples": result.valid_samples,
                    "errors": result.errors,
                    "message": (
                        f"Ingestion complete: {result.train_samples} train / "
                        f"{result.valid_samples} valid samples"
                    ),
                }
            )
    except Exception as e:
        logger.exception("Ingestion pipeline failed")
        with _ingest_lock:
            _ingest_state["status"] = "failed"
            _ingest_state["message"] = f"Ingestion failed: {e}"


@router.post("", summary="Start UIDAI document ingestion")
async def start_ingestion(req: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Download UIDAI policy PDFs, parse them, chunk into instruction-response pairs,
    and write train/valid JSONL files for fine-tuning.

    This runs asynchronously. Poll GET /ingest/status for progress.
    """
    with _ingest_lock:
        if _ingest_state["status"] == "running":
            raise HTTPException(status_code=409, detail="Ingestion already in progress")

    background_tasks.add_task(_run_ingestion, req)
    return {
        "message": "Ingestion started in background",
        "config": req.model_dump(),
        "status_endpoint": "/api/v1/ingest/status",
    }


@router.get("/status", response_model=IngestStatusResponse, summary="Get ingestion status")
async def get_ingest_status() -> IngestStatusResponse:
    """Returns current ingestion progress and dataset statistics."""
    with _ingest_lock:
        state = dict(_ingest_state)
    return IngestStatusResponse(
        status=state["status"],
        docs_downloaded=state["docs_downloaded"],
        chunks_created=state["chunks_created"],
        train_samples=state["train_samples"],
        valid_samples=state["valid_samples"],
        message=state["message"],
    )


@router.get("/samples", summary="Preview training samples")
async def preview_samples(n: int = 5) -> dict:
    """
    Return the first N samples from train.jsonl for inspection.
    Useful for verifying dataset format before training.
    """
    train_file = settings.train_jsonl
    if not train_file.exists():
        raise HTTPException(status_code=404, detail="No training data found. Run /ingest first.")

    samples = []
    with open(train_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line.strip()))

    return {"file": str(train_file), "total_preview": len(samples), "samples": samples}


@router.delete("/reset", summary="Clear all ingested data")
async def reset_ingestion() -> dict:
    """
    Delete all downloaded PDFs, processed text, and JSONL datasets.
    Use with caution — this cannot be undone.
    """
    import shutil

    settings.ensure_dirs()
    dirs_cleared = []

    for d in [settings.raw_data_dir, settings.processed_data_dir, settings.train_data_dir]:
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            dirs_cleared.append(str(d))

    with _ingest_lock:
        _ingest_state.update(
            {
                "status": "idle",
                "docs_downloaded": 0,
                "chunks_created": 0,
                "train_samples": 0,
                "valid_samples": 0,
                "message": "Data reset. Run /ingest to start fresh.",
                "errors": [],
                "log": [],
            }
        )

    return {"message": "Data cleared successfully", "directories_reset": dirs_cleared}
