"""
app/models/schemas.py
─────────────────────
Pydantic v2 request / response schemas for all endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Shared Enums ──────────────────────────────────────────────────────────────

class TrainingBackend(str, Enum):
    mlx = "mlx"
    hf = "hf"


class TrainingStatus(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    failed = "failed"
    stopped = "stopped"


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    max_docs: int = Field(default=20, ge=1, le=100, description="Max UIDAI documents to download")
    chunk_size: int = Field(default=512, ge=64, description="Token chunk size for splitting docs")
    chunk_overlap: int = Field(default=64, ge=0, description="Overlap tokens between chunks")
    train_split: float = Field(default=0.8, ge=0.5, le=0.95, description="Fraction for train set")


class IngestStatusResponse(BaseModel):
    status: str
    docs_downloaded: int
    chunks_created: int
    train_samples: int
    valid_samples: int
    message: str


# ── Fine-Tuning ───────────────────────────────────────────────────────────────

class FinetuneRequest(BaseModel):
    base_model: str = Field(default="google/gemma-3-1b-it")
    backend: TrainingBackend = Field(default=TrainingBackend.mlx)
    # LoRA hyperparameters
    lora_r: int = Field(default=8, ge=1, le=256, description="LoRA rank r — controls adapter capacity")
    lora_alpha: int = Field(default=16, ge=1, description="LoRA alpha — scales adapter updates by alpha/r")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="Dropout on adapter activations")
    target_modules: List[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Weight matrices to attach LoRA adapters to"
    )
    # Training hyperparameters
    num_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=4, ge=1)
    grad_accum_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    max_seq_len: int = Field(default=512, ge=64)
    warmup_steps: int = Field(default=50, ge=0)
    eval_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=100, ge=1)
    run_name: Optional[str] = Field(default=None, description="Optional name for this training run")


class TrainingProgressEvent(BaseModel):
    event: str = "progress"
    step: int
    total_steps: int
    epoch: float
    loss: float
    val_loss: Optional[float] = None
    learning_rate: float
    tokens_per_sec: Optional[float] = None
    eta_seconds: Optional[int] = None


class FinetuneStatusResponse(BaseModel):
    status: TrainingStatus
    run_name: Optional[str]
    current_step: int
    total_steps: int
    current_loss: Optional[float]
    best_val_loss: Optional[float]
    adapter_path: Optional[str]
    message: str


class FinetuneArtifactsResponse(BaseModel):
    adapters: List[Dict[str, Any]]


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    adapter_path: Optional[str] = Field(
        default=None, description="Path to adapter; uses latest if None"
    )
    num_samples: int = Field(default=50, ge=1, description="Number of validation samples to evaluate")
    use_llm_judge: bool = Field(default=True, description="Run LLM-as-a-judge for faithfulness/relevance")
    reference_answers: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional list of {question, reference} for retrieval metrics"
    )


class EvalMetrics(BaseModel):
    # ── Classical NLP ──────────────────────────────────────────
    val_loss: float
    perplexity: float
    bleu_1: float
    bleu_4: float
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float

    # ── Semantic Faithfulness ──────────────────────────────────
    faithfulness_score: float = Field(description="0-1: factual consistency with source documents")
    answer_relevance_score: float = Field(description="0-1: how relevant the answer is to the question")

    # ── LLM-as-a-Judge ────────────────────────────────────────
    llm_judge_score: Optional[float] = Field(
        default=None, description="0-10: GPT/local model quality rating"
    )
    llm_judge_reasoning: Optional[str] = None

    # ── Ranking Metrics (retrieval quality) ───────────────────
    ndcg_at_5: Optional[float] = Field(default=None, description="nDCG@5 for retrieved context ranking")
    ndcg_at_10: Optional[float] = Field(default=None, description="nDCG@10")
    map_at_5: Optional[float] = Field(default=None, description="MAP@5")
    map_at_10: Optional[float] = Field(default=None, description="MAP@10")

    # ── Metadata ──────────────────────────────────────────────
    num_samples_evaluated: int
    adapter_path: Optional[str]
    evaluation_duration_seconds: float


class EvalHistoryResponse(BaseModel):
    steps: List[int]
    train_losses: List[float]
    val_losses: List[float]
    perplexities: List[float]


# ── Inference ─────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str = Field(description="User question or instruction")
    system_prompt: str = Field(
        default="You are a helpful assistant specialized in UIDAI Aadhaar policy and regulations. "
                "Answer accurately based on official UIDAI documentation.",
    )
    adapter_path: Optional[str] = Field(default=None, description="Adapter path; uses latest if None")
    max_tokens: int = Field(default=512, ge=16, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class InferenceResponse(BaseModel):
    prompt: str
    response: str
    model: str
    adapter_used: Optional[str]
    tokens_generated: int
    generation_time_seconds: float
