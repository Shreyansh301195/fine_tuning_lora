"""
app/core/config.py
──────────────────
Centralised settings loaded from .env via pydantic-settings.
Import `settings` anywhere in the app.

Root cause of TARGET_MODULES error:
  pydantic-settings v2 attempts to JSON-decode any List[str] field read
  from the env file BEFORE field_validators run. A bare comma-separated
  string like "q_proj,k_proj,..." is not valid JSON, so it crashes.

Fix:
  - TARGET_MODULES is stored as a plain `str` field (_target_modules_str)
    so pydantic-settings reads it without any JSON decoding attempt.
  - `settings.target_modules` is exposed as a @computed_field that parses
    the string flexibly: accepts both comma-separated AND JSON array formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Disable automatic JSON-decoding of env values for complex types.
        # We handle List[str] parsing manually via computed_field below.
    )

    # ── HuggingFace ──────────────────────────────────────────
    hf_token: str = Field(default="", description="HuggingFace access token for gated models")

    # ── Model ─────────────────────────────────────────────────
    base_model_id: str = Field(
        default="google/gemma-3-1b-it",
        description="HuggingFace model ID or local path",
    )

    # ── LoRA Hyperparameters ──────────────────────────────────
    lora_r: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)

    # Stored as plain str to avoid pydantic-settings JSON-decoding List[str]
    # from the .env file.  Accepts either format in .env:
    #   TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj   ← comma-separated
    #   TARGET_MODULES=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]  ← JSON
    target_modules_str: str = Field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        validation_alias="target_modules",  # reads TARGET_MODULES from .env
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_modules(self) -> List[str]:
        """
        Parse TARGET_MODULES from .env into a Python list.
        Supports both comma-separated and JSON array formats.
        """
        raw = self.target_modules_str.strip()
        if raw.startswith("["):
            # JSON array: ["q_proj","k_proj",...]
            return json.loads(raw)
        # Comma-separated: q_proj,k_proj,...
        return [m.strip() for m in raw.split(",") if m.strip()]

    # ── Training Hyperparameters ──────────────────────────────
    batch_size: int = Field(default=4, ge=1)
    grad_accum_steps: int = Field(default=4, ge=1)
    num_epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=3e-4)
    max_seq_len: int = Field(default=512, ge=64)
    warmup_steps: int = Field(default=50, ge=0)
    eval_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=100, ge=1)

    # ── Backend ───────────────────────────────────────────────
    training_backend: str = Field(default="mlx", pattern="^(mlx|hf)$")

    # ── LLM-as-a-Judge (Ollama — local, no API key needed) ──────
    # Install Ollama: https://ollama.com  →  brew install ollama
    # Pull a judge model: ollama pull llama3.2:3b
    # Ollama exposes an OpenAI-compatible API at localhost:11434
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama OpenAI-compatible endpoint",
    )
    judge_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model tag used for LLM-as-a-Judge scoring",
    )
    judge_max_samples: int = Field(default=10, ge=1)

    # ── Paths ─────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("./data"))
    adapter_dir: Path = Field(default=Path("./adapters"))
    log_dir: Path = Field(default=Path("./logs"))

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def train_data_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def train_jsonl(self) -> Path:
        return self.train_data_dir / "train.jsonl"

    @property
    def valid_jsonl(self) -> Path:
        return self.train_data_dir / "valid.jsonl"

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.train_data_dir,
            self.adapter_dir,
            self.log_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
