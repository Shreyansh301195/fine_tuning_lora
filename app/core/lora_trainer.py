"""
app/core/lora_trainer.py
─────────────────────────
MLX-native LoRA fine-tuning for Apple Silicon (M2).

Uses mlx-lm's Python API to:
  - Load the Gemma model in MLX format
  - Inject LoRA adapters into attention + FFN projections
  - Run the training loop with streaming loss logs
  - Save adapter weights to disk
  - Support graceful stops via a threading.Event

How LoRA works here (brief recap):
  For each target weight matrix W₀ ∈ ℝ^(d×k):
    - W₀ is frozen (no gradients)
    - We add: ΔW = B·A  where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << d,k
    - B is zero-initialized, A is random-normal initialized
    - Forward pass: output = W₀x + (α/r)·B·(A·x)
    - Only A and B are updated by the optimizer
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Training State (shared across threads)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingState:
    status: str = "idle"           # idle | running | completed | failed | stopped
    run_name: str = ""
    current_step: int = 0
    total_steps: int = 0
    current_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    adapter_path: Optional[str] = None
    message: str = ""
    loss_history: List[dict] = None

    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []


# Global singleton state
_training_state = TrainingState()
_stop_event = threading.Event()
_training_thread: Optional[threading.Thread] = None


def get_training_state() -> TrainingState:
    return _training_state


def request_stop():
    global _stop_event
    _stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# MLX LoRA Trainer
# ─────────────────────────────────────────────────────────────────────────────

class MLXLoRATrainer:
    """
    Wraps mlx_lm.lora training via subprocess for clean process management
    and real-time log streaming to the FastAPI SSE endpoint.
    """

    def __init__(self, config: dict):
        self.config = config
        self.process: Optional[subprocess.Popen] = None

    def _write_yaml_config(self, adapter_path: str) -> Path:
        """
        Write a YAML config file for mlx_lm lora.

        mlx-lm 0.18+ reads LoRA parameters (rank, scale, dropout, target keys)
        from a config YAML rather than CLI flags. This avoids all 'unrecognized
        argument' errors and keeps the invocation clean.

        Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/lora.py
        """
        import yaml  # bundled with PyYAML (already installed via transformers deps)

        cfg = self.config

        # mlx-lm uses 'scale' = lora_alpha (it applies scale/r internally)
        lora_scale = float(cfg.get("lora_alpha", 16))

        yaml_config = {
            # ── Model & data ───────────────────────────────────────────
            "model": cfg["base_model"],
            "train": True,
            "data": str(settings.train_data_dir),
            "adapter_path": adapter_path,

            # ── Training hyperparameters ───────────────────────────────
            "batch_size": cfg["batch_size"],
            "iters": cfg["total_iters"],
            "learning_rate": cfg["learning_rate"],
            "steps_per_report": 1,          # report every step (for SSE streaming)
            "steps_per_eval": cfg["eval_steps"],
            "save_every": cfg["save_steps"],
            "max_seq_length": cfg["max_seq_len"],
            "grad_checkpoint": True,        # halves RAM at ~20% speed cost
            "seed": 42,
            "fine_tune_type": "lora",       # lora | dora | full

            # ── LoRA-specific parameters ───────────────────────────────
            # num_layers: how many transformer layers get LoRA adapters (-1 = all)
            "num_layers": -1,
            "lora_parameters": {
                "rank": cfg.get("lora_r", 8),
                "scale": lora_scale,        # = lora_alpha; effective scale = alpha/r
                "dropout": cfg.get("lora_dropout", 0.05),
                # Target the Gemma attention + FFN projection matrices
                "keys": cfg.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]),
            },
        }

        config_path = settings.train_data_dir / "mlx_train_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"MLX config written to {config_path}")
        logger.debug(f"Config contents:\n{yaml.dump(yaml_config, default_flow_style=False)}")
        return config_path

    def _build_cmd(self):
        """
        Build the mlx_lm lora command.

        mlx-lm 0.18+:
          - Invocation:  python -m mlx_lm lora -c config.yaml
            (NOT python -m mlx_lm.lora — that form is deprecated)
          - All hyperparameters go in the YAML config file
          - No --lora-layers or --warmup flags (they were removed)
        """
        cfg = self.config
        adapter_path = str(settings.adapter_dir / cfg["run_name"])
        Path(adapter_path).mkdir(parents=True, exist_ok=True)

        config_path = self._write_yaml_config(adapter_path)

        # "python -m mlx_lm lora" (space between mlx_lm and lora — not a dot)
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "-c", str(config_path),
        ]

        # Set HF token env var for model download
        if settings.hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token
            os.environ["HF_TOKEN"] = settings.hf_token

        return cmd, adapter_path


    def train(
        self,
        on_progress: Optional[Callable[[dict], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Run the MLX LoRA training loop.
        Streams stdout, parses JSON-like progress lines, calls on_progress callback.
        Returns the adapter_path on success.
        """
        global _training_state

        cmd, adapter_path = self._build_cmd()
        logger.info(f"Starting MLX LoRA training: {' '.join(cmd)}")

        _training_state.adapter_path = adapter_path
        _training_state.message = "MLX training process started"

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            step = 0
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue

                logger.debug(f"[MLX] {line}")

                # Check stop signal
                if stop_event and stop_event.is_set():
                    self.process.terminate()
                    _training_state.status = "stopped"
                    _training_state.message = "Training stopped by user"
                    return adapter_path

                # Parse progress line
                # mlx-lm emits lines like:
                # Iter 10: Train loss 2.456, Learning Rate 3.0e-04, It/Sec 1.23
                # Iter 10: Val loss 2.567, Val took 3.45s
                event = _parse_mlx_log_line(line, step)
                if event:
                    step = event.get("step", step)
                    _training_state.current_step = step
                    _training_state.current_loss = event.get("loss")

                    val_loss = event.get("val_loss")
                    if val_loss is not None:
                        if (
                            _training_state.best_val_loss is None
                            or val_loss < _training_state.best_val_loss
                        ):
                            _training_state.best_val_loss = val_loss

                    _training_state.loss_history.append(event)

                    if on_progress:
                        on_progress(event)

            self.process.wait()
            if self.process.returncode == 0:
                _training_state.status = "completed"
                _training_state.message = f"Training completed. Adapter saved to {adapter_path}"
                logger.success(f"MLX training complete. Adapter: {adapter_path}")
            else:
                _training_state.status = "failed"
                _training_state.message = f"MLX process exited with code {self.process.returncode}"
                logger.error(_training_state.message)

        except Exception as e:
            _training_state.status = "failed"
            _training_state.message = f"Training error: {e}"
            logger.exception("MLX training failed")

        return adapter_path


def _parse_mlx_log_line(line: str, current_step: int) -> Optional[dict]:
    """
    Parse mlx-lm training log lines into structured events.

    Expected formats:
    - "Iter 10: Train loss 2.456, Learning Rate 3.0e-04, It/Sec 1.23"
    - "Iter 10: Val loss 2.567, Val took 3.45s"
    - "Saved adapter weights to adapters/..."
    """
    import re

    # Train step
    train_match = re.match(
        r"Iter\s+(\d+):\s+Train\s+loss\s+([\d.]+),\s+Learning\s+Rate\s+([\de.+-]+)(?:,\s+It/Sec\s+([\d.]+))?",
        line, re.IGNORECASE
    )
    if train_match:
        step = int(train_match.group(1))
        return {
            "event": "progress",
            "step": step,
            "loss": float(train_match.group(2)),
            "learning_rate": float(train_match.group(3)),
            "tokens_per_sec": float(train_match.group(4)) if train_match.group(4) else None,
        }

    # Val step
    val_match = re.match(
        r"Iter\s+(\d+):\s+Val\s+loss\s+([\d.]+)",
        line, re.IGNORECASE
    )
    if val_match:
        return {
            "event": "val",
            "step": int(val_match.group(1)),
            "val_loss": float(val_match.group(2)),
        }

    # Checkpoint saved
    if "saved adapter" in line.lower():
        return {"event": "checkpoint", "message": line}

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def start_training(
    config: dict,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> None:
    """
    Launch MLX LoRA training in a background thread.
    config keys: base_model, run_name, lora_r, lora_alpha, batch_size,
                 learning_rate, num_epochs, eval_steps, save_steps,
                 max_seq_len, warmup_steps
    """
    global _training_state, _stop_event, _training_thread

    if _training_state.status == "running":
        raise RuntimeError("A training job is already running.")

    # Estimate total iterations
    train_jsonl = settings.train_jsonl
    if train_jsonl.exists():
        num_samples = sum(1 for _ in open(train_jsonl))
    else:
        num_samples = 100  # fallback estimate

    total_iters = max(1, (num_samples // config["batch_size"]) * config["num_epochs"])
    config["total_iters"] = total_iters

    # Reset state
    _stop_event = threading.Event()
    _training_state = TrainingState(
        status="running",
        run_name=config.get("run_name", f"run_{int(time.time())}"),
        total_steps=total_iters,
        message="Initializing MLX model and LoRA adapters...",
    )

    trainer = MLXLoRATrainer(config)

    def _run():
        trainer.train(on_progress=on_progress, stop_event=_stop_event)

    _training_thread = threading.Thread(target=_run, daemon=True, name="mlx-lora-trainer")
    _training_thread.start()
    logger.info(f"Training thread started — run_name={_training_state.run_name}, total_iters={total_iters}")
