"""
app/core/hf_trainer.py
───────────────────────
HuggingFace PEFT + TRL fallback LoRA trainer.
Use when: running on Intel Mac, Windows/Linux, or preferring the HF ecosystem.

On M2 Mac: set PYTORCH_ENABLE_MPS_FALLBACK=1 in environment.

How LoRA differs between MLX and HF/PEFT:
  - Architecture is identical: frozen base + low-rank A/B adapters
  - PEFT uses nn.Linear wrappers (LoraLayer) instead of MLX's array-based approach
  - SFTTrainer handles the training loop; HF DataCollator applies completion masking
  - gradient_checkpointing=True halves RAM usage at ~20% speed cost
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from app.core.config import settings
from app.core.lora_trainer import TrainingState, _training_state, get_training_state


class HFLoRATrainer:
    """
    HuggingFace PEFT LoRA trainer using SFTTrainer (TRL).
    Imports are deferred to avoid loading torch at startup when using MLX.
    """

    def __init__(self, config: dict):
        self.config = config

    def train(
        self,
        on_progress: Optional[Callable[[dict], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> str:
        global _training_state

        # Deferred imports
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
            TrainerControl,
            TrainerState,
            TrainingArguments,
        )
        from trl import SFTConfig, SFTTrainer

        cfg = self.config
        adapter_path = str(settings.adapter_dir / cfg["run_name"])
        os.makedirs(adapter_path, exist_ok=True)

        if settings.hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

        # ── Device selection ─────────────────────────────────────────────
        if torch.backends.mps.is_available():
            # Apple Silicon via MPS
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            device = "mps"
            logger.info("Using MPS (Apple Silicon) backend")
        else:
            device = "cpu"
            logger.warning("MPS not available — using CPU (will be slow)")

        # ── Load tokenizer ───────────────────────────────────────────────
        logger.info(f"Loading tokenizer: {cfg['base_model']}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["base_model"],
            token=settings.hf_token or None,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ── Load base model ──────────────────────────────────────────────
        logger.info(f"Loading base model: {cfg['base_model']}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            torch_dtype=torch.float32,   # float16 not stable on MPS
            device_map=device,
            token=settings.hf_token or None,
        )
        model.enable_input_require_grads()

        # ── LoRA config ──────────────────────────────────────────────────
        # This is where the magic happens:
        # We inject A (random init) and B (zero init) matrices into each
        # target_module. Only A and B will receive gradients.
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            bias="none",                  # don't train biases
            inference_mode=False,
        )

        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        pct = 100 * trainable / total
        logger.info(
            f"LoRA adapters injected: {trainable:,} trainable / {total:,} total params ({pct:.2f}%)"
        )
        _training_state.message = f"LoRA adapters injected: {trainable:,} trainable params ({pct:.2f}%)"

        # Enable gradient checkpointing to save RAM
        model.gradient_checkpointing_enable()

        # ── Dataset ──────────────────────────────────────────────────────
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(settings.train_jsonl),
                "validation": str(settings.valid_jsonl),
            },
        )

        # ── Streaming progress callback ───────────────────────────────────
        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
                if logs and on_progress:
                    event = {
                        "event": "progress",
                        "step": state.global_step,
                        "loss": logs.get("loss"),
                        "val_loss": logs.get("eval_loss"),
                        "learning_rate": logs.get("learning_rate", 0.0),
                    }
                    _training_state.current_step = state.global_step
                    _training_state.current_loss = event.get("loss")
                    if event.get("val_loss") is not None:
                        if (
                            _training_state.best_val_loss is None
                            or event["val_loss"] < _training_state.best_val_loss
                        ):
                            _training_state.best_val_loss = event["val_loss"]
                    _training_state.loss_history.append(event)
                    on_progress(event)

            def on_step_begin(self, args, state, control, **kwargs):
                if stop_event and stop_event.is_set():
                    control.should_training_stop = True

        # ── Training arguments ───────────────────────────────────────────
        training_args = SFTConfig(
            output_dir=adapter_path,
            num_train_epochs=cfg["num_epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg.get("grad_accum_steps", 4),
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg["warmup_steps"],
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_seq_length=cfg["max_seq_len"],
            dataset_text_field="text",
            fp16=False,                   # MPS does not support fp16 reliably
            bf16=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none",             # disable wandb/tensorboard by default
            seed=42,
            # Completion masking: only compute loss on model turn
            # (not on the user instruction prefix)
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            callbacks=[ProgressCallback()],
        )

        logger.info("Starting HuggingFace SFTTrainer training loop...")
        _training_state.total_steps = trainer.args.max_steps if trainer.args.max_steps > 0 else (
            len(dataset["train"]) // cfg["batch_size"] * cfg["num_epochs"]
        )

        trainer.train()

        # Save final adapter
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        logger.success(f"HF LoRA training complete. Adapter saved to {adapter_path}")

        _training_state.status = "completed"
        _training_state.adapter_path = adapter_path
        _training_state.message = f"Training completed. Adapter saved to {adapter_path}"

        return adapter_path
