# Gemma LoRA Fine-Tuning API 🦾

**Fine-tune a Gemma model locally on Apple Silicon (M2) using LoRA with UIDAI policy documents.**

Built with **FastAPI** + **MLX** (Apple's native ML framework) + **HuggingFace PEFT** (fallback).

---

## Architecture

```
fine_tuning_lora/
├── app/
│   ├── main.py                       # FastAPI entrypoint
│   ├── api/routes/
│   │   ├── ingest.py                 # /ingest — UIDAI PDF download & processing
│   │   ├── finetune.py               # /finetune — LoRA training + SSE streaming
│   │   ├── evaluate.py               # /evaluate — metrics suite
│   │   └── inference.py              # /inference — query fine-tuned model
│   ├── core/
│   │   ├── config.py                 # Pydantic settings (.env)
│   │   ├── document_processor.py     # PDF → JSONL pipeline
│   │   ├── lora_trainer.py           # MLX LoRA trainer (primary)
│   │   ├── hf_trainer.py             # HuggingFace PEFT trainer (fallback)
│   │   ├── evaluator.py              # All eval metrics
│   │   └── inference_engine.py       # Lazy model loader + generation
│   └── models/schemas.py             # Pydantic request/response schemas
├── data/
│   ├── raw/                          # Downloaded UIDAI PDFs
│   ├── processed/                    # Cleaned text
│   └── train/                        # train.jsonl, valid.jsonl
├── adapters/                         # Saved LoRA adapter weights
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Prerequisites

- macOS 13+ with Apple Silicon (M1/M2/M3/M4) — uses MLX
- Python 3.10+
- HuggingFace account with Gemma access

### 2. Accept Gemma License

Visit: https://huggingface.co/google/gemma-3-1b-it  
Click **"Agree and access repository"**

### 3. Install Dependencies

```bash
cd fine_tuning_lora
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and set:
#   HF_TOKEN=hf_your_token_here
#   OPENAI_API_KEY=sk-...   (optional, for LLM-as-Judge evaluation)
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000/docs

---

## Usage

### Step 1 — Ingest UIDAI Documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"max_docs": 5, "chunk_size": 512, "train_split": 0.8}'
```

Poll status:
```bash
curl http://localhost:8000/api/v1/ingest/status
```

### Step 2 — Start LoRA Fine-Tuning

```bash
curl -X POST http://localhost:8000/api/v1/finetune/start \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "google/gemma-3-1b-it",
    "backend": "mlx",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 3e-4
  }'
```

### Step 3 — Stream Training Progress

```bash
curl -N http://localhost:8000/api/v1/finetune/stream
# SSE events: {"event": "progress", "step": 10, "loss": 2.456, "lr": 3e-4}
```

### Step 4 — Evaluate

```bash
curl "http://localhost:8000/api/v1/evaluate?num_samples=30&use_llm_judge=false"
```

Returns: BLEU, ROUGE, Perplexity, Faithfulness, Answer Relevance, nDCG@5, MAP@5

### Step 5 — Inference

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the privacy obligations of Requesting Entities under UIDAI regulations?",
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Compare Base vs Fine-Tuned

```bash
curl "http://localhost:8000/api/v1/inference/compare?prompt=What+is+Aadhaar+VID"
```

---

## How LoRA Works

### The Math

For a frozen weight matrix **W₀ ∈ ℝ^(d×k)**:

```
Forward: output = W₀·x + (α/r)·B·(A·x)

Where:
  A ∈ ℝ^(r×k)  — initialized with Gaussian noise  (TRAINED)
  B ∈ ℝ^(d×r)  — initialized to ZERO              (TRAINED)
  W₀            — original Gemma weight             (FROZEN)
  r             — rank (controls adapter size)
  α             — scaling factor
```

**Why zero-initialize B?** So that at step 0, `B·A = 0`, meaning the model starts identically to the base — no sudden distribution shift.

### Parameters in This Setup (r=8, Gemma-1B)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `lora_r` | 8 | Each adapter adds 2 matrices of rank 8 |
| `lora_alpha` | 16 | Updates scaled by α/r = 2× |
| `lora_dropout` | 0.05 | 5% of adapter activations zeroed per step |
| `target_modules` | q,k,v,o,gate,up,down | 7 projection types × 18 layers |
| **Trainable params** | ~6M | **0.6% of 1B total** |

### Loss Function

```
L = -Σ log P(token_t | token_{<t})

Only computed on response tokens (completion masking).
User instruction tokens have label = -100 (ignored).
```

---

## Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Classical | `val_loss` | Cross-entropy on validation set |
| Classical | `perplexity` | exp(val_loss) — lower = better |
| Classical | `bleu_1`, `bleu_4` | N-gram precision (sacrebleu) |
| Classical | `rouge_1`, `rouge_2`, `rouge_l` | Token overlap F1 |
| Semantic | `faithfulness_score` | Cosine sim of answer vs source context |
| Semantic | `answer_relevance_score` | Cosine sim of answer vs question |
| LLM Judge | `llm_judge_score` | GPT-4o-mini rates 0-10 (requires API key) |
| Ranking | `ndcg_at_5`, `ndcg_at_10` | Normalised DCG |
| Ranking | `map_at_5`, `map_at_10` | Mean Average Precision |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Start UIDAI document ingestion |
| `GET` | `/api/v1/ingest/status` | Ingestion progress |
| `GET` | `/api/v1/ingest/samples` | Preview training data |
| `DELETE` | `/api/v1/ingest/reset` | Clear all data |
| `POST` | `/api/v1/finetune/start` | Launch LoRA training |
| `GET` | `/api/v1/finetune/status` | Training state |
| `GET` | `/api/v1/finetune/stream` | SSE training progress |
| `POST` | `/api/v1/finetune/stop` | Stop training |
| `GET` | `/api/v1/finetune/artifacts` | List saved adapters |
| `GET` | `/api/v1/finetune/lora-explain` | LoRA math explanation |
| `GET` | `/api/v1/evaluate` | Run full evaluation |
| `GET` | `/api/v1/evaluate/history` | Training loss curve data |
| `GET` | `/api/v1/evaluate/metrics` | Last eval results |
| `POST` | `/api/v1/inference` | Query fine-tuned model |
| `POST` | `/api/v1/inference/stream` | Streaming generation (SSE) |
| `GET` | `/api/v1/inference/compare` | Base vs fine-tuned comparison |
| `GET` | `/health` | Service health |

---

## Memory Requirements (M2 16GB)

| Model | Full FT | LoRA (r=8) | MLX (quantized) |
|-------|---------|-----------|-----------------|
| Gemma-3 1B | ~14 GB | ~4 GB | ~2.5 GB |
| Gemma-3 4B | OOM | ~8 GB | ~5 GB |

**Recommendation:** Use `batch_size=4` + `grad_accum_steps=4` for stable training on 16GB M2.
