"""
app/core/evaluator.py
──────────────────────
Comprehensive evaluation suite for the fine-tuned Gemma model.

Metrics implemented:
──────────────────────────────────────────────────────────────────
CLASSICAL NLP
  - Validation Loss (cross-entropy)
  - Perplexity = exp(val_loss)
  - BLEU-1, BLEU-4 (sacrebleu)
  - ROUGE-1, ROUGE-2, ROUGE-L (rouge-score)

SEMANTIC / RAG QUALITY
  - Faithfulness Score  — semantic similarity between generated answer and source context
                          (sentence-transformers cosine similarity)
  - Answer Relevance    — semantic similarity between generated answer and the question

LLM-AS-A-JUDGE
  - Judge Score (0–10)  — Open-source model via Ollama (local, no API key)
                          Recommended: llama3.2:3b  or  gemma3:4b
                          API: http://localhost:11434/v1  (OpenAI-compatible)

RANKING / RETRIEVAL METRICS
  - nDCG@5, nDCG@10     — normalised discounted cumulative gain
  - MAP@5, MAP@10        — mean average precision
  (These require reference relevance judgements or can be computed via
   self-consistency ranking if no gold labels provided)
──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_valid_samples(num_samples: int = 50) -> List[Dict[str, str]]:
    """
    Parse valid.jsonl and extract (instruction, response) pairs.
    Returns list of dicts with keys: 'question', 'reference'
    """
    samples = []
    if not settings.valid_jsonl.exists():
        return samples

    with open(settings.valid_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            raw = json.loads(line.strip())
            text: str = raw.get("text", "")
            # Parse Gemma chat format: extract user/model turns
            parts = text.split("<start_of_turn>")
            q, a = "", ""
            for part in parts:
                if part.startswith("user\n"):
                    q = part[len("user\n"):].replace("<end_of_turn>", "").strip()
                elif part.startswith("model\n"):
                    a = part[len("model\n"):].replace("<end_of_turn>", "").strip()
            if q and a:
                samples.append({"question": q, "reference": a})
    return samples


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classical NLP Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU-1 and BLEU-4 using sacrebleu."""
    try:
        import sacrebleu
        bleu1 = sacrebleu.corpus_bleu(predictions, [references], max_ngram_order=1)
        bleu4 = sacrebleu.corpus_bleu(predictions, [references], max_ngram_order=4)
        return {"bleu_1": round(bleu1.score / 100, 4), "bleu_4": round(bleu4.score / 100, 4)}
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return {"bleu_1": 0.0, "bleu_4": 0.0}


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_f, r2_f, rl_f = [], [], []
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            r1_f.append(scores["rouge1"].fmeasure)
            r2_f.append(scores["rouge2"].fmeasure)
            rl_f.append(scores["rougeL"].fmeasure)
        return {
            "rouge_1_f": round(float(np.mean(r1_f)), 4),
            "rouge_2_f": round(float(np.mean(r2_f)), 4),
            "rouge_l_f": round(float(np.mean(rl_f)), 4),
        }
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Faithfulness & Answer Relevance (Semantic)
# ─────────────────────────────────────────────────────────────────────────────

_embedding_model = None  # lazy-loaded sentence-transformer

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer for semantic evaluation...")
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def compute_faithfulness(
    predictions: List[str],
    contexts: List[str],
) -> float:
    """
    Faithfulness: how well does the generated answer stay true to the source context?
    Measured as mean cosine similarity between prediction embedding and context embedding.

    Score range: [0, 1] where 1 = perfectly faithful to source text.
    """
    try:
        model = _get_embedding_model()
        pred_embs = model.encode(predictions, normalize_embeddings=True)
        ctx_embs = model.encode(contexts, normalize_embeddings=True)
        sims = [
            _cosine_sim(pred_embs[i], ctx_embs[i])
            for i in range(len(predictions))
        ]
        return round(float(np.mean(sims)), 4)
    except Exception as e:
        logger.warning(f"Faithfulness computation failed: {e}")
        return 0.0


def compute_answer_relevance(
    predictions: List[str],
    questions: List[str],
) -> float:
    """
    Answer Relevance: does the generated answer actually address the question?
    Measured as mean cosine similarity between prediction embedding and question embedding.

    Score range: [0, 1] where 1 = perfectly relevant to the question.
    """
    try:
        model = _get_embedding_model()
        pred_embs = model.encode(predictions, normalize_embeddings=True)
        q_embs = model.encode(questions, normalize_embeddings=True)
        sims = [_cosine_sim(pred_embs[i], q_embs[i]) for i in range(len(predictions))]
        return round(float(np.mean(sims)), 4)
    except Exception as e:
        logger.warning(f"Answer relevance computation failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM-as-a-Judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for AI systems specializing in UIDAI/Aadhaar policy.
Rate the following AI-generated answer on a scale of 0 to 10 based on:
1. Accuracy — Is the information factually correct per UIDAI regulations?
2. Completeness — Does it fully address the question?
3. Policy Alignment — Does it correctly reflect official UIDAI policy?
4. Clarity — Is the answer clear and well-structured?

Question: {question}

Reference Answer: {reference}

AI Generated Answer: {prediction}

Provide your evaluation in JSON format:
{{
  "score": <0-10>,
  "accuracy": <0-10>,
  "completeness": <0-10>,
  "policy_alignment": <0-10>,
  "clarity": <0-10>,
  "reasoning": "<brief explanation>"
}}

JSON only, no other text:"""


def _check_ollama_available() -> bool:
    """
    Ping the Ollama server and verify the judge model is pulled.
    Returns True if Ollama is running and the model is available.
    """
    import urllib.request
    try:
        url = settings.ollama_base_url.replace("/v1", "/api/tags")
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
        available = [m["name"] for m in data.get("models", [])]
        judge = settings.judge_model
        # Ollama tags can be "llama3.2:3b" or "llama3.2" — check prefix
        judge_base = judge.split(":")[0]
        matched = any(judge_base in tag for tag in available)
        if not matched:
            logger.warning(
                f"Ollama is running but judge model '{judge}' is not pulled.\n"
                f"  Run:  ollama pull {judge}\n"
                f"  Available: {available}"
            )
        return matched
    except Exception as e:
        logger.warning(
            f"Ollama not reachable at {settings.ollama_base_url} — {e}\n"
            "  Install Ollama: brew install ollama\n"
            "  Start server:  ollama serve\n"
            f"  Pull model:    ollama pull {settings.judge_model}"
        )
        return False


def compute_llm_judge_score(
    questions: List[str],
    predictions: List[str],
    references: List[str],
    judge_model: str = "llama3.2:3b",
    max_samples: int = 10,
) -> Tuple[float, str]:
    """
    LLM-as-a-Judge using a local open-source model served by Ollama.

    Uses the OpenAI-compatible REST API that Ollama exposes at:
        http://localhost:11434/v1

    No API key is needed — Ollama accepts any string (we use "ollama").

    Recommended judge models (run `ollama pull <model>` to download):
      • llama3.2:3b   — best balance of speed + quality on M2 (~2 GB RAM)
      • gemma3:4b     — same model family as fine-tuned model (~3 GB RAM)
      • mistral:7b    — stronger reasoning, slower (~4 GB RAM)
      • qwen2.5:7b    — excellent instruction following (~4 GB RAM)

    Returns:
        (mean_score: float,  combined_reasoning: str)
        Score range: [0, 10].
        Returns (None, reason_str) if Ollama is not available.
    """
    if not _check_ollama_available():
        return None, (
            f"Ollama not available. Start it with: ollama serve "
            f"and pull the model: ollama pull {settings.judge_model}"
        )

    try:
        from openai import OpenAI   # reuse openai client, pointed at Ollama

        # Ollama's OpenAI-compatible endpoint — api_key can be any non-empty string
        client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",        # required by client but ignored by Ollama
        )

        scores = []
        reasonings = []
        n = min(len(questions), max_samples)
        model_to_use = judge_model or settings.judge_model

        logger.info(f"LLM-as-a-Judge: scoring {n} samples with '{model_to_use}' via Ollama...")

        for i in range(n):
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=questions[i],
                reference=references[i],
                prediction=predictions[i],
            )
            try:
                resp = client.chat.completions.create(
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,          # deterministic scoring
                    max_tokens=400,
                    # Note: Ollama does not support response_format JSON mode
                    # for all models — we parse the JSON manually below
                )
                raw_text = resp.choices[0].message.content.strip()

                # Extract JSON from response (model may add surrounding text)
                json_start = raw_text.find("{")
                json_end = raw_text.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError(f"No JSON found in judge response: {raw_text[:200]}")

                parsed = json.loads(raw_text[json_start:json_end])
                score = float(parsed.get("score", 0))
                scores.append(max(0.0, min(10.0, score)))  # clamp to [0, 10]
                reasonings.append(parsed.get("reasoning", ""))
                logger.debug(f"  Sample {i+1}/{n}: score={score:.1f} — {reasonings[-1][:60]}...")

            except Exception as e:
                logger.warning(f"  Judge call {i+1} failed: {e}")

        if not scores:
            return None, "All judge calls failed — check Ollama logs with: ollama logs"

        mean_score = round(float(np.mean(scores)), 2)
        summary = " | ".join(r for r in reasonings[:3] if r)
        logger.success(f"LLM judge complete: mean_score={mean_score} over {len(scores)} samples")
        return mean_score, summary

    except Exception as e:
        logger.error(f"LLM judge failed unexpectedly: {e}")
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ranking Metrics (nDCG, MAP)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ndcg_map(
    predictions: List[str],
    references: List[str],
    ks: List[int] = [5, 10],
) -> Dict[str, float]:
    """
    Approximate nDCG@k and MAP@k via self-consistency ranking.

    Since we don't have a retrieval corpus here, we simulate relevance grading:
    - For each prediction, we compute its similarity to all references
    - The "true" item is the paired reference (ideal rank = 1)
    - We check whether the model's output is semantically close enough
      to suggest it would be ranked correctly

    This gives an approximation useful for relative comparison across
    fine-tuning runs; for precise IR metrics, provide gold relevance labels.
    """
    try:
        from sklearn.metrics import ndcg_score

        model = _get_embedding_model()
        pred_embs = model.encode(predictions, normalize_embeddings=True)
        ref_embs = model.encode(references, normalize_embeddings=True)

        results = {}
        for k in ks:
            k_actual = min(k, len(predictions))
            ndcg_scores = []
            map_scores = []

            for i in range(len(predictions)):
                # Similarity of prediction[i] to all references
                sims = np.array([_cosine_sim(pred_embs[i], ref_embs[j]) for j in range(len(references))])

                # Ideal: reference[i] should be ranked top
                ideal_relevance = np.zeros(len(references))
                ideal_relevance[i] = 1.0

                # Predicted relevance = similarity scores
                top_k_idx = np.argsort(sims)[::-1][:k_actual]
                pred_relevance = np.zeros(len(references))
                for rank, idx in enumerate(top_k_idx):
                    pred_relevance[idx] = 1.0 / (rank + 1)  # soften

                # nDCG
                ndcg = ndcg_score(
                    ideal_relevance.reshape(1, -1),
                    sims.reshape(1, -1),
                    k=k_actual,
                )
                ndcg_scores.append(ndcg)

                # MAP: was the paired reference in top-k?
                if i in top_k_idx.tolist():
                    rank_of_relevant = top_k_idx.tolist().index(i) + 1
                    ap = 1.0 / rank_of_relevant
                else:
                    ap = 0.0
                map_scores.append(ap)

            results[f"ndcg_at_{k}"] = round(float(np.mean(ndcg_scores)), 4)
            results[f"map_at_{k}"] = round(float(np.mean(map_scores)), 4)

        return results

    except Exception as e:
        logger.warning(f"nDCG/MAP computation failed: {e}")
        return {f"ndcg_at_{k}": 0.0 for k in ks} | {f"map_at_{k}": 0.0 for k in ks}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Loss History
# ─────────────────────────────────────────────────────────────────────────────

def get_loss_history() -> Dict[str, List]:
    """Return training loss history from the trainer state."""
    from app.core.lora_trainer import get_training_state
    state = get_training_state()
    steps, train_losses, val_losses, perplexities = [], [], [], []

    for entry in state.loss_history:
        step = entry.get("step", 0)
        loss = entry.get("loss")
        val_loss = entry.get("val_loss")

        if loss is not None:
            steps.append(step)
            train_losses.append(round(loss, 4))
            val_losses.append(round(val_loss, 4) if val_loss else None)
            perplexities.append(round(math.exp(min(val_loss, 20)), 4) if val_loss else None)

    return {
        "steps": steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "perplexities": perplexities,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Evaluation Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    generate_fn,        # Callable[[str], str] — runs inference on a prompt
    num_samples: int = 50,
    use_llm_judge: bool = True,
    reference_data: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline.

    Args:
        generate_fn: function that takes a question string and returns generated answer
        num_samples: how many validation samples to evaluate
        use_llm_judge: whether to call GPT-4o-mini for judge scoring
        reference_data: optional list of {question, reference} with gold labels

    Returns:
        dict with all metric scores
    """
    start_time = time.time()

    # Load validation samples
    samples = reference_data or _load_valid_samples(num_samples)
    if not samples:
        raise ValueError("No validation samples found. Run /ingest first.")

    questions = [s["question"] for s in samples]
    references = [s["reference"] for s in samples]

    logger.info(f"Evaluating on {len(samples)} samples...")

    # Generate predictions
    predictions = []
    for q in questions:
        try:
            pred = generate_fn(q)
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Generation failed for sample: {e}")
            predictions.append("")

    # ── Compute all metrics ────────────────────────────────────────────
    logger.info("Computing BLEU...")
    bleu_scores = compute_bleu(predictions, references)

    logger.info("Computing ROUGE...")
    rouge_scores = compute_rouge(predictions, references)

    logger.info("Computing semantic faithfulness...")
    faithfulness = compute_faithfulness(predictions, references)

    logger.info("Computing answer relevance...")
    relevance = compute_answer_relevance(predictions, questions)

    logger.info("Computing nDCG/MAP...")
    ranking_scores = compute_ndcg_map(predictions, references, ks=[5, 10])

    # LLM judge via Ollama (runs if Ollama is up, gracefully skipped if not)
    judge_score, judge_reasoning = None, None
    if use_llm_judge:
        logger.info("Running LLM-as-a-Judge via Ollama...")
        judge_score, judge_reasoning = compute_llm_judge_score(
            questions, predictions, references,
            judge_model=settings.judge_model,
            max_samples=settings.judge_max_samples,
        )

    # Get val_loss from training state
    from app.core.lora_trainer import get_training_state
    state = get_training_state()
    val_loss = state.best_val_loss or 99.0
    perplexity = round(math.exp(min(val_loss, 20)), 4)

    duration = round(time.time() - start_time, 2)

    return {
        # Classical NLP
        "val_loss": round(val_loss, 4),
        "perplexity": perplexity,
        **bleu_scores,
        **rouge_scores,
        # Semantic
        "faithfulness_score": faithfulness,
        "answer_relevance_score": relevance,
        # LLM Judge
        "llm_judge_score": judge_score,
        "llm_judge_reasoning": judge_reasoning,
        # Ranking
        **ranking_scores,
        # Meta
        "num_samples_evaluated": len(samples),
        "adapter_path": state.adapter_path,
        "evaluation_duration_seconds": duration,
    }
