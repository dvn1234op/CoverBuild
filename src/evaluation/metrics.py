"""src/evaluation/metrics.py — Evaluation metrics for cover letter generation.

Computes:
  - ROUGE-L (recall-oriented, measures content overlap)
  - BLEU (precision-oriented, measures n-gram match)
  - Reward model scores
  - Before/After comparison table (pre-SFT vs post-PPO)
"""

from __future__ import annotations

import re
from typing import Optional

import mlflow
import nltk
import numpy as np

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


# Ensure NLTK punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_rouge(
    hypothesis: str, reference: str, rouge_type: str = "rougeL"
) -> float:
    """Compute ROUGE-L F-score between hypothesis and reference."""
    if not HAS_ROUGE:
        return 0.0
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return round(scores[rouge_type].fmeasure, 4)


def compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute sentence-level BLEU score."""
    if not HAS_NLTK:
        return 0.0
    try:
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())
        if not hyp_tokens or not ref_tokens:
            return 0.0
        smoothie = SmoothingFunction().method4
        return round(
            sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie), 4
        )
    except Exception:
        return 0.0


def compute_lexical_diversity(text: str) -> float:
    """Type-token ratio — measures vocabulary richness."""
    words = text.lower().split()
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def compute_professionalism_score(text: str) -> float:
    """Simple rule-based professionalism heuristic."""
    score = 0.0
    text_lower = text.lower()

    good_phrases = [
        "i am pleased", "i am excited", "i would welcome",
        "thank you for", "i look forward", "sincerely", "best regards",
        "my expertise", "i have demonstrated", "proven track record",
    ]
    score += sum(0.08 for p in good_phrases if p in text_lower)

    bad_phrases = ["hire me", "i need", "i want the job", "stuff", "things", "whatnot"]
    score -= sum(0.1 for p in bad_phrases if p in text_lower)

    # Length bonus
    n_words = len(text.split())
    if 100 <= n_words <= 400:
        score += 0.2

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# High-level evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    policy_model,
    reward_model,
    tokenizer,
    prompts: list[str],
    reference_completions: Optional[list[str]] = None,
    n_samples: int = 20,
    mlflow_step: Optional[int] = None,
    tag: str = "eval",
) -> dict:
    """
    Evaluate the current policy model.

    Args:
        policy_model:   The policy (GPT-2) with generate() method.
        reward_model:   Reward model with score() method.
        tokenizer:      Tokenizer.
        prompts:        List of prompts to evaluate on.
        reference_completions: Gold reference texts for ROUGE/BLEU.
        n_samples:      Number of prompts to evaluate.
        mlflow_step:    Step for MLflow logging.
        tag:            Label prefix for logged metrics.

    Returns:
        dict of aggregated metrics.
    """
    import torch

    sample_prompts = prompts[:n_samples]
    device = next(policy_model.parameters()).device if hasattr(policy_model, "parameters") else "cpu"

    rewards, rouge_scores, bleu_scores, diversity_scores, prof_scores = [], [], [], [], []
    generated_texts = []

    for i, prompt in enumerate(sample_prompts):
        # Generate completion
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(device)

        with torch.no_grad():
            out_ids = policy_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_texts.append(completion)

        # Reward
        r = reward_model.score(prompt, completion)
        rewards.append(r)

        # ROUGE / BLEU
        if reference_completions and i < len(reference_completions):
            ref = reference_completions[i]
            rouge_scores.append(compute_rouge(completion, ref))
            bleu_scores.append(compute_bleu(completion, ref))

        # Lexical diversity & professionalism
        diversity_scores.append(compute_lexical_diversity(completion))
        prof_scores.append(compute_professionalism_score(completion))

    metrics = {
        f"{tag}/reward_mean":      round(float(np.mean(rewards)), 4),
        f"{tag}/reward_std":       round(float(np.std(rewards)), 4),
        f"{tag}/diversity_mean":   round(float(np.mean(diversity_scores)), 4),
        f"{tag}/prof_score_mean":  round(float(np.mean(prof_scores)), 4),
    }
    if rouge_scores:
        metrics[f"{tag}/rouge_l_mean"] = round(float(np.mean(rouge_scores)), 4)
    if bleu_scores:
        metrics[f"{tag}/bleu_mean"] = round(float(np.mean(bleu_scores)), 4)

    if mlflow.active_run() and mlflow_step is not None:
        mlflow.log_metrics(metrics, step=mlflow_step)

    # Print summary
    print(f"\n📊 {tag.upper()} Evaluation ({n_samples} samples):")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    return metrics


def print_comparison_table(
    pre_ppo_metrics: dict,
    post_ppo_metrics: dict,
):
    """Print a before/after PPO comparison table."""
    print("\n" + "=" * 70)
    print("  📈 Before vs After PPO — Improvement Summary")
    print("=" * 70)
    print(f"  {'Metric':<35} {'Before SFT':>12} {'After PPO':>12} {'Δ':>8}")
    print("  " + "-" * 67)

    all_keys = sorted(set(pre_ppo_metrics) | set(post_ppo_metrics))
    for key in all_keys:
        short_key = key.replace("eval/", "").replace("_mean", "")
        pre  = pre_ppo_metrics.get(key, float("nan"))
        post = post_ppo_metrics.get(key, float("nan"))
        delta = post - pre if not (isinstance(pre, float) and (pre != pre)) else 0.0
        delta_str = f"{delta:+.4f}" if delta != 0 else "  —   "
        print(f"  {short_key:<35} {pre:>12.4f} {post:>12.4f} {delta_str:>8}")

    print("=" * 70 + "\n")
