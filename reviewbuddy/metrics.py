"""
Metrics for the flat reviewer-level GEPA training system.

GEPA metric (used for optimization):
  - Semantic similarity: 0.5 weight  (sentence-transformers cosine similarity)
  - Score similarity:    0.5 weight  (strict normalized: 1 - |pred-human| / 9)

Per-example metrics (compute_all_metrics):
  - ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, Semantic Similarity
  - Length diff (signed: positive = generated longer)
  - LLM-as-judge (optional, eval only)

Aggregate metrics (compute_aggregate_metrics):
  - Per-reviewer-pair: MAE mean±std, RMSE mean±std, Spearman (pooled)
  - Final score: MAE, RMSE
  - Overall: Spearman (p-value), AUC-ROC
  - Accept/Reject: Accuracy, MCC, Confusion matrix (TP/FP/FN/TN)
  - Text quality: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, Semantic Similarity, Length Diff
"""

import math
import dspy
import numpy as np
from functools import lru_cache
from typing import List, Optional
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import matthews_corrcoef, roc_auc_score, cohen_kappa_score
from reviewbuddy.prompts import LLMJudge

# ICLR score range: min=1, max=10 - max distance = 9
_ICLR_MAX_DIST = 9.0


def _safe_int_score(value, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Sentence-transformers model (loaded once, cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(generated: str, reference: str) -> float:
    """Cosine similarity between sentence embeddings. Returns float in [0, 1]."""
    if not generated.strip() or not reference.strip():
        return 0.0
    try:
        model = _get_st_model()
        embeddings = model.encode([generated, reference], convert_to_tensor=True)
        from sentence_transformers import util
        score = float(util.cos_sim(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"  [semantic_sim] Error: {e}")
        return 0.0


def review_semantic_similarity(
    generated_comments: str,
    generated_strengths: str,
    generated_weaknesses: str,
    human_comments: str,
    human_strengths: str,
    human_weaknesses: str,
) -> float:
    """Average semantic similarity across comments, strengths, weaknesses."""
    pairs = [
        (generated_comments, human_comments),
        (generated_strengths, human_strengths),
        (generated_weaknesses, human_weaknesses),
    ]
    scores = [semantic_similarity(gen, ref) for gen, ref in pairs if ref.strip()]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Score similarity
# ---------------------------------------------------------------------------

def score_similarity(pred_score: int, human_score: int) -> float:
    """Normalized distance with extra penalty for wrong accept/reject side.
    Crossing the accept threshold (6) multiplies the distance by 1.6,
    making wrong-side errors worse without a binary cliff."""
    dist = abs(pred_score - human_score) / _ICLR_MAX_DIST
    if (pred_score >= 6) != (human_score >= 6):
        dist = min(1.0, dist * 1.6)
    return 1.0 - dist


# ---------------------------------------------------------------------------
# GEPA metric (combined, no LLM judge)
# ---------------------------------------------------------------------------

def gepa_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """GEPA metric: 0.5 * semantic_similarity + 0.5 * score_similarity. Returns float in [0, 1]."""
    pred_score = _safe_int_score(getattr(prediction, "score", 1))

    sem_sim = review_semantic_similarity(
        generated_comments=getattr(prediction, "comments", ""),
        generated_strengths=getattr(prediction, "strengths", ""),
        generated_weaknesses=getattr(prediction, "weaknesses", ""),
        human_comments=example.human_comments,
        human_strengths=example.human_strengths,
        human_weaknesses=example.human_weaknesses,
    )
    score_sim = score_similarity(pred_score, example.human_score)
    return 0.5 * sem_sim + 0.5 * score_sim


# ---------------------------------------------------------------------------
# Text quality metrics
# ---------------------------------------------------------------------------

_judge = None


def _get_judge():
    global _judge
    if _judge is None:
        _judge = dspy.Predict(LLMJudge)
    return _judge


def llm_judge_score(
    generated_comments: str,
    generated_strengths: str,
    generated_weaknesses: str,
    generated_score: int,
    human_comments: str,
    human_strengths: str,
    human_weaknesses: str,
    human_score: int,
) -> float:
    """LLM-as-judge, logging only. Returns float 0-1."""
    generated_review = (
        f"Comments:\n{generated_comments}\n\n"
        f"Strengths:\n{generated_strengths}\n\n"
        f"Weaknesses:\n{generated_weaknesses}"
    )
    human_review = (
        f"Comments:\n{human_comments}\n\n"
        f"Strengths:\n{human_strengths}\n\n"
        f"Weaknesses:\n{human_weaknesses}"
    )
    try:
        result = _get_judge()(
            generated_review=generated_review,
            human_review=human_review,
            generated_score=generated_score,
            human_score=human_score,
        )
        return max(0.0, min(1.0, float(result.evaluation)))
    except Exception as e:
        print(f"  [judge] Error: {e}")
        return 0.0


def rouge_scores(generated: str, reference: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        result = scorer.score(reference, generated)
        return {
            "rouge_1": result["rouge1"].fmeasure,
            "rouge_2": result["rouge2"].fmeasure,
            "rouge_l": result["rougeL"].fmeasure,
        }
    except Exception:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


def bertscore_f1(generated: str, reference: str, lang: str = "en") -> float:
    """BERTScore F1. Returns 0.0 on failure."""
    try:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        from bert_score import score as bs_score
        _, _, F = bs_score([generated], [reference], lang=lang, verbose=False)
        return float(F[0])
    except Exception:
        return 0.0


def length_diff(generated: str, reference: str) -> int:
    """Signed word count difference: positive = generated is longer."""
    return len(generated.split()) - len(reference.split())


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    example: dspy.Example,
    prediction: dspy.Prediction,
    include_llm_judge: bool = False,
) -> dict:
    """Compute all per-example metrics. LLM judge is off by default — enable only at eval time."""
    pred_score = _safe_int_score(getattr(prediction, "score", 1))

    pred_comments = getattr(prediction, "comments", "")
    pred_strengths = getattr(prediction, "strengths", "")
    pred_weaknesses = getattr(prediction, "weaknesses", "")

    sem_sim = review_semantic_similarity(
        pred_comments, pred_strengths, pred_weaknesses,
        example.human_comments, example.human_strengths, example.human_weaknesses,
    )
    score_sim = score_similarity(pred_score, example.human_score)
    judge = llm_judge_score(
        pred_comments, pred_strengths, pred_weaknesses, pred_score,
        example.human_comments, example.human_strengths, example.human_weaknesses, example.human_score,
    ) if include_llm_judge else None

    rouge = rouge_scores(pred_comments, example.human_comments)
    bs = bertscore_f1(pred_comments, example.human_comments)
    ld = length_diff(pred_comments, example.human_comments)

    return {
        "gepa_score": 0.5 * sem_sim + 0.5 * score_sim,
        "semantic_similarity": sem_sim,
        "score_similarity": score_sim,
        "llm_judge": judge,
        "pred_score": pred_score,
        "human_score": example.human_score,
        "rouge_1": rouge["rouge_1"],
        "rouge_2": rouge["rouge_2"],
        "rouge_l": rouge["rouge_l"],
        "bertscore_f1": bs,
        "length_diff": ld,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics (over all evaluated papers/pairs)
# ---------------------------------------------------------------------------

def compute_aggregate_metrics(results: List[dict]) -> dict:
    """
    Compute comprehensive aggregate metrics.

    Each result dict must have:
      pred_score, human_score, final_decision, human_decision,
      rouge_l, rouge_1, rouge_2, bertscore_f1, semantic_similarity, length_diff
      Optional: per_pair_pred_scores, per_pair_human_scores (for GEPA eval per-pair stats)
      Optional: llm_judge
    """
    if not results:
        return {}

    n = len(results)

    # ── Score fields ─────────────────────────────────────────────────────────
    pred_scores = [float(r["pred_score"]) for r in results]
    human_scores = [float(r["human_score"]) for r in results]
    decisions = [r["final_decision"].strip().lower() for r in results]
    human_decisions = [r["human_decision"].strip().lower() for r in results]

    # ── Regression (final scores) ─────────────────────────────────────────────
    errors = [p - h for p, h in zip(pred_scores, human_scores)]
    abs_errors = [abs(e) for e in errors]
    final_mae = float(np.mean(abs_errors))
    final_rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))

    # ── Per-reviewer-pair stats (GEPA eval only) ──────────────────────────────
    pair_pred, pair_human = [], []
    for r in results:
        pp = r.get("per_pair_pred_scores")
        ph = r.get("per_pair_human_scores")
        if pp and ph:
            pair_pred.extend(pp)
            pair_human.extend(ph)

    pair_mae_vals = [abs(p - h) for p, h in zip(pair_pred, pair_human)]
    pair_rmse_vals = [(p - h) ** 2 for p, h in zip(pair_pred, pair_human)]
    pair_mae_mean = float(np.mean(pair_mae_vals)) if pair_mae_vals else None
    pair_mae_std = float(np.std(pair_mae_vals)) if pair_mae_vals else None
    pair_rmse_mean = float(np.sqrt(np.mean(pair_rmse_vals))) if pair_rmse_vals else None
    pair_spearman, pair_spearman_p = (None, None)
    if len(pair_pred) > 2:
        r_val, p_val = spearmanr(pair_pred, pair_human)
        pair_spearman, pair_spearman_p = float(r_val), float(p_val)

    # ── Overall Spearman, Pearson, Kendall ───────────────────────────────────
    overall_spearman, overall_spearman_p = (None, None)
    overall_pearson, overall_pearson_p = (None, None)
    overall_kendall, overall_kendall_p = (None, None)
    if n > 2:
        r_val, p_val = spearmanr(pred_scores, human_scores)
        overall_spearman, overall_spearman_p = float(r_val), float(p_val)
        r_val, p_val = pearsonr(pred_scores, human_scores)
        overall_pearson, overall_pearson_p = float(r_val), float(p_val)
        r_val, p_val = kendalltau(pred_scores, human_scores)
        overall_kendall, overall_kendall_p = float(r_val), float(p_val)

    # ── Per-pair Pearson, Kendall ─────────────────────────────────────────────
    pair_pearson, pair_pearson_p = (None, None)
    pair_kendall, pair_kendall_p = (None, None)
    if len(pair_pred) > 2:
        r_val, p_val = pearsonr(pair_pred, pair_human)
        pair_pearson, pair_pearson_p = float(r_val), float(p_val)
        r_val, p_val = kendalltau(pair_pred, pair_human)
        pair_kendall, pair_kendall_p = float(r_val), float(p_val)

    # ── Weighted Kappa, Exact match, Off-by-one ───────────────────────────────
    pred_scores_int = [int(round(p)) for p in pred_scores]
    human_scores_int = [int(round(h)) for h in human_scores]

    try:
        weighted_kappa = float(cohen_kappa_score(human_scores_int, pred_scores_int, weights="quadratic"))
    except Exception:
        weighted_kappa = None

    exact_match = sum(p == h for p, h in zip(pred_scores_int, human_scores_int)) / n
    off_by_one = sum(abs(p - h) <= 1 for p, h in zip(pred_scores_int, human_scores_int)) / n

    # per-pair exact match / off-by-one
    pair_exact_match = None
    pair_off_by_one = None
    if pair_pred:
        pair_pred_int = [int(round(p)) for p in pair_pred]
        pair_human_int = [int(round(h)) for h in pair_human]
        pair_exact_match = sum(p == h for p, h in zip(pair_pred_int, pair_human_int)) / len(pair_pred)
        pair_off_by_one = sum(abs(p - h) <= 1 for p, h in zip(pair_pred_int, pair_human_int)) / len(pair_pred)

    # ── AUC-ROC ───────────────────────────────────────────────────────────────
    auc = None
    y_true_binary = [1 if d == "accept" else 0 for d in human_decisions]
    if sum(y_true_binary) > 0 and sum(y_true_binary) < n:
        try:
            auc = float(roc_auc_score(y_true_binary, pred_scores))
        except Exception:
            pass

    # ── Accept/Reject classification ─────────────────────────────────────────
    pred_binary = [1 if d == "accept" else 0 for d in decisions]
    accuracy = sum(p == h for p, h in zip(pred_binary, y_true_binary)) / n

    tp = sum(1 for p, h in zip(pred_binary, y_true_binary) if p == 1 and h == 1)
    fp = sum(1 for p, h in zip(pred_binary, y_true_binary) if p == 1 and h == 0)
    fn = sum(1 for p, h in zip(pred_binary, y_true_binary) if p == 0 and h == 1)
    tn = sum(1 for p, h in zip(pred_binary, y_true_binary) if p == 0 and h == 0)

    try:
        mcc = float(matthews_corrcoef(y_true_binary, pred_binary))
    except Exception:
        mcc = 0.0

    # ── Text quality ──────────────────────────────────────────────────────────
    def _avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    avg_rouge_1 = _avg("rouge_1")
    avg_rouge_2 = _avg("rouge_2")
    avg_rouge_l = _avg("rouge_l")
    avg_bert = _avg("bertscore_f1")
    avg_sem = _avg("semantic_similarity")
    avg_judge = _avg("llm_judge")

    ld_vals = [r["length_diff"] for r in results if "length_diff" in r]
    avg_length_diff = float(np.mean(ld_vals)) if ld_vals else None

    return {
        "n": n,
        "n_pairs": len(pair_pred),
        # per-pair (GEPA eval)
        "pair_mae_mean": pair_mae_mean,
        "pair_mae_std": pair_mae_std,
        "pair_rmse_mean": pair_rmse_mean,
        "pair_spearman": pair_spearman,
        "pair_spearman_p": pair_spearman_p,
        "pair_pearson": pair_pearson,
        "pair_pearson_p": pair_pearson_p,
        "pair_kendall": pair_kendall,
        "pair_kendall_p": pair_kendall_p,
        "pair_exact_match": pair_exact_match,
        "pair_off_by_one": pair_off_by_one,
        # final score
        "final_mae": final_mae,
        "final_rmse": final_rmse,
        # overall correlations
        "overall_spearman": overall_spearman,
        "overall_spearman_p": overall_spearman_p,
        "overall_pearson": overall_pearson,
        "overall_pearson_p": overall_pearson_p,
        "overall_kendall": overall_kendall,
        "overall_kendall_p": overall_kendall_p,
        "weighted_kappa": weighted_kappa,
        "exact_match": exact_match,
        "off_by_one": off_by_one,
        "auc_roc": auc,
        # classification
        "accuracy": accuracy,
        "mcc": mcc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        # text quality
        "avg_rouge_1": avg_rouge_1,
        "avg_rouge_2": avg_rouge_2,
        "avg_rouge_l": avg_rouge_l,
        "avg_bertscore_f1": avg_bert,
        "avg_semantic_similarity": avg_sem,
        "avg_llm_judge": avg_judge,
        "avg_length_diff": avg_length_diff,
    }


def print_aggregate_metrics(metrics: dict, label: str = "") -> None:
    header = f"RESULTS{' — ' + label if label else ''}"
    n = metrics.get("n", "?")
    n_pairs = metrics.get("n_pairs", 0)

    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"Papers evaluated:         {n}")

    if n_pairs:
        print(f"Total reviewer pairs:     {n_pairs}")
        print(f"\n--- Per-reviewer-pair score ---")
        if metrics["pair_mae_mean"] is not None:
            print(f"MAE  (mean ± std):        {metrics['pair_mae_mean']:.3f} ± {metrics['pair_mae_std']:.3f}")
        if metrics["pair_rmse_mean"] is not None:
            print(f"RMSE (mean):              {metrics['pair_rmse_mean']:.3f}")
        if metrics["pair_spearman"] is not None:
            print(f"Spearman (pooled):        {metrics['pair_spearman']:.4f}  (p={metrics['pair_spearman_p']:.4f})")
        if metrics["pair_pearson"] is not None:
            print(f"Pearson  (pooled):        {metrics['pair_pearson']:.4f}  (p={metrics['pair_pearson_p']:.4f})")
        if metrics["pair_kendall"] is not None:
            print(f"Kendall  (pooled):        {metrics['pair_kendall']:.4f}  (p={metrics['pair_kendall_p']:.4f})")
        if metrics["pair_exact_match"] is not None:
            print(f"Exact match:              {metrics['pair_exact_match']*100:.1f}%")
        if metrics["pair_off_by_one"] is not None:
            print(f"Off-by-one:               {metrics['pair_off_by_one']*100:.1f}%")

    print(f"\n--- Final score (mean gen vs mean human) ---")
    print(f"Final score MAE:          {metrics['final_mae']:.3f}")
    print(f"Final score RMSE:         {metrics['final_rmse']:.3f}")
    print(f"\n--- Overall score metrics ---")
    if metrics["overall_spearman"] is not None:
        print(f"Spearman:                 {metrics['overall_spearman']:.4f}  (p={metrics['overall_spearman_p']:.4f})")
    if metrics["overall_pearson"] is not None:
        print(f"Pearson:                  {metrics['overall_pearson']:.4f}  (p={metrics['overall_pearson_p']:.4f})")
    if metrics["overall_kendall"] is not None:
        print(f"Kendall:                  {metrics['overall_kendall']:.4f}  (p={metrics['overall_kendall_p']:.4f})")
    if metrics["weighted_kappa"] is not None:
        print(f"Weighted Kappa:           {metrics['weighted_kappa']:.4f}")
    print(f"Exact match:              {metrics['exact_match']*100:.1f}%")
    print(f"Off-by-one:               {metrics['off_by_one']*100:.1f}%")
    if metrics["auc_roc"] is not None:
        print(f"AUC-ROC:                  {metrics['auc_roc']:.4f}")

    print(f"\n--- Accept/Reject ---")
    print(f"Accuracy:                 {metrics['accuracy']:.4f}")
    print(f"MCC:                      {metrics['mcc']:.4f}")
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    print(f"Confusion matrix:         TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    print(f"\n--- Text Quality ---")
    if metrics["avg_rouge_1"] is not None:
        print(f"Avg ROUGE-1:              {metrics['avg_rouge_1']:.4f}")
    if metrics["avg_rouge_2"] is not None:
        print(f"Avg ROUGE-2:              {metrics['avg_rouge_2']:.4f}")
    if metrics["avg_rouge_l"] is not None:
        print(f"Avg ROUGE-L:              {metrics['avg_rouge_l']:.4f}")
    if metrics["avg_bertscore_f1"] is not None:
        print(f"Avg BERTScore F1:         {metrics['avg_bertscore_f1']:.4f}")
    if metrics["avg_semantic_similarity"] is not None:
        print(f"Avg Semantic Similarity:  {metrics['avg_semantic_similarity']:.4f}")
    if metrics["avg_llm_judge"] is not None:
        print(f"Avg LLM Judge:            {metrics['avg_llm_judge']:.4f}")
    if metrics["avg_length_diff"] is not None:
        sign = "+" if metrics["avg_length_diff"] >= 0 else ""
        print(f"Avg Length Diff:          {sign}{metrics['avg_length_diff']:.1f} words  "
              f"({'generated longer' if metrics['avg_length_diff'] > 0 else 'generated shorter'})")
    print(f"{'='*60}")
