"""First proof-of-concept: generates paper reviews via DSPy with Attachments and evaluates against human reviews (ROUGE-L, BERTScore, Spearman)."""

import os
import json
import dspy
import numpy as np
from dotenv import load_dotenv
from evaluate import load
from bert_score import score
from attachments.dspy import Attachments
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_absolute_error, mean_squared_error

load_dotenv()

DEFAULT_USER_PROMPT = "Vygeneruj plnú, štruktúrovanú akademickú recenziu tohto článku. Zameraj sa na silné a slabé stránky."

REVIEWS_DIR = "C:\\Users\\katka\\BAKALARKA\\test_reviews3"
PDF_DIR = "C:\\Users\\katka\\BAKALARKA\\test_pdfs3"
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")


# -------------------------------------------------------------------------
#   Načítanie reálnych ľudských recenzií + ciest k PDF
# -------------------------------------------------------------------------
def load_human_reviews(limit=20):
    human_reviews = []

    review_files = [f for f in os.listdir(REVIEWS_DIR) if f.endswith(".json")]
    for filename in review_files:
        base_name = filename.replace(".json", "")
        pdf_path = os.path.join(PDF_DIR, base_name + ".pdf")
        review_path = os.path.join(REVIEWS_DIR, filename)

        if not os.path.exists(pdf_path):
            continue

        with open(review_path, "r", encoding="utf-8") as f:
            review = json.load(f)
        reviews = review.get("reviews", [])
        if not reviews:
            continue

        selected_review = reviews[0]

        example = dspy.Example(
            review=selected_review,
            user_prompt=DEFAULT_USER_PROMPT,
            pdf_path=pdf_path,
            paper_id=base_name
        ).with_inputs("user_prompt")

        human_reviews.append(example)

        if len(human_reviews) >= limit:
            break

    print(f" Načítaných {len(human_reviews)} párov review + pdf.")
    return human_reviews


# -------------------------------------------------------------------------
#   Extrakcia textov, skóre a rozhodnutí z ľudských recenzií
# -------------------------------------------------------------------------
def extract_human_review_data(human_reviews):
    data = {"texts": [], "scores": [], "decisions": []}

    for sample in human_reviews:
        review_data = sample["review"]
        if not isinstance(review_data, dict):
            continue

        text = review_data.get("comments", "")
        data["texts"].append(text)

        rec = review_data.get("RECOMMENDATION", review_data.get("recommendation", 0))
        try:
            score_val = float(rec)
            score_val = max(1.0, min(10.0, score_val))
        except (ValueError, TypeError):
            score_val = 5.0
        data["scores"].append(score_val)

        decision = "Accept" if score_val >= 6.0 else "Reject"
        data["decisions"].append(decision)

    print(f" Extracted {len(data['texts'])} reviews with scores and decisions")
    return data


# -------------------------------------------------------------------------
#   DSPy Signature: generovanie recenzií + skóre + rozhodnutie
# -------------------------------------------------------------------------
class DocumentReviews(dspy.Signature):
    """Review an academic paper. Generate a detailed review, recommend a score, and predict acceptance."""
    document: Attachments = dspy.InputField()
    reviews: str = dspy.OutputField(desc="Full structured review text.")
    predicted_score: float = dspy.OutputField(desc="Recommendation score from 1.0 (strong reject) to 10.0 (strong accept).")
    predicted_decision: str = dspy.OutputField(desc="Final decision: exactly 'Accept' or 'Reject'.")


# -------------------------------------------------------------------------
#   Generovanie recenzií pomocou dspy + Attachments
# -------------------------------------------------------------------------
def generate_reviews(human_reviews):
    results = []
    dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano-fiit", api_key=API_KEY, api_base=API_BASE, temperature=0.0, max_tokens=4000), track_usage=True)
    chain = dspy.Predict(DocumentReviews)

    for sample in human_reviews:
        pdf_path = sample.pdf_path
        try:
            attachment = Attachments(pdf_path)
            result = chain(document=attachment)

            try:
                pred_score = float(result.predicted_score)
                pred_score = max(1.0, min(10.0, pred_score))
            except (ValueError, TypeError):
                pred_score = 5.0

            pred_decision = str(result.predicted_decision).strip()
            if "accept" not in pred_decision.lower() and "reject" not in pred_decision.lower():
                pred_decision = "Accept" if pred_score >= 6.0 else "Reject"

            results.append({
                "review": result.reviews,
                "score": pred_score,
                "decision": pred_decision,
                "paper_id": sample.paper_id
            })
            print(f" [{sample.paper_id}] score={pred_score:.1f}, decision={pred_decision}")

        except Exception as e:
            print(f" Error processing {pdf_path}: {e}")

    return results


# -------------------------------------------------------------------------
#   Výpočet metrik
# -------------------------------------------------------------------------
def compute_text_metrics(generated_reviews, human_texts):
    rougel = load("rouge")
    rouge_scores = []
    bert_scores = []
    length_diffs = []

    for i, (result, hum_rev) in enumerate(zip(generated_reviews, human_texts)):
        gen_rev = result["review"]

        rouge = rougel.compute(predictions=[gen_rev], references=[hum_rev])
        rouge_scores.append(rouge["rougeL"])

        _, _, F1 = score([gen_rev], [hum_rev], lang="en", verbose=False)
        bert_scores.append(float(F1.item()))

        gen_len = len(gen_rev.split())
        hum_len = len(hum_rev.split())
        length_diffs.append(abs(gen_len - hum_len))

        print(f"  Article {i+1} [{result['paper_id']}]: ROUGE-L={rouge['rougeL']:.4f}, BERTScore={F1.item():.4f}, LenDiff={abs(gen_len - hum_len)}")

    return rouge_scores, bert_scores, length_diffs


def compute_token_cost(lm):
    total_input = 0
    total_output = 0
    total_cost_litellm = 0.0
    for call in lm.history:
        usage = call.get("usage", {})
        total_input += usage.get("prompt_tokens", 0)
        total_output += usage.get("completion_tokens", 0)
        total_cost_litellm += call.get("cost", 0) or 0
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "cost_litellm": total_cost_litellm,
    }


def compute_decision_metrics(generated_reviews, human_data):
    pred_scores = [r["score"] for r in generated_reviews]
    pred_decisions = [r["decision"] for r in generated_reviews]
    human_scores = human_data["scores"]
    human_decisions = human_data["decisions"]

    n = min(len(pred_scores), len(human_scores))
    pred_scores = pred_scores[:n]
    pred_decisions = pred_decisions[:n]
    human_scores = human_scores[:n]
    human_decisions = human_decisions[:n]

    y_true = [1 if "accept" in d.lower() else 0 for d in human_decisions]
    y_pred = [1 if "accept" in d.lower() else 0 for d in pred_decisions]

    # Spearman
    corr, p_val = spearmanr(pred_scores, human_scores) if n > 2 else (float("nan"), float("nan"))

    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, pred_scores)
    except ValueError:
        auc = float("nan")

    # MAE and RMSE on recommendation scores
    mae = mean_absolute_error(human_scores, pred_scores)
    rmse = mean_squared_error(human_scores, pred_scores) ** 0.5

    return {
        "spearman": corr,
        "spearman_p": p_val,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc,
        "mae": mae,
        "rmse": rmse,
    }


# -------------------------------------------------------------------------
#   Hlavný workflow
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("FIRST EXPERIMENT - Baseline Review Generation")
    print("=" * 60)

    human_reviews = load_human_reviews(limit=70)
    if not human_reviews:
        print(" No reviews loaded. Check REVIEWS_DIR and PDF_DIR paths.")
        exit(1)

    human_data = extract_human_review_data(human_reviews)

    print(f"\n Generating reviews for {len(human_reviews)} papers...")
    generated_reviews = generate_reviews(human_reviews)

    if not generated_reviews:
        print(" No reviews generated.")
        exit(1)

    # Token cost
    token_stats = compute_token_cost(dspy.settings.lm)
    n = len(generated_reviews)
    print(f"\n Token usage: input={token_stats['input_tokens']:,}  output={token_stats['output_tokens']:,}  total={token_stats['total_tokens']:,}")
    print(f" Cost (LiteLLM):     ${token_stats['cost_litellm']:.4f}")
    if n > 0:
        print(f" Cost per article:   ${token_stats['cost_litellm']/n:.4f}")

    # Align human texts with successfully generated reviews
    paper_ids_generated = {r["paper_id"] for r in generated_reviews}
    human_texts_aligned = [
        human_data["texts"][i]
        for i, sample in enumerate(human_reviews)
        if sample.paper_id in paper_ids_generated
    ]
    human_data_aligned = {
        "scores": [human_data["scores"][i] for i, s in enumerate(human_reviews) if s.paper_id in paper_ids_generated],
        "decisions": [human_data["decisions"][i] for i, s in enumerate(human_reviews) if s.paper_id in paper_ids_generated],
    }

    # -------------------------------------------------------------------------
    #   Text Quality Metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEXT QUALITY METRICS")
    print("=" * 60)
    rouge_scores, bert_scores, length_diffs = compute_text_metrics(generated_reviews, human_texts_aligned)

    print("\n--- Averages ---")
    print(f"Avg ROUGE-L:     {np.mean(rouge_scores):.4f}")
    print(f"Avg BERTScore:   {np.mean(bert_scores):.4f}")
    print(f"Avg Length Diff: {np.mean(length_diffs):.1f} words")

    # -------------------------------------------------------------------------
    #   Decision Metrics (Spearman, Accuracy, AUC-ROC)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DECISION METRICS")
    print("=" * 60)

    metrics = compute_decision_metrics(generated_reviews, human_data_aligned)
    print(f"Spearman Correlation: {metrics['spearman']:.4f} (p={metrics['spearman_p']:.4f})")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall:               {metrics['recall']:.4f}")
    print(f"F1 Score:             {metrics['f1']:.4f}")
    print(f"AUC-ROC:              {metrics['auc_roc']:.4f}")
    print(f"MAE:                  {metrics['mae']:.4f}")
    print(f"RMSE:                 {metrics['rmse']:.4f}")

    print("=" * 60)
