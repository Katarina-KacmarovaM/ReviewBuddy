"""
Augment score-1 and score-10 reviews by paraphrasing existing ones using Claude.
Adds paraphrased reviews to the same paper as additional reviewers.
"""
import json
import os
import re
import random
import shutil
import anthropic
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

REVIEWS_DIR = "C:/Users/katka/BAKALARKA/train_reviews3"
BACKUP_DIR = "C:/Users/katka/BAKALARKA/train_reviews3_pre_augment"

TARGET_SCORES = {1, 10}
TARGET_ADD = {1: 20, 10: 35}

SEED = 42
random.seed(SEED)

claude = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


def parse_score(value):
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def paraphrase_review(review: dict) -> dict | None:
    """Use Claude Haiku to paraphrase a review, keeping the same score and key points."""
    score = review["RECOMMENDATION"]
    comments = review.get("comments", "")
    strengths = review.get("STRENGTHS", "")
    weaknesses = review.get("WEAKNESSES", "")

    prompt = f"""You are paraphrasing an academic paper review. Rewrite the review below using different wording while keeping the same score ({score}), the same key points, and the same overall sentiment. Do NOT change the recommendation score.

Return a JSON object with exactly these fields:
- "comments": paraphrased main review text
- "STRENGTHS": paraphrased strengths (keep empty string if original is empty)
- "WEAKNESSES": paraphrased weaknesses (keep empty string if original is empty)

Original review:
SCORE: {score}
COMMENTS: {comments}
STRENGTHS: {strengths}
WEAKNESSES: {weaknesses}

Return only valid JSON, no extra text."""

    try:
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)

        return {
            "DATE": review["DATE"],
            "RECOMMENDATION": review["RECOMMENDATION"],
            "comments": parsed.get("comments", ""),
            "IS_META_REVIEW": False,
            "IS_ANNOTATED": True,
            "REVIEWER_CONFIDENCE": review.get("REVIEWER_CONFIDENCE", ""),
            "SOUNDNESS_CORRECTNESS": review.get("SOUNDNESS_CORRECTNESS", ""),
            "CLARITY": review.get("CLARITY", ""),
            "STRENGTHS": parsed.get("STRENGTHS", ""),
            "WEAKNESSES": parsed.get("WEAKNESSES", ""),
        }
    except Exception as e:
        print(f"    Paraphrase error: {e}")
        return None


# =============================================================================
# COLLECT existing reviews by score
# =============================================================================
reviews_by_score = {1: [], 10: []}

for fname in os.listdir(REVIEWS_DIR):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(REVIEWS_DIR, fname), encoding="utf-8") as f:
        data = json.load(f)
    for r in data["reviews"]:
        s = parse_score(r["RECOMMENDATION"])
        if s in reviews_by_score:
            reviews_by_score[s].append((fname, r))

print("Existing reviews:")
for s in TARGET_SCORES:
    print(f"  Score {s}: {len(reviews_by_score[s])} reviews")

# =============================================================================
# BACKUP files that will be modified
# =============================================================================
files_to_modify = set(fname for score in TARGET_SCORES for fname, _ in reviews_by_score[score])
os.makedirs(BACKUP_DIR, exist_ok=True)
backed_up = 0
for fname in files_to_modify:
    dst = os.path.join(BACKUP_DIR, fname)
    if not os.path.exists(dst):
        shutil.copy2(os.path.join(REVIEWS_DIR, fname), dst)
        backed_up += 1
print(f"Backed up {backed_up} new files to {BACKUP_DIR} ({len(os.listdir(BACKUP_DIR))} total)")

# =============================================================================
# AUGMENT
# =============================================================================
added_counts = {1: 0, 10: 0}

for score, n_to_add in TARGET_ADD.items():
    print(f"\n--- Augmenting score {score}: adding {n_to_add} paraphrased reviews ---")

    pool = reviews_by_score[score]
    if not pool:
        print(f"  No existing reviews for score {score}, skipping.")
        continue

    candidates = random.choices(pool, k=n_to_add)

    for i, (fname, original_review) in enumerate(candidates):
        path = os.path.join(REVIEWS_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        print(f"  [{i+1}/{n_to_add}] Paraphrasing score-{score} from {fname}...")
        new_review = paraphrase_review(original_review)
        if not new_review or len(new_review.get("comments", "")) < 50:
            print(f"    Skipped: too short or failed")
            continue

        existing_ids = [r["review_id"] for r in data["reviews"]]
        new_review["review_id"] = max(existing_ids) + 1

        data["reviews"].append(new_review)

        scores = [parse_score(r["RECOMMENDATION"]) for r in data["reviews"]]
        score_counts = dict(Counter(scores))
        data["review_count"] = len(data["reviews"])
        data["average_score"] = round(sum(scores) / len(scores), 1)
        data["recommendation_scores"] = sorted(score_counts.keys())
        data["review_score_counts"] = {str(k): v for k, v in score_counts.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        added_counts[score] += 1
        print(f"    OK -> review_id={new_review['review_id']}")

# =============================================================================
# FINAL DISTRIBUTION
# =============================================================================
score_totals = Counter()
for fname in os.listdir(REVIEWS_DIR):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(REVIEWS_DIR, fname), encoding="utf-8") as f:
        data = json.load(f)
    for r in data["reviews"]:
        s = parse_score(r["RECOMMENDATION"])
        if s:
            score_totals[s] += 1

print(f"\nAdded: score-1={added_counts[1]}, score-10={added_counts[10]}")
print("\nFinal score distribution:")
for s in sorted(score_totals):
    print(f"  Score {s:2d}: {score_totals[s]:3d} reviews")
