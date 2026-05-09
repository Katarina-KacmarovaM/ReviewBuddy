"""
Reduce over-represented scores in train_reviews3:
  - Score 6: papers with >4 score-6 reviews → keep only 4
  - Score 8: papers where ALL reviews are 8 AND have 5+ reviews → keep only 4
Recalculates average_score, review_count, recommendation_scores, review_score_counts.
"""
import json
import os
import re
from collections import Counter


def parse_score(value):
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None

REVIEWS_DIR = "C:/Users/katka/BAKALARKA/train_reviews3"

files = sorted(os.listdir(REVIEWS_DIR), key=lambda x: int(x.replace(".json", "")))

total_removed_6 = 0
total_removed_8 = 0
papers_changed = 0

for fname in files:
    path = os.path.join(REVIEWS_DIR, fname)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    reviews = data["reviews"]
    original_count = len(reviews)
    changed = False

    # --- Score 6 reduction ---
    score6_reviews = [r for r in reviews if parse_score(r["RECOMMENDATION"]) == 6]
    if len(score6_reviews) > 3:
        # Keep only 3 score-6 reviews (keep first 3), remove the rest
        to_remove = set(r["review_id"] for r in score6_reviews[3:])
        removed = len(to_remove)
        reviews = [r for r in reviews if r["review_id"] not in to_remove]
        total_removed_6 += removed
        changed = True
        print(f"  {fname}: score-6 {len(score6_reviews)} -> 3 (removed {removed})")

    # --- Score 8 reduction: only all-8 papers with 4+ reviews ---
    all_scores = [parse_score(r["RECOMMENDATION"]) for r in reviews]
    if all(s == 8 for s in all_scores) and len(reviews) >= 4:
        excess = len(reviews) - 3
        reviews = reviews[:3]
        total_removed_8 += excess
        changed = True
        print(f"  {fname}: all-8 paper {len(all_scores)} -> 3 (removed {excess})")

    if changed:
        papers_changed += 1
        # Recalculate metadata
        scores = [parse_score(r["RECOMMENDATION"]) for r in reviews]
        score_counts = dict(Counter(scores))
        data["reviews"] = reviews
        data["review_count"] = len(reviews)
        data["average_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0
        data["recommendation_scores"] = sorted(score_counts.keys())
        data["review_score_counts"] = {str(k): v for k, v in score_counts.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\nDone. Papers changed: {papers_changed}")
print(f"Score-6 reviews removed: {total_removed_6}")
print(f"Score-8 reviews removed: {total_removed_8}")

# Final distribution
score_totals = Counter()
for fname in files:
    path = os.path.join(REVIEWS_DIR, fname)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data["reviews"]:
        s = parse_score(r["RECOMMENDATION"])
        if s is not None:
            score_totals[s] += 1

print("\nFinal score distribution:")
for s in sorted(score_totals):
    print(f"  Score {s:2d}: {score_totals[s]:3d} reviews")
