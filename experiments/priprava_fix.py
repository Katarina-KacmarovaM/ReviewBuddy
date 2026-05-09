"""
Oprava reviews pre existujúce PDFs v new_human_reviews_GEPA100.
Overí decision priamo z OpenReview API a prefiltruje podľa strán.
"""
import json
from pathlib import Path
from openreview.api import OpenReviewClient as Client

# Zdroje
REVIEWS_DIR = Path("C:/Users/katka/BAKALARKA/human_reviews_2")

client = Client(baseurl='https://api2.openreview.net')

# 1. Načítaj existujúce reviews
review_files = list(REVIEWS_DIR.glob("*.json"))
print(f"Načítaných reviews: {len(review_files)}")

# 2. Pre každý článok over decision z API
results = []
skipped_no_decision = 0
skipped_no_score = 0
fixed_decisions = 0

for i, review_file in enumerate(sorted(review_files)):
    article_id = review_file.stem

    # Načítaj existujúci JSON
    with open(review_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    old_decision = data.get("decision", None)
    score = data.get("average_score", 0)

    if not score or score == 0:
        skipped_no_score += 1
        continue

    # Over decision z OpenReview API
    try:
        note = client.get_note(article_id)
        replies = client.get_all_notes(forum=article_id)

        real_decision = None
        for reply in replies:
            invitations = reply.invitations if hasattr(reply, 'invitations') else []
            for inv in invitations:
                if 'Decision' in inv:
                    raw = reply.content.get('decision', {})
                    if isinstance(raw, dict):
                        raw = raw.get('value', '')
                    if "Accept" in str(raw) or "Oral" in str(raw) or "Spotlight" in str(raw):
                        real_decision = "Accept"
                    else:
                        real_decision = "Reject"

        if real_decision is None:
            print(f"  [{i+1}] {article_id}: NO DECISION from API (score={score:.2f}, old={old_decision})")
            skipped_no_decision += 1
            continue

        if real_decision != old_decision:
            print(f"  [{i+1}] {article_id}: FIXED {old_decision} -> {real_decision} (score={score:.2f})")
            fixed_decisions += 1
        else:
            print(f"  [{i+1}] {article_id}: OK {real_decision} (score={score:.2f})")

        # Updatuj data
        data["decision"] = real_decision
        data["accepted"] = real_decision == "Accept"

        results.append({
            "article_id": article_id,
            "decision": real_decision,
            "score": score,
            "data": data
        })

    except Exception as e:
        print(f"  [{i+1}] {article_id}: API ERROR - {e}")
        continue

# 3. Štatistiky
accept = [r for r in results if r["decision"] == "Accept"]
reject = [r for r in results if r["decision"] == "Reject"]

print(f"\n{'='*60}")
print("VÝSLEDKY PO OVERENÍ:")
print(f"  Accept: {len(accept)}")
print(f"  Reject: {len(reject)}")
print(f"  Total:  {len(results)}")
print(f"  Fixed decisions: {fixed_decisions}")
print(f"\nSkipped:")
print(f"  No score:    {skipped_no_score}")
print(f"  No decision: {skipped_no_decision}")
print(f"{'='*60}")

# 4. Ulož opravené reviews na miesto
for item in results:
    review_path = REVIEWS_DIR / f"{item['article_id']}.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(item["data"], f, indent=2, ensure_ascii=False)

print(f"\nOpravené reviews uložené do: {REVIEWS_DIR} ({len(results)} files)")
