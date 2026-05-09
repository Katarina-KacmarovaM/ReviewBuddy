"""
Download additional papers with score 1 or score 10 from ICLR 2025/2024/2023,
skipping any already present in train_reviews3, and append them starting from
the next available number.
"""
from openreview.api import OpenReviewClient as Client
import os
import json
import re
import time
import threading
from collections import Counter

# =============================================================================
# CONFIG
# =============================================================================
TRAIN_REVIEWS = "C:/Users/katka/BAKALARKA/train_reviews3"
TRAIN_PDFS = "C:/Users/katka/BAKALARKA/train_pdf3"

TARGET_SCORES = {1, 10}          # only papers that have at least one of these
TARGET_NEW = 40                   # stop after adding this many new papers
MIN_PAGES = 8
MAX_PAGES = 50
PREFERRED_MAX_PAGES = 30

OPENREVIEW_BASEURL = "https://api2.openreview.net"
OPENREVIEW_USERNAME = "xkacmarovamadar@stuba.sk"
OPENREVIEW_PASSWORD = "KAWyrm62GBej-H-"


# =============================================================================
# HELPERS (same as priprava.py)
# =============================================================================
def get_pdf_page_count(pdf_bytes):
    count = len(re.findall(rb'/Type\s*/Page[\s/>\r\n]', pdf_bytes))
    if count > 0:
        return count
    matches = re.findall(rb'/Count\s+(\d+)', pdf_bytes)
    if matches:
        return max(int(m) for m in matches)
    return None


def build_review_obj(reply):
    content = reply.get("content", {})
    if "rating" not in content:
        return None
    rating_value = content["rating"].get("value") if isinstance(content.get("rating"), dict) else content.get("rating")
    text_value = ""
    for field in ["review", "summary", "comment"]:
        if field in content:
            v = content[field].get("value") if isinstance(content[field], dict) else content[field]
            if v:
                text_value = v
                break
    if not text_value or len(str(text_value)) < 50:
        return None

    def get_field(field_name):
        if field_name not in content:
            return ""
        v = content[field_name].get("value") if isinstance(content[field_name], dict) else content[field_name]
        return str(v) if v else ""

    return {
        "DATE": str(reply.get("cdate")),
        "RECOMMENDATION": str(rating_value),
        "comments": text_value,
        "IS_META_REVIEW": False,
        "IS_ANNOTATED": True,
        "REVIEWER_CONFIDENCE": get_field("confidence"),
        "SOUNDNESS_CORRECTNESS": get_field("soundness"),
        "CLARITY": get_field("presentation"),
        "STRENGTHS": get_field("strengths"),
        "WEAKNESSES": get_field("weaknesses"),
    }


def build_clean_reviews(reviews):
    revs_clean = [r for r in (build_review_obj(rv) for rv in reviews) if r]
    for idx, rev in enumerate(revs_clean, start=1):
        rev["review_id"] = idx
    return revs_clean


def build_article_payload(paper, page_count, clean_reviews):
    sub = paper["sub"]
    content = sub.content
    title = content.get("title", {})
    title = title.get("value") if isinstance(title, dict) else title
    abstract = content.get("abstract", {})
    abstract = abstract.get("value") if isinstance(abstract, dict) else abstract
    authors = content.get("authors", {})
    authors = authors.get("value") if isinstance(authors, dict) else authors
    if not isinstance(authors, list):
        authors = []
    return {
        "conference": f"ICLR {paper['year']} Conference",
        "id": sub.id,
        "title": str(title) if title else "",
        "abstract": str(abstract) if abstract else "",
        "authors": authors,
        "accepted": paper["decision"] == "Accept",
        "decision": paper["decision"],
        "average_score": round(paper["avg_score"], 1),
        "recommendation_scores": paper["recommendation_scores"],
        "review_score_counts": paper["review_score_counts"],
        "review_count": paper["review_count"],
        "page_count": page_count,
        "reviews": clean_reviews,
    }


def download_pdf(sub_id, timeout=60):
    result = [None]
    error = [None]

    def _fetch():
        try:
            result[0] = client.get_pdf(id=sub_id)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"Download exceeded {timeout}s for {sub_id}")
    if error[0]:
        raise error[0]
    return result[0]


def parse_submissions(submissions, year):
    pool = []
    for sub in submissions:
        reviews = []
        decision = None
        for reply in sub.details.get("directReplies", []):
            invitation = reply.get("invitations", [""])[0]
            if "Review" in invitation:
                reviews.append(reply)
            if "Decision" in invitation:
                decision_field = reply.get("content", {}).get("decision", {})
                decision_value = decision_field.get("value") if isinstance(decision_field, dict) else decision_field
                if decision_value:
                    decision = "Accept" if any(x in decision_value for x in ["Accept", "Oral", "Spotlight"]) else "Reject"
        if not decision or not reviews:
            continue
        scores = []
        for review in reviews:
            rating = review.get("content", {}).get("rating")
            if rating is None:
                continue
            rating_value = rating.get("value") if isinstance(rating, dict) else rating
            m = re.search(r"\d+", str(rating_value))
            if m:
                scores.append(int(m.group()))
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        score_counts = Counter(scores)
        pool.append({
            "sub": sub,
            "decision": decision,
            "avg_score": avg_score,
            "recommendation_scores": sorted(score_counts.keys()),
            "review_score_counts": dict(score_counts),
            "individual_scores": scores,
            "review_count": sum(score_counts.values()),
            "reviews": reviews,
            "year": year,
        })
    return pool


# =============================================================================
# LOAD EXISTING IDs FROM train_reviews3
# =============================================================================
def load_existing_ids(directory):
    ids = set()
    for fname in os.listdir(directory):
        if fname.endswith(".json"):
            with open(os.path.join(directory, fname), encoding="utf-8") as f:
                data = json.load(f)
            ids.add(data.get("id"))
    return ids


def next_counter(directory):
    nums = []
    for fname in os.listdir(directory):
        if fname.endswith(".json"):
            try:
                nums.append(int(fname.replace(".json", "")))
            except ValueError:
                pass
    return max(nums) + 1 if nums else 1


# =============================================================================
# EXECUTION
# =============================================================================
client = Client(
    baseurl=OPENREVIEW_BASEURL,
    username=OPENREVIEW_USERNAME,
    password=OPENREVIEW_PASSWORD,
)

existing_ids = load_existing_ids(TRAIN_REVIEWS)
print(f"Existing papers in train_reviews3: {len(existing_ids)}")

counter = next_counter(TRAIN_REVIEWS)
print(f"Next counter: {counter}")

pool = []
for year, invitation in [
    (2025, "ICLR.cc/2025/Conference/-/Submission"),
    (2024, "ICLR.cc/2024/Conference/-/Submission"),
    (2023, "ICLR.cc/2023/Conference/-/Submission"),
]:
    print(f"Fetching ICLR {year}...")
    submissions = client.get_all_notes(invitation=invitation, details="directReplies")
    parsed = parse_submissions(submissions, year)
    print(f"  {len(parsed)} papers parsed")
    pool.extend(parsed)

# Filter: score 1/10 must be majority (strictly more than other scores)
# max 1 non-target review allowed per paper
candidates = [
    p for p in pool
    if p["sub"].id not in existing_ids
    and sum(1 for s in p["individual_scores"] if s in TARGET_SCORES) > sum(1 for s in p["individual_scores"] if s not in TARGET_SCORES)
    and sum(1 for s in p["individual_scores"] if s not in TARGET_SCORES) <= 1
]

# Sort: prefer papers with more TARGET_SCORES reviews (most 1s/10s first)
candidates.sort(
    key=lambda p: sum(1 for s in p["individual_scores"] if s in TARGET_SCORES),
    reverse=True,
)

print(f"\nCandidates with score 1 or 10 not in dataset: {len(candidates)}")
print(f"Downloading up to {TARGET_NEW} papers...\n")

added = 0
long_buffer = []
remaining = list(candidates)

def save_paper(paper, pdf_bytes, page_count):
    global counter, added
    clean_reviews = build_clean_reviews(paper["reviews"])
    if not clean_reviews:
        print(f"    Skipped {paper['sub'].id}: no valid reviews")
        return False

    sub_id = paper["sub"].id
    tmp_pdf_path = f"{TRAIN_PDFS}/{sub_id}_tmp.pdf"
    with open(tmp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    final_pdf_path = f"{TRAIN_PDFS}/{counter}.pdf"
    os.replace(tmp_pdf_path, final_pdf_path)

    payload = build_article_payload(paper, page_count, clean_reviews)
    with open(f"{TRAIN_REVIEWS}/{counter}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    scores = paper["individual_scores"]
    target_scores_in_paper = [s for s in scores if s in TARGET_SCORES]
    print(f"  [{added+1}/{TARGET_NEW}] {counter}.json | {page_count}p | scores={scores} | target={target_scores_in_paper}")
    counter += 1
    added += 1
    time.sleep(1)
    return True


while added < TARGET_NEW and (remaining or long_buffer):
    if remaining:
        paper = remaining.pop(0)
    else:
        paper, pdf_bytes, page_count = long_buffer.pop(0)
        save_paper(paper, pdf_bytes, page_count)
        continue

    sub_id = paper["sub"].id
    try:
        print(f"  Downloading {sub_id}...")
        pdf_bytes = download_pdf(sub_id)
        page_count = get_pdf_page_count(pdf_bytes)

        if not page_count or page_count < MIN_PAGES or page_count > MAX_PAGES:
            print(f"    Skipped {sub_id}: {page_count} pages")
            continue

        if page_count > PREFERRED_MAX_PAGES and remaining:
            long_buffer.append((paper, pdf_bytes, page_count))
            print(f"    Deferred {sub_id}: {page_count} pages")
            continue

        save_paper(paper, pdf_bytes, page_count)

    except Exception as e:
        print(f"  Error on {sub_id}: {e}")

print(f"\nDone. Added {added} new papers.")
print(f"Total papers now: {len(os.listdir(TRAIN_REVIEWS))}")
