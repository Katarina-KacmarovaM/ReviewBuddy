"""Downloads and prepares the ICLR paper+review dataset from OpenReview API into local train/test splits."""

from openreview.api import OpenReviewClient as Client
import os
import json
import re
import random
import time
import threading
from collections import Counter

import requests


# =============================================================================
# CONFIG
# =============================================================================
TRAIN_REVIEWS = "C:/Users/katka/BAKALARKA/train_reviews3"
TRAIN_PDFS = "C:/Users/katka/BAKALARKA/train_pdf3"
TEST_REVIEWS = "C:/Users/katka/BAKALARKA/test_reviews3"
TEST_PDFS = "C:/Users/katka/BAKALARKA/test_pdfs3"

for d in [TRAIN_REVIEWS, TRAIN_PDFS, TEST_REVIEWS, TEST_PDFS]:
    os.makedirs(d, exist_ok=True)

ICLR_SCORES = [1, 3, 5, 6, 8, 10]
SEED = 42
random.seed(SEED)

TARGET_TRAIN = 150
TARGET_TEST = 150

OPENREVIEW_BASEURL = "https://api2.openreview.net"
OPENREVIEW_USERNAME = "xkacmarovamadar@stuba.sk"
OPENREVIEW_PASSWORD = "KAWyrm62GBej-H-"


# =============================================================================
# HELPERS
# =============================================================================
MIN_PAGES = 8
MAX_PAGES = 50
PREFERRED_MAX_PAGES = 30  # prefer papers under this length; longer ones used as fallback


def get_pdf_page_count(pdf_bytes):
    """
    Count PDF pages without a library.
    1. Try counting /Type /Page objects (works for uncompressed PDFs)
    2. Fallback: read /Count from the Pages tree (always uncompressed, works for PDF 1.5+ compressed streams too)
    """
    # Method 1: count individual page objects
    count = len(re.findall(rb'/Type\s*/Page[\s/>\r\n]', pdf_bytes))
    if count > 0:
        return count

    # Method 2: /Count N in the Pages dictionary (root node has the total)
    matches = re.findall(rb'/Count\s+(\d+)', pdf_bytes)
    if matches:
        return max(int(m) for m in matches)

    return None


def build_review_obj(reply):
    """Build a normalized review object used in exported JSON files."""
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


# =============================================================================
# FETCH DATA
# =============================================================================
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

        individual_scores = []
        for s, cnt in score_counts.items():
            individual_scores.extend([s] * cnt)

        pool.append({
            "sub": sub,
            "decision": decision,
            "avg_score": avg_score,
            "recommendation_scores": sorted(score_counts.keys()),
            "review_score_counts": dict(score_counts),
            "individual_scores": individual_scores,  # cached for greedy
            "review_count": sum(score_counts.values()),
            "reviews": reviews,
            "year": year,
        })

    return pool


def filter_noisy(pool):
    """
    Remove papers where decision contradicts review scores:
    - Accept + avg_score < 5  → reviewers disagree with accept
    - Reject + avg_score > 7  → reviewers disagree with reject
    """
    clean = []
    removed = 0
    for p in pool:
        if p["decision"] == "Accept" and p["avg_score"] < 5:
            removed += 1
            continue
        if p["decision"] == "Reject" and p["avg_score"] > 7:
            removed += 1
            continue
        clean.append(p)
    print(f"  Filtered out {removed} noisy papers ({len(clean)} remaining)")
    return clean


# =============================================================================
# WEIGHTED GREEDY DOWNLOADER
# =============================================================================
def paper_weight(paper, score_counts, target_per_score):
    """
    Combined weight = distribution_weight * agreement_bonus.
    - distribution_weight: how much this paper fills underrepresented scores
    - agreement_bonus: 1/(1+std_dev) — higher when all reviewers agree on the same score
    """
    scores = paper["individual_scores"]

    # Primary: hard cap — once score reaches target_per_score, contributes 0
    distribution_weight = sum(max(0.0, 1 - score_counts[s] / target_per_score) for s in scores if s in score_counts)

    # Fallback: if all scores saturated, still prefer least-represented
    if distribution_weight == 0.0:
        distribution_weight = sum(1 / (score_counts[s] + 1) for s in scores if s in score_counts) * 0.001

    # Agreement bonus: prefer papers where all reviewers agree (low std_dev)
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5
    agreement_bonus = 1 / (1 + std_dev)

    return distribution_weight * agreement_bonus


def compute_target_per_score(pool, target):
    """Calculate target reviews per score based on actual avg reviews in pool."""
    if not pool:
        return 100
    avg_reviews = sum(p["review_count"] for p in pool) / len(pool)
    return int(target * avg_reviews / len(ICLR_SCORES))


def select_and_download(pool, target, pdf_dir, reviews_dir, label, start_counter=0, score_counts=None):
    """
    Greedy weighted selection. If score_counts is passed in, continues
    filling from where a previous run left off (used to chain 2025 → 2024).
    """
    if score_counts is None:
        score_counts = {s: 0 for s in ICLR_SCORES}

    target_per_score = compute_target_per_score(pool, target)
    print(f"  Target per score: {target_per_score} (avg {sum(p['review_count'] for p in pool)/len(pool):.1f} reviews/paper)")

    selected = []
    remaining = list(pool)
    long_buffer = []  # papers that passed filters but exceeded PREFERRED_MAX_PAGES
    counter = start_counter

    def save_paper(paper, pdf_bytes, page_count):
        nonlocal counter
        clean_reviews = build_clean_reviews(paper["reviews"])
        if not clean_reviews:
            print(f"    Skipped {paper['sub'].id}: no valid reviews")
            return False

        sub_id = paper["sub"].id
        tmp_pdf_path = f"{pdf_dir}/{sub_id}_tmp.pdf"
        with open(tmp_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        counter += 1
        final_pdf_path = f"{pdf_dir}/{counter}.pdf"
        os.replace(tmp_pdf_path, final_pdf_path)

        payload = build_article_payload(paper, page_count, clean_reviews)
        with open(f"{reviews_dir}/{counter}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        for s in paper["individual_scores"]:
            if s in score_counts:
                score_counts[s] += 1

        selected.append(paper)
        dist_str = " | ".join(f"{s}:{score_counts[s]}" for s in ICLR_SCORES)
        print(f"  [{label}] {len(selected)}/{target} | {page_count}p | scores: {paper['individual_scores']} | [{dist_str}]")
        time.sleep(1)
        return True

    while len(selected) < target and (remaining or long_buffer):
        # exhaust short papers first; fall back to long buffer when remaining is empty
        if remaining:
            remaining.sort(key=lambda p: paper_weight(p, score_counts, target_per_score), reverse=True)
            paper = remaining.pop(0)
        else:
            long_buffer.sort(key=lambda x: paper_weight(x[0], score_counts, target_per_score), reverse=True)
            paper, pdf_bytes, page_count = long_buffer.pop(0)
            save_paper(paper, pdf_bytes, page_count)
            continue

        sub_id = paper["sub"].id
        try:
            print(f"  Downloading {sub_id}...")
            pdf_bytes = download_pdf(sub_id)

            page_count = get_pdf_page_count(pdf_bytes)
            if not page_count or page_count < MIN_PAGES or page_count > MAX_PAGES:
                print(f"    Skipped {sub_id}: {page_count} pages (limit {MIN_PAGES}-{MAX_PAGES})")
                continue

            if page_count > PREFERRED_MAX_PAGES and remaining:
                # defer long paper — try shorter ones first
                long_buffer.append((paper, pdf_bytes, page_count))
                print(f"    Deferred {sub_id}: {page_count} pages (prefer shorter)")
                continue

            save_paper(paper, pdf_bytes, page_count)

        except Exception as e:
            print(f"  Error on {sub_id}: {e}")

    print(f"\nScore distribution after {label}:")
    for s in ICLR_SCORES:
        print(f"  Score {s:2d}: {score_counts[s]:3d} reviews")

    return selected, counter, score_counts, {p["sub"].id for p in selected}


# =============================================================================
# EXECUTION
# =============================================================================
client = Client(
    baseurl=OPENREVIEW_BASEURL,
    username=OPENREVIEW_USERNAME,
    password=OPENREVIEW_PASSWORD,
)

print("Fetching ICLR 2025 submissions...")
submissions_2025 = client.get_all_notes(
    invitation="ICLR.cc/2025/Conference/-/Submission",
    details="directReplies",
)
papers_pool_2025 = parse_submissions(submissions_2025, 2025)
papers_pool_2025 = filter_noisy(papers_pool_2025)
print(f"  ICLR 2025: {len(papers_pool_2025)} papers after noise filter")

print("Fetching ICLR 2024 submissions...")
submissions_2024 = client.get_all_notes(
    invitation="ICLR.cc/2024/Conference/-/Submission",
    details="directReplies",
)
papers_pool_2024 = parse_submissions(submissions_2024, 2024)
papers_pool_2024 = filter_noisy(papers_pool_2024)
print(f"  ICLR 2024: {len(papers_pool_2024)} papers after noise filter")

print("Fetching ICLR 2023 submissions...")
submissions_2023 = client.get_all_notes(
    invitation="ICLR.cc/2023/Conference/-/Submission",
    details="directReplies",
)
papers_pool_2023 = parse_submissions(submissions_2023, 2023)
papers_pool_2023 = filter_noisy(papers_pool_2023)
print(f"  ICLR 2023: {len(papers_pool_2023)} papers after noise filter")

combined_pool = papers_pool_2025 + papers_pool_2024 + papers_pool_2023
random.shuffle(combined_pool)
print(f"  Combined pool: {len(combined_pool)} papers total")

# =============================================================================
# TRAIN
# =============================================================================
print(f"\n{'=' * 60}\nTRAIN ({TARGET_TRAIN} papers)\n{'=' * 60}")
saved_train, counter, train_score_counts, train_used_ids = select_and_download(
    combined_pool, TARGET_TRAIN, TRAIN_PDFS, TRAIN_REVIEWS, "TRAIN"
)

# =============================================================================
# TEST: from remaining papers not used in train
# =============================================================================
test_pool = [p for p in combined_pool if p["sub"].id not in train_used_ids]

print(f"\n{'=' * 60}\nTEST ({TARGET_TEST} papers)\n{'=' * 60}")
saved_test, counter, test_score_counts, _ = select_and_download(
    test_pool, TARGET_TEST, TEST_PDFS, TEST_REVIEWS, "TEST",
    start_counter=counter
)


# =============================================================================
# FINAL VALIDATION
# =============================================================================
def count_pdfs(directory):
    return len([name for name in os.listdir(directory) if name.endswith(".pdf")])


print(f"\n{'=' * 60}\nFINAL COUNT CHECK\n{'=' * 60}")
print(f"TRAIN FOLDER: {count_pdfs(TRAIN_PDFS)} PDFs (Target: {TARGET_TRAIN})")
print(f"TEST FOLDER:  {count_pdfs(TEST_PDFS)} PDFs (Target: {TARGET_TEST})")
