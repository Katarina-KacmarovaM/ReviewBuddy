"""
Build a flattened reviewer-level DSPy dataset from train reviews and test reviews.

Each (paper, reviewer_idx) pair becomes one dspy.Example:
  - inputs:  pdf_path
  - labels:  human_comments, human_strengths, human_weaknesses, human_score
  - meta:    reviewer_idx, paper_id, title, decision

The number of reviewer slots equals the maximum review count across all papers.
"""

import json
import os
import re
import dspy
from typing import Optional


def _parse_score(value) -> Optional[int]:
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def build_flat_dataset(
    reviews_dir: str,
    pdfs_dir: str,
    md_dir: Optional[str] = None,
) -> dict[int, list[dspy.Example]]:
    """
    Returns a dict: {reviewer_idx: [dspy.Example, ...]}

    reviewer_idx 0 = 1st human review of each paper
    reviewer_idx 1 = 2nd human review of each paper
    ...up to (max reviews per paper - 1)

    If md_dir is provided, each example also contains article_text (the pre-converted
    markdown) and uses article_text as the DSPy input (for RLM extraction mode).
    Otherwise pdf_path is the input (for pageindex mode).
    """
    json_files = sorted(
        [f for f in os.listdir(reviews_dir) if f.endswith(".json")],
        key=lambda x: int(x.replace(".json", "")),
    )

    # First pass: find max reviewer count
    max_reviewers = 0
    for fname in json_files:
        with open(os.path.join(reviews_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        max_reviewers = max(max_reviewers, len(data.get("reviews", [])))

    print(f"  [dataset] Max reviews per paper: {max_reviewers} -> creating {max_reviewers} reviewer slots")

    by_reviewer: dict[int, list[dspy.Example]] = {i: [] for i in range(max_reviewers)}

    # Second pass: build examples
    for fname in json_files:
        paper_num = fname.replace(".json", "")
        pdf_path = os.path.join(pdfs_dir, f"{paper_num}.pdf")

        if not os.path.exists(pdf_path):
            print(f"  [dataset] Missing PDF for {fname}, skipping.")
            continue

        article_text = None
        if md_dir:
            md_path = os.path.join(md_dir, paper_num, f"{paper_num}.md")
            if not os.path.exists(md_path):
                print(f"  [dataset] Missing markdown for {paper_num}, skipping.")
                continue
            with open(md_path, encoding="utf-8") as f:
                article_text = f.read()

        with open(os.path.join(reviews_dir, fname), encoding="utf-8") as f:
            data = json.load(f)

        reviews = data.get("reviews", [])

        for idx, r in enumerate(reviews):
            score = _parse_score(r.get("RECOMMENDATION", ""))
            if score is None:
                continue

            ex = dspy.Example(
                pdf_path=pdf_path,
                article_text=article_text,
                reviewer_idx=idx,
                paper_id=data.get("id", ""),
                title=data.get("title", ""),
                decision=data.get("decision", ""),
                human_comments=r.get("comments", ""),
                human_strengths=r.get("STRENGTHS", ""),
                human_weaknesses=r.get("WEAKNESSES", ""),
                human_score=score,
            ).with_inputs(
                *(["article_text"] if md_dir else ["pdf_path"]),
                "human_comments", "human_strengths", "human_weaknesses", "human_score",
            )

            by_reviewer[idx].append(ex)

    for idx in range(max_reviewers):
        print(f"  Reviewer {idx}: {len(by_reviewer[idx])} examples")

    return by_reviewer
