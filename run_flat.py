"""
run_flat.py — main entrypoint for training and evaluating ReviewBuddy.

Modes (select via CLI flags):
  --gepa      Train one ReviewPipeline per reviewer slot using GEPA optimization.
              Each slot learns to mimic a different human reviewer style.
              Saves trained pipelines to flat_optimized/.

  --eval      Evaluate all trained reviewer slots on the test set.
              Runs all slots per paper, consolidates via Area Chair,
              and computes full metrics (MAE, RMSE, Spearman, BERTScore, ...).

  --baseline  Run a single unoptimized ReviewPipeline as a baseline.

  --review    Generate a review for a single PDF and print it to console.
              Loads trained reviewers from flat_optimized/.

Extraction mode (add to any flag):
  --pageindex   Use pageindex-local for PDF section extraction (default for --review).
  --rlm         Use dspy.RLM markdown extraction instead.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import dspy
from dotenv import load_dotenv

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from collections import defaultdict

from experiments.dataset import build_flat_dataset
from reviewbuddy.pipeline import ReviewPipeline
from reviewbuddy.metrics import gepa_metric, compute_all_metrics, compute_aggregate_metrics, print_aggregate_metrics
from reviewbuddy.prompts import (
    AreaChairSignature, ICLR_RUBRIC,
    CAReviewer, MethodologyReviewer, ClarityReviewer,
)
from experiments.cost_tracker import get_lm_usage, print_cost_summary, register_lm, snapshot_per_model
import contextlib
from contextlib import nullcontext
import time

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

TRAIN_REVIEWS = str(BASE_DIR / "data" / "reviews" / "train")
TRAIN_PDFS    = str(BASE_DIR / "data" / "pdf" / "train")
VAL_REVIEWS   = str(BASE_DIR / "data" / "reviews" / "val")
VAL_PDFS      = str(BASE_DIR / "data" / "pdf" / "val")
TEST_REVIEWS  = str(BASE_DIR / "data" / "reviews" / "test")
TEST_PDFS     = str(BASE_DIR / "data" / "pdf" / "test")
PAGEINDEX_CACHE = str(BASE_DIR / "pageindex_cache")
RLM_TRAIN     = str(BASE_DIR / "RLM_train")
RLM_VAL       = str(BASE_DIR / "RLM_val")
RLM_TEST      = str(BASE_DIR / "RLM_test")
OUTPUT_DIR    = str(BASE_DIR / "flat_optimized")

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

VALID_ICLR_SCORES = {1, 3, 5, 6, 8, 10}


def _nearest_iclr_score(raw):
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return 1
    return min(VALID_ICLR_SCORES, key=lambda v: abs(v - raw))


def create_lms(api_key, api_base):
    """LMs for GEPA training and baseline: fast=gpt-4.1-nano, powerful/review=gpt-4.1-mini, reflection=gpt-5.4."""
    fast_lm = dspy.LM(
        model="openai/gpt-4.1-nano-fiit",
        api_key=api_key,
        api_base=api_base,
        temperature=0,
        max_tokens=2000,
    )
    powerful_lm = dspy.LM(
        model="openai/gpt-4.1-mini-fiit",
        api_key=api_key,
        api_base=api_base,
        temperature=0,
        max_tokens=4000,
    )
    review_lm = dspy.LM(
        model="openai/gpt-4.1-mini-fiit",
        api_key=api_key,
        api_base=api_base,
        temperature=0,
        max_tokens=4000,
    )
    reflection_lm = dspy.LM(
        model="openai/gpt-5.4-fiit",
        api_key=api_key,
        api_base=api_base,
        temperature=1.0,
        max_tokens=8000,
    )
    return fast_lm, powerful_lm, review_lm, reflection_lm


def create_eval_lms(api_key, api_base):
    """LMs for GEPA evaluation: all gpt-4.1-mini-fiit, temp=0."""
    fast_lm = dspy.LM(
        model="openai/gpt-4.1-nano-fiit",
        api_key=api_key,
        api_base=api_base,
        temperature=0,
        max_tokens=2000,
    )
    powerful_lm = dspy.LM(
        model="openai/gpt-4.1-mini-fiit",
        api_key=api_key,
        api_base=api_base,
        temperature=0,
        max_tokens=4000,
    )
    review_lm = dspy.LM(
        model="openai/gpt-4.1-mini-fiit",
        api_key=api_key,
        api_base=api_base,
        temperature=0,
        max_tokens=4000,
    )
    return fast_lm, powerful_lm, review_lm


def _save_cost_log(label: str, prompt_tokens: int, completion_tokens: int, cost: float, elapsed: float, n_examples: int = 0):
    import datetime
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "cost_log.json")
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "label": label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost, 4),
        "elapsed_sec": round(elapsed, 1),
        "n_examples": n_examples,
        "cost_per_example": round(cost / n_examples, 4) if n_examples else None,
    }
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    existing.append(entry)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Cost log saved to {log_path}")


# ---------------------------------------------------------------------------
# GEPA training
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fast_lm, powerful_lm, review_lm, reflection_lm = create_lms(API_KEY, API_BASE)
    dspy.configure(lm=powerful_lm, track_usage=True, num_threads=4)
    register_lm(fast_lm, "gpt-4.1-nano-fiit", "fast_lm")
    register_lm(powerful_lm, "gpt-4.1-mini-fiit", "powerful_lm")
    register_lm(review_lm, "gpt-4.1-mini-fiit", "review_lm")
    register_lm(reflection_lm, "gpt-5.4-fiit")
    t0 = time.time()

    use_rlm = args.use_rlm
    md_train = RLM_TRAIN if use_rlm else None
    md_val = RLM_VAL if use_rlm else None
    if use_rlm and not os.path.exists(RLM_VAL):
        raise FileNotFoundError(
            f"Validation markdown directory not found: {RLM_VAL}. "
            "Create a validation RLM split or run without --rlm."
        )

    print("Building flattened training dataset...")
    by_reviewer = build_flat_dataset(TRAIN_REVIEWS, TRAIN_PDFS, md_dir=md_train)
    n_reviewers = len(by_reviewer)
    print(f"Total reviewer slots: {n_reviewers}")

    print("Building validation dataset from held-out validation split...")
    by_reviewer_val = build_flat_dataset(VAL_REVIEWS, VAL_PDFS, md_dir=md_val)

    import random as _random
    from collections import defaultdict as _dd

    shared_extraction_cache = {}  # shared across all slots — each paper extracted only once

    for reviewer_idx, examples in sorted(by_reviewer.items()):
        if reviewer_idx > args.max_slot:
            print(f"\n[Reviewer {reviewer_idx}] Slot > --max-slot ({args.max_slot}), skipping.")
            continue
        if not examples:
            print(f"\n[Reviewer {reviewer_idx}] No training examples, skipping.")
            continue

        suffix = "_rlm" if use_rlm else ""
        save_path = os.path.join(OUTPUT_DIR, f"reviewer_{reviewer_idx}{suffix}.json")
        if os.path.exists(save_path) and not args.force:
            print(f"\n[Reviewer {reviewer_idx}] Already optimized ({save_path}), skipping. Use --force to retrain.")
            continue

        # Optionally limit trainset size for cost estimation
        if args.subset and args.subset < len(examples):
            _random.seed(42 + reviewer_idx)
            trainset = _random.sample(examples, args.subset)
        else:
            trainset = examples

        # Valset: stratified sample from validation slot, ~20% of train size
        val_examples = by_reviewer_val.get(reviewer_idx, [])
        if not val_examples:
            print(f"  [Reviewer {reviewer_idx}] No validation examples in held-out val split, skipping.")
            continue
        _random.seed(42 + reviewer_idx)
        by_score = _dd(list)
        for ex in val_examples:
            by_score[ex.human_score].append(ex)
        valset = []
        n_val_target = max(15, min(25, int(0.2 * len(trainset))))
        for score_val, group in sorted(by_score.items()):
            _random.shuffle(group)
            n_val = max(1, round(n_val_target * len(group) / len(val_examples)))
            valset.extend(group[:n_val])
        valset = valset[:n_val_target]

        print(f"\n{'='*60}")
        print(f"[Reviewer {reviewer_idx}] GEPA on {len(trainset)} train / {len(valset)} val examples")
        print(f"{'='*60}")

        pipeline = ReviewPipeline(
            fast_lm=fast_lm,
            powerful_lm=powerful_lm,
            review_lm=review_lm,
            use_pageindex=args.pageindex,
            use_rlm=use_rlm,
            pageindex_cache_dir=PAGEINDEX_CACHE if args.pageindex else None,
        )
        pipeline._intermediate_cache = shared_extraction_cache

        gepa_kwargs = {}
        if args.auto:
            gepa_kwargs["auto"] = args.auto
        else:
            gepa_kwargs["max_full_evals"] = args.max_full_evals

        optimizer = dspy.GEPA(
            metric=gepa_metric,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=args.reflection_minibatch_size,
            **gepa_kwargs,
        )

        optimized = optimizer.compile(
            pipeline,
            trainset=trainset,
            valset=valset,
        )

        optimized.save(save_path)
        print(f"  [Reviewer {reviewer_idx}] Saved to {save_path}")

    p, c, cost = get_lm_usage()
    print_cost_summary("GEPA TRAINING", p, c, cost, time.time() - t0)
    _save_cost_log("GEPA TRAINING", p, c, cost, time.time() - t0)


# ---------------------------------------------------------------------------
# Baseline (unoptimized single-reviewer)
# ---------------------------------------------------------------------------

def baseline(args):
    """Run a single unoptimized ReviewPipeline on test_reviews3 as baseline.
    Uses gpt-4.1-mini, temperature=0 (same model as GEPA training, no optimization).
    Metric: 1 generated review compared against each human review → averaged.
    """
    fast_lm, powerful_lm, review_lm, _ = create_lms(API_KEY, API_BASE)
    dspy.configure(lm=powerful_lm, track_usage=True)
    register_lm(fast_lm, "gpt-4.1-nano-fiit", "fast_lm")
    register_lm(powerful_lm, "gpt-4.1-mini-fiit", "powerful_lm")
    register_lm(review_lm, "gpt-4.1-mini-fiit", "review_lm")
    t0 = time.time()
    n_skipped = 0

    use_rlm = args.use_rlm
    pipeline = ReviewPipeline(
        fast_lm=fast_lm,
        powerful_lm=powerful_lm,
        review_lm=review_lm,
        use_pageindex=args.pageindex,
        use_rlm=use_rlm,
        pageindex_cache_dir=PAGEINDEX_CACHE if args.pageindex else None,
    )
    # One reviewer slot → treat the pipeline as reviewers[0]
    reviewers = [pipeline]

    # Area Chair setup (identical to evaluate())
    _ca_reviewer = dspy.Predict(CAReviewer)
    _methodology_reviewer = dspy.Predict(MethodologyReviewer)
    _clarity_reviewer = dspy.Predict(ClarityReviewer)
    _shared = {"sections": None, "article_text": None}

    def re_analyze_ca() -> str:
        """Re-analyze the paper's contribution and novelty independently."""
        if _shared["sections"] is None:
            return "Error: no sections available."
        with (dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()):
            s = _shared["sections"]
            result = _ca_reviewer(
                introduction_section=s.Introduction,
                related_work_section=s.Related_Work,
                conclusion_section=s.Conclusion,
            )
        return (f"CA Re-analysis\nType: {result.contribution_type}\n"
                f"Summary: {result.summary_of_contribution_assessment}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    def re_analyze_methodology() -> str:
        """Re-analyze the paper's methodology and experimental rigor independently."""
        if _shared["sections"] is None:
            return "Error: no sections available."
        with (dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()):
            s = _shared["sections"]
            result = _methodology_reviewer(
                methods_section=s.Methods,
                experiments_section=s.Experiments,
            )
        return (f"Methodology Re-analysis\nSummary: {result.summary_of_methods_and_experiments}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    def re_analyze_clarity() -> str:
        """Re-analyze the paper's clarity and presentation quality independently."""
        if _shared["article_text"] is None:
            return "Error: no article text available."
        with (dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()):
            result = _clarity_reviewer(article_text=_shared["article_text"])
        return (f"Clarity Re-analysis\nSummary: {result.clarity_summary}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    area_chair = dspy.ReAct(
        AreaChairSignature,
        tools=[re_analyze_ca, re_analyze_methodology, re_analyze_clarity],
        max_iters=3,
    )

    import random
    import re as _re

    print("\nBuilding test dataset...")
    by_reviewer = build_flat_dataset(TEST_REVIEWS, TEST_PDFS, md_dir=RLM_TEST if use_rlm else None)

    papers = defaultdict(list)
    for idx, examples in sorted(by_reviewer.items()):
        for ex in examples:
            papers[ex.pdf_path].append(ex)

    paper_list = list(papers.items())

    if args.test_whitelist:
        with open(args.test_whitelist) as _f:
            _wl = json.load(_f)
        whitelist_ids = set(str(x) for x in _wl.get("paper_ids", _wl if isinstance(_wl, list) else []))
        paper_list = [
            (pdf, exs) for pdf, exs in paper_list
            if (_m := _re.search(r"(\d+)\.pdf", pdf)) and _m.group(1) in whitelist_ids
        ]
        print(f"  Whitelist filter: {len(paper_list)} papers selected.")
        selected = paper_list[:args.n]
    else:
        # Stratified sample by human score of first reviewer
        by_score = defaultdict(list)
        for pdf, exs in paper_list:
            by_score[exs[0].human_score].append((pdf, exs))

        random.seed(42)
        selected = []
        for score_val, group in sorted(by_score.items()):
            k = max(1, round(args.n * len(group) / len(paper_list)))
            selected.extend(random.sample(group, min(k, len(group))))
        random.shuffle(selected)
        selected = selected[:args.n]

    all_metrics = []
    results = []

    for paper_pdf, paper_examples in selected:
        paper_examples.sort(key=lambda e: e.reviewer_idx)
        title = paper_examples[0].title if paper_examples else ""
        print(f"\nEvaluating (baseline): {title[:60]}")

        # Run the single pipeline once per paper
        try:
            first_example = paper_examples[0]
            if use_rlm:
                pred = pipeline(article_text=first_example.article_text)
            else:
                pred = pipeline(pdf_path=first_example.pdf_path)
            _shared["sections"] = getattr(pred, "sections", None)
            _shared["article_text"] = getattr(pred, "article_text", None)
        except Exception as e:
            print(f"  [Baseline] Error: {e}")
            n_skipped += 1
            continue

        # Compare 1 generated review against each human review → average
        per_human_metrics = [
            compute_all_metrics(ex, pred, include_llm_judge=args.llm_judge)
            for ex in paper_examples
        ]
        # average numeric fields, keep None if all None
        def _avg_field(key):
            vals = [x[key] for x in per_human_metrics if x.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        m = {k: _avg_field(k) for k in per_human_metrics[0]}

        review_text = (
            f"Comments:\n{pred.comments}\n\n"
            f"Strengths:\n{pred.strengths}\n\n"
            f"Weaknesses:\n{pred.weaknesses}"
        )
        score = pred.score

        judge_str = f"{m['llm_judge']:.2f}" if m['llm_judge'] is not None else "N/A"
        print(
            f"  pred={score} avg_human_score={m['human_score']:.1f} "
            f"sem_sim={m['semantic_similarity']:.2f} score_sim={m['score_similarity']:.2f} "
            f"judge={judge_str} rouge={m['rouge_l']:.2f} bert={m['bertscore_f1']:.2f}"
        )

        # Area Chair consolidation (single reviewer → trivial)
        final_score = _nearest_iclr_score(score)
        try:
            ac = area_chair(
                reviews=[review_text],
                scores=[score],
                rubric=ICLR_RUBRIC,
                final_score=float(score),
            )
            final_decision = ac.final_decision
            meta_review = ac.meta_review
        except Exception as e:
            print(f"  [AC] Error: {e}")
            final_decision = "Accept" if score >= 6 else "Reject"
            meta_review = ""

        human_decision = paper_examples[0].decision
        decision_correct = final_decision.lower() == human_decision.lower()

        print(
            f"  AC decision: {final_decision} (human: {human_decision}) "
            f"{'OK' if decision_correct else 'WRONG'} | gepa={m['gepa_score']:.3f}"
        )

        all_metrics.append({
            "gepa": m["gepa_score"], "sem": m["semantic_similarity"], "sim": m["score_similarity"],
            "judge": m["llm_judge"], "rouge": m["rouge_l"], "bert": m["bertscore_f1"],
            "decision_correct": decision_correct,
        })

        results.append({
            "title": title,
            "pdf_path": paper_pdf,
            "pred_score": float(score),
            "human_score": m["human_score"],
            "final_score": float(score),
            "final_decision": final_decision,
            "human_decision": human_decision,
            "decision_correct": decision_correct,
            "meta_review": meta_review,
            "gepa_score": m["gepa_score"],
            "llm_judge": m["llm_judge"],
            "score_similarity": m["score_similarity"],
            "semantic_similarity": m["semantic_similarity"],
            "rouge_1": m["rouge_1"],
            "rouge_2": m["rouge_2"],
            "rouge_l": m["rouge_l"],
            "bertscore_f1": m["bertscore_f1"],
            "length_diff": m["length_diff"],
        })

    if results:
        agg = compute_aggregate_metrics(results)
        print_aggregate_metrics(agg, label="BASELINE")

    print(f"\nPapers evaluated: {len(results)}  |  skipped: {n_skipped}")
    p, c, cost = get_lm_usage()
    print_cost_summary("BASELINE", p, c, cost, time.time() - t0, n_examples=len(results))
    _save_cost_log("BASELINE", p, c, cost, time.time() - t0, n_examples=len(results))

    out_path = os.path.join(OUTPUT_DIR, "baseline_results.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    fast_lm, powerful_lm, review_lm = create_eval_lms(API_KEY, API_BASE)
    dspy.configure(lm=powerful_lm, track_usage=True)
    register_lm(fast_lm, "gpt-4.1-nano-fiit", "fast_lm")
    register_lm(powerful_lm, "gpt-4.1-mini-fiit", "powerful_lm")
    register_lm(review_lm, "gpt-4.1-mini-fiit", "review_lm")
    t0 = time.time()
    n_skipped = 0

    use_rlm = args.use_rlm

    # Load optimized reviewers (rlm or pageindex depending on mode)
    suffix = "_rlm" if use_rlm else ""
    reviewer_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR)
         if f.startswith("reviewer_") and f.endswith(f"{suffix}.json")],
        key=lambda f: int(f.replace("reviewer_", "").replace(f"{suffix}.json", "")),
    )
    if not reviewer_files:
        print("No optimized reviewers found. Run with --gepa first.")
        return
    shared_intermediate_cache = {}
    reviewers = []
    for fname in reviewer_files:
        pipeline = ReviewPipeline(
            fast_lm=fast_lm,
            powerful_lm=powerful_lm,
            review_lm=review_lm,
            use_pageindex=args.pageindex,
            use_rlm=use_rlm,
            pageindex_cache_dir=PAGEINDEX_CACHE if args.pageindex else None,
        )
        pipeline.load(os.path.join(OUTPUT_DIR, fname))
        pipeline._intermediate_cache = shared_intermediate_cache
        reviewers.append(pipeline)
        print(f"  Loaded {fname}")


    # Agentic Area Chair with re-analysis tools (same as AgenticReviewPipeline)
    _ca_reviewer = dspy.Predict(CAReviewer)
    _methodology_reviewer = dspy.Predict(MethodologyReviewer)
    _clarity_reviewer = dspy.Predict(ClarityReviewer)

    # Shared state: sections and article_text set before each paper's AC call
    _shared = {"sections": None, "article_text": None}

    def re_analyze_ca() -> str:
        """Re-analyze the paper's contribution and novelty independently.
        Use this when reviewer opinions on novelty or significance diverge substantially."""
        if _shared["sections"] is None:
            return "Error: no sections available."
        ctx = dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()
        with ctx:
            s = _shared["sections"]
            result = _ca_reviewer(
                introduction_section=s.Introduction,
                related_work_section=s.Related_Work,
                conclusion_section=s.Conclusion,
            )
        return (f"CA Re-analysis\nType: {result.contribution_type}\n"
                f"Summary: {result.summary_of_contribution_assessment}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    def re_analyze_methodology() -> str:
        """Re-analyze the paper's methodology and experimental rigor independently.
        Use this when reviewer opinions on technical soundness diverge substantially."""
        if _shared["sections"] is None:
            return "Error: no sections available."
        ctx = dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()
        with ctx:
            s = _shared["sections"]
            result = _methodology_reviewer(
                methods_section=s.Methods,
                experiments_section=s.Experiments,
            )
        return (f"Methodology Re-analysis\nSummary: {result.summary_of_methods_and_experiments}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    def re_analyze_clarity() -> str:
        """Re-analyze the paper's clarity and presentation quality independently.
        Use this when reviewer opinions on writing quality diverge substantially."""
        if _shared["article_text"] is None:
            return "Error: no article text available."
        ctx = dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()
        with ctx:
            result = _clarity_reviewer(article_text=_shared["article_text"])
        return (f"Clarity Re-analysis\nSummary: {result.clarity_summary}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    area_chair = dspy.ReAct(
        AreaChairSignature,
        tools=[re_analyze_ca, re_analyze_methodology, re_analyze_clarity],
        max_iters=3,
    )

    import random
    import re as _re

    # Build test dataset (flat, all reviewers merged for per-paper evaluation)
    print("\nBuilding test dataset...")
    by_reviewer = build_flat_dataset(TEST_REVIEWS, TEST_PDFS, md_dir=RLM_TEST if use_rlm else None)

    # Group by paper (pdf_path) for full-paper evaluation
    papers = defaultdict(list)
    for idx, examples in sorted(by_reviewer.items()):
        for ex in examples:
            papers[ex.pdf_path].append(ex)

    paper_list = list(papers.items())

    # Filter to whitelist if provided
    if args.test_whitelist:
        with open(args.test_whitelist) as _f:
            _wl = json.load(_f)
        whitelist_ids = set(str(x) for x in _wl.get("paper_ids", _wl if isinstance(_wl, list) else []))
        paper_list = [
            (pdf, exs) for pdf, exs in paper_list
            if (_m := _re.search(r"(\d+)\.pdf", pdf)) and _m.group(1) in whitelist_ids
        ]
        print(f"  Whitelist filter: {len(paper_list)} papers selected.")
        selected = paper_list[:args.n]
    else:
        # Stratified sample by human score of first reviewer
        by_score = defaultdict(list)
        for pdf, exs in paper_list:
            by_score[exs[0].human_score].append((pdf, exs))

        random.seed(42)
        selected = []
        for score_val, group in sorted(by_score.items()):
            k = max(1, round(args.n * len(group) / len(paper_list)))
            selected.extend(random.sample(group, min(k, len(group))))
        random.shuffle(selected)
        selected = selected[:args.n]

    results = []

    for paper_pdf, paper_examples in selected:
        paper_examples.sort(key=lambda e: e.reviewer_idx)
        title = paper_examples[0].title if paper_examples else ""
        print(f"\nEvaluating: {title[:60]}")

        reviews_text = []
        scores = []
        per_reviewer_metrics = []

        for idx, reviewer in enumerate(reviewers):
            if idx >= len(paper_examples):
                continue  # paper has fewer reviews than reviewer slots
            ex = paper_examples[idx]

            try:
                if use_rlm:
                    pred = reviewer(article_text=ex.article_text)
                else:
                    pred = reviewer(pdf_path=ex.pdf_path)
                # Share sections from first reviewer for Area Chair tools
                if idx == 0:
                    _shared["sections"] = getattr(pred, "sections", None)
                    _shared["article_text"] = getattr(pred, "article_text", None)
            except Exception as e:
                print(f"  [Reviewer {idx}] Error: {e}")
                continue

            m = compute_all_metrics(ex, pred, include_llm_judge=(args.llm_judge and idx == 0))
            per_reviewer_metrics.append(m)

            review_text = (
                f"Comments:\n{pred.comments}\n\n"
                f"Strengths:\n{pred.strengths}\n\n"
                f"Weaknesses:\n{pred.weaknesses}"
            )
            reviews_text.append(review_text)
            scores.append(pred.score)

            judge_str = f"{m['llm_judge']:.2f}" if m['llm_judge'] is not None else "N/A"
            print(
                f"  Reviewer {idx}: pred={pred.score} human={ex.human_score} "
                f"sem_sim={m['semantic_similarity']:.2f} score_sim={m['score_similarity']:.2f} "
                f"judge={judge_str} rouge={m['rouge_l']:.2f} bert={m['bertscore_f1']:.2f}"
            )

        if not reviews_text:
            n_skipped += 1
            continue

        # Area Chair consolidation
        final_score = sum(scores) / len(scores)
        try:
            ac = area_chair(
                reviews=reviews_text,
                scores=scores,
                rubric=ICLR_RUBRIC,
                final_score=final_score,
            )
            final_decision = ac.final_decision
            meta_review = ac.meta_review
        except Exception as e:
            print(f"  [AC] Error: {e}")
            final_decision = "Accept" if final_score >= 6 else "Reject"
            meta_review = ""

        human_decision = paper_examples[0].decision
        decision_correct = final_decision.lower() == human_decision.lower()

        def _avg_m(key):
            vals = [m[key] for m in per_reviewer_metrics if m.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        avg_pred_score = sum(scores) / len(scores)
        avg_human_score = _avg_m("human_score")

        print(
            f"  AC decision: {final_decision} (human: {human_decision}) "
            f"{'OK' if decision_correct else 'WRONG'} | "
            f"gepa={_avg_m('gepa_score'):.3f}"
        )

        results.append({
            "title": title,
            "pdf_path": paper_pdf,
            "pred_score": avg_pred_score,
            "human_score": avg_human_score,
            "final_score": final_score,
            "final_decision": final_decision,
            "human_decision": human_decision,
            "decision_correct": decision_correct,
            "meta_review": meta_review,
            # per-pair data for aggregate stats
            "per_pair_pred_scores": scores,
            "per_pair_human_scores": [m["human_score"] for m in per_reviewer_metrics],
            # averages
            "gepa_score": _avg_m("gepa_score"),
            "llm_judge": _avg_m("llm_judge"),
            "semantic_similarity": _avg_m("semantic_similarity"),
            "score_similarity": _avg_m("score_similarity"),
            "rouge_1": _avg_m("rouge_1"),
            "rouge_2": _avg_m("rouge_2"),
            "rouge_l": _avg_m("rouge_l"),
            "bertscore_f1": _avg_m("bertscore_f1"),
            "length_diff": _avg_m("length_diff"),
        })

    if results:
        agg = compute_aggregate_metrics(results)
        print_aggregate_metrics(agg, label="GEPA EVAL")

    print(f"\nPapers evaluated: {len(results)}  |  skipped: {n_skipped}")
    p, c, cost = get_lm_usage()
    print_cost_summary("GEPA EVAL", p, c, cost, time.time() - t0, n_examples=len(results))
    _save_cost_log("GEPA EVAL", p, c, cost, time.time() - t0, n_examples=len(results))

    out_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Single paper review
# ---------------------------------------------------------------------------

def review_single(args):
    """Load optimized reviewers and generate a review for a single PDF."""
    fast_lm, powerful_lm, _ = create_eval_lms(API_KEY, API_BASE)
    review_lm = dspy.LM(
        model="openai/gpt-4.1-mini-fiit",
        api_key=API_KEY,
        api_base=API_BASE,
        temperature=0.3,
        max_tokens=4000,
    )
    dspy.configure(lm=powerful_lm)

    use_rlm = args.use_rlm
    suffix = "_rlm" if use_rlm else ""
    reviewer_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR)
         if f.startswith("reviewer_") and f.endswith(f"{suffix}.json")
         and (suffix != "" or not f.endswith("_rlm.json"))],
        key=lambda f: int(f.replace("reviewer_", "").replace(f"{suffix}.json", "")),
    )
    if not reviewer_files:
        print("No optimized reviewers found in flat_optimized/. Run --gepa first.")
        return

    reviewers = []
    for fname in reviewer_files:
        pipeline = ReviewPipeline(
            fast_lm=fast_lm,
            powerful_lm=powerful_lm,
            review_lm=review_lm,
            use_pageindex=args.pageindex,
            use_rlm=use_rlm,
            pageindex_cache_dir=PAGEINDEX_CACHE if args.pageindex else None,
        )
        pipeline.load(os.path.join(OUTPUT_DIR, fname))
        reviewers.append(pipeline)
        print(f"  Loaded {fname}")

    shared_cache = {}
    for r in reviewers:
        r._intermediate_cache = shared_cache

    _ca_reviewer = dspy.Predict(CAReviewer)
    _methodology_reviewer = dspy.Predict(MethodologyReviewer)
    _clarity_reviewer = dspy.Predict(ClarityReviewer)
    _shared = {"sections": None, "article_text": None}

    def re_analyze_ca() -> str:
        """Re-analyze the paper's contribution and novelty independently."""
        if _shared["sections"] is None:
            return "Error: no sections available."
        with (dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()):
            s = _shared["sections"]
            result = _ca_reviewer(
                introduction_section=s.Introduction,
                related_work_section=s.Related_Work,
                conclusion_section=s.Conclusion,
            )
        return (f"CA Re-analysis\nType: {result.contribution_type}\n"
                f"Summary: {result.summary_of_contribution_assessment}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    def re_analyze_methodology() -> str:
        """Re-analyze the paper's methodology and experimental rigor independently."""
        if _shared["sections"] is None:
            return "Error: no sections available."
        with (dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()):
            s = _shared["sections"]
            result = _methodology_reviewer(
                methods_section=s.Methods,
                experiments_section=s.Experiments,
            )
        return (f"Methodology Re-analysis\nSummary: {result.summary_of_methods_and_experiments}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    def re_analyze_clarity() -> str:
        """Re-analyze the paper's clarity and presentation quality independently."""
        if _shared["article_text"] is None:
            return "Error: no article text available."
        with (dspy.context(lm=powerful_lm) if powerful_lm else nullcontext()):
            result = _clarity_reviewer(article_text=_shared["article_text"])
        return (f"Clarity Re-analysis\nSummary: {result.clarity_summary}\n"
                f"Strengths: {result.strengths}\nWeaknesses: {result.weaknesses}")

    area_chair = dspy.ReAct(
        AreaChairSignature,
        tools=[re_analyze_ca, re_analyze_methodology, re_analyze_clarity],
        max_iters=3,
    )

    pdf_path = args.review
    print(f"\nReviewing: {pdf_path}\n{'='*60}")

    reviews_text = []
    scores = []

    for idx, reviewer in enumerate(reviewers):
        print(f"  Generating review {idx + 1}/{len(reviewers)}...", end="", flush=True)
        try:
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                pred = reviewer(pdf_path=pdf_path)
            if idx == 0:
                _shared["sections"] = getattr(pred, "sections", None)
                _shared["article_text"] = getattr(pred, "article_text", None)
            print(" done")
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        review_text = (
            f"Comments:\n{pred.comments}\n\n"
            f"Strengths:\n{pred.strengths}\n\n"
            f"Weaknesses:\n{pred.weaknesses}\n\n"
            f"Questions for authors:\n{pred.questions}"
        )
        reviews_text.append(review_text)
        scores.append(pred.score)

        print(f"\n{'─'*60}")
        print(f"REVIEWER {idx + 1}  (score: {pred.score})")
        print(f"{'─'*60}")
        print(review_text)

    if not reviews_text:
        print("No reviews generated.")
        return

    final_score = sum(scores) / len(scores)
    try:
        ac = area_chair(
            reviews=reviews_text,
            scores=scores,
            rubric=ICLR_RUBRIC,
            final_score=final_score,
        )
        final_decision = ac.final_decision
        meta_review = ac.meta_review
    except Exception as e:
        print(f"  [AC] Error: {e}")
        final_decision = "Accept" if final_score >= 6 else "Reject"
        meta_review = ""

    print(f"\n{'='*60}")
    print(f"AREA CHAIR META-REVIEW")
    print(f"{'='*60}")
    print(meta_review)
    print(f"\nFINAL DECISION: {final_decision}")
    print(f"FINAL SCORE:    {_nearest_iclr_score(final_score)}  (avg: {final_score:.1f})")
    print(f"{'='*60}")

    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
    reviews_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "reviews")
    os.makedirs(reviews_dir, exist_ok=True)
    out_path = os.path.join(reviews_dir, f"review_{paper_name}.json")

    generated_reviews = []
    for i, reviewer_pred in enumerate(zip(reviews_text, scores)):
        text, score = reviewer_pred
        parts = {}
        for field in ["Comments", "Strengths", "Weaknesses", "Questions for authors"]:
            marker = f"{field}:\n"
            if marker in text:
                start = text.index(marker) + len(marker)
                next_markers = [f"{f}:\n" for f in ["Comments", "Strengths", "Weaknesses", "Questions for authors"] if f != field and f"{f}:\n" in text[start:]]
                end = text.index(next_markers[0], start) if next_markers else len(text)
                parts[field] = text[start:end].strip()
            else:
                parts[field] = ""
        generated_reviews.append({
            "review_id": i + 1,
            "RECOMMENDATION": score,
            "comments": parts.get("Comments", ""),
            "STRENGTHS": parts.get("Strengths", ""),
            "WEAKNESSES": parts.get("Weaknesses", ""),
            "QUESTIONS": parts.get("Questions for authors", ""),
        })

    output = {
        "pdf_path": pdf_path,
        "decision": final_decision,
        "final_score": _nearest_iclr_score(final_score),
        "average_score": round(final_score, 2),
        "meta_review": meta_review,
        "reviews": generated_reviews,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flat reviewer-level GEPA training")
    parser.add_argument("--gepa", action="store_true", help="Run GEPA optimization")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on test set")
    parser.add_argument("--baseline", action="store_true", help="Run unoptimized baseline (1 review, no GEPA)")
    parser.add_argument("--review", type=str, default=None, metavar="PDF", help="Review a single paper (path to PDF)")
    parser.add_argument("--pageindex", action="store_true", help="Use PageIndex PDF extraction")
    parser.add_argument("--rlm", action="store_true", help="Use DspyRLM markdown extraction (default when --pageindex not set)")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM-as-judge metric during eval/baseline (expensive, off by default)")
    parser.add_argument("--force", action="store_true", help="Retrain even if already optimized")
    parser.add_argument("--auto", type=str, default=None, choices=["light", "medium", "heavy"], help="GEPA auto budget (light=6, medium=12, heavy=18 candidates)")
    parser.add_argument("--max-full-evals", type=int, default=3, help="GEPA max full evaluations (alternative to --auto)")
    parser.add_argument("--reflection-minibatch-size", type=int, default=10, help="GEPA reflection minibatch size")
    parser.add_argument("--n", type=int, default=130, help="Number of test papers to evaluate")
    parser.add_argument("--max-slot", type=int, default=2, help="Max reviewer slot index to optimize (0-indexed)")
    parser.add_argument("--subset", type=int, default=80, help="Limit trainset to N examples per slot")
    parser.add_argument("--test-whitelist", type=str, default=None, help="JSON file with paper_ids list to restrict test/eval set")
    args = parser.parse_args()
    args.use_rlm = not args.pageindex

    if not args.gepa and not args.eval and not args.baseline and not args.review:
        parser.print_help()
        return

    if args.gepa:
        train(args)
    if args.eval:
        evaluate(args)
    if args.baseline:
        baseline(args)
    if args.review:
        review_single(args)


if __name__ == "__main__":
    main()
