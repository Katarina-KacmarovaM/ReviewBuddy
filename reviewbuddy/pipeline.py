"""
ReviewPipeline — generates one structured review for an academic paper.

Pipeline stages:
  Stage 0 — Section extraction: PDF -> Abstract, Introduction, Related Work,
             Methods, Experiments, Conclusion. Supports pageindex-local (default),
             dspy.RLM, or plain dspy.Predict.
  Stage 1 — Validation: checks section quality and detects prompt injection.
  Stage 2a — Specialist reviewers: CA (contribution), Methodology, Clarity.
             Run once per paper and cached — shared across all GEPA candidates.
  Stage 2b — Summary + Score: ChainOfThought modules optimized by GEPA.

Area Chair (meta-review + final decision) is NOT part of this pipeline —
it runs separately at inference time in run_flat.py.
"""

import dspy
from contextlib import nullcontext
from typing import Optional
from pathlib import Path

from reviewbuddy.prompts import (
    ICLR_RUBRIC,
    SectionExtractor,
    PromptInjectionDetector,
    CAReviewer,
    MethodologyReviewer,
    ClarityReviewer,
    SummaryReview,
    ScoreForReview,
)
from reviewbuddy.extraction import LocalPDFSectionExtractor

VALID_ICLR_SCORES = {1, 3, 5, 6, 8, 10}

# Keywords that must appear in a suspicious excerpt for it to count as a real injection.
# This prevents false positives from math formulas, citations, and normal academic text.
_INJECTION_KEYWORDS = frozenset([
    "ignore previous", "ignore all previous", "ignore the previous",
    "disregard previous", "disregard the previous",
    "forget previous", "forget your instructions", "forget all",
    "you must give this paper", "give this paper a score",
    "assign a score of", "rate this paper as",
    "new instructions", "your new task", "new task:",
    "system prompt", "system message",
    "jailbreak",
    "override instructions", "override your",
])


def _nearest_iclr_score(raw: float) -> int:
    return min(VALID_ICLR_SCORES, key=lambda v: abs(v - raw))


def _is_real_injection(excerpt: str) -> bool:
    """Return True only when excerpt contains explicit injection phrasing."""
    lower = excerpt.lower()
    return any(kw in lower for kw in _INJECTION_KEYWORDS)


class ReviewPipeline(dspy.Module):
    """
    Generates one structured review for a paper.
    Optimized by GEPA — Area Chair is excluded.

    forward() returns dspy.Prediction with:
        comments, strengths, weaknesses, questions, score
    """

    def __init__(
        self,
        fast_lm=None,
        powerful_lm=None,
        review_lm=None,
        use_pageindex: bool = False,
        use_rlm: bool = False,
        pageindex_cache_dir: Optional[str] = None,
        pdf_parser: str = "pageindex",
    ):
        super().__init__()
        self.fast_lm = fast_lm
        self.powerful_lm = powerful_lm
        self.review_lm = review_lm
        self.use_pageindex = use_pageindex
        self.use_rlm = use_rlm
        self.rubric = ICLR_RUBRIC

        # Stage 0: extractor
        if use_pageindex:
            ref_lm = fast_lm or powerful_lm
            lm_model = getattr(ref_lm, "model", "gpt-5.4-nano") if ref_lm else "gpt-5.4-nano"
            if "/" in lm_model:
                lm_model = lm_model.split("/", 1)[1]
            lm_kwargs = getattr(ref_lm, "kwargs", {}) if ref_lm else {}
            self.extractor = LocalPDFSectionExtractor(
                model=lm_model,
                api_key=lm_kwargs.get("api_key"),
                api_base=lm_kwargs.get("api_base"),
                fast_lm=fast_lm,
                cache_dir=pageindex_cache_dir,
                pdf_parser=pdf_parser,
            )
        elif use_rlm:
            from experiments.dspy_rlm_extraction import DspyRLMSectionExtractor
            self.extractor = DspyRLMSectionExtractor()
        else:
            self.extractor = dspy.Predict(SectionExtractor)

        # Stage 1: validation
        self.injection_detector = dspy.Predict(PromptInjectionDetector)

        # Stage 2: specialist agents
        self.ca_reviewer = dspy.Predict(CAReviewer)
        self.methodology_reviewer = dspy.Predict(MethodologyReviewer)
        self.clarity_reviewer = dspy.Predict(ClarityReviewer)
        self.summary_reviewer = dspy.ChainOfThought(SummaryReview)
        self.score_predictor = dspy.ChainOfThought(ScoreForReview)

        # Cache for intermediate results (extraction, validation, CA/methodology/clarity)
        # Keyed by pdf_path so each article is processed only once across GEPA candidates
        self._intermediate_cache: dict = {}

    def named_predictors(self):
        # Only optimize SummaryReview and ScoreForReview — others are frozen
        # Must return inner Predict objects (not ChainOfThought wrappers)
        return [
            ("summary_reviewer.predict", self.summary_reviewer.predict),
            ("score_predictor.predict", self.score_predictor.predict),
        ]

    def forward(
        self,
        pdf_path: Optional[str] = None,
        article_text: Optional[str] = None,
        **kwargs,  # human_* fields passed by GEPA for reflection LM visibility, not used in generation
    ) -> dspy.Prediction:
        if pdf_path is None and article_text is None:
            raise ValueError("Provide either pdf_path or article_text.")

        cache_key = pdf_path or article_text[:200]

        if cache_key not in self._intermediate_cache:
            # Stage 0: extract
            sections, article_text = self._extract(article_text, pdf_path)
            if sections is None:
                return self._error_result("Section extraction failed.")

            # Stage 1: validate (injection detection runs once per article)
            validation_issues = self._validate(sections)
            quality_failures = sum(1 for v in validation_issues.values() if v.startswith("QUALITY:"))
            if quality_failures >= 4:
                return self._error_result("Section extraction insufficient — skipping.")
            if validation_issues:
                print(f"  [Pipeline] Validation issues: {validation_issues}")

            # Stage 2a: specialist agents run once per article
            lm = self.review_lm or self.powerful_lm
            ctx = dspy.context(lm=lm) if lm else nullcontext()
            with ctx:
                ca = self.ca_reviewer(
                    introduction_section=sections.Introduction,
                    related_work_section=sections.Related_Work,
                    conclusion_section=sections.Conclusion,
                )
                methodology = self.methodology_reviewer(
                    methods_section=sections.Methods,
                    experiments_section=sections.Experiments,
                )
                clarity = self.clarity_reviewer(article_text=article_text)

            ca_text = (
                f"Contribution type: {ca.contribution_type}\n"
                f"{ca.summary_of_contribution_assessment}\n"
                f"Strengths: {ca.strengths}\n"
                f"Weaknesses: {ca.weaknesses}\n"
                f"Rejection reasons: {ca.rejection_reasons}"
            )
            methodology_text = (
                f"{methodology.summary_of_methods_and_experiments}\n"
                f"Strengths: {methodology.strengths}\n"
                f"Weaknesses: {methodology.weaknesses}\n"
                f"Rejection reasons: {methodology.rejection_reasons}"
            )
            clarity_text = (
                f"{clarity.clarity_summary}\n"
                f"Strengths: {clarity.strengths}\n"
                f"Weaknesses: {clarity.weaknesses}\n"
                f"Rejection reasons: {clarity.rejection_reasons}"
            )
            rejection_reasons = (
                f"CA: {ca.rejection_reasons}\n"
                f"Methodology: {methodology.rejection_reasons}\n"
                f"Clarity: {clarity.rejection_reasons}"
            )

            self._intermediate_cache[cache_key] = {
                "sections": sections,
                "article_text": article_text,
                "validation_issues": validation_issues,
                "ca_text": ca_text,
                "methodology_text": methodology_text,
                "clarity_text": clarity_text,
                "rejection_reasons": rejection_reasons,
                "soundness": methodology.soundness_score,
                "presentation": clarity.presentation_score,
                "contribution": ca.contribution_score,
            }
        else:
            cached = self._intermediate_cache[cache_key]
            sections = cached["sections"]
            article_text = cached["article_text"]
            validation_issues = cached["validation_issues"]
            ca_text = cached["ca_text"]
            methodology_text = cached["methodology_text"]
            clarity_text = cached["clarity_text"]
            rejection_reasons = cached["rejection_reasons"]

        cached = self._intermediate_cache[cache_key]

        # Stage 2b: summary + score — optimized, run per GEPA candidate
        lm = self.review_lm or self.powerful_lm
        ctx = dspy.context(lm=lm) if lm else nullcontext()
        with ctx:
            summary = self.summary_reviewer(
                ca_review=ca_text,
                methodology_review=methodology_text,
                clarity_review=clarity_text,
            )

            full_review = (
                f"Comments:\n{summary.comments}\n\n"
                f"Strengths:\n{summary.strengths}\n\n"
                f"Weaknesses:\n{summary.weaknesses}\n\n"
                f"Questions for authors:\n{summary.questions}"
            )
            score_pred = self.score_predictor(
                review_comments=full_review,
                rubric=self.rubric,
                rejection_reasons=rejection_reasons,
            )

        raw_score = getattr(score_pred, "score", 1)
        try:
            raw_score = float(raw_score)
            if not (1 <= raw_score <= 10):
                raw_score = 1
        except (TypeError, ValueError):
            raw_score = 1
        score = _nearest_iclr_score(raw_score)

        return dspy.Prediction(
            comments=summary.comments,
            strengths=summary.strengths,
            weaknesses=summary.weaknesses,
            questions=summary.questions,
            score=score,
            score_justification=getattr(score_pred, "score_justification", ""),
            soundness=cached["soundness"],
            presentation=cached["presentation"],
            contribution=cached["contribution"],
            validation_issues=validation_issues,
            sections=sections,
            article_text=article_text,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _extract(self, article_text, pdf_path, force_pymupdf: bool = False):
        try:
            if self.use_pageindex:
                ctx = dspy.context(lm=self.fast_lm) if self.fast_lm else nullcontext()
                with ctx:
                    sections = self.extractor(pdf_path=pdf_path, text=article_text, force_pymupdf=force_pymupdf)
                if not article_text:
                    article_text = "\n\n".join(filter(None, [
                        sections.Abstract, sections.Introduction,
                        sections.Related_Work, sections.Methods,
                        sections.Experiments, sections.Conclusion,
                    ]))
            elif self.use_rlm:
                # RLM needs a capable model to write correct Python extraction code
                rlm_lm = self.powerful_lm or self.fast_lm
                ctx = dspy.context(lm=rlm_lm) if rlm_lm else nullcontext()
                with ctx:
                    sections = self.extractor(full_text=article_text)
            else:
                ctx = dspy.context(lm=self.fast_lm) if self.fast_lm else nullcontext()
                with ctx:
                    sections = self.extractor(article_text=article_text)
            return sections, article_text
        except Exception as e:
            print(f"  [Pipeline] Extraction error: {e}")
            return None, None

    def _validate(self, sections, check_injection: bool = True) -> dict:
        issues = {}
        section_names = ["Abstract", "Introduction", "Related_Work", "Methods", "Experiments", "Conclusion"]
        ctx = dspy.context(lm=self.fast_lm) if self.fast_lm else nullcontext()
        with ctx:
            for name in section_names:
                text = getattr(sections, name, "")
                if not text or not text.strip() or len(text.split()) < 80:
                    issues[name] = f"QUALITY: too short or empty"
                    continue
                if check_injection:
                    inj = self.injection_detector(text=text)
                    excerpt = (inj.suspicious_excerpt or "").strip()
                    excerpt_lower = excerpt.lower()
                    if (
                        not inj.is_safe
                        and excerpt_lower not in ("", "none", "n/a", "no injection detected")
                        and _is_real_injection(excerpt)
                    ):
                        issues[name] = f"INJECTION: {excerpt}"
        return issues

    @staticmethod
    def _error_result(msg: str) -> dspy.Prediction:
        return dspy.Prediction(
            comments=msg, strengths="", weaknesses="", questions="",
            score=1, score_justification="", soundness=1, presentation=1,
            contribution=1, validation_issues={},
        )
