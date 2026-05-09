"""Early agentic multi-agent workflow with a coordinator — precursor to the main ReviewBuddy pipeline."""

#TODO DAJ si pozor na metriky.. porovnavaj spravne !!!
#TODO Checkni este raz clanky a a zvaz pocet
# AGENTIC MULTI-AGENT WORKFLOW WITH COORDINATOR

from unittest import result
import dspy
import os
from dotenv import load_dotenv
import litellm
from bert_score import score
from evaluate import load
import json
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import csv
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# GEPA
from dspy.teleprompt import MIPROv2
from dspy.evaluate import evaluate
from dspy.teleprompt import GEPA
import sys


load_dotenv()
TF_ENABLE_ONEDNN_OPTS=0
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

model = "openai/gpt-4.1-nano-fiit"

lmm = dspy.LM(model=model, api_key=API_KEY, api_base=API_BASE, temperature=0.3, cache=False)
dspy.configure(lm=lmm)

# =============================================================================
# DATA STRUCTURES FOR MULTI-AGENT COMMUNICATION
# =============================================================================

@dataclass
class AgentReview:
    """Output from a specialist agent"""
    agent_name: str
    review_text: str
    score: float  # 1-5 scale
    confidence: float  # 0.0 - 1.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

@dataclass
class ReviewState:
    """Shared state for the collaborative review process"""
    article_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    agent_reviews: Dict[str, AgentReview] = field(default_factory=dict)
    coordinator_notes: List[str] = field(default_factory=list)
    final_summary: Optional[str] = None
    final_decision: Optional[str] = None
    final_score: Optional[float] = None


# =============================================================================
# SIGNATURES (Low-level LLM calls)
# =============================================================================

class SectionExtractor(dspy.Signature):
    """
    Extract specific sections from a scientific article provided in markdown.
    
    RULES:
    1. DO NOT summarize. Extract the text verbatim including all subheadings (e.g., ## 2.1, ###).
    2. INCLUDE all image descriptions found within the tags [ POPIS OBRÁZKA] that belong to the section.
    3. IGNORE page markers like [--- START PAGE X ---], but extract all text between them.
    4. PRESERVE markdown formatting (bold, tables, LaTeX).
    5. ADAPT to the paper structure: Titles vary. Look for the CONTENT type described below.
    
    """
    article_text: str = dspy.InputField(description="Full text of the article in markdown format.")
    
    Abstract: str = dspy.OutputField(description="Content under the Abstract heading.")
    
    Introduction: str = dspy.OutputField(description="Content from the Introduction heading up to the start of the technical/background sections.")
    
    Methods: str = dspy.OutputField(description="The core technical contribution. This typically includes sections named 'Methodology', 'Proposed Method', 'Framework', 'Preliminaries', or specific technical titles (e.g., 'Stabilizing...', 'Simplifying...'). It contains mathematical definitions, algorithms, and architectural details. It usually sits between Introduction and Experiments.")
    
    Experiments: str = dspy.OutputField(description="The empirical evaluation. Look for sections named 'Experiments', 'Results', 'Evaluation', 'Benchmarks', 'Ablation Study' or 'Scaling Up'. Includes tables, metrics (FID, accuracy), and comparisons.")
    
    Conclusion: str = dspy.OutputField(description="Content under 'Conclusion', 'Discussion', or 'Limitations'.")

class CAReview(dspy.Signature):
    """
    Review the contribution and novelty of the article.
    
    CRITICAL CHECKLIST (The "Impact" Test):
    1. MOTIVATION: Does the paper address a REAL and SIGNIFICANT bottleneck?
       - If it solves a "strawman" (fake) problem or lacks justification ("Why do we need this?"), mark it as WEAK MOTIVATION.
    
    2. NOVELTY TYPE:
       - POSITIVE: "Root Cause Analysis" (explains WHY something fails), "Unification" (connects disjoint theories), or "Simpson's Paradox" (reveals a counter-intuitive truth).
       - NEGATIVE: "Incremental" (just another loss term), "Application" (applying existing method to new data without insight), or "Complexity without Gain".
       
    3. COMPLEXITY CHECK: 
       - Is the method complex just to appear rigorous ("Math Salad")? 
       - If the math is dense but the gain over simple baselines is marginal -> MAJOR WEAKNESS.
    
    
    """
    introduction_section: str = dspy.InputField(description="Introduction text: Look for the 'Why' and the 'What'.")
    conclusion_section: str = dspy.InputField(description="Conclusion text: Look for the claims of impact.")
    contribution_type: str = dspy.OutputField(description="Classify as: 'Incremental' (Score < 3.8), 'Significant' (Score > 3.9), or 'Breakthrough'.")
    review: str = dspy.OutputField(description="Critique focusing on novelty, significance, and motivation.")


class MethodologyReview(dspy.Signature):
    """
    Review methods and experiments.
    
    CRITICAL CHECKLIST (The "SOTA" Test):
    1. DATASETS: Does the paper use large-scale standard benchmarks (e.g., ImageNet, extensive NLP suites) or only toy datasets (CIFAR, MNIST)? 
       - Toy datasets only = Negative signal.
    2. BASELINES: Do they compare against state-of-the-art (SOTA) methods from the last 1-2 years? 
       - Missing recent baselines = Fatal flaw.
    3. ABLATION: Did they prove *which* part of their method causes the improvement?

    """
    methods_section: str = dspy.InputField(description="Text of the article methods section to be reviewed.")
    experiments_section: str = dspy.InputField(description="Text of the article experiments section to be reviewed.")
    review: str = dspy.OutputField(description="Critique focusing on experimental rigor, scale, and baselines.")


class ClarityReview(dspy.Signature):
    """
    Review the clarity of the article. 
    
    CRITICAL INSTRUCTION: 
    You must act as a "Syntax Strict Reviewer". If the paper uses mathematical symbols (e.g., H, U, beta) without explicitly defining them in the text, you MUST flag this as a major weakness.
    
    Checklist:
    1. Are all variables defined upon first use? (If NO -> Score < 3)
    2. Is the high-level intuition provided before the complex math?
    3. Are figures self-explanatory?
    """
    article_text: str = dspy.InputField(description="Full text of the article in markdown format to be reviewed for clarity.")
    review: str = dspy.OutputField(description="Critique focusing on notation definitions and readability.")


class SummaryGenerator(dspy.Signature):
    """Generate a specific summary of the article based on the reviews provided. Be objective and strict. Mention specific strong and weak points of the article. Give suggestions for improvement."""
    ca_review: str = dspy.InputField(description="Contribution assessment review text.")
    methodology_review: str = dspy.InputField(description="Methodology review text.")
    clarity_review: str = dspy.InputField(description="Clarity review text.")
    summary: str = dspy.OutputField(description="Generated specific summary.")


class DecisionPredictor(dspy.Signature):
    """
    You are an Area Chair at ICLR 2025. Make a final acceptance decision.
    
    CALIBRATION EXAMPLES (Derived from actual ICLR data):
    
    [SCENARIO: REJECT] (Score ~2.5)
    - Signs: "Notation unclear", "Undefined matrices", "Weak motivation", "Tested only on small datasets".
    - Action: REJECT. Even if the math looks fancy, if it's not defined, it's worthless.
    
    [SCENARIO: WEAK ACCEPT] (Score ~3.5 )  # 3.9
    - Signs: "Good performance", "Clear method", but "Incremental novelty" or "Small experiments".
    - Action: ACCEPT (Borderline).
    
    [SCENARIO: STRONG ACCEPT] (Score > 4.5)
    - Signs: "Unifies existing theories", "Scales to 1.5B+ parameters", "Identifies root cause of a problem", "Simple and effective".
    - Action: ACCEPT (Oral/Spotlight).

    If score is under 3.9, then REJECT.
    
    Evaluate the input reviews based on these scenarios. Be harsh on clarity and experimental scale.


    """
    #summary: str = dspy.InputField(description="Summary of the article reviews.")
    ca_review: str = dspy.InputField(description="Critique of contribution/novelty.")
    methodology_review: str = dspy.InputField(description="Critique of scale and experiments.")
    clarity_review: str = dspy.InputField(description="Critique of writing and notation.")

    final_justification: str = dspy.OutputField(description="Briefly explain the main reason for the final score, focusing on why it met or failed the bar.")
    score: float = dspy.OutputField(description="Predicted score (1-5).")
    decision: str = dspy.OutputField(description="Accept or Reject decision.")



class LLMJudge(dspy.Signature):
    """You are an expert strict reviewer. Compare the generated review with the human review.
    Rate the accuracy and significance on a scale from 1 to 5.

    Be objective in scoring.
    """
    generated_review: str = dspy.InputField(description="Generated review text by the LLM.")
    human_review: str = dspy.InputField(description="Human-written review text to compare against.")
    evaluation: float = dspy.OutputField(description="LLM-based evaluation score (1-5).")


# =============================================================================
# MARG-INSPIRED MULTI-AGENT SIGNATURES (Original Adaptation)
# =============================================================================

class ReviewPlan(dspy.Signature):
    """
    You are the COORDINATOR planning a review strategy for a scientific paper.

    Analyze the paper's abstract and structure to create a focused review plan.
    Identify what aspects need the most attention from specialist reviewers.

    This helps specialists focus on the most important aspects of this specific paper.
    """
    abstract: str = dspy.InputField(description="Paper abstract.")
    paper_type: str = dspy.InputField(description="Detected paper type: empirical/theoretical/survey/system.")

    focus_contribution: str = dspy.OutputField(description="What should the Contribution reviewer focus on for THIS paper?")
    focus_methodology: str = dspy.OutputField(description="What should the Methodology reviewer focus on for THIS paper?")
    focus_clarity: str = dspy.OutputField(description="What should the Clarity reviewer focus on for THIS paper?")
    key_claims: str = dspy.OutputField(description="List the main claims that need verification.")


class PaperTypeDetector(dspy.Signature):
    """Detect the type of scientific paper based on abstract."""
    abstract: str = dspy.InputField(description="Paper abstract.")
    paper_type: str = dspy.OutputField(description="One of: empirical, theoretical, survey, system, hybrid")
    reasoning: str = dspy.OutputField(description="Brief reasoning for classification.")


class QualityGate(dspy.Signature):
    """
    You are a QUALITY CHECKER for review comments.

    Check if a review comment is specific and actionable, or too generic/vague.
    Do NOT judge the paper - only judge the QUALITY of the review comment itself.

    Generic comments like "needs more experiments" or "writing could be improved"
    should be flagged. Specific comments with concrete examples are good.
    """
    review_text: str = dspy.InputField(description="The review comment to check.")
    aspect: str = dspy.InputField(description="Which aspect: contribution/methodology/clarity.")

    is_specific: bool = dspy.OutputField(description="True if the review is specific and actionable.")
    quality_score: float = dspy.OutputField(description="Quality score 0.0-1.0 (1.0 = very specific).")
    improvement_hint: str = dspy.OutputField(description="If not specific, what's missing? Empty if good.")


class ScoreCalibrator(dspy.Signature):
    """
    You are a SCORE CALIBRATOR ensuring review scores use the full 1-5 range.

    STEP 1: Classify the paper into ONE bucket based on the review findings.

    CALIBRATION EXAMPLES (from real ICLR reviews):

    [STRONG_REJECT] Score: 1.0 - 1.9
    Signs: "Fundamentally flawed methodology", "Claims not supported by any evidence",
           "Contribution is trivial or already published", "Ethical concerns".
    Example: Paper claims novelty but the method is identical to prior work.

    [WEAK_REJECT] Score: 2.0 - 2.9
    Signs: "Notation unclear", "Undefined matrices/variables", "Weak motivation",
           "Tested only on toy datasets (CIFAR, MNIST)", "Missing recent baselines",
           "Incremental contribution - just another loss term".
    Example: Method works but only tested on MNIST, no comparison to SOTA from 2023+.

    [BORDERLINE] Score: 3.0 - 3.5
    Signs: "Good performance but incremental novelty", "Clear method but small experiments",
           "Some merit but notable weaknesses", "Needs major revision".
    Example: Solid empirical results but contribution is an obvious extension.

    [WEAK_ACCEPT] Score: 3.6 - 4.2
    Signs: "Clear contribution", "Good experiments on standard benchmarks",
           "Minor clarity issues", "Would benefit the community".
    Example: Novel approach with comprehensive experiments, minor writing issues.

    [STRONG_ACCEPT] Score: 4.3 - 5.0
    Signs: "Unifies existing theories", "Root cause analysis - explains WHY",
           "Scales to large models (1B+ params)", "Simple yet effective",
           "Will change how people think about this problem".
    Example: Identifies fundamental flaw in common practice, proposes elegant fix.

    STEP 2: Assign a score WITHIN that bucket's range.

    CRITICAL RULES:
    - If contribution_type is "Incremental" → Cannot be above BORDERLINE (max 3.5)
    - If "missing baselines" or "toy datasets only" in weaknesses → WEAK_REJECT or lower
    - If "undefined variables" or "notation unclear" in weaknesses → Deduct at least 0.5
    - If contribution_type is "Breakthrough" → At least WEAK_ACCEPT (min 3.6)

    Be decisive. Use the full range. Most papers are NOT borderline.
    """
    synthesis: str = dspy.InputField(description="The synthesized review from coordinator.")
    strengths: str = dspy.InputField(description="Key strengths identified.")
    weaknesses: str = dspy.InputField(description="Key weaknesses identified.")
    contribution_type: str = dspy.InputField(description="Incremental/Significant/Breakthrough.")

    bucket: str = dspy.OutputField(description="One of: STRONG_REJECT, WEAK_REJECT, BORDERLINE, WEAK_ACCEPT, STRONG_ACCEPT")
    bucket_reasoning: str = dspy.OutputField(description="Why this bucket? Reference specific issues from the review.")
    calibrated_score: float = dspy.OutputField(description="Final score within the bucket's range (1.0-5.0).")
    decision: str = dspy.OutputField(description="Accept if score >= 3.6, Reject otherwise.")


class CoordinatorSynthesis(dspy.Signature):
    """
    You are the COORDINATOR (Area Chair) synthesizing reviews from your committee.

    You receive reviews from three specialists, each with a quality score indicating
    how specific/actionable their feedback is. Weight higher-quality reviews more.

    Your role:
    1. Synthesize the key findings from all reviewers
    2. Give more weight to specific, well-justified critiques
    3. Identify clear strengths and weaknesses
    4. Be fair - if specialists disagree, acknowledge the uncertainty

    Do NOT assign a final score here - that will be done by the Score Calibrator.
    Focus on accurately summarizing the findings.
    """
    review_plan: str = dspy.InputField(description="The original review plan/focus areas.")
    ca_review: str = dspy.InputField(description="Contribution review.")
    ca_quality: float = dspy.InputField(description="Quality score of contribution review (0-1).")
    ca_contribution_type: str = dspy.InputField(description="Incremental/Significant/Breakthrough.")
    methodology_review: str = dspy.InputField(description="Methodology review.")
    methodology_quality: float = dspy.InputField(description="Quality score of methodology review (0-1).")
    clarity_review: str = dspy.InputField(description="Clarity review.")
    clarity_quality: float = dspy.InputField(description="Quality score of clarity review (0-1).")

    synthesis: str = dspy.OutputField(description="Synthesized summary of all reviews.")
    strengths: str = dspy.OutputField(description="Key strengths identified - be specific.")
    weaknesses: str = dspy.OutputField(description="Key weaknesses identified - be specific.")
    suggestions: str = dspy.OutputField(description="Constructive suggestions for improvement.")


# =============================================================================
# SPECIALIST AGENT MODULES (wrap existing Signatures with focus injection)
# =============================================================================

class ContributionAgent(dspy.Module):
    """
    Specialist agent for contribution/novelty assessment.
    Uses your existing CAReview prompt but can receive focus guidance.
    """
    def __init__(self):
        super().__init__()
        self.reviewer = dspy.ChainOfThought(CAReview)

    def forward(self, introduction: str, conclusion: str, focus_hint: str = "") -> tuple:
        # Inject focus hint into introduction if provided
        if focus_hint:
            introduction = f"[COORDINATOR FOCUS: {focus_hint}]\n\n{introduction}"

        result = self.reviewer(
            introduction_section=introduction,
            conclusion_section=conclusion
        )

        # Extract contribution type for scoring hint
        contribution_type = getattr(result, 'contribution_type', 'Unknown')
        score_estimate = 3.0
        if "Breakthrough" in contribution_type:
            score_estimate = 4.5
        elif "Significant" in contribution_type:
            score_estimate = 3.8
        elif "Incremental" in contribution_type:
            score_estimate = 2.5

        review = AgentReview(
            agent_name="ContributionAgent",
            review_text=result.review,
            score=score_estimate,
            confidence=0.8,
            strengths=[],
            weaknesses=[]
        )
        return review, contribution_type


class MethodologyAgent(dspy.Module):
    """
    Specialist agent for methodology and experiments assessment.
    Uses your existing MethodologyReview prompt but can receive focus guidance.
    """
    def __init__(self):
        super().__init__()
        self.reviewer = dspy.ChainOfThought(MethodologyReview)

    def forward(self, methods: str, experiments: str, focus_hint: str = "") -> AgentReview:
        # Inject focus hint if provided
        if focus_hint:
            methods = f"[COORDINATOR FOCUS: {focus_hint}]\n\n{methods}"

        result = self.reviewer(
            methods_section=methods,
            experiments_section=experiments
        )

        return AgentReview(
            agent_name="MethodologyAgent",
            review_text=result.review,
            score=3.0,
            confidence=0.8,
            strengths=[],
            weaknesses=[]
        )


class ClarityAgent(dspy.Module):
    """
    Specialist agent for clarity and presentation assessment.
    Uses your existing ClarityReview prompt but can receive focus guidance.
    """
    def __init__(self):
        super().__init__()
        self.reviewer = dspy.ChainOfThought(ClarityReview)

    def forward(self, article_text: str, focus_hint: str = "") -> AgentReview:
        # Inject focus hint if provided
        if focus_hint:
            article_text = f"[COORDINATOR FOCUS: {focus_hint}]\n\n{article_text}"

        result = self.reviewer(article_text=article_text)

        return AgentReview(
            agent_name="ClarityAgent",
            review_text=result.review,
            score=3.0,
            confidence=0.8,
            strengths=[],
            weaknesses=[]
        )


# =============================================================================
# COORDINATOR AGENT (Orchestrates the MARG-inspired workflow)
# =============================================================================

class CoordinatorAgent(dspy.Module):
    """
    Coordinator that orchestrates the multi-agent review process.

    PHASE 1: SECTION EXTRACTION
       - Extract Abstract, Introduction, Methods, Experiments, Conclusion

    PHASE 2: PLANNING
       - Detect paper type (empirical/theoretical/survey/system)
       - Create focused review plan with specific focus hints for each specialist

    PHASE 3: SPECIALIST AGENTS (with focus hints from plan)
       - ContributionAgent: Novelty, motivation, impact (uses CAReview prompt)
       - MethodologyAgent: Experiments, baselines, rigor (uses MethodologyReview prompt)
       - ClarityAgent: Writing, notation, figures (uses ClarityReview prompt)

    PHASE 4: QUALITY GATE
       - Check each review for specificity (not generic)
       - Assign quality scores (0-1) to weight reviews

    PHASE 5: SYNTHESIS
       - Weighted aggregation of specialist reviews
       - Identify strengths, weaknesses, suggestions

    PHASE 6: SCORE CALIBRATION (ensures score diversity)
       - Classify into bucket: STRONG_REJECT/WEAK_REJECT/BORDERLINE/WEAK_ACCEPT/STRONG_ACCEPT
       - Assign calibrated score within bucket range
       - Final Accept/Reject decision
       
    """

    def __init__(self):
        super().__init__()
        # Planning
        self.type_detector = dspy.Predict(PaperTypeDetector)
        self.planner = dspy.ChainOfThought(ReviewPlan)

        # Section extraction
        self.section_extractor = dspy.ChainOfThought(SectionExtractor)

        # Specialist agents
        self.contribution_agent = ContributionAgent()
        self.methodology_agent = MethodologyAgent()
        self.clarity_agent = ClarityAgent()

        # Quality gate
        self.quality_checker = dspy.Predict(QualityGate)

        # Synthesis
        self.synthesizer = dspy.ChainOfThought(CoordinatorSynthesis)

        # Score calibration (ensures score diversity)
        self.score_calibrator = dspy.ChainOfThought(ScoreCalibrator)

    def _check_quality(self, review_text: str, aspect: str) -> tuple:
        """Run quality gate on a review."""
        result = self.quality_checker(
            review_text=review_text,
            aspect=aspect
        )
        return float(result.quality_score), result.improvement_hint

    def forward(self, article_text: str) -> ReviewState:
        """Execute the full multi-agent review workflow."""

        state = ReviewState(article_text=article_text)

        # =====================================================================
        # PHASE 1: SECTION EXTRACTION
        # =====================================================================
        sections = self.section_extractor(article_text=article_text)
        state.sections = {
            "abstract": sections.Abstract,
            "introduction": sections.Introduction,
            "methods": sections.Methods,
            "experiments": sections.Experiments,
            "conclusion": sections.Conclusion
        }

        # =====================================================================
        # PHASE 2: PLANNING (Coordinator creates review strategy)
        # =====================================================================
        paper_type_result = self.type_detector(abstract=state.sections["abstract"])
        paper_type = paper_type_result.paper_type

        plan = self.planner(
            abstract=state.sections["abstract"],
            paper_type=paper_type
        )

        review_plan_text = f"""
                Paper Type: {paper_type}
                Key Claims: {plan.key_claims}
                Focus Areas:
                - Contribution: {plan.focus_contribution}
                - Methodology: {plan.focus_methodology}
                - Clarity: {plan.focus_clarity}
                """
        state.coordinator_notes.append(f"Review Plan: {review_plan_text}")

        # =====================================================================
        # PHASE 3: SPECIALIST AGENTS (with focus hints from plan)
        # =====================================================================

        # Contribution Agent
        ca_review, contribution_type = self.contribution_agent(
            introduction=state.sections["introduction"],
            conclusion=state.sections["conclusion"],
            focus_hint=plan.focus_contribution
        )
        state.agent_reviews["contribution"] = ca_review

        # Methodology Agent
        meth_review = self.methodology_agent(
            methods=state.sections["methods"],
            experiments=state.sections["experiments"],
            focus_hint=plan.focus_methodology
        )
        state.agent_reviews["methodology"] = meth_review

        # Clarity Agent
        clarity_review = self.clarity_agent(
            article_text=article_text,
            focus_hint=plan.focus_clarity
        )
        state.agent_reviews["clarity"] = clarity_review

        # =====================================================================
        # PHASE 4: QUALITY GATE (check specificity of each review)
        # =====================================================================
        ca_quality, ca_hint = self._check_quality(ca_review.review_text, "contribution")
        meth_quality, meth_hint = self._check_quality(meth_review.review_text, "methodology")
        clarity_quality, clarity_hint = self._check_quality(clarity_review.review_text, "clarity")

        # Update confidence based on quality
        ca_review.confidence = ca_quality
        meth_review.confidence = meth_quality
        clarity_review.confidence = clarity_quality

        state.coordinator_notes.append(
            f"Quality Scores - CA: {ca_quality:.2f}, Meth: {meth_quality:.2f}, Clarity: {clarity_quality:.2f}"
        )

        # =====================================================================
        # PHASE 5: SYNTHESIS (weighted by quality scores)
        # =====================================================================
        synthesis = self.synthesizer(
            review_plan=review_plan_text,
            ca_review=ca_review.review_text,
            ca_quality=ca_quality,
            ca_contribution_type=contribution_type,
            methodology_review=meth_review.review_text,
            methodology_quality=meth_quality,
            clarity_review=clarity_review.review_text,
            clarity_quality=clarity_quality
        )

        state.final_summary = synthesis.synthesis
        state.coordinator_notes.extend([
            f"Strengths: {synthesis.strengths}",
            f"Weaknesses: {synthesis.weaknesses}",
            f"Suggestions: {synthesis.suggestions}"
        ])

        # =====================================================================
        # PHASE 6: SCORE CALIBRATION (ensures score diversity)
        # =====================================================================
        calibration = self.score_calibrator(
            synthesis=synthesis.synthesis,
            strengths=synthesis.strengths,
            weaknesses=synthesis.weaknesses,
            contribution_type=contribution_type
        )

        state.final_score = float(calibration.calibrated_score)
        state.final_decision = calibration.decision
        state.coordinator_notes.extend([
            f"Score Bucket: {calibration.bucket}",
            f"Bucket Reasoning: {calibration.bucket_reasoning}",
            f"Calibrated Score: {calibration.calibrated_score}",
            f"Decision: {calibration.decision}"
        ])

        return state


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_article(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def load_human_review(file_path):
    """
    Loads text from valid reviews only.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Use the centralized filtering logic
    valid_reviews = get_valid_reviews(data)
    
    reviews_text = []
    for review in valid_reviews:
        comments = review.get("comments", "")
        reviews_text.append(comments)
    
    combined_review = "\n\n".join(reviews_text)
    return combined_review


def get_human_scores(file_path):
    """
    Extracts and averages human scores from valid reviews only.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Use the centralized filtering logic
    valid_reviews = get_valid_reviews(data)
    
    scores = []
    for review in valid_reviews:
        try:
            # Handle both string "8" and int 8
            val = float(review["RECOMMENDATION"])
            # if val > 5  :
            #     val = val / 2  # Normalize 1-10 scale to 1-5
            scores.append(val)
        except (ValueError, TypeError):
            continue
    
    if not scores:
        return 0.0
    
    
    avg_score = sum(scores) / len(scores)
    return avg_score


rougel = load("rouge")

# -------------------------------------------------------------------------
# ROUGE-L Score
# -------------------------------------------------------------------------

def calculate_rouge_l(generated_text, reference_text):
    """Calculate ROUGE-L score between generated and reference text."""
    results = rougel.compute(
        predictions=[generated_text],
        references=[reference_text],
        rouge_types=["rougeL"]
    )
    return results["rougeL"]




# -------------------------------------------------------------------------
# BERTScore
# -------------------------------------------------------------------------
def calculate_bertscore(generated_text, reference_text):
    """Calculate BERTScore F1 between generated and reference text."""
    P, R, F1 = score(
        [generated_text],
        [reference_text],
        lang="en",
        verbose=False,
    )
    return F1.item()


# -------------------------------------------------------------------------
# Length Difference
# -------------------------------------------------------------------------
def calculate_length_difference(generated_text, reference_text):
    """Calculate the absolute length difference between generated and reference text."""
    gen_length = len(generated_text.split())
    ref_length = len(reference_text.split())
    length_diff = abs(gen_length - ref_length)
    return length_diff  



# AUC for accept/reject classification would require multiple samples with known labels, which is not implemented here.



def evaluate_single_article(article_path, review_path):
    """
    Evaluuje jeden článok proti každej ľudskej recenzii osobitne.
    """
    try:
        # Load data
        article_text = load_article(article_path)
        
        with open(review_path, "r", encoding="utf-8") as f:
            review_data = json.load(f)
        
        #all_reviews = review_data.get("reviews", [])
        human_reviews = get_valid_reviews(review_data)
        
        # Check if we have any valid reviews
        if len(human_reviews) == 0:
            print(f"  No valid reviews found for article")
            return {"success": False, "error": "No valid reviews"}
        
        # Calculate average human score (from valid reviews only)

        # Use pre-calculated values from priprava.py
        human_avg_score = review_data.get("average_score", 0)
        human_decision = review_data.get("decision", "Unknown")

        # Fallback: Calculate if missing (for old format)
        if human_avg_score == 0 or human_decision == "Unknown":
            human_scores = []
            for r in human_reviews:
                try:
                    val = float(r["RECOMMENDATION"])
                    
                    human_scores.append(val)
                except (ValueError, TypeError):
                    continue

            if not human_scores:
                 return {"success": False, "error": "No valid scores"}

            human_avg_score = sum(human_scores) / len(human_scores)
            human_decision = "Accept" if human_avg_score >= 6.0 else "Reject"
        
        # =================================================================
        # MULTI-AGENT WORKFLOW (MARG-Inspired)
        # =================================================================
        coordinator = CoordinatorAgent()
        review_state = coordinator(article_text=article_text)

        # Print workflow phases
        print("\n" + "=" * 60)
        print("PHASE 1: SECTION EXTRACTION")
        print("=" * 60)
        print(f"Abstract: {review_state.sections.get('abstract', 'N/A')[:150]}...")
        print(f"Introduction: {review_state.sections.get('introduction', 'N/A')[:150]}...")
        print(f"Methods: {review_state.sections.get('methods', 'N/A')[:150]}...")
        print(f"Experiments: {review_state.sections.get('experiments', 'N/A')[:150]}...")
        print(f"Conclusion: {review_state.sections.get('conclusion', 'N/A')[:150]}...")

        print("\n" + "=" * 60)
        print("PHASE 2: REVIEW PLANNING")
        print("=" * 60)
        # First note contains the review plan
        if review_state.coordinator_notes:
            print(review_state.coordinator_notes[0])

        print("\n" + "=" * 60)
        print("PHASE 3: SPECIALIST AGENTS")
        print("=" * 60)
        for agent_name, agent_review in review_state.agent_reviews.items():
            print(f"\n--- {agent_review.agent_name} (confidence: {agent_review.confidence:.2f}) ---")
            print(f"{agent_review.review_text[:400]}...")

        print("\n" + "=" * 60)
        print("PHASE 4: QUALITY GATE")
        print("=" * 60)
        # Second note contains quality scores
        if len(review_state.coordinator_notes) > 1:
            print(review_state.coordinator_notes[1])

        print("\n" + "=" * 60)
        print("PHASE 5: COORDINATOR SYNTHESIS")
        print("=" * 60)
        print(f"Synthesis: {review_state.final_summary[:500]}...")
        # Print strengths, weaknesses, suggestions (notes 2-4)
        for note in review_state.coordinator_notes[2:5]:
            print(f"  {note}")

        print("\n" + "=" * 60)
        print("PHASE 6: SCORE CALIBRATION")
        print("=" * 60)
        print(f"Final Decision: {review_state.final_decision} (Score: {review_state.final_score})")
        # Print calibration notes (bucket, reasoning, etc.)
        for note in review_state.coordinator_notes[5:]:
            print(f"  {note}")
        print("=" * 60)

        # Map to old variable names for compatibility with metrics calculation
        class PredictionCompat:
            def __init__(self, state):
                self.final_justification = state.coordinator_notes[-1] if state.coordinator_notes else ""
                self.score = state.final_score
                self.decision = state.final_decision

        class SummaryCompat:
            def __init__(self, state):
                self.summary = state.final_summary

        prediction = PredictionCompat(review_state)
        final_summary = SummaryCompat(review_state)
        
        
        # Comparing with each human reviewer separately
        per_reviewer_metrics = []
        
        for human_review in human_reviews:
            human_comments = human_review.get("comments", "")
            
            # LLM Judge
            judge = dspy.Predict(LLMJudge)
            llm_evaluation = judge(
                generated_review=final_summary.summary,
                human_review=human_comments
            )

            #print(f" LLM as Judge: {llm_evaluation.evaluation}")
            
            # Calculate metrics per reviewer
            rouge_l = calculate_rouge_l(final_summary.summary, human_comments)
            bertscore_f1 = calculate_bertscore(final_summary.summary, human_comments)
            length_diff = calculate_length_difference(final_summary.summary, human_comments)
            
            per_reviewer_metrics.append({
                "llm_judge_score": float(llm_evaluation.evaluation),
                "rouge_l": rouge_l,
                "bertscore_f1": bertscore_f1,
                "length_diff": length_diff
            })
        
        
        avg_llm_judge = np.mean([m["llm_judge_score"] for m in per_reviewer_metrics])
        avg_rouge_l = np.mean([m["rouge_l"] for m in per_reviewer_metrics])
        avg_bertscore = np.mean([m["bertscore_f1"] for m in per_reviewer_metrics])
        avg_length_diff = np.mean([m["length_diff"] for m in per_reviewer_metrics])
        
        return {
            "generated_review": final_summary.summary,
            "human_reviews": human_reviews,
            "success": True,
            "final_justification": prediction.final_justification,
            "predicted_score": float(prediction.score),
            "predicted_decision": prediction.decision,
            "human_score": human_avg_score,  # spriemeruju sa vsetky recenzie pre prislusny article
            "human_decision": human_decision,
            "num_reviewers": len(human_reviews),
            "llm_judge_score": avg_llm_judge,
            "rouge_l": avg_rouge_l,
            "bertscore_f1": avg_bertscore,
            "length_diff": avg_length_diff,
            "generated_review": final_summary.summary,
            "per_reviewer_metrics": per_reviewer_metrics 
        }
    
    except Exception as e:
        print(f"Error processing {article_path}: {str(e)}")
        return {"success": False, "error": str(e)}



def calculate_aggregate_metrics(results):
    """
    Vypočíta agregované metriky zo všetkých výsledkov.
    """
    predicted_scores = [r["predicted_score"] for r in results]
    human_scores = [r["human_score"] for r in results]
    predicted_decisions = [r["predicted_decision"] for r in results]
    human_decisions = [r["human_decision"] for r in results]

    print(f"\n DEBUG - Unique human decisions: {set(human_decisions)}")
    print(f" DEBUG - Unique predicted decisions: {set(predicted_decisions)}")
    print(f" DEBUG - Count human: {len(human_decisions)}")
    print(f" DEBUG - Count predicted: {len(predicted_decisions)}")
    
    
    # Spearman correlation
    corr, _ = spearmanr(predicted_scores, human_scores)
    
    
    accuracy = accuracy_score(human_decisions, predicted_decisions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        human_decisions, 
        predicted_decisions, 
        average='binary', 
        pos_label='Accept',
        zero_division=0
    )
    
    # AUC-ROC (potrebuje numerické hodnoty)
    try:
        # Convert Accept/Reject to 1/0
        #y_true = [1 if d == "Accept" else 0 for d in human_decisions]
        #y_pred = [1 if d == "Accept" else 0 for d in predicted_decisions]
        y_scores=[r["predicted_score"] for r in results]
        y_true=[1 if d == "Accept" else 0 for d in human_decisions]
        auc = roc_auc_score(y_true, y_scores) #if len(set(y_true)) > 1 else None
    except:
        auc = None
    
    # Text quality metrics
    avg_rouge_l = np.mean([r["rouge_l"] for r in results])
    avg_bertscore = np.mean([r["bertscore_f1"] for r in results])
    avg_llm_judge = np.mean([r["llm_judge_score"] for r in results])
    avg_length_diff = np.mean([r["length_diff"] for r in results])
    
    return {
        "n_articles": len(results),
        "spearman_corr": corr,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "avg_rouge_l": avg_rouge_l,
        "avg_bertscore": avg_bertscore,
        "avg_llm_judge": avg_llm_judge,
        "avg_length_diff": avg_length_diff
    }
    


def evaluate_dataset(articles_dir, reviews_dir, output_file="evaluation_results.json"):
    """
    Evaluuje celý dataset článkov.
    
    Args:
        articles_dir: Cesta k priečinku s článkami (markdown)
        reviews_dir: Cesta k priečinku s human reviews (json)
        output_file: Kam uložiť výsledky
    
    Returns:
        dict: Agregované metriky
    """
    articles_path = Path(articles_dir)
    reviews_path = Path(reviews_dir)
    
    results = []
    
    # Prejdi všetky články
    for article_file in articles_path.glob("reconstructed_article_temp*.md"):
        # Extract article ID (napr. "12" z "reconstructed_article_temp_12.md")
        article_id = article_file.stem.replace("reconstructed_article_temp_", "")
        review_file = reviews_path / f"{article_id}.json"
        
        if not review_file.exists():
            print(f"Skipping {article_file.name} - no review found")
            continue
        
        print(f"\nProcessing article {article_id}...")
        result = evaluate_single_article(str(article_file), str(review_file))
        
        if result["success"]:
            result["article_id"] = article_id
            results.append(result)
            print(f" Article {article_id} processed successfully")
        else:
            print(f" Article {article_id} failed: {result.get('error', 'Unknown error')}")
    
    # Calculate aggregate metrics
    if results:
        metrics = calculate_aggregate_metrics(results)
        
        # Saving results
        output_data = {
            "individual_results": results,
            "aggregate_metrics": metrics
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("AGGREGATE METRICS")
        print(f"{'='*60}")
        print(f"Number of articles: {metrics['n_articles']}")
        print(f"\n--- Score Prediction ---")
        print(f"Spearman Correlation: {metrics['spearman_corr']:.4f}")
        print(f"\n--- Accept/Reject Classification ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        #print(f"Precision: {metrics['precision']:.4f}")
        #print(f"Recall: {metrics['recall']:.4f}")
        #print(f"F1 Score: {metrics['f1']:.4f}")
        if metrics['auc'] is not None:
            print(f"AUC-ROC: {metrics['auc']:.4f}")
        print(f"\n--- Text Quality ---")
        print(f"Avg ROUGE-L: {metrics['avg_rouge_l']:.4f}")
        print(f"Avg BERTScore F1: {metrics['avg_bertscore']:.4f}")
        print(f"Avg LLM Judge Score: {metrics['avg_llm_judge']:.4f}")
        print(f"Avg Length Difference: {metrics['avg_length_diff']:.1f} words")

        print(f"\nResults saved to: {output_file}")
        print(f"{'='*60}\n")
        print("GENERATED REVIEWS SAMPLES:")
        print(f"{'='*60}\n")


        for idx, result in enumerate(results, 1):
            article_id = result.get("article_id", "unknown")
            generated_review = result.get("generated_review", "N/A")
            predicted_score = result.get("predicted_score", "N/A")
            predicted_decision = result.get("predicted_decision", "N/A")
            human_score = result.get("human_score", "N/A")
            human_decision = result.get("human_decision", "N/A")
            
            print(f"{'─'*60}")
            print(f"ARTICLE {idx}: {article_id}")
            print(f"{'─'*60}")
            print(f"Predicted: {predicted_decision} (Score: {predicted_score:.2f})")
            print(f"Human:     {human_decision} (Score: {human_score:.2f})")
            print(f"\n--- Generated Review ---")
            print(generated_review)
            print(f"\n")

        print(f"{'='*60}\n")

        
        return metrics
    else:
        print("No results to aggregate!")
        return None



# ICLR 2017 - filtrovanie validnych recenzii , IBA tie ktore maju RECOMMENDATION a dostatocne dlhe comments
# def get_valid_reviews(review_data):
#     """
#     Filtruje a vráti len validné recenzie.
#     """
#     all_reviews = review_data.get("reviews", [])
#     valid_reviews = []
    
#     for review in all_reviews:
#         # Must have RECOMMENDATION
#         if "RECOMMENDATION" not in review or review["RECOMMENDATION"] is None:
#             continue
        
#         # Must have meaningful comments
#         comments = review.get("comments", "")
#         if not comments or len(comments.strip()) < 50:  

#             continue
        
#         is_meta = review.get("is_meta_review", False) or review.get("IS_META_REVIEW", False)
#         if is_meta:
#             continue
        
#         valid_reviews.append(review)
    
#     return valid_reviews


# def get_valid_reviews(review_data):
#     """
#     Robust filtering for reviews. 
#     Handles:
#     - Exclusion of Meta Reviews and Committee Decisions
#     - Missing RECOMMENDATION keys
#     - String vs Integer scores
#     - Short/Empty comments
#     """
#     all_reviews = review_data.get("reviews", [])
#     valid_reviews = []
    
#     for review in all_reviews:
#         # 1. Check for Meta Reviews (Key can be lowercase or uppercase)
#         is_meta = review.get("IS_META_REVIEW", False) or review.get("is_meta_review", False)
#         if is_meta:
#             continue

#         # 2. Check Titles for administrative entries 
#         # (e.g., 'ICLR committee final decision' or 'Source code')
#         title = review.get("TITLE", "").lower()
#         if "committee final decision" in title or "meta review" in title or "source code" in title:
#             continue

#         # 3. Must have a RECOMMENDATION (score)
#         # We skip entries that have no score (like questions about code)
#         if "RECOMMENDATION" not in review or review["RECOMMENDATION"] is None:
#             continue
            
#         # 4. Must have meaningful comments
#         comments = review.get("comments", "")
#         if not comments or len(comments.strip()) < 50:  
#             continue
        
#         valid_reviews.append(review)
    
#     return valid_reviews

def get_valid_reviews(review_data):
    """
    Robust filtering for reviews. 
    Handles both ICLR 2017 and ICLR 2025 formats.
    - ICLR 2017: {"reviews": [...]}
    - ICLR 2025: {"conference": "...", "reviews": [...], "title": ..., etc.}
    
    Filters out:
    - Meta Reviews and Committee Decisions
    - Missing RECOMMENDATION keys
    - Short/Empty comments
    """
    # Handle both old format (just "reviews" key) and new format (full structure)
    if "reviews" in review_data:
        all_reviews = review_data.get("reviews", [])
    else:
        # Legacy support: if review_data IS the reviews array
        all_reviews = review_data if isinstance(review_data, list) else []
    
    valid_reviews = []
    
    for review in all_reviews:
        # 1. Check for Meta Reviews (Key can be lowercase or uppercase)
        is_meta = review.get("IS_META_REVIEW", False) or review.get("is_meta_review", False)
        if is_meta:
            continue

        # 2. Check Titles for administrative entries 
        # (e.g., 'ICLR committee final decision' or 'Source code')
        title = review.get("TITLE", "").lower()
        if "committee final decision" in title or "meta review" in title or "source code" in title:
            continue

        # 3. Must have a RECOMMENDATION (score)
        # We skip entries that have no score (like questions about code)
        if "RECOMMENDATION" not in review or review["RECOMMENDATION"] is None:
            continue
            
        # 4. Must have meaningful comments
        comments = review.get("comments", "")
        if not comments or len(comments.strip()) < 50:  
            continue
        
        valid_reviews.append(review)
    
    return valid_reviews



def log_usage_to_csv(total_prompt, total_completion, total_cost, n_articles, filename="usage_log.csv"):
    file_exists = Path(filename).is_file()
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Hlavička, ak súbor ešte neexistuje
        if not file_exists:
            writer.writerow(['timestamp', 'n_articles', 'prompt_tokens', 'completion_tokens', 'total_cost_usd'])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_articles,
            total_prompt,
            total_completion,
            f"{total_cost:.5f}"
        ])

## main
if __name__ == "__main__":
   
    # ARTICLES_PATH = "C:\\Users\\katka\\BAKALARKA\\reconstructed_articles\\reconstructed_article_7.md"
    # REVIEWS_PATH = "C:\\Users\\katka\\BAKALARKA\\human_reviews\\7.json"
    # ARTICLES_FOLDER = "C:\\Users\\katka\\BAKALARKA\\ready_md"
    # REVIEWS_FOLDER = "C:\\Users\\katka\\BAKALARKA\\PeerRead\\data\\iclr_2017\\test\\reviews"
    # ARTICLES_FOLDER = "C:\\Users\\katka\\BAKALARKA\\reconstructed_articles"
    # REVIEWS_FOLDER = "C:\\Users\\katka\\BAKALARKA\\human_reviews"

    ARTICLES_FOLDER = "C:/Users/katka/BAKALARKA/new_ready_GEPA100md"
    REVIEWS_FOLDER = "C:/Users/katka/BAKALARKA/new_human_reviews_GEPA100"

           
    
    metrics = evaluate_dataset(
        articles_dir=ARTICLES_FOLDER,
        reviews_dir=REVIEWS_FOLDER
    )


    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


    # history=dspy.inspect_history()

    if metrics:
        # ... (výpis metrík ostáva rovnaký) ...

        print(f"\n{'='*60}")
        print("USAGE & COST REPORT")
        print(f"{'='*60}")
        
        total_prompt = 0
        total_completion = 0
        
        # Ceny (skontrolujte si aktuálne ceny vášho providera)
        PRICE_INPUT = 0.15 / 1_000_000  # Cena za 1 token
        PRICE_OUTPUT = 0.60 / 1_000_000 # Cena za 1 token

        if hasattr(lmm, 'history') and lmm.history:
            for interaction in lmm.history:
                # Extrakcia usage z rôznych možných štruktúr
                usage = interaction.get('usage') or {}
                if not usage and 'response' in interaction:
                    # Skús hľadať v response (ak litellm vracia objekt)
                    resp = interaction['response']
                    if hasattr(resp, 'usage'):
                        if isinstance(resp, dict):
                            # Ak je resp slovník
                            usage_data = resp.get('usage', {})
                            p = usage_data.get('prompt_tokens', 0)
                            c = usage_data.get('completion_tokens', 0)
                        elif hasattr(resp, 'usage'):
                            # Ak je resp objekt (ako ho vracajú niektoré verzie LiteLLM/OpenAI)
                            if isinstance(resp.usage, dict):
                                p = resp.usage.get('prompt_tokens', 0)
                                c = resp.usage.get('completion_tokens', 0)
                            else:
                                p = getattr(resp.usage, 'prompt_tokens', 0)
                                c = getattr(resp.usage, 'completion_tokens', 0)
                        else:
                            p, c = 0, 0

                        usage = {'prompt_tokens': p, 'completion_tokens': c}
                
                total_prompt += usage.get('prompt_tokens', 0)
                total_completion += usage.get('completion_tokens', 0)
        
        else:
            print("Warning: No history found in LM object.")

        total_cost = (total_prompt * PRICE_INPUT) + (total_completion * PRICE_OUTPUT)

        log_usage_to_csv(total_prompt, total_completion, total_cost, metrics['n_articles'])
        

        print(f"Total Prompt Tokens:     {total_prompt}")
        print(f"Total Completion Tokens: {total_completion}")
        print(f"Total All Tokens:        {total_prompt + total_completion}")
        print(f"{'-'*30}")
        print(f"ESTIMATED COST:          ${total_cost:.5f}")
        print(f"Usage logged to usage_log.csv")




