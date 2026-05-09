"""Early LLM workflow using dspy.Predict only (non-agentic) — precursor to the main pipeline."""

### teraz pouzijeme len dspy.predict  - LLM workflow nie agentic system
from unittest import result
import dspy
import os
from dotenv import load_dotenv
import litellm
#from dspy.evaluate import  
from bert_score import score
from evaluate import load
import json
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from pathlib import Path
import numpy as np
import csv
from datetime import datetime

# GEPA
from dspy.teleprompt import MIPROv2
from dspy.evaluate import evaluate
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
import sys



load_dotenv()
TF_ENABLE_ONEDNN_OPTS=0
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


model="openai/gpt-4.1-nano-fiit"

#lmm=dspy.configure(lm=dspy.LM(model=model, api_key=API_KEY, api_base=API_BASE,temperature=0.3,cache=False))


lmm = dspy.LM(model=model, api_key=API_KEY, api_base=API_BASE, temperature=0.0, max_tokens=8000)  # Reduced tokens + cache enabled


llm2= dspy.LM(model="openai/gpt-4.1-mini-fiit", api_key=API_KEY,api_base=API_BASE, temperature=0.0,max_tokens=8000)

    
# )
dspy.configure(lm=llm2)

# -------------------------------------------------------------------------

# PRIDAT DETEKCIU PROMPT INJECTION - NOVA SIGNATURE ASI !! 
# UROBIT VIAC AGENTIC - PROTOCOL MULTI AGENT + JEDEN CENTRALNY COORDINATOR
# PRIDAT USER PROMPT !! - INPUT FIELD

class SectionExtractor(dspy.Signature):
    """
    Extract specific sections from a scientific article provided in markdown.
    
    RULES:
    1. DO NOT summarize. Extract the text including all subheadings (e.g., ## 2.1, ###).
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
    contribution_type: str = dspy.OutputField(description="Classify as: 'Incremental' (Score < 7), 'Significant' (Score 7-8.5), or 'Breakthrough' (Score > 8.5).")
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
    1. Are all variables defined upon first use? (If NO -> Score < 7)
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
    summary: str = dspy.OutputField(description="Generated specific summary with section-grounded evidence, strengths, weaknesses, methodological concerns, clarity issues, and rubric-aligned recommendations.")


class DecisionPredictor(dspy.Signature):
    """
    You are an Area Chair at a top-tier AI conference. Your role is to
    synthesize specialized reviews and make a final high-stakes decision.

    DECISION LOGIC :
        
        1. METHODOLOGY FIRST: If the Methodology Review identifies FUNDAMENTAL flaws 
        (e.g., incorrect mathematical proofs, impossible results, or a total 
        absence of standard SOTA baselines), the paper MUST be REJECTED. 
        Minor experimental gaps should only lower the score, not trigger an 
        automatic rejection.
        
        2. CONTRIBUTION: If the work is PURELY incremental (e.g., a known method 
        applied to a trivial dataset without new insight), it should be REJECTED.
        
        3. HOLISTIC VETO: A paper should only be ACCEPTED if it is technically sound 
        AND provides a non-trivial contribution. However, exceptional novelty 
        can occasionally compensate for minor clarity issues.

    SCORING GUIDELINES:
    - 9.0 - 10.0: Rare. Groundbreaking, flawless. Perfect methodology and impact.
    - 7.5 - 8.9: Accept. Solid contribution, technically sound, well-supported.
    - 7.0 - 7.4: Borderline. Lean towards Reject UNLESS the contribution is 
      unique/original or solves a long-standing niche problem.
    - 4.0 - 6.9: Reject. Significant methodological gaps, very weak novelty, 
      or major clarity issues that hinder understanding.
    - 1.0 - 3.9: Strong Reject. Fundamental errors, zero novelty, or 
      unethical/plagiarized methods.

   
    """
    # ca_review: str = dspy.InputField(description="Contribution review.")
    # methodology_review: str = dspy.InputField(description="Methodology review.")
    # clarity_review: str = dspy.InputField(description="Clarity review.")
    summary: str = dspy.InputField(description="Summary of the reviews to inform the decision.")

    final_justification: str = dspy.OutputField(description="Brief justification for decision. Weigh weaknesses heavily.")
    score: float = dspy.OutputField(description="Quality score from 1 to 10. EACH paper must get a DIFFERENT score based on its specific strengths and weaknesses. Do NOT default to the same score for every paper.")
    decision: str = dspy.OutputField(description="ONLY 'Accept' or 'Reject'. When in doubt, Reject.")



class LLMJudge(dspy.Signature):
    """You are an expert strict reviewer. Compare the generated review with the human review.
    Rate the accuracy and significance on a scale from 1 to 10.

    Be objective in scoring.
    """
    generated_review: str = dspy.InputField(description="Generated review text by the LLM.")
    human_review: str = dspy.InputField(description="Human-written review text to compare against.")
    evaluation: float = dspy.OutputField(description="LLM-based evaluation score (1-10).")


# -------------------------------------------------------------------------
# GEPA: Review Pipeline Module
# -------------------------------------------------------------------------

class ReviewPipeline(dspy.Module):
    """
    Enhanced pipeline with validation tracking
    """
    def __init__(self):
        super().__init__()
        self.section_extractor = dspy.Predict(SectionExtractor)
        
        
        self.ca_reviewer = dspy.Predict(CAReview)
        self.methodology_reviewer = dspy.Predict(MethodologyReview)
        self.clarity_reviewer = dspy.Predict(ClarityReview)
        self.decision_predictor = dspy.ChainOfThought(DecisionPredictor)
        self.summary_generator = dspy.Predict(SummaryGenerator)
        
        # Track intermediate predictions for debugging
        self.debug_mode = False
    
    def forward(self, article_text):
        # Extract sections with error handling
        try:
            sections = self.section_extractor(article_text=article_text)
        
            
        except Exception as e:
            print(f"Section extraction failed: {e}")
            # Return dummy prediction to avoid crashes
            return dspy.Prediction(
                decision="Reject",
                score=1.0,
                justification="Failed to extract sections",
                summary="Error in processing"
            )
        
        # Validate extracted sections
        if len(sections.Introduction) < 50:
            print("Warning: Short introduction extracted")
        
        # Generate reviews
        ca_review = self.ca_reviewer(
            introduction_section=sections.Introduction,
            conclusion_section=sections.Conclusion
        )
        
        methodology_review = self.methodology_reviewer(
            methods_section=sections.Methods,
            experiments_section=sections.Experiments
        )
        
        clarity_review = self.clarity_reviewer(article_text=article_text)
        
        ca_combined = f"Type: {ca_review.contribution_type}\nReview: {ca_review.review}"

        # Generate summary
        summary = self.summary_generator(
            ca_review=ca_combined,
            methodology_review=methodology_review.review,
            clarity_review=clarity_review.review
        )
        
        # Final decision
        prediction = self.decision_predictor(
            summary=ca_combined + "\n\n" + "Methodology Review:\n" + methodology_review.review + "\n\n" + "Clarity Review:\n" + clarity_review.review
        )
        
        
        result = dspy.Prediction(
            decision=prediction.decision,
            score=prediction.score,
            justification=prediction.final_justification,
            summary=summary.summary,
            ca_review=ca_review.review,
            methodology_review=methodology_review.review,
            clarity_review=clarity_review.review,
            # Include sections for debugging
            extracted_sections=sections if self.debug_mode else None
        )
        
        return result

# -------------------------------------------------------------------------
# GEPA: Metric Function (GEPA requires 5 arguments)
# -------------------------------------------------------------------------


def gepa_combined_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Simple metric for GEPA optimization.

    Returns ScoreWithFeedback with:
    - score: 0-1 (70% decision, 30% score proximity)
    - feedback: Clear explanation of what went wrong/right

    GEPA learns from feedback - keep it simple and informative.
    """
    try:
        # 1. Rozhodnutie (0.0 alebo 1.0)
        pred_accept = "accept" in pred.decision.strip().lower()
        true_accept = "accept" in gold.decision.strip().lower()
        decision_correct = (pred_accept == true_accept)
        
        # 2. Proximita skóre (0.0 až 1.0)
        pred_score = float(pred.score)
        true_score = float(gold.score)
        score_error = abs(pred_score - true_score)
        score_proximity = max(0.0, 1.0 - (score_error / 9.0))

        # =================================================================
        # NOVÝ VÝPOČET SKÓRE (Bez drastického násobenia)
        # =================================================================
        # Dáme 60% váhu správnemu rozhodnutiu a 40% presnosti číselného skóre
        final_score = (0.6 * float(decision_correct)) + (0.4 * score_proximity)

        # Jemná penalizácia za False Accept (miesto 0.5x použijeme fixný odpočet)
        # Toto model stále učí opatrnosti, ale neničí to jeho ochotu prijať článok.
        if pred_accept and not true_accept:
            final_score = max(0.0, final_score - 0.15) 

        # =================================================================
        # VYLEPŠENÝ FEEDBACK PRE GEPA
        # =================================================================
        feedback_lines = []
        if decision_correct:
            feedback_lines.append(f"STRENGTH: Correct decision ({pred.decision}).")
        else:
            type_err = "False Positive" if pred_accept else "False Negative"
            feedback_lines.append(f"WEAKNESS: {type_err}. Human said {gold.decision} (Score {true_score}).")
            
        if score_error > 1.5:
            feedback_lines.append(f"ADVICE: Calibrate score. Your {pred_score} is far from human {true_score}.")

        feedback = " ".join(feedback_lines)
        return ScoreWithFeedback(score=float(final_score), feedback=feedback)

    except Exception as e:
        return ScoreWithFeedback(score=0.0, feedback=f"Metric error: {e}")


# -------------------------------------------------------------------------
# GEPA: Training Data Loading
# -------------------------------------------------------------------------

def load_gepa_training_data(articles_dir, reviews_dir, max_samples=60):
    """
    Load training examples for GEPA optimization.

    Args:
        articles_dir: Path to reconstructed articles (markdown)
        reviews_dir: Path to human reviews (JSON)
        max_samples: Maximum number of examples to load

    Returns:
        List of dspy.Example objects
    """
    examples = []
    articles_path = Path(articles_dir)
    reviews_path = Path(reviews_dir)

    article_files = list(articles_path.glob("reconstructed_article_temp*.md"))
    print(f"Found {len(article_files)} article files for GEPA training")

    for article_file in article_files[:max_samples]:
        article_id = article_file.stem.replace("reconstructed_article_temp_", "")
        review_file = reviews_path / f"{article_id}.json"

        if not review_file.exists():
            continue

        try:
            # Load article
            article_text = load_article(str(article_file))

            # Load review data
            with open(review_file, "r", encoding="utf-8") as f:
                review_data = json.load(f)

            human_decision = review_data.get("decision", "Unknown")
            human_score = review_data.get("average_score", 0)

            if human_decision == "Unknown" or human_score == 0:
                continue

            # Load human review text for feedback
            human_review_text = load_human_review(str(review_file))

            # Create example with article_id and human_review for GEPA feedback
            example = dspy.Example(
                article_text=article_text,
                decision=human_decision,
                score=float(human_score),
                article_id=article_id,
                human_review=human_review_text  # For detailed GEPA feedback
            ).with_inputs("article_text")

            examples.append(example)
            print(f"  Loaded {article_id}: {human_decision} (score: {human_score})")

        except Exception as e:
            print(f"  Error loading {article_id}: {e}")
            continue

    return examples

def stratified_split(trainset, train_ratio=0.68, val_ratio=0.16, test_ratio=0.16, min_per_class=2, seed=42):
    """
    Stratified split into train/val/test sets.

    Args:   
        trainset: List of dspy.Example objects
        train_ratio: Proportion for training (default 0.6 = 60%)
        val_ratio: Proportion for validation (default 0.2 = 20%)
        test_ratio: Proportion for testing (default 0.2 = 20%)
        min_per_class: Minimum samples per class in each split
        seed: Random seed for reproducibility

    Returns:
        train_examples, val_examples, test_examples
    """
    import random
    random.seed(seed)  # Fixný seed pre reprodukovateľnosť

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"

    accepts = [ex for ex in trainset if 'accept' in ex.decision.lower()]
    rejects = [ex for ex in trainset if 'reject' in ex.decision.lower()]

    print(f"Dataset composition: {len(accepts)} accepts, {len(rejects)} rejects")
    print(f"Split ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")

    # Ensure minimum samples per class
    min_needed = min_per_class * 3  # Need min_per_class for each of train/val/test
    if len(accepts) < min_needed or len(rejects) < min_needed:
        raise ValueError(f"Insufficient samples per class. Need at least {min_needed} each, "
                        f"got {len(accepts)} accepts and {len(rejects)} rejects.")

    # Shuffle before splitting
    random.shuffle(accepts)
    random.shuffle(rejects)

    def split_class(examples, train_r, val_r):
        """Split a single class into train/val/test"""
        n = len(examples)
        n_train = max(min_per_class, int(n * train_r))
        n_val = max(min_per_class, int(n * val_r))
        n_test = n - n_train - n_val

        # Ensure test has at least min_per_class
        if n_test < min_per_class:
            # Borrow from train (which is largest)
            diff = min_per_class - n_test
            n_train -= diff
            n_test = min_per_class

        return (examples[:n_train],
                examples[n_train:n_train + n_val],
                examples[n_train + n_val:])

    # Split each class
    train_acc, val_acc, test_acc = split_class(accepts, train_ratio, val_ratio)
    train_rej, val_rej, test_rej = split_class(rejects, train_ratio, val_ratio)

    # Combine and shuffle
    train_examples = train_acc + train_rej
    val_examples = val_acc + val_rej
    test_examples = test_acc + test_rej

    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)

    # Print summary
    print(f"Train: {len(train_examples)} ({sum(1 for e in train_examples if 'accept' in e.decision.lower())} accepts, "
          f"{sum(1 for e in train_examples if 'reject' in e.decision.lower())} rejects)")
    print(f"Val:   {len(val_examples)} ({sum(1 for e in val_examples if 'accept' in e.decision.lower())} accepts, "
          f"{sum(1 for e in val_examples if 'reject' in e.decision.lower())} rejects)")
    print(f"Test:  {len(test_examples)} ({sum(1 for e in test_examples if 'accept' in e.decision.lower())} accepts, "
          f"{sum(1 for e in test_examples if 'reject' in e.decision.lower())} rejects)")

    return train_examples, val_examples, test_examples


# -------------------------------------------------------------------------
# GEPA: Optimization Function
# -------------------------------------------------------------------------

def optimize_with_gepa(
    train_examples,
    val_examples,
    output_path="optimized_review_pipeline3.json"
):
    """
    Run GEPA optimization on the review pipeline.

    Args:
        train_examples: Pre-split training examples (list of dspy.Example)
        val_examples: Pre-split validation examples (list of dspy.Example)
        output_path: Where to save optimized pipeline

    Returns:
        Optimized ReviewPipeline module
    """
    print("=" * 60)
    print("GEPA OPTIMIZATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Validate inputs
    if len(train_examples) < 3:
        raise ValueError(f"Not enough training examples: {len(train_examples)}. Need at least 3.")

    print(f"Train set: {len(train_examples)} examples")
    print(f"  - Accept: {sum(1 for e in train_examples if 'accept' in e.decision.lower())}")
    print(f"  - Reject: {sum(1 for e in train_examples if 'reject' in e.decision.lower())}")
    print(f"Val set: {len(val_examples)} examples")
    print(f"  - Accept: {sum(1 for e in val_examples if 'accept' in e.decision.lower())}")
    print(f"  - Reject: {sum(1 for e in val_examples if 'reject' in e.decision.lower())}")
    print()

    # Initialize pipeline
    print("Initializing ReviewPipeline...")
    pipeline = ReviewPipeline()

    # Evaluate baseline (before optimization)
    print("\nEvaluating baseline (before optimization)...")
    baseline_scores = []
    for example in val_examples[:3]:
        try:
            pred = pipeline(article_text=example.article_text)
            metric_result = gepa_combined_metric(example, pred, None, None, None)
            metric_score = metric_result.score if hasattr(metric_result, 'score') else float(metric_result)
            baseline_scores.append(metric_score)
            print(f"  Baseline: {pred.decision} (human: {example.decision}) -> metric: {metric_score:.2f}")
        except Exception as e:
            print(f"  Baseline error: {e}")

    if baseline_scores:
        print(f"Baseline average metric: {sum(baseline_scores)/len(baseline_scores):.3f}")
    print()

    # Initialize GEPA optimizer
    print("Initializing GEPA optimizer...")
    print()

    # GEPA requires a reflection LM for analyzing failures and proposing better prompts
    reflection_lm = dspy.LM(
        model="openai/gpt-4.1-mini-fiit",  # Smaller, cheaper model for reflection
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        # api_base=API_BASE,
        
        max_tokens=4000,   # Reduced to save budget and prevent prompt bloat
        #cache=True,        # Cache repeated calls
    )
    print(f"  Reflection LM: openai/gpt-4.1-mini ")

    # ANTI-OVERFITTING: Reduce max_full_evals to prevent over-optimization
    # With only 8 val examples, 5-6 evals is enough
    optimizer = GEPA(
        metric=gepa_combined_metric,
        reflection_lm=reflection_lm,
        max_full_evals=5,  # Reduced to prevent overfitting on small val set
    )


    print("Running GEPA optimization (this may take a while)...")
    print("-" * 60)

    optimized_pipeline = optimizer.compile(
        pipeline,
        trainset=train_examples,
        valset=val_examples,

    )

    print("-" * 60)
    print("Optimization complete!")
    print()

    # Evaluate optimized pipeline
    print("Evaluating optimized pipeline...")
    optimized_scores = []
    for example in val_examples:
        try:
            pred = optimized_pipeline(article_text=example.article_text)
            metric_result = gepa_combined_metric(example, pred, None, None, None)
            # Handle ScoreWithFeedback
            metric_score = metric_result.score if hasattr(metric_result, 'score') else float(metric_result)
            optimized_scores.append(metric_score)
            print(f"  Optimized: {pred.decision} (human: {example.decision}) -> metric: {metric_score:.2f}")
        except Exception as e:
            print(f"  Evaluation error: {e}")

    if optimized_scores:
        avg_optimized = sum(optimized_scores) / len(optimized_scores)
        print(f"\nOptimized average metric: {avg_optimized:.3f}")

        if baseline_scores:
            avg_baseline = sum(baseline_scores) / len(baseline_scores)
            improvement = avg_optimized - avg_baseline
            print(f"Improvement: {improvement:+.3f} ({improvement/max(avg_baseline, 0.001)*100:+.1f}%)")

    # Save optimized pipeline
    print(f"\nSaving optimized pipeline to: {output_path}")
    optimized_pipeline.save(output_path)

    print()
    print("=" * 60)
    print("GEPA OPTIMIZATION COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return optimized_pipeline


def load_optimized_pipeline(path="optimized_review_pipeline3.json"):
    """
    Load a previously optimized pipeline.

    Usage:
        pipeline = load_optimized_pipeline()
        result = pipeline(article_text="...")
    """
    pipeline = ReviewPipeline()
    pipeline.load(path)
    return pipeline


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
            human_decision = "Accept" if human_avg_score >= 7.5 else "Reject"
        
        sections = dspy.ChainOfThought(SectionExtractor)



        response_sections = sections(article_text=article_text)

        print("\nExtracted Sections:")
        print(f"-Abstract: {response_sections.Abstract}\n")
        print(f"-Introduction: {response_sections.Introduction}\n")
        print(f"-Methods: {response_sections.Methods}\n")
        print(f"-Experiments: {response_sections.Experiments}\n")
        print(f"-Conclusion: {response_sections.Conclusion}\n")
        print("End of Sections\n")


        # chain of thought -
        ca_reviewer = dspy.Predict(CAReview)
        ca_review = ca_reviewer(
            introduction_section=response_sections.Introduction,
            conclusion_section=response_sections.Conclusion
        )
        
        methodology_reviewer = dspy.Predict(MethodologyReview)
        methodology_review = methodology_reviewer(
            methods_section=response_sections.Methods,
            experiments_section=response_sections.Experiments
        )
        
        clarity_reviewer = dspy.Predict(ClarityReview)
        clarity_review = clarity_reviewer(article_text=article_text)
        
        # Predict decision
        decision_predictor = dspy.ChainOfThought(DecisionPredictor)
        #prediction = decision_predictor(summary=final_summary.summary)
        ca_input_combined = f"Type: {ca_review.contribution_type}\nReview: {ca_review.review}"
        prediction = decision_predictor(ca_review=ca_input_combined,
                                       methodology_review=methodology_review.review,
                                       clarity_review=clarity_review.review)
        

        summary_generator = dspy.Predict(SummaryGenerator)
        final_summary = summary_generator(
            ca_review=ca_input_combined,
            methodology_review=methodology_review.review,
            clarity_review=clarity_review.review
        )
        
        
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
    corr, spearman_p = spearmanr(predicted_scores, human_scores)
    if np.isnan(corr):
        corr = None
        spearman_p = None
    
    
    accuracy = accuracy_score(human_decisions, predicted_decisions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        human_decisions, 
        predicted_decisions, 
        average='binary', 
        pos_label='Accept',
        zero_division=0
    )
    
    # Decision metrics (numeric)
    y_scores = [float(r["predicted_score"]) for r in results]
    y_true = [1 if str(d).strip().lower() == "accept" else 0 for d in human_decisions]
    y_pred = [1 if "accept" in str(d).strip().lower() else 0 for d in predicted_decisions]

    # Confusion matrix (fixed shape)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # MCC & balanced accuracy
    mcc = matthews_corrcoef(y_true, y_pred) if len(y_true) else None
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(y_true) else None

    # AUC-ROC & PR-AUC need both classes in y_true
    if len(set(y_true)) >= 2:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = None
        try:
            pr_auc = average_precision_score(y_true, y_scores)
        except Exception:
            pr_auc = None
    else:
        auc = None
        pr_auc = None
    
    # Text quality metrics
    avg_rouge_l = np.mean([r["rouge_l"] for r in results])
    avg_bertscore = np.mean([r["bertscore_f1"] for r in results])
    avg_llm_judge = np.mean([r["llm_judge_score"] for r in results])
    avg_length_diff = np.mean([r["length_diff"] for r in results])
    
    return {
        "n_articles": len(results),
        "spearman_corr": corr,
        "spearman_p": spearman_p,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
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

def evaluate_with_detailed_logging(pipeline, examples, name="Evaluation", show_feedback=False):
    """
    Evaluate with detailed per-example logging.

    Args:
        pipeline: The DSPy pipeline to evaluate
        examples: List of dspy.Example objects
        name: Name for the evaluation (displayed in header)
        show_feedback: If True, print GEPA feedback for each example
    """
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}\n")

    scores = []
    decision_matches = []
    score_errors = []
    predictions = []

    for i, example in enumerate(examples):
        try:
            pred = pipeline(article_text=example.article_text)
            predictions.append(pred)
            metric_result = gepa_combined_metric(example, pred, None, None, None)

            # Handle ScoreWithFeedback or plain float
            if hasattr(metric_result, 'score'):
                metric_score = metric_result.score
                feedback = metric_result.feedback if hasattr(metric_result, 'feedback') else ""
            else:
                metric_score = float(metric_result)
                feedback = ""

            decision_match = (pred.decision.lower().strip() ==
                            example.decision.lower().strip())
            score_error = abs(float(pred.score) - float(example.score))

            scores.append(metric_score)
            decision_matches.append(decision_match)
            score_errors.append(score_error)

            # Detailed logging
            print(f"Example {i+1}:")
            print(f"  Predicted: {pred.decision} (score: {pred.score:.2f})")
            print(f"  Human:     {example.decision} (score: {example.score:.2f})")
            print(f"  Match: {'✓' if decision_match else '✗'}, "
                  f"Score Error: {score_error:.2f}, "
                  f"Metric: {metric_score:.3f}")

            # Optionally show feedback
            if show_feedback and feedback:
                print(f"  --- Feedback ---")
                for line in feedback.split('\n'):
                    print(f"  {line}")
                print()

        except Exception as e:
            print(f"Example {i+1}: ERROR - {e}")
            scores.append(0.0)
            decision_matches.append(False)
            score_errors.append(10.0)
            predictions.append(None)

    # Summary statistics
    avg_metric = sum(scores) / len(scores) if scores else 0.0
    accuracy = sum(decision_matches) / len(decision_matches) if decision_matches else 0.0
    avg_score_error = sum(score_errors) / len(score_errors) if score_errors else 0.0

    print(f"\n{'-'*60}")
    print(f"Summary:")
    print(f"  Average Metric: {avg_metric:.3f}")
    print(f"  Decision Accuracy: {accuracy:.3f} ({sum(decision_matches)}/{len(decision_matches)})")
    print(f"  Average Score Error: {avg_score_error:.3f}")
    print(f"{'='*60}\n")

    return {
        'avg_metric': avg_metric,
        'accuracy': accuracy,
        'avg_score_error': avg_score_error,
        'individual_scores': scores,
        'predictions': predictions
    }

## main
if __name__ == "__main__":
   
    # ARTICLES_PATH = "C:\\Users\\katka\\BAKALARKA\\reconstructed_articles\\reconstructed_article_7.md"
    # REVIEWS_PATH = "C:\\Users\\katka\\BAKALARKA\\human_reviews\\7.json"
    # ARTICLES_FOLDER = "C:\\Users\\katka\\BAKALARKA\\ready_md"
    # REVIEWS_FOLDER = "C:\\Users\\katka\\BAKALARKA\\PeerRead\\data\\iclr_2017\\test\\reviews"
    # ARTICLES_FOLDER = "C:\\Users\\katka\\BAKALARKA\\reconstructed_articles"
    # REVIEWS_FOLDER = "C:\\Users\\katka\\BAKALARKA\\human_reviews"

    ARTICLES_FOLDER = "C:/Users/katka/BAKALARKA/new_ready_GEPA300_subset"
    REVIEWS_FOLDER = "C:/Users/katka/BAKALARKA/new_human_reviews_GEPA100"

    mode = sys.argv[1] if len(sys.argv) > 1 else "evaluate"

    def get_int_flag(flag: str):
        """Minimal flag parser: returns int after --flag, else None."""
        if flag not in sys.argv:
            return None
        try:
            i = sys.argv.index(flag)
            return int(sys.argv[i + 1])
        except Exception:
            return None

    def get_str_flag(flag: str):
        """Minimal flag parser: returns value after --flag, else None."""
        if flag not in sys.argv:
            return None
        try:
            i = sys.argv.index(flag)
            return str(sys.argv[i + 1])
        except Exception:
            return None
     
    if mode == "gepa":
        # Run GEPA optimization
        print("\n" + "="*60)
        print("MODE: GEPA OPTIMIZATION")
        print("="*60 + "\n")

        # Configuration for 88 articles
        TOTAL_ARTICLES = 88
        RANDOM_SEED = 42  # Fixný seed pre reprodukovateľnosť

        # Load all data ONCE
        print(f"Loading {TOTAL_ARTICLES} articles...")
        trainset = load_gepa_training_data(
            ARTICLES_FOLDER,
            REVIEWS_FOLDER,
            max_samples=TOTAL_ARTICLES
        )

        if len(trainset) < 12:  # Need at least 12 for meaningful 60/20/20 split
            raise ValueError(f"Need at least 12 examples, got {len(trainset)}")

        # Split data ONCE with fixed seed (60% train, 20% val, 20% test)
        print(f"\nSplitting data with seed={RANDOM_SEED}...")
        train_examples, val_examples, test_examples = stratified_split(
            trainset,
            train_ratio=0.68,
            val_ratio=0.16,
            test_ratio=0.16,
            seed=RANDOM_SEED
        )

        # Initialize baseline pipeline
        baseline_pipeline = ReviewPipeline()

        # Evaluate baseline on VALIDATION set
        print("\n" + "-"*60)
        baseline_results = evaluate_with_detailed_logging(
            baseline_pipeline,
            val_examples,
            name="BASELINE EVALUATION (Validation Set)"
        )

        # Run GEPA optimization using the SAME train/val split
        optimized_pipeline = optimize_with_gepa(
            train_examples=train_examples,
            val_examples=val_examples,
            output_path="optimized_review_pipeline3.json"
        )

        # Evaluate optimized on VALIDATION set
        optimized_results = evaluate_with_detailed_logging(
            optimized_pipeline,
            val_examples,
            name="OPTIMIZED EVALUATION (Validation Set)"
        )

        # Compare validation results
        print("\n" + "="*60)
        print("IMPROVEMENT SUMMARY (Validation Set)")
        print("="*60)
        print(f"Metric:   {baseline_results['avg_metric']:.3f} → {optimized_results['avg_metric']:.3f} "
              f"({optimized_results['avg_metric'] - baseline_results['avg_metric']:+.3f})")
        print(f"Accuracy: {baseline_results['accuracy']:.3f} → {optimized_results['accuracy']:.3f} "
              f"({optimized_results['accuracy'] - baseline_results['accuracy']:+.3f})")
        print("="*60 + "\n")

        # Final evaluation on TEST set (unseen data) - with feedback to see what went wrong
        print("\n" + "="*60)
        print("FINAL TEST SET EVALUATION (with detailed feedback)")
        print("="*60)
        test_results = evaluate_with_detailed_logging(
            optimized_pipeline,
            test_examples,
            name="OPTIMIZED EVALUATION (Test Set - Final)",
            show_feedback=True  # Show GEPA feedback for test set
        )

        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Validation Accuracy: {optimized_results['accuracy']:.3f}")
        print(f"Test Accuracy:       {test_results['accuracy']:.3f}")
        print(f"Validation Metric:   {optimized_results['avg_metric']:.3f}")
        print(f"Test Metric:         {test_results['avg_metric']:.3f}")
        print("="*60 + "\n")

        # =========================================================
        # COMPREHENSIVE METRICS EVALUATION (reuse predictions from above)
        # =========================================================
        print("\n" + "="*60)
        print("COMPREHENSIVE METRICS (Test Set)")
        print("="*60)

        # Reuse predictions from evaluate_with_detailed_logging (no duplicate API calls)
        test_preds = []
        reviews_path = Path(REVIEWS_FOLDER)

        for i, (example, pred) in enumerate(zip(test_examples, test_results['predictions'])):
            if pred is None:
                continue
            try:
                generated_summary = pred.summary if hasattr(pred, 'summary') else ""

                # Load human review text for comparison
                human_review_text = ""
                if hasattr(example, 'article_id'):
                    review_file = reviews_path / f"{example.article_id}.json"
                    if review_file.exists():
                        human_review_text = load_human_review(str(review_file))

                test_preds.append({
                    "pred_score": float(pred.score),
                    "pred_decision": pred.decision,
                    "human_score": float(example.score),
                    "human_decision": example.decision,
                    "generated_summary": generated_summary,
                    "human_review": human_review_text,
                })
            except Exception as e:
                print(f"  [{i+1}/{len(test_examples)}] Error building metrics: {e}")
                continue

        if test_preds:
            pred_scores = [p["pred_score"] for p in test_preds]
            human_scores = [p["human_score"] for p in test_preds]
            pred_decisions = [p["pred_decision"] for p in test_preds]
            human_decisions = [p["human_decision"] for p in test_preds]

            print("\n--- Decision Metrics ---")

            # Spearman correlation
            corr, p_val = spearmanr(pred_scores, human_scores)
            print(f"Spearman Correlation: {corr:.4f} (p={p_val:.4f})")

            # Decision metrics
            y_true = [1 if "accept" in h.lower() else 0 for h in human_decisions]
            y_pred = [1 if "accept" in p.lower() else 0 for p in pred_decisions]

            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            print(f"Accuracy:            {accuracy:.4f}")
            print(f"Precision:           {precision:.4f}")
            print(f"Recall:              {recall:.4f}")
            print(f"F1 Score:            {f1:.4f}")

            # AUC-ROC
            try:
                auc = roc_auc_score(y_true, pred_scores)
                print(f"AUC-ROC:             {auc:.4f}")
            except:
                print(f"AUC-ROC:             N/A (need both classes)")

            # Score MAE
            mae = np.mean([abs(p - h) for p, h in zip(pred_scores, human_scores)])
            print(f"Score MAE:           {mae:.4f}")

            # --- Text Quality Metrics ---
            print("\n--- Text Quality Metrics ---")

            rouge_scores = []
            bert_scores = []
            length_diffs = []

            for p in test_preds:
                gen_text = p["generated_summary"]
                human_text = p["human_review"]

                if gen_text and human_text:
                    try:
                        rouge_scores.append(calculate_rouge_l(gen_text, human_text))
                    except:
                        pass
                    try:
                        bert_scores.append(calculate_bertscore(gen_text, human_text))
                    except:
                        pass
                    try:
                        length_diffs.append(calculate_length_difference(gen_text, human_text))
                    except:
                        pass

            if rouge_scores:
                print(f"Avg ROUGE-L:         {np.mean(rouge_scores):.4f}")
            else:
                print(f"Avg ROUGE-L:         N/A (no summaries)")

            if bert_scores:
                print(f"Avg BERTScore F1:    {np.mean(bert_scores):.4f}")
            else:
                print(f"Avg BERTScore F1:    N/A (no summaries)")

            if length_diffs:
                print(f"Avg Length Diff:     {np.mean(length_diffs):.1f} words")
            else:
                print(f"Avg Length Diff:     N/A (no summaries)")

        print("="*60 + "\n")

    elif mode == "baseline":
        # =================================================================
        # BASELINE MODE: Run unoptimized pipeline on independent test set
        # =================================================================
        TEST_ARTICLES = "C:/Users/katka/BAKALARKA/new_ready_GEPA100md"
        TEST_REVIEWS = "C:/Users/katka/BAKALARKA/new_human_reviews_GEPA100"

        # Allow overriding dirs from CLI (useful to run on RLM markdown folders)
        # Example:
        #   uv run python experiment/review_process2.py baseline --articles-dir test_RLMmd --reviews-dir test_reviews --limit 10
        override_articles = get_str_flag("--articles-dir")
        override_reviews = get_str_flag("--reviews-dir")
        if override_articles:
            TEST_ARTICLES = override_articles
        if override_reviews:
            TEST_REVIEWS = override_reviews

        print("\n" + "="*60)
        print("MODE: BASELINE (Unoptimized Pipeline on Independent Test Set)")
        print("="*60)
        print(f"Test articles: {TEST_ARTICLES}")
        print(f"Test reviews:  {TEST_REVIEWS}")

        print("\nInitializing baseline pipeline...")
        pipeline = ReviewPipeline()
        print("Baseline pipeline ready.")

        test_articles_path = Path(TEST_ARTICLES)
        test_reviews_path = Path(TEST_REVIEWS)
        all_article_files = sorted(test_articles_path.glob("reconstructed_article_temp*.md"))
        limit = get_int_flag("--limit")
        article_files = all_article_files[:limit] if limit else all_article_files
        if limit:
            print(f"Found {len(all_article_files)} test articles (using first {len(article_files)}).\n")
        else:
            print(f"Found {len(article_files)} test articles.\n")

        results = []
        for idx, article_file in enumerate(article_files, 1):
            article_id = article_file.stem.replace("reconstructed_article_temp_", "")
            review_file = test_reviews_path / f"{article_id}.json"

            if not review_file.exists():
                print(f"  [{idx}/{len(article_files)}] Skipping {article_id} - no review found")
                continue

            print(f"  [{idx}/{len(article_files)}] Processing article {article_id}...")

            try:
                article_text = load_article(str(article_file))
                with open(review_file, "r", encoding="utf-8") as f:
                    review_data = json.load(f)

                human_reviews = get_valid_reviews(review_data)
                if not human_reviews:
                    print(f"    No valid reviews, skipping.")
                    continue

                human_avg_score = review_data.get("average_score", 0)
                human_decision = review_data.get("decision", "Unknown")

                if human_avg_score == 0 or human_decision == "Unknown":
                    scores_list = []
                    for r in human_reviews:
                        try:
                            scores_list.append(float(r["RECOMMENDATION"]))
                        except (ValueError, TypeError):
                            continue
                    if not scores_list:
                        continue
                    human_avg_score = sum(scores_list) / len(scores_list)
                    human_decision = "Accept" if human_avg_score >= 7.5 else "Reject"

                pred = pipeline(article_text=article_text)

                per_reviewer_metrics = []
                for human_review in human_reviews:
                    human_comments = human_review.get("comments", "")

                    judge = dspy.Predict(LLMJudge)
                    llm_eval = judge(
                        generated_review=pred.summary,
                        human_review=human_comments
                    )

                    rouge_l = calculate_rouge_l(pred.summary, human_comments)
                    bertscore_f1 = calculate_bertscore(pred.summary, human_comments)
                    length_diff = calculate_length_difference(pred.summary, human_comments)

                    per_reviewer_metrics.append({
                        "llm_judge_score": float(llm_eval.evaluation),
                        "rouge_l": rouge_l,
                        "bertscore_f1": bertscore_f1,
                        "length_diff": length_diff
                    })

                result = {
                    "article_id": article_id,
                    "success": True,
                    "predicted_score": float(pred.score),
                    "predicted_decision": pred.decision,
                    "human_score": human_avg_score,
                    "human_decision": human_decision,
                    "num_reviewers": len(human_reviews),
                    "llm_judge_score": np.mean([m["llm_judge_score"] for m in per_reviewer_metrics]),
                    "rouge_l": np.mean([m["rouge_l"] for m in per_reviewer_metrics]),
                    "bertscore_f1": np.mean([m["bertscore_f1"] for m in per_reviewer_metrics]),
                    "length_diff": np.mean([m["length_diff"] for m in per_reviewer_metrics]),
                    "generated_review": pred.summary,
                    "final_justification": pred.justification,
                    "per_reviewer_metrics": per_reviewer_metrics
                }
                results.append(result)

                match = "OK" if pred.decision.strip().lower() == human_decision.strip().lower() else "MISS"
                print(f"    {match} | Pred: {pred.decision} ({float(pred.score):.2f}) | Human: {human_decision} ({human_avg_score:.2f})")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        if results:
            metrics = calculate_aggregate_metrics(results)

            output_file = "evaluation_results_baseline.json"
            output_data = {
                "pipeline": "baseline",
                "test_articles_dir": TEST_ARTICLES,
                "n_articles": len(results),
                "individual_results": results,
                "aggregate_metrics": metrics
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"\n{'='*60}")
            print(f"AGGREGATE METRICS (n={len(results)} articles)")
            print(f"{'='*60}")

            print(f"\n--- Decision Metrics ---")
            if metrics.get('spearman_corr') is None:
                print("Spearman Correlation: N/A")
            else:
                print(f"Spearman Correlation: {metrics['spearman_corr']:.4f} (p={metrics['spearman_p']:.4f})")
            print(f"Accuracy:            {metrics['accuracy']:.4f}")
            print(f"Precision:           {metrics['precision']:.4f}")
            print(f"Recall:              {metrics['recall']:.4f}")
            print(f"F1 Score:            {metrics['f1']:.4f}")
            print(f"MCC:                 {metrics['mcc']:.4f}")
            print(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
            if metrics['auc'] is not None:
                print(f"AUC-ROC:             {metrics['auc']:.4f}")
            if metrics.get('pr_auc') is not None:
                print(f"PR-AUC:              {metrics['pr_auc']:.4f}")

            mae = np.mean([abs(r["predicted_score"] - r["human_score"]) for r in results])
            print(f"Score MAE:           {mae:.4f}")

            print(f"\n--- Text Quality Metrics ---")
            print(f"Avg ROUGE-L:         {metrics['avg_rouge_l']:.4f}")
            print(f"Avg BERTScore F1:    {metrics['avg_bertscore']:.4f}")
            print(f"Avg LLM Judge Score: {metrics['avg_llm_judge']:.4f}")
            print(f"Avg Length Diff:     {metrics['avg_length_diff']:.1f} words")

            print(f"\nResults saved to: {output_file}")

        else:
            print("No results to aggregate!")

    else:
        # =================================================================
        # EVALUATE MODE: Load optimized pipeline, test on independent set
        # =================================================================
        # Cesty k NEZÁVISLÉMU test setu (nesmie sa prekrývať s GEPA dátami!)
        TEST_ARTICLES = "C:/Users/katka/BAKALARKA/test_articles"
        TEST_REVIEWS = "C:/Users/katka/BAKALARKA/test_reviews"
        PIPELINE_PATH = "optimized_review_pipeline4.json"

        # Allow overriding dirs from CLI (useful to run on RLM markdown folders)
        # Example:
        #   uv run python experiment/review_process2.py evaluate --articles-dir test_RLMmd --reviews-dir test_reviews --limit 10
        override_articles = get_str_flag("--articles-dir")
        override_reviews = get_str_flag("--reviews-dir")
        if override_articles:
            TEST_ARTICLES = override_articles
        if override_reviews:
            TEST_REVIEWS = override_reviews

        print("\n" + "="*60)
        print("MODE: EVALUATE (Optimized Pipeline on Independent Test Set)")
        print("="*60)
        print(f"Pipeline:      {PIPELINE_PATH}")
        print(f"Test articles: {TEST_ARTICLES}")
        print(f"Test reviews:  {TEST_REVIEWS}")

        # 1. Načítaj optimalizovaný pipeline
        print("\nLoading optimized pipeline...")
        pipeline = load_optimized_pipeline(PIPELINE_PATH)
        print("Pipeline loaded successfully.")

        # 2. Nájdi všetky test články
        test_articles_path = Path(TEST_ARTICLES)
        test_reviews_path = Path(TEST_REVIEWS)
        all_article_files = sorted(test_articles_path.glob("reconstructed_article_temp*.md"))
        limit = get_int_flag("--limit")
        article_files = all_article_files[:limit] if limit else all_article_files
        if limit:
            print(f"Found {len(all_article_files)} test articles (using first {len(article_files)}).\n")
        else:
            print(f"Found {len(article_files)} test articles.\n")

        # 3. Spracuj každý článok cez optimalizovaný pipeline
        results = []
        for idx, article_file in enumerate(article_files, 1):
            article_id = article_file.stem.replace("reconstructed_article_temp_", "")
            review_file = test_reviews_path / f"{article_id}.json"

            if not review_file.exists():
                print(f"  [{idx}/{len(article_files)}] Skipping {article_id} - no review found")
                continue

            print(f"  [{idx}/{len(article_files)}] Processing article {article_id}...")

            try:
                # Načítaj článok a review dáta
                article_text = load_article(str(article_file))
                with open(review_file, "r", encoding="utf-8") as f:
                    review_data = json.load(f)

                human_reviews = get_valid_reviews(review_data)
                if not human_reviews:
                    print(f"    No valid reviews, skipping.")
                    continue

                human_avg_score = review_data.get("average_score", 0)
                human_decision = review_data.get("decision", "Unknown")

                if human_avg_score == 0 or human_decision == "Unknown":
                    scores_list = []
                    for r in human_reviews:
                        try:
                            scores_list.append(float(r["RECOMMENDATION"]))
                        except (ValueError, TypeError):
                            continue
                    if not scores_list:
                        continue
                    human_avg_score = sum(scores_list) / len(scores_list)
                    human_decision = "Accept" if human_avg_score >= 7.5 else "Reject"

                # Spusti optimalizovaný pipeline
                pred = pipeline(article_text=article_text)

                # Porovnaj s každým ľudským recenzentom
                per_reviewer_metrics = []
                for human_review in human_reviews:
                    human_comments = human_review.get("comments", "")

                    # LLM Judge
                    judge = dspy.Predict(LLMJudge)
                    llm_eval = judge(
                        generated_review=pred.summary,
                        human_review=human_comments
                    )

                    rouge_l = calculate_rouge_l(pred.summary, human_comments)
                    bertscore_f1 = calculate_bertscore(pred.summary, human_comments)
                    length_diff = calculate_length_difference(pred.summary, human_comments)

                    per_reviewer_metrics.append({
                        "llm_judge_score": float(llm_eval.evaluation),
                        "rouge_l": rouge_l,
                        "bertscore_f1": bertscore_f1,
                        "length_diff": length_diff
                    })

                result = {
                    "article_id": article_id,
                    "success": True,
                    "predicted_score": float(pred.score),
                    "predicted_decision": pred.decision,
                    "human_score": human_avg_score,
                    "human_decision": human_decision,
                    "num_reviewers": len(human_reviews),
                    "llm_judge_score": np.mean([m["llm_judge_score"] for m in per_reviewer_metrics]),
                    "rouge_l": np.mean([m["rouge_l"] for m in per_reviewer_metrics]),
                    "bertscore_f1": np.mean([m["bertscore_f1"] for m in per_reviewer_metrics]),
                    "length_diff": np.mean([m["length_diff"] for m in per_reviewer_metrics]),
                    "generated_review": pred.summary,
                    "final_justification": pred.justification,
                    "per_reviewer_metrics": per_reviewer_metrics
                }
                results.append(result)

                match = "OK" if pred.decision.strip().lower() == human_decision.strip().lower() else "MISS"
                print(f"    {match} | Pred: {pred.decision} ({float(pred.score):.2f}) | Human: {human_decision} ({human_avg_score:.2f})")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        # 4. Agregované metriky
        if results:
            metrics = calculate_aggregate_metrics(results)

            # Uloženie výsledkov do JSON
            output_file = "evaluation_results_optimized4.json"
            output_data = {
                "pipeline": PIPELINE_PATH,
                "test_articles_dir": TEST_ARTICLES,
                "n_articles": len(results),
                "individual_results": results,
                "aggregate_metrics": metrics
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            # Výpis metrík
            print(f"\n{'='*60}")
            print(f"AGGREGATE METRICS (n={len(results)} articles)")
            print(f"{'='*60}")

            print(f"\n--- Decision Metrics ---")
            if metrics.get('spearman_corr') is None:
                print("Spearman Correlation: N/A")
            else:
                print(f"Spearman Correlation: {metrics['spearman_corr']:.4f} (p={metrics['spearman_p']:.4f})")
            print(f"Accuracy:            {metrics['accuracy']:.4f}")
            print(f"Precision:           {metrics['precision']:.4f}")
            print(f"Recall:              {metrics['recall']:.4f}")
            print(f"F1 Score:            {metrics['f1']:.4f}")
            print(f"MCC:                 {metrics['mcc']:.4f}")
            print(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
            if metrics['auc'] is not None:
                print(f"AUC-ROC:             {metrics['auc']:.4f}")
            if metrics.get('pr_auc') is not None:
                print(f"PR-AUC:              {metrics['pr_auc']:.4f}")

            mae = np.mean([abs(r["predicted_score"] - r["human_score"]) for r in results])
            print(f"Score MAE:           {mae:.4f}")

            print(f"\n--- Text Quality Metrics ---")
            print(f"Avg ROUGE-L:         {metrics['avg_rouge_l']:.4f}")
            print(f"Avg BERTScore F1:    {metrics['avg_bertscore']:.4f}")
            print(f"Avg LLM Judge Score: {metrics['avg_llm_judge']:.4f}")
            print(f"Avg Length Diff:     {metrics['avg_length_diff']:.1f} words")

            print(f"\nResults saved to: {output_file}")

            # Ukážky vygenerovaných recenzií
            print(f"\n{'='*60}")
            print("GENERATED REVIEWS SAMPLES")
            print(f"{'='*60}")
            for idx, result in enumerate(results, 1):
                print(f"\n{'─'*60}")
                print(f"ARTICLE {idx}: {result['article_id']}")
                print(f"{'─'*60}")
                print(f"Predicted: {result['predicted_decision']} (Score: {result['predicted_score']:.2f})")
                print(f"Human:     {result['human_decision']} (Score: {result['human_score']:.2f})")
                print(f"\n--- Generated Review ---")
                print(result['generated_review'])

            print(f"\n{'='*60}")

            # =============================================================
            # USAGE & COST REPORT (pre článok o udržateľnosti)
            # =============================================================
            print(f"\n{'='*60}")
            print("USAGE & COST REPORT")
            print(f"{'='*60}")

            total_prompt = 0
            total_completion = 0

            PRICE_INPUT = 0.15 / 1_000_000   # Cena za 1 input token
            PRICE_OUTPUT = 0.60 / 1_000_000  # Cena za 1 output token

            if hasattr(llm2, 'history') and llm2.history:
                for interaction in llm2.history:
                    usage = interaction.get('usage') or {}
                    if not usage and 'response' in interaction:
                        resp = interaction['response']
                        if hasattr(resp, 'usage'):
                            if isinstance(resp, dict):
                                usage_data = resp.get('usage', {})
                                p = usage_data.get('prompt_tokens', 0)
                                c = usage_data.get('completion_tokens', 0)
                            elif hasattr(resp, 'usage'):
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

            log_usage_to_csv(total_prompt, total_completion, total_cost, len(results))

            print(f"Total Prompt Tokens:     {total_prompt}")
            print(f"Total Completion Tokens: {total_completion}")
            print(f"Total All Tokens:        {total_prompt + total_completion}")
            print(f"{'-'*30}")
            print(f"ESTIMATED COST:          ${total_cost:.5f}")
            print(f"Usage logged to usage_log.csv")
            print(f"{'='*60}")




