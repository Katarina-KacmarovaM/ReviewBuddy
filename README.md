# Agentic Academic Paper Review System

Modular agentic system for automated academic paper review generation using DSPy. Built for research evaluation on the ICLR dataset.

## Architecture

```
PDF → LocalPDFSectionExtractor (pageindex-local, runs via your LLM endpoint)
    ↓
Section Extraction (IdentifySections → tree-based, or SectionExtractor → markdown)
    ↓
Specialist Agents: ContributionReview, MethodologyReview, ClarityReview, CrossSectionAnalysis, PaperSummary
    ↓
ReviewSynthesizer → DecisionPredictor (GEPA-optimized)
    ↓
Evaluation: ROUGE-L, BERTScore, Spearman, Accuracy, AUC-ROC, LLM-as-Judge
```

## Key Files

| File | Description |
|---|---|
| `run.py` | Main entry point — GEPA optimization + evaluation |
| `quick_test.py` | Fast pipeline validation without GEPA |
| `preview.py` | Run pipeline on a single article and print output |
| `reviewbuddy/prompts.py` | DSPy Signature definitions (IdentifySections, SectionExtractor, review agents) |
| `reviewbuddy/extraction.py` | Section extractors: `LocalPDFSectionExtractor` (pageindex-local), `RLMSectionExtractor` (markdown) |
| `reviewbuddy/pipeline.py` | `ReviewPipeline` orchestration |
| `reviewbuddy/reviewer.py` | Public `Reviewer` API class |
| `experiments/data_utils.py` | Data loading for train/test sets |
| `experiments/metrics.py` | Evaluation metrics + GEPA metric function |
| `experiments/evaluation.py` | `evaluate()` — per-example evaluation with logging |
| `experiments/optimizer.py` | `optimize_with_gepa()` + precompute helpers |
| `experiments/cost_tracker.py` | LM usage tracking and cost estimation |

## Setup

```bash
git clone https://github.com/Katarina-KacmarovaM/Bachelor-s-Thesis.git
cd Bachelor-s-Thesis

# Install with uv (recommended)
uv sync --extra experiments
```

> **Note:** `torch` is part of the experiment dependencies. For GPU support, replace it manually after install: see [pytorch.org](https://pytorch.org/get-started/locally/).

Create a `.env` file in the root directory:
```
API_KEY=...
API_BASE=...
MODEL=...
```

## Data

Test PDFs are stored on Google Drive (too large for git):

📁 [Download test_pdfs/](https://drive.google.com/drive/folders/1-akPfjyLX39rPlchSn7gKyHfc7PIVBZI?usp=sharing)

After downloading, place the folder in the repo root:
```
BAKALARKA/
├── test_pdfs/       ← place here
├── test_reviews/    ← already in repo
```

> **Note:** Training data (`train_pdfs/`, `train_reviews/`) is not included. GEPA optimization requires training data and cannot be run without it.

## Using the Package (`reviewbuddy`)

The `reviewbuddy` package exposes a single public class: `Reviewer`.

```python
from reviewbuddy import Reviewer

agent = Reviewer(
    api_key="your-api-key",
    api_base="https://your-api-base/",
    use_local_pdf=True,   # uses pageindex-local via your LLM endpoint
)
review = agent.review(pdf_path="paper.pdf")

print(review.decision)               # "Accept" or "Reject"
print(review.score)                  # 1–10
print(review.comments)
print(review.strengths)
print(review.weaknesses)
print(review.clarification_questions)
```

To use a pre-optimized pipeline saved after GEPA training:
```python
agent = Reviewer(api_key="...", api_base="...", use_local_pdf=True).load("optimized_review_pipeline.json")
review = agent.review(pdf_path="paper.pdf")
```

## Quick Start (Evaluation Only)

**1. Preview the pipeline on a single article:**
```bash
python preview.py --pageindex
```

**2. Quick validation on multiple articles (no GEPA):**
```bash
python quick_test.py --pageindex
```

**3. Full evaluation on the test set:**
```bash
python run.py --pageindex
```
If `optimized_review_pipeline.json` is present, it will be loaded automatically.

## GEPA Optimization (requires training data)

Training data must be placed in `train_pdfs/` and `train_reviews/`:

```bash
# Full training set
python run.py --gepa --pageindex

# Smaller subset (faster)
python run.py --gepa --pageindex --subset
```

The optimized pipeline is saved to `optimized_review_pipeline.json` and used automatically by evaluate mode.

## Experiments

Evaluation and optimization utilities are in `experiments/`:
- `data_utils.py` — data loading and train/val/test splitting
- `metrics.py` — ROUGE-L, BERTScore, Spearman, LLM-as-Judge
- `evaluation.py` — pipeline evaluation with per-example logging
- `optimizer.py` — GEPA optimization loop
- `cost_tracker.py` — token usage and cost estimation
