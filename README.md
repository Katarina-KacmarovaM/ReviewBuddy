# ReviewBuddy

AI-powered academic paper review system. Generates structured peer reviews for PDF papers using DSPy and LLM-based section extraction.

## Installation

```bash
git clone https://github.com/Katarina-KacmarovaM/ReviewBuddy.git
cd ReviewBuddy
uv sync
```

## Configuration

Create a `.env` file in the root directory:

```
API_KEY=your-api-key
API_BASE=https://your-api-base/
```

## Usage

### Review a single paper

```bash
uv run python run_flat.py --review path/to/paper.pdf --pageindex
```

### Train reviewer slots (requires training data)

```bash
uv run python run_flat.py --gepa --pageindex
```

Training data (`data/reviews/train/`, `data/pdfs/train/`) must be placed in the repo root before running.

### Evaluate on test set (requires trained reviewers)

```bash
uv run python run_flat.py --eval --pageindex
```

### Run unoptimized baseline

```bash
uv run python run_flat.py --baseline --pageindex
```

## Trained Reviewers

Optimized reviewer slots are saved to `flat_optimized/` after `--gepa`. The `--review` and `--eval` modes load them automatically.
