"""
Preview - výstup pipeline na jeden článok.

Použitie:
    python preview.py              # vezme prvý dostupný článok
    python preview.py --pageindex  # PageIndex mód (potrebuje PDF)
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signatures import ReviewPipeline
from data_utils import load_article

load_dotenv()

ARTICLES_FOLDER = "C:/Users/katka/BAKALARKA/new_ready_GEPA100md"
PDFS_FOLDER     = "C:/Users/katka/BAKALARKA/test_pdfs"

use_pageindex = "--pageindex" in sys.argv

if use_pageindex:
    api_key  = os.getenv("MY_PAGEINDEX_API_KEY")
    pdf_path = next(Path(PDFS_FOLDER).glob("*.pdf"), None)
    if not pdf_path:
        print("Žiadne PDF súbory nenájdené v", PDFS_FOLDER)
        sys.exit(1)
    print(f"Používam PDF: {pdf_path.name}\n")
    pipe = ReviewPipeline(use_pageindex=True, pageindex_api_key=api_key)
    pred = pipe(pdf_path=str(pdf_path))
else:
    article_path = next(Path(ARTICLES_FOLDER).glob("reconstructed_article_temp_*.md"), None)
    if not article_path:
        print("Žiadne článkové súbory nenájdené v", ARTICLES_FOLDER)
        sys.exit(1)
    print(f"Používam článok: {article_path.name}\n")
    article_text = load_article(str(article_path))
    pipe = ReviewPipeline()
    pred = pipe(article_text=article_text)

print(pred.comments)
print(f"\nStrengths:\n{pred.strengths}")
print(f"\nWeaknesses:\n{pred.weaknesses}")
print(f"\nClarification Questions:\n{pred.clarification_questions}")
print(f"\nRating: {pred.recommendation}/10  |  Decision: {pred.decision}  |  Soundness: {pred.soundness}/4  |  Clarity: {pred.clarity}/4")
