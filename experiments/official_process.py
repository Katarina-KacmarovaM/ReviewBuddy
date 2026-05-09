"""
AgentReviewer — Official Review Pipeline
=========================================
Multi-agent review system with:
- Rubric as explicit input
- Output structure matching ICLR human review format
- Separated strengths, weaknesses, sub-scores
- ReviewSynthesizer for final structured output
"""

import dspy
import os
from dotenv import load_dotenv
from bert_score import score
from evaluate import load
import json
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from pathlib import Path
import numpy as np
import csv
from datetime import datetime
import sys
import re
import shutil
import tempfile
import time

# GEPA
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

load_dotenv()
TF_ENABLE_ONEDNN_OPTS = 0
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    from pageindex import PageIndexClient
    import pageindex.utils as pageindex_utils
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PAGEINDEX_AVAILABLE = False

# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------

# Smaller, faster model for structured extraction and focused tasks
fast_model = dspy.LM(
    model="openai/gpt-4.1-nano-fiit",
    api_key=API_KEY,
    api_base=API_BASE,
    temperature=0.1,  # CHANGE: was 0.3 → 0.1 (extraction needs determinism)
    max_tokens=8000,
    cache=False
)

# Larger, more powerful model for analysis, synthesis, and critical reasoning
powerful_model = dspy.LM(
    model="openai/gpt-4.1-mini-fiit",
    api_key=API_KEY,
    api_base=API_BASE,
    temperature=0.0,  
    max_tokens=8000,
    cache=False
)

# Configure default LM for the pipeline
dspy.configure(lm=powerful_model, track_usage=True)


# -------------------------------------------------------------------------
# Rubric Definition
# -------------------------------------------------------------------------

def load_rubric(path="rubric_iclr.txt"):
    """Load review rubric from external text file."""
    rubric_path = Path(path)
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path.resolve()}")
    return rubric_path.read_text(encoding="utf-8")

ICLR_RUBRIC = load_rubric()

# -------------------------------------------------------------------------
# DSPy Signatures
# -------------------------------------------------------------------------

class SectionExtractor(dspy.Signature):
    """Extract sections from a scientific article in markdown. Do not summarize."""

    article_text: str = dspy.InputField(description="Full article text in markdown.")
    Abstract: str = dspy.OutputField(description="Abstract section content.")
    Introduction: str = dspy.OutputField(description="Introduction section content.")
    Related_Work: str = dspy.OutputField(description="Related Work / Background section content.")
    Methods: str = dspy.OutputField(description="Methods / Methodology section content.")
    Experiments: str = dspy.OutputField(description="Experiments / Results section content.")
    Conclusion: str = dspy.OutputField(description="Conclusion / Discussion section content.")


class CAReview(dspy.Signature):
    """
    Evaluate the contribution and novelty of the article.
    Before listing strengths, ask: Could this result be explained by a simpler baseline or by data leakage?
    For every weakness, cite the specific section where the problem occurs.
    """
    introduction_section: str = dspy.InputField(description="Introduction text with motivation and problem statement.")
    related_work_section: str = dspy.InputField(description="Related Work text for assessing novelty context.")
    conclusion_section: str = dspy.InputField(description="Conclusion text with authors' contribution summary.")
    contribution_type: str = dspy.OutputField(description="Classify as: 'Incremental', 'Significant', or 'Breakthrough'.")
    novelty_review: str = dspy.OutputField(description="Detailed review of novelty, significance, and motivation with strengths, weaknesses, and section-grounded evidence.")


class MethodologyReview(dspy.Signature):
    """
    Evaluate the soundness of methodology and experiments.
    A single missing key baseline is sufficient for rejection. State which baseline is missing and why it is critical.
    For every weakness, name the exact missing baseline or dataset and state which claim cannot be trusted without it.
    """
    methods_section: str = dspy.InputField(description="Methods section text with technical contribution.")
    experiments_section: str = dspy.InputField(description="Experiments section text with empirical evaluation.")
    technical_review: str = dspy.OutputField(description="Detailed review of methodology soundness and experimental rigor with strengths, weaknesses, and section-grounded evidence.")


class ClarityReview(dspy.Signature):
    """
    Evaluate the clarity and presentation quality of the article.
    Critical issues: main method unclear, key equations unexplained, results without context.
    Minor issues: typos, formatting, missing references.
    Quote the exact sentence or equation that is unclear.
    """
    article_text: str = dspy.InputField(description="Full article text.")
    clarity_review: str = dspy.OutputField(description="Detailed review of clarity and presentation with strengths, weaknesses, and section-grounded evidence.")


class ConflictResolver(dspy.Signature):
    """
    Identifies and resolves conflicts between two different review assessments.

    Your task is to analyze the Contribution/Novelty review and the Methodology/Experiments review.
    If they present conflicting judgments (e.g., one praises novelty while the other criticizes weak experiments),
    you must create a balanced, synthesized viewpoint.

    If there is no conflict, simply state that the reviews are consistent.
    """
    ca_review: str = dspy.InputField(description="The review assessing contribution and novelty.")
    methodology_review: str = dspy.InputField(description="The review assessing methodology and experiments.")
    resolved_critique: str = dspy.OutputField(description="A brief, synthesized critique that resolves any conflicts, or a statement of consistency.")


class ReviewSynthesizer(dspy.Signature):
    """
    Synthesize specialist reviews into a final structured review.
    Identify consensus and conflicts. Weight weaknesses heavily.
    A single major methodological flaw is a dealbreaker.
    Good clarity cannot compensate for weak methodology.
    """
    ca_review: str = dspy.InputField(description="Contribution assessment review with strengths and weaknesses.")
    methodology_review: str = dspy.InputField(description="Methodology review with strengths and weaknesses.")
    clarity_review: str = dspy.InputField(description="Clarity review with strengths and weaknesses.")
    rubric: str = dspy.InputField(description="Review criteria and scoring guidelines.")

    comments: str = dspy.OutputField(description="Summary of the paper and overall assessment (2-4 sentences).")
    strengths: str = dspy.OutputField(description="Specific strengths with bullet points. Reference concrete sections or results.")
    weaknesses: str = dspy.OutputField(description="Specific weaknesses with bullet points. Reference concrete gaps or flaws.")
    clarification_questions: str = dspy.OutputField(description="Clarification questions for the authors with bullet points.")
    recommendation: float = dspy.OutputField(description="Score from 1.0 to 10.0 (use decimals, e.g. 6.5). Use full range: 9-10 breakthrough, 7-8 accept, 5-6 borderline, 3-4 reject, 1-2 strong reject.")
    soundness: int = dspy.OutputField(description="Soundness score 1-4.")
    clarity: int = dspy.OutputField(description="Clarity score 1-4.")
    decision: str = dspy.OutputField(description="'Accept' or 'Reject'.")


class LLMJudge(dspy.Signature):
    """You are an expert strict reviewer. Compare the generated review with the human review.
    Rate the accuracy and significance on a scale from 1 to 10.

    Be objective in scoring.
    """
    generated_review: str = dspy.InputField(description="Generated review text by the LLM.")
    human_review: str = dspy.InputField(description="Human review text to compare with.")


class IdentifySections(dspy.Signature):
    """
    Map nodes of a scientific paper tree to standard review sections.
    Output JSON with exactly these 6 keys: abstract, introduction, related_work, methods, experiments, conclusion.
    Each value is a list of node_ids. Use [] if section is not found.
    Alternative names — introduction: background/motivation/overview/preliminaries;
    related_work: literature review/prior work/background and related work;
    methods: methodology/approach/framework/architecture/proposed method;
    experiments: evaluation/results/benchmarks/empirical study;
    conclusion: discussion/limitations/future work/concluding remarks.
    Each node_id must appear in at most one category.
    Do NOT map References or Bibliography sections to related_work — skip them entirely.
    """
    tree_structure = dspy.InputField(desc="Simplified JSON tree with id and title fields.")
    section_mapping = dspy.OutputField(desc='JSON with keys: abstract, introduction, related_work, methods, experiments, conclusion. Values are lists of node_ids.')



class PageIndexSectionExtractor(dspy.Module):
    """
    Extract sections from PDF using PageIndex tree structure.
    Maps PageIndex lowercase keys to standard pipeline keys.
    """
    DOC_ID_CACHE_FILE = "pageindex_doc_ids.json"
    KEY_MAP = {
        "abstract": "Abstract", "introduction": "Introduction",
        "related_work": "Related_Work", "methods": "Methods",
        "experiments": "Experiments", "conclusion": "Conclusion"
    }

    def __init__(self, api_key):
        super().__init__()
        if not PAGEINDEX_AVAILABLE:
            raise ImportError("pageindex package is not installed. Run: pip install pageindex")
        self.pi_client = PageIndexClient(api_key=api_key)
        self.doc_cache = {}
        self._persistent_ids = self._load_persistent_ids()
        self.identify_sections = dspy.Predict(IdentifySections)

    def _load_persistent_ids(self):
        """Load saved pdf_path → doc_id mapping from disk."""
        if os.path.exists(self.DOC_ID_CACHE_FILE):
            with open(self.DOC_ID_CACHE_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save_persistent_ids(self):
        """Save pdf_path → doc_id mapping to disk."""
        with open(self.DOC_ID_CACHE_FILE, "w") as f:
            json.dump(self._persistent_ids, f, indent=2)

    def _upload_and_process(self, pdf_path):
        """Upload PDF (or reuse existing doc_id), return tree + node_map."""
        # 1. In-memory cache (current run)
        if pdf_path in self.doc_cache:
            print(f"  [PageIndex CACHE] {pdf_path}")
            return self.doc_cache[pdf_path]

        # 2. Persistent cache (previous runs)
        if pdf_path in self._persistent_ids:
            doc_id = self._persistent_ids[pdf_path]
            print(f"  [PageIndex REUSE] {pdf_path} → {doc_id}")
            try:
                status = self.pi_client.get_document(doc_id)["status"]
            except Exception:
                status = "expired"

            if status not in ("completed", "processing"):
                print(f"  [PageIndex] Cached doc_id expired, re-uploading...")
                del self._persistent_ids[pdf_path]
                self._save_persistent_ids()
                result = self.pi_client.submit_document(pdf_path)
                doc_id = result["doc_id"]
                print(f"  [PageIndex DOC_ID] {doc_id}")
                self._persistent_ids[pdf_path] = doc_id
                self._save_persistent_ids()
        else:
            # 3. First time — upload
            print(f"  [PageIndex UPLOAD] {pdf_path}")
            try:
                result = self.pi_client.submit_document(pdf_path)
            except Exception as e:
                if "Too many files with similar names" in str(e):
                    unique_name = os.path.splitext(os.path.basename(pdf_path))[0] + f"_{os.getpid()}.pdf"
                    tmp_path = os.path.join(tempfile.gettempdir(), unique_name)
                    shutil.copy2(pdf_path, tmp_path)
                    print(f"  [PageIndex] Retrying with unique name: {unique_name}")
                    try:
                        result = self.pi_client.submit_document(tmp_path)
                    finally:
                        os.remove(tmp_path)
                else:
                    raise
            doc_id = result["doc_id"]
            print(f"  [PageIndex DOC_ID] {doc_id}")
            self._persistent_ids[pdf_path] = doc_id
            self._save_persistent_ids()

        print("  [PageIndex] Waiting for processing...")
        timeout_sec = 180  # 3 minutes max
        elapsed = 0
        while True:
            doc_info = self.pi_client.get_document(doc_id)
            status = doc_info.get("status", "unknown")
            if status == "completed":
                break
            if status in ("failed", "error", "cancelled"):
                raise ValueError(f"PageIndex processing failed (status={status}) for {pdf_path}")
            time.sleep(3)
            elapsed += 3
            if elapsed >= timeout_sec:
                raise TimeoutError(f"PageIndex processing timeout after {timeout_sec}s for {pdf_path}")
        print("  [PageIndex] Ready.")

        tree = self.pi_client.get_tree(doc_id, node_summary=True)['result']   # why ??
        node_map = pageindex_utils.create_node_mapping(tree)

        self.doc_cache[pdf_path] = {
            'doc_id': doc_id,
            'tree': tree,
            'node_map': node_map
        }
        return self.doc_cache[pdf_path]

    def _simplify_tree(self, node):
        """Recursively extract only node_id and title to reduce token usage."""
        if isinstance(node, list):
            return [self._simplify_tree(n) for n in node]
        item = {"id": node.get("node_id"), "title": node.get("title", "")}
        children = node.get("nodes", [])
        if children:
            item["children"] = [self._simplify_tree(c) for c in children]
        return item

    def _identify_section_nodes(self, tree):
        """Use DSPy Predict to identify which tree nodes belong to which section."""
        simple_tree = self._simplify_tree(tree)
        pred = self.identify_sections(
            tree_structure=json.dumps(simple_tree, indent=2, ensure_ascii=False)
        )
        try:
            raw = pred.section_mapping
            match = re.search(r'\{.*\}', str(raw), re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")
            result = json.loads(match.group())
            for key in ["abstract", "introduction", "related_work", "methods", "experiments", "conclusion"]:
                val = result.get(key, [])
                result[key] = [val] if isinstance(val, str) else val
            # Deduplicate node_ids within each section
            for key in result:
                seen_ids = set()
                result[key] = [nid for nid in result[key] if not (nid in seen_ids or seen_ids.add(nid))]
            # Remove cross-section duplicates (first section in order wins)
            all_assigned = set()
            for key in ["abstract", "introduction", "related_work", "methods", "experiments", "conclusion"]:
                result[key] = [nid for nid in result[key] if nid not in all_assigned]
                all_assigned.update(result[key])
            return result
        except Exception as e:
            print(f"  [PageIndex] Section parsing error: {e}")
            return {k: [] for k in ["abstract", "introduction", "related_work", "methods", "experiments", "conclusion"]}

    def _get_section_text(self, node_id, node_map):
        """Collect text from a node and all its descendant nodes recursively (deduplication included)."""
        texts = []
        seen = set()
        def collect(node):
            text = node.get('text', '').strip()
            if text and text not in seen:
                texts.append(text)
                seen.add(text)
            for child in node.get('nodes', []):
                collect(child)
        if node_id in node_map:
            collect(node_map[node_id])
        return "\n\n".join(texts)

    def forward(self, pdf_path):
        """
        Extract sections from PDF via PageIndex.
        Returns dspy.Prediction with standard keys.
        """
        data = self._upload_and_process(pdf_path)
        node_map = data['node_map']
        tree = data['tree']

        section_nodes = self._identify_section_nodes(tree)

        # Fallback: for empty sections, search unmapped nodes by title keywords
        FALLBACK_KEYWORDS = {
            "abstract":     ["abstract", "summary"],
            "introduction": ["introduction", "motivation", "overview", "background", "preliminaries",
                             "problem statement", "problem formulation"],
            "related_work": ["related work", "related literature", "background and related",
                             "literature review", "prior work", "previous work", "review of"],
            "methods":      ["method", "approach", "framework", "architecture", "proposed method",
                             "methodology", "model", "algorithm", "formulation", "our approach",
                             "technical", "system design"],
            "experiments":  ["experiment", "result", "evaluation", "benchmark", "ablation",
                             "empirical", "analysis", "performance", "quantitative"],
            "conclusion":   ["conclusion", "concluding", "discussion", "limitations",
                             "future work", "future direction", "closing", "final remark"],
        }
        all_assigned = set(nid for nids in section_nodes.values() for nid in nids)
        for key, kws in FALLBACK_KEYWORDS.items():
            if not section_nodes.get(key):
                for nid, node in node_map.items():
                    if nid in all_assigned:
                        continue
                    if any(kw in node.get("title", "").lower() for kw in kws):
                        section_nodes[key] = [nid]
                        all_assigned.add(nid)
                        print(f"  [PageIndex] Fallback: '{node.get('title')}' → {key}")
                        break

        # Post-processing: remove reference/bibliography nodes from related_work
        ref_keywords = {"reference", "bibliography"}
        cleaned_rw = []
        for nid in section_nodes.get("related_work", []):
            title = node_map.get(nid, {}).get("title", "").lower()
            if any(kw in title for kw in ref_keywords):
                print(f"  [PageIndex] Removed '{title}' (node {nid}) from related_work — is a references section")
            else:
                cleaned_rw.append(nid)
        section_nodes["related_work"] = cleaned_rw

        print(f"  [PageIndex] Section mapping: { {k: v for k, v in section_nodes.items()} }")

        results = {}
        for key, output_key in self.KEY_MAP.items():
            node_ids = section_nodes.get(key, [])
            texts = []
            for nid in node_ids:
                txt = self._get_section_text(nid, node_map)
                if txt:
                    texts.append(txt)
            combined = "\n\n".join(texts)
            if key == "related_work":
                for marker in ["\n## References", "\n### References", "\n# References",
                                "\n## Bibliography", "\n### Bibliography"]:
                    if marker.lower() in combined.lower():
                        idx = combined.lower().index(marker.lower())
                        combined = combined[:idx]
            results[output_key] = combined
            preview = results[output_key][:100].replace('\n', ' ') if results[output_key] else "EMPTY"
            status = "OK" if results[output_key] else "MISSING"
            print(f"  [PageIndex] {output_key:15s} [{status}] {len(results[output_key]):6d} chars | {preview!r}")

        return dspy.Prediction(**results)



class ReviewPipeline(dspy.Module):
    """Assign models based on task complexity"""

    def __init__(self,use_pageindex=False,pageindex_api_key=None):
        super().__init__()
        self.use_pageindex = use_pageindex
        self.pageindex_api_key = pageindex_api_key
        

        if use_pageindex:
            if not pageindex_api_key:
                raise ValueError("pageindex_api_key is required when use_pageindex=True")
            self.pageindex_extractor = PageIndexSectionExtractor(api_key=pageindex_api_key)
        else:
            dspy.configure(lm=fast_model, track_usage=True)
            self.section_extractor = dspy.Predict(SectionExtractor)

        
        self.ca_reviewer = dspy.Predict(CAReview)
        self.methodology_reviewer = dspy.Predict(MethodologyReview)
        self.clarity_reviewer = dspy.Predict(ClarityReview)

        dspy.configure(lm=powerful_model, track_usage=True)

        self.conflict_resolver = dspy.Predict(ConflictResolver)
        self.synthesizer = dspy.ChainOfThought(ReviewSynthesizer)
        self.debug_mode = False
        self.rubric = ICLR_RUBRIC

    def forward(self, article_text=None, pdf_path=None):
        # Stage 1: Extract sections
        try:
            if self.use_pageindex:
                if not pdf_path:
                    raise ValueError("pdf_path is required when use_pageindex=True")
                structure = self.pageindex_extractor(pdf_path=pdf_path)
                if not article_text:
                    article_text = "\n\n".join(filter(None, [
                        structure.Abstract, structure.Introduction,
                        structure.Related_Work, structure.Methods, structure.Experiments,
                        structure.Conclusion
                    ]))
            else:
                structure = self.section_extractor(article_text=article_text)
                
        except Exception as e:
            print(f"Section extraction failed: {e}")
            return dspy.Prediction(
                decision="Reject",
                recommendation=1,
                comments="Failed to extract sections",
                strengths="N/A",
                weaknesses="N/A",
                soundness=1,
                clarity=1,
                summary="Error in processing"
            )

        if structure:
            sections = structure
        
        if len(sections.Introduction) < 50:
            print("Warning: Short introduction extracted")

        # Stage 2: Role-specialized reviews
        ca_review = self.ca_reviewer(
            introduction_section=sections.Introduction,
            related_work_section=sections.Related_Work,
            conclusion_section=sections.Conclusion
        )

        methodology_review = self.methodology_reviewer(
            methods_section=sections.Methods,
            experiments_section=sections.Experiments
        )

        clarity_review = self.clarity_reviewer(article_text=article_text)

        # Stage 2.5: Conflict Resolution
        resolved = self.conflict_resolver(
            ca_review=ca_review.novelty_review,
            methodology_review=methodology_review.technical_review
        )

        ca_combined = f"Type: {ca_review.contribution_type}\nReview: {ca_review.novelty_review}"
        methodology_combined = f"{methodology_review.technical_review}\n\nConflict Resolution: {resolved.resolved_critique}"


        # Stage 3: Synthesize into structured review
        synthesis = self.synthesizer(
            ca_review=ca_combined,
            methodology_review=methodology_combined,
            clarity_review=clarity_review.clarity_review,
            rubric=self.rubric
        )

        # Build full review text for metrics comparison
        full_review_text = (
            f"{synthesis.comments}\n\n"
            f"Strengths:\n{synthesis.strengths}\n\n"
            f"Weaknesses:\n{synthesis.weaknesses}\n\n"
            f"Clarification Questions:\n{synthesis.clarification_questions}"
        )

        result = dspy.Prediction(
            decision=synthesis.decision,
            recommendation=synthesis.recommendation,
            score=float(synthesis.recommendation),  # backward compat
            comments=synthesis.comments,
            strengths=synthesis.strengths,
            weaknesses=synthesis.weaknesses,
            clarification_questions=synthesis.clarification_questions,
            soundness=synthesis.soundness,
            clarity=synthesis.clarity,
            summary=full_review_text,  # backward compat
            ca_review=ca_review.novelty_review,
            methodology_review=methodology_review.technical_review,
            clarity_review=clarity_review.clarity_review,
            resolved_critique=resolved.resolved_critique,
            extracted_sections=sections if self.debug_mode else None
        )

        return result
