"""PageIndex-based section extraction pipeline (v3) — refined version with improved section handling."""

import dspy
import json
import os
import re
import time
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import shutil, tempfile

try:
    from pageindex import PageIndexClient
    import pageindex.utils as pageindex_utils
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PAGEINDEX_AVAILABLE = False


# -------------------------------------------------------------------------
# RLM Infrastructure (unchanged)
# -------------------------------------------------------------------------

class ContextTools:
    def __init__(self, full_text):
        self.context = full_text

    def search_headers(self):
        pattern = r"^## (.+)$"
        matches = []
        for match in re.finditer(pattern, self.context, re.MULTILINE):
            title = match.group(1).strip()
            slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            matches.append({
                "title": title,
                "id": slug,
                "start_index": match.start()
            })
        return matches

    def get_text_slice(self, start_idx, end_idx):
        return self.context[start_idx:end_idx]

    def peek(self, start, length=500):
        end = min(start + length, len(self.context))
        return self.context[start:end]

    def get_length(self):
        return len(self.context)


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


class SectionBounds(BaseModel):
    start: int
    end: int

class NavigatorOutput(BaseModel):
    thought: str = Field(description="Reasoning about the document structure.")
    action: str = Field(description="search_headers, peek, get_length, or FINISH")
    action_input: Dict = Field(default_factory=dict)
    section_mapping: Optional[Dict[str, SectionBounds]] = Field(default=None)


class RLMNavigator(dspy.Signature):
    """
    You are an agent exploring a document.
    Cycle:
    1. THOUGHT: Plan your next move.
    2. ACTION: Choose a tool (search_headers, peek, get_length, FINISH).
    3. ACTION_INPUT: Parameters for the tool.

    When you find all sections, output action 'FINISH' and fill 'section_mapping'.
    """
    instruction = dspy.InputField()
    available_tools = dspy.InputField()
    history = dspy.InputField()

    thought = dspy.OutputField(description="Reasoning about what to do next.")
    action = dspy.OutputField(description="One of: search_headers, peek, get_length, FINISH.")
    action_input = dspy.OutputField(description="JSON params, e.g. {'start': 0}. Empty {} if none.")
    section_mapping = dspy.OutputField(description="JSON mapping ONLY if action is FINISH. E.g. {'Abstract': {'start':0, 'end':100}}.")


class SectionExtractor(dspy.Signature):
    """Extract sections from a scientific article in markdown. Do not summarize."""

    article_text: str = dspy.InputField(description="Full article text in markdown.")
    Abstract: str = dspy.OutputField(description="Abstract section content.")
    Introduction: str = dspy.OutputField(description="Introduction section content.")
    Related_Work: str = dspy.OutputField(description="Related Work / Background section content.")
    Methods: str = dspy.OutputField(description="Methods / Methodology section content.")
    Experiments: str = dspy.OutputField(description="Experiments / Results section content.")
    Conclusion: str = dspy.OutputField(description="Conclusion / Discussion section content.")


class SectionSingleExtractor(dspy.Signature):
    """Extract a specific section from a markdown paper without summarizing."""
    article_text = dspy.InputField(description="The content text fragment.")
    section_name = dspy.InputField(description="e.g., 'Methods', 'Abstract'")
    specific_instructions = dspy.InputField(description="Specific rules for this section type.")
    extracted_content = dspy.OutputField(description="Clean extracted text.")


# -------------------------------------------------------------------------
# Simplified Review Signatures
# -------------------------------------------------------------------------

class CAReview(dspy.Signature):
    """
    You are a conference reviewer evaluating contribution and novelty. Apply domain-agnostic criteria only.
    Be critical by default — most submitted papers have only incremental contributions.

    Key criteria:
      Motivation: Does it address a real, significant bottleneck?
      Novelty: Is the idea genuinely new? Combining existing methods without new insight = incremental = weakness.
      Evidence: Missing ablations or single-dataset evaluation = weakness.
      Prior Work: Based on your knowledge, does this idea already exist? Repackaging known techniques = weakness.

    Output: Strengths / Weaknesses (cite specific sections) / Suggestions for improvement.
    """
    introduction_section: str = dspy.InputField(description="Introduction text with motivation and problem statement.")
    related_work_section: str = dspy.InputField(description="Related Work text for assessing novelty context.")
    conclusion_section: str = dspy.InputField(description="Conclusion text with authors' contribution summary.")
    novelty_review: str = dspy.OutputField(description=" Detailed novelty review with strengths and weaknesses and suggestions for improvement.")

class MethodologyReview(dspy.Signature):
    """
    You are a conference reviewer evaluating methodological soundness. Apply domain-agnostic criteria only.
    Be critical by default — missing baselines alone is sufficient grounds for rejection.

    Key criteria:
      Baselines: Are comparisons with strong, recent baselines included? Missing or weak baselines = rejection risk.
      Scale: Is the evaluation large enough? Small sample size or single dataset = weakness.
      Ablations: Is each component isolated and validated? Missing ablations = weakness.
      Results: Are improvements statistically significant, not marginal or cherry-picked?
      Reproducibility: Enough implementation detail to reimplement?

    Output: Strengths / Weaknesses (cite specific sections) / Suggestions for improvement.
    """
    methods_section: str = dspy.InputField(description="Methods section text with technical contribution.")
    experiments_section: str = dspy.InputField(description="Experiments section text with empirical evaluation.")
    technical_review: str = dspy.OutputField(description="Detailed methodology review with strengths and weaknesses and suggestions for improvement.")
    

class ClarityReview(dspy.Signature):
    """
    You are a conference reviewer evaluating clarity and presentation. Apply domain-agnostic criteria only.
    Be critical by default — poor presentation is a valid rejection reason.

    Critical issues (each alone can justify Reject): main method unclear, unexplained key equations,
      results presented without context, algorithm not reproducible from description alone.
    Minor issues: typos, unclear figure captions, inconsistent notation.
    Quote the exact sentence or equation that is unclear.

    Output: Strengths / Weaknesses (cite specific sections) / Suggestions for improvement.
    """
    article_text: str = dspy.InputField(description="Full article text.")
    clarity_review: str = dspy.OutputField(description="Detailed clarity review with Strengths and Weaknesses and suggestions for improvement.")

class SummaryGenerator(dspy.Signature):
    """
    Synthesize three specialist reviews into a final Meta-Review. Apply domain-agnostic judgment.
    Identify consensus and conflicts across reviews.

    DEALBREAKERS (any one is sufficient to recommend Reject):
      - No genuine novelty over existing work
      - Missing comparison with strong recent baselines
      - Results not statistically significant or marginal
      - Main method or key equations are unclear

    Methodology soundness outweighs clarity. Output: Strengths / Weaknesses (major to minor) / Suggestions.
    """
    ca_review: str = dspy.InputField(description="Contribution assessment review with strengths and weaknesses.")
    methodology_review: str = dspy.InputField(description="Methodology review with strengths and weaknesses.")
    clarity_review: str = dspy.InputField(description="Clarity review with strengths and weaknesses.")
    summary: str = dspy.OutputField(description="Structured meta-review (Strengths, Weaknesses, Suggestions).")


class DecisionPredictor(dspy.Signature):
    """
    You are a conference Area Chair predicting accept/reject decision. Apply domain-agnostic judgment only.

    Default score is 5.0 (Reject). Add points only for confirmed strengths, subtract for every unresolved weakness.
    To score >= 6.1 (Accept), ALL must be present: genuine novelty, strong recent baselines, ablations, significant results.

    SCORING:
    9.0-10.0: Breakthrough, no flaws.
    7.5-8.5:  Strong Accept — ablations + strong baselines.
    6.1-7.4:  Accept — solid, minor gaps only.
    4.0-6.0:  Reject — missing baselines, incremental, or limited scale.
    1.0-3.5:  Strong Reject — errors or no contribution.

    Output 'Accept' only if score >= 6.1, else 'Reject'.
    """

    ca_review: str = dspy.InputField(description="Contribution assessment review with strengths and weaknesses.")      
    methodology_review: str = dspy.InputField(description="Methodology review with strengths and weaknesses.")
    clarity_review: str = dspy.InputField(description="Clarity review with strengths and weaknesses.")
    #summary: str = dspy.InputField(description="Final meta-review summary with all identified strengths and weaknesses.")
    decision: str = dspy.OutputField(description="'Accept' or 'Reject'.")
    score: float = dspy.OutputField(description="Score from 1.0 to 10.0.")
   
    final_justification: str = dspy.OutputField(description="Brief justification for the chosen score.")


class LLMJudge(dspy.Signature):
    """Compare the generated review with the human review and explain why the generated review is better or worse than the human review. 
    Rate the accuracy and significance on a scale from 1 to 10."""
    generated_review: str = dspy.InputField(description="Generated review text.")
    human_review: str = dspy.InputField(description="Human review text.")
    evaluation: float = dspy.OutputField(description="Score (1-10).")


# -------------------------------------------------------------------------
# RLM Section Extractors (unchanged)
# -------------------------------------------------------------------------

class RecursiveSectionExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.navigator = dspy.Predict(RLMNavigator)

    def _robust_json_parse(self, raw_text):
        if not raw_text: return None
        if isinstance(raw_text, dict): return raw_text

        text = str(raw_text).strip()
        text = re.sub(r"^```json", "", text).strip()
        text = re.sub(r"^```", "", text).strip()
        text = text.strip("`")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return None

    def _execute_tool(self, tools, action, action_input):
        act_name = str(action).strip().lower()
        params = self._robust_json_parse(action_input)
        if not params: params = {}

        if "search_headers" in act_name:
            return json.dumps(tools.search_headers(), ensure_ascii=False)
        elif "peek" in act_name:
            start = int(params.get("start", 0))
            length = int(params.get("length", 1000))
            return tools.peek(start, length)
        elif "get_length" in act_name:
            return str(tools.get_length())
        return f"Error: Unknown tool '{act_name}'"

    def _fallback_mapping(self, headers, doc_length):
        section_keywords = {
            "Abstract": ["abstract"],
            "Introduction": ["introduction"],
            "Related_Work": ["related work", "background", "literature review", "prior work"],
            "Methods": ["method", "approach", "framework", "preliminar", "proposed", "model", "architecture"],
            "Experiments": ["experiment", "result", "evaluation", "benchmark", "ablation", "scaling"],
            "Conclusion": ["conclusion", "discussion", "limitation", "future work", "summary"]
        }
        section_order = ["Abstract", "Introduction", "Related_Work", "Methods", "Experiments", "Conclusion"]
        sorted_headers = sorted(headers, key=lambda x: x['start_index'])

        section_starts = {}
        for section in section_order:
            keywords = section_keywords[section]
            for h in sorted_headers:
                title_lower = h["title"].lower()
                if any(kw in title_lower for kw in keywords):
                    section_starts[section] = h["start_index"]
                    break

        found = sorted(section_starts.items(), key=lambda x: x[1])
        mapping = {}
        for i, (section, start) in enumerate(found):
            if i + 1 < len(found):
                end = found[i + 1][1]
            else:
                end = doc_length
            mapping[section] = {"start": start, "end": end}
        return mapping

    def forward(self, full_text):
        tools = ContextTools(full_text)

        available_tools_desc = (
            "1. search_headers(): Find all titles and indices.\n"
            "2. peek(start, length): Read text around an index.\n"
            "3. get_length(): Total characters.\n"
            "4. FINISH: Output the section_mapping."
        )

        history = ""
        max_steps = 6
        final_mapping = None

        print(f"--- RLM Start (Doc Len: {len(full_text)}) ---")

        for step in range(max_steps):
            current_instruction = "Find indices for: Abstract, Intro, Related Work, Methods, Experiments, Conclusion."
            if step == 0:
                current_instruction += " Start by calling 'search_headers'."

            pred = self.navigator(
                instruction=current_instruction,
                available_tools=available_tools_desc,
                history=history if history else "Start."
            )

            thought = getattr(pred, 'thought', '')
            action = getattr(pred, 'action', '')
            action_input = getattr(pred, 'action_input', '{}')

            print(f" [Step {step+1}] Thought: {thought[:50]}... -> Action: {action}")

            if "FINISH" in str(action).upper():
                mapping_str = getattr(pred, 'section_mapping', '{}')
                parsed = self._robust_json_parse(mapping_str)
                if parsed and len(parsed) > 0:
                    final_mapping = parsed
                    break
                else:
                    print("  [RLM] FINISH called but mapping invalid. Retrying...")

            tool_res = self._execute_tool(tools, action, action_input)
            history += f"\n[Step {step+1}]\nThought: {thought}\nAction: {action}\nInput: {action_input}\nResult: {tool_res[:600]}..."

        if not final_mapping:
            print(" [RLM] Fallback to heuristics.")
            final_mapping = self._fallback_mapping(tools.search_headers(), len(full_text))

        return final_mapping


class RobustSectionExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(SectionSingleExtractor)
        self.section_rules = {
            "Abstract": "Extract the summary. Usually one or two paragraphs.",
            "Introduction": "Focus on motivation and problem statement. Stop before Related Work/Methods.",
            "Related_Work": "Include citations and comparisons to prior work.",
            "Methods": "CRITICAL: Preserve all LaTeX equations, algorithm blocks, and technical definitions.",
            "Experiments": "Preserve all tables, metric results (FID, Accuracy), and hardware setups.",
            "Conclusion": "Extract final summary, limitations, and future work."
        }

    def forward(self, full_text, mapping):
        extracted_sections = {}
        if not mapping:
             mapping = {}

        for section_name in ["Abstract", "Introduction", "Related_Work", "Methods", "Experiments", "Conclusion"]:
            bounds = mapping.get(section_name)
            if not bounds:
                extracted_sections[section_name] = ""
                continue

            start = int(bounds.get("start") or 0)
            end = int(bounds.get("end") or start)
            content = full_text[start:end]

            if len(content) < 50:
                 extracted_sections[section_name] = ""
                 continue

            print(f" [Extractor] {section_name} ({len(content)} chars)")
            extracted_sections[section_name] = content

        return dspy.Prediction(**extracted_sections)


# -------------------------------------------------------------------------
# PageIndex Section Extractor
# -------------------------------------------------------------------------

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
        timeout_sec = 300  # 5 minutes max
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

        tree = self.pi_client.get_tree(doc_id, node_summary=True)['result']
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
            # Ensure all 6 keys exist and values are lists
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

        results = {}
        print(f"  [PageIndex] Section mapping: { {k: v for k, v in section_nodes.items()} }")
        for key, output_key in self.KEY_MAP.items():
            node_ids = section_nodes.get(key, [])
            texts = []
            for nid in node_ids:
                txt = self._get_section_text(nid, node_map)
                if txt:
                    texts.append(txt)
            combined = "\n\n".join(texts)
            # Remove any References/Bibliography block that leaked into related_work
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


# -------------------------------------------------------------------------
# Review Pipeline
# -------------------------------------------------------------------------

class ReviewPipeline(dspy.Module):
    """
    Review pipeline with simplified prompts.

    Args:
        use_rlm: If True, uses RecursiveSectionExtractor (agentic, multi-step).
                 If False, uses simple dspy.Predict(SectionExtractor) (single-step).
        use_pageindex: If True, uses PageIndex API for section extraction from PDF.
        pageindex_api_key: API key for PageIndex (required when use_pageindex=True).
    """
    def __init__(self, use_rlm=True, use_pageindex=False, pageindex_api_key=None):
        super().__init__()
        self.use_rlm = use_rlm
        self.use_pageindex = use_pageindex

        if use_pageindex:
            if not pageindex_api_key:
                raise ValueError("pageindex_api_key is required when use_pageindex=True")
            self.pageindex_extractor = PageIndexSectionExtractor(api_key=pageindex_api_key)
        else:
            self.navigator = RecursiveSectionExtractor()
            if use_rlm:
                self.section_extractor = RobustSectionExtractor()
            else:
                self.section_extractor = dspy.Predict(SectionExtractor)

        self.ca_reviewer = dspy.Predict(CAReview)
        self.methodology_reviewer = dspy.Predict(MethodologyReview)
        self.clarity_reviewer = dspy.Predict(ClarityReview)
        self.summary_generator = dspy.Predict(SummaryGenerator)
        self.decision_predictor = dspy.Predict(DecisionPredictor)
        self.llm_judge = dspy.ChainOfThought(LLMJudge)

    def forward(self, article_text=None, pdf_path=None):
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
            elif self.use_rlm:
                mapping = self.navigator(full_text=article_text)
                structure = self.section_extractor(full_text=article_text, mapping=mapping)
            else:
                structure = self.section_extractor(article_text=article_text)
        except Exception as e:
            print(f"Section extraction failed: {e}")
            raise ValueError(f"Section extraction failed: {e}")

        # Check if extraction produced enough content
        critical = [structure.Abstract, structure.Introduction, structure.Methods, structure.Experiments]
        if sum(1 for s in critical if s and len(s.strip()) > 100) < 2:
            raise ValueError("Extraction failed: too many critical sections are empty")

        # Reviews
        ca_result = self.ca_reviewer(
            introduction_section=structure.Introduction,
            related_work_section=structure.Related_Work,
            conclusion_section=structure.Conclusion
        )
        meth_result = self.methodology_reviewer(
            methods_section=structure.Methods,
            experiments_section=structure.Experiments
        )
        clarity_result = self.clarity_reviewer(article_text=article_text)

        # Summary
        final_summary = self.summary_generator(
            ca_review=ca_result.novelty_review,
            methodology_review=meth_result.technical_review,
            clarity_review=clarity_result.clarity_review,
            
        )

        # Decision — gets full reviews for more accurate scoring
        prediction = self.decision_predictor(
            #summary=final_summary.summary
            ca_review=ca_result.novelty_review,
            methodology_review=meth_result.technical_review,
            clarity_review=clarity_result.clarity_review,

        )


        # Enforce consistency between score and decision
        try:
            score_val = float(prediction.score)
        except (ValueError, TypeError):
            score_val = 5.0
        enforced_decision = "Accept" if score_val >= 6.1 else "Reject"
        if enforced_decision != prediction.decision.strip():
            print(f"  [Pipeline] Decision corrected: {prediction.decision} → {enforced_decision} (score={score_val})")

        return dspy.Prediction(
            decision=enforced_decision,
            score=score_val,
            justification=prediction.final_justification,
            summary=final_summary.summary,
            ca_review=ca_result.novelty_review,
            methodology_review=meth_result.technical_review,
            sections=structure,
            clarity_review=clarity_result.clarity_review,
        )
