
"""
PDF section extraction for ReviewBuddy.

Uses pageindex-local to parse a PDF into a structured node tree, then maps
nodes to canonical paper sections (Abstract, Introduction, Related Work,
Methods, Experiments, Conclusion).

Pre-processing: the raw tree is sanitized (UTF-16 surrogates) and split at
embedded section headings to recover nodes that pageindex merged together.

Post-processing: keyword-based fallback when LLM mapping fails, page-header
stripping, Methods/Experiments rebalancing, conclusion extraction from
misplaced nodes, and optional vision-LLM figure descriptions appended to
the relevant tree nodes.

Falls back to PyMuPDF when pageindex times out or is unavailable. Results are cached to disk to avoid reprocessing.

Main class: LocalPDFSectionExtractor
"""

import base64
import concurrent.futures
import dspy
import hashlib
import json
import os
import re
from contextlib import nullcontext
from collections import Counter
import fitz  # PyMuPDF

from reviewbuddy.prompts import (
    DescribeFigures, IdentifySections, SectionExtractor,
)

try:
    from pageindex import page_index as _pageindex_local
    PAGEINDEX_LOCAL_AVAILABLE = True
except ImportError:
    PAGEINDEX_LOCAL_AVAILABLE = False


def _call_pageindex_with_timeout(pdf_path, model, api_key, api_base, timeout=180):
    """Run pageindex-local with a timeout. Returns (tree, None) on success or (None, reason) on failure."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _executor:
            _future = _executor.submit(
                _pageindex_local,
                doc=pdf_path,
                model=model,
                api_key=api_key,
                api_base=api_base,
                if_add_node_id="yes",
                if_add_node_text="yes",
            )
            result = _future.result(timeout=timeout)
        tree = result.get("structure", result.get("result", result)) if isinstance(result, dict) else result
        tree = tree if isinstance(tree, list) else [tree]
        return tree, None
    except concurrent.futures.TimeoutError:
        return None, f"pageindex timed out after {timeout // 60}min"
    except Exception as e:
        return None, str(e)


def _sanitize_text(text: str) -> str:
    """Remove lone UTF-16 surrogate characters that PyMuPDF sometimes produces."""
    if not isinstance(text, str):
        return text
    return text.encode("utf-8", errors="replace").decode("utf-8")



# -------------------------------------------------------------------------
# Local PDF Section Extractor — pageindex-local (no cloud)
# -------------------------------------------------------------------------

class LocalPDFSectionExtractor(dspy.Module):
    """
    Extract sections from a PDF locally using pageindex-local.
    Runs entirely via your own LLM endpoint — no data sent to pageindex.ai.

    Pass cache_dir to avoid re-processing PDFs: the raw tree JSON is saved after the
    first run and reloaded on subsequent runs, skipping the expensive pageindex call.

    Requires: uv add pageindex-local
    """

    SECTIONS = ["abstract", "introduction", "related_work", "methods", "experiments", "conclusion"]
    OUTPUT_KEYS = {
        "abstract":     "Abstract",
        "introduction": "Introduction",
        "related_work": "Related_Work",
        "methods":      "Methods",
        "experiments":  "Experiments",
        "conclusion":   "Conclusion",
    }
    FALLBACK_KEYWORDS = {
        "abstract":     ["abstract", "summary"],
        "introduction": ["introduction", "motivation", "overview"],
        "related_work": ["related work", "related literature", "background",
                         "literature review", "prior work", "previous work",
                         "state of the art", "survey", "prior art", "preliminaries", "theoretical background"],
        "methods":      ["method", "approach", "framework", "architecture", "proposed",
                         "algorithm", "formulation", "problem statement", "system design", "methodology", "embedding"],
        "experiments":  ["experiment", "result", "evaluation", "benchmark", "ablation",
                         "empirical", "analysis", "performance", "quantitative", 
                         "case study", "numerical", "simulation", "setup", "implementation"],
        "conclusion":   ["conclusion", "concluding", "discussion", "limitation",
                         "future work", "future direction", "future scope",
                         "closing", "final remark"],
    }

    # ── Init ───────────────────────────────────────────────────────────────────

    def __init__(self, model: str, api_key: str, api_base: str,
                 fast_lm=None, cache_dir: str = None,
                 pdf_parser: str = "pageindex", vision_lm=None,
                 use_llm_mapping: bool = False):
        """
        Args:
            pdf_parser: One of:
                - "pageindex" (default) — uses pageindex-local LLM-based TOC extraction
                - "pymupdf"             — extracts text page-by-page via PyMuPDF,
                                          handles image-heavy PDFs better than PyPDF2
            vision_lm: Pre-built dspy.LM for vision tasks. Created from lm_config if omitted.
            use_llm_mapping: If False (default), skip the nested dspy LLM call for section
                             identification and rely on keyword fallback only. This avoids
                             concurrent.futures thread-pool corruption when running inside
                             dspy.Evaluate's ThreadPoolExecutor (Python 3.13 issue).
        """
        super().__init__()
        if not PAGEINDEX_LOCAL_AVAILABLE:
            raise ImportError("pageindex-local is not installed. Run: uv add pageindex-local")
        self.model           = model
        self.api_key         = api_key
        self.api_base        = api_base
        self._fast_lm        = fast_lm
        self.cache_dir       = cache_dir
        self.pdf_parser      = pdf_parser
        self._use_llm_mapping = use_llm_mapping
        self.identify_sections   = dspy.Predict(IdentifySections)
        self.section_extractor   = dspy.Predict(SectionExtractor)
        self.figure_describer    = dspy.Predict(DescribeFigures)
        if vision_lm is None:
            from lm_config import create_lms
            _, vision_lm, _ = create_lms(api_key, api_base)
        self._vision_lm = vision_lm
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cache_path(self, pdf_path: str) -> str | None:
        if not self.cache_dir:
            return None
        stem      = os.path.splitext(os.path.basename(pdf_path))[0]
        path_hash = hashlib.md5(os.path.abspath(pdf_path).encode()).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"{stem}_{path_hash}.json")

    def _load_cache(self, pdf_path: str):
        path = self._cache_path(pdf_path)
        if path and os.path.exists(path):
            print(f"  [LocalPageIndex] Loading from cache: {path}")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, pdf_path: str, tree) -> None:
        path = self._cache_path(pdf_path)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(tree, f, ensure_ascii=True)
            print(f"  [LocalPageIndex] Cached to: {path}")

    # ── Tree utilities ─────────────────────────────────────────────────────────

    def preprocess_json_tree(self, tree: list) -> list:
        """
        Chirurgicky prejde JSON strom a rozdelí uzly.
        Ošetrí podkapitoly, zlé medzery a izoluje titulnú stranu.
        """
        words = [
            'ABSTRACT', 'INTRODUCTION', 'RELATED WORK', 'METHODS', 
            'EXPERIMENTS', 'CONCLUSION', 'ACKNOWLEDGEMENTS', 'REFERENCES', 'DISCUSSION'
        ]
        
        # 1. Regex imunita voči medzerám ("R\s*E\s*L\s*A\s*T\s*E\s*D\s*W\s*O\s*R\s*K")
        flex_words = [r'\s*'.join(list(w.replace(' ', ''))) for w in words]
        kw_group = r'(?i:' + r'|'.join(flex_words) + r')'
        
        # 2. Vzor: Zachytí CELÝ nadpis (aj s číslom podkapitoly napr. "1.4 ") do jednej skupiny
        pattern_str = r'\n\s*((?:\d+(?:\.\d+)*\s*)?(?:' + kw_group + r'\b|[A-Z][A-Z\s\-]{3,}))\.?\s*\n'
        pattern = re.compile(pattern_str)

        new_tree = []

        for node in tree:
            text = node.get("text", "")
            
            if not text:
                if "nodes" in node:
                    node["nodes"] = self.preprocess_json_tree(node["nodes"])
                new_tree.append(node)
                continue
            
            parts = pattern.split(text)
            
            if len(parts) == 1:
                if "nodes" in node:
                    node["nodes"] = self.preprocess_json_tree(node["nodes"])
                new_tree.append(node)
            else:
                original_text = parts[0].strip()
                
                if original_text:
                    old_node = node.copy()
                    old_node["text"] = original_text
                    # Premenujeme ho, aby ho LLM nedala omylom do uvodu
                    old_node["title"] = "TITLE_AND_AUTHORS" 
                    old_node["nodes"] = []
                    new_tree.append(old_node)
                
                for i in range(1, len(parts), 2):
                    raw_header = parts[i].strip()
                    content = parts[i+1].strip()
                    
                    # 4. AUTO-KOREKTOR: Očistenie od čísel a medzier (z "1.4 R ELATED WORK" spraví "RELATEDWORK")
                    clean_header = re.sub(r'\s+', '', raw_header).upper()
                    final_title = raw_header # Predvolený názov
                    
                    # Ak v tom "rozbitom" názve spoznáme našu kľúčovú sekciu, opravíme ju
                    for w in words:
                        if w.replace(' ', '') in clean_header:
                            final_title = w 
                            break
                    
                    original_id = node.get("node_id", "000")
                    clean_id = re.sub(r'\W+', '', final_title)[:4].upper()
                    new_id = f"{original_id}_{clean_id}"
                    
                    new_node = {
                        "node_id": new_id,
                        "title": final_title, # Teraz je to  čitateľné pre Fallback aj LLM
                        "text": f"{raw_header}\n{content}",
                        "nodes": []
                    }
                    new_tree.append(new_node)
                

        return new_tree


    def _build_node_map(self, tree) -> dict:
        node_map = {}
        def traverse(node):
            if isinstance(node, dict):
                if nid := node.get("node_id"):
                    node_map[nid] = node
                for child in node.get("nodes", []):
                    traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)
        traverse(tree)
        return node_map

    def _simplify_tree(self, node) -> dict | list:
        if isinstance(node, list):
            return [self._simplify_tree(n) for n in node]
        item = {
            "id":       node.get("node_id"),
            "title":    node.get("title", ""),
            "preview":  node.get("text", "")[:200],
            "text_len": len(node.get("text", "")),
        }
        if children := node.get("nodes", []):
            item["children"] = [self._simplify_tree(c) for c in children]
        return item

    def _get_node_text(self, node_id: str, node_map: dict) -> str:
        node = node_map.get(node_id)
        if not node:
            return ""
        parent_text = node.get("text", "").strip()

        # Recurse through all descendants to keep continuations that page-level nodes
        # sometimes store in children (instead of in the parent text blob).
        def extract_descendant_texts(n):
            parts = []
            for child in n.get("nodes", []):
                child_text = child.get("text", "").strip()
                if child_text:
                    parts.append(child_text)
                deeper = extract_descendant_texts(child)
                if deeper:
                    parts.append(deeper)
            return "\n\n".join(filter(None, parts))

        child_text = extract_descendant_texts(node)

        if parent_text and child_text:
            # Avoid obvious duplication, but do not drop child continuation text.
            if child_text in parent_text:
                # print(f"[DBG] dedup branch: child_in_parent for {node_id}")
                return parent_text
            if parent_text in child_text:
                # print(f"[DBG] dedup branch: parent_in_child for {node_id}")
                return child_text
            return f"{parent_text}\n\n{child_text}".strip()
        # print(f"[DBG] get_node_text nid={node_id} parent_len={len(parent_text)} child_len={len(child_text)}")
        return parent_text if parent_text else child_text

    # ── Section identification ─────────────────────────────────────────────────

    def _llm_identify_sections(self, tree) -> dict:
        simple = self._simplify_tree(tree)
        ctx    = dspy.context(lm=self._fast_lm) if self._fast_lm else nullcontext()
        with ctx:
            pred = self.identify_sections(
                tree_structure=json.dumps(simple, indent=2, ensure_ascii=False)
            )

        # print(f"[DBG] LLM raw section_mapping: {pred.section_mapping}")
        raw = pred.section_mapping
        if not isinstance(raw, dict):
            match = re.search(r'\{.*\}', str(raw), re.DOTALL)
            if not match:
                raise ValueError(f"No JSON in IdentifySections response: {raw!r}")
            raw = json.loads(match.group())
        if not isinstance(raw, dict):
            raw = {}

        for key in self.SECTIONS:
            val = raw.get(key, [])
            # LLM may return null/int/object; normalize safely to a list of non-empty IDs.
            if val is None:
                candidates = []
            elif isinstance(val, str):
                candidates = [val]
            elif isinstance(val, (list, tuple, set)):
                candidates = list(val)
            else:
                candidates = [val]

            normalized = []
            for item in candidates:
                if item is None:
                    continue
                nid = str(item).strip()
                if nid:
                    normalized.append(nid)
            raw[key] = normalized

        for key in self.SECTIONS:
            seen, deduped = set(), []
            for nid in raw[key]:
                if nid not in seen:
                    seen.add(nid)
                    deduped.append(nid)
            raw[key] = deduped

        assigned: set = set()
        for key in self.SECTIONS:
            raw[key] = [nid for nid in raw[key] if nid not in assigned]
            assigned.update(raw[key])
        
        # print(f"\n[DBG] normalized mapping: {raw}")

        return raw

    def _resolve_section_nodes(self, tree, node_map: dict, label: str) -> dict:
        if not self._use_llm_mapping:
            section_nodes = {k: [] for k in self.SECTIONS}
        else:
            try:
                section_nodes = self._llm_identify_sections(tree)
            except Exception as e:
                print(f"  [{label}] LLM failed: {e}. Keyword fallback only.")
                section_nodes = {k: [] for k in self.SECTIONS}

        # 1. ODSTRÁNENIE HALUCINÁCIÍ LLM: Necháme len platné ID uzlov
        valid_ids = set(node_map.keys())
        for key in self.SECTIONS:
            section_nodes[key] = [nid for nid in section_nodes.get(key, []) if nid in valid_ids]

        assigned = {nid for nids in section_nodes.values() for nid in nids}

        # Pomocná funkcia: Spočíta dĺžku textu uzla AJ všetkých jeho podkapitol
        def get_full_len(nid):
            node = node_map.get(nid, {})
            length = len(node.get("text", ""))
            for c in node.get("nodes", []):
                length += get_full_len(c.get("node_id"))
            return length

        # 2. FALLBACK FÁZA 1: Hľadanie v nadpisoch (Najbezpečnejšie)
        for key, kws in self.FALLBACK_KEYWORDS.items():
            if section_nodes[key]: 
                continue
            
            best_nid, best_len = None, -1
            for nid, node in node_map.items():
                title = node.get("title", "").lower()
                if nid not in assigned and any(kw in title for kw in kws):
                    n_len = get_full_len(nid)
                    if n_len > best_len:
                        best_len, best_nid = n_len, nid
            
            if best_nid:
                section_nodes[key] = [best_nid]
                assigned.add(best_nid)
                print(f"  [{label}] Fallback (Title): '{node_map[best_nid].get('title')}' -> {key}")

        # 3. FALLBACK FÁZA 2: Hľadanie v prvých 100 znakoch textu (Poistka)
        for key, kws in self.FALLBACK_KEYWORDS.items():
            if section_nodes[key]: 
                continue
            
            best_nid, best_len = None, -1
            for nid, node in node_map.items():
                if nid in assigned: 
                    continue
                # Pozeráme len na úplný začiatok textu, aby sme sa vyhli náhodným zmienkam
                preview = node.get("text", "")[:100].lower()
                if any(kw in preview for kw in kws):
                    n_len = get_full_len(nid)
                    if n_len > best_len:
                        best_len, best_nid = n_len, nid

            if best_nid:
                section_nodes[key] = [best_nid]
                assigned.add(best_nid)
                print(f"  [{label}] Fallback (Text-Start): '{node_map[best_nid].get('title')}' -> {key}")

        # 4. FILTER REFERENCIÍ (Zabraňuje, aby sa referencie tvárili ako iná sekcia)
        ref_kws = {"reference", "bibliography", "acknowledgment"}
        for key in self.SECTIONS:
            section_nodes[key] = [
                nid for nid in section_nodes[key]
                if not any(kw in node_map.get(nid, {}).get("title", "").lower() for kw in ref_kws)
            ]

        # 5. HLBOKÉ ODSTRÁNENIE REDUNDANCIE (Deti, vnúčatá...)
        def get_all_descendants(nid):
            desc = set()
            for c in node_map.get(nid, {}).get("nodes", []):
                cid = c.get("node_id")
                if cid:
                    desc.add(cid)
                    desc.update(get_all_descendants(cid))
            return desc

        for key in self.SECTIONS:
            nids = set(section_nodes[key])
            redundant = set()
            for nid in nids:
                redundant.update(get_all_descendants(nid))
            
            if redundant:
                section_nodes[key] = [nid for nid in section_nodes[key] if nid not in redundant]

        print(f"  [{label}] Mapping: { {k: v for k, v in section_nodes.items() if v} }")
        return section_nodes
        
        

    # ── Text cleaning ──────────────────────────────────────────────────────────

    def _detect_page_headers(self, node_map: dict, min_occurrences: int = 4) -> list[str]:
        line_counts: Counter = Counter()
        for node in node_map.values():
            text = node.get("text", "")
            if not text:
                continue
            # Strip likely visual line numbers only when followed by alphabetic text.
            # Remove standalone numeric lines and numeric prefixes stuck to text
            # at line start (e.g., "323Under review").
            text = re.sub(r'(?m)^[ \t]*\d+[ \t]*\r?$', '', text)
            text = re.sub(r'(?m)^[ \t]*\d+(?=[A-Za-z])', '', text)
            lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 10]
            for line in lines[:2] + lines[-2:]:
                line_counts[line] += 1
        return [line for line, count in line_counts.items() if count >= min_occurrences]

    def _strip_page_headers(self, text: str, headers: list[str]) -> str:
        return "\n".join(
            line for line in text.splitlines()
            if not any(line.strip() == h or (len(h) > 20 and h in line.strip()) for h in headers)
        ) 

    def _clean_text(self, text: str, key: str, page_headers: list[str]) -> str:
        # Odstráni osamotené 3-ciferné čísla (často čísla riadkov pre reviewerov)
        text = re.sub(r'(?m)^\d{3}(?!\d)', '', text)
        # Odstráni prázdne riadky
        text = re.sub(r'(?m)^\s*$\n', '', text) 
        
        if page_headers:
            text = self._strip_page_headers(text, page_headers)
            
        # BEZPEČNÁ GILOTÍNA PRE VŠETKY SEKCIE
        # Hľadá samostatne stojaci nadpis typu "References", "7. References", "VIII. Bibliography"
        m = re.search(r'(?im)^\s*(?:\d+(?:\.\d+)*\s*|[IVX]+\.?\s*)?(References|Bibliography)\s*$', text)
        if m:
            text = text[:m.start()]
        return text.strip()

    def _fix_abstract(self, abstract: str, intro: str, label: str) -> str:
        if not abstract and intro:
            print(f"  [{label}] Abstract: fallback to Introduction head (1500 chars)")
            return intro[:1500]
        if len(abstract) > 3500:
            print(f"  [{label}] Abstract trimmed (oversized container node)")
            return abstract[:3500]
        return abstract
    

    def _fix_conclusion(self, extracted_texts):
        target_key = "Conclusion"
        if target_key not in extracted_texts:
            extracted_texts[target_key] = ""

        # Prísny regex: nadpis musí byť na vlastnom riadku (za ním hneď \n alebo koniec).
        # Zabraňuje matchovaniu inline fráz ako "we provide a discussion on future work."
        pattern = r'((?:^|\n)[ \t]*(?:\d+[\.\d]*[ \t]+)?(?:Conclusions?|Concluding [Rr]emarks|Future Work|Discussion)[ \t]*[:\.]?[ \t]*\n)'

        # Preskočíme sekcie, kde sa tieto slová bežne vyskytujú ako bežný text
        SKIP_KEYS = {"References", "Abstract", "Introduction", target_key}

        for key in list(extracted_texts.keys()):
            if key in SKIP_KEYS:
                continue

            text = extracted_texts[key]
            casti = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)

            if len(casti) == 3:
                extracted_texts[key] = casti[0].strip()
                found_conclusion = (casti[1] + casti[2]).strip()
                if extracted_texts[target_key]:
                    extracted_texts[target_key] += "\n\n" + found_conclusion
                else:
                    extracted_texts[target_key] = found_conclusion
                print(f"  [FIX] Odrezaný záver zo sekcie '{key}' a pridaný do '{target_key}'")

    # ── Core pipeline ──────────────────────────────────────────────────────────

    def _sanitize_tree(self, tree) -> list:
        """Recursively sanitize surrogate characters in all text fields of the tree."""
        if isinstance(tree, list):
            return [self._sanitize_tree(node) for node in tree]
        if isinstance(tree, dict):
            return {k: (_sanitize_text(v) if isinstance(v, str) else self._sanitize_tree(v))
                    for k, v in tree.items()}
        return tree

    def _process_tree(self, tree, label: str = "") -> dspy.Prediction:

        tree = self._sanitize_tree(tree)
        tree = self.preprocess_json_tree(tree)
        node_map      = self._build_node_map(tree)
        section_nodes = self._resolve_section_nodes(tree, node_map, label)
        page_headers  = self._detect_page_headers(node_map)

        
        print(f"  [{label}] Expanded mapping: { {k: v for k, v in section_nodes.items() if v} }")

        results = {}
        for key, output_key in self.OUTPUT_KEYS.items():
            raw  = "\n\n".join(
                self._get_node_text(nid, node_map)
                for nid in section_nodes.get(key, [])
            )


            text = self._clean_text(raw, key, page_headers)

            MAX_CHARS = 30000
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "\n\n...[ZVYŠOK SEKCIE BOL SKRÁTENÝ]..."

            results[output_key] = text
            status  = "OK" if text else "MISSING"
            preview = text[:200].replace("\n", " ") if text else ""
            print(f"  [{label}] {output_key:15s} [{status}] {len(text):6d} chars | {preview!r}")

        results["Abstract"] = self._fix_abstract(
            results["Abstract"], results.get("Introduction", ""), label
        )

        self._fix_conclusion(results)

        return dspy.Prediction(**results)

    # ── PyMuPDF extraction ─────────────────────────────────────────────────────

    def _extract_text_pymupdf(self, pdf_path: str) -> list:
        """Extract structured nodes from PDF using PyMuPDF.

        3-tier strategy:
          1. Embedded TOC  — most reliable, uses PDF's own section titles.
          2. Font/heading detection — falls back when TOC is absent.
          3. Flat pages    — last resort, original behaviour.
        """
        doc = fitz.open(pdf_path)

        # Tier 1: embedded TOC
        toc = doc.get_toc()
        if toc:
            nodes = self._pymupdf_from_toc(doc, toc)
            if nodes:
                doc.close()
                print(f"  [PyMuPDF] TOC-based: {len(nodes)} sections")
                return nodes

        # Tier 2: font-size / heading-pattern detection
        nodes = self._pymupdf_from_headings(doc)
        if nodes:
            doc.close()
            print(f"  [PyMuPDF] Heading-based: {len(nodes)} sections")
            return nodes

        # Tier 3: flat pages (original fallback)
        nodes = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                nodes.append({
                    "node_id": f"page_{i + 1}",
                    "title":   f"Page {i + 1}",
                    "text":    text,
                    "nodes":   [],
                })
        doc.close()
        print(f"  [PyMuPDF] Flat-page fallback: {len(nodes)} pages")
        return nodes

    def _pymupdf_from_toc(self, doc, toc: list) -> list:
        """Build structured nodes from the PDF's embedded TOC.

        Each TOC entry becomes a node whose text spans from its start page
        to the start page of the next entry.
        """
        pages_text = [page.get_text() for page in doc]
        nodes = []
        for i, (level, title, page_1based) in enumerate(toc):
            start = page_1based - 1  # convert to 0-based
            end = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(pages_text)
            text = "\n\n".join(pages_text[start:end]).strip()
            if not text:
                continue
            node_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or f"sec_{i}"
            nodes.append({
                "node_id": node_id,
                "title":   title,
                "text":    text,
                "nodes":   [],
            })
        # Require at least 3 sections to be considered usable
        return nodes if len(nodes) >= 3 else []

    def _pymupdf_from_headings(self, doc) -> list:
        """Detect section headings via font size and numbering patterns.

        A line is treated as a heading when:
          - its maximum font size is >= 115 % of the document body size, OR
          - it is bold AND matches an Arabic/Roman numbered section pattern.

        Handles both Arabic (1., 2.1) and Roman numeral (I., II., IV.) prefixes.
        """
        # Collect every text line with its max font size and bold flag
        page_lines = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    spans = line["spans"]
                    text = " ".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    max_size = max(s["size"] for s in spans)
                    is_bold  = any(s["flags"] & 16 for s in spans)
                    page_lines.append((text, max_size, is_bold))

        if not page_lines:
            return []

        # Body font size = most common rounded size
        body_size = Counter(round(sz) for _, sz, _ in page_lines).most_common(1)[0][0]
        threshold = body_size * 1.15

        # Matches: "1 Introduction", "2.1 Related Work", "IV. Experiments", "II Background"
        heading_re = re.compile(
            r"^(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?)\s+[A-Za-z]"
        )

        def is_heading(text: str, size: float, bold: bool) -> bool:
            if len(text) > 100:
                return False
            return size >= threshold or (bold and heading_re.match(text))

        # Split lines into sections at each detected heading
        nodes = []
        current_title = None
        current_lines: list[str] = []

        def flush():
            if current_title and current_lines:
                nid = re.sub(r"[^a-z0-9]+", "_", current_title.lower()).strip("_") or f"sec_{len(nodes)}"
                nodes.append({
                    "node_id": nid,
                    "title":   current_title,
                    "text":    "\n".join(current_lines),
                    "nodes":   [],
                })

        for text, size, bold in page_lines:
            if is_heading(text, size, bold):
                flush()
                current_title = text
                current_lines = []
            else:
                current_lines.append(text)

        flush()

        return nodes if len(nodes) >= 3 else []

    # ── Vision LLM figure description ─────────────────────────────────────────

    def _describe_figures_with_vision(self, pdf_path: str, tree: list) -> list:
        """Render pages that contain figures as images and ask the vision LLM to
        describe what is shown. The description is appended to the matching tree node.
        """
        doc = fitz.open(pdf_path)

        # Find pages that have embedded images
        pages_with_figures = {i + 1 for i, page in enumerate(doc) if page.get_images()}
        if not pages_with_figures:
            doc.close()
            print("  [Vision] No image pages found.")
            return tree

        # Map page number - deepest tree node covering that page
        # pageindex uses start_index/end_index, fallback to page_start/page_end
        page_to_node = {}
        def map_pages(node):
            start = node.get("start_index") or node.get("page_start")
            end   = node.get("end_index") or node.get("page_end") or start
            if start is not None:
                for p in range(start, end + 1):
                    page_to_node[p] = node
            for child in node.get("nodes", []):
                map_pages(child)
        for node in tree:
            map_pages(node)

        # Fallback: ak pageindex nevracia page indices, mapujeme cez text overlap.
        # Pre každú stránku nájdeme node, ktorého text sa najviac prekrýva s textom stránky.
        if not page_to_node:
            print("  [Vision] No page indices in tree — using text-overlap fallback mapping.")
            all_nodes = []
            def collect_nodes(node):
                if node.get("text"):
                    all_nodes.append(node)
                for child in node.get("nodes", []):
                    collect_nodes(child)
            for node in tree:
                collect_nodes(node)

            def best_node_for_page(page_text: str):
                page_words = set(page_text.lower().split())
                best_node, best_score = None, 0
                for n in all_nodes:
                    node_words = set(n.get("text", "").lower().split())
                    score = len(page_words & node_words)
                    if score > best_score:
                        best_score, best_node = score, n
                return best_node if best_score > 10 else None

            for page_num in pages_with_figures:
                page_to_node[page_num] = best_node_for_page(doc[page_num - 1].get_text())

        described = 0
        for page_num in sorted(pages_with_figures):
            page    = doc[page_num - 1]
            pix     = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_b64 = base64.b64encode(pix.tobytes("jpeg")).decode()

            try:
                # Extract figure/table captions from page text to include in prompt
                page_text = page.get_text()
                captions = re.findall(
                    r'(?:Figure|Fig\.|Table|Tab\.)\s*\d+[.:\s]+[^\n]{5,}',
                    page_text, re.IGNORECASE
                )
                caption_hint = ""
                if captions:
                    caption_hint = (
                        "\nThis page contains the following figure/table captions: "
                        + "; ".join(c.strip() for c in captions)
                        + ".\nDescribe each one specifically by name."
                    )

                image = dspy.Image(url=f"data:image/jpeg;base64,{img_b64}")
                with dspy.context(lm=self._vision_lm):
                    pred = self.figure_describer(
                        page_image=image,
                        caption_hint=caption_hint,
                    )
                description = pred.description.strip()
                node = page_to_node.get(page_num)
                if node:
                    node["text"] = (
                        node.get("text", "") +
                        f"\n\n[FIGURE DESCRIPTION page {page_num}]\n{description}"
                    ).strip()
                    described += 1
                    print(f"  [Vision] Page {page_num} → '{node.get('title', '')}'")
            except Exception as e:
                print(f"  [Vision] Page {page_num}: failed ({e})")

        doc.close()
        print(f"  [Vision] Described figures on {described}/{len(pages_with_figures)} pages")
        return tree

    # ── Entry point ────────────────────────────────────────────────────────────

    def forward(self, pdf_path: str = None, text: str = None, force_pymupdf: bool = False) -> dspy.Prediction:
        if text and not force_pymupdf:
            print("  [LocalPageIndex] Using provided text directly")
            ctx = dspy.context(lm=self._fast_lm) if self._fast_lm else nullcontext()
            with ctx:
                return self.section_extractor(article_text=text)
        if not pdf_path:
            raise ValueError("pdf_path or text is required")

        if force_pymupdf:
            print("  [LocalPageIndex] Forced PyMuPDF extraction (validation retry)")
            tree = self._extract_text_pymupdf(pdf_path)
            return self._process_tree(tree, label="pymupdf-forced")

        tree = self._load_cache(pdf_path)
        if tree is None:
            if self.pdf_parser == "pymupdf":
                tree = self._extract_text_pymupdf(pdf_path)
            else:
                print(f"  [LocalPageIndex] Processing: {pdf_path}")
                tree, error = _call_pageindex_with_timeout(pdf_path, self.model, self.api_key, self.api_base)
                if tree is None:
                    print(f"  [LocalPageIndex] {error}, falling back to PyMuPDF")
                    tree = self._extract_text_pymupdf(pdf_path)
                else:
                    try:
                        tree = self._describe_figures_with_vision(pdf_path, tree)
                    except Exception as e:
                        print(f"  [Vision] failed ({e}), skipping figure descriptions")

            self._save_cache(pdf_path, tree)

        return self._process_tree(tree, label=self.pdf_parser)
