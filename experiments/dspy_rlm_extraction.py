"""
DspyRLMSectionExtractor — section extraction using dspy.RLM.

dspy.RLM runs a sandboxed Python REPL where the LM explores the markdown
document via code (regex, string slicing, etc.) instead of loading the
full text into the prompt at once.

"""

import dspy
from dspy import Prediction


class DspyRLMSectionExtractor(dspy.Module):
    """
    Extracts standard paper sections from markdown using dspy.RLM.

    The LM receives the full markdown as `context` and a per-section query.
    Inside the REPL sandbox it can write Python (regex, slicing, etc.) to
    locate and extract each section without loading the whole document into
    the prompt at once.

    Returns dspy.Prediction with fields:
        Abstract, Introduction, Related_Work, Methods, Experiments, Conclusion
    (same interface as the heuristic RLMSectionExtractor in extraction.py)
    """

    SECTIONS = [
        "Abstract",
        "Introduction",
        "Related_Work",
        "Methods",
        "Experiments",
        "Conclusion",
    ]

    # Per-section instructions passed as `section_query` to the RLM.
    SECTION_QUERIES = {
        "Abstract": (
            "Extract the full Abstract section verbatim. "
            "Headings may be numbered, e.g. '## Abstract', '## ABSTRACT'. "
            "Return only the section body — no heading line, no other sections."
        ),
        "Introduction": (
            "Extract the full Introduction section verbatim. "
            "The heading may be numbered, e.g. '## 1 INTRODUCTION' or '## 1. Introduction'. "
            "Stop before the next top-level section (Related Work, Methods, etc.). "
            "Return only the section body — no heading line."
        ),
        "Related_Work": (
            "Extract the full Related Work section verbatim. "
            "The heading may be numbered, e.g. '## 2 RELATED WORK', '## 3 Related Literature'. "
            "It may also be called 'Prior Work', 'Background', or 'Literature Review'. "
            "Stop before the next top-level section. "
            "Exclude the References/Bibliography list. "
            "Return only the section body — no heading line."
        ),
        "Methods": (
            "Extract the full Methods / Methodology section verbatim. "
            "The heading may be numbered, e.g. '## 3 METHOD', '## 4 OUR APPROACH'. "
            "It may also be called 'Approach', 'Framework', 'Architecture', 'Model', "
            "'Proposed Method', 'Algorithm', 'Problem Statement', or 'Preliminaries'. "
            "CRITICAL: preserve all LaTeX equations, algorithm blocks, and technical definitions. "
            "Stop before the Experiments/Results section. "
            "Return only the section body — no heading line."
        ),
        "Experiments": (
            "Extract the full Experiments / Evaluation section verbatim, including all subsections. "
            "The heading may be numbered, e.g. '## 4 EXPERIMENTS', '## 5 EVALUATION'. "
            "It may also be called 'Results', 'Benchmarks', 'Empirical Study', or 'Ablation Study'. "
            "Include all subsections (Setup, Results, Ablation, Analysis). "
            "Stop before Conclusion or Discussion. "
            "Preserve all tables, metric results, and hardware setups. "
            "Return only the section body — no heading line."
        ),
        "Conclusion": (
            "Extract the full Conclusion section verbatim. "
            "The heading may be numbered, e.g. '## 6 CONCLUSION', '## 5 SUMMARY AND CONCLUSION'. "
            "It may also be called 'Discussion', 'Summary', 'Concluding Remarks', or 'Future Work'. "
            "Include all remaining content until References. "
            "Return only the section body — no heading line."
        ),
    }

    # Signature: the LM gets `context` (full markdown) as a Python variable in
    # the REPL sandbox, plus `section_query` explaining what to extract.
    _SIGNATURE = "context, section_query -> section_text"

    def __init__(self, max_iterations: int = 5):
        super().__init__()
        self.rlm = dspy.RLM(self._SIGNATURE, max_iterations=max_iterations)

    def forward(self, full_text: str) -> Prediction:
        """
        Extract all sections from markdown text.

        Args:
            full_text: Full paper text in markdown format.

        Returns:
            dspy.Prediction with Abstract, Introduction, Related_Work,
            Methods, Experiments, Conclusion fields.
        """
        extracted: dict[str, str] = {}

        for section in self.SECTIONS:
            for attempt in range(2):
                try:
                    pred = self.rlm(
                        context=full_text,
                        section_query=self.SECTION_QUERIES[section],
                    )
                    text = getattr(pred, "section_text", "") or ""
                    extracted[section] = text.strip()
                    break
                except Exception as exc:
                    if attempt == 0:
                        print(f"  [DspyRLM] {section} retry after error: {exc}")
                    else:
                        print(f"  [DspyRLM] {section} extraction failed: {exc}")
                        extracted[section] = ""

        return Prediction(**extracted)
