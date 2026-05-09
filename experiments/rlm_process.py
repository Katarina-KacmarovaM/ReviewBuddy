"""PageIndex-based section extraction pipeline (v1) — early experiment using the PageIndex client."""

import dspy
import json
import os
import re
import time
from pydantic import BaseModel, Field
from typing import Dict, Optional, List

try:
    from pageindex import PageIndexClient
    import pageindex.utils as pageindex_utils
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PAGEINDEX_AVAILABLE = False

# -------------------------------------------------------------------------

# PRIDAT DETEKCIU PROMPT INJECTION - NOVA SIGNATURE ASI !! 
# UROBIT VIAC AGENTIC - PROTOCOL MULTI AGENT + JEDEN CENTRALNY COORDINATOR
# PRIDAT USER PROMPT !! - INPUT FIELD



class ContextTools:
    def __init__(self, full_text):
        self.context = full_text

    def search_headers(self):
        # Regex pre formát: ## ABSTRACT, ## 1 INTRODUCTION, ## 2.1 SUBSECTION, etc.
        pattern = r"^## (.+)$"
        matches = []
        for match in re.finditer(pattern, self.context, re.MULTILINE):
            title = match.group(1).strip()
            # Generate slugified ID from title: "1 INTRODUCTION" -> "1_introduction"
            slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            matches.append({
                "title": title,
                "id": slug,
                "start_index": match.start()
            })
        return matches

    def get_text_slice(self, start_idx, end_idx):
        """Vráti konkrétny výsek textu na základe indexov."""
        return self.context[start_idx:end_idx]

    def peek(self, start, length=500):
        """Preview a chunk of text starting at `start` for `length` characters."""
        end = min(start + length, len(self.context))
        return self.context[start:end]

    def get_length(self):
        """Vráti celkový počet znakov/tokenov."""
        return len(self.context)
    

class SectionBounds(BaseModel):
    start: int
    end: int

class NavigatorOutput(BaseModel):
    thought: str = Field(description="Reasoning about the document structure.")
    action: str = Field(description="search_headers, peek, get_length, or FINISH")
    action_input: Dict = Field(default_factory=dict)
    # Pridáme default=None, aby Pydantic nenadával, keď agent ešte len hľadá
    section_mapping: Optional[Dict[str, SectionBounds]] = Field(default=None)

# class RLMNavigator(dspy.Signature):
#     """
#     You are an RLM (Recursive Language Model) navigator agent.
#     Your job is to explore a scientific document step-by-step using tools,
#     then produce a JSON mapping of standard sections to character-index ranges.

#     Standard sections: Abstract, Introduction,Related Work, Methods, Experiments, Conclusion.

#     Strategy:
#     1. Call search_headers to discover all section headings and their positions.
#     2. Use peek to preview text around ambiguous boundaries.
#     3. When you have enough information, output action="FINISH" with a section_mapping JSON.

#     The section_mapping must be a JSON object like:
#     {"Abstract": {"start": 0, "end": 500}, "Introduction": {"start": 500, "end": 3000}, ...}
#     Use character indices from the tools. If a section is not found, omit it.

#     Args:
#         -instruction: The task to accomplish.
#         -available_tools: Description of available tools and their parameters.
#         -history: Log of previous tool calls and their results. Empty string on first step.
#     Outputs:
#         -thought: Your reasoning about what to do next.
#         -action: Tool to call: 'search_headers', 'peek', 'get_text_slice', 'get_length', or 'FINISH'.
#         -action_input: JSON parameters for the tool (e.g. {"start": 0, "length": 500}). Empty '{}' for no-arg tools. Ignored when action=FINISH.
#         -section_mapping: ONLY when action=FINISH: JSON mapping of section names to {start, end} indices. Empty string otherwise.
#     """
#     instruction = dspy.InputField(description="The task to accomplish.")
#     available_tools = dspy.InputField(description="Description of available tools and their parameters.")
#     history = dspy.InputField(description="Log of previous tool calls and their results. Empty string on first step.")

#     thought = dspy.OutputField(description="Your reasoning about what to do next.")
#     action = dspy.OutputField(description="Tool to call: 'search_headers', 'peek', 'get_text_slice', 'get_length', or 'FINISH'.")
#     action_input = dspy.OutputField(description="JSON parameters for the tool (e.g. {\"start\": 0, \"length\": 500}). Empty '{}' for no-arg tools. Ignored when action=FINISH.")
#     section_mapping = dspy.OutputField(description="ONLY when action=FINISH: JSON mapping of section names to {start, end} indices. Empty string otherwise.")

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
    
    # Rozdelíme výstup na samostatné polia - DSPy to lepšie zvláda
    thought = dspy.OutputField(description="Reasoning about what to do next.")
    action = dspy.OutputField(description="One of: search_headers, peek, get_length, FINISH.")
    action_input = dspy.OutputField(description="JSON params, e.g. {'start': 0}. Empty {} if none.")
    section_mapping = dspy.OutputField(description="JSON mapping ONLY if action is FINISH. E.g. {'Abstract': {'start':0, 'end':100}}.")

class SectionExtractor(dspy.Signature):
    """
    Extract specific sections from a scientific article provided in markdown.
    
    RULES:
    1. DO NOT summarize. Extract the text including all subheadings (e.g., ## 2.1, ###).
    2. INCLUDE all image descriptions found within the tags [ POPIS OBRÁZKA] that belong to the section.
    3. IGNORE page markers like [--- START PAGE X ---], but extract all text between them.
    4. PRESERVE markdown formatting (bold, tables, LaTeX).
    5. ADAPT to the paper structure: Titles vary. Look for the CONTENT type described below.

    Args:
        - full text (article): The entire article text in markdown format.
    Outputs:
        - Abstract: Content under the Abstract heading.
        - Introduction: Content from the Introduction heading up to the start of the technical/background sections.
        - Related Work: Content under the Related Work heading, if it exists. This section often appears after Introduction and before Methods, but can vary.Section is usually named 'Related Work', 'Background', 'Literature Review' or similar. It contains discussion of prior research and how the current work differs.
        - Methods: The core technical contribution. This typically includes sections named 'Methodology', 'Proposed Method', 'Framework', 'Preliminaries', or specific technical titles (e.g., 'Stabilizing...', 'Simplifying...'). It contains mathematical definitions, algorithms, and architectural details. It usually sits between Introduction and Experiments.
        - Experiments: The empirical evaluation. Section is usually named 'Experiments', 'Results', 'Evaluation', 'Benchmarks', 'Ablation Study' or 'Scaling Up'. Includes tables, metrics (FID, accuracy), and comparisons.
        - Conclusion: Content under 'Conclusion', 'Discussion', or 'Limitations'.
    
    """


    article_text: str = dspy.InputField(description="Full text of the article in markdown format.")
    
    Abstract: str = dspy.OutputField(description="Content under the Abstract heading.")
    
    Introduction: str = dspy.OutputField(description="Content from the Introduction heading up to the start of the technical/background sections.")

    Related_Work: str = dspy.OutputField(description="Content under the Related Work heading, if it exists. This section often appears after Introduction and before Methods, but can vary.Section is usually named 'Related Work', 'Background', 'Literature Review' or similar. It contains discussion of prior research and how the current work differs.")
    
    Methods: str = dspy.OutputField(description="The core technical contribution. This typically includes sections named 'Methodology', 'Proposed Method', 'Framework', 'Preliminaries', or specific technical titles (e.g., 'Stabilizing...', 'Simplifying...'). It contains mathematical definitions, algorithms, and architectural details. It usually sits between Introduction and Experiments.")
    
    Experiments: str = dspy.OutputField(description="The empirical evaluation. Section is usually named 'Experiments', 'Results', 'Evaluation', 'Benchmarks', 'Ablation Study' or 'Scaling Up'. Includes tables, metrics (FID, accuracy), and comparisons.")
    
    Conclusion: str = dspy.OutputField(description="Content under 'Conclusion', 'Discussion', or 'Limitations'.")

class SectionSingleExtractor(dspy.Signature):
    """Extract a SPECIFIC section from a markdown paper without summarizing."""
    # Vstupy
    article_text = dspy.InputField(description="The content text fragment.")
    section_name = dspy.InputField(description="e.g., 'Methods', 'Abstract'")
    specific_instructions = dspy.InputField(description="Specific rules for this section type.")
    
    # Výstupy
    extracted_content = dspy.OutputField(description="Clean extracted text.")


class CAReview(dspy.Signature):
    """
    You are ICLR reviewer and your task is to evaluate the contribution and novelty of the article and give a detailed critique with numerical score for novelty.
    
    This is what you should look for when evaluating the contribution and novelty:
        1. MOTIVATION: Does the paper address a REAL and SIGNIFICANT bottleneck?
        - If it solves a "strawman" (fake) problem or lacks justification ("Why do we need this?"), mark it as WEAK MOTIVATION.
        - If it is solid, new method that addresses a real issue, with real impact, it's a STRENGTH.
        - If it open a new research direction,has strong theoretical insights , it is a STRENGTH.
        
        2. NOVELTY TYPE:
        - POSITIVE: "Root Cause Analysis" (explains WHY something fails), "Unification" (connects disjoint theories), or "Simpson's Paradox" (reveals a counter-intuitive truth).
        - NEGATIVE: "Incremental" (just another loss term), "Application" (applying existing method to new data without insight), or "Complexity without Gain".
        
        3. COMPLEXITY CHECK: 
        - If the math is dense but the gain over simple baselines is marginal -> MAJOR WEAKNESS.
        - If the method is complex but provides a significant leap in performance or understanding -> STRENGTH.

    Based on the above criteria, provide a detailed,contribution assesment review with explicit Strengths and Weaknesses sections.

    Args:
        - introduction_section: The Introduction section text, which contains the motivation and problem statement.
        - related_work_section: The Related Work section text, which contains the discussion of prior research and how the current work differs. This can provide additional context for assessing novelty.
        - conclusion_section: The Conclusion section text, which contains the authors' own summary of their contribution and its significance.
    Outputs:
        - review: Detailed review focusing on novelty, significance, and motivation vs. gain trade-off. It contains weaknesses and strengths of the contribution, and suggestions for improvement.


    """
    introduction_section: str = dspy.InputField(description="Introduction text: It contains the motivation for the work, the problem statement, and a high-level overview of the contribution.")
    related_work_section: str = dspy.InputField(description="Related Work text: It contains the discussion of prior research and how the current work differs. This can provide additional context for assessing novelty.")
    conclusion_section: str = dspy.InputField(description="Conclusion text: It contains the authors' own summary of their contribution and its significance.")
    review: str = dspy.OutputField(description="Detailed review focusing on novelty, significance, and motivation vs. gain trade-off. It contains weaknesses and strengths of the contribution, and suggestions for improvement.")


class MethodologyReview(dspy.Signature):
    """
    You are ICLR reviewer and your task is to evaluate the soundness of the methodology and experiments and give a detailed critique with numerical score for soundness.

    This is what you should look for when evaluating the methodology and experiments:

        1. DATASETS: Are the datasets appropriate for the claims?
            Large-scale benchmarks = strength. Only small/toy datasets = weakness.
        2. BASELINES: Are comparisons fair and recent?
            Strong recent baselines = strength. Missing key baselines = weakness.
        3. ABLATION: Is each component's contribution isolated?
            Comprehensive ablation = strength. No ablation = weakness.
        4. REPRODUCIBILITY: Enough detail to reimplement?
        5. RESULTS: Are improvements substantial or marginal/within noise?
    
    Based on the above criteria, provide a detailed, methodology review with explicit Strengths and Weaknesses sections.

    Args:
        - methods_section: The Methods section text, which contains the technical contribution and experimental setup.
        - experiments_section: The Experiments section text, which contains the empirical evaluation and results.
    Outputs:
        - review: Detailed review focusing on methodology and experiments with mentioned strengths and weaknesses of the methodology or experiments, and suggestions for improvement.

    """
    methods_section: str = dspy.InputField(description="Methods section text.")
    experiments_section: str = dspy.InputField(description="Experiments section text.")
    review: str = dspy.OutputField(description="Detailed review focusing on methodology and experiments with mentioned strengths and weaknesses of the methodology or experiments, and suggestions for improvement.")



class ClarityReview(dspy.Signature):
    """
    You are ICLR reviewer and your task is to evaluate the clarity of the article and give a detailed critique with numerical score for clarity.
    Review the clarity and presentation quality of the article.

    This is what good paper clarity looks like:
    - Clear problem formulation and well-stated contributions.
    - Good paper structure and logical flow.
    - Variables defined before use, intuition before formalism.
    - Self-explanatory figures and tables.

    This is what poor paper clarity looks like:
    - Unclear problem definition or missing motivation.
    - Poor organization or hard-to-follow arguments.
    - Undefined notation or inconsistent symbols.
    - Figures/tables that require text to understand.

    Based on the above criteria, provide a detailed, clarity review with explicit Strengths and Weaknesses sections


    Args:
        - article_text: The full article text in markdown format, which you can refer to for specific examples of clarity issues or strengths.
    Outputs:ions
        - review: Detailed review focusing on clarity and presentation with mentioned strengths and  weaknesses of the clarity, and suggest for improvement.
    """
    article_text: str = dspy.InputField(description="Full article text in markdown.")
    review: str = dspy.OutputField(description="Detailed review focusing on clarity and presentation with mentioned strengths and  weaknesses of the clarity, and suggestions for improvement.")


class SummaryGenerator(dspy.Signature):
    """
    You are an expert reviewer tasked with synthesizing the insights from three specialist reviews (contribution, methodology, clarity) into a final summary that will inform the Area Chair's decision.
    Synthesize three specialist reviews into a final summary.
    Structure: Strengths, Weaknesses, Suggestions for each part.
    Be specific - cite concrete issues from the reviews.

    Args:
        - ca_review: Contribution assessment review.
        - methodology_review: Methodology review.
        - clarity_review: Clarity review.
    Output:
        - summary: Structured summary with Strengths, Weaknesses, and Suggestions sections.
    """
    ca_review: str = dspy.InputField(description="Contribution assessment review.")
    methodology_review: str = dspy.InputField(description="Methodology review.")
    clarity_review: str = dspy.InputField(description="Clarity review.")
    summary: str = dspy.OutputField(description="Structured summary with Strengths, Weaknesses, and Suggestions sections.")


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
    
    ca_review: str = dspy.InputField(description="Contribution assessment review with weaknesses and strengths.")
    methodology_review: str = dspy.InputField(description="Methodology review with weaknesses and strengths.")
    clarity_review: str = dspy.InputField(description="Clarity review with strengths and weaknesses.")
    #summary: str = dspy.InputField(description="Review summary with weaknesses and strengths.")
    final_justification: str = dspy.OutputField(description="Brief justification.")
    score: float = dspy.OutputField(description="Score from 1 to 10.")
    decision: str = dspy.OutputField(description="'Accept' or 'Reject'.")



class LLMJudge(dspy.Signature):
    """
    You are an expert strict reviewer. Compare the generated review with the human review.
    Rate the accuracy and significance on a scale from 1 to 10.
    """
    generated_review: str = dspy.InputField(description="Generated review text by the LLM.")
    human_review: str = dspy.InputField(description="Human-written review text to compare against.")
    evaluation: float = dspy.OutputField(description="LLM-based evaluation score (1-10).")



# class RecursiveSectionExtractor(dspy.Module):
#     def __init__(self):
#         super().__init__()
#         self.navigator = dspy.Predict(RLMNavigator)
#         #self.child_reviewer = dspy.Predict(SectionExtractor)

#     def _parse_action(self, raw_action, raw_action_input):
#         """Parse action and params, handling messy LLM outputs like 'peek(168, 500)' or 'search_headers()'."""
#         action = raw_action.strip()
#         params = {}

#         # Try to parse action_input as JSON first
#         try:
#             if raw_action_input and raw_action_input.strip() and raw_action_input.strip() != "{}":
#                 params = json.loads(raw_action_input)
#         except json.JSONDecodeError:
#             pass

#         # Handle function-call style: "peek(168, 500)" or "search_headers()"
#         func_match = re.match(r"(\w+)\((.*)\)", action)
#         if func_match:
#             action = func_match.group(1)
#             args_str = func_match.group(2).strip()
#             if args_str and not params:
#                 # Try JSON first: peek({"start": 168, "length": 500})
#                 try:
#                     params = json.loads(args_str)
#                 except json.JSONDecodeError:
#                     # Try positional args: peek(168, 500)
#                     parts = [p.strip() for p in args_str.split(",")]
#                     action_lower = action.strip().lower()
#                     if action_lower == "peek" and len(parts) >= 1:
#                         params["start"] = int(parts[0])
#                         if len(parts) >= 2:
#                             params["length"] = int(parts[1])
#                     elif action_lower == "get_text_slice" and len(parts) >= 2:
#                         params["start"] = int(parts[0])
#                         params["end"] = int(parts[1])

#         return action.strip().lower(), params

#     def _execute_tool(self, tools, action, action_input):
#         """Execute a tool call from the Navigator and return the result string."""
#         action_name, params = self._parse_action(action, action_input)

#         if action_name == "search_headers":
#             result = tools.search_headers()
#             return json.dumps(result, ensure_ascii=False)
#         elif action_name == "peek":
#             start = int(params.get("start", 0))
#             length = int(params.get("length", 500))
#             return tools.peek(start, length)
#         elif action_name == "get_text_slice":
#             start = int(params.get("start", 0))
#             end = int(params.get("end", start + 1000))
#             return tools.get_text_slice(start, end)
#         elif action_name == "get_length":
#             return str(tools.get_length())
#         else:
#             return f"ERROR: Unknown tool '{action}'. Use: search_headers, peek, get_text_slice, get_length."


        

#     def forward(self, full_text, task="Extract all sections"):
    
#         # 1. PRÍPRAVA NÁSTROJOV
#         tools = ContextTools(full_text)

#         available_tools_desc = (
#             "search_headers() -> returns JSON list of {title, id, start_index} for all ## headings.\n"
#             "peek(start, length) -> returns `length` characters of text starting at `start`.\n"
#             "get_text_slice(start, end) -> returns text between character indices start..end.\n"
#             "get_length() -> returns total character count of the document.\n"
#             "FINISH -> end exploration; you MUST provide section_mapping JSON."
#         )

#         instruction = (
#             "Map the document's sections to standard academic sections: "
#             "Abstract, Introduction,Related Work, Methods, Experiments, Conclusion. "
#             "Use search_headers first to discover headings. Then use peek if boundaries are unclear. "
#             "Related Work includes: Related Work, Background, Literature Review."
#             "Methods includes: Methodology, Proposed Method, Framework, Preliminaries, or any core technical section. "
#             "Experiments includes: Results, Evaluation, Benchmarks, Ablation Study. "
#             "Conclusion includes: Discussion, Limitations. "
#             "When ready, set action=FINISH and provide section_mapping JSON with {start, end} character indices."
#         )

#         # 2. ITERATÍVNY REPL CYKLUS (RLM pattern)
#         history = ""
#         max_steps = 4
#         mapping = None

#         for step in range(max_steps):
#             # FIX A: Na poslednom kroku vynútiť FINISH
#             step_instruction = instruction
#             if step == max_steps - 1:
#                 step_instruction += (
#                     " THIS IS YOUR LAST STEP. You MUST output action=FINISH "
#                     "and provide section_mapping JSON NOW."
#                 )

#             step_history = history if history else "(no previous actions)"
#             step_history += f"\n[Budget: step {step+1} of {max_steps}]"

#             nav_result = self.navigator(
#                 instruction=step_instruction,
#                 available_tools=available_tools_desc,
#                 history=step_history
#             )

#             raw_action = nav_result.action.strip()
#             # Normalize: strip parens, quotes, whitespace, check for FINISH
#             action_clean = re.sub(r"[\"'()\s]", "", raw_action).strip().upper()
#             print(f"  [RLM Step {step+1}] thought={nav_result.thought[:80]}... action={raw_action}")

#             # Check if Navigator decided to finish
#             if action_clean == "FINISH":
#                 mapping = self._parse_mapping(nav_result.section_mapping)
#                 if mapping:
#                     print(f"  [RLM] Navigator finished with mapping: {list(mapping.keys())}")
#                 break

#             # FIX C: Detekovať JSON halucináciu v action poli
#             mapping = self._parse_mapping(raw_action)
#             if mapping:
#                 print(f"  [RLM] Detected mapping in action field: {list(mapping.keys())}")
#                 break

#             # FIX C: Skontrolovať aj section_mapping na každom kroku
#             if hasattr(nav_result, 'section_mapping') and nav_result.section_mapping.strip():
#                 mapping = self._parse_mapping(nav_result.section_mapping)
#                 if mapping:
#                     print(f"  [RLM] Detected early mapping in section_mapping field: {list(mapping.keys())}")
#                     break

#             # Execute the requested tool
#             tool_output = self._execute_tool(tools, raw_action, nav_result.action_input)

#             # Append to history (REPL pattern - agent sees accumulated context)
#             truncated_output = tool_output[:2000]
#             history += f"\n[Step {step+1}] Action: {raw_action}({nav_result.action_input}) -> {truncated_output}"

#         # 3. FALLBACK: ak Navigator nedal mapping, použijeme heuristiku z headerov
#         if mapping is None:
#             print("  [RLM] Fallback: Navigator did not produce mapping, using header-based heuristic.")
#             found_headers = tools.search_headers()
#             mapping = self._fallback_mapping(found_headers, len(full_text))

#         # 4. EXTRAKCIA SEKCIÍ Z MAPPINGU
#         extracted_sections = {}
#         standard_sections = ["Abstract", "Introduction", "Related_Work", "Methods", "Experiments", "Conclusion"]

#         for section in standard_sections:
#             if section in mapping:
#                 bounds = mapping[section]
#                 start = int(bounds.get("start", 0))
#                 end = int(bounds.get("end", start))
#                 content = tools.get_text_slice(start, end)

#                 # RLM REKURZIA: Pre príliš dlhé sekcie delegujeme na child agenta
#                 if len(content) > 12000:
#                     print(f"  --- RLM REKURZIA: Sekcia {section} je príliš dlhá ({len(content)} zn.), volám pod-agenta ---")
#                     recursive_pred = self.child_reviewer(article_text=content)
#                     extracted_sections[section] = getattr(recursive_pred, section, content)
#                 else:
#                     extracted_sections[section] = content
#             else:
#                 extracted_sections[section] = "Section not found in document."

#         # 5. FINÁLNA SYNTÉZA
#         return dspy.Prediction(
#             Abstract=extracted_sections.get("Abstract", ""),
#             Introduction=extracted_sections.get("Introduction", ""),
#             Related_Work=extracted_sections.get("Related_Work", ""),
#             Methods=extracted_sections.get("Methods", ""),
#             Experiments=extracted_sections.get("Experiments", ""),
#             Conclusion=extracted_sections.get("Conclusion", "")
#         )


#     def _fallback_mapping(self, headers, doc_length):
#         """Heuristic mapping when Navigator fails to produce one."""
#         mapping = {}
#         section_keywords = {
#             "Abstract": ["abstract"],
#             "Introduction": ["introduction"],
#             "Related_Work": ["related work", "background", "literature review"],
#             "Methods": ["method", "approach", "framework", "preliminar", "proposed", "model", "architecture"],
#             "Experiments": ["experiment", "result", "evaluation", "benchmark", "ablation"],
#             "Conclusion": ["conclusion", "discussion", "limitation", "summary"]
#         }

#         for section, keywords in section_keywords.items():
#             for h in headers:
#                 title_lower = h["title"].lower()
#                 if any(kw in title_lower for kw in keywords):
#                     start_idx = h["start_index"]
#                     # End = next header's start or end of document
#                     h_idx = headers.index(h)
#                     end_idx = headers[h_idx + 1]["start_index"] if h_idx + 1 < len(headers) else doc_length

#                     # For Methods/Experiments, extend to include subsections
#                     if section in ("Methods", "Experiments"):
#                         for subsequent in headers[h_idx + 1:]:
#                             sub_lower = subsequent["title"].lower()
#                             if any(kw in sub_lower for kw in keywords):
#                                 s_idx = headers.index(subsequent)
#                                 end_idx = headers[s_idx + 1]["start_index"] if s_idx + 1 < len(headers) else doc_length
#                             else:
#                                 break

#                     mapping[section] = {"start": start_idx, "end": end_idx}
#                     break

#         return mapping


class RecursiveSectionExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Predict stačí — RLMNavigator už má vlastné 'thought' pole
        self.navigator = dspy.Predict(RLMNavigator)

    def _robust_json_parse(self, raw_text):
        """Bezpečné parsovanie JSONu (či už je to dict alebo string)."""
        if not raw_text: return None
        if isinstance(raw_text, dict): return raw_text
        
        text = str(raw_text).strip()
        # Odstránenie markdown blokov ```json ... ```
        text = re.sub(r"^```json", "", text).strip()
        text = re.sub(r"^```", "", text).strip()
        text = text.strip("`")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Skúsime nájsť {} blok
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return None

    def _execute_tool(self, tools, action, action_input):
        act_name = str(action).strip().lower()
        
        # Parsovanie inputu
        params = self._robust_json_parse(action_input)
        if not params: params = {}

        if "search_headers" in act_name:
            return json.dumps(tools.search_headers(), ensure_ascii=False)
        elif "peek" in act_name:
            start = int(params.get("start", 0))
            length = int(params.get("length", 1000)) # Zväčšíme okno pre lepší kontext
            return tools.peek(start, length)
        elif "get_length" in act_name:
            return str(tools.get_length())
            
        return f"Error: Unknown tool '{act_name}'"

    def _fallback_mapping(self, headers, doc_length):
        """Heuristika: nájdi štart každej štandardnej sekcie, koniec = štart nasledujúcej."""
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

        # 1. Nájdi PRVÝ header pre každú štandardnú sekciu
        section_starts = {}  # section_name -> start_index
        for section in section_order:
            keywords = section_keywords[section]
            for h in sorted_headers:
                title_lower = h["title"].lower()
                if any(kw in title_lower for kw in keywords):
                    section_starts[section] = h["start_index"]
                    break

        # 2. Zoraď nájdené sekcie podľa pozície v dokumente
        found = sorted(section_starts.items(), key=lambda x: x[1])

        # 3. Koniec každej sekcie = začiatok nasledujúcej nájdenej sekcie
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
        
        # Zjednodušený popis nástrojov pre LLM
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
            # Dynamická inštrukcia
            current_instruction = "Find indices for: Abstract, Intro, Related Work, Methods, Experiments, Conclusion."
            if step == 0:
                current_instruction += " Start by calling 'search_headers'."
            
            pred = self.navigator(
                instruction=current_instruction,
                available_tools=available_tools_desc,
                history=history if history else "Start."
            )
            
            # Bezpečné vytiahnutie polí
            thought = getattr(pred, 'thought', '')
            action = getattr(pred, 'action', '')
            action_input = getattr(pred, 'action_input', '{}')
            
            print(f" [Step {step+1}] Thought: {thought[:50]}... -> Action: {action}")

            # Kontrola FINISH
            if "FINISH" in str(action).upper():
                mapping_str = getattr(pred, 'section_mapping', '{}')
                parsed = self._robust_json_parse(mapping_str)
                if parsed and len(parsed) > 0:
                    final_mapping = parsed
                    break
                else:
                    print("  [RLM] FINISH called but mapping invalid. Retrying...")
            
            # Vykonanie nástroja
            tool_res = self._execute_tool(tools, action, action_input)
            
            # Skrátenie histórie aby sa nezaplnil kontext
            history += f"\n[Step {step+1}]\nThought: {thought}\nAction: {action}\nInput: {action_input}\nResult: {tool_res[:600]}..."

        if not final_mapping:
            print(" [RLM] Fallback to heuristics.")
            final_mapping = self._fallback_mapping(tools.search_headers(), len(full_text))
            
        return final_mapping



class RobustSectionExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Použijeme Predict alebo ChainOfThought pre lepšiu kvalitu
        self.extractor = dspy.Predict(SectionSingleExtractor)
        
        # Špecifické inštrukcie pre každý typ obsahu
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

        # Validácia mappingu
        if not mapping:
             mapping = {}

        for section_name in ["Abstract", "Introduction", "Related_Work", "Methods", "Experiments", "Conclusion"]:
            bounds = mapping.get(section_name)

            # Fallback ak sekcia chýba v mappingu
            if not bounds:
                extracted_sections[section_name] = ""
                continue

            start = int(bounds.get("start") or 0)
            end = int(bounds.get("end") or start)

            # Priame vyrezanie textu — žiadne ďalšie LLM volanie
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
                    # API has too many copies of this filename — upload under unique name
                    import shutil, tempfile
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
        while self.pi_client.get_document(doc_id)["status"] != "completed":
            time.sleep(3)
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

    def get_full_text(self, pdf_path):
        """Get concatenated full text from all nodes (for clarity review)."""
        data = self._upload_and_process(pdf_path)
        node_map = data['node_map']
        all_texts = []
        for node_id, node in node_map.items():
            text = node.get('text', '')
            if text:
                all_texts.append(text)
        return "\n\n".join(all_texts)


# -------------------------------------------------------------------------
# GEPA: Review Pipeline Module
# -------------------------------------------------------------------------

class ReviewPipeline(dspy.Module):
    """
    Enhanced pipeline with validation tracking.

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
        self.decision_predictor = dspy.ChainOfThought(DecisionPredictor)

    def forward(self, article_text=None, pdf_path=None):
        # Extract sections with error handling


        try:
            if self.use_pageindex:
                if not pdf_path:
                    raise ValueError("pdf_path is required when use_pageindex=True")
                structure = self.pageindex_extractor(pdf_path=pdf_path)
                # Build full text for clarity review if not provided
                if not article_text:
                    article_text = "\n\n".join(filter(None, [
                        structure.Abstract, structure.Introduction,
                        structure.Methods, structure.Experiments,
                        structure.Conclusion
                    ]))
            elif self.use_rlm:
                mapping = self.navigator(full_text=article_text)
                structure = self.section_extractor(full_text=article_text, mapping=mapping)
            else:
                structure = self.section_extractor(article_text=article_text)
        except Exception as e:
            print(f"Section extraction failed: {e}")
            # Return dummy prediction to avoid crashes
            return dspy.Prediction(
                decision="Reject",
                score=1.0,
                justification="Failed to extract sections",
                summary="Error in processing"
            )


        # 2. Sekvenčné volanie špecialistov
        ca_result = self.ca_reviewer(introduction_section=structure.Introduction, related_work_section=structure.Related_Work, conclusion_section=structure.Conclusion)
        meth_result = self.methodology_reviewer(methods_section=structure.Methods, experiments_section=structure.Experiments)
        clarity_result = self.clarity_reviewer(article_text=article_text)

        # 3. Syntéza výsledkov
        final_summary = self.summary_generator(
            ca_review=ca_result.review,
            methodology_review=meth_result.review,
            clarity_review=clarity_result.review
        )

        # 4. Finálne rozhodnutie Area Chaira
        prediction = self.decision_predictor(
            ca_review=ca_result.review,
            methodology_review=meth_result.review,
            clarity_review=clarity_result.review,
            #summary=final_summary.summary
        )

        return dspy.Prediction(
            decision=prediction.decision,
            score=prediction.score,
            justification=prediction.final_justification,
            summary=final_summary.summary,
            ca_review=ca_result.review,
            methodology_review=meth_result.review,
            sections=structure,  
            clarity_review=clarity_result.review
        )
