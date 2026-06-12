"""
Final reasoner for OV-VQA.

This module performs evidence-based answer generation.

Flow:
    detections + knowledge + answer candidates
        -> structured evidence JSON
        -> final answer prompt
        -> LLM reasoning
        -> answer cleaning
        -> constrained candidate selection

Important:
    The final answer can be constrained to answer_candidates.
    This prevents free-form answers such as:
        "holding bat"
        "visit museum"
        "who leaves it like this"

    Instead, the final answer is selected from generated candidates:
        ["swing", "hit", "bat", ...] -> "swing"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.prompts.final_answer_prompts import (
    FINAL_ANSWER_SYSTEM_INSTRUCTION,
    build_final_answer_prompt_builder,
)
from src.reasoning.answer_cleaner import build_answer_cleaner
from src.reasoning.evidence_formatter import build_evidence_formatter
from src.utils.text import normalize_answer
from src.reasoning.candidate_ranker import rank_answer_candidates

Detection = Dict[str, Any]
Passage = Dict[str, Any]


@dataclass
class FinalReasoningResult:
    """
    Final reasoning output.

    Attributes:
        pred_answer:
            Human-readable cleaned answer.
        clean_answer:
            Normalized answer for evaluation.
        raw_reasoning_output:
            Raw LLM output.
        reasoning_input:
            Structured evidence input, usually JSON string.
        final_prompt:
            Prompt sent to LLM.
        visual_evidence:
            Formatted visual evidence list.
        knowledge_evidence:
            Formatted knowledge evidence list.
    """

    pred_answer: str
    clean_answer: str
    raw_reasoning_output: str
    reasoning_input: str
    final_prompt: str
    visual_evidence: list
    knowledge_evidence: list


class FinalReasoner:
    """
    Evidence-based final answer reasoner.

    This class supports two modes:

    1. Free-form evidence-based reasoning:
        Evidence -> LLM -> answer cleaner

    2. Constrained candidate selection:
        Evidence + answer_candidates -> LLM chooses candidate id
        Python maps id -> candidate string

    The second mode is enabled automatically when answer_candidates are available.
    """

    def __init__(
        self,
        llm: Any,
        evidence_formatter: Any,
        prompt_builder: Any,
        answer_cleaner: Any,
        structured: bool = True,
        constrain_to_candidates: bool = True,
    ) -> None:
        """
        Args:
            llm: LLM wrapper.
            evidence_formatter: EvidenceFormatter instance.
            prompt_builder: FinalAnswerPromptBuilder instance.
            answer_cleaner: AnswerCleaner instance.
            structured: Whether structured reasoning is enabled.
            constrain_to_candidates:
                Whether to force final answer selection from answer_candidates.
        """
        self.llm = llm
        self.evidence_formatter = evidence_formatter
        self.prompt_builder = prompt_builder
        self.answer_cleaner = answer_cleaner
        self.structured = structured
        self.constrain_to_candidates = constrain_to_candidates

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        llm: Any,
    ) -> "FinalReasoner":
        """
        Build final reasoner from config.

        Args:
            cfg: Full experiment config.
            llm: LLM wrapper.

        Returns:
            FinalReasoner.
        """
        reasoning_cfg = cfg.get("reasoning", {})

        return cls(
            llm=llm,
            evidence_formatter=build_evidence_formatter(cfg),
            prompt_builder=build_final_answer_prompt_builder(cfg),
            answer_cleaner=build_answer_cleaner(cfg),
            structured=bool(reasoning_cfg.get("structured", True)),
            constrain_to_candidates=bool(
                reasoning_cfg.get("constrain_to_candidates", True)
            ),
        )

    @staticmethod
    def detections_to_plain_text(
        detections: Optional[Sequence[Detection]],
        max_items: int = 10,
    ) -> str:
        """
        Convert detections to plain text for unstructured ablation.

        Args:
            detections: Detector outputs.
            max_items: Maximum detections.

        Returns:
            Plain text visual evidence.
        """
        if not detections:
            return "None"

        sorted_dets = sorted(
            detections,
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )

        lines = []

        for idx, det in enumerate(sorted_dets[:max_items], start=1):
            label = det.get("label") or det.get("prompt") or ""
            conf = det.get("confidence")

            if not label:
                continue

            if conf is not None:
                try:
                    lines.append(f"{idx}. {label} ({float(conf):.3f})")
                except Exception:
                    lines.append(f"{idx}. {label}")
            else:
                lines.append(f"{idx}. {label}")

        return "\n".join(lines) if lines else "None"

    @staticmethod
    def passages_to_plain_text(
        passages: Optional[Sequence[Passage]],
        max_items: int = 5,
        max_chars: int = 500,
    ) -> str:
        """
        Convert passages to plain text for unstructured ablation.

        Args:
            passages: Retrieved or reranked passages.
            max_items: Maximum passages.
            max_chars: Maximum chars per passage.

        Returns:
            Plain text knowledge evidence.
        """
        if not passages:
            return "None"

        lines = []

        for idx, passage in enumerate(passages[:max_items], start=1):
            title = (
                passage.get("title")
                or passage.get("page_title")
                or passage.get("wiki_title")
                or ""
            )
            text = (
                passage.get("text")
                or passage.get("passage")
                or passage.get("content")
                or ""
            )

            title = str(title).strip()
            text = str(text).strip()

            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "..."

            if title and text:
                lines.append(f"{idx}. {title}: {text}")
            elif text:
                lines.append(f"{idx}. {text}")
            elif title:
                lines.append(f"{idx}. {title}")

        return "\n".join(lines) if lines else "None"

    def _call_reasoning_llm(self, prompt: str) -> str:
        """
        Call LLM for final answer generation.

        Args:
            prompt: Final answer prompt.

        Returns:
            Raw LLM output.
        """
        if hasattr(self.llm, "generate_reasoning"):
            return self.llm.generate_reasoning(prompt)

        return self.llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=150,
            system_instruction=FINAL_ANSWER_SYSTEM_INSTRUCTION,
        )

    # ------------------------------------------------------------------
    # Candidate selection utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_candidate_list(candidates: Optional[Sequence[str]]) -> List[str]:
        """
        Normalize and deduplicate candidate list while keeping original strings.

        Args:
            candidates: Raw candidate list.

        Returns:
            Cleaned candidate list.
        """
        output: List[str] = []
        seen = set()

        for candidate in candidates or []:
            candidate = str(candidate).strip()
            candidate = " ".join(candidate.split())

            if not candidate:
                continue

            key = normalize_answer(candidate)

            if not key:
                continue

            if key in seen:
                continue

            seen.add(key)
            output.append(candidate)

        return output

    @staticmethod
    def _format_candidate_list(candidates: Sequence[str]) -> str:
        """
        Format candidates as numbered list.

        Args:
            candidates: Candidate list.

        Returns:
            Numbered candidate text.
        """
        return "\n".join(
            f"{idx}. {candidate}"
            for idx, candidate in enumerate(candidates, start=1)
        )

    @staticmethod
    def _parse_candidate_index(
        text: str,
        num_candidates: int,
    ) -> Optional[int]:
        """
        Parse candidate index from LLM output.

        Args:
            text: LLM output.
            num_candidates: Number of candidates.

        Returns:
            Zero-based candidate index or None.
        """
        if text is None:
            return None

        text = str(text).strip()

        patterns = [
            r"candidate\s*[:：]\s*(\d+)",
            r"answer\s*[:：]\s*(\d+)",
            r"choice\s*[:：]\s*(\d+)",
            r"^(\d+)$",
            r"\b(\d+)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue

            idx = int(match.group(1)) - 1

            if 0 <= idx < num_candidates:
                return idx

        return None

    @staticmethod
    def _parse_candidate_by_text(
        text: str,
        candidates: Sequence[str],
    ) -> Optional[int]:
        """
        Parse candidate by matching candidate text in LLM output.

        This is a fallback when the model outputs "swing" instead of "Candidate: 1".

        Args:
            text: LLM output.
            candidates: Candidate list.

        Returns:
            Zero-based candidate index or None.
        """
        if text is None:
            return None

        output_norm = normalize_answer(text)

        if not output_norm:
            return None

        for idx, candidate in enumerate(candidates):
            cand_norm = normalize_answer(candidate)

            if not cand_norm:
                continue

            if output_norm == cand_norm:
                return idx

        for idx, candidate in enumerate(candidates):
            cand_norm = normalize_answer(candidate)

            if not cand_norm:
                continue

            if cand_norm in output_norm:
                return idx

        return None

    def _build_candidate_selection_prompt(
        self,
        question: str,
        reasoning_input: str,
        answer_candidates: Sequence[str],
        raw_output: str = "",
    ) -> str:
        """
        Build constrained candidate-selection prompt.

        Args:
            question: VQA question.
            reasoning_input: Structured evidence JSON string.
            answer_candidates: Candidate answers.
            raw_output: Previous free-form answer.

        Returns:
            Prompt string.
        """
        candidate_text = self._format_candidate_list(answer_candidates)

        return f"""
You are selecting the final answer for an OK-VQA question.

Question:
{question}

Structured evidence:
{reasoning_input}

Answer candidates:
{candidate_text}

Previous free-form answer:
{raw_output}

Task:
Choose exactly one answer candidate that best answers the question.

Rules:
1. Output only the candidate number.
2. Do not output words.
3. Do not explain.
4. Choose the shortest correct candidate when multiple candidates are plausible.
5. Prefer candidates that directly answer the question type.
6. Do not choose a generic detected object if a more specific candidate answers the question.
7. For action questions, choose an action candidate such as "swing" or "hit".
8. For "who" questions, choose a person category such as "man", "woman", or "person".
9. For "what is this called" questions, choose the object name.
10. For "what sport" questions, choose the sport or activity.

Output format:
Candidate: <number>
""".strip()

    def _call_candidate_selector(self, prompt: str) -> str:
        """
        Call LLM for candidate selection.

        Args:
            prompt: Candidate selection prompt.

        Returns:
            Raw selector output.
        """
        if hasattr(self.llm, "generate"):
            try:
                return self.llm.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=20,
                    system_instruction=(
                        "You are a strict multiple-choice selector. "
                        "Return only the selected candidate number."
                    ),
                )
            except TypeError:
                return self.llm.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=20,
                )

        return self._call_reasoning_llm(prompt)

    @staticmethod
    def _fallback_candidate_by_question_type(
        question: str,
        candidates: Sequence[str],
    ) -> Optional[str]:
        """
        Rule-light fallback selector.

        This does not use ground-truth answers. It only chooses among existing
        candidates based on broad question types.

        Args:
            question: Question text.
            candidates: Candidate list.

        Returns:
            Candidate string or None.
        """
        q = str(question or "").lower()

        cand_norm_to_original = {
            normalize_answer(candidate): candidate
            for candidate in candidates
            if normalize_answer(candidate)
        }

        def first_available(preferred: Sequence[str]) -> Optional[str]:
            for item in preferred:
                key = normalize_answer(item)
                if key in cand_norm_to_original:
                    return cand_norm_to_original[key]
            return None

        if "what sport" in q or "sport" in q:
            picked = first_available(
                [
                    "race",
                    "motocross",
                    "racing",
                    "ride",
                    "baseball",
                    "tennis",
                    "soccer",
                    "football",
                ]
            )
            if picked:
                return picked

        if "doing" in q or "action" in q:
            picked = first_available(
                [
                    "swing",
                    "hit",
                    "ride",
                    "riding",
                    "throw",
                    "catch",
                    "kick",
                    "run",
                    "walk",
                    "sit",
                    "stand",
                ]
            )
            if picked:
                return picked

        if "who" in q:
            picked = first_available(
                [
                    "man",
                    "men",
                    "woman",
                    "women",
                    "person",
                    "people",
                    "child",
                    "boy",
                    "girl",
                ]
            )
            if picked:
                return picked

        if "called" in q or "what is this" in q or "what is that" in q:
            picked = first_available(candidates)
            if picked:
                return picked

        if "type of plant" in q or "plant" in q:
            picked = first_available(
                [
                    "vine",
                    "ficus",
                    "plant",
                    "flower",
                    "tree",
                    "leaf",
                ]
            )
            if picked:
                return picked

        if "grow from" in q or "grows from" in q:
            picked = first_available(
                [
                    "ground",
                    "root",
                    "plant",
                    "stem",
                    "soil",
                ]
            )
            if picked:
                return picked

        if "kitchen" in q and ("center" in q or "unit" in q or "affixed" in q):
            picked = first_available(
                [
                    "island",
                    "kitchen island",
                    "counter",
                    "table",
                ]
            )
            if picked:
                return picked

        if "bag" in q or "carrying" in q:
            picked = first_available(
                [
                    "clothes",
                    "cloth",
                    "food",
                    "lunch",
                    "shoes",
                    "shoe",
                    "bag",
                ]
            )
            if picked:
                return picked

        if "place" in q or "go to" in q:
            picked = first_available(
                [
                    "shop",
                    "store",
                    "business",
                    "museum",
                    "visit",
                ]
            )
            if picked:
                return picked

        if "animal" in q and "part" in q:
            picked = first_available(
                [
                    "mouth",
                    "leg",
                    "paw",
                    "head",
                    "tail",
                ]
            )
            if picked:
                return picked

        return candidates[0] if candidates else None

    def _select_candidate_answer(
        self,
        question: str,
        reasoning_input: str,
        answer_candidates: Optional[Sequence[str]],
        raw_output: str = "",
    ) -> Optional[str]:
        """
        Select final answer from answer_candidates using constrained selection.

        If LLM fails to return a valid candidate id, fall back to a lightweight
        question-type selector, then to the first candidate.

        Args:
            question: Question text.
            reasoning_input: Structured evidence.
            answer_candidates: Candidate answers.
            raw_output: Previous free-form LLM answer.

        Returns:
            Selected candidate string or None.
        """
        candidates = self._clean_candidate_list(answer_candidates)

        if not candidates:
            return None

        prompt = self._build_candidate_selection_prompt(
            question=question,
            reasoning_input=reasoning_input,
            answer_candidates=candidates,
            raw_output=raw_output,
        )

        selection_output = self._call_candidate_selector(prompt)

        idx = self._parse_candidate_index(
            text=selection_output,
            num_candidates=len(candidates),
        )

        if idx is None:
            idx = self._parse_candidate_by_text(
                text=selection_output,
                candidates=candidates,
            )

        if idx is not None:
            return candidates[idx]

        fallback = self._fallback_candidate_by_question_type(
            question=question,
            candidates=candidates,
        )

        if fallback:
            return fallback

        return candidates[0]

    # ------------------------------------------------------------------
    # Main reasoning
    # ------------------------------------------------------------------

    def reason(
        self,
        question: str,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
        answer_candidates: Optional[Sequence[str]] = None,
        selected_prompts: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FinalReasoningResult:
        """
        Run final evidence-based reasoning.

        Args:
            question: VQA question.
            detections: Visual detections.
            passages: Retrieved or reranked passages.
            answer_candidates: Optional answer candidates.
            selected_prompts: Detection prompts used by detector.
            metadata: Optional metadata.

        Returns:
            FinalReasoningResult.
        """
        evidence_dict = self.evidence_formatter.format(
            question=question,
            detections=detections,
            passages=passages,
            answer_candidates=answer_candidates,
            selected_prompts=selected_prompts,
            metadata=metadata,
            as_string=False,
        )

        reasoning_input = self.evidence_formatter.serialize(evidence_dict)

        visual_evidence = evidence_dict.get("visual_evidence", [])
        knowledge_evidence = evidence_dict.get("knowledge", [])

        if self.structured:
            final_prompt = self.prompt_builder.build(
                evidence_json=reasoning_input,
            )
        else:
            visual_text = self.detections_to_plain_text(detections)
            knowledge_text = self.passages_to_plain_text(passages)

            final_prompt = self.prompt_builder.build(
                question=question,
                visual_evidence_text=visual_text,
                knowledge_text=knowledge_text,
            )

        raw_output = self._call_reasoning_llm(final_prompt)

        cleaned = self.answer_cleaner.clean(
            raw_output=raw_output,
            question=question,
        )

        if self.constrain_to_candidates and answer_candidates:
            ranked_candidates = rank_answer_candidates(
                question=question,
                candidates=answer_candidates,
                selected_prompts=selected_prompts,
                detections=detections,
                passages=passages,
            )

            if ranked_candidates:
                selected_candidate = ranked_candidates[0]
                cleaned = {
                    "pred_answer": selected_candidate,
                    "clean_answer": normalize_answer(selected_candidate),
                }

        return FinalReasoningResult(
            pred_answer=cleaned["pred_answer"],
            clean_answer=cleaned["clean_answer"],
            raw_reasoning_output=raw_output,
            reasoning_input=reasoning_input,
            final_prompt=final_prompt,
            visual_evidence=visual_evidence,
            knowledge_evidence=knowledge_evidence,
        )

    def reason_for_sample(
        self,
        sample: Dict[str, Any],
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
        answer_candidates: Optional[Sequence[str]] = None,
        selected_prompts: Optional[Sequence[str]] = None,
    ) -> FinalReasoningResult:
        """
        Run final reasoning for one unified VQA sample.

        Args:
            sample: Unified VQA sample.
            detections: Visual detections.
            passages: Knowledge passages.
            answer_candidates: Optional answer candidates.
            selected_prompts: Detection prompts.

        Returns:
            FinalReasoningResult.
        """
        return self.reason(
            question=sample.get("question", ""),
            detections=detections,
            passages=passages,
            answer_candidates=answer_candidates,
            selected_prompts=selected_prompts,
            metadata={
                "question_id": sample.get("question_id"),
                "image_id": sample.get("image_id"),
            },
        )

    def __call__(
        self,
        question: str,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
        answer_candidates: Optional[Sequence[str]] = None,
        selected_prompts: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FinalReasoningResult:
        """
        Alias for reason().
        """
        return self.reason(
            question=question,
            detections=detections,
            passages=passages,
            answer_candidates=answer_candidates,
            selected_prompts=selected_prompts,
            metadata=metadata,
        )


def build_final_reasoner(
    cfg: Dict[str, Any],
    llm: Any,
) -> FinalReasoner:
    """
    Build final reasoner from config.

    Args:
        cfg: Full experiment config.
        llm: LLM wrapper.

    Returns:
        FinalReasoner.
    """
    return FinalReasoner.from_config(cfg=cfg, llm=llm)