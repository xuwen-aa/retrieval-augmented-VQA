"""
Evidence formatter for OV-VQA reasoning.

This module converts visual detections and retrieved knowledge into
a structured JSON-like evidence input for the final LLM reasoner.

The formatted evidence is used by:
- evidence-based reasoning
- intermediate.jsonl saving
- case study analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.utils.text import truncate_text


Detection = Dict[str, Any]
Passage = Dict[str, Any]
Evidence = Dict[str, Any]


@dataclass
class EvidenceFormatterConfig:
    """
    Evidence formatter configuration.
    """

    max_visual_evidence: int = 10
    max_knowledge_passages: int = 5
    max_passage_chars: int = 600
    include_bbox: bool = True
    include_detection_confidence: bool = True
    include_knowledge_scores: bool = True
    include_answer_candidates: bool = True


class EvidenceFormatter:
    """
    Format visual and textual evidence for final reasoning.
    """

    def __init__(
        self,
        max_visual_evidence: int = 10,
        max_knowledge_passages: int = 5,
        max_passage_chars: int = 600,
        include_bbox: bool = True,
        include_detection_confidence: bool = True,
        include_knowledge_scores: bool = True,
        include_answer_candidates: bool = True,
    ) -> None:
        """
        Args:
            max_visual_evidence: Maximum number of detections to include.
            max_knowledge_passages: Maximum number of knowledge passages.
            max_passage_chars: Maximum characters per passage.
            include_bbox: Whether to include bbox in visual evidence.
            include_detection_confidence: Whether to include detection confidence.
            include_knowledge_scores: Whether to include retrieval/rerank scores.
            include_answer_candidates: Whether to include answer candidates.
        """
        self.config = EvidenceFormatterConfig(
            max_visual_evidence=max_visual_evidence,
            max_knowledge_passages=max_knowledge_passages,
            max_passage_chars=max_passage_chars,
            include_bbox=include_bbox,
            include_detection_confidence=include_detection_confidence,
            include_knowledge_scores=include_knowledge_scores,
            include_answer_candidates=include_answer_candidates,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "EvidenceFormatter":
        """
        Build formatter from config.

        Args:
            cfg: Full experiment config.

        Returns:
            EvidenceFormatter.
        """
        reasoning_cfg = cfg.get("reasoning", {})
        formatter_cfg = cfg.get("evidence_formatter", {})

        return cls(
            max_visual_evidence=int(
                formatter_cfg.get(
                    "max_visual_evidence",
                    cfg.get("detector", {}).get("max_detections", 10),
                )
            ),
            max_knowledge_passages=int(
                formatter_cfg.get(
                    "max_knowledge_passages",
                    cfg.get("reranker", {}).get("top_k", 5),
                )
            ),
            max_passage_chars=int(formatter_cfg.get("max_passage_chars", 600)),
            include_bbox=bool(formatter_cfg.get("include_bbox", True)),
            include_detection_confidence=bool(
                formatter_cfg.get("include_detection_confidence", True)
            ),
            include_knowledge_scores=bool(
                formatter_cfg.get("include_knowledge_scores", True)
            ),
            include_answer_candidates=bool(
                reasoning_cfg.get("use_answer_candidates", True)
            ),
        )

    @staticmethod
    def _round_float(value: Any, digits: int = 4) -> Optional[float]:
        """
        Safely round a float value.

        Args:
            value: Raw value.
            digits: Decimal digits.

        Returns:
            Rounded float or None.
        """
        if value is None:
            return None

        try:
            return round(float(value), digits)
        except Exception:
            return None

    @staticmethod
    def _round_bbox(bbox: Any, digits: int = 2) -> Optional[List[float]]:
        """
        Round bbox coordinates.

        Args:
            bbox: Raw bbox.
            digits: Decimal digits.

        Returns:
            Rounded bbox list or None.
        """
        if bbox is None:
            return None

        if not isinstance(bbox, (list, tuple)):
            return None

        output = []

        for value in bbox:
            try:
                output.append(round(float(value), digits))
            except Exception:
                return None

        return output

    def format_visual_evidence(
        self,
        detections: Optional[Sequence[Detection]],
    ) -> List[Evidence]:
        """
        Format detector outputs as visual evidence.

        Args:
            detections: Detector output list.

        Returns:
            Visual evidence list.
        """
        if not detections:
            return []

        sorted_detections = sorted(
            detections,
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )

        visual_evidence: List[Evidence] = []

        for det in sorted_detections[: self.config.max_visual_evidence]:
            label = det.get("label") or det.get("prompt") or det.get("class_name")
            prompt = det.get("prompt") or label

            if not label:
                continue

            item: Evidence = {
                "concept": str(label),
                "prompt": str(prompt),
            }

            if self.config.include_detection_confidence:
                confidence = self._round_float(det.get("confidence"))
                if confidence is not None:
                    item["confidence"] = confidence

            if self.config.include_bbox:
                bbox_xyxy = self._round_bbox(det.get("bbox_xyxy"))
                bbox_xywh = self._round_bbox(det.get("bbox_xywh"))

                if bbox_xyxy is not None:
                    item["bbox_xyxy"] = bbox_xyxy

                if bbox_xywh is not None:
                    item["bbox_xywh"] = bbox_xywh

            source = det.get("source")
            if source:
                item["source"] = source

            visual_evidence.append(item)

        return visual_evidence

    def format_knowledge(
        self,
        passages: Optional[Sequence[Passage]],
    ) -> List[Evidence]:
        """
        Format retrieved / reranked passages as knowledge evidence.

        Args:
            passages: Retrieved or reranked passages.

        Returns:
            Knowledge evidence list.
        """
        if not passages:
            return []

        knowledge: List[Evidence] = []

        for passage in passages[: self.config.max_knowledge_passages]:
            text = (
                passage.get("text")
                or passage.get("passage")
                or passage.get("content")
                or ""
            )
            title = (
                passage.get("title")
                or passage.get("page_title")
                or passage.get("wiki_title")
                or ""
            )

            text = str(text).strip()
            title = str(title).strip()

            if not text and not title:
                continue

            item: Evidence = {
                "title": title,
                "text": truncate_text(text, self.config.max_passage_chars),
            }

            if "id" in passage:
                item["id"] = passage.get("id")

            if self.config.include_knowledge_scores:
                retrieval_score = self._round_float(
                    passage.get("retrieval_score", passage.get("score"))
                )
                rerank_score = self._round_float(passage.get("rerank_score"))

                if retrieval_score is not None:
                    item["retrieval_score"] = retrieval_score

                if rerank_score is not None:
                    item["rerank_score"] = rerank_score

            rank = passage.get("rerank_rank", passage.get("rank"))
            if rank is not None:
                item["rank"] = rank

            source = passage.get("source")
            if source:
                item["source"] = source

            knowledge.append(item)

        return knowledge

    @staticmethod
    def format_answer_candidates(
        answer_candidates: Optional[Sequence[str]],
        max_candidates: int = 8,
    ) -> List[str]:
        """
        Format answer candidates.

        Args:
            answer_candidates: Candidate answer list.
            max_candidates: Maximum number of candidates.

        Returns:
            Clean candidate list.
        """
        if not answer_candidates:
            return []

        output = []
        seen = set()

        for candidate in answer_candidates:
            if candidate is None:
                continue

            candidate = str(candidate).strip()
            candidate = " ".join(candidate.split())

            if not candidate:
                continue

            key = candidate.lower()
            if key in seen:
                continue

            seen.add(key)
            output.append(candidate)

            if len(output) >= max_candidates:
                break

        return output

    def build_evidence_dict(
        self,
        question: str,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
        answer_candidates: Optional[Sequence[str]] = None,
        selected_prompts: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build structured evidence dictionary.

        Args:
            question: VQA question.
            detections: Visual detections.
            passages: Retrieved or reranked knowledge passages.
            answer_candidates: Optional answer candidates.
            selected_prompts: Detection prompts used by detector.
            metadata: Optional metadata.

        Returns:
            Structured evidence dictionary.
        """
        visual_evidence = self.format_visual_evidence(detections)
        knowledge = self.format_knowledge(passages)

        evidence: Dict[str, Any] = {
            "question": str(question).strip(),
            "visual_evidence": visual_evidence,
            "knowledge": knowledge,
        }

        if selected_prompts is not None:
            evidence["selected_detection_prompts"] = [
                str(p).strip() for p in selected_prompts if str(p).strip()
            ]

        if self.config.include_answer_candidates:
            candidates = self.format_answer_candidates(answer_candidates)
            if candidates:
                evidence["answer_candidates"] = candidates

        if metadata:
            evidence["metadata"] = metadata

        return evidence

    def serialize(
        self,
        evidence: Dict[str, Any],
        indent: int = 2,
    ) -> str:
        """
        Serialize evidence dictionary to JSON string.

        Args:
            evidence: Evidence dictionary.
            indent: JSON indentation.

        Returns:
            JSON string.
        """
        return json.dumps(
            evidence,
            ensure_ascii=False,
            indent=indent,
        )

    def format(
        self,
        question: str,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
        answer_candidates: Optional[Sequence[str]] = None,
        selected_prompts: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        as_string: bool = True,
    ) -> str | Dict[str, Any]:
        """
        Format all evidence.

        Args:
            question: VQA question.
            detections: Visual detections.
            passages: Knowledge passages.
            answer_candidates: Optional answer candidates.
            selected_prompts: Detection prompts.
            metadata: Optional metadata.
            as_string: If True, return JSON string. Otherwise return dict.

        Returns:
            Serialized evidence or evidence dictionary.
        """
        evidence = self.build_evidence_dict(
            question=question,
            detections=detections,
            passages=passages,
            answer_candidates=answer_candidates,
            selected_prompts=selected_prompts,
            metadata=metadata,
        )

        if as_string:
            return self.serialize(evidence)

        return evidence

    def __call__(
        self,
        question: str,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
        answer_candidates: Optional[Sequence[str]] = None,
        selected_prompts: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        as_string: bool = True,
    ) -> str | Dict[str, Any]:
        """
        Alias for format().
        """
        return self.format(
            question=question,
            detections=detections,
            passages=passages,
            answer_candidates=answer_candidates,
            selected_prompts=selected_prompts,
            metadata=metadata,
            as_string=as_string,
        )


def build_evidence_formatter(cfg: Dict[str, Any]) -> EvidenceFormatter:
    """
    Build evidence formatter from config.

    Args:
        cfg: Full experiment config.

    Returns:
        EvidenceFormatter.
    """
    return EvidenceFormatter.from_config(cfg)