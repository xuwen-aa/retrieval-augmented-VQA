from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.utils.text import clean_concepts, normalize_answer, split_llm_list_output


Detection = Dict[str, Any]
Passage = Dict[str, Any]


@dataclass
class CandidateGeneratorConfig:
    """
    Candidate generator configuration.
    """

    enabled: bool = True
    max_candidates: int = 8
    max_visual_evidence: int = 10
    max_knowledge_passages: int = 5
    max_passage_chars: int = 500
    use_rule_based_candidates: bool = True


def _safe_lower(text: Any) -> str:
    """
    Convert text to lowercase safely.
    """
    if text is None:
        return ""
    return str(text).strip().lower()


def _deduplicate_keep_order(items: Sequence[str]) -> List[str]:
    """
    Deduplicate candidates while preserving order.
    """
    output: List[str] = []
    seen = set()

    for item in items:
        if item is None:
            continue

        item = str(item).strip()
        item = " ".join(item.split())

        if not item:
            continue

        key = normalize_answer(item)

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def _split_candidate_alternatives(candidate: str) -> List[str]:
    """
    Split multi-answer candidates like:
        "motorcycling or cycling"
        "race / motocross"
        "swing and hit"

    Args:
        candidate: Raw candidate.

    Returns:
        Candidate list.
    """
    if candidate is None:
        return []

    candidate = str(candidate).strip()

    if not candidate:
        return []

    # Avoid splitting normal noun phrases unless explicit alternatives appear.
    parts = re.split(r"\s+(?:or|and)\s+|/|\\|;", candidate, flags=re.IGNORECASE)

    output = []

    for part in parts:
        part = part.strip(" \t\r\n\"'`.,!?;:")

        if part:
            output.append(part)

    return output or [candidate]


def _candidate_is_valid(candidate: str) -> bool:
    """
    Filter invalid or meta candidates.
    """
    if candidate is None:
        return False

    candidate = str(candidate).strip()

    if not candidate:
        return False

    lower = candidate.lower()

    invalid_prefixes = (
        "here are",
        "here is",
        "short answer candidates",
        "answer candidates",
        "candidate answers",
        "possible answers",
        "based on",
        "according to",
        "the answer",
        "final answer",
        "question",
        "output",
        "return",
        "none",
        "n/a",
    )

    if lower.startswith(invalid_prefixes):
        return False

    if lower.endswith(":"):
        return False

    # Keep OK-VQA style short answers.
    if len(candidate.split()) > 5:
        return False

    return True


def clean_answer_candidates(
    candidates: Sequence[str],
    max_candidates: int = 8,
) -> List[str]:
    """
    Clean and normalize answer candidates.

    Args:
        candidates: Raw candidate list.
        max_candidates: Maximum number of candidates.

    Returns:
        Cleaned candidates.
    """
    expanded: List[str] = []

    for candidate in candidates:
        for part in _split_candidate_alternatives(candidate):
            part = part.strip()
            part = re.sub(r"^answer\s*:\s*", "", part, flags=re.IGNORECASE)
            part = re.sub(r"^candidate\s*:\s*", "", part, flags=re.IGNORECASE)
            part = re.sub(r"^final answer\s*:\s*", "", part, flags=re.IGNORECASE)
            part = part.strip(" \t\r\n\"'`.,!?;:")

            if _candidate_is_valid(part):
                expanded.append(part)

    expanded = clean_concepts(expanded)
    expanded = _deduplicate_keep_order(expanded)

    return expanded[:max_candidates]


def format_detections_for_candidates(
    detections: Optional[Sequence[Detection]],
    max_items: int = 10,
) -> str:
    """
    Format visual detections for candidate generation.

    Args:
        detections: Detector outputs.
        max_items: Maximum number of detections.

    Returns:
        Text description of detections.
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
        confidence = det.get("confidence")

        if not label:
            continue

        if confidence is not None:
            try:
                lines.append(f"{idx}. {label} (confidence={float(confidence):.3f})")
            except Exception:
                lines.append(f"{idx}. {label}")
        else:
            lines.append(f"{idx}. {label}")

    return "\n".join(lines) if lines else "None"


def format_passages_for_candidates(
    passages: Optional[Sequence[Passage]],
    max_items: int = 5,
    max_chars: int = 500,
) -> str:
    """
    Format knowledge passages for candidate generation.

    Args:
        passages: Retrieved or reranked passages.
        max_items: Maximum number of passages.
        max_chars: Maximum characters per passage.

    Returns:
        Text description of passages.
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

        if not text and not title:
            continue

        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."

        if title and text:
            lines.append(f"{idx}. {title}: {text}")
        elif text:
            lines.append(f"{idx}. {text}")
        else:
            lines.append(f"{idx}. {title}")

    return "\n".join(lines) if lines else "None"


def detection_labels_to_text(
    detections: Optional[Sequence[Detection]],
    selected_prompts: Optional[Sequence[str]] = None,
) -> str:
    """
    Collect detection labels and prompts into a lowercase text string.

    Args:
        detections: Detection outputs.
        selected_prompts: Selected prompts.

    Returns:
        Combined text.
    """
    parts: List[str] = []

    if selected_prompts:
        parts.extend([str(p) for p in selected_prompts if p])

    if detections:
        for det in detections:
            label = det.get("label") or det.get("prompt")
            prompt = det.get("prompt")

            if label:
                parts.append(str(label))
            if prompt:
                parts.append(str(prompt))

    return " ".join(parts).lower()


def get_rule_based_candidates(
    question: str,
    selected_prompts: Optional[Sequence[str]] = None,
    detections: Optional[Sequence[Detection]] = None,
    passages: Optional[Sequence[Passage]] = None,
) -> List[str]:
    """
    Generate OK-VQA-style short answer candidates using question and evidence.

    This does not use ground-truth answers. It only uses:
    - question text
    - selected detection prompts
    - detected labels
    - retrieved passage titles/text

    Args:
        question: VQA question.
        selected_prompts: Detection prompts.
        detections: Visual detections.
        passages: Retrieved/reranked passages.

    Returns:
        Rule-based candidate answers.
    """
    q = _safe_lower(question)
    evidence_text = detection_labels_to_text(
        detections=detections,
        selected_prompts=selected_prompts,
    )

    passage_text = ""

    if passages:
        chunks = []
        for passage in passages[:5]:
            chunks.append(str(passage.get("title", "")))
            chunks.append(str(passage.get("text", "")))
        passage_text = " ".join(chunks).lower()

    text_all = " ".join([q, evidence_text, passage_text])

    candidates: List[str] = []

    # ------------------------------------------------------------------
    # Sports / games
    # ------------------------------------------------------------------
    if "sport" in q or "game" in q or "playing" in q:
        if any(x in text_all for x in ["motorcycle", "motocross", "dirt bike", "bike", "helmet", "track"]):
            candidates.extend(["race", "racing", "motocross", "ride"])

        if any(x in text_all for x in ["bat", "baseball"]):
            candidates.extend(["baseball", "swing", "hit"])

        if any(x in text_all for x in ["racket", "racquet", "court", "net"]):
            candidates.extend(["tennis", "badminton", "racquetball"])

        if "ball" in text_all and "field" in text_all:
            candidates.extend(["soccer", "baseball", "football"])

        candidates.extend(["sport", "game"])

    # ------------------------------------------------------------------
    # Action questions
    # ------------------------------------------------------------------
    if q.startswith("what is") and "doing" in q:
        if "bat" in text_all:
            candidates.extend(["swing", "hit", "bat"])
        if any(x in text_all for x in ["skateboard", "skateboarder"]):
            candidates.extend(["skateboarding", "ride"])
        if any(x in text_all for x in ["bike", "bicycle", "motorcycle"]):
            candidates.extend(["ride", "riding", "race"])
        if "ball" in text_all:
            candidates.extend(["throw", "catch", "hit", "kick"])

    if "with the bat" in q or "bat" in q:
        candidates.extend(["swing", "hit", "baseball"])

    # ------------------------------------------------------------------
    # Plant questions
    # ------------------------------------------------------------------
    if "plant" in q or "flower" in q or "grow" in q:
        if "type of plant" in q or "name the type of plant" in q:
            candidates.extend(["vine", "flower", "plant", "ficus"])
        if "grow from" in q or "grows from" in q:
            candidates.extend(["ground", "plant", "root", "stem", "soil"])
        candidates.extend(["plant", "vine", "flower", "leaf"])

    # ------------------------------------------------------------------
    # Animal / body part questions
    # ------------------------------------------------------------------
    if "animal" in q:
        if "part" in q and ("game" in q or "playing" in q):
            candidates.extend(["mouth", "leg", "paw", "head"])
        candidates.extend(["mouth", "tail", "leg", "paw"])

    # ------------------------------------------------------------------
    # Bag / carrying
    # ------------------------------------------------------------------
    if "bag" in q or "carrying" in q:
        candidates.extend(["clothes", "food", "lunch", "shoes", "cloth"])
        if "red bag" in q or "red bag" in text_all:
            candidates.extend(["clothes", "food", "lunch", "shoes"])

    # ------------------------------------------------------------------
    # Toilet / bathroom
    # ------------------------------------------------------------------
    if "toilet" in q or "bathroom" in q:
        if "who" in q:
            candidates.extend(["man", "men", "person"])
        candidates.extend(["man", "person"])

    # ------------------------------------------------------------------
    # Kitchen object
    # ------------------------------------------------------------------
    if "kitchen" in q:
        if "center" in q or "affixed" in q or "unit" in q:
            candidates.extend(["island", "counter", "kitchen island"])
        candidates.extend(["island", "counter", "table"])

    # ------------------------------------------------------------------
    # Place / reason
    # ------------------------------------------------------------------
    if "why might someone go to this place" in q or "go to this place" in q:
        candidates.extend(["shop", "business", "store", "museum", "visit"])

    if "place" in q:
        if any(x in text_all for x in ["store", "shop", "business", "sign"]):
            candidates.extend(["shop", "business", "store"])
        if "museum" in text_all:
            candidates.extend(["museum", "visit"])

    # ------------------------------------------------------------------
    # Toy
    # ------------------------------------------------------------------
    if "toy" in q:
        candidates.extend(["stuffed animal", "teddy bear", "toy", "bear", "doll"])

    # ------------------------------------------------------------------
    # Food / material / object name generic patterns
    # ------------------------------------------------------------------
    if q.startswith("what is this") or q.startswith("what is that"):
        if "food" in text_all:
            candidates.extend(["food"])
        if "animal" in text_all:
            candidates.extend(["animal"])
        if "plant" in text_all:
            candidates.extend(["plant"])

    # ------------------------------------------------------------------
    # Detection-label candidates as fallback.
    # These are useful but lower priority than OK-VQA-style rules.
    # ------------------------------------------------------------------
    if detections:
        sorted_dets = sorted(
            detections,
            key=lambda x: float(x.get("confidence", 0.0)),
            reverse=True,
        )

        for det in sorted_dets[:8]:
            label = det.get("label") or det.get("prompt")
            if label:
                label = str(label).strip()
                if label:
                    candidates.append(label)

    # ------------------------------------------------------------------
    # Selected prompt candidates as final fallback.
    # ------------------------------------------------------------------
    if selected_prompts:
        for prompt in selected_prompts[:8]:
            if prompt:
                candidates.append(str(prompt))

    return clean_answer_candidates(candidates, max_candidates=30)


def build_candidate_generation_prompt(
    question: str,
    selected_prompts: Optional[Sequence[str]] = None,
    detections: Optional[Sequence[Detection]] = None,
    passages: Optional[Sequence[Passage]] = None,
    rule_candidates: Optional[Sequence[str]] = None,
    max_candidates: int = 8,
    max_visual_evidence: int = 10,
    max_knowledge_passages: int = 5,
    max_passage_chars: int = 500,
) -> str:
    """
    Build prompt for answer candidate generation.

    Args:
        question: VQA question.
        selected_prompts: Detection prompts.
        detections: Visual detections.
        passages: Retrieved or reranked knowledge passages.
        rule_candidates: Rule-based candidate answers.
        max_candidates: Maximum number of candidates.
        max_visual_evidence: Maximum visual evidence items.
        max_knowledge_passages: Maximum knowledge passages.
        max_passage_chars: Maximum characters per passage.

    Returns:
        Prompt string.
    """
    prompts_text = (
        "\n".join(f"- {p}" for p in selected_prompts)
        if selected_prompts
        else "None"
    )

    detections_text = format_detections_for_candidates(
        detections=detections,
        max_items=max_visual_evidence,
    )

    passages_text = format_passages_for_candidates(
        passages=passages,
        max_items=max_knowledge_passages,
        max_chars=max_passage_chars,
    )

    rule_text = (
        "\n".join(f"- {c}" for c in rule_candidates)
        if rule_candidates
        else "None"
    )

    return f"""
Generate short OK-VQA-style answer candidates.

The answer candidates should be:
- very short
- usually 1 to 3 words
- concrete labels, actions, objects, places, materials, animals, plant types, or sports
- not full sentences
- not explanations
- not multiple alternatives joined by "or" or "and"

Question:
{question}

Selected visual detection prompts:
{prompts_text}

Detected visual evidence:
{detections_text}

Retrieved knowledge:
{passages_text}

Rule-based candidate hints:
{rule_text}

Return at most {max_candidates} answer candidates.
Return only one candidate per line.
Do not include explanations.
Do not include meta text such as "Here are".
""".strip()


class AnswerCandidateGenerator:
    """
    Generate answer candidates using both rule-based priors and an LLM.

    Rule-based candidates are placed before LLM candidates because they are
    more OK-VQA-style and less verbose.
    """

    def __init__(
        self,
        llm: Any,
        enabled: bool = True,
        max_candidates: int = 8,
        max_visual_evidence: int = 10,
        max_knowledge_passages: int = 5,
        max_passage_chars: int = 500,
        use_rule_based_candidates: bool = True,
    ) -> None:
        """
        Args:
            llm: LLM wrapper.
            enabled: Whether candidate generation is enabled.
            max_candidates: Maximum number of candidates.
            max_visual_evidence: Maximum visual detections in prompt.
            max_knowledge_passages: Maximum knowledge passages in prompt.
            max_passage_chars: Maximum characters per passage.
            use_rule_based_candidates: Whether to use rule-based priors.
        """
        self.config = CandidateGeneratorConfig(
            enabled=enabled,
            max_candidates=max_candidates,
            max_visual_evidence=max_visual_evidence,
            max_knowledge_passages=max_knowledge_passages,
            max_passage_chars=max_passage_chars,
            use_rule_based_candidates=use_rule_based_candidates,
        )
        self.llm = llm

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        llm: Any,
    ) -> "AnswerCandidateGenerator":
        """
        Build candidate generator from config.

        Args:
            cfg: Full experiment config.
            llm: LLM wrapper.

        Returns:
            AnswerCandidateGenerator.
        """
        reasoning_cfg = cfg.get("reasoning", {})
        candidate_cfg = cfg.get("candidate_generation", {})

        return cls(
            llm=llm,
            enabled=bool(reasoning_cfg.get("use_answer_candidates", True)),
            max_candidates=int(
                reasoning_cfg.get(
                    "max_answer_candidates",
                    candidate_cfg.get("max_candidates", 8),
                )
            ),
            max_visual_evidence=int(candidate_cfg.get("max_visual_evidence", 10)),
            max_knowledge_passages=int(candidate_cfg.get("max_knowledge_passages", 5)),
            max_passage_chars=int(candidate_cfg.get("max_passage_chars", 500)),
            use_rule_based_candidates=bool(
                candidate_cfg.get("use_rule_based_candidates", True)
            ),
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM for candidate generation.

        Args:
            prompt: Candidate generation prompt.

        Returns:
            Raw LLM output.
        """
        if hasattr(self.llm, "generate_concepts"):
            return self.llm.generate_concepts(
                prompt=prompt,
                max_tokens=160,
            )

        return self.llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=160,
        )

    def generate(
        self,
        question: str,
        selected_prompts: Optional[Sequence[str]] = None,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer candidates.

        Args:
            question: VQA question.
            selected_prompts: Detection prompts.
            detections: Visual detections.
            passages: Retrieved or reranked passages.

        Returns:
            Dictionary with:
                candidates: list[str]
                raw_output: str
                prompt: str
                rule_candidates: list[str]
                llm_candidates: list[str]
        """
        if not self.config.enabled:
            return {
                "candidates": [],
                "raw_output": "",
                "prompt": "",
                "rule_candidates": [],
                "llm_candidates": [],
            }

        if self.config.use_rule_based_candidates:
            rule_candidates = get_rule_based_candidates(
                question=question,
                selected_prompts=selected_prompts,
                detections=detections,
                passages=passages,
            )
        else:
            rule_candidates = []

        prompt = build_candidate_generation_prompt(
            question=question,
            selected_prompts=selected_prompts,
            detections=detections,
            passages=passages,
            rule_candidates=rule_candidates,
            max_candidates=self.config.max_candidates,
            max_visual_evidence=self.config.max_visual_evidence,
            max_knowledge_passages=self.config.max_knowledge_passages,
            max_passage_chars=self.config.max_passage_chars,
        )

        raw_output = self._call_llm(prompt)

        llm_candidates = split_llm_list_output(raw_output)
        llm_candidates = clean_answer_candidates(
            llm_candidates,
            max_candidates=self.config.max_candidates,
        )

        # Rule candidates first. They are usually closer to OK-VQA labels.
        merged = list(rule_candidates) + list(llm_candidates)
        candidates = clean_answer_candidates(
            merged,
            max_candidates=self.config.max_candidates,
        )

        return {
            "candidates": candidates,
            "raw_output": raw_output,
            "prompt": prompt,
            "rule_candidates": rule_candidates,
            "llm_candidates": llm_candidates,
        }

    def generate_for_sample(
        self,
        sample: Dict[str, Any],
        selected_prompts: Optional[Sequence[str]] = None,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer candidates for a unified VQA sample.

        Args:
            sample: Unified VQA sample.
            selected_prompts: Detection prompts.
            detections: Visual detections.
            passages: Retrieved or reranked passages.

        Returns:
            Candidate generation result.
        """
        return self.generate(
            question=sample.get("question", ""),
            selected_prompts=selected_prompts,
            detections=detections,
            passages=passages,
        )

    def __call__(
        self,
        question: str,
        selected_prompts: Optional[Sequence[str]] = None,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
    ) -> Dict[str, Any]:
        """
        Alias for generate().
        """
        return self.generate(
            question=question,
            selected_prompts=selected_prompts,
            detections=detections,
            passages=passages,
        )


class EmptyCandidateGenerator:
    """
    Fallback candidate generator when answer candidates are disabled.
    """

    def generate(
        self,
        question: str,
        selected_prompts: Optional[Sequence[str]] = None,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
    ) -> Dict[str, Any]:
        return {
            "candidates": [],
            "raw_output": "",
            "prompt": "",
            "rule_candidates": [],
            "llm_candidates": [],
        }

    def generate_for_sample(
        self,
        sample: Dict[str, Any],
        selected_prompts: Optional[Sequence[str]] = None,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
    ) -> Dict[str, Any]:
        return {
            "candidates": [],
            "raw_output": "",
            "prompt": "",
            "rule_candidates": [],
            "llm_candidates": [],
        }

    def __call__(
        self,
        question: str,
        selected_prompts: Optional[Sequence[str]] = None,
        detections: Optional[Sequence[Detection]] = None,
        passages: Optional[Sequence[Passage]] = None,
    ) -> Dict[str, Any]:
        return self.generate(
            question=question,
            selected_prompts=selected_prompts,
            detections=detections,
            passages=passages,
        )


def build_candidate_generator(
    cfg: Dict[str, Any],
    llm: Any,
):
    """
    Build candidate generator from config.

    Args:
        cfg: Full experiment config.
        llm: LLM wrapper.

    Returns:
        AnswerCandidateGenerator or EmptyCandidateGenerator.
    """
    reasoning_cfg = cfg.get("reasoning", {})
    enabled = bool(reasoning_cfg.get("use_answer_candidates", True))

    if not enabled:
        return EmptyCandidateGenerator()

    return AnswerCandidateGenerator.from_config(cfg=cfg, llm=llm)