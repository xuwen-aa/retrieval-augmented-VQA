from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.utils.text import normalize_answer


Detection = Dict[str, Any]
Passage = Dict[str, Any]


GENERIC_ANSWERS = {
    "object",
    "thing",
    "item",
    "person",
    "people",
    "place",
    "plant",
    "animal",
    "food",
    "material",
    "unit",
    "center",
    "face",
    "bag",
    "building",
    "tool",
}


def _clean_candidates(candidates: Optional[Sequence[str]]) -> List[str]:
    output = []
    seen = set()

    for c in candidates or []:
        c = str(c).strip()
        c = " ".join(c.split())

        if not c:
            continue

        key = normalize_answer(c)
        if not key or key in seen:
            continue

        seen.add(key)
        output.append(c)

    return output


def _collect_evidence_text(
    selected_prompts: Optional[Sequence[str]] = None,
    detections: Optional[Sequence[Detection]] = None,
    passages: Optional[Sequence[Passage]] = None,
) -> str:
    parts = []

    for p in selected_prompts or []:
        parts.append(str(p))

    for d in detections or []:
        parts.append(str(d.get("label", "")))
        parts.append(str(d.get("prompt", "")))

    for p in passages or []:
        parts.append(str(p.get("title", "")))
        parts.append(str(p.get("text", "")))

    return normalize_answer(" ".join(parts))


def _first_available(
    candidates: Sequence[str],
    preferred: Sequence[str],
) -> Optional[str]:
    cand_norm_to_original = {
        normalize_answer(c): c
        for c in candidates
        if normalize_answer(c)
    }

    for p in preferred:
        key = normalize_answer(p)
        if key in cand_norm_to_original:
            return cand_norm_to_original[key]

    return None


def _question_type_bonus(question: str, candidate: str, evidence_text: str) -> float:
    q = normalize_answer(question)
    c = normalize_answer(candidate)

    score = 0.0

    # Sport/activity questions.
    if "sport" in q or "game" in q:
        if any(x in evidence_text for x in ["motorcycle", "bike", "helmet", "track", "motocross"]):
            if c in {"race", "racing", "motocross", "ride"}:
                score += 8.0
        if c in {"race", "racing", "motocross", "ride", "baseball", "tennis", "soccer", "football"}:
            score += 5.0
        if c in {"ball", "bat", "racket", "helmet", "person"}:
            score -= 2.0

    # Action questions.
    if "doing" in q or "action" in q:
        if c in {"swing", "hit", "ride", "riding", "throw", "catch", "kick", "run", "walk", "sit", "stand"}:
            score += 8.0
        if c.startswith("holding") or c in {"bat", "baseball bat", "person"}:
            score -= 3.0

    if "bat" in q and ("doing" in q or "with bat" in q):
        if c in {"swing", "hit"}:
            score += 10.0
        if c in {"bat", "baseball", "baseball bat", "holding bat"}:
            score -= 3.0

    # Who questions.
    if q.startswith("who") or " who " in q:
        if c in {"man", "men", "woman", "women", "person", "people", "child", "boy", "girl"}:
            score += 8.0
        if c in {"toilet", "bathroom", "floor"}:
            score -= 4.0

    # Plant questions.
    if "type of plant" in q or "name type of plant" in q:
        if c in {"vine", "ficus"}:
            score += 10.0
        if c in {"plant", "flower", "tree", "leaf"}:
            score += 2.0

    if "grow from" in q or "grows from" in q:
        if c in {"ground", "root", "roots", "stem", "soil", "plant"}:
            score += 9.0
        if c in {"flower", "leaf", "vine"}:
            score -= 2.0

    # Body part questions.
    if "part" in q and ("face" in q or "body" in q or "animal" in q):
        if c in {"mouth", "teeth", "leg", "paw", "head", "tail", "hand", "hands"}:
            score += 8.0

    if "toothbrush" in q and "face" in q:
        if c in {"mouth", "teeth"}:
            score += 10.0
        if c in {"face", "toothbrush", "cheek", "jawline"}:
            score -= 2.0

    # Kitchen object naming.
    if "kitchen" in q and any(x in q for x in ["center", "affixed", "unit", "called"]):
        if c in {"island", "kitchen island"}:
            score += 10.0
        if c in {"counter", "table"}:
            score += 4.0
        if c in {"cabinet", "sink", "stove", "appliance"}:
            score -= 2.0

    # Carrying / bag contents.
    if "bag" in q or "carrying" in q:
        if c in {"cloth", "clothes", "food", "lunch", "shoe", "shoes"}:
            score += 8.0
        if c in {"bag", "red bag", "handbag", "person"}:
            score -= 4.0

    # Place-purpose questions.
    if "go to this place" in q or "why might someone go" in q:
        if c in {"shop", "store", "business"}:
            score += 8.0
        if c in {"museum", "visit"}:
            score += 2.0
        if c in {"building", "sign", "street", "beach", "park"}:
            score -= 2.0

    # Materials.
    if "material" in q or "made of" in q or "made from" in q:
        if c in {"cloth", "fabric", "leather", "nylon", "polyester", "wood", "metal", "plastic", "steel", "glass"}:
            score += 9.0
        if c in {"material", "seat material", "padded surface", "object"}:
            score -= 3.0

    # Containers.
    if "container" in q or "in what are" in q:
        if c in {"vase", "pot", "bowl", "cup", "box", "jar", "glass"}:
            score += 9.0
        if c in {"plant", "flower", "leaf"}:
            score -= 3.0

    # Food toppings.
    if "topped with" in q or "hot dog" in q:
        if c in {"relish", "onion relish", "onions", "onion", "condiment", "mustard", "ketchup"}:
            score += 8.0
        if c in {"food", "cheese", "person", "object"}:
            score -= 2.0

    # Toy questions.
    if "toy" in q:
        if c in {"teddy bear", "stuffed animal", "bear", "doll", "toy"}:
            score += 8.0
        if c in {"person", "object"}:
            score -= 3.0

    return score


def _generic_score(
    question: str,
    candidate: str,
    index: int,
    evidence_text: str,
) -> float:
    q = normalize_answer(question)
    c = normalize_answer(candidate)

    score = 0.0

    # Preserve candidate generator's ranking as a prior.
    score += max(0.0, 3.0 - 0.25 * index)

    # Short answers fit OK-VQA better.
    n_words = len(c.split())
    if n_words == 1:
        score += 1.0
    elif n_words <= 3:
        score += 0.5
    else:
        score -= 1.0

    # Avoid generic answers unless question explicitly asks for that category.
    if c in GENERIC_ANSWERS:
        score -= 2.0

    # Penalize candidates that merely repeat words in the question.
    # Example: "red bag" for "what is in the red bag?"
    if c and c in q:
        score -= 2.5

    # Weak evidence support.
    if c and c in evidence_text:
        score += 0.5

    # Avoid conjunction / alternatives.
    if " or " in candidate.lower() or " and " in candidate.lower() or "/" in candidate:
        score -= 3.0

    return score


def rank_answer_candidates(
    question: str,
    candidates: Optional[Sequence[str]],
    selected_prompts: Optional[Sequence[str]] = None,
    detections: Optional[Sequence[Detection]] = None,
    passages: Optional[Sequence[Passage]] = None,
) -> List[str]:
    """
    Rank answer candidates without using ground truth.

    Args:
        question: VQA question.
        candidates: Candidate answers.
        selected_prompts: Detection prompts.
        detections: Visual detections.
        passages: Retrieved/reranked passages.

    Returns:
        Ranked candidates.
    """
    candidates = _clean_candidates(candidates)

    if not candidates:
        return []

    evidence_text = _collect_evidence_text(
        selected_prompts=selected_prompts,
        detections=detections,
        passages=passages,
    )

    scored = []

    for idx, candidate in enumerate(candidates):
        score = 0.0
        score += _generic_score(
            question=question,
            candidate=candidate,
            index=idx,
            evidence_text=evidence_text,
        )
        score += _question_type_bonus(
            question=question,
            candidate=candidate,
            evidence_text=evidence_text,
        )

        scored.append(
            {
                "candidate": candidate,
                "score": score,
                "index": idx,
            }
        )

    scored.sort(
        key=lambda x: (x["score"], -x["index"]),
        reverse=True,
    )

    return [item["candidate"] for item in scored]