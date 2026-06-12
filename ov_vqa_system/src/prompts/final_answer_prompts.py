from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FinalAnswerPromptConfig:
    """
    Final answer prompt configuration.
    """

    short_answer: bool = True
    max_answer_words: int = 5
    evidence_constraint: bool = True
    allow_insufficient: bool = False
    use_answer_candidates: bool = True


FINAL_ANSWER_SYSTEM_INSTRUCTION = (
    "You are a factual visual question answering agent. "
    "You must answer using only the provided evidence. "
    "The evidence may include visual detections, retrieved knowledge, and answer candidates. "
    "Do not introduce unsupported information. "
    "Return a concise final answer."
)


def build_structured_reasoning_prompt(
    evidence_json: str,
    short_answer: bool = True,
    max_answer_words: int = 5,
    evidence_constraint: bool = True,
    allow_insufficient: bool = False,
    use_answer_candidates: bool = True,
) -> str:
    """
    Build structured final reasoning prompt.

    Args:
        evidence_json: Serialized evidence JSON string.
        short_answer: Whether to force a short answer.
        max_answer_words: Maximum final answer words.
        evidence_constraint: Whether to require evidence-only reasoning.
        allow_insufficient: Whether model may answer insufficient information.
        use_answer_candidates: Whether to prefer answer candidates when available.

    Returns:
        Prompt string.
    """
    rules = []

    if evidence_constraint:
        rules.append(
            "Use only the provided visual evidence and retrieved knowledge."
        )
        rules.append(
            "Do not rely on unsupported prior assumptions."
        )

    if use_answer_candidates:
        rules.append(
            "If answer_candidates are provided, prefer one of them when it is supported by the evidence."
        )

    if short_answer:
        rules.append(
            f"Your final answer must be a short phrase with at most {max_answer_words} words."
        )

    if allow_insufficient:
        rules.append(
            "If the evidence is insufficient, answer exactly: Insufficient information."
        )
    else:
        rules.append(
            "Choose the best supported answer even if the evidence is incomplete."
        )

    rules.append(
        "Do not output long explanations."
    )
    rules.append(
        "The last line must follow this exact format: Final answer: <answer>"
    )

    rules_text = "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))

    return f"""
    You are given structured evidence for an OK-VQA task.

    Evidence:
    {evidence_json}

    Instructions:
    {rules_text}

    Your job is to output the most likely SHORT ground-truth style answer.

    Critical rules:
    1. Output exactly one line.
    2. The line must follow this exact format: Final answer: <answer>
    3. <answer> must be 1 to {max_answer_words} words.
    4. If answer_candidates are provided, choose exactly ONE candidate or a shorter synonym of one candidate.
    5. Do not output a sentence.
    6. Do not repeat any part of the question.
    7. Do not use "or", "/", "and", or multiple alternatives.
    8. Do not start with "you can", "it is", "this is", "the answer is", "using", "holding", or "type of".
    9. For action questions, output only the action verb when possible.
    10. For "what sport" questions, output the sport/activity label only.

    Bad outputs:
    Final answer: motorcycling or cycling
    Final answer: type of plant this is
    Final answer: holding a baseball bat
    Final answer: this grows from roots

    Good outputs:
    Final answer: race
    Final answer: motocross
    Final answer: vine
    Final answer: island
    Final answer: swing
    Final answer: root
    Final answer: man

    Now answer.

    Final answer:
    """.strip()


def build_unstructured_reasoning_prompt(
    question: str,
    visual_evidence_text: str = "",
    knowledge_text: str = "",
    short_answer: bool = True,
    max_answer_words: int = 5,
) -> str:
    """
    Build a weaker unstructured prompt.

    This is mainly used for the w/o structured reasoning ablation.

    Args:
        question: VQA question.
        visual_evidence_text: Plain text visual evidence.
        knowledge_text: Plain text knowledge.
        short_answer: Whether to force short answer.
        max_answer_words: Maximum answer words.

    Returns:
        Prompt string.
    """
    length_rule = (
        f"Answer with at most {max_answer_words} words."
        if short_answer
        else "Answer concisely."
    )

    return f"""
Question:
{question}

Visual evidence:
{visual_evidence_text or "None"}

Knowledge:
{knowledge_text or "None"}

Instruction:
Answer the question using the provided information. {length_rule}
End with: Final answer: <answer>

Final answer:
""".strip()


class FinalAnswerPromptBuilder:
    """
    Prompt builder for final answer generation.

    It supports:
    - structured evidence-based reasoning
    - unstructured reasoning ablation
    """

    def __init__(
        self,
        short_answer: bool = True,
        max_answer_words: int = 5,
        evidence_constraint: bool = True,
        allow_insufficient: bool = False,
        use_answer_candidates: bool = True,
        structured: bool = True,
    ) -> None:
        """
        Args:
            short_answer: Whether to force short answer.
            max_answer_words: Maximum answer words.
            evidence_constraint: Whether to require evidence-only reasoning.
            allow_insufficient: Whether insufficient information is allowed.
            use_answer_candidates: Whether answer candidates are used.
            structured: Whether to use structured prompt.
        """
        self.short_answer = short_answer
        self.max_answer_words = max_answer_words
        self.evidence_constraint = evidence_constraint
        self.allow_insufficient = allow_insufficient
        self.use_answer_candidates = use_answer_candidates
        self.structured = structured

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "FinalAnswerPromptBuilder":
        """
        Build from experiment config.

        Args:
            cfg: Full experiment config.

        Returns:
            FinalAnswerPromptBuilder.
        """
        reasoning_cfg = cfg.get("reasoning", {})

        return cls(
            short_answer=bool(reasoning_cfg.get("short_answer", True)),
            max_answer_words=int(reasoning_cfg.get("max_answer_words", 5)),
            evidence_constraint=bool(reasoning_cfg.get("evidence_constraint", True)),
            allow_insufficient=bool(reasoning_cfg.get("allow_insufficient", False)),
            use_answer_candidates=bool(
                reasoning_cfg.get("use_answer_candidates", True)
            ),
            structured=bool(reasoning_cfg.get("structured", True)),
        )

    def build(
        self,
        evidence_json: Optional[str] = None,
        question: Optional[str] = None,
        visual_evidence_text: str = "",
        knowledge_text: str = "",
    ) -> str:
        """
        Build final answer prompt.

        Args:
            evidence_json: Structured serialized evidence.
            question: Question text for unstructured ablation.
            visual_evidence_text: Plain visual evidence text.
            knowledge_text: Plain knowledge text.

        Returns:
            Prompt string.
        """
        if self.structured:
            if evidence_json is None:
                raise ValueError(
                    "evidence_json is required when structured reasoning is enabled."
                )

            return build_structured_reasoning_prompt(
                evidence_json=evidence_json,
                short_answer=self.short_answer,
                max_answer_words=self.max_answer_words,
                evidence_constraint=self.evidence_constraint,
                allow_insufficient=self.allow_insufficient,
                use_answer_candidates=self.use_answer_candidates,
            )

        if question is None:
            raise ValueError(
                "question is required when structured reasoning is disabled."
            )

        return build_unstructured_reasoning_prompt(
            question=question,
            visual_evidence_text=visual_evidence_text,
            knowledge_text=knowledge_text,
            short_answer=self.short_answer,
            max_answer_words=self.max_answer_words,
        )


def build_final_answer_prompt_builder(
    cfg: Dict[str, Any],
) -> FinalAnswerPromptBuilder:
    """
    Build final answer prompt builder from config.

    Args:
        cfg: Full experiment config.

    Returns:
        FinalAnswerPromptBuilder.
    """
    return FinalAnswerPromptBuilder.from_config(cfg)