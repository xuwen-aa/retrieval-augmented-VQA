from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.text import (
    clean_predicted_answer,
    contains_insufficient_info,
    force_yes_no_if_needed,
    normalize_answer,
    normalize_whitespace,
    truncate_words,
)


@dataclass
class AnswerCleanerConfig:
    """
    Answer cleaner configuration.
    """

    short_answer: bool = True
    max_answer_words: int = 5
    normalize_for_eval: bool = True
    allow_insufficient: bool = False
    force_yes_no: bool = True


class AnswerCleaner:
    """
    Clean raw LLM outputs into final predicted answers.
    """

    def __init__(
        self,
        short_answer: bool = True,
        max_answer_words: int = 5,
        normalize_for_eval: bool = True,
        allow_insufficient: bool = False,
        force_yes_no: bool = True,
    ) -> None:
        """
        Args:
            short_answer: Whether to truncate long answers.
            max_answer_words: Maximum number of words for final answer.
            normalize_for_eval: Whether to produce VQA-normalized answer.
            allow_insufficient: Whether to keep "Insufficient information".
            force_yes_no: Whether to simplify yes/no answers for yes/no questions.
        """
        self.config = AnswerCleanerConfig(
            short_answer=short_answer,
            max_answer_words=max_answer_words,
            normalize_for_eval=normalize_for_eval,
            allow_insufficient=allow_insufficient,
            force_yes_no=force_yes_no,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "AnswerCleaner":
        """
        Build answer cleaner from config.

        Args:
            cfg: Full experiment config.

        Returns:
            AnswerCleaner.
        """
        reasoning_cfg = cfg.get("reasoning", {})
        evaluation_cfg = cfg.get("evaluation", {})

        return cls(
            short_answer=bool(reasoning_cfg.get("short_answer", True)),
            max_answer_words=int(reasoning_cfg.get("max_answer_words", 5)),
            normalize_for_eval=bool(evaluation_cfg.get("normalize_answer", True)),
            allow_insufficient=bool(reasoning_cfg.get("allow_insufficient", False)),
            force_yes_no=bool(reasoning_cfg.get("force_yes_no", True)),
        )

    @staticmethod
    def remove_prefix_phrases(answer: str) -> str:
        """
        Remove common answer prefixes.

        Args:
            answer: Extracted answer.

        Returns:
            Answer without generic prefixes.
        """
        if answer is None:
            return ""

        answer = str(answer).strip()

        prefix_patterns = [
            r"^the answer is\s+",
            r"^answer is\s+",
            r"^it is\s+",
            r"^it's\s+",
            r"^this is\s+",
            r"^that is\s+",
            r"^probably\s+",
            r"^most likely\s+",
            r"^likely\s+",
        ]

        for pattern in prefix_patterns:
            answer = re.sub(pattern, "", answer, flags=re.IGNORECASE).strip()

        return answer

    @staticmethod
    def remove_trailing_explanation(answer: str) -> str:
        """
        Remove trailing explanation after the short answer.

        Args:
            answer: Raw extracted answer.

        Returns:
            Shortened answer.
        """
        if answer is None:
            return ""

        answer = str(answer).strip()

        # Keep text before common explanation separators.
        separators = [
            " because ",
            " since ",
            " as ",
            " based on ",
            " according to ",
            " due to ",
        ]

        lower = answer.lower()

        for sep in separators:
            idx = lower.find(sep)
            if idx > 0:
                answer = answer[:idx].strip()
                break

        # Remove trailing sentence if model wrote multiple sentences.
        sentence_split = re.split(r"[.!?]\s+", answer)
        if sentence_split:
            answer = sentence_split[0].strip()

        return answer

    @staticmethod
    def strip_quotes_and_punctuation(answer: str) -> str:
        """
        Strip wrapping quotes and light punctuation.

        Args:
            answer: Answer string.

        Returns:
            Cleaned answer.
        """
        if answer is None:
            return ""

        answer = str(answer).strip()

        answer = answer.strip(" \t\r\n\"'`")
        answer = answer.strip()

        # Keep internal punctuation, but remove obvious trailing punctuation.
        answer = answer.rstrip(".。!！?？,，;；:")

        return answer.strip()

    @staticmethod
    def extract_final_answer_line(raw_output: str) -> str:
        """
        Extract answer using common final-answer markers.

        Args:
            raw_output: Raw LLM output.

        Returns:
            Extracted answer string.
        """
        if raw_output is None:
            return ""

        text = str(raw_output).strip()
        if not text:
            return ""

        # Prefer the last occurrence of Final answer.
        patterns = [
            r"final\s*answer\s*[:：]\s*(.+)",
            r"answer\s*[:：]\s*(.+)",
            r"prediction\s*[:：]\s*(.+)",
        ]

        for pattern in patterns:
            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if matches:
                candidate = matches[-1].strip()
                # If marker captured multiple lines, keep first non-empty line.
                lines = [line.strip() for line in candidate.splitlines() if line.strip()]
                return lines[0] if lines else candidate

        # Fallback: use helper from utils, then last non-empty line if needed.
        answer = clean_predicted_answer(
            raw_output=text,
            max_answer_words=None,
            normalize=False,
        )

        if answer:
            return answer

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def clean_raw_answer(
        self,
        raw_output: str,
        question: Optional[str] = None,
    ) -> str:
        """
        Clean raw LLM output without VQA normalization.

        Args:
            raw_output: Raw LLM output.
            question: Optional question for yes/no simplification.

        Returns:
            Human-readable cleaned answer.
        """
        answer = self.extract_final_answer_line(raw_output)

        answer = self.remove_prefix_phrases(answer)
        answer = self.remove_trailing_explanation(answer)
        answer = self.strip_quotes_and_punctuation(answer)
        answer = normalize_whitespace(answer)

        if not self.config.allow_insufficient and contains_insufficient_info(answer):
            # For OK-VQA scoring, avoid writing a long refusal string.
            # Keep it compact and explicit.
            answer = "unknown"

        if self.config.force_yes_no and question:
            answer = force_yes_no_if_needed(question, answer)

        if self.config.short_answer:
            answer = truncate_words(answer, self.config.max_answer_words)

        answer = self.strip_quotes_and_punctuation(answer)
        answer = normalize_whitespace(answer)

        return answer

    def clean_for_eval(self, answer: str) -> str:
        """
        Normalize answer for VQA evaluation.

        Args:
            answer: Cleaned human-readable answer.

        Returns:
            Normalized answer.
        """
        if not self.config.normalize_for_eval:
            return answer

        return normalize_answer(answer)

    def clean(
        self,
        raw_output: str,
        question: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Clean raw LLM output.

        Args:
            raw_output: Raw LLM output.
            question: Optional question.

        Returns:
            Dictionary with:
                pred_answer: human-readable cleaned answer
                clean_answer: normalized answer for evaluation
        """
        pred_answer = self.clean_raw_answer(
            raw_output=raw_output,
            question=question,
        )

        clean_answer = self.clean_for_eval(pred_answer)

        return {
            "pred_answer": pred_answer,
            "clean_answer": clean_answer,
        }

    def __call__(
        self,
        raw_output: str,
        question: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Alias for clean().
        """
        return self.clean(
            raw_output=raw_output,
            question=question,
        )


def build_answer_cleaner(cfg: Dict[str, Any]) -> AnswerCleaner:
    """
    Build answer cleaner from config.

    Args:
        cfg: Full experiment config.

    Returns:
        AnswerCleaner.
    """
    return AnswerCleaner.from_config(cfg)