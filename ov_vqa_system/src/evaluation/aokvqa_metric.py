from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


_ARTICLES = {"a", "an", "the"}

_MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "ive": "i've",
    "lets": "let's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
}


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _remove_punctuation(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    out = []

    for char in text:
        if char in string.punctuation:
            out.append(" ")
        else:
            out.append(char)

    return "".join(out)


def normalize_answer(answer: Any) -> str:
    answer = _to_str(answer).lower().strip()

    answer = re.sub(r"^final answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"^answer\s*:\s*", "", answer, flags=re.IGNORECASE)

    answer = answer.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')
    answer = _remove_punctuation(answer)

    words = []
    for word in answer.split():
        word = _MANUAL_MAP.get(word, word)
        word = _CONTRACTIONS.get(word, word)
        if word not in _ARTICLES:
            words.append(word)

    return " ".join(words).strip()


def _extract_direct_answer(prediction: Any) -> str:
    if isinstance(prediction, str):
        return prediction

    if isinstance(prediction, dict):
        for key in (
            "clean_answer",
            "pred_answer",
            "direct_answer",
            "answer",
            "prediction",
            "text",
        ):
            value = prediction.get(key)
            if value not in (None, ""):
                return _to_str(value)

    return _to_str(prediction)


def _extract_choice_idx(prediction: Any) -> Optional[int]:
    if isinstance(prediction, int):
        return prediction

    if not isinstance(prediction, dict):
        return None

    for key in (
        "choice_idx",
        "pred_choice_idx",
        "predicted_choice_idx",
        "multiple_choice_idx",
        "mc_idx",
    ):
        value = prediction.get(key)
        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def _extract_choice_text(prediction: Any) -> str:
    if isinstance(prediction, str):
        return prediction

    if isinstance(prediction, dict):
        for key in (
            "choice",
            "pred_choice",
            "predicted_choice",
            "multiple_choice",
            "mc_answer",
            "answer",
            "pred_answer",
            "clean_answer",
        ):
            value = prediction.get(key)
            if value not in (None, ""):
                return _to_str(value)

    return _to_str(prediction)


def _get_gold_direct_answers(sample: Dict[str, Any]) -> List[str]:
    for key in ("direct_answers", "answers", "gt_answers", "gold_answers"):
        value = sample.get(key)
        if isinstance(value, list):
            return [_to_str(v) for v in value]

    value = sample.get("answer")
    if value is not None:
        return [_to_str(value)]

    return []


def _get_choices(sample: Dict[str, Any]) -> List[str]:
    value = sample.get("choices")
    if isinstance(value, list):
        return [_to_str(v) for v in value]

    value = sample.get("multiple_choice_answers")
    if isinstance(value, list):
        return [_to_str(v) for v in value]

    return []


def _get_gold_choice_idx(sample: Dict[str, Any]) -> Optional[int]:
    for key in ("correct_choice_idx", "answer_idx", "gold_choice_idx", "label"):
        value = sample.get(key)
        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def _get_gold_choice_text(sample: Dict[str, Any]) -> Optional[str]:
    idx = _get_gold_choice_idx(sample)
    choices = _get_choices(sample)

    if idx is not None and 0 <= idx < len(choices):
        return choices[idx]

    for key in ("correct_choice", "gold_choice", "multiple_choice_answer"):
        value = sample.get(key)
        if value not in (None, ""):
            return _to_str(value)

    return None


@dataclass
class AOKVQAScore:
    direct_answer_accuracy: Optional[float] = None
    multiple_choice_accuracy: Optional[float] = None
    num_samples: int = 0
    num_direct_answer_samples: int = 0
    num_multiple_choice_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direct_answer_accuracy": self.direct_answer_accuracy,
            "multiple_choice_accuracy": self.multiple_choice_accuracy,
            "num_samples": self.num_samples,
            "num_direct_answer_samples": self.num_direct_answer_samples,
            "num_multiple_choice_samples": self.num_multiple_choice_samples,
        }


class AOKVQAMetric:
    """
    Metric implementation for A-OKVQA.

    Supported tasks:
    - Direct Answer: VQA-style soft accuracy using direct_answers.
    - Multiple Choice: exact accuracy using correct_choice_idx or correct choice text.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def _norm(self, value: Any) -> str:
        if self.normalize:
            return normalize_answer(value)
        return _to_str(value).strip().lower()

    def score_direct_answer(
        self,
        prediction: Any,
        gold_answers: Sequence[Any],
    ) -> float:
        pred = self._norm(_extract_direct_answer(prediction))

        if not pred:
            return 0.0

        gold = [self._norm(a) for a in gold_answers if _to_str(a).strip()]
        if not gold:
            return 0.0

        match_count = sum(1 for ans in gold if ans == pred)

        return min(1.0, match_count / 3.0)

    def score_multiple_choice(
        self,
        prediction: Any,
        sample: Dict[str, Any],
    ) -> float:
        pred_idx = _extract_choice_idx(prediction)
        gold_idx = _get_gold_choice_idx(sample)

        if pred_idx is not None and gold_idx is not None:
            return 1.0 if pred_idx == gold_idx else 0.0

        gold_choice = _get_gold_choice_text(sample)
        if gold_choice is None:
            return 0.0

        pred_text = self._norm(_extract_choice_text(prediction))
        gold_text = self._norm(gold_choice)

        if not pred_text or not gold_text:
            return 0.0

        return 1.0 if pred_text == gold_text else 0.0

    def score_sample(
        self,
        prediction: Any,
        sample: Dict[str, Any],
    ) -> Dict[str, Optional[float]]:
        gold_answers = _get_gold_direct_answers(sample)

        direct_score: Optional[float]
        if gold_answers:
            direct_score = self.score_direct_answer(prediction, gold_answers)
        else:
            direct_score = None

        gold_choice = _get_gold_choice_text(sample)
        gold_idx = _get_gold_choice_idx(sample)

        mc_score: Optional[float]
        if gold_choice is not None or gold_idx is not None:
            mc_score = self.score_multiple_choice(prediction, sample)
        else:
            mc_score = None

        return {
            "direct_answer_accuracy": direct_score,
            "multiple_choice_accuracy": mc_score,
        }

    def evaluate(
        self,
        predictions: Sequence[Any],
        samples: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if len(predictions) != len(samples):
            raise ValueError(
                f"Number of predictions and samples must match. "
                f"Got {len(predictions)} predictions and {len(samples)} samples."
            )

        direct_scores: List[float] = []
        mc_scores: List[float] = []

        for pred, sample in zip(predictions, samples):
            scores = self.score_sample(pred, sample)

            if scores["direct_answer_accuracy"] is not None:
                direct_scores.append(float(scores["direct_answer_accuracy"]))

            if scores["multiple_choice_accuracy"] is not None:
                mc_scores.append(float(scores["multiple_choice_accuracy"]))

        direct_acc = sum(direct_scores) / len(direct_scores) if direct_scores else None
        mc_acc = sum(mc_scores) / len(mc_scores) if mc_scores else None

        result = AOKVQAScore(
            direct_answer_accuracy=direct_acc,
            multiple_choice_accuracy=mc_acc,
            num_samples=len(samples),
            num_direct_answer_samples=len(direct_scores),
            num_multiple_choice_samples=len(mc_scores),
        )

        return result.to_dict()

    def evaluate_from_records(
        self,
        records: Sequence[Dict[str, Any]],
        prediction_key: str = "pred_answer",
    ) -> Dict[str, Any]:
        predictions = []
        samples = []

        for record in records:
            if prediction_key in record:
                pred = record[prediction_key]
            elif "clean_answer" in record:
                pred = record["clean_answer"]
            elif "answer" in record:
                pred = record["answer"]
            else:
                pred = record

            predictions.append(pred)
            samples.append(record)

        return self.evaluate(predictions, samples)


def evaluate_aokvqa(
    predictions: Sequence[Any],
    samples: Sequence[Dict[str, Any]],
    normalize: bool = True,
) -> Dict[str, Any]:
    metric = AOKVQAMetric(normalize=normalize)
    return metric.evaluate(predictions, samples)


__all__ = [
    "AOKVQAMetric",
    "AOKVQAScore",
    "evaluate_aokvqa",
    "normalize_answer",
]