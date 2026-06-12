"""
OK-VQA evaluation metric.

This module implements VQA-style accuracy for OK-VQA.

Standard VQA accuracy:
    acc = min(1, number_of_matching_human_answers / 3)

For each question, OK-VQA usually provides 10 human answers.
A predicted answer is compared against the normalized human answers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.utils.text import normalize_answer


@dataclass
class OKVQAMetricResult:
    """
    OK-VQA metric result.
    """

    accuracy: float
    total: int
    correct_sum: float


class OKVQAMetric:
    """
    OK-VQA VQA-style accuracy metric.

    For one prediction:
        score = min(1.0, count(prediction in gt_answers) / 3.0)

    Example:
        gt_answers = ["cat"] repeated 6 times
        pred = "cat"
        score = min(1, 6 / 3) = 1.0

        gt_answers = ["cat"] repeated 1 time
        pred = "cat"
        score = min(1, 1 / 3) = 0.3333
    """

    def __init__(self, normalize: bool = True) -> None:
        """
        Args:
            normalize: Whether to normalize predictions and ground-truth answers.
        """
        self.normalize = normalize
        self.scores: List[float] = []
        self.records: List[Dict[str, Any]] = []

    @staticmethod
    def _ensure_answer_list(answers: Any) -> List[str]:
        """
        Convert raw ground-truth answers into a list of strings.

        Args:
            answers: Raw answers.

        Returns:
            List of answer strings.
        """
        if answers is None:
            return []

        if isinstance(answers, str):
            return [answers]

        output = []

        if isinstance(answers, list):
            for item in answers:
                if isinstance(item, str):
                    output.append(item)
                elif isinstance(item, dict):
                    if "answer" in item:
                        output.append(str(item["answer"]))
                    elif "raw_answer" in item:
                        output.append(str(item["raw_answer"]))
                    elif "direct_answer" in item:
                        output.append(str(item["direct_answer"]))
                else:
                    output.append(str(item))

        return [x.strip() for x in output if str(x).strip()]

    def normalize_text(self, text: str) -> str:
        """
        Normalize text if enabled.

        Args:
            text: Input text.

        Returns:
            Normalized or raw text.
        """
        if self.normalize:
            return normalize_answer(text)

        return str(text).strip().lower()

    def score_prediction(
        self,
        prediction: str,
        gt_answers: Sequence[str],
    ) -> float:
        """
        Score one predicted answer.

        Args:
            prediction: Predicted answer.
            gt_answers: Ground-truth human answers.

        Returns:
            VQA-style score in [0, 1].
        """
        if prediction is None:
            prediction = ""

        gt_answers = self._ensure_answer_list(gt_answers)

        if not gt_answers:
            return 0.0

        pred_norm = self.normalize_text(prediction)
        gt_norm = [self.normalize_text(ans) for ans in gt_answers]

        match_count = sum(1 for ans in gt_norm if ans == pred_norm)

        return min(1.0, match_count / 3.0)

    def add(
        self,
        prediction: str,
        gt_answers: Sequence[str],
        question_id: Optional[Any] = None,
        image_id: Optional[Any] = None,
        question: Optional[str] = None,
    ) -> float:
        """
        Add one prediction and return its score.

        Args:
            prediction: Predicted answer.
            gt_answers: Ground-truth answers.
            question_id: Optional question ID.
            image_id: Optional image ID.
            question: Optional question text.

        Returns:
            Per-sample score.
        """
        gt_answers = self._ensure_answer_list(gt_answers)
        score = self.score_prediction(prediction, gt_answers)

        self.scores.append(score)

        record = {
            "question_id": question_id,
            "image_id": image_id,
            "question": question,
            "prediction": prediction,
            "gt_answers": gt_answers,
            "score": score,
        }

        self.records.append(record)

        return score

    def compute(self) -> Dict[str, Any]:
        """
        Compute final metrics.

        Returns:
            Metric dictionary.
        """
        total = len(self.scores)
        correct_sum = float(sum(self.scores))
        accuracy = correct_sum / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "okvqa_accuracy": accuracy,
            "correct_sum": correct_sum,
            "num_samples": total,
        }

    def reset(self) -> None:
        """
        Reset metric state.
        """
        self.scores = []
        self.records = []

    def get_records(self) -> List[Dict[str, Any]]:
        """
        Return per-sample metric records.

        Returns:
            List of records.
        """
        return self.records


def evaluate_predictions(
    predictions: Sequence[Dict[str, Any]],
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a list of prediction records.

    Expected prediction record format:
        {
            "question_id": ...,
            "image_id": ...,
            "question": ...,
            "clean_answer": "...",
            "gt_answers": [...]
        }

    Args:
        predictions: Prediction records.
        normalize: Whether to normalize answers.

    Returns:
        Metric dictionary.
    """
    metric = OKVQAMetric(normalize=normalize)

    for record in predictions:
        prediction = (
            record.get("clean_answer")
            or record.get("pred_answer")
            or record.get("prediction")
            or ""
        )

        gt_answers = record.get("gt_answers") or record.get("answers") or []

        metric.add(
            prediction=prediction,
            gt_answers=gt_answers,
            question_id=record.get("question_id"),
            image_id=record.get("image_id"),
            question=record.get("question"),
        )

    return metric.compute()


def build_okvqa_metric(cfg: Dict[str, Any]) -> OKVQAMetric:
    """
    Build OK-VQA metric from config.

    Args:
        cfg: Full experiment config.

    Returns:
        OKVQAMetric.
    """
    evaluation_cfg = cfg.get("evaluation", {})

    return OKVQAMetric(
        normalize=bool(evaluation_cfg.get("normalize_answer", True))
    )