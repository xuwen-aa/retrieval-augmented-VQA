"""
Error analysis utilities for OV-VQA.

This module analyzes predictions and intermediate outputs.

Used for:
- wrong case extraction
- correct case extraction
- case study selection
- error type statistics
- comparing two experiment outputs
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.text import normalize_answer


Record = Dict[str, Any]


def is_correct_record(record: Record, threshold: float = 1.0) -> bool:
    """
    Determine whether a prediction record is correct.

    Args:
        record: Prediction or intermediate record.
        threshold: Score threshold.

    Returns:
        True if score >= threshold.
    """
    score = record.get("score")

    if score is None:
        return False

    try:
        return float(score) >= threshold
    except Exception:
        return False


def split_correct_wrong(
    records: Sequence[Record],
    threshold: float = 1.0,
) -> Dict[str, List[Record]]:
    """
    Split records into correct and wrong groups.

    Args:
        records: Prediction records.
        threshold: Correctness threshold.

    Returns:
        Dictionary with correct and wrong records.
    """
    correct = []
    wrong = []

    for record in records:
        if is_correct_record(record, threshold=threshold):
            correct.append(record)
        else:
            wrong.append(record)

    return {
        "correct": correct,
        "wrong": wrong,
    }


def count_empty_fields(
    records: Sequence[Record],
    fields: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """
    Count how often important fields are empty.

    Args:
        records: Intermediate records.
        fields: Fields to inspect.

    Returns:
        Empty count per field.
    """
    if fields is None:
        fields = [
            "concepts",
            "expanded_concepts",
            "selected_prompts",
            "detections",
            "retrieved_passages",
            "reranked_passages",
            "visual_evidence",
            "pred_answer",
            "clean_answer",
        ]

    counts = Counter()

    for record in records:
        for field in fields:
            value = record.get(field)

            if value is None:
                counts[field] += 1
            elif isinstance(value, (list, dict, str)) and len(value) == 0:
                counts[field] += 1

    return dict(counts)


def infer_error_types(record: Record) -> List[str]:
    """
    Infer rough error types from one intermediate record.

    Error labels:
    - no_concepts
    - no_detection_prompts
    - no_detections
    - no_retrieval
    - no_visual_evidence
    - empty_answer
    - answer_not_in_gt

    Args:
        record: Intermediate record.

    Returns:
        List of error type strings.
    """
    errors = []

    if not record.get("concepts"):
        errors.append("no_concepts")

    if not record.get("selected_prompts"):
        errors.append("no_detection_prompts")

    if not record.get("detections"):
        errors.append("no_detections")

    if not record.get("retrieved_passages") and not record.get("reranked_passages"):
        errors.append("no_retrieval")

    if not record.get("visual_evidence"):
        errors.append("no_visual_evidence")

    pred = record.get("clean_answer") or record.get("pred_answer") or ""
    if not str(pred).strip():
        errors.append("empty_answer")

    gt_answers = record.get("gt_answers") or record.get("answers") or []

    if pred and gt_answers:
        pred_norm = normalize_answer(pred)
        gt_norm = [normalize_answer(ans) for ans in gt_answers]

        if pred_norm not in gt_norm:
            errors.append("answer_not_in_gt")

    return errors


def summarize_error_types(records: Sequence[Record]) -> Dict[str, Any]:
    """
    Summarize error types over records.

    Args:
        records: Intermediate records.

    Returns:
        Error type statistics.
    """
    counter = Counter()

    for record in records:
        error_types = infer_error_types(record)
        counter.update(error_types)

    total = len(records)

    return {
        "num_records": total,
        "error_type_counts": dict(counter),
        "error_type_rates": {
            key: value / total if total > 0 else 0.0
            for key, value in counter.items()
        },
    }


def merge_prediction_and_intermediate(
    predictions: Sequence[Record],
    intermediate: Sequence[Record],
) -> List[Record]:
    """
    Merge prediction records with intermediate records by question_id.

    Args:
        predictions: Prediction records.
        intermediate: Intermediate records.

    Returns:
        Merged records.
    """
    pred_by_qid = {
        str(record.get("question_id")): record
        for record in predictions
        if record.get("question_id") is not None
    }

    merged = []

    for item in intermediate:
        qid = str(item.get("question_id"))
        pred = pred_by_qid.get(qid, {})

        merged_item = dict(item)
        for key, value in pred.items():
            if key not in merged_item or merged_item.get(key) is None:
                merged_item[key] = value

        merged.append(merged_item)

    return merged


def select_case_studies(
    records: Sequence[Record],
    max_cases: int = 20,
    require_correct: Optional[bool] = None,
    min_detections: int = 1,
    min_knowledge: int = 1,
) -> List[Record]:
    """
    Select useful case studies.

    Criteria:
    - has visual detections
    - has knowledge passages
    - has reasoning output
    - optionally correct or wrong

    Args:
        records: Intermediate records.
        max_cases: Maximum number of cases.
        require_correct:
            True: only correct cases
            False: only wrong cases
            None: no correctness filter
        min_detections: Minimum number of detections.
        min_knowledge: Minimum number of knowledge passages.

    Returns:
        Selected case records.
    """
    selected = []

    for record in records:
        if require_correct is not None:
            correct = is_correct_record(record, threshold=1.0)
            if correct != require_correct:
                continue

        detections = record.get("detections") or []
        knowledge = record.get("reranked_passages") or record.get("retrieved_passages") or []
        reasoning = record.get("raw_reasoning_output") or ""

        if len(detections) < min_detections:
            continue

        if len(knowledge) < min_knowledge:
            continue

        if not str(reasoning).strip():
            continue

        selected.append(record)

        if len(selected) >= max_cases:
            break

    return selected


def compare_experiments(
    baseline_records: Sequence[Record],
    ours_records: Sequence[Record],
) -> Dict[str, List[Record]]:
    """
    Compare two experiment outputs by question_id.

    Useful for selecting cases where:
    - ours correct, baseline wrong
    - baseline correct, ours wrong
    - both correct
    - both wrong

    Args:
        baseline_records: Baseline records.
        ours_records: Ours records.

    Returns:
        Dictionary of comparison groups.
    """
    base_by_qid = {
        str(record.get("question_id")): record
        for record in baseline_records
        if record.get("question_id") is not None
    }

    ours_by_qid = {
        str(record.get("question_id")): record
        for record in ours_records
        if record.get("question_id") is not None
    }

    groups = {
        "ours_correct_baseline_wrong": [],
        "baseline_correct_ours_wrong": [],
        "both_correct": [],
        "both_wrong": [],
    }

    for qid, ours in ours_by_qid.items():
        baseline = base_by_qid.get(qid)
        if baseline is None:
            continue

        ours_correct = is_correct_record(ours, threshold=1.0)
        base_correct = is_correct_record(baseline, threshold=1.0)

        merged = {
            "question_id": qid,
            "ours": ours,
            "baseline": baseline,
        }

        if ours_correct and not base_correct:
            groups["ours_correct_baseline_wrong"].append(merged)
        elif base_correct and not ours_correct:
            groups["baseline_correct_ours_wrong"].append(merged)
        elif ours_correct and base_correct:
            groups["both_correct"].append(merged)
        else:
            groups["both_wrong"].append(merged)

    return groups


def analyze_files(
    predictions_path: str,
    intermediate_path: Optional[str] = None,
    output_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze prediction and intermediate files.

    Args:
        predictions_path: Path to predictions.jsonl.
        intermediate_path: Optional path to intermediate.jsonl.
        output_prefix: Optional prefix for saving analysis outputs.

    Returns:
        Analysis summary.
    """
    predictions = read_jsonl(predictions_path)

    if intermediate_path:
        intermediate = read_jsonl(intermediate_path)
        records = merge_prediction_and_intermediate(predictions, intermediate)
    else:
        records = predictions

    split = split_correct_wrong(records, threshold=1.0)
    empty_counts = count_empty_fields(records)
    error_summary = summarize_error_types(split["wrong"])

    correct_cases = select_case_studies(
        split["correct"],
        max_cases=20,
        require_correct=True,
    )

    wrong_cases = select_case_studies(
        split["wrong"],
        max_cases=20,
        require_correct=False,
    )

    summary = {
        "num_records": len(records),
        "num_correct": len(split["correct"]),
        "num_wrong": len(split["wrong"]),
        "accuracy_strict": len(split["correct"]) / len(records) if records else 0.0,
        "empty_counts": empty_counts,
        "wrong_error_summary": error_summary,
        "num_selected_correct_cases": len(correct_cases),
        "num_selected_wrong_cases": len(wrong_cases),
    }

    if output_prefix:
        write_json(summary, f"{output_prefix}_summary.json")
        write_jsonl(split["correct"], f"{output_prefix}_correct.jsonl")
        write_jsonl(split["wrong"], f"{output_prefix}_wrong.jsonl")
        write_jsonl(correct_cases, f"{output_prefix}_case_correct.jsonl")
        write_jsonl(wrong_cases, f"{output_prefix}_case_wrong.jsonl")

    return summary