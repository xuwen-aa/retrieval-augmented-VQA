"""
I/O utilities for OV-VQA experiments.

This module provides safe helpers for reading and writing:
- JSON
- JSONL
- YAML
- plain text files
- experiment outputs

All writing functions create parent directories automatically.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


JsonDict = Dict[str, Any]


def ensure_dir(path: str | os.PathLike) -> Path:
    """
    Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: str | os.PathLike) -> Path:
    """
    Create the parent directory of a file path.

    Args:
        path: File path.

    Returns:
        Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | os.PathLike) -> Any:
    """
    Read a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    data: Any,
    path: str | os.PathLike,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Write data to a JSON file.

    Args:
        data: JSON-serializable object.
        path: Output path.
        indent: JSON indentation.
        ensure_ascii: Whether to escape non-ASCII characters.
    """
    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )


def read_jsonl(path: str | os.PathLike) -> List[JsonDict]:
    """
    Read a JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        List of JSON objects.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: List[JsonDict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {e}"
                ) from e

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Each JSONL line must be a JSON object. "
                    f"Got {type(obj)} at {path}:{line_no}"
                )

            records.append(obj)

    return records


def write_jsonl(
    records: Iterable[JsonDict],
    path: str | os.PathLike,
    ensure_ascii: bool = False,
) -> None:
    """
    Write records to a JSONL file.

    Args:
        records: Iterable of dictionaries.
        path: Output JSONL path.
        ensure_ascii: Whether to escape non-ASCII characters.
    """
    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=ensure_ascii) + "\n"
            )


def append_jsonl(
    record: JsonDict,
    path: str | os.PathLike,
    ensure_ascii: bool = False,
) -> None:
    """
    Append one record to a JSONL file.

    This is useful during long-running experiments.
    If the process crashes, previous records are already saved.

    Args:
        record: JSON-serializable dictionary.
        path: Output JSONL path.
        ensure_ascii: Whether to escape non-ASCII characters.
    """
    path = ensure_parent_dir(path)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=ensure_ascii) + "\n"
        )


def read_yaml(path: str | os.PathLike) -> Any:
    """
    Read a YAML file.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML object.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(
    data: Any,
    path: str | os.PathLike,
    allow_unicode: bool = True,
) -> None:
    """
    Write data to a YAML file.

    Args:
        data: YAML-serializable object.
        path: Output YAML path.
        allow_unicode: Whether to preserve unicode characters.
    """
    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=allow_unicode,
            sort_keys=False,
            default_flow_style=False,
        )


def read_text(path: str | os.PathLike) -> str:
    """
    Read a plain text file.

    Args:
        path: Text file path.

    Returns:
        File content.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return f.read()


def write_text(text: str, path: str | os.PathLike) -> None:
    """
    Write text to a file.

    Args:
        text: Text content.
        path: Output path.
    """
    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def get_output_paths(cfg: JsonDict) -> JsonDict:
    """
    Build standard output paths from config.

    Args:
        cfg: Experiment config.

    Returns:
        Dictionary of standard output file paths.
    """
    output_dir = Path(cfg["experiment"]["output_dir"])

    output_cfg = cfg.get("output", {})

    return {
        "output_dir": output_dir,
        "config": output_dir / output_cfg.get("config_file", "config.yaml"),
        "metrics": output_dir / output_cfg.get("metrics_file", "metrics.json"),
        "predictions": output_dir / output_cfg.get(
            "predictions_file", "predictions.jsonl"
        ),
        "intermediate": output_dir / output_cfg.get(
            "intermediate_file", "intermediate.jsonl"
        ),
        "command": output_dir / output_cfg.get("command_file", "command.txt"),
        "log": output_dir / output_cfg.get("log_file", "run.log"),
    }


def prepare_output_dir(cfg: JsonDict) -> JsonDict:
    """
    Create output directory and return standard output paths.

    Args:
        cfg: Experiment config.

    Returns:
        Dictionary of output paths.
    """
    paths = get_output_paths(cfg)
    ensure_dir(paths["output_dir"])
    return paths


def save_command(path: str | os.PathLike, argv: Optional[List[str]] = None) -> None:
    """
    Save the command used to run an experiment.

    Args:
        path: Output command file.
        argv: Command arguments. If None, use sys.argv.
    """
    if argv is None:
        argv = sys.argv

    command = " ".join(argv)
    write_text(command + "\n", path)


def remove_file_if_exists(path: str | os.PathLike) -> None:
    """
    Remove a file if it exists.

    Useful when rerunning experiments with overwrite enabled.

    Args:
        path: File path.
    """
    path = Path(path)

    if path.exists() and path.is_file():
        path.unlink()


def initialize_output_files(cfg: JsonDict) -> JsonDict:
    """
    Prepare output directory and initialize JSONL output files.

    If overwrite is true, old predictions and intermediate files
    will be removed before the experiment starts.

    Args:
        cfg: Experiment config.

    Returns:
        Dictionary of output paths.
    """
    paths = prepare_output_dir(cfg)

    overwrite = bool(cfg.get("experiment", {}).get("overwrite", False))

    if overwrite:
        remove_file_if_exists(paths["predictions"])
        remove_file_if_exists(paths["intermediate"])
        remove_file_if_exists(paths["metrics"])
        remove_file_if_exists(paths["log"])

    save_command(paths["command"])

    return paths


def save_prediction(
    path: str | os.PathLike,
    question_id: Any,
    image_id: Any,
    question: str,
    pred_answer: str,
    clean_answer: str,
    gt_answers: Optional[List[str]] = None,
    score: Optional[float] = None,
    extra: Optional[JsonDict] = None,
) -> None:
    """
    Append one prediction record to predictions.jsonl.

    Args:
        path: predictions.jsonl path.
        question_id: Question ID.
        image_id: Image ID.
        question: Question text.
        pred_answer: Raw predicted answer.
        clean_answer: Normalized or cleaned answer.
        gt_answers: Ground-truth answers.
        score: VQA score for this sample.
        extra: Additional fields.
    """
    record: JsonDict = {
        "question_id": question_id,
        "image_id": image_id,
        "question": question,
        "pred_answer": pred_answer,
        "clean_answer": clean_answer,
        "gt_answers": gt_answers or [],
    }

    if score is not None:
        record["score"] = score

    if extra:
        record.update(extra)

    append_jsonl(record, path)


def save_intermediate(
    path: str | os.PathLike,
    sample: JsonDict,
    concepts: Optional[List[str]] = None,
    expanded_concepts: Optional[List[str]] = None,
    selected_prompts: Optional[List[str]] = None,
    detections: Optional[List[JsonDict]] = None,
    retrieved_passages: Optional[List[JsonDict]] = None,
    reranked_passages: Optional[List[JsonDict]] = None,
    visual_evidence: Optional[List[JsonDict]] = None,
    reasoning_input: Optional[Any] = None,
    raw_reasoning_output: Optional[str] = None,
    pred_answer: Optional[str] = None,
    clean_answer: Optional[str] = None,
    score: Optional[float] = None,
    extra: Optional[JsonDict] = None,
) -> None:
    """
    Append one intermediate record to intermediate.jsonl.

    This file is used for:
    - debugging
    - case study selection
    - error analysis
    - threshold analysis

    Args:
        path: intermediate.jsonl path.
        sample: Original dataset sample.
        concepts: Raw generated concepts.
        expanded_concepts: Knowledge-expanded concepts.
        selected_prompts: Detection prompts.
        detections: Detector outputs.
        retrieved_passages: Retrieved knowledge passages.
        reranked_passages: Reranked knowledge passages.
        visual_evidence: Evidence passed to the reasoner.
        reasoning_input: Serialized input to the LLM reasoner.
        raw_reasoning_output: Raw LLM output.
        pred_answer: Final raw answer.
        clean_answer: Cleaned answer for evaluation.
        score: Per-sample score.
        extra: Additional fields.
    """
    record: JsonDict = {
        "question_id": sample.get("question_id"),
        "image_id": sample.get("image_id"),
        "image_path": sample.get("image_path"),
        "question": sample.get("question"),
        "gt_answers": sample.get("answers", []),
        "choices": sample.get("choices", []),
        "concepts": concepts or [],
        "expanded_concepts": expanded_concepts or [],
        "selected_prompts": selected_prompts or [],
        "detections": detections or [],
        "retrieved_passages": retrieved_passages or [],
        "reranked_passages": reranked_passages or [],
        "visual_evidence": visual_evidence or [],
        "reasoning_input": reasoning_input,
        "raw_reasoning_output": raw_reasoning_output,
        "pred_answer": pred_answer,
        "clean_answer": clean_answer,
        "score": score,
    }

    if extra:
        record.update(extra)

    append_jsonl(record, path)


def save_metrics(
    path: str | os.PathLike,
    metrics: JsonDict,
    cfg: Optional[JsonDict] = None,
) -> None:
    """
    Save metrics.json.

    Args:
        path: metrics output path.
        metrics: Metric dictionary.
        cfg: Optional config. If provided, key experiment fields are saved.
    """
    output: JsonDict = dict(metrics)

    if cfg is not None:
        output["experiment"] = {
            "name": cfg.get("experiment", {}).get("name"),
            "output_dir": cfg.get("experiment", {}).get("output_dir"),
            "dataset_name": cfg.get("data", {}).get("dataset_name"),
            "detector_model_path": cfg.get("detector", {}).get("model_path"),
            "concept_strategy": cfg.get("concept", {}).get("strategy"),
            "retrieval_enabled": cfg.get("retrieval", {}).get("enabled"),
            "reranker_enabled": cfg.get("reranker", {}).get("enabled"),
            "structured_reasoning": cfg.get("reasoning", {}).get("structured"),
        }

    write_json(output, path)


def file_exists(path: str | os.PathLike) -> bool:
    """
    Check whether a file exists.

    Args:
        path: File path.

    Returns:
        True if the path exists and is a file.
    """
    return Path(path).is_file()


def dir_exists(path: str | os.PathLike) -> bool:
    """
    Check whether a directory exists.

    Args:
        path: Directory path.

    Returns:
        True if the path exists and is a directory.
    """
    return Path(path).is_dir()