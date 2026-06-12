"""
Logging utilities for OV-VQA experiments.

This module creates a logger that writes messages to both:
1. terminal stdout
2. experiment log file

The logger is used by scripts and pipeline modules.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "ov_vqa",
    log_file: Optional[str | os.PathLike] = None,
    level: int = logging.INFO,
    reset_handlers: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a logger.

    Args:
        name: Logger name.
        log_file: Optional path to a log file.
        level: Logging level.
        reset_handlers: Whether to remove old handlers before adding new ones.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if reset_handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config_summary(logger: logging.Logger, cfg: dict) -> None:
    """
    Log important experiment configuration fields.

    Args:
        logger: Logger instance.
        cfg: Experiment config dictionary.
    """
    experiment = cfg.get("experiment", {})
    data = cfg.get("data", {})
    detector = cfg.get("detector", {})
    concept = cfg.get("concept", {})
    retrieval = cfg.get("retrieval", {})
    reranker = cfg.get("reranker", {})
    llm = cfg.get("llm", {})
    reasoning = cfg.get("reasoning", {})

    logger.info("=" * 80)
    logger.info("Experiment configuration summary")
    logger.info("=" * 80)

    logger.info("Experiment name: %s", experiment.get("name"))
    logger.info("Output dir: %s", experiment.get("output_dir"))
    logger.info("Seed: %s", experiment.get("seed"))

    logger.info("Dataset: %s", data.get("dataset_name"))
    logger.info("Annotation path: %s", data.get("annotation_path"))
    logger.info("Image root: %s", data.get("image_root"))
    logger.info("Limit: %s", data.get("limit"))

    logger.info("Detector: %s", detector.get("name"))
    logger.info("Detector model path: %s", detector.get("model_path"))
    logger.info("Detector conf threshold: %s", detector.get("conf_threshold"))
    logger.info("Detector IoU threshold: %s", detector.get("iou_threshold"))
    logger.info("Detector max prompts: %s", detector.get("max_prompts"))
    logger.info("Detector max detections: %s", detector.get("max_detections"))

    logger.info("Concept enabled: %s", concept.get("enabled"))
    logger.info("Concept strategy: %s", concept.get("strategy"))
    logger.info("Max selected concepts: %s", concept.get("max_selected_concepts"))

    logger.info("Retrieval enabled: %s", retrieval.get("enabled"))
    logger.info("Retrieval top_k: %s", retrieval.get("top_k"))
    logger.info("Retrieval index path: %s", retrieval.get("index_path"))

    logger.info("Reranker enabled: %s", reranker.get("enabled"))
    logger.info("Reranker top_k: %s", reranker.get("top_k"))

    logger.info("LLM provider: %s", llm.get("provider"))
    logger.info("LLM model name: %s", llm.get("model_name"))

    logger.info("Structured reasoning: %s", reasoning.get("structured"))
    logger.info("Use answer candidates: %s", reasoning.get("use_answer_candidates"))
    logger.info("Short answer: %s", reasoning.get("short_answer"))

    logger.info("=" * 80)


def log_sample_start(
    logger: logging.Logger,
    index: int,
    total: int,
    question_id: object,
    image_id: object,
    question: str,
) -> None:
    """
    Log the start of one sample.

    Args:
        logger: Logger instance.
        index: Current sample index, starting from 1.
        total: Total number of samples.
        question_id: Question ID.
        image_id: Image ID.
        question: Question text.
    """
    logger.info("-" * 80)
    logger.info(
        "[%d/%d] question_id=%s image_id=%s",
        index,
        total,
        question_id,
        image_id,
    )
    logger.info("Question: %s", question)


def log_sample_result(
    logger: logging.Logger,
    pred_answer: str,
    clean_answer: str,
    gt_answers: list | None = None,
    score: float | None = None,
) -> None:
    """
    Log one sample result.

    Args:
        logger: Logger instance.
        pred_answer: Raw predicted answer.
        clean_answer: Cleaned answer.
        gt_answers: Ground-truth answers.
        score: Optional per-sample score.
    """
    logger.info("Raw answer: %s", pred_answer)
    logger.info("Clean answer: %s", clean_answer)

    if gt_answers is not None:
        logger.info("GT answers: %s", gt_answers)

    if score is not None:
        logger.info("Score: %.4f", score)


def log_final_metrics(logger: logging.Logger, metrics: dict) -> None:
    """
    Log final experiment metrics.

    Args:
        logger: Logger instance.
        metrics: Metrics dictionary.
    """
    logger.info("=" * 80)
    logger.info("Final metrics")
    logger.info("=" * 80)

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info("%s: %.6f", key, value)
        else:
            logger.info("%s: %s", key, value)

    logger.info("=" * 80)


def get_log_level(level_name: str | None) -> int:
    """
    Convert a string log level to logging level.

    Args:
        level_name: String such as 'info', 'debug', 'warning'.

    Returns:
        logging level.
    """
    if level_name is None:
        return logging.INFO

    level_name = level_name.lower()

    mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if level_name not in mapping:
        raise ValueError(
            f"Unsupported log level: {level_name}. "
            f"Choose from {list(mapping.keys())}."
        )

    return mapping[level_name]