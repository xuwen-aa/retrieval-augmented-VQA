"""
Dataset builder for OV-VQA experiments.

This module creates dataset objects from experiment config.
Currently supported:
- OK-VQA

Reserved:
- A-OKVQA
"""

from __future__ import annotations

from typing import Any, Dict

from src.datasets.okvqa_dataset import OKVQADataset
from src.datasets.aokvqa_dataset import AOKVQADataset


def build_dataset(cfg: Dict[str, Any]):
    """
    Build a dataset from config.

    Args:
        cfg: Experiment configuration dictionary.

    Returns:
        Dataset object.

    Raises:
        ValueError: If dataset_name is unsupported.
    """
    data_cfg = cfg.get("data", {})

    dataset_name = str(data_cfg.get("dataset_name", "")).lower()
    annotation_path = data_cfg.get("annotation_path")
    image_root = data_cfg.get("image_root")
    split = data_cfg.get("split", "val")
    limit = data_cfg.get("limit")
    check_image_exists = bool(data_cfg.get("check_image_exists", False))

    if annotation_path is None:
        raise ValueError("data.annotation_path is required.")

    if image_root is None:
        raise ValueError("data.image_root is required.")

    if dataset_name in {"okvqa", "ok-vqa"}:
        return OKVQADataset(
            annotation_path=annotation_path,
            image_root=image_root,
            split=split,
            limit=limit,
            check_image_exists=check_image_exists,
            question_file=data_cfg.get("question_file"),
            annotation_file=data_cfg.get("answer_annotation_file"),
            coco_split=data_cfg.get("coco_split"),
        )

    if dataset_name in {"aokvqa", "a-okvqa", "a_okvqa"}:
        return AOKVQADataset(
            annotation_path=annotation_path,
            image_root=image_root,
            split=split,
            limit=limit,
            check_image_exists=check_image_exists,
        )

    raise ValueError(
        f"Unsupported dataset_name: {dataset_name}. "
        "Currently supported: okvqa. A-OKVQA is reserved but not implemented."
    )


def dataset_summary(dataset) -> Dict[str, Any]:
    """
    Return a dataset summary dictionary.

    Args:
        dataset: Dataset object.

    Returns:
        Summary dictionary.
    """
    if hasattr(dataset, "summary"):
        return dataset.summary()

    return {
        "dataset": dataset.__class__.__name__,
        "num_samples": len(dataset),
    }