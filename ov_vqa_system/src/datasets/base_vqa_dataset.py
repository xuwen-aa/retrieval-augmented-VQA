"""
Base dataset interface for OV-VQA experiments.

This module defines a unified sample format for all VQA datasets.

Unified sample format:
{
    "question_id": str | int,
    "image_id": str | int,
    "image_path": str,
    "question": str,
    "answers": list[str],
    "choices": list[str],
    "metadata": dict,
}
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


VQASample = Dict[str, Any]


class BaseVQADataset(ABC):
    """
    Base class for VQA datasets.

    Subclasses should implement:
    - _load_samples()
    - _build_image_path()
    """

    def __init__(
        self,
        annotation_path: str | os.PathLike,
        image_root: str | os.PathLike,
        split: str = "val",
        limit: Optional[int] = None,
        check_image_exists: bool = False,
    ) -> None:
        """
        Args:
            annotation_path: Path to dataset annotation file.
            image_root: Root directory of images.
            split: Dataset split name.
            limit: Optional number of samples for smoke tests.
            check_image_exists: Whether to verify image files exist.
        """
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.split = split
        self.limit = limit
        self.check_image_exists = check_image_exists

        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_path}"
            )

        if not self.image_root.exists():
            raise FileNotFoundError(
                f"Image root not found: {self.image_root}"
            )

        samples = self._load_samples()

        if limit is not None:
            samples = samples[:limit]

        if check_image_exists:
            self._check_images(samples)

        self.samples = samples

    @abstractmethod
    def _load_samples(self) -> List[VQASample]:
        """
        Load and convert raw annotations into unified VQA samples.

        Returns:
            List of unified VQA samples.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_image_path(self, image_id: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Build image path from image_id.

        Args:
            image_id: Image ID.
            metadata: Optional raw metadata.

        Returns:
            Absolute or root-relative image path.
        """
        raise NotImplementedError

    def _check_images(self, samples: List[VQASample]) -> None:
        """
        Check whether image files exist.

        Args:
            samples: List of samples.

        Raises:
            FileNotFoundError: If any image is missing.
        """
        missing = []

        for sample in samples:
            image_path = Path(sample["image_path"])
            if not image_path.exists():
                missing.append(str(image_path))

            if len(missing) >= 10:
                break

        if missing:
            raise FileNotFoundError(
                "Some image files are missing. Examples:\n"
                + "\n".join(missing)
            )

    def __len__(self) -> int:
        """
        Returns:
            Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> VQASample:
        """
        Get one sample.

        Args:
            index: Sample index.

        Returns:
            Unified VQA sample.
        """
        return self.samples[index]

    def __iter__(self) -> Iterator[VQASample]:
        """
        Iterate over samples.

        Yields:
            Unified VQA samples.
        """
        return iter(self.samples)

    def get_samples(self) -> List[VQASample]:
        """
        Returns:
            All samples.
        """
        return self.samples

    @staticmethod
    def normalize_image_id(image_id: Any) -> str:
        """
        Normalize image ID to a string.

        Args:
            image_id: Raw image ID.

        Returns:
            Normalized image ID string.
        """
        return str(image_id)

    @staticmethod
    def coco_image_name(image_id: Any, split: str = "val2014") -> str:
        """
        Build standard COCO image filename.

        Examples:
            image_id=123, split=val2014
            -> COCO_val2014_000000000123.jpg

        Args:
            image_id: COCO image ID.
            split: COCO split name, such as train2014 or val2014.

        Returns:
            COCO image filename.
        """
        image_id_int = int(image_id)
        return f"COCO_{split}_{image_id_int:012d}.jpg"

    @staticmethod
    def ensure_list(value: Any) -> List[Any]:
        """
        Convert a value to a list.

        Args:
            value: Raw value.

        Returns:
            List value.
        """
        if value is None:
            return []

        if isinstance(value, list):
            return value

        return [value]

    @staticmethod
    def normalize_answers(answers: Any) -> List[str]:
        """
        Convert dataset-specific answers into a list of strings.

        Supports:
        - ["cat", "kitty"]
        - [{"answer": "cat"}, {"answer": "kitty"}]
        - "cat"

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

        return [ans.strip() for ans in output if str(ans).strip()]

    @staticmethod
    def make_sample(
        question_id: Any,
        image_id: Any,
        image_path: str | os.PathLike,
        question: str,
        answers: Optional[List[str]] = None,
        choices: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VQASample:
        """
        Create a unified VQA sample.

        Args:
            question_id: Question ID.
            image_id: Image ID.
            image_path: Image path.
            question: Question text.
            answers: Ground-truth answers.
            choices: Multiple-choice candidates.
            metadata: Additional raw fields.

        Returns:
            Unified sample dictionary.
        """
        return {
            "question_id": question_id,
            "image_id": image_id,
            "image_path": str(image_path),
            "question": str(question).strip(),
            "answers": answers or [],
            "choices": choices or [],
            "metadata": metadata or {},
        }

    def summary(self) -> Dict[str, Any]:
        """
        Return dataset summary.

        Returns:
            Dataset summary dictionary.
        """
        return {
            "dataset": self.__class__.__name__,
            "annotation_path": str(self.annotation_path),
            "image_root": str(self.image_root),
            "split": self.split,
            "num_samples": len(self.samples),
            "limit": self.limit,
        }