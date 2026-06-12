from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.datasets.base_vqa_dataset import BaseVQADataset, VQASample


class OKVQADataset(BaseVQADataset):
    """
    OK-VQA dataset loader.

    Supports AutoDL image layout:
        /root/autodl-tmp/data/common_images/000000297147.jpg

    Also supports original COCO layout:
        /root/autodl-tmp/data/common_images/COCO_val2014_000000297147.jpg
        /root/autodl-tmp/data/common_images/val2014/COCO_val2014_000000297147.jpg
    """

    QUESTION_FILENAME_CANDIDATES = [
        "OpenEnded_mscoco_val2014_questions.json",
        "OpenEnded_mscoco_train2014_questions.json",
        "questions.json",
        "okvqa_questions.json",
    ]

    ANNOTATION_FILENAME_CANDIDATES = [
        "mscoco_val2014_annotations.json",
        "mscoco_train2014_annotations.json",
        "annotations.json",
        "okvqa_annotations.json",
    ]

    def __init__(
        self,
        annotation_path: str | os.PathLike,
        image_root: str | os.PathLike,
        split: str = "val",
        limit: Optional[int] = None,
        check_image_exists: bool = False,
        question_file: Optional[str | os.PathLike] = None,
        annotation_file: Optional[str | os.PathLike] = None,
        coco_split: Optional[str] = None,
    ) -> None:
        """
        Args:
            annotation_path:
                Either the raw OK-VQA directory, such as data/raw/vqa,
                or a specific question json file.
            image_root:
                Root directory of images.
            split:
                Dataset split. Usually "val" for OK-VQA validation.
            limit:
                Optional sample limit for smoke tests.
            check_image_exists:
                Whether to check image file existence.
            question_file:
                Optional explicit question file path.
            annotation_file:
                Optional explicit annotation file path.
            coco_split:
                COCO split name, e.g. val2014 or train2014.
        """
        self.raw_annotation_path = Path(annotation_path)
        self.question_file = Path(question_file) if question_file else None
        self.answer_annotation_file = Path(annotation_file) if annotation_file else None
        self.coco_split = coco_split

        super().__init__(
            annotation_path=annotation_path,
            image_root=image_root,
            split=split,
            limit=limit,
            check_image_exists=check_image_exists,
        )

    def _resolve_files(self) -> Tuple[Path, Path]:
        """
        Resolve OK-VQA question file and annotation file.

        Returns:
            (question_file, annotation_file)
        """
        if self.question_file is not None and self.answer_annotation_file is not None:
            if not self.question_file.exists():
                raise FileNotFoundError(f"Question file not found: {self.question_file}")

            if not self.answer_annotation_file.exists():
                raise FileNotFoundError(
                    f"Annotation file not found: {self.answer_annotation_file}"
                )

            return self.question_file, self.answer_annotation_file

        path = self.raw_annotation_path

        if path.is_dir():
            question_file = self._find_file(path, self.QUESTION_FILENAME_CANDIDATES)
            annotation_file = self._find_file(path, self.ANNOTATION_FILENAME_CANDIDATES)

            if question_file is None:
                raise FileNotFoundError(
                    f"Could not find OK-VQA question file under {path}. "
                    f"Expected one of: {self.QUESTION_FILENAME_CANDIDATES}"
                )

            if annotation_file is None:
                raise FileNotFoundError(
                    f"Could not find OK-VQA annotation file under {path}. "
                    f"Expected one of: {self.ANNOTATION_FILENAME_CANDIDATES}"
                )

            return question_file, annotation_file

        if path.is_file():
            if self.answer_annotation_file is None:
                raise FileNotFoundError(
                    "annotation_path points to a file, but annotation_file was not "
                    "provided. For OK-VQA, both question and annotation files are needed."
                )

            if not self.answer_annotation_file.exists():
                raise FileNotFoundError(
                    f"Annotation file not found: {self.answer_annotation_file}"
                )

            return path, self.answer_annotation_file

        raise FileNotFoundError(f"OK-VQA annotation path not found: {path}")

    @staticmethod
    def _find_file(directory: Path, candidates: List[str]) -> Optional[Path]:
        """
        Find the first existing file from candidate names.

        Args:
            directory: Directory to search.
            candidates: Candidate filenames.

        Returns:
            Found path or None.
        """
        for filename in candidates:
            path = directory / filename
            if path.exists():
                return path

        return None

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        """
        Read a JSON file.

        Args:
            path: JSON file path.

        Returns:
            Parsed JSON dictionary.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}")

        return data

    def _load_samples(self) -> List[VQASample]:
        """
        Load OK-VQA samples and convert them into unified format.

        Returns:
            List of unified VQA samples.
        """
        question_file, annotation_file = self._resolve_files()

        questions_json = self._read_json(question_file)
        annotations_json = self._read_json(annotation_file)

        raw_questions = questions_json.get("questions", [])
        raw_annotations = annotations_json.get("annotations", [])

        if not raw_questions:
            raise ValueError(f"No questions found in {question_file}")

        if not raw_annotations:
            raise ValueError(f"No annotations found in {annotation_file}")

        annotation_by_qid = {
            ann["question_id"]: ann
            for ann in raw_annotations
            if "question_id" in ann
        }

        samples: List[VQASample] = []

        for q in raw_questions:
            question_id = q.get("question_id")
            image_id = q.get("image_id")
            question = q.get("question", "")

            if question_id is None or image_id is None:
                continue

            ann = annotation_by_qid.get(question_id, {})
            answers = self.normalize_answers(ann.get("answers", []))

            image_path = self._build_image_path(
                image_id=image_id,
                metadata={
                    "question": q,
                    "annotation": ann,
                },
            )

            sample = self.make_sample(
                question_id=question_id,
                image_id=image_id,
                image_path=image_path,
                question=question,
                answers=answers,
                choices=[],
                metadata={
                    "dataset": "okvqa",
                    "split": self.split,
                    "question_file": str(question_file),
                    "annotation_file": str(annotation_file),
                    "raw_question": q,
                    "raw_annotation": ann,
                },
            )

            samples.append(sample)

        return samples

    def _build_image_path(
        self,
        image_id: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build image path for OK-VQA.

        AutoDL common_images stores COCO images as:
            000000297147.jpg

        Original COCO may store images as:
            COCO_val2014_000000297147.jpg
        """
        coco_split = self.coco_split

        if coco_split is None:
            if self.split.lower().startswith("train"):
                coco_split = "train2014"
            else:
                coco_split = "val2014"

        image_id_int = int(image_id)

        plain_name = f"{image_id_int:012d}.jpg"
        coco_name = self.coco_image_name(image_id_int, split=coco_split)

        candidates = [
            # AutoDL common_images format
            self.image_root / plain_name,
            self.image_root / "images" / plain_name,

            # Original COCO format
            self.image_root / coco_name,
            self.image_root / coco_split / coco_name,
            self.image_root / "images" / coco_name,
            self.image_root / "images" / coco_split / coco_name,

            # Plain name inside split folders
            self.image_root / coco_split / plain_name,
            self.image_root / "images" / coco_split / plain_name,
        ]

        # Try both val2014 and train2014 variants just in case.
        for alt_split in ["val2014", "train2014"]:
            alt_coco_name = self.coco_image_name(image_id_int, split=alt_split)
            candidates.extend(
                [
                    self.image_root / alt_coco_name,
                    self.image_root / alt_split / alt_coco_name,
                    self.image_root / "images" / alt_split / alt_coco_name,
                    self.image_root / alt_split / plain_name,
                    self.image_root / "images" / alt_split / plain_name,
                ]
            )

        for path in candidates:
            if path.exists():
                return str(path)

        # Fallback to AutoDL expected path.
        return str(self.image_root / plain_name)

    def summary(self) -> Dict[str, Any]:
        """
        Return dataset summary.

        Returns:
            Summary dictionary.
        """
        base = super().summary()

        try:
            question_file, annotation_file = self._resolve_files()
            base["question_file"] = str(question_file)
            base["answer_annotation_file"] = str(annotation_file)
        except Exception:
            base["question_file"] = None
            base["answer_annotation_file"] = None

        return base