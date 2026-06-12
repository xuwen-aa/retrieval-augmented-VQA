"""YOLO-World wrapper for RefCOCOg grounding adaptation.

This module intentionally keeps the data interface region-level:
    image + referring expression + ground-truth box

It does NOT use BGE embeddings. BGE should be used in the retrieval stage, while
YOLO-World fine-tuning should rely on the detector's own text-conditioning path.

The actual Ultralytics/YOLO-World training API may differ depending on the
installed package version. Therefore, this wrapper focuses on a stable adapter
interface and provides utility methods for prompt-conditioned inference and
training-target formatting. The training loop should call these methods rather
than treating YOLO-World as a global image-text contrastive model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from ultralytics import YOLO, YOLOWorld


Tensor = torch.Tensor


@dataclass
class GroundingBatch:
    """A batch for phrase-conditioned grounding.

    Attributes:
        images: Tensor of shape [B, 3, H, W].
        boxes: List of tensors. Each tensor has shape [Ni, 4] in xyxy format.
        phrases: List of phrase lists. phrases[i] has length Ni.
        image_ids: Optional image ids.
        file_names: Optional image file names.
    """

    images: Tensor
    boxes: List[Tensor]
    phrases: List[List[str]]
    image_ids: Optional[List[Union[int, str]]] = None
    file_names: Optional[List[str]] = None


class YOLOWorldGroundingAdapter(nn.Module):
    """Adapter around YOLO-World for RefCOCOg-style grounding fine-tuning.

    The previous version used global image-text contrastive learning, where one
    whole-image feature was matched with one sentence embedding. That is not a
    faithful implementation of RefCOCOg grounding. This adapter preserves the
    region-level supervision needed by our paper: each phrase is associated with
    a target bounding box.
    """

    def __init__(
        self,
        base_model_path: str = "yolov8s-world.pt",
        freeze_backbone: bool = False,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()

        # Prefer YOLOWorld when available because it exposes prompt-conditioned
        # open-vocabulary detection. If a custom checkpoint is passed, YOLOWorld
        # can still load standard YOLO-World weights in recent Ultralytics builds.
        self.detector = YOLOWorld(base_model_path)
        self.model = self.detector.model

        if freeze_backbone:
            self.freeze_backbone()
        if freeze_text_encoder:
            self.freeze_text_encoder()

    def freeze_backbone(self) -> None:
        """Freeze non-head parameters as a conservative adaptation setting."""
        for name, param in self.model.named_parameters():
            # Keep detection/head-related parameters trainable when their names
            # expose a head/detect pattern. Ultralytics names vary by version, so
            # this is deliberately conservative.
            trainable = any(key in name.lower() for key in ["detect", "head", "cv3"])
            param.requires_grad = trainable

    def freeze_text_encoder(self) -> None:
        """Freeze text-side parameters if the loaded model exposes them."""
        for name, param in self.model.named_parameters():
            if any(key in name.lower() for key in ["text", "clip", "language", "txt"]):
                param.requires_grad = False

    @staticmethod
    def unique_phrases(phrases: Sequence[Sequence[str]]) -> List[str]:
        """Collect unique prompts while preserving order."""
        seen = set()
        out: List[str] = []
        for phrase_list in phrases:
            for phrase in phrase_list:
                phrase = str(phrase).strip().lower()
                if phrase and phrase not in seen:
                    seen.add(phrase)
                    out.append(phrase)
        return out

    def set_prompts(self, phrases: Sequence[str]) -> None:
        """Set open-vocabulary classes/prompts for YOLO-World."""
        classes = [str(p).strip().lower() for p in phrases if str(p).strip()]
        if not classes:
            raise ValueError("At least one non-empty phrase is required.")
        self.detector.set_classes(classes)

    def format_targets(self, batch: GroundingBatch) -> List[Dict[str, Any]]:
        """Format phrase-box supervision for a trainer.

        Returned targets keep phrase strings instead of forcing them into fixed
        COCO class ids. The trainer can map each phrase to the prompt index after
        calling `set_prompts`.
        """
        targets: List[Dict[str, Any]] = []
        for i, (boxes_i, phrases_i) in enumerate(zip(batch.boxes, batch.phrases)):
            if len(boxes_i) != len(phrases_i):
                raise ValueError(
                    f"Image {i} has {len(boxes_i)} boxes but {len(phrases_i)} phrases."
                )
            targets.append(
                {
                    "boxes": boxes_i.float(),
                    "phrases": [str(p).strip().lower() for p in phrases_i],
                    "image_id": None if batch.image_ids is None else batch.image_ids[i],
                    "file_name": None if batch.file_names is None else batch.file_names[i],
                }
            )
        return targets

    def forward(self, images: Tensor, phrases: Optional[Sequence[str]] = None) -> Any:
        """Run prompt-conditioned detection.

        For training, the exact loss call depends on the installed Ultralytics
        YOLO-World version. The trainer should use `format_targets` and the
        underlying `self.model` loss API. For evaluation/inference, this forward
        returns raw model predictions.
        """
        if phrases is not None:
            self.set_prompts(phrases)
        return self.model(images)

    @torch.no_grad()
    def predict_with_phrases(
        self,
        image: Union[str, Tensor],
        phrases: Sequence[str],
        conf: float = 0.25,
        iou: float = 0.7,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Any:
        """Convenience method for phrase-conditioned YOLO-World inference."""
        self.set_prompts(phrases)
        kwargs: Dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        return self.detector.predict(image, **kwargs)


def build_yoloworld_grounding_model(
    base_model_path: str = "yolov8s-world.pt",
    freeze_backbone: bool = False,
    freeze_text_encoder: bool = True,
) -> YOLOWorldGroundingAdapter:
    """Factory used by main.py/trainer.py."""
    return YOLOWorldGroundingAdapter(
        base_model_path=base_model_path,
        freeze_backbone=freeze_backbone,
        freeze_text_encoder=freeze_text_encoder,
    )
