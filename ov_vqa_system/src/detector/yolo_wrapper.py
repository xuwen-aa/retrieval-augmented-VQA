

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


Detection = Dict[str, Any]


@dataclass
class YoloWorldConfig:
    """
    Configuration for YOLO-World inference.
    """

    model_path: str
    device: str = "cuda:0"
    image_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.70
    max_detections: int = 20
    max_prompts: int = 5


class YoloWorldWrapper:
    """
    Wrapper for YOLO-World open-vocabulary detection.

    This class supports both:
    - original YOLO-World checkpoint, e.g. yolov8s-world.pt
    - RefCOCOg-adapted checkpoint, e.g. detector_refcocog/weights/best.pt
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        device: str = "cuda:0",
        image_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        max_detections: int = 20,
        max_prompts: int = 5,
        lazy_load: bool = False,
    ) -> None:
        """
        Args:
            model_path: Path to YOLO-World checkpoint.
            device: Device string, e.g. cuda:0 or cpu.
            image_size: Inference image size.
            conf_threshold: Confidence threshold.
            iou_threshold: NMS IoU threshold.
            max_detections: Maximum number of detections to return.
            max_prompts: Maximum number of text prompts used per image.
            lazy_load: If True, delay model loading until first detection.
        """
        self.config = YoloWorldConfig(
            model_path=str(model_path),
            device=device,
            image_size=image_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_prompts=max_prompts,
        )

        self.model = None

        if not lazy_load:
            self._load_model()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "YoloWorldWrapper":
        """
        Build wrapper from experiment config.

        Args:
            cfg: Full experiment config.

        Returns:
            YoloWorldWrapper instance.
        """
        detector_cfg = cfg.get("detector", {})

        return cls(
            model_path=detector_cfg.get("model_path"),
            device=detector_cfg.get("device", "cuda:0"),
            image_size=int(detector_cfg.get("image_size", 640)),
            conf_threshold=float(detector_cfg.get("conf_threshold", 0.25)),
            iou_threshold=float(detector_cfg.get("iou_threshold", 0.70)),
            max_detections=int(detector_cfg.get("max_detections", 20)),
            max_prompts=int(detector_cfg.get("max_prompts", 5)),
        )

    def _load_model(self) -> None:
        """
        Load YOLO-World model.

        We first try YOLOWorld. If unavailable or incompatible,
        we fall back to YOLO. This makes the wrapper more robust
        across Ultralytics versions and fine-tuned checkpoints.
        """
        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO-World model checkpoint not found: {model_path}"
            )

        try:
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(str(model_path))
            self.model_type = "YOLOWorld"
            return

        except Exception:
            pass

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(model_path))
            self.model_type = "YOLO"
            return

        except Exception as e:
            raise RuntimeError(
                "Failed to load YOLO-World checkpoint. "
                "Please make sure ultralytics is installed and the checkpoint is valid. "
                f"Checkpoint: {model_path}"
            ) from e

    @staticmethod
    def _normalize_device(device: str) -> str:
        """
        Normalize device string for Ultralytics.

        Ultralytics commonly accepts:
        - "0"
        - "cpu"
        - "cuda:0" may work in some versions, but "0" is safer.

        Args:
            device: Raw device string.

        Returns:
            Normalized device string.
        """
        if device is None:
            return "0"

        device = str(device)

        if device.startswith("cuda:"):
            return device.split("cuda:", 1)[1]

        if device == "cuda":
            return "0"

        return device
    @staticmethod
    def _torch_device(device: str) -> str:
        """
        Convert config device string to torch device string.

        Args:
            device: Raw device, e.g. cuda:0, cuda, 0, cpu.

        Returns:
            Torch device string.
        """
        if device is None:
            return "cuda:0"

        device = str(device)

        if device == "cpu":
            return "cpu"

        if device.startswith("cuda"):
            return device

        if device.isdigit():
            return f"cuda:{device}"

        return device

    def _move_model_to_device(self, torch_device: str) -> None:
        """
        Move YOLO-World model and text encoder to the same device.

        This fixes YOLO-World set_classes device mismatch:
            token ids on cuda, CLIP/text embedding on cpu
        """
        if self.model is None:
            self._load_model()

        # Move outer Ultralytics model.
        try:
            self.model.to(torch_device)
        except Exception:
            pass

        # Move inner PyTorch model.
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None:
            try:
                inner_model.to(torch_device)
            except Exception:
                pass

            # YOLO-World may cache CLIP/text encoder in different attributes
            # depending on Ultralytics version.
            possible_text_attrs = [
                "clip_model",
                "text_model",
                "txt_model",
                "model",
            ]

            for attr in possible_text_attrs:
                module = getattr(inner_model, attr, None)
                if module is not None and hasattr(module, "to"):
                    try:
                        module.to(torch_device)
                    except Exception:
                        pass
    @staticmethod
    def _clean_prompts(prompts: Sequence[str], max_prompts: int) -> List[str]:
        """
        Clean and truncate detection prompts.

        Args:
            prompts: Raw prompt strings.
            max_prompts: Maximum number of prompts.

        Returns:
            Cleaned prompt list.
        """
        cleaned = []
        seen = set()

        for prompt in prompts:
            if prompt is None:
                continue

            prompt = str(prompt).strip()
            prompt = " ".join(prompt.split())

            if not prompt:
                continue

            key = prompt.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(prompt)

            if len(cleaned) >= max_prompts:
                break

        return cleaned

    def _set_classes(self, prompts: List[str]) -> None:
        """
        Set YOLO-World text classes.

        Args:
            prompts: Detection prompt list.
        """
        if self.model is None:
            self._load_model()

        if not prompts:
            return

        torch_device = self._torch_device(self.config.device)

        # Important: move YOLO-World and text encoder before set_classes.
        self._move_model_to_device(torch_device)

        if hasattr(self.model, "set_classes"):
            try:
                self.model.set_classes(prompts)
                return
            except RuntimeError as e:
                error_text = str(e)

                # Retry once after forcing all internal modules to target device.
                if "Expected all tensors to be on the same device" in error_text:
                    self._move_model_to_device(torch_device)
                    self.model.set_classes(prompts)
                    return

                raise

        return

    @staticmethod
    def _xyxy_to_xywh(bbox_xyxy: List[float]) -> List[float]:
        """
        Convert xyxy box to xywh.

        Args:
            bbox_xyxy: [x1, y1, x2, y2]

        Returns:
            [x, y, w, h]
        """
        x1, y1, x2, y2 = bbox_xyxy
        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def _clip_bbox_xyxy(
        bbox_xyxy: List[float],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> List[float]:
        """
        Clip bounding box to image boundary if width and height are known.

        Args:
            bbox_xyxy: [x1, y1, x2, y2]
            width: Image width.
            height: Image height.

        Returns:
            Clipped bbox.
        """
        x1, y1, x2, y2 = bbox_xyxy

        if width is not None:
            x1 = max(0.0, min(float(width), x1))
            x2 = max(0.0, min(float(width), x2))

        if height is not None:
            y1 = max(0.0, min(float(height), y1))
            y2 = max(0.0, min(float(height), y2))

        return [float(x1), float(y1), float(x2), float(y2)]

    @staticmethod
    def _get_result_image_size(result: Any) -> tuple[Optional[int], Optional[int]]:
        """
        Get image width and height from an Ultralytics result.

        Args:
            result: Ultralytics result object.

        Returns:
            (width, height), either can be None.
        """
        orig_shape = getattr(result, "orig_shape", None)

        if orig_shape is None:
            return None, None

        if len(orig_shape) >= 2:
            height, width = orig_shape[:2]
            return int(width), int(height)

        return None, None

    def _class_id_to_label(
        self,
        result: Any,
        class_id: int,
        prompts: List[str],
    ) -> str:
        """
        Convert class id to label.

        Priority:
        1. result.names
        2. prompts[class_id]
        3. class_{id}

        Args:
            result: Ultralytics result object.
            class_id: Predicted class id.
            prompts: Prompt list.

        Returns:
            Label string.
        """
        names = getattr(result, "names", None)

        if isinstance(names, dict) and class_id in names:
            return str(names[class_id])

        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])

        if 0 <= class_id < len(prompts):
            return prompts[class_id]

        return f"class_{class_id}"

    def _parse_result(
        self,
        result: Any,
        prompts: List[str],
    ) -> List[Detection]:
        """
        Parse one Ultralytics result object.

        Args:
            result: Ultralytics result.
            prompts: Detection prompts.

        Returns:
            List of detection dictionaries.
        """
        detections: List[Detection] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        width, height = self._get_result_image_size(result)

        xyxy_tensor = getattr(boxes, "xyxy", None)
        conf_tensor = getattr(boxes, "conf", None)
        cls_tensor = getattr(boxes, "cls", None)

        if xyxy_tensor is None or conf_tensor is None or cls_tensor is None:
            return detections

        xyxy_list = xyxy_tensor.detach().cpu().tolist()
        conf_list = conf_tensor.detach().cpu().tolist()
        cls_list = cls_tensor.detach().cpu().tolist()

        for bbox_xyxy, confidence, class_id in zip(xyxy_list, conf_list, cls_list):
            confidence = float(confidence)

            if confidence < self.config.conf_threshold:
                continue

            class_id = int(class_id)
            label = self._class_id_to_label(
                result=result,
                class_id=class_id,
                prompts=prompts,
            )

            bbox_xyxy = [float(x) for x in bbox_xyxy]
            bbox_xyxy = self._clip_bbox_xyxy(
                bbox_xyxy=bbox_xyxy,
                width=width,
                height=height,
            )
            bbox_xywh = self._xyxy_to_xywh(bbox_xyxy)

            prompt = prompts[class_id] if 0 <= class_id < len(prompts) else label

            detection = {
                "label": label,
                "prompt": prompt,
                "class_id": class_id,
                "confidence": confidence,
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": bbox_xywh,
                "source": "yolo_world",
            }

            detections.append(detection)

        detections.sort(key=lambda x: x["confidence"], reverse=True)

        return detections

    def detect(
        self,
        image_path: str | os.PathLike,
        prompts: Sequence[str],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> List[Detection]:
        """
        Run open-vocabulary detection on one image.

        Args:
            image_path: Image file path.
            prompts: Text prompts/classes for YOLO-World.
            conf_threshold: Optional override confidence threshold.
            iou_threshold: Optional override IoU threshold.
            max_detections: Optional override maximum detections.

        Returns:
            List of detections.
        """
        if self.model is None:
            self._load_model()

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        prompts = self._clean_prompts(
            prompts=prompts,
            max_prompts=self.config.max_prompts,
        )

        if not prompts:
            return []

        conf = (
            float(conf_threshold)
            if conf_threshold is not None
            else self.config.conf_threshold
        )
        iou = (
            float(iou_threshold)
            if iou_threshold is not None
            else self.config.iou_threshold
        )
        max_det = (
            int(max_detections)
            if max_detections is not None
            else self.config.max_detections
        )

        self._set_classes(prompts)

        device = self._normalize_device(self.config.device)

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.config.image_size,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )

        all_detections: List[Detection] = []

        for result in results:
            detections = self._parse_result(result=result, prompts=prompts)
            all_detections.extend(detections)

        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        if max_det is not None:
            all_detections = all_detections[:max_det]

        return all_detections

    def detect_from_sample(
        self,
        sample: Dict[str, Any],
        prompts: Sequence[str],
    ) -> List[Detection]:
        """
        Run detection using unified VQA sample.

        Args:
            sample: Unified VQA sample.
            prompts: Detection prompts.

        Returns:
            List of detections.
        """
        image_path = sample.get("image_path")

        if image_path is None:
            raise ValueError("Sample does not contain image_path.")

        return self.detect(
            image_path=image_path,
            prompts=prompts,
        )

    def __call__(
        self,
        image_path: str | os.PathLike,
        prompts: Sequence[str],
    ) -> List[Detection]:
        """
        Alias for detect().
        """
        return self.detect(image_path=image_path, prompts=prompts)


def build_detector(cfg: Dict[str, Any]) -> YoloWorldWrapper:
    """
    Build detector from experiment config.

    Args:
        cfg: Full experiment config.

    Returns:
        YoloWorldWrapper.
    """
    return YoloWorldWrapper.from_config(cfg)