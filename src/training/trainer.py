from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLOWorld


@dataclass
class GroundingTrainConfig:
    """Configuration for RefCOCOg-to-YOLO training conversion."""

    processed_json: str
    image_root: str
    work_dir: str = "./runs/refcocog_yoloworld_data"
    split_name: str = "train"
    epochs: int = 10
    imgsz: int = 640
    batch: int = 8
    lr0: float = 1e-4
    device: str | int = 0
    workers: int = 4
    project: str = "./runs/yoloworld_finetune"
    name: str = "refcocog_grounding"
    copy_images: bool = False


class RefCOCOgYOLOFormatter:
    """
    Convert RefCOCOg phrase-box samples to a YOLO-style directory.

    YOLO labels require numeric class ids. RefCOCOg expressions are free-form,
    so we use one pseudo class, "referring_object", for box supervision. The
    free-form phrases are saved in a sidecar JSONL file and can be used by
    custom YOLO-World prompt logic or evaluation code.

    This is a practical compromise:
      - detection loss is grounded by the RefCOCOg boxes;
      - phrase supervision is preserved explicitly;
      - no BGE or global image-text contrastive objective is mixed into this
        grounding module.
    """

    def __init__(self, cfg: GroundingTrainConfig):
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)
        self.images_dir = self.work_dir / "images" / cfg.split_name
        self.labels_dir = self.work_dir / "labels" / cfg.split_name
        self.phrase_file = self.work_dir / f"phrases_{cfg.split_name}.jsonl"
        self.data_yaml = self.work_dir / "data.yaml"

    @staticmethod
    def _xyxy_to_yolo(
        box_xyxy: Sequence[float], image_w: int, image_h: int
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = map(float, box_xyxy)
        x1 = max(0.0, min(x1, image_w - 1.0))
        y1 = max(0.0, min(y1, image_h - 1.0))
        x2 = max(0.0, min(x2, image_w - 1.0))
        y2 = max(0.0, min(y2, image_h - 1.0))

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        return cx / image_w, cy / image_h, bw / image_w, bh / image_h

    def _resolve_image_path(self, file_name: str, image_id: int | str) -> Path:
        candidates = []
        if file_name:
            candidates.append(Path(self.cfg.image_root) / file_name)

        image_id_str = str(image_id).zfill(12)
        candidates.extend(
            [
                Path(self.cfg.image_root) / f"{image_id_str}.jpg",
                Path(self.cfg.image_root) / f"COCO_train2014_{image_id_str}.jpg",
                Path(self.cfg.image_root) / f"COCO_val2014_{image_id_str}.jpg",
            ]
        )

        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Cannot find image for image_id={image_id}, file_name={file_name}. "
            f"Checked: {[str(p) for p in candidates]}"
        )

    def build(self) -> str:
        with open(self.cfg.processed_json, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Group phrase-box records by image, because YOLO expects one label file
        # per image.
        by_image: Dict[str, List[dict]] = {}
        for item in records:
            key = str(item["image_id"])
            by_image.setdefault(key, []).append(item)

        with open(self.phrase_file, "w", encoding="utf-8") as phrase_out:
            for image_id, items in tqdm(by_image.items(), desc="Formatting RefCOCOg"):
                first = items[0]
                src_img = self._resolve_image_path(
                    first.get("file_name", ""), first.get("image_id", image_id)
                )

                # Use original file name when available to preserve COCO naming.
                dst_name = src_img.name
                dst_img = self.images_dir / dst_name

                if self.cfg.copy_images:
                    if not dst_img.exists():
                        shutil.copy2(src_img, dst_img)
                else:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    os.symlink(src_img.resolve(), dst_img)

                with Image.open(src_img) as im:
                    image_w, image_h = im.size

                label_lines: List[str] = []
                phrase_payload = {
                    "image_id": first.get("image_id", image_id),
                    "file_name": dst_name,
                    "objects": [],
                }

                for item in items:
                    box_xyxy = item.get("bbox_xyxy")
                    if box_xyxy is None and "bbox_xywh" in item:
                        x, y, w, h = item["bbox_xywh"]
                        box_xyxy = [x, y, x + w, y + h]
                    if box_xyxy is None:
                        continue

                    cx, cy, bw, bh = self._xyxy_to_yolo(box_xyxy, image_w, image_h)
                    # One pseudo class for all referring objects.
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                    phrase_payload["objects"].append(
                        {
                            "phrase": item.get("phrase", ""),
                            "bbox_xyxy": box_xyxy,
                            "ann_id": item.get("ann_id"),
                            "ref_id": item.get("ref_id"),
                        }
                    )

                label_path = self.labels_dir / f"{Path(dst_name).stem}.txt"
                label_path.write_text("\n".join(label_lines), encoding="utf-8")
                phrase_out.write(json.dumps(phrase_payload, ensure_ascii=False) + "\n")

        self.data_yaml.write_text(
            "\n".join(
                [
                    f"path: {self.work_dir.resolve()}",
                    f"train: images/{self.cfg.split_name}",
                    f"val: images/{self.cfg.split_name}",
                    "names:",
                    "  0: referring_object",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        return str(self.data_yaml)


class YOLOWorldGroundingTrainer:
    """
    Trainer wrapper for the open-vocabulary grounding adaptation stage.

    It intentionally does not optimize a global image-text InfoNCE objective.
    The training signal is RefCOCOg bounding-box supervision. Referring phrases
    are preserved and used to set YOLO-World classes/prompts when supported.
    """

    def __init__(self, base_model_path: str, cfg: GroundingTrainConfig):
        self.base_model_path = base_model_path
        self.cfg = cfg
        self.formatter = RefCOCOgYOLOFormatter(cfg)
        self.model = YOLOWorld(base_model_path)

    @staticmethod
    def collect_prompt_classes(processed_json: str, max_prompts: int = 2000) -> List[str]:
        """
        Collect frequent unique referring phrases as open-vocabulary prompts.

        For very large RefCOCOg training sets, using every expression as a class
        is inefficient. This function keeps unique phrases up to max_prompts.
        """
        with open(processed_json, "r", encoding="utf-8") as f:
            records = json.load(f)

        prompts: List[str] = []
        seen = set()
        for item in records:
            phrase = str(item.get("phrase", "")).strip().lower()
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            prompts.append(phrase)
            if len(prompts) >= max_prompts:
                break

        # Fallback pseudo prompt for detection loss.
        return prompts or ["object"]

    def train(self) -> None:
        data_yaml = self.formatter.build()

        prompts = self.collect_prompt_classes(self.cfg.processed_json)
        try:
            self.model.set_classes(prompts)
        except Exception as exc:  # noqa: BLE001
            print(
                "⚠️ 当前 Ultralytics/YOLOWorld 版本不支持 set_classes 或调用失败。"
                "将继续使用 data.yaml 中的 pseudo class 进行 box-supervised adaptation。"
            )
            print(f"原因: {exc}")

        self.model.train(
            data=data_yaml,
            epochs=self.cfg.epochs,
            imgsz=self.cfg.imgsz,
            batch=self.cfg.batch,
            lr0=self.cfg.lr0,
            device=self.cfg.device,
            workers=self.cfg.workers,
            project=self.cfg.project,
            name=self.cfg.name,
        )

def train_refcocog_grounding(
    base_model_path: str,
    processed_json: str,
    image_root: str,
    **kwargs,
) -> None:
    """
    Convenience entry point used by main.py.

    Supports save_dir from main.py by mapping it to Ultralytics project/name.
    """
    save_dir = kwargs.pop("save_dir", None)

    if save_dir is not None:
        save_dir_path = Path(save_dir)
        kwargs.setdefault("project", str(save_dir_path.parent))
        kwargs.setdefault("name", save_dir_path.name)

    cfg = GroundingTrainConfig(
        processed_json=processed_json,
        image_root=image_root,
        **kwargs,
    )
    trainer = YOLOWorldGroundingTrainer(base_model_path=base_model_path, cfg=cfg)
    trainer.train()


# Deprecated compatibility stub. Kept only to avoid silent misuse.
def train_offline_alignment(*args, **kwargs):
    raise RuntimeError(
        "train_offline_alignment has been removed because global image-text "
        "InfoNCE is not the grounding adaptation described in the paper. "
        "Use train_refcocog_grounding(...) with phrase-box RefCOCOg samples instead."
    )
