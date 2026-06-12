import os
import json
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


# ==========================================
# 1. 数据预处理类：RefCOCOg -> phrase-region grounding samples
# ==========================================
class RefCOCOgDatasetProcessor:
    """
    将 RefCOCOg 的 refs(google).p 和 instances.json 对齐为可训练的 grounding 样本。

    输出样本粒度：
        one referring expression sentence -> one region box

    每条样本包含：
        image_id, file_name, ann_id, ref_id, phrase, bbox_xywh, bbox_xyxy, category_id

    注意：
        这里不再只构建 image-phrase pair，而是构建 phrase-bbox pair。
        这更符合 YOLO-World 微调中的 text-region alignment / grounding supervision。
    """

    def __init__(
        self,
        raw_dir: str = "./data/raw",
        processed_dir: str = "./data/processed",
        split: str = "train",
        keep_all_sentences: bool = True,
        min_words: int = 1,
        max_words: int = 30,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.split = split
        self.keep_all_sentences = keep_all_sentences
        self.min_words = min_words
        self.max_words = max_words

        self.instances_file = os.path.join(raw_dir, "instances.json")
        self.refs_file = os.path.join(raw_dir, "refs(google).p")
        self.out_file = os.path.join(processed_dir, f"{split}_grounding.json")

    @staticmethod
    def _xywh_to_xyxy(bbox: List[float]) -> List[float]:
        x, y, w, h = bbox
        return [float(x), float(y), float(x + w), float(y + h)]

    @staticmethod
    def _clean_phrase(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _valid_phrase(self, phrase: str) -> bool:
        n_words = len(phrase.split())
        return self.min_words <= n_words <= self.max_words

    def process(self) -> str:
        print(" 开始解析 RefCOCOg 标注文件...")

        with open(self.instances_file, "r", encoding="utf-8") as f:
            instances = json.load(f)

        with open(self.refs_file, "rb") as f:
            refs = pickle.load(f)

        # image_id -> file_name
        image_id_to_file_name: Dict[int, str] = {
            int(img["id"]): img["file_name"] for img in instances.get("images", [])
        }

        # ann_id -> annotation, annotation 中通常包含 image_id, bbox, category_id
        ann_id_to_ann: Dict[int, Dict[str, Any]] = {
            int(ann["id"]): ann for ann in instances.get("annotations", [])
        }

        grounding_samples: List[Dict[str, Any]] = []
        skipped_missing_ann = 0
        skipped_missing_image = 0
        skipped_invalid_phrase = 0
        skipped_invalid_box = 0

        for ref in refs:
            if ref.get("split") != self.split:
                continue

            ann_id = int(ref["ann_id"])
            ref_id = int(ref.get("ref_id", -1))

            ann = ann_id_to_ann.get(ann_id)
            if ann is None:
                skipped_missing_ann += 1
                continue

            image_id = int(ref.get("image_id", ann["image_id"]))
            file_name = image_id_to_file_name.get(image_id)
            if file_name is None:
                skipped_missing_image += 1
                continue

            bbox_xywh = ann.get("bbox", None)
            if (
                bbox_xywh is None
                or len(bbox_xywh) != 4
                or float(bbox_xywh[2]) <= 0
                or float(bbox_xywh[3]) <= 0
            ):
                skipped_invalid_box += 1
                continue

            sentences = ref.get("sentences", [])
            if not sentences:
                skipped_invalid_phrase += 1
                continue

            if self.keep_all_sentences:
                selected_sentences = sentences
            else:
                # 如果只想保留一句，不取最短句，而取长度适中的句子，避免 "man" 这类过泛表达。
                valid_sentences = []
                for sent_obj in sentences:
                    phrase = self._clean_phrase(sent_obj["sent"])
                    if self._valid_phrase(phrase):
                        valid_sentences.append(sent_obj)

                selected_sentences = valid_sentences[:1] if valid_sentences else [sentences[0]]

            for sent_obj in selected_sentences:
                phrase = self._clean_phrase(sent_obj["sent"])
                if not self._valid_phrase(phrase):
                    skipped_invalid_phrase += 1
                    continue

                sample = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "ann_id": ann_id,
                    "ref_id": ref_id,
                    "sent_id": int(sent_obj.get("sent_id", -1)),
                    "phrase": phrase,
                    "bbox_xywh": [float(v) for v in bbox_xywh],
                    "bbox_xyxy": self._xywh_to_xyxy([float(v) for v in bbox_xywh]),
                    "category_id": int(ann.get("category_id", -1)),
                    "split": self.split,
                }
                grounding_samples.append(sample)

        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.out_file, "w", encoding="utf-8") as f:
            json.dump(grounding_samples, f, ensure_ascii=False, indent=2)

        print(f" 数据预处理完成！")
        print(f"   split: {self.split}")
        print(f"   grounding samples: {len(grounding_samples)}")
        print(f"   saved to: {self.out_file}")
        print(f"   skipped_missing_ann: {skipped_missing_ann}")
        print(f"   skipped_missing_image: {skipped_missing_image}")
        print(f"   skipped_invalid_phrase: {skipped_invalid_phrase}")
        print(f"   skipped_invalid_box: {skipped_invalid_box}")

        return self.out_file


# ==========================================
# 2. PyTorch Dataset：返回 image + phrase + bbox
# ==========================================
class RefCOCOGroundingDataset(Dataset):
    """
    RefCOCOg grounding dataset.

    返回格式更接近 open-vocabulary detector 微调：
        {
            "image": PIL image 或 transform 后的 tensor,
            "boxes": Tensor[num_boxes, 4],  # xyxy
            "phrases": List[str],
            "labels": Tensor[num_boxes],
            "image_id": int,
            "file_name": str,
            "ann_ids": List[int],
        }

    默认 group_by_image=False：
        每个 referring expression sentence 是一个样本。

    如果 group_by_image=True：
        将同一张图中的多个 phrase-region 样本聚合，适合检测器按 image batch 训练。
    """

    def __init__(
        self,
        json_path: str,
        img_dir: str,
        transform=None,
        group_by_image: bool = False,
        return_pil: bool = True,
    ):
        self.json_path = json_path
        self.img_dir = img_dir
        self.transform = transform
        self.group_by_image = group_by_image
        self.return_pil = return_pil

        with open(json_path, "r", encoding="utf-8") as f:
            self.annotations: List[Dict[str, Any]] = json.load(f)

        if self.group_by_image:
            grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
            for ann in self.annotations:
                grouped[(int(ann["image_id"]), ann["file_name"])].append(ann)

            self.samples: List[Dict[str, Any]] = []
            for (image_id, file_name), anns in grouped.items():
                self.samples.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "items": anns,
                    }
                )
        else:
            self.samples = self.annotations

        print(f"加载 RefCOCOg grounding dataset: {len(self.samples)} samples")
        print(f"   source json: {json_path}")
        print(f"   image dir: {img_dir}")
        print(f"   group_by_image: {group_by_image}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, file_name: str, image_id: Optional[int] = None) -> str:
        candidates = [os.path.join(self.img_dir, file_name)]

        # 兼容 common_images 中可能直接使用 12 位数字命名的情况
        if image_id is not None:
            img_id_str = str(image_id).zfill(12)
            candidates.extend(
                [
                    os.path.join(self.img_dir, f"{img_id_str}.jpg"),
                    os.path.join(self.img_dir, f"COCO_train2014_{img_id_str}.jpg"),
                    os.path.join(self.img_dir, f"COCO_val2014_{img_id_str}.jpg"),
                ]
            )

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"找不到图片：file_name={file_name}, image_id={image_id}. "
            f"已尝试：{candidates}"
        )

    def _load_image(self, file_name: str, image_id: int):
        img_path = self._resolve_image_path(file_name, image_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        if self.group_by_image:
            image_id = int(sample["image_id"])
            file_name = sample["file_name"]
            items = sample["items"]
        else:
            image_id = int(sample["image_id"])
            file_name = sample["file_name"]
            items = [sample]

        image, img_path = self._load_image(file_name, image_id)

        boxes = torch.tensor(
            [item["bbox_xyxy"] for item in items],
            dtype=torch.float32,
        )
        phrases = [item["phrase"] for item in items]

        # 这里的 labels 只作为占位：每个 phrase 对应一个 foreground object。
        # 真正的 open-vocabulary 语义来自 phrases，而不是固定 class id。
        labels = torch.arange(len(items), dtype=torch.long)

        return {
            "image": image,
            "boxes": boxes,
            "phrases": phrases,
            "labels": labels,
            "image_id": image_id,
            "file_name": file_name,
            "img_path": img_path,
            "ann_ids": [int(item["ann_id"]) for item in items],
            "ref_ids": [int(item["ref_id"]) for item in items],
        }


# 为了兼容旧代码名，保留一个别名。
# 但建议后续 trainer.py 统一改用 RefCOCOGroundingDataset。
RefCOCOContrastiveDataset = RefCOCOGroundingDataset


def grounding_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    检测/grounding 任务中，每张图的 box 数量不同，不能直接用默认 collate。
    这个 collate_fn 保留 list 结构，交给 trainer/model 自己处理。
    """
    return {
        "images": [item["image"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "phrases": [item["phrases"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "file_names": [item["file_name"] for item in batch],
        "img_paths": [item["img_path"] for item in batch],
        "ann_ids": [item["ann_ids"] for item in batch],
        "ref_ids": [item["ref_ids"] for item in batch],
    }
