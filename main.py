import argparse
import os
from pathlib import Path

from src.data.dataset import RefCOCOgDatasetProcessor
from src.training.trainer import train_refcocog_grounding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune/adapt YOLO-World with RefCOCOg phrase-box grounding supervision."
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="./data/raw",
        help="Directory containing instances.json and refs(google).p.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="./data/processed",
        help="Directory used to save processed RefCOCOg grounding samples.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/root/autodl-tmp/data/common_images",
        help="COCO image directory on the remote server.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="./weights/yolov8s-world.pt",
        help="Path to the base YOLO-World weight file.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/detector",
        help="Directory for adapted YOLO-World checkpoints and training artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--force-preprocess", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("==========================================")
    print(" RefCOCOg-based YOLO-World grounding adaptation")
    print("==========================================")

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    image_root = Path(args.image_root)
    save_dir = Path(args.save_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    instances_file = raw_dir / "instances.json"
    refs_file = raw_dir / "refs(google).p"
    train_json = processed_dir / "train_grounding.json"

    if not instances_file.exists():
        raise FileNotFoundError(f"Missing RefCOCOg instance file: {instances_file}")
    if not refs_file.exists():
        raise FileNotFoundError(f"Missing RefCOCOg refs file: {refs_file}")
    if not image_root.exists():
        raise FileNotFoundError(
            f"Image root does not exist: {image_root}. "
            "On AutoDL, set --image-root to the COCO/common_images directory."
        )

    if args.force_preprocess or not train_json.exists():
        print("⚠ Processed grounding file not found. Building RefCOCOg phrase-box samples...")
        processor = RefCOCOgDatasetProcessor(
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        )
        processor.process()
    else:
        print(f" Found processed grounding file: {train_json}")

    if not train_json.exists():
        raise FileNotFoundError(
            f"Expected processed file was not created: {train_json}. "
            "Check whether RefCOCOg preprocessing writes train_grounding.json."
        )

    print("\n Starting YOLO-World grounding adaptation...")
    print(f"   Raw annotations : {raw_dir}")
    print(f"   Processed data  : {train_json}")
    print(f"   Image root      : {image_root}")
    print(f"   Base model      : {args.base_model}")
    print(f"   Save directory  : {save_dir}")

    train_refcocog_grounding(
        base_model_path=args.base_model,
        processed_json=str(train_json),
        image_root=str(image_root),
        save_dir=str(save_dir),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )

    print(f"\n Grounding adaptation finished. Checkpoints/artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
