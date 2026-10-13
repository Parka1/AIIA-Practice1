"""Export YOLOv8 Nano weights to a CPU-friendly ONNX model for the Azure VM."""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("local_vm/models/yolov8n_960.onnx"),
    )
    parser.add_argument("--image-size", type=int, default=960)
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("--image-size must be greater than zero")

    from ultralytics import YOLO

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.image_size,
            dynamic=False,
            simplify=False,
            opset=12,
            device="cpu",
            project=str(args.out.parent),
            name=args.out.stem,
            exist_ok=True,
        )
    )
    if exported.resolve() != args.out.resolve():
        shutil.copy2(exported, args.out)
    print(f"Exported YOLOv8 model to {args.out}")


if __name__ == "__main__":
    main()
