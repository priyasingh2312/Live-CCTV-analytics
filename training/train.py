"""Train a custom model for one use case (cap, gloves, shoes, mask, beard, button, pen/diary).

Dataset layout (YOLO format) for a detector, e.g. cap_detection.
A Roboflow "YOLOv11" zip can be unzipped into datasets\\<detector>\\ as it is (train\\images, valid\\images).
Or use this plain layout:
    datasets\\cap_detection\\images\\train\\*.jpg
    datasets\\cap_detection\\images\\val\\*.jpg
    datasets\\cap_detection\\labels\\train\\*.txt     (one .txt per image: class x_center y_center w h, normalised)
    datasets\\cap_detection\\labels\\val\\*.txt
Class ids follow the order of `classes:` in config.yaml (cap=0, no_cap=1).

Usage:
    python training\\train.py --detector cap_detection --epochs 50
    python training\\train.py --detector object_detection --epochs 60   (trains the custom pen/diary model)

CPU training is slow. For more than a few hundred images, train on Google Colab
(free GPU) with the same command, then copy the resulting weights file into models\\.
The best weights are copied automatically to the path given in config.yaml.
"""
import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from common.base import ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, help="block name in config.yaml, e.g. cap_detection")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", default="yolo11n.pt", help="pretrained starting weights (downloaded automatically)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dataset", help="folder name under datasets\\ (default: the detector name)")
    ap.add_argument("--keep-dataset-names", action="store_true",
                    help="use the class names/order from the dataset's own data.yaml (for datasets with several "
                         "classes shared by more than one detector, e.g. gloves + shoes)")
    ap.add_argument("--out", help="where to save the weights (default: the path in config.yaml)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="use only this share of the training images (e.g. 0.4) to cut CPU training time")
    ap.add_argument("--export-openvino", action="store_true",
                    help="also export an OpenVINO copy (faster on Intel CPUs)")
    args = ap.parse_args()

    cfg = load_config()
    if args.detector not in cfg["detectors"]:
        sys.exit(f"unknown detector {args.detector}")
    custom = [m for m in cfg["detectors"][args.detector]["models"] if not m.get("coco")]
    if not custom:
        sys.exit(f"{args.detector} uses only pretrained COCO weights - nothing to train.")
    entry = custom[0]
    classes = entry["classes"]

    ds = ROOT / "datasets" / (args.dataset or args.detector)
    # Two dataset layouts are accepted:
    #   Roboflow export : train/images, valid/images   (+ train/labels, valid/labels)
    #   Plain YOLO      : images/train, images/val     (+ labels/train, labels/val)
    if (ds / "train" / "images").exists():
        val_dir = "valid" if (ds / "valid" / "images").exists() else "val"
        train_rel, val_rel = "train/images", f"{val_dir}/images"
        needed = [ds / "train" / "images", ds / "train" / "labels",
                  ds / val_dir / "images", ds / val_dir / "labels"]
    else:
        train_rel, val_rel = "images/train", "images/val"
        needed = [ds / "images/train", ds / "images/val", ds / "labels/train", ds / "labels/val"]
    for folder in needed:
        if not folder.exists():
            sys.exit(f"missing folder: {folder}\nSee the dataset layout at the top of this file / README.")

    data_yaml = ds / "data.yaml"
    if args.keep_dataset_names:
        ds_names = yaml.safe_load(data_yaml.read_text(encoding="utf-8")).get("names")
        classes = [ds_names[i] for i in sorted(ds_names)] if isinstance(ds_names, dict) else list(ds_names)
    data_yaml.write_text(yaml.safe_dump({
        "path": str(ds), "train": train_rel, "val": val_rel,
        "names": {i: n for i, n in enumerate(classes)},
    }), encoding="utf-8")
    print("classes:", dict(enumerate(classes)))

    from ultralytics import YOLO

    model = YOLO(args.base)
    results = model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                          batch=args.batch, device=args.device, fraction=args.fraction,
                          project=str(ROOT / "runs"), name=args.detector, exist_ok=True)
    best = pathlib.Path(results.save_dir) / "weights" / "best.pt"
    target = ROOT / (args.out or entry["weights"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, target)
    print(f"\nDone. Best weights copied to {target}")
    if args.export_openvino:
        YOLO(str(target)).export(format="openvino")
        print("OpenVINO export created next to the weights file.")


if __name__ == "__main__":
    main()
