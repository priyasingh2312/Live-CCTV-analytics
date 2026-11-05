r"""Auto-label heads as 'no_cap' in frames where nobody wears a cap.

Finds faces with YuNet (models/face_detection_yunet_2023mar.onnx), grows each face box upward to include
the hair, and writes a YOLO dataset (train / valid / test + data.yaml) with classes [cap, no_cap].
Every box is class 1 (no_cap). It also saves preview pictures so you can check the boxes.

    python tools\prelabel_no_cap.py --frames datasets\captures\cap_frames2 --name cap_cctv_nocap

Output: datasets\<name>\  and  datasets\<name>.zip  (ready for tools\merge_datasets.py)
Split: each video is cut in time order (75 % train, 15 % valid, 10 % test) so neighbouring frames
that look alike never sit in both train and valid.
"""
import argparse
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def head_box(face, w, h):
    """YuNet face box -> head box (face + hair above, a little margin at the sides)."""
    x, y, bw, bh = face
    x1, y1 = x - 0.12 * bw, y - 0.38 * bh
    x2, y2 = x + 1.12 * bw, y + 1.05 * bh
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder with the frames (all people WITHOUT a cap)")
    ap.add_argument("--name", default="cap_cctv_nocap")
    ap.add_argument("--score", type=float, default=0.65, help="face confidence, 0.5-0.9 (higher = fewer false boxes)")
    ap.add_argument("--min-face", type=int, default=14, help="ignore faces smaller than this many pixels")
    ap.add_argument("--class-id", type=int, default=1, help="1 = no_cap in the [cap, no_cap] list")
    a = ap.parse_args()

    models = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
    if not models.exists():
        sys.exit(f"missing {models}  (it is downloaded once by the face-recognition detector)")
    if not hasattr(cv2, "FaceDetectorYN"):
        sys.exit('this OpenCV has no face detector. Run: pip install "opencv-python>=4.8,<5"')
    det = cv2.FaceDetectorYN.create(str(models), "", (320, 320), a.score, 0.3, 5000)

    frames = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not frames:
        sys.exit("no pictures found in " + a.frames)
    by_video = defaultdict(list)
    for p in frames:
        by_video[re.sub(r"_\d+$", "", p.stem)].append(p)

    out = ROOT / "datasets" / a.name
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "valid", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)
    prev = out / "preview"
    prev.mkdir()

    total = defaultdict(int)
    boxes_total = 0
    for video, files in by_video.items():
        n = len(files)
        for i, p in enumerate(files):
            split = "train" if i < 0.75 * n else ("valid" if i < 0.90 * n else "test")
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
            det.setInputSize((w, h))
            _, rows = det.detect(im)
            lines, shown = [], im.copy()
            for r in (rows if rows is not None else []):
                x, y, bw, bh = (float(v) for v in r[:4])
                if min(bw, bh) < a.min_face:
                    continue
                x1, y1, x2, y2 = head_box((x, y, bw, bh), w, h)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                lines.append(f"{a.class_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            shutil.copy(p, out / split / "images" / p.name)
            (out / split / "labels" / (p.stem + ".txt")).write_text("\n".join(lines))
            total[split] += 1
            boxes_total += len(lines)
            if i % 25 == 0:
                cv2.imwrite(str(prev / f"{p.stem}.jpg"), cv2.resize(shown, (w // 2, h // 2)))
        print(f"  {video}: {n} frames")

    (out / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: ['cap', 'no_cap']\n")
    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file() and "preview" not in f.parts:
                z.write(f, f.relative_to(out))
    print(f"\nDone: {sum(total.values())} frames, {boxes_total} head boxes (class no_cap)  "
          f"train {total['train']} / valid {total['valid']} / test {total['test']}")
    print(f"Check the boxes in:  {prev}")
    print(f"Dataset zip:         {zpath}")


if __name__ == "__main__":
    main()
