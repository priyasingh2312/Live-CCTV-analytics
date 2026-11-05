r"""Check how many people yolov8n finds in a CCTV video at different confidence / image sizes.
Read-only test: it changes no detector, no config, and does not touch person_count.py.

  python tools\test_person_conf.py --source "D:\priya\cctv_analytics\CCTV Footage\Bc-08 - People Count.mp4"

It takes --frames still frames spread over the video, runs yolov8n on each with every
(imgsz, conf) pair, prints a table of person counts, and saves annotated pictures to
outputs\person_test\ so you can count the real people yourself and compare.
"""
import argparse
import pathlib

import cv2
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolov8n.pt"))
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 960, 1280])
    ap.add_argument("--conf", nargs="+", type=float, default=[0.7, 0.5, 0.35, 0.25])
    args = ap.parse_args()

    out = ROOT / "outputs" / "person_test"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit("cannot read the video")
    picks = [int(total * (i + 1) / (args.frames + 1)) for i in range(args.frames)]

    for n, pos in enumerate(picks, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        print(f"\nframe {n}  (video frame {pos}, {w}x{h})")
        print("  imgsz | " + " | ".join(f"conf {c:<4}" for c in args.conf))
        for sz in args.imgsz:
            row = []
            for c in args.conf:
                r = model.predict(frame, imgsz=sz, conf=c, classes=[0], verbose=False)[0]
                row.append(f"{len(r.boxes):>9}")
                if c == min(args.conf):
                    cv2.imwrite(str(out / f"frame{n}_imgsz{sz}_conf{c}.jpg"), r.plot())
            print(f"  {sz:>5} | " + " | ".join(row))
    print(f"\nPictures saved in: {out}")
    print("Count the real people in one picture and compare with the table.")


if __name__ == "__main__":
    main()
