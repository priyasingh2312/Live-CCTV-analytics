r"""Save frames from a camera so you can build your own training set (for example for pens).

    python tools\capture_frames.py --source "http://10.210.87.87:8080/video" --name pen
    python tools\capture_frames.py --source 0 --name pen --every 0.7 --max 250

Hold the object in front of the camera and keep changing the pose between saves:
angle, distance, hand, background, light. A frame is saved every --every seconds
into datasets\captures\<name>\ . Press SPACE to pause or resume saving, q to quit.
Upload that folder to Roboflow, draw a box on the object in each image, export as
YOLOv11, then train with training\colab_train.ipynb.
"""
import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.base import ROOT, load_config  # noqa: E402
from common.stream import StreamReader  # noqa: E402

import cv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="rtsp://, http:// camera URL, or webcam index (default: config / .env)")
    ap.add_argument("--name", required=True, help="short name of the object, e.g. pen (used for the folder and file names)")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between saved frames (default 1.0)")
    ap.add_argument("--max", type=int, default=200, help="stop after saving this many frames (default 200)")
    ap.add_argument("--width", type=int, default=1280, help="shrink saved frames to this width (0 = keep original size)")
    args = ap.parse_args()

    cfg = load_config()
    src = args.source or os.environ.get("RTSP_URL") or cfg["camera"]["source"]
    out = ROOT / "datasets" / "captures" / args.name
    out.mkdir(parents=True, exist_ok=True)
    start_no = len(list(out.glob("*.jpg")))
    print(f"Source : {str(src).split('@')[-1]}")
    print(f"Saving : {out}  (already has {start_no} images)")
    print("SPACE = pause/resume saving, q = quit. Change the pose, angle, distance and background between saves.")

    reader = StreamReader(src, 3).start()
    saved, last_save, paused = 0, 0.0, False
    try:
        while saved < args.max:
            fid, frame = reader.read()
            if frame is None:
                time.sleep(0.1)
                continue
            now = time.time()
            if not paused and now - last_save >= args.every:
                last_save = now
                img = frame
                if args.width and img.shape[1] > args.width:
                    h = int(img.shape[0] * args.width / img.shape[1])
                    img = cv2.resize(img, (args.width, h))
                name = f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}_{start_no + saved:04d}.jpg"
                cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
            view = frame.copy()
            label = f"{args.name}: saved {saved}/{args.max}" + ("  [PAUSED]" if paused else "")
            cv2.putText(view, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(view, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Capture - SPACE pause, q quit", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        cv2.destroyAllWindows()
    print(f"Done. Saved {saved} new images in {out}")


if __name__ == "__main__":
    main()
