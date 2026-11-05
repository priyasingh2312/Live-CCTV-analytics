r"""Save sample frames from video files (e.g. CCTV footage) as images.

Example (PowerShell, from the project folder):
    python tools\video_to_frames.py --videos "D:\priya\cctv_analytics\CCTV Footage" --name cctv_negatives --every 10 --max-per-video 120

--videos        a video file or a folder of videos (mp4, avi, mkv, mov)
--name          output folder: datasets\captures\<name>\  (same place as tools\capture_frames.py)
--every         seconds between saved frames (default 10)
--max-per-video keep at most N frames per video, spread evenly (default 150)
--min-diff      skip a frame that is almost identical to the last saved one (default 1, 0 = keep all)
--resize        shrink frames so the longest side is at most this many pixels (default 1280)

Black or blank frames are skipped. The videos are only read, never changed.
Use the result as negatives for training:  merge_datasets.py ... --negatives datasets\captures\<name>
Look through the frames first and delete any that show the object you are training (e.g. a pen).
"""
import argparse
import pathlib
import re
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".wmv"}


def small_gray(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 45)).astype(np.int16)


def process(video, out_dir, every, max_frames, min_diff, resize):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  cannot open {video.name}, skipped")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(every * fps)))
    if total and max_frames:
        step = max(step, int(total / max_frames))
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", video.stem).strip("_")[:40]
    print(f"  {video.name}: {fps:.1f} fps, {total} frames, one frame every {step / fps:.1f} s")
    saved, idx, last = 0, 0, None
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                g = small_gray(frame)
                blank = g.mean() < 12 or g.std() < 6
                same = last is not None and min_diff > 0 and np.abs(g - last).mean() < min_diff
                if not blank and not same:
                    h, w = frame.shape[:2]
                    if max(h, w) > resize:
                        sc = resize / max(h, w)
                        frame = cv2.resize(frame, (int(w * sc), int(h * sc)))
                    sec = int(idx / fps)
                    cv2.imwrite(str(out_dir / f"{stem}_{sec:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    last, saved = g, saved + 1
                    if max_frames and saved >= max_frames:
                        break
        idx += 1
    cap.release()
    print(f"    saved {saved} frames")
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--max-per-video", type=int, default=150)
    ap.add_argument("--min-diff", type=float, default=1)
    ap.add_argument("--resize", type=int, default=1280)
    a = ap.parse_args()

    src = pathlib.Path(a.videos)
    vids = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not vids:
        sys.exit(f"no videos found in {src}")
    out_dir = ROOT / "datasets" / "captures" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for v in vids:
        total += process(v, out_dir, a.every, a.max_per_video, a.min_diff, a.resize)
    print(f"Done: {total} frames in {out_dir}")


if __name__ == "__main__":
    main()
