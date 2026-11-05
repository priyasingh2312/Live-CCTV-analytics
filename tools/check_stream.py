"""Quick camera test: opens the stream, grabs one frame, saves it as test_frame.jpg.

    python tools\\check_stream.py
    python tools\\check_stream.py --source "rtsp://user:pass@192.168.1.64:554/stream1"
    python tools\\check_stream.py --source 0
Run this BEFORE the detectors to make sure the RTSP URL works.
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
    ap.add_argument("--source")
    ap.add_argument("--timeout", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config()
    src = args.source or os.environ.get("RTSP_URL") or cfg["camera"]["source"]
    print("Opening:", str(src).split("@")[-1])
    r = StreamReader(src, 3).start()
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        fid, frame = r.read()
        if frame is not None:
            out = ROOT / "test_frame.jpg"
            cv2.imwrite(str(out), frame)
            h, w = frame.shape[:2]
            print(f"OK - got a {w}x{h} frame. Saved {out}")
            r.stop()
            return
        time.sleep(0.2)
    r.stop()
    print("FAILED - no frame received. Check URL, username/password, IP, port 554, and that "
          "this PC can reach the camera (ping it).")
    sys.exit(1)


if __name__ == "__main__":
    main()
