"""Run several (or all) detectors on ONE camera stream.

Examples (PowerShell, from D:\\priya\\Live CCTV analytics):
    python run_all.py                                  # all 9 detectors
    python run_all.py --only person_count,mask_detection
    python run_all.py --exclude button_detection,mask_detection
    python run_all.py --source 0 --no-window           # webcam, no window

On CPU the detectors take turns (one AI run per `frame_skip` frames), so
running many at once lowers the refresh rate of each. Run only what you need.
Detectors whose custom weights are missing are skipped with a message.
"""
import argparse
import os
import sys
from datetime import datetime

from common.base import load_config
from common.runner import build_parser, run
from detectors import NAMES, get_class


class _Stamp:
    """Puts [date time|camera] in front of every printed line, so the dashboard's System tab can show times."""

    def __init__(self, out):
        self._out, self._buf = out, ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                cam = os.environ.get("CAMERA_NAME", "")
                self._out.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}|{cam}] {line}\n")
            else:
                self._out.write("\n")
        return len(text)

    def flush(self):
        self._out.flush()

    def __getattr__(self, name):
        return getattr(self._out, name)


def main():
    if "--no-window" in sys.argv:                      # started by the dashboard: its log gets time stamps
        sys.stdout = _Stamp(sys.stdout)
    parser = build_parser("Run multiple CCTV detectors on one stream")
    parser.add_argument("--only", help="comma-separated detector names to run")
    parser.add_argument("--exclude", help="comma-separated detector names to skip")
    args = parser.parse_args()

    chosen = list(NAMES)
    if args.only:
        chosen = [n.strip() for n in args.only.split(",") if n.strip()]
    if args.exclude:
        skip = {n.strip() for n in args.exclude.split(",")}
        chosen = [n for n in chosen if n not in skip]
    unknown = [n for n in chosen if n not in NAMES]
    if unknown:
        parser.error(f"unknown detector(s): {unknown}. Valid: {NAMES}")

    cfg = load_config(args.config)
    show = cfg["runtime"].get("show_window", True) and not args.no_window
    run([get_class(n) for n in chosen], cfg, source=args.source, show=show,
        max_frames=args.max_frames)


if __name__ == "__main__":
    main()
