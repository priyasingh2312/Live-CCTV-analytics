r"""Gloves detection

Detects gloves. Needs custom weights models/gloves.pt with classes [gloves, no_gloves].
An alert is raised when 'no_gloves' is seen.

Run on its own:
    python detectors\gloves_detection.py --source "rtsp://user:pass@ip:554/stream"
    python detectors\gloves_detection.py --source 0            (webcam test)
Settings for this detector are in config.yaml -> detectors -> gloves_detection.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.base import BaseDetector  # noqa: E402
from common.runner import main  # noqa: E402


class GlovesDetector(BaseDetector):
    name = "gloves_detection"


DETECTOR = GlovesDetector

if __name__ == "__main__":
    main([DETECTOR], "Gloves detection")
