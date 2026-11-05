r"""Shoes detection

Detects shoes. Needs custom weights models/shoes.pt with classes [shoes, no_shoes].
An alert is raised when 'no_shoes' is seen.

Run on its own:
    python detectors\shoes_detection.py --source "rtsp://user:pass@ip:554/stream"
    python detectors\shoes_detection.py --source 0            (webcam test)
Settings for this detector are in config.yaml -> detectors -> shoes_detection.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.base import BaseDetector  # noqa: E402
from common.runner import main  # noqa: E402


class ShoesDetector(BaseDetector):
    name = "shoes_detection"


DETECTOR = ShoesDetector

if __name__ == "__main__":
    main([DETECTOR], "Shoes detection")
