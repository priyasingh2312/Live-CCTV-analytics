r"""Object detection: laptop, pen, diary, chair, mobile phone

Uses two models: pretrained COCO for laptop / chair / mobile phone, and custom
weights models/pen_diary.pt for pen and diary. If the custom file is missing, the
detector still runs for laptop / chair / mobile phone and prints a warning.

Run on its own:
    python detectors\object_detection.py --source "rtsp://user:pass@ip:554/stream"
    python detectors\object_detection.py --source 0            (webcam test)
Settings for this detector are in config.yaml -> detectors -> object_detection.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.base import BaseDetector  # noqa: E402
from common.runner import main  # noqa: E402


class ObjectDetector(BaseDetector):
    name = "object_detection"


DETECTOR = ObjectDetector

if __name__ == "__main__":
    main([DETECTOR], "Object detection: laptop, pen, diary, chair, mobile phone")
