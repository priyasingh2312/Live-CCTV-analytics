r"""Person count

Counts people in view using the pretrained COCO model (no custom weights needed).
Raises an event when the count goes above max_counts.person in config.yaml.

Run on its own:
    python detectors\person_count.py --source "rtsp://user:pass@ip:554/stream"
    python detectors\person_count.py --source 0            (webcam test)
Settings for this detector are in config.yaml -> detectors -> person_count.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.base import BaseDetector  # noqa: E402
from common.runner import main  # noqa: E402


class PersonCountDetector(BaseDetector):
    name = "person_count"

    def summary_text(self, counts):
        return f"People: {counts.get('person', 0)}"


DETECTOR = PersonCountDetector

if __name__ == "__main__":
    main([DETECTOR], "Person count")
