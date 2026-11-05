r"""Cap detection

Detects caps. Classes [cap, no_cap]. An alert is raised when 'no_cap' is seen.

Uses trained weights models/cap.pt when that file exists. If it does not, the built-in
CLIP fallback (common/appearance.py) is used, so the detector still runs. Needs for the
fallback:  pip install transformers

Run on its own:
    python detectors\cap_detection.py --source "rtsp://user:pass@ip:554/stream"
    python detectors\cap_detection.py --source 0            (webcam test)
Settings for this detector are in config.yaml -> detectors -> cap_detection.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.appearance import AppearanceDetector  # noqa: E402
from common.runner import main  # noqa: E402


class CapDetector(AppearanceDetector):
    name = "cap_detection"
    kind = "cap"


DETECTOR = CapDetector

if __name__ == "__main__":
    main([DETECTOR], "Cap detection")
