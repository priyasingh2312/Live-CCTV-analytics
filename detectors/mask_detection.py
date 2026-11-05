r"""Mask detection

Detects face masks. Classes [mask, no_mask]. An alert is raised when 'no_mask' is seen.

Uses trained weights models/mask.pt when that file exists. If it does not, the built-in
CLIP fallback (common/appearance.py) is used, so the detector still runs. Needs for the
fallback:  pip install transformers

Run on its own:
    python detectors\mask_detection.py --source "rtsp://user:pass@ip:554/stream"
    python detectors\mask_detection.py --source 0            (webcam test)
Settings for this detector are in config.yaml -> detectors -> mask_detection.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.appearance import AppearanceDetector  # noqa: E402
from common.runner import main  # noqa: E402


class MaskDetector(AppearanceDetector):
    name = "mask_detection"
    kind = "mask"


DETECTOR = MaskDetector

if __name__ == "__main__":
    main([DETECTOR], "Mask detection")
