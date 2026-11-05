r"""Button detection with garment profiles (shirt, blazer, t-shirt)

With `torso_gate: true` (config.yaml) the button model only looks at the chest/torso of each PERSON: the person is
found first (yolov8n), the torso is cut out and enlarged, and buttons are searched inside it. Bracelets, watches,
wall clocks, badges and ceiling lights outside the torso can no longer be mistaken for buttons, and small buttons
are found more easily because the torso crop is bigger than the same area in the full picture.

Counts buttons and raises "button below N" when SOME buttons are visible but fewer than the
garment normally has. N comes from config.yaml -> button_detection -> garments, chosen with
--garment (or `garment:` in the config). Needs custom weights models/button.pt.
Buttons are small: hold the garment flat, front facing, close to the camera.

Run on its own:
    python detectors\button_detection.py --garment shirt  --source 0
    python detectors\button_detection.py --garment blazer --source "http://10.210.87.87:8080/video"
    python detectors\button_detection.py --garment tshirt
Settings for this detector are in config.yaml -> detectors -> button_detection.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.base import BaseDetector  # noqa: E402
from common.models import load_model  # noqa: E402
from common.runner import main  # noqa: E402


def _take_garment_arg():
    """Remove --garment <name> from sys.argv (the shared runner does not know it) and return the name."""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--garment" and i + 1 < len(argv):
            name = argv[i + 1]
            del argv[i:i + 2]
            return name
        if a.startswith("--garment="):
            name = a.split("=", 1)[1]
            del argv[i]
            return name
    return None


class ButtonDetector(BaseDetector):
    name = "button_detection"

    def __init__(self, cfg, sink):
        super().__init__(cfg, sink)
        self.cfg = dict(self.cfg)  # private copy so the shared config is not changed
        profiles = self.cfg.get("garments") or {}
        self.garment = os.environ.get("BUTTON_GARMENT") or self.cfg.get("garment")
        if profiles:
            if self.garment not in profiles:
                raise RuntimeError(f"unknown garment '{self.garment}'. Choose one of: {', '.join(profiles)}")
            self.cfg["min_counts"] = {"button": int(profiles[self.garment])}
            print(f"[{self.name}] garment: {self.garment}  (alert below {profiles[self.garment]} buttons)")

    # ------------------------------------------------------------ torso gate
    def _people(self, frame):
        """Person boxes (x1, y1, x2, y2) found with the COCO model."""
        if getattr(self, "_pm", None) is None:
            self._pm = load_model(self.cfg.get("person_weights", "models/yolov8n.pt"), coco=True)
            if self._pm is None:
                raise RuntimeError("torso_gate needs the person model (models/yolov8n.pt)")
        res = self._pm.predict(frame, imgsz=int(self.cfg.get("person_imgsz", 960)),
                               conf=float(self.cfg.get("person_conf", 0.40)), classes=[0],
                               device=self.device, verbose=False)[0]
        return [tuple(int(v) for v in b.xyxy[0].tolist()) for b in res.boxes]

    def detect(self, frame):
        if not self.cfg.get("torso_gate"):
            return super().detect(frame)
        y0, y1 = (self.cfg.get("torso_y") or [0.18, 0.80])      # share of the person box height: below the head, above the legs
        x0, x1 = (self.cfg.get("torso_x") or [0.12, 0.88])      # share of the width: leaves out arms hanging at the sides
        max_frac = float(self.cfg.get("max_button_frac", 0.12))  # a button is never wider than this share of the torso crop
        min_crop = int(self.cfg.get("min_torso_px", 48))
        dets = []
        for (px1, py1, px2, py2) in self._people(frame):
            w, h = px2 - px1, py2 - py1
            cx1, cx2 = px1 + int(x0 * w), px1 + int(x1 * w)
            cy1, cy2 = py1 + int(y0 * h), py1 + int(y1 * h)
            crop = frame[max(0, cy1):cy2, max(0, cx1):cx2]
            if crop.size == 0 or min(crop.shape[:2]) < min_crop:
                continue
            ox, oy = max(0, cx1), max(0, cy1)
            for d in super().detect(crop):
                bx1, by1, bx2, by2 = d["box"]
                if (bx2 - bx1) > max_frac * crop.shape[1] or (by2 - by1) > max_frac * crop.shape[1]:
                    continue
                d["box"] = (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy)
                dets.append(d)
        return dets

    def summary_text(self, counts):
        base = super().summary_text(counts)
        return f"{self.garment}: {base}" if self.garment else base


DETECTOR = ButtonDetector

if __name__ == "__main__":
    g = _take_garment_arg()
    if g:
        os.environ["BUTTON_GARMENT"] = g
    main([DETECTOR], "Button detection")
