r"""Mask detection

Detects face masks. Classes [mask, no_mask]. An alert is raised when 'no_mask' is seen.

Uses trained weights models/mask.pt when that file exists. If it does not, the built-in
CLIP fallback (common/appearance.py) is used, so the detector still runs. Needs for the
fallback:  pip install transformers

Side faces: the trained model learned mostly front-facing faces, so a face turned away from the camera gets
wrong answers. With  side_face: enabled: true  (config.yaml -> mask_detection -> side_face) a mask / no_mask
box is dropped when the face is turned too far, or when a large box has no front-facing face under it.
This only affects mask_detection.

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
from common.turn import is_turned  # noqa: E402


def _face_under(box, faces):
    """The face whose (slightly enlarged) box holds the centre of `box`."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    best, area = None, None
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        mx, my = (x2 - x1) * 0.2, (y2 - y1) * 0.2
        if x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my:
            a = (x2 - x1) * (y2 - y1)
            if area is None or a < area:
                best, area = f, a
    return best


class MaskDetector(AppearanceDetector):
    name = "mask_detection"
    kind = "mask"

    def _load_models(self):
        self._side = None
        super()._load_models()
        opts = self.cfg.get("side_face") or {}
        if self._clip or not opts.get("enabled", False):
            return
        try:
            from common.faces import FaceEngine
            self._fe = FaceEngine(det_score=float(opts.get("face_score", 0.7)))
        except Exception as e:
            print(f"[{self.name}] side-face filter is off: {e}")
            return
        self._side = {
            "max_turn": float(opts.get("max_turn", 0.35)),            # higher = allows more turned faces
            "min_eye_ratio": float(opts.get("min_eye_ratio", 0.28)),  # lower = allows more turned faces
            "drop_unseen": bool(opts.get("drop_unseen", True)),       # drop a large box with no front-facing face
            "min_box": int(opts.get("min_box", 70)),                  # 'large' = box taller than this (pixels)
            "sure_mask": float(opts.get("sure_mask", 0.75)),          # a 'mask' this sure is always kept
        }
        self._side_warned = False
        print(f"[{self.name}] side-face filter is ON (turned faces get no mask / no_mask label)")

    def detect(self, frame):
        dets = super().detect(frame)
        if self._side is None or not dets:
            return dets
        try:
            faces = self._fe.detect(frame)
        except Exception as e:                  # never lose the normal result
            if not self._side_warned:
                print(f"[{self.name}] side-face filter failed: {e}")
                self._side_warned = True
            return dets
        o, keep = self._side, []
        for d in dets:
            if d["label"] == "mask" and d["conf"] >= o["sure_mask"]:
                keep.append(d)                  # a clearly visible mask can hide the face from the face finder
                continue
            f = _face_under(d["box"], faces)
            if f is None:
                if o["drop_unseen"] and (d["box"][3] - d["box"][1]) >= o["min_box"]:
                    continue                    # big face-sized box, but no front-facing face: probably turned away
                keep.append(d)
            elif is_turned(f["row"], o["max_turn"], o["min_eye_ratio"]):
                continue
            else:
                keep.append(d)
        return keep


DETECTOR = MaskDetector

if __name__ == "__main__":
    main([DETECTOR], "Mask detection")
