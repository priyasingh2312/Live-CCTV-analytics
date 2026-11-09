r"""Cap detection

Detects caps. Classes [cap, no_cap]. An alert is raised when 'no_cap' is seen.

Uses trained weights models/cap.pt when that file exists. If it does not, the built-in
CLIP fallback (common/appearance.py) is used, so the detector still runs. Needs for the
fallback:  pip install transformers

Close-up faces (someone sitting right at the camera): the trained model learned caps on people seen from a
distance, so a big face fills the picture and gets no box. With  near_face: enabled: true  (config.yaml ->
cap_detection -> near_face) each big face is cut out together with the space above it, shrunk to the size of a
person seen from far away and checked by the same model. This only affects cap_detection.

Run on its own:
    python detectors\cap_detection.py --source "rtsp://user:pass@ip:554/stream"
    python detectors\cap_detection.py --source 0            (webcam test)
Settings for this detector are in config.yaml -> detectors -> cap_detection.
"""
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.appearance import AppearanceDetector  # noqa: E402
from common.runner import main  # noqa: E402


def _inside(box, region):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


class CapDetector(AppearanceDetector):
    name = "cap_detection"
    kind = "cap"

    def _load_models(self):
        self._nf = None
        super()._load_models()
        opts = self.cfg.get("near_face") or {}
        if self._clip or not opts.get("enabled", False):
            return
        try:
            from common.faces import FaceEngine
            self._nf_engine = FaceEngine(det_score=float(opts.get("face_score", 0.8)))
        except Exception as e:
            print(f"[{self.name}] close-up face check is off: {e}")
            return
        self._nf = {
            "min_face": int(opts.get("min_face", 90)),     # faces smaller than this are already handled by the normal check
            # face heights after shrinking (a far-away person's face); every size is tried, the best answer wins
            "sizes": [float(v) for v in (opts.get("face_px_list") or [36, 56, 80])],
            # lower limits than the normal check, used ONLY for close-up faces
            "limits": {str(k): float(v) for k, v in (opts.get("class_conf") or {"no_cap": 0.15, "cap": 0.50}).items()},
            "debug": bool(opts.get("debug", True)),
            "canvas": int(opts.get("canvas", 320)),        # size of the picture the model looks at for each face
        }
        self._nf_warned = False
        self._nf_last_log = 0.0
        print(f"[{self.name}] close-up face check is ON (faces taller than {self._nf['min_face']} px)")

    def detect(self, frame):
        dets = super().detect(frame)
        if self._nf is None:
            return dets
        try:
            return dets + self._near_faces(frame, dets)
        except Exception as e:                  # never lose the normal result
            if not self._nf_warned:
                print(f"[{self.name}] close-up face check failed: {e}")
                self._nf_warned = True
            return dets

    def _near_faces(self, frame, dets):
        o = self._nf
        fh, fw = frame.shape[:2]
        m = self.models[0]
        extra = []
        for f in self._nf_engine.detect(frame):
            x1, y1, x2, y2 = f["box"]
            w, h = x2 - x1, y2 - y1
            if h < o["min_face"]:
                continue
            # head = face + the space above it (where a cap sits) + a little at the sides
            hx1, hx2 = max(0, x1 - int(0.35 * w)), min(fw, x2 + int(0.35 * w))
            hy1, hy2 = max(0, y1 - int(0.90 * h)), min(fh, y2 + int(0.15 * h))
            head = (hx1, hy1, hx2, hy2)
            if any(_inside(d["box"], head) for d in dets):
                continue                        # the normal check already sees a cap / no cap here
            crop = frame[hy1:hy2, hx1:hx2]
            if crop.size == 0:
                continue
            best, seen = None, {}
            for face_px in o["sizes"]:
                s = face_px / float(h)
                cw, ch = max(8, int(crop.shape[1] * s)), max(8, int(crop.shape[0] * s))
                small = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
                S = o["canvas"]
                canvas = np.full((S, S, 3), 114, np.uint8)
                ox, oy = (S - cw) // 2, (S - ch) // 2
                canvas[oy:oy + ch, ox:ox + cw] = small
                res = m["model"].predict(canvas, imgsz=S, conf=min(0.05, m["conf"]), device=self.device,
                                         verbose=False)[0]
                for b in res.boxes:
                    raw = res.names[int(b.cls[0])]
                    label = m["rename"].get(raw, raw)
                    if m["classes"] and raw not in m["classes"] and label not in m["classes"]:
                        continue
                    conf = float(b.conf[0])
                    seen[label] = max(seen.get(label, 0.0), conf)
                    if conf < o["limits"].get(label, o["limits"].get(raw, m["conf"])):
                        continue
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    box = (int((bx1 - ox) / s + hx1), int((by1 - oy) / s + hy1),
                           int((bx2 - ox) / s + hx1), int((by2 - oy) / s + hy1))
                    if not _inside(box, head):
                        continue
                    if best is None or conf > best["conf"]:
                        best = {"label": label, "conf": conf, "box": box}
            if o["debug"] and time.time() - self._nf_last_log > 4:
                self._nf_last_log = time.time()
                found = f"{best['label']} {best['conf']:.2f}" if best else "no box"
                print(f"[{self.name}] close-up face {h}px: best scores "
                      + (", ".join(f"{k} {v:.2f}" for k, v in seen.items()) or "none") + f" -> {found}")
            if best:
                extra.append(best)
        return extra


DETECTOR = CapDetector

if __name__ == "__main__":
    main([DETECTOR], "Cap detection")
