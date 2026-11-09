r"""Beard detection

Detects beards. Classes [beard, no_beard]. Alerts when 'beard' (an unshaved face) is seen: config.yaml -> alert_classes: [beard].

How it works, in order of preference:
  1. models/beard_cls.pt exists  -> YuNet finds each face, a small trained classifier (trained with
     training/colab_train_cls.ipynb) says beard / no_beard for that face. A moustache alone counts as no_beard.
  2. otherwise models/beard.pt (YOLO detector) if you have one
  3. otherwise the built-in CLIP fallback (common/appearance.py, needs: pip install transformers)

Run on its own:
    python detectors\beard_detection.py --source "rtsp://user:pass@ip:554/stream"
    python detectors\beard_detection.py --source 0            (webcam test)
Settings for this detector are in config.yaml -> detectors -> beard_detection -> classifier.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.appearance import AppearanceDetector  # noqa: E402
from common.runner import main  # noqa: E402
from common.turn import is_turned  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


class BeardDetector(AppearanceDetector):
    name = "beard_detection"
    kind = "beard"

    def _load_models(self):
        self._cls = None
        sf = self.cfg.get("side_face") or {}
        # turned faces get no beard / no_beard label (side_face: enabled: true in config.yaml -> beard_detection)
        self._side = ((float(sf.get("max_turn", 0.35)), float(sf.get("min_eye_ratio", 0.28)))
                      if sf.get("enabled", False) else None)
        opts = self.cfg.get("classifier") or {}
        path = pathlib.Path(opts.get("weights", "models/beard_cls.pt"))
        if not path.is_absolute():
            path = ROOT / path
        if opts.get("enabled", True) and path.exists():
            from ultralytics import YOLO

            from common.faces import FaceEngine
            self._cls = YOLO(str(path))
            self._cls_imgsz = int(opts.get("imgsz", 224))
            self._cls_threshold = float(opts.get("threshold", 0.70))   # how sure the classifier must be
            self._min_face = int(opts.get("min_face", 40))             # ignore faces smaller than this (pixels)
            self._face_score = float(opts.get("face_score", 0.8))
            self._smooth = min(1.0, max(0.05, float(opts.get("smoothing", 0.35))))  # 1 = no smoothing, lower = steadier
            self._tracks = []                                                        # faces followed from run to run
            self._faces_yn = FaceEngine(det_score=self._face_score)
            self._clip = False
            print(f"[{self.name}] using the trained classifier {path.name} "
                  f"(threshold {self._cls_threshold:.2f}, smoothing {self._smooth:.2f}, min face {self._min_face}px)")
            return
        super()._load_models()          # beard.pt, or the CLIP fallback

    def detect(self, frame):
        if self._cls is None:
            return super().detect(frame)
        fh, fw = frame.shape[:2]
        crops, boxes = [], []
        for f in self._faces_yn.detect(frame):
            x1, y1, x2, y2 = f["box"]
            w, h = x2 - x1, y2 - y1
            if min(w, h) < self._min_face:
                continue
            if self._side and is_turned(f["row"], *self._side):
                continue
            # face plus a margin (hair, jaw and chin) - the training pictures are framed like this
            cx1, cy1 = max(0, x1 - int(0.25 * w)), max(0, y1 - int(0.30 * h))
            cx2, cy2 = min(fw, x2 + int(0.25 * w)), min(fh, y2 + int(0.25 * h))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size:
                crops.append(crop)
                boxes.append((x1, y1, x2, y2))
        if not crops:
            self._age_tracks()
            return []
        results = self._cls.predict(crops, imgsz=self._cls_imgsz, device=self.device, verbose=False)
        names = results[0].names
        idx = next((i for i, n in names.items() if n == "beard"), 0)
        dets = []
        for box, res in zip(boxes, results):
            p = self._smoothed(box, float(res.probs.data[idx]))
            t = self._cls_threshold
            if p >= t:
                label, conf = "beard", p
            elif p <= 1.0 - t:
                label, conf = "no_beard", 1.0 - p
            else:
                continue                # unsure: no label, so no false alert
            dets.append({"label": label, "conf": conf, "box": box})
        self._age_tracks()
        return dets

    # -- smoothing: follow each face between AI runs and average its beard score, so the label stops flickering
    def _smoothed(self, box, p):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = max(box[2] - box[0], box[3] - box[1])
        best, best_d = None, 0.8 * size                  # a face moved less than 0.8 face-widths = the same face
        for t in self._tracks:
            if t["seen"]:
                continue
            d = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5
            if d < best_d:
                best, best_d = t, d
        if best is None:
            best = {"p": p}
            self._tracks.append(best)
        else:
            best["p"] = (1 - self._smooth) * best["p"] + self._smooth * p
        best.update(cx=cx, cy=cy, seen=True, miss=0)
        return best["p"]

    def _age_tracks(self):
        keep = []
        for t in self._tracks:
            if not t["seen"]:
                t["miss"] += 1
            t["seen"] = False
            if t["miss"] <= 4:                           # forget a face after 5 runs without seeing it
                keep.append(t)
        self._tracks = keep


DETECTOR = BeardDetector

if __name__ == "__main__":
    main([DETECTOR], "Beard detection")
