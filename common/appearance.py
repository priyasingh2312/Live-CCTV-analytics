"""Cap / mask / beard detection that works WITHOUT trained weights.

How it fits the pipeline
------------------------
cap_detection, mask_detection and beard_detection are normal detectors (config.yaml, alerts,
min_consecutive, dashboard, run_all.py). They first look for trained YOLO weights
(models/cap.pt, models/mask.pt, models/beard.pt). When a weights file is missing they switch
to the built-in fallback in this file:

    1. find faces (OpenCV Haar cascades, run on a down-scaled picture so it stays fast)
    2. cut a face-relative region (above the face for caps, lower face for masks and beards)
    3. ask a pretrained CLIP model which of two sentences describes the crop better
       ("wearing a cap" / "no cap" ...). The score is relative, not a calibrated confidence.

The fallback is based on the prototype by the team mate (cctv-appearance-detection).
Once trained weights exist, just drop them into models/ and the fallback is not used.

Needs (only for the fallback):  pip install transformers
First run downloads the CLIP weights (about 600 MB) and needs internet once.
"""
import threading
from pathlib import Path

import cv2

from .base import BaseDetector

CLIP_MODEL = "openai/clip-vit-base-patch32"

# (positive sentence, negative sentence)
PROMPTS = {
    "beard": (
        "a close-up photograph of a man's chin and cheeks with visible facial hair, a beard or stubble",
        "a close-up photograph of a clean-shaven person's chin and cheeks, with no facial hair",
    ),
    "cap": (
        "a close-up photograph of a person wearing a baseball cap or other cap on the head",
        "a close-up photograph of a person's bare head or uncovered hair, no hat or cap",
    ),
    "mask": (
        "a close-up photograph of a person wearing a protective face mask over nose and mouth",
        "a close-up photograph of a person's uncovered nose and mouth with no face mask",
    ),
    # used only by the optional face check (clip: face_check in config.yaml)
    "face": (
        "a photograph of a real person's face with eyes, nose and mouth",
        "a photograph of a wall, glass door, poster, curtain, clothing, scarf or furniture, not a face",
    ),
}

# class names shown / used in alert_classes (same as the trained-model class names)
LABELS = {"cap": ("cap", "no_cap"), "mask": ("mask", "no_mask"), "beard": ("beard", "no_beard")}


def extract_region(frame, face, kind):
    """Face-relative region to classify. Returns the cropped image or None."""
    x, y, w, h = face
    if kind == "cap":      # the cap sits ABOVE the detected face
        x1, y1, x2, y2 = x - int(0.30 * w), y - int(0.85 * h), x + int(1.30 * w), y + int(0.45 * h)
    elif kind == "mask":
        x1, y1, x2, y2 = x - int(0.10 * w), y + int(0.30 * h), x + int(1.10 * w), y + int(1.12 * h)
    elif kind == "beard":
        x1, y1, x2, y2 = x - int(0.18 * w), y + int(0.32 * h), x + int(1.18 * w), y + int(1.34 * h)
    else:
        raise ValueError(f"unsupported kind: {kind}")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _same_face(a, b):
    """True when two boxes describe the same face: they overlap a lot (IoU), or the smaller
    one lies mostly inside the larger one (a cascade sometimes reports a 'face' on the nose/mouth)."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    smaller = min(a[2] * a[3], b[2] * b[3])
    union = a[2] * a[3] + b[2] * b[3] - inter
    iou = inter / union if union else 0.0
    inside = inter / smaller if smaller else 0.0
    return iou >= 0.4 or inside >= 0.6


class FaceFinder:
    """Frontal-face finder (two Haar cascades, results merged)."""

    def __init__(self):
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError("this OpenCV version has no face cascades. Install a 4.x version: "
                               'pip install "opencv-python<5"')
        base = Path(cv2.data.haarcascades)
        self.cascades = []
        for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
            c = cv2.CascadeClassifier(str(base / name))
            if not c.empty():
                self.cascades.append(c)
        if not self.cascades:
            raise RuntimeError("could not load the OpenCV face cascades")

    def find(self, frame, min_face=40):
        h, w = frame.shape[:2]
        scale = min(1.0, 640.0 / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        size = max(24, int(min_face * scale))
        found = []
        for c in self.cascades:
            for b in c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(size, size)):
                found.append(tuple(int(v / scale) for v in b))
        merged = []
        for b in sorted(found, key=lambda r: r[2] * r[3], reverse=True):
            if not any(_same_face(b, m) for m in merged):
                merged.append(b)
        return sorted(merged, key=lambda r: r[0])


class ClipScorer:
    """One CLIP model shared by cap, mask and beard detectors."""

    _inst = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, device="cpu"):
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls(device)
            return cls._inst

    def __init__(self, device="cpu"):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("the CLIP fallback needs the 'transformers' package: pip install transformers") from exc
        self.torch = torch
        self.device = "cpu" if str(device) == "cpu" or not torch.cuda.is_available() else "cuda"
        print(f"[appearance] loading {CLIP_MODEL} on {self.device} (first run downloads about 600 MB) ...")
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self._text = {}

    def _tensor(self, out):
        if self.torch.is_tensor(out):
            return out
        for attr in ("pooler_output", "image_embeds", "text_embeds"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        return out[0]

    def _text_features(self, kind):
        if kind not in self._text:
            with self.torch.inference_mode():
                inp = self.processor(text=list(PROMPTS[kind]), return_tensors="pt", padding=True)
                inp = {k: v.to(self.device) for k, v in inp.items()}
                f = self._tensor(self.model.get_text_features(**inp))
                self._text[kind] = f / f.norm(dim=-1, keepdim=True)
        return self._text[kind]

    def positive_scores(self, kind, bgr_crops):
        """Probability (0..1) that each crop matches the 'positive' sentence (cap / mask / beard)."""
        from PIL import Image
        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in bgr_crops]
        with self.torch.inference_mode():
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            f = self._tensor(self.model.get_image_features(pixel_values=pixels))
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ self._text_features(kind).T).softmax(dim=-1)[:, 0]
        return [float(p) for p in probs.tolist()]


class AppearanceDetector(BaseDetector):
    """Base class of cap_detection, mask_detection and beard_detection."""

    kind = None  # "cap" | "mask" | "beard"

    def _load_models(self):
        self._clip = False
        try:
            super()._load_models()          # trained YOLO weights, if the file exists
            return
        except RuntimeError:
            pass
        opts = self.cfg.get("clip") or {}
        if not opts.get("enabled", True):
            raise RuntimeError("no model weights available (CLIP fallback is switched off in config.yaml)")
        self.threshold = float(opts.get("threshold", 0.65))
        # optional: 0 / missing = off. 0.6 = keep a face box only if CLIP is at least 60% sure it is a face
        self.face_check = float(opts.get("face_check", 0) or 0)
        self._scorer = ClipScorer.get(self.device)
        self._faces = FaceFinder()
        self._clip = True
        print(f"[{self.name}] trained weights not found - using the CLIP fallback "
              f"(threshold {self.threshold:.2f}). Drop models/{self.kind}.pt in to use trained weights.")

    def _real_faces(self, frame, found):
        """Drop 'faces' that CLIP says are not faces (a door, a poster, a scarf pattern, a sleeve ...)."""
        height, width = frame.shape[:2]
        crops, keep = [], []
        for (x, y, w, h) in found:
            pad_x, pad_y = int(0.15 * w), int(0.15 * h)
            c = frame[max(0, y - pad_y):min(height, y + h + pad_y), max(0, x - pad_x):min(width, x + w + pad_x)]
            if c.size:
                crops.append(c)
                keep.append((x, y, w, h))
        if not crops:
            return []
        scores = self._scorer.positive_scores("face", crops)
        return [f for f, s in zip(keep, scores) if s >= self.face_check]

    def detect(self, frame):
        if not self._clip:
            return super().detect(frame)
        crops, faces = [], []
        found = self._faces.find(frame)
        if self.face_check and found:
            found = self._real_faces(frame, found)
        for face in found:
            region = extract_region(frame, face, self.kind)
            if region is None or min(region.shape[:2]) < 35:
                continue
            crops.append(region)
            faces.append(face)
        if not crops:
            return []
        positive, negative = LABELS[self.kind]
        dets = []
        for (x, y, w, h), p in zip(faces, self._scorer.positive_scores(self.kind, crops)):
            if p >= self.threshold:
                label, conf = positive, p
            elif p <= 1.0 - self.threshold:
                label, conf = negative, 1.0 - p
            else:
                continue                    # unsure: no label, so no false alert
            dets.append({"label": label, "conf": conf, "box": (x, y, x + w, y + h)})
        return dets
