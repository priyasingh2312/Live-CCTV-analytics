"""Face finding and face embeddings with two small OpenCV models (CPU friendly, no extra packages):
    YuNet  - finds faces and 5 landmarks
    SFace  - turns an aligned face into 128 numbers; two pictures of the same person give similar numbers
Both .onnx files (about 40 MB together) are downloaded once into models/ the first time they are needed.
Needs opencv-python 4.8 or newer (but below 5, see requirements.txt).
"""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"
URLS = {
    DETECTOR_FILE: _ZOO + "face_detection_yunet/" + DETECTOR_FILE,
    RECOGNIZER_FILE: _ZOO + "face_recognition_sface/" + RECOGNIZER_FILE,
}


def ensure_model(filename, models_dir=None):
    path = Path(models_dir or ROOT / "models") / filename
    if path.exists() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[faces] downloading {filename} (one time) ...")
    try:
        tmp = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(URLS[filename], tmp)
        if tmp.stat().st_size < 100_000:                       # a git-lfs pointer or an error page, not the model
            raise RuntimeError("downloaded file is too small")
        os.replace(tmp, path)
    except Exception as e:
        raise RuntimeError(
            f"could not download {filename} ({e}). Download it by hand from {URLS[filename]} "
            f"and save it as {path}") from e
    return path


class FaceEngine:
    def __init__(self, det_score=0.8, det_width=1280, models_dir=None):
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("this OpenCV has no face models. Run: pip install \"opencv-python>=4.8,<5\"")
        self.det_width = int(det_width)
        self.detector = cv2.FaceDetectorYN.create(str(ensure_model(DETECTOR_FILE, models_dir)), "", (320, 320),
                                                  float(det_score), 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(ensure_model(RECOGNIZER_FILE, models_dir)), "")

    def detect(self, frame):
        """[{box:(x1,y1,x2,y2), score, row}] in the coordinates of `frame`. `row` is what embed() needs."""
        h, w = frame.shape[:2]
        scale = min(1.0, self.det_width / max(h, w))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self.detector.detect(small)
        out = []
        for r in (rows if rows is not None else []):
            r = r.astype(np.float32).copy()
            r[:14] /= scale
            x, y, bw, bh = r[:4]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w - 1, int(x + bw)), min(h - 1, int(y + bh))
            if x2 > x1 and y2 > y1:
                out.append({"box": (x1, y1, x2, y2), "score": float(r[14]), "row": r})
        return out

    def embed(self, frame, face):
        """128 numbers (unit length) describing the face."""
        aligned = self.recognizer.alignCrop(frame, face["row"])
        f = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        return f / max(float(np.linalg.norm(f)), 1e-9)
