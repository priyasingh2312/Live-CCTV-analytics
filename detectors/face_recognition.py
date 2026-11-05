r"""Live face recognition

Finds faces on the live picture and says who they are. People are added from the dashboard
("Faces" tab: type a name, press Start capture, look at the camera). A face that matches nobody is
shown as Unknown and raises an alert with a snapshot (config.yaml -> alert_classes: [Unknown]).

Needs the two face models in models/ (downloaded automatically on first start, about 40 MB).

Run on its own:
    python detectors\face_recognition.py --source "rtsp://user:pass@ip:554/stream1"
    python detectors\face_recognition.py --source 0            (webcam test)
Settings: config.yaml -> detectors -> face_recognition.
"""
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from common.base import BaseDetector, _atomic_write  # noqa: E402
from common.face_db import UNKNOWN, FaceDB  # noqa: E402
from common.faces import FaceEngine  # noqa: E402
from common.runner import main  # noqa: E402

ENROLL_TIMEOUT = 90          # seconds a capture may take
MIN_SAMPLES = 8              # a capture that timed out is still saved when it got at least this many
STALE_REQUEST = 600          # a capture request older than this (seconds) is ignored


class FaceRecognitionDetector(BaseDetector):
    name = "face_recognition"

    # -------------------------------------------------------------- setup
    def _load_models(self):
        o = self.cfg
        self.threshold = float(o.get("threshold", 0.40))
        self.min_face = int(o.get("min_face", 48))
        self.log_known = bool(o.get("log_known", False))
        self.target = int(o.get("enroll_samples", 20))
        self.engine = FaceEngine(det_score=float(o.get("det_score", 0.8)), det_width=int(o.get("det_width", 1280)))
        self.db = FaceDB()
        self.names, self.embs, self._sig, self._sig_check = [], np.zeros((0, 128), np.float32), None, 0.0
        self._reload(force=True)
        self.enroll_dir = self.sink.out / "face_enroll"
        self.enroll_dir.mkdir(parents=True, exist_ok=True)
        self._enr, self._last_id = None, None
        self.models = [self.engine]
        print(f"[face_recognition] {len(set(self.names))} people enrolled")

    def _reload(self, force=False):
        now = time.time()
        if not force and now - self._sig_check < 2:
            return
        self._sig_check = now
        sig = self.db.signature()
        if force or sig != self._sig:
            self._sig = sig
            self.names, self.embs = self.db.load()

    # -------------------------------------------------------------- matching
    def _match(self, emb):
        if not len(self.embs):
            return UNKNOWN, 0.0
        sims = self.embs @ emb
        i = int(np.argmax(sims))
        s = float(sims[i])
        return (self.names[i] if s >= self.threshold else UNKNOWN), max(0.0, s)

    def detect(self, frame):
        self._reload()
        self._poll_request()
        faces = [f for f in self.engine.detect(frame) if f["box"][2] - f["box"][0] >= self.min_face]
        if self._enr:
            return self._enroll_step(frame, faces)
        dets = []
        for f in faces:
            name, sim = self._match(self.engine.embed(frame, f))
            dets.append({"label": name, "conf": sim, "box": f["box"]})
        return dets

    def evaluate(self, counts, dets):
        if self._enr:
            return {}                                   # the person being added must not raise an Unknown alert
        events = super().evaluate(counts, dets)         # alert_classes: [Unknown]
        if self.log_known:
            for label, n in counts.items():
                if label != UNKNOWN:
                    events[f"{label} seen"] = n
        return events

    # -------------------------------------------------------------- adding a person (asked for by the dashboard)
    def _status(self, state, message="", count=0):
        e = self._enr or {}
        body = {"id": e.get("id"), "name": e.get("name"), "state": state, "count": count,
                "target": self.target, "message": message, "updated": time.time()}
        _atomic_write(self.enroll_dir / "status.json", json.dumps(body).encode())

    def _poll_request(self):
        req = self.enroll_dir / "request.json"
        try:
            r = json.loads(req.read_text())
        except (OSError, ValueError):
            r = None
        if r is None:
            if self._enr:                               # the dashboard removed the request: cancelled
                self._status("cancelled", "Capture cancelled.", len(self._enr["embs"]))
                self._enr = None
            return
        if r.get("id") == self._last_id or (self._enr and self._enr["id"] == r.get("id")):
            return
        if time.time() - float(r.get("ts", 0)) > STALE_REQUEST:
            req.unlink(missing_ok=True)
            return
        self._enr = {"id": r.get("id"), "name": str(r.get("name", "")).strip(), "t0": time.time(),
                     "last": 0.0, "embs": [], "thumb": None, "best": 0.0}
        self.target = int(r.get("target") or self.target)
        self._status("collecting", "Look at the camera.", 0)

    def _finish(self, ok, message):
        e = self._enr
        if ok:
            thumb = cv2.imencode(".jpg", e["thumb"])[1].tobytes() if e["thumb"] is not None else None
            self.db.add(e["name"], e["embs"], thumb)
            self._reload(force=True)
        self._status("done" if ok else "failed", message, len(e["embs"]))
        (self.enroll_dir / "request.json").unlink(missing_ok=True)
        self._last_id, self._enr, self._streak = e["id"], None, {}

    def _enroll_step(self, frame, faces):
        e, now = self._enr, time.time()
        msg, dets = "No face seen. Look at the camera.", []
        if faces:
            f = max(faces, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            x1, y1, x2, y2 = f["box"]
            dets = [{"label": "Enrolling", "conf": f["score"], "box": f["box"]}]
            if f["score"] < 0.85:
                msg = "Face the camera."
            elif x2 - x1 < self.min_face * 1.3:
                msg = "Come a little closer."
            else:
                emb = self.engine.embed(frame, f)
                new = not e["embs"] or float(np.max(np.stack(e["embs"]) @ emb)) < 0.93
                if now - e["last"] >= 0.4 and new:
                    e["embs"].append(emb)
                    e["last"] = now
                    quality = f["score"] * (x2 - x1)
                    if quality > e["best"]:
                        pad = int(0.25 * (x2 - x1))
                        crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                        e["thumb"], e["best"] = cv2.resize(crop, (160, 160)), quality
                    msg = "Good. Turn your head a little, slowly."
                else:
                    msg = "Turn your head a little, slowly."
        n = len(e["embs"])
        if n >= self.target:
            self._finish(True, f"Saved {n} samples.")
        elif now - e["t0"] > ENROLL_TIMEOUT:
            if n >= MIN_SAMPLES:
                self._finish(True, f"Time is up. Saved {n} samples.")
            else:
                self._finish(False, "Time is up and too few good pictures were taken. Try again with better light.")
        else:
            self._status("collecting", msg, n)
        return dets

    # -------------------------------------------------------------- drawing
    def summary_text(self, counts):
        if self._enr:
            return f"Adding {self._enr['name']}: {len(self._enr['embs'])}/{self.target}"
        return super().summary_text(counts) if counts else "no face"

    @staticmethod
    def _color(label):
        return (0, 200, 255) if label == "Enrolling" else (0, 200, 0)    # amber while adding, green = known


DETECTOR = FaceRecognitionDetector

if __name__ == "__main__":
    main([DETECTOR], "Live face recognition")
