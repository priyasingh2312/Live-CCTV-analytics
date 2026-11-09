"""Core detector engine shared by every use-case script.

BaseDetector  : loads the model(s) for one use case, runs detection on a frame,
                decides if an event happened, draws boxes, publishes dashboard data.
EventSink     : saves a snapshot, stores the event in outputs/events.db (SQLite) and sends alerts,
                with a cooldown so the same event does not spam you.
"""
import json
import math
import os
import re
import time
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from .alerts import AlertManager
from .models import load_model
from .people import PersonTracker, Tracks
from .store import EventStore

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # config.local.yaml (your own settings) is read on top of config.yaml, so an editor overwriting config.yaml
    # cannot lose them. Dicts are merged key by key; lists and single values are replaced.
    local = ROOT / "config.local.yaml"
    if path is None and local.exists():
        with open(local, "r", encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def _merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base[k], v) if k in base else v
        return out
    return over


def _atomic_write(path, data):
    """Write a file so readers never see half of it. On Windows the dashboard may be reading the file at that
    very moment (os.replace then fails with 'Access is denied'): retry briefly, and if it still fails just skip
    this write - the next one comes a moment later. A busy file must never stop the detector."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        for i in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.02 * (i + 1))
    except OSError:
        pass
    try:
        tmp.unlink()
    except OSError:
        pass


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def _one_box_per_object(dets):
    """Keep the highest-confidence box when two boxes cover (mostly) the same object, whatever their class."""
    kept = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        ax1, ay1, ax2, ay2 = d["box"]
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        dup = False
        for k in kept:
            bx1, by1, bx2, by2 = k["box"]
            iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            if inter / (area_a + area_b - inter) >= 0.45 or inter / min(area_a, area_b) >= 0.7:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def _iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = max(1.0, (a[2] - a[0]) * (a[3] - a[1])) + max(1.0, (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / ua


class EventSink:
    def __init__(self, cfg):
        self.out = ROOT / cfg["runtime"].get("outputs_dir", "outputs")
        for sub in ("snapshots", "stats", "frames"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.store = EventStore(self.out / "events.db")
        self.camera = os.environ.get("CAMERA_NAME") or (cfg.get("camera") or {}).get("name", "camera-1")
        alerts_cfg = cfg.get("alerts") or {}
        self.cooldown = alerts_cfg.get("cooldown_seconds", 120)
        self.alerts = AlertManager(alerts_cfg)
        self._last = {}
        self.people = None
        if (cfg.get("runtime") or {}).get("per_person_alerts"):
            self.people = PersonTracker(cfg["runtime"])
            if not self.people.ready:
                self.people = None      # no person model: keep the old cooldown rule

    def emit(self, detector, event, count, frame, force=False):
        """force=True: the caller already decided this is a NEW case (a new person), so no cooldown applies."""
        key = (detector, event)
        now = time.time()
        if not force and now - self._last.get(key, 0) < self.cooldown:
            return False
        self._last[key] = now
        ts = datetime.now()
        snap = self.out / "snapshots" / f"{ts:%Y%m%d_%H%M%S}_{detector}_{_safe(event)}.jpg"
        cv2.imwrite(str(snap), frame)
        try:
            self.store.add(ts.isoformat(timespec="seconds"), self.camera, detector, event, count, snap.name)
        except Exception as e:      # never lose the alert because the database is busy
            print("[store] could not save event:", e)
        msg = f"[{detector}] {event} (count={count}) at {ts:%Y-%m-%d %H:%M:%S}"
        print("[EVENT]", msg)
        self.alerts.send(msg, snap)
        return True


class BaseDetector:
    name = "base"  # must match the block name in config.yaml -> detectors

    def __init__(self, cfg, sink):
        self.cfg = cfg["detectors"][self.name]
        rt = cfg["runtime"]
        self.imgsz = rt.get("imgsz", 640)
        self.default_conf = rt.get("conf", 0.35)
        self.device = rt.get("device", "cpu")
        self.interval = rt.get("dashboard_interval", 1.0)
        self.sink = sink
        self.models = []
        self._load_models()

        self.box_smooth = min(1.0, max(0.05, float(rt.get("box_smooth", 0.5))))   # 1 = off; lower = steadier boxes
        self.box_hold = int(rt.get("box_hold", 1))                  # AI runs a box is kept after it is missed (stops blinking)
        self.box_hold_sec = float(rt.get("box_hold_sec", 3.0))      # ... but never longer than this many seconds
        self.box_glide = float(rt.get("box_glide", 0.12))           # seconds a box takes to glide to its new place (0 = jump)
        self._btracks, self._bnext, self._disp = [], 1, {}
        self.last_dets = []
        self.last_counts = {}
        self.last_active = {}
        self.infer_ms = 0.0
        self._streak = {}
        self._gap = {}
        self._alerted = set()      # (person, event) pairs already alerted while that person stays in view
        self._orph = {}            # per event: tracks for violations that are not inside any person box
        self.person_lost = float(rt.get("person_lost_sec", 8.0))
        self.streak_gap = int(rt.get("streak_gap", 2))   # AI runs an event may be missed before its count restarts
        self._runs, self._ms_total = 0, 0.0
        self._last_publish = 0.0

    # ---------------------------------------------------------- setup
    def _load_models(self):
        for m in self.cfg.get("models", []):
            model = load_model(m["weights"], coco=bool(m.get("coco")))
            if model is None:
                print(f"[{self.name}] WARNING: weights not found: {m['weights']} "
                      f"(see README -> custom models). Skipping this model.")
                continue
            self.models.append({
                "model": model,
                "classes": set(m.get("classes") or []),
                "rename": m.get("rename") or {},
                "conf": m.get("conf", self.default_conf),
                # optional: a different minimum confidence for some classes, e.g. {cap: 0.75, no_cap: 0.4}.
                # `conf` above must then be the lowest of them (it is what the model itself uses).
                "class_conf": {str(k): float(v) for k, v in (m.get("class_conf") or {}).items()},
                # optional: ignore boxes taller/wider than this share of the frame (0-1)
                "max_h": m.get("max_box_height"),
                "max_w": m.get("max_box_width"),
                # optional: ignore boxes narrower than this (width / height). 0.75 drops tall thin boxes such as chair legs
                "min_ar": m.get("min_box_aspect"),
                # optional: when boxes of different classes sit on the same object, keep only the best one
                "one_box": bool(m.get("one_box_per_object")),
                # optional: image size for this model only (default = runtime imgsz); bigger finds small people
                "imgsz": int(m.get("imgsz") or self.imgsz),
                # optional: overlap limit for merging boxes of the same class (model default 0.7; lower merges more, e.g. 0.4)
                "iou": m.get("iou"),
            })
        if not self.models:
            raise RuntimeError("no model weights available")

    # ---------------------------------------------------------- inference
    def detect(self, frame):
        dets = []
        fh, fw = frame.shape[:2]
        for m in self.models:
            start = len(dets)
            extra = {"iou": float(m["iou"])} if m.get("iou") else {}
            res = m["model"].predict(frame, imgsz=m["imgsz"], conf=m["conf"],
                                     device=self.device, verbose=False, **extra)[0]
            names = res.names
            for b in res.boxes:
                raw = names[int(b.cls[0])]
                label = m["rename"].get(raw, raw)
                if m["classes"] and raw not in m["classes"] and label not in m["classes"]:
                    continue
                cc = m["class_conf"].get(label, m["class_conf"].get(raw))
                if cc is not None and float(b.conf[0]) < cc:
                    continue
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                if m["max_h"] and (y2 - y1) > m["max_h"] * fh:
                    continue
                if m["max_w"] and (x2 - x1) > m["max_w"] * fw:
                    continue
                if m["min_ar"] and (x2 - x1) < m["min_ar"] * (y2 - y1):
                    continue
                dets.append({"label": label, "conf": float(b.conf[0]), "box": (x1, y1, x2, y2)})
            if m["one_box"]:
                dets[start:] = _one_box_per_object(dets[start:])
        return dets

    def evaluate(self, counts, dets):
        """Return {event_name: count} for things worth logging/alerting.

        Default rules come from config.yaml:
          alert_classes -> event whenever the class is visible
          max_counts    -> event when count goes above the limit
        Override this method in a detector file for custom logic.
        """
        events = {}
        for label in self.cfg.get("alert_classes") or []:
            if counts.get(label, 0) > 0:
                events[label] = counts[label]
        for label, limit in (self.cfg.get("max_counts") or {}).items():
            if counts.get(label, 0) > limit:
                events[f"{label} above {limit}"] = counts[label]
        # min_counts -> event when SOME are visible but fewer than expected
        # (e.g. a shirt with 3 of 5 buttons). Nothing visible = no event.
        for label, minimum in (self.cfg.get("min_counts") or {}).items():
            seen = counts.get(label, 0)
            if 0 < seen < minimum:
                events[f"{label} below {minimum}"] = seen
        return events

    @staticmethod
    def nice(label):
        """no_cap -> No cap,  beard -> Beard,  Unknown stays Unknown"""
        t = str(label).replace("_", " ").strip()
        return t[:1].upper() + t[1:]

    def title(self):
        """cap_detection -> Cap,  person_count -> Person count"""
        special = {"object_detection": "Objects", "button_detection": "Buttons"}
        if self.name in special:
            return special[self.name]
        return self.nice(self.name.replace("_detection", ""))

    def summary_text(self, counts):
        if not counts:
            return "Nothing detected"
        return ", ".join(f"{self.nice(k)} ({v})" for k, v in sorted(counts.items()))

    # ---------------------------------------------------------- per-frame step
    def _steady(self, dets):
        """Steady boxes: follow each box from run to run, average its position (no jitter) and keep it for a short
        while when one run misses it (no blinking). Counts, alerts and the picture all use these steadied boxes."""
        now = time.time()
        tracks, used = self._btracks, set()
        pairs = []
        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                if t["label"] == d["label"]:
                    v = _iou(d["box"], t["box"])
                    if v >= 0.2:
                        pairs.append((v, i, j))
        taken_d = set()
        a = self.box_smooth
        for v, i, j in sorted(pairs, reverse=True):
            if i in taken_d or j in used:
                continue
            taken_d.add(i)
            used.add(j)
            t = tracks[j]
            t["box"] = tuple(a * n + (1 - a) * o for n, o in zip(dets[i]["box"], t["box"]))
            t.update(det=dets[i], miss=0, seen=now)
        for i, d in enumerate(dets):
            if i not in taken_d:
                tracks.append({"id": self._bnext, "label": d["label"], "box": tuple(float(v) for v in d["box"]),
                               "det": d, "miss": 0, "seen": now})
                self._bnext += 1
                used.add(len(tracks) - 1)
        keep, out = [], []
        for j, t in enumerate(tracks):
            if j not in used:
                t["miss"] += 1
            if t["miss"] <= self.box_hold and now - t["seen"] <= self.box_hold_sec:
                keep.append(t)
                r = dict(t["det"])
                r["box"] = tuple(int(round(v)) for v in t["box"])
                r["_id"] = t["id"]
                out.append(r)
        self._btracks = keep
        return out

    def step(self, frame):
        t0 = time.time()
        dets = self._steady(self.detect(frame))
        counts = dict(Counter(d["label"] for d in dets))
        active = self.evaluate(counts, dets)
        self.last_dets, self.last_counts, self.last_active = dets, counts, active
        self.infer_ms = (time.time() - t0) * 1000

        need = self.cfg.get("min_consecutive", 3)
        # An event keeps its count through a few missed runs (face lost for a moment, confidence just
        # under the limit). Before, ONE miss restarted the count, so alerts could take minutes.
        for k in list(self._streak):
            if k not in active:
                self._gap[k] = self._gap.get(k, 0) + 1
                if self._gap[k] > self.streak_gap:
                    del self._streak[k]
                    self._gap.pop(k, None)
        self._runs += 1
        self._ms_total += self.infer_ms
        if self._runs == 10:
            print(f"[{self.name}] speed: about {self._ms_total / 10:.0f} ms per AI run")
        annotated = None
        people = getattr(self.sink, "people", None)
        alert_set = set(self.cfg.get("alert_classes") or [])
        if people is not None:
            people.refresh(frame)          # keeps every person's id alive even while nobody breaks a rule
        for event, cnt in active.items():
            self._gap[event] = 0
            self._streak[event] = self._streak.get(event, 0) + 1
            if self._streak[event] >= need:
                if people is not None and event in alert_set:
                    fresh = self._new_people(event, dets, people)
                    if not fresh:
                        continue            # this person was already alerted and is still in view
                    if annotated is None:
                        annotated = self.draw(frame.copy())
                    self.sink.emit(self.name, event, fresh, annotated, force=True)
                    continue
                if annotated is None:
                    annotated = self.draw(frame.copy())
                self.sink.emit(self.name, event, cnt, annotated)
        self._publish(frame)

    def _new_people(self, event, dets, people):
        """How many people showing `event` have NOT been alerted yet. A person stays 'alerted' until they leave the picture."""
        rows = [d for d in dets if d["label"] == event]
        keys, loose = [], []
        for d in rows:
            pid = people.owner(d["box"])
            if pid is not None:
                keys.append(("p", pid, event))
            else:
                loose.append(d["box"])      # violation not inside a person box (person missed): follow the box itself
        orph = self._orph.setdefault(event, Tracks(self.person_lost, reach=2.0))
        for oid in orph.update(loose):
            keys.append(("o", oid, event))
        live_p, live_o = people.alive(), orph.alive()
        self._alerted = {k for k in self._alerted
                         if (k[0] == "p" and k[1] in live_p) or (k[0] == "o" and (k[2] != event or k[1] in live_o))}
        fresh = set(keys) - self._alerted
        self._alerted |= fresh
        for k in sorted(fresh):
            print(f"[people] {self.name} {event}: new {'person' if k[0] == 'p' else 'unmatched box'} #{k[1]} "
                  f"({len(live_p)} people in view)")
        return len(fresh)

    # ---------------------------------------------------------- drawing
    @staticmethod
    def _color(label):
        h = zlib.crc32(label.encode()) & 0xFFFFFF
        return (h & 255, (h >> 8) & 255, (h >> 16) & 255)

    def _glide(self, d):
        """The box position to draw now: it slides smoothly to the newest detected position instead of jumping."""
        tgt, key = d["box"], d.get("_id")
        if key is None or self.box_glide <= 0:
            return tgt
        now = time.time()
        cur = self._disp.get(key)
        if cur is None:
            cur = self._disp[key] = [[float(v) for v in tgt], now]
        else:
            k = 1.0 - math.exp(-max(0.0, now - cur[1]) / self.box_glide)
            cur[0] = [c + (t - c) * k for c, t in zip(cur[0], tgt)]
            cur[1] = now
        return tuple(int(round(v)) for v in cur[0])

    def draw(self, frame, row=0):
        live = self.last_dets
        if self.box_glide > 0:
            ids = {d.get("_id") for d in live}
            for k in [k for k in self._disp if k not in ids]:
                self._disp.pop(k, None)
        for d in live:
            x1, y1, x2, y2 = self._glide(d)
            c = (0, 0, 255) if d["label"] in (self.cfg.get("alert_classes") or []) else self._color(d["label"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f'{self.nice(d["label"])} {int(round(d["conf"] * 100))}%', (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        if self.name == "person_count":
            n = self.last_counts.get("person", 0)
            summary = f"{n} {'person' if n == 1 else 'people'}"
        else:
            summary = self.summary_text(self.last_counts)
        text = f"{self.title()}: {summary}"
        # right side of the picture (the camera's own date/time sits on the left): big white text on a dark band,
        # so it stays readable on bright ceilings and walls. Size grows with the picture width.
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = max(0.7, w / 1280 * 0.95)
        thick = 2 if scale >= 0.9 else 1
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        pad = 6
        band_h = th + base + 2 * pad
        x = max(pad, w - tw - 2 * pad - 8)
        y0 = 8 + row * (band_h + 4)
        ov = frame.copy()
        cv2.rectangle(ov, (x - pad, y0), (min(w - 1, x + tw + pad), y0 + band_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, text, (x, y0 + pad + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return frame

    # ---------------------------------------------------------- dashboard feed
    def _publish(self, frame):
        now = time.time()
        if now - self._last_publish < self.interval:
            return
        self._last_publish = now
        out = self.sink.out
        stats = {
            "detector": self.name,
            "timestamp": now,
            "counts": self.last_counts,
            "events": self.last_active,
            "inference_ms": round(self.infer_ms, 1),
        }
        _atomic_write(out / "stats" / f"{self.name}.json", json.dumps(stats).encode())
        img = self.draw(frame.copy())
        h, w = img.shape[:2]
        if w > 960:
            img = cv2.resize(img, (960, int(h * 960 / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            _atomic_write(out / "frames" / f"{self.name}.jpg", buf.tobytes())
