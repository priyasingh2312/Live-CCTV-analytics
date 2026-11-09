"""Who is in the picture? One alert per person.

PersonTracker finds people (small COCO person model, a few times a second) and gives every person an id that stays
the same while the person is in view. A person who leaves for more than `person_lost_sec` and comes back is new.
BaseDetector uses it so a 'no_gloves' / 'no_cap' / ... alert is sent ONCE per person, not once per cooldown.
"""
import time


def _iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = max(1.0, (a[2] - a[0]) * (a[3] - a[1])) + max(1.0, (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / ua


def _centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


class Tracks:
    """Keeps an id for each box from one update to the next. A track is dropped after `lost_sec` without a match."""

    def __init__(self, lost_sec=8.0, reach=0.6):
        self.lost_sec, self.reach = float(lost_sec), float(reach)
        self.tracks, self._next = [], 1

    def update(self, boxes, now=None):
        now = time.time() if now is None else now
        self.tracks = [t for t in self.tracks if now - t["seen"] <= self.lost_sec]
        pairs = []
        for i, b in enumerate(boxes):
            for j, t in enumerate(self.tracks):
                v = _iou(b, t["box"])
                (cx, cy), (tx, ty) = _centre(b), _centre(t["box"])
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                size = max(b[2] - b[0], b[3] - b[1], t["box"][2] - t["box"][0], t["box"][3] - t["box"][1])
                if v >= 0.15 or dist <= self.reach * size:
                    pairs.append((v, -dist, i, j))
        pairs.sort(reverse=True)
        ids, used = [None] * len(boxes), set()
        for _, _, i, j in pairs:
            if ids[i] is not None or j in used:
                continue
            used.add(j)
            t = self.tracks[j]
            t["box"], t["seen"] = tuple(boxes[i]), now
            ids[i] = t["id"]
        for i, b in enumerate(boxes):
            if ids[i] is None:
                self.tracks.append({"id": self._next, "box": tuple(b), "seen": now})
                ids[i] = self._next
                self._next += 1
        return ids

    def alive(self):
        return {t["id"] for t in self.tracks}


class PersonTracker:
    def __init__(self, rt):
        self.ready = False
        self.every = float(rt.get("person_every", 0.5))        # seconds between person searches
        self.conf = float(rt.get("person_conf", 0.30))
        self.hold = float(rt.get("person_hold_sec", 4.0))      # a person missed by one search is still "there" for this long
        self.imgsz = int(rt.get("person_imgsz", 640))
        self.device = rt.get("device", "cpu")
        self.tracks = Tracks(rt.get("person_lost_sec", 8.0), reach=0.6)
        self.people = []            # [(id, box)]
        self._last = 0.0
        self.model = None
        try:
            from .models import load_model
            self.model = load_model(rt.get("person_model", "models/yolov8n.pt"), coco=True)
            self.ready = self.model is not None
        except Exception as e:
            print(f"[people] person tracking is off: {e}")
        if self.ready:
            print("[people] one alert per person is ON (a person is new after being out of view "
                  f"for {self.tracks.lost_sec:g} s)")

    def refresh(self, frame):
        now = time.time()
        if not self.ready or now - self._last < self.every:
            return
        self._last = now
        try:
            res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=[0],
                                     device=self.device, verbose=False)[0]
            boxes = [tuple(int(v) for v in b.xyxy[0].tolist()) for b in res.boxes]
        except Exception as e:
            print(f"[people] search failed: {e}")
            return
        ids = self.tracks.update(boxes, now)
        self.people = list(zip(ids, boxes))

    def owner(self, box):
        """Id of the person whose box holds the centre of `box` (smallest such person), or None.
        Uses every person seen in the last `hold` seconds, so one missed search does not make the person vanish."""
        now = time.time()
        cx, cy = _centre(box)
        best, area = None, None
        for t in self.tracks.tracks:
            if now - t["seen"] > self.hold:
                continue
            x1, y1, x2, y2 = t["box"]
            mx, my = (x2 - x1) * 0.15, (y2 - y1) * 0.15
            if x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my:
                a = (x2 - x1) * (y2 - y1)
                if area is None or a < area:
                    best, area = t["id"], a
        return best

    def alive(self):
        return self.tracks.alive()
