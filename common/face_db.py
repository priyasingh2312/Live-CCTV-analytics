"""Face database: who is enrolled. numpy + json only (no OpenCV), so the dashboard can use it too.

One file per person in  <project>/faces/ :
    <slug>.npz   the face embeddings (one row per saved sample) and the display name
    <slug>.jpg   a small picture of the face, shown in the dashboard
Embeddings are numbers, not pictures, but they still identify a person: keep the faces/ folder private
(it is in .gitignore).
"""
import os
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACES_DIR = ROOT / "faces"
UNKNOWN = "Unknown"
MAX_SAMPLES = 60                                  # per person; the oldest are dropped
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._'-]{0,39}$")


def valid_name(name):
    """Letters, digits, space . _ ' - (so the name can be drawn on the picture), at most 40 characters."""
    name = (name or "").strip()
    return bool(NAME_RE.match(name)) and name.lower() != UNKNOWN.lower()


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "person"


def _atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class FaceDB:
    def __init__(self, root=None):
        self.dir = Path(root) if root else FACES_DIR

    # ------------------------------------------------------------ read
    def signature(self):
        """Changes whenever someone is added, changed or deleted (the detector reloads when it changes)."""
        try:
            return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(self.dir.glob("*.npz")))
        except OSError:
            return ()

    def people(self):
        """[{name, samples, created, thumb}] sorted by name."""
        out = []
        for p in sorted(self.dir.glob("*.npz")):
            try:
                with np.load(p, allow_pickle=False) as z:
                    out.append({"name": str(z["name"]), "samples": int(len(z["emb"])),
                                "created": float(z["created"]),
                                "thumb": (p.with_suffix(".jpg") if p.with_suffix(".jpg").exists() else None)})
            except Exception:
                continue
        return sorted(out, key=lambda d: d["name"].lower())

    def load(self):
        """(names, embeddings): one name per embedding row, rows are unit length."""
        names, rows = [], []
        for p in sorted(self.dir.glob("*.npz")):
            try:
                with np.load(p, allow_pickle=False) as z:
                    e, n = z["emb"].astype(np.float32), str(z["name"])
            except Exception:
                continue
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
            rows.append(e)
            names += [n] * len(e)
        return names, (np.vstack(rows) if rows else np.zeros((0, 128), np.float32))

    # ------------------------------------------------------------ write
    def add(self, name, embs, thumb_jpg=None):
        """Save samples for a person (added to the ones already saved under that name)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        base = self.dir / slug(name)
        embs = np.asarray(embs, np.float32).reshape(-1, 128)
        created = time.time()
        if base.with_suffix(".npz").exists():
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as z:
                embs = np.vstack([z["emb"], embs])[-MAX_SAMPLES:]
                name, created = str(z["name"]), float(z["created"])
        tmp = base.with_name(base.name + ".tmp.npz")
        np.savez(tmp, emb=embs, name=np.array(name), created=np.array(created))
        os.replace(tmp, base.with_suffix(".npz"))
        if thumb_jpg:
            _atomic(base.with_suffix(".jpg"), thumb_jpg)
        return name

    def delete(self, name):
        base = self.dir / slug(name)
        for ext in (".npz", ".jpg"):
            try:
                base.with_suffix(ext).unlink()
            except OSError:
                pass
