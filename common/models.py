"""Model loading with a cache so detectors can share the same weights file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_cache = {}


def resolve_weights(weights):
    p = Path(weights)
    return p if p.is_absolute() else ROOT / p


def load_model(weights, coco=False):
    """Load a YOLO model. Returns None if a custom weights file is missing.

    coco=True weights (e.g. yolov8n.pt) are downloaded automatically into
    the models/ folder the first time (needs internet once).
    """
    path = resolve_weights(weights)
    key = str(path)
    if key in _cache:
        return _cache[key]

    from ultralytics import YOLO

    if path.exists():
        model = YOLO(str(path))
    elif coco:
        path.parent.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(path.parent)  # ultralytics downloads into the current folder
        try:
            model = YOLO(path.name)
        finally:
            os.chdir(cwd)
    else:
        return None
    _cache[key] = model
    return model
