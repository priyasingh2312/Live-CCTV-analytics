r"""Draft hand boxes (gloves / no_gloves) from your own recordings, to train the gloves model on your cameras.

  python tools\prelabel_hands.py --name gloves_cctv --gloves recordings\gloves_cam1.mp4 --no-gloves recordings\bare_cam1.mp4

--gloves    clips where EVERY hand in view wears a glove
--no-gloves clips where EVERY hand in view is bare
(a clip with one gloved and one bare hand can not be labelled automatically, so do not use it here)
MediaPipe finds each hand and draws the box; the class is the one of the clip.
Output: datasets\<name>\ (train/valid/test, YOLO format) and datasets\<name>\preview\ with the boxes drawn on
every 5th frame so you can check them quickly. Delete wrong frames from images\ and labels\ if you see any.
Needs:  pip install mediapipe   (the hand model, about 8 MB, is downloaded once into models\hand_landmarker.task)
"""
import argparse, os, pathlib, shutil, sys, urllib.request
import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "hand_landmarker.task"
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
NAMES = ["gloves", "no_gloves"]


def make_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    if not MODEL.exists():
        print("downloading hand model ...")
        MODEL.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, MODEL)
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
        num_hands=4, min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4)
    det = vision.HandLandmarker.create_from_options(opts)

    def find(frame):
        h, w = frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = []
        for lm in det.detect(img).hand_landmarks:
            xs = np.array([p.x for p in lm]) * w
            ys = np.array([p.y for p in lm]) * h
            bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
            pad = 0.12 * max(bw, bh)           # landmarks sit inside the fingertips, so widen a little
            boxes.append((max(0, xs.min() - pad), max(0, ys.min() - pad),
                          min(w, xs.max() + pad), min(h, ys.max() + pad)))
        return boxes
    return find


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gloves", nargs="*", default=[], help="clips where every hand wears a glove")
    ap.add_argument("--no-gloves", nargs="*", default=[], dest="no_gloves", help="clips where every hand is bare")
    ap.add_argument("--only-pairs", action="store_true",
                    help="for glove clips: keep only frames where exactly 2 hands were found (the finder misses many gloved "
                         "hands; a frame with a missed glove would teach the model that gloves are background)")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between saved frames (default 0.5)")
    a = ap.parse_args()

    find = make_detector()
    out = ROOT / "datasets" / a.name
    if out.exists():
        shutil.rmtree(out)                                          # start clean on every run
    items = []                                                      # (image, label lines, preview)
    jobs = [(v, 0) for v in a.gloves] + [(v, 1) for v in a.no_gloves]
    if not jobs:
        sys.exit("give at least one clip with --gloves or --no-gloves")
    for v, cls in jobs:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        step = max(1, int(round(fps * a.every)))
        stem = pathlib.Path(v).stem
        i = n = hands = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = frame.shape[:2]
                lines, shown = [], frame.copy()
                boxes = find(frame)
                skip = not boxes or (cls == 0 and a.only_pairs and len(boxes) != 2)
                for box in ([] if skip else boxes):
                    c = cls
                    x1, y1, x2, y2 = box
                    lines.append(f"{c} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
                    cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255) if c else (0, 200, 0), 2)
                    cv2.putText(shown, NAMES[c], (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255) if c else (0, 200, 0), 2)
                if lines:                                           # a frame with no hand found is not saved
                    items.append((f"{stem}_{n:04d}", frame, lines, shown))
                    hands += len(lines); n += 1
            i += 1
        cap.release()
        print(f"{v}: {n} frames, {hands} hands")

    # contiguous blocks (80/10/10) so neighbouring near-identical frames do not leak across splits
    N = len(items)
    cut1, cut2 = int(N * .8), int(N * .9)
    for k, (name, frame, lines, shown) in enumerate(items):
        split = "train" if k < cut1 else ("valid" if k < cut2 else "test")
        for d in ("images", "labels"):
            (out / split / d).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / split / "images" / (name + ".jpg")), frame)
        (out / split / "labels" / (name + ".txt")).write_text("\n".join(lines))
        if k % 5 == 0:
            (out / "preview").mkdir(exist_ok=True)
            cv2.imwrite(str(out / "preview" / (name + ".jpg")), cv2.resize(shown, (640, int(640 * shown.shape[0] / shown.shape[1]))))
    (out / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
                                   f"nc: 2\nnames: {NAMES}\n")
    cnt = {0: 0, 1: 0}
    for t in out.glob("*/labels/*.txt"):
        for row in t.read_text().splitlines():
            cnt[int(row.split()[0])] += 1
    print(f"\nsaved {N} frames to {out}\nboxes: gloves {cnt[0]}, no_gloves {cnt[1]}\nCheck the pictures in {out / 'preview'}")


if __name__ == "__main__":
    main()
