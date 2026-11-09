r"""Merge several YOLO-format datasets (Roboflow YOLOv11 zips) into one training zip.

Example (gloves):
  python tools\merge_datasets.py --name gloves_merged --names gloves no_gloves ^
      --add "D:\priya\zips\Gloves Detection.v1i.yolov11.zip" "0:0,1:1" ^
      --add "D:\priya\zips\glove.v1i.yolov11.zip" "0:0"

--names : class names of the merged dataset, in order (index 0, 1, ...)
--add   : a dataset zip (or an unzipped folder) and a class map "source_id:merged_id,...".
          A source class that is not listed in the map is dropped (its boxes are removed).
Every source keeps its own train / valid / test split. Output: datasets\<name>\ and <name>.zip
--negatives : (optional, repeatable) a folder of images that contain NO object of interest, e.g. frames of
          scarves, striped clothes and chair corners that the model wrongly detects. They are added with empty
          label files, so the model learns that these are background:
              --negatives "D:\\priya\\Live CCTV analytics\\datasets\\captures\\pen_negatives"
--max-negatives N : keep at most N negative images (spread evenly over the folder). Default: no limit.
          A good size is 25-35 percent of the images that contain the object.
Polygon labels are kept as they are (Ultralytics converts them to boxes while training).
"""
import argparse
import hashlib
import pathlib
import shutil
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def _short(name):
    """Very long file names (some web datasets have 200+ characters) break Windows' 260-character path limit.
    Shorten the name of the picture and of its label the same way, so they still belong together."""
    stem, dot, ext = name.rpartition(".")
    if len(stem) > 60:
        stem = stem[:40] + "_" + hashlib.md5(stem.encode()).hexdigest()[:10]
    return stem + dot + ext


def _extract_short(z, dest):
    dest = pathlib.Path(dest)
    for info in z.infolist():
        if info.is_dir():
            continue
        parts = info.filename.replace("\\", "/").split("/")
        parts[-1] = _short(parts[-1])
        target = dest.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as f, open(target, "wb") as o:
            shutil.copyfileobj(f, o)


def parse_map(text):
    m = {}
    for part in text.split(","):
        a, b = part.split(":")
        m[int(a)] = int(b)
    return m


def add_negatives(args, out, totals):
    """Copy background-only images with empty label files. Frames stay in time order and are split in
    contiguous blocks (80% train, 10% valid, 10% test), so near-identical neighbours do not leak across splits."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    imgs = []
    for folder in args.negatives:
        f = pathlib.Path(folder)
        if not f.is_dir():
            sys.exit(f"negatives folder not found: {f}")
        imgs += sorted(p for p in f.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not imgs:
        sys.exit("no images found in the --negatives folder(s)")
    if args.max_negatives and len(imgs) > args.max_negatives:
        step = len(imgs) / args.max_negatives
        imgs = [imgs[int(i * step)] for i in range(args.max_negatives)]
    n = len(imgs)
    cut1, cut2 = int(n * 0.8), int(n * 0.9)
    before = {k: v for k, v in totals.items()}
    for i, img in enumerate(imgs):
        target = "train" if i < cut1 else ("valid" if i < cut2 else "test")
        totals[target] = totals.get(target, 0) + 1
        key = hashlib.md5(str(img).encode()).hexdigest()[:12]
        stem = f"neg_{key}.rf.{totals[target]:05d}"
        dst = out / target / "images" / (stem + ".jpg")
        im = cv2.imread(str(img)) if cv2 is not None else None
        if im is not None:                              # shrink big phone photos, save as jpg
            h, w = im.shape[:2]
            if max(h, w) > 1280:
                sc = 1280 / max(h, w)
                im = cv2.resize(im, (int(w * sc), int(h * sc)))
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            dst = out / target / "images" / (stem + img.suffix.lower())
            shutil.copyfile(img, dst)
        (out / target / "labels" / f"{stem}.txt").write_text("")
    pos_train = before.get("train", 0)
    print(f"negatives: {n} images added (train {cut1}, valid {cut2 - cut1}, test {n - cut2}); "
          f"{n / max(1, pos_train + cut1):.0%} of the training images are negatives")
    if cut1 > 0.45 * (pos_train + cut1):
        print("WARNING: more than 45% negatives; use --max-negatives to keep the model confident on real objects.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output name, e.g. gloves_merged")
    ap.add_argument("--names", nargs="+", required=True, help="merged class names in order")
    ap.add_argument("--add", nargs=2, action="append", metavar=("SOURCE", "MAP"), required=True)
    ap.add_argument("--negatives", action="append", default=[], metavar="FOLDER",
                    help="folder of images with no object (empty labels); can be given several times")
    ap.add_argument("--max-negatives", type=int, help="keep at most this many negative images")
    args = ap.parse_args()

    out = ROOT / "datasets" / args.name
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "valid", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)

    totals = {}
    for n, (src, cmap_text) in enumerate(args.add, 1):
        cmap = parse_map(cmap_text)
        bad = [v for v in cmap.values() if not 0 <= v < len(args.names)]
        if bad:
            sys.exit(f"map {cmap_text}: class id {bad} is outside --names (0..{len(args.names) - 1})")
        src = pathlib.Path(src)
        tmp = None
        if src.suffix.lower() == ".zip":
            tmp = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(src) as z:
                _extract_short(z, tmp.name)
            base = pathlib.Path(tmp.name)
        else:
            base = src
        if not (base / "data.yaml").exists():
            sub = [p for p in base.iterdir() if (p / "data.yaml").exists()]
            if sub:
                base = sub[0]
        tag = f"d{n}"
        kept = dropped = 0
        for folder, target in SPLITS.items():
            img_dir = base / folder / "images"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lab = base / folder / "labels" / (img.stem + ".txt")
                lines = []
                if lab.exists():
                    for row in lab.read_text().splitlines():
                        p = row.split()
                        if not p:
                            continue
                        c = int(float(p[0]))
                        if c in cmap:
                            lines.append(" ".join([str(cmap[c])] + p[1:]))
                            kept += 1
                        else:
                            dropped += 1
                # Short file names: Roboflow names are very long and break the Windows 260-character path limit.
                # The first part is a hash of the original photo name and is kept the same for every augmented copy
                # (Roboflow's ".rf." marker), so the Colab notebook can still find and remove leaked validation images.
                totals[target] = totals.get(target, 0) + 1
                photo = img.name.split(".rf.")[0]
                key = hashlib.md5(f"{tag}/{photo}".encode()).hexdigest()[:12]
                stem = f"{tag}_{key}.rf.{totals[target]:05d}"
                shutil.copyfile(img, out / target / "images" / (stem + img.suffix.lower()))
                (out / target / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"source {n}: {src.name}  boxes kept {kept}, dropped {dropped}")
        if tmp:
            tmp.cleanup()

    if args.negatives:
        add_negatives(args, out, totals)

    names = ", ".join(repr(x) for x in args.names)
    (out / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
        f"nc: {len(args.names)}\nnames: [{names}]\n"
    )
    zip_path = ROOT / "datasets" / f"{args.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out).as_posix())
    print("images per split:", totals)
    print(f"Merged dataset: {out}")
    print(f"Zip for Colab : {zip_path}")


if __name__ == "__main__":
    main()
