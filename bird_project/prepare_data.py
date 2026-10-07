"""
Build the training data for the two-stage bird pipeline from the two Roboflow
YOLO exports (herons + ducks).

Creates, inside --out:
    bird_det/      one-class detector dataset (everything labelled 'bird')
    species_cls/   classifier dataset:  train|val|test / heron|duck[|other] / *.jpg

Usage:
    python prepare_data.py --herons herons-21k_v1i_yolo26.zip --ducks Ducks_v1i_yolo26.zip
    python prepare_data.py --herons <folder> --ducks <folder> --other <folder of other-bird photos>

Zips or already-extracted folders both work.

Why the crops are degraded: herons here are big close-ups (~350 px) and ducks are
tiny (~50 px). Left alone, a classifier would learn "sharp = heron, blurry = duck".
So every crop of every species is shrunk to a random 20-64 px effective
resolution and scaled back up, which makes resolution useless as a clue.
"""
import argparse
import math
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # source name -> classifier name
PAD = 0.08  # context around each box; the demo uses the same value


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def resolve_root(src: str, work: Path) -> Path:
    p = Path(src).expanduser()
    if p.suffix.lower() == ".zip":
        dest = work / "_extracted" / p.stem
        if not dest.exists():
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
        p = dest
    for cand in [p, *sorted(p.iterdir())]:
        if cand.is_dir() and not cand.name.startswith("__MACOSX") and (cand / "train").is_dir():
            return cand
    raise SystemExit(f"{src}: couldn't find a 'train' folder inside it")


def read_boxes(label_path: Path, W: int, H: int):
    """YOLO txt -> list of (x1, y1, x2, y2) in pixels. Handles polygon labels too."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        if len(p) == 5:
            cx, cy, w, h = map(float, p[1:5])
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs = [float(v) for v in p[1::2]]
            ys = [float(v) for v in p[2::2]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append((max(x1, 0) * W, max(y1, 0) * H, min(x2, 1) * W, min(y2, 1) * H))
    return boxes


def square_crop(img: Image.Image, box, pad=PAD):
    """Square region centred on the box (box + pad), kept inside the image."""
    W, H = img.size
    x1, y1, x2, y2 = box
    side = min(max(x2 - x1, y2 - y1) * (1 + 2 * pad), W, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - side / 2, 0), W - side)
    top = min(max(cy - side / 2, 0), H - side)
    return img.crop((round(left), round(top), round(left + side), round(top + side)))


def degrade(crop: Image.Image, rng, lo, hi, out):
    """Shrink to a random effective resolution, then scale back up to out x out."""
    r = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    r = int(max(8, min(r, crop.width)))  # can't be sharper than the source
    return crop.resize((r, r), Image.BOX).resize((out, out), Image.BICUBIC)


def collect(root: Path, split: str, min_box: int):
    entries = []
    imgdir, lbldir = root / split / "images", root / split / "labels"
    if not imgdir.is_dir():
        return entries
    for img in sorted(imgdir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS or img.name.startswith("._"):
            continue
        with Image.open(img) as im:
            W, H = im.size
        for box in read_boxes(lbldir / (img.stem + ".txt"), W, H):
            if max(box[2] - box[0], box[3] - box[1]) >= min_box:
                entries.append((img, box))
    return entries


def write_crops(items, label, out_dir: Path, args, rng):
    """items: list of (image_path, box, variant). Opens each image once."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_img = defaultdict(list)
    for img, box, v in items:
        by_img[img].append((box, v))
    n = 0
    for img, group in by_img.items():
        with Image.open(img) as im:
            im = im.convert("RGB")
            for k, (box, v) in enumerate(group):
                crop = square_crop(im, box)
                out = degrade(crop, rng, args.res_min, args.res_max, args.size)
                if args.grayscale:
                    out = out.convert("L").convert("RGB")
                out.save(out_dir / f"{label}_{img.stem}_{k}_{v}.jpg", quality=92)
                n += 1
    return n


# --------------------------------------------------------------------------
# Stage 1 data: merged one-class detector set
# --------------------------------------------------------------------------
def build_detector(herons: Path, ducks: Path, out: Path):
    if out.exists():
        shutil.rmtree(out)
    counts = defaultdict(int)
    for src, prefix in ((herons, "h"), (ducks, "d")):
        for split in SPLITS:
            imgdir, lbldir = src / split / "images", src / split / "labels"
            if not imgdir.is_dir():
                continue
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            for img in imgdir.iterdir():
                if img.suffix.lower() not in IMG_EXTS or img.name.startswith("._"):
                    continue
                shutil.copy2(img, out / split / "images" / f"{prefix}_{img.name}")
                lines = []
                lf = lbldir / (img.stem + ".txt")
                if lf.exists():
                    for line in lf.read_text().splitlines():
                        p = line.split()
                        if p:
                            lines.append(" ".join(["0"] + p[1:]))  # every class -> 0 (bird)
                (out / split / "labels" / f"{prefix}_{img.stem}.txt").write_text("\n".join(lines))
                counts[split] += 1
    # No 'path:' on purpose: Ultralytics then resolves paths relative to this yaml,
    # so the folder still works after you move or upload it.
    (out / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: bird\n")
    print("bird_det images:", dict(counts))


# --------------------------------------------------------------------------
# Stage 2 data: species crops
# --------------------------------------------------------------------------
def build_classifier(herons: Path, ducks: Path, out: Path, args):
    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(args.seed)
    summary = []
    for src_split, dst in SPLITS.items():
        h = collect(herons, src_split, args.min_box)
        d = collect(ducks, src_split, args.min_box)
        variants = args.heron_variants if src_split == "train" else 1
        h_items = [(img, box, v) for img, box in h for v in range(variants)]
        n_d = min(len(d), round(len(h_items) * args.duck_ratio))  # balance classes
        d_items = [(img, box, 0) for img, box in rng.sample(d, n_d)]
        nh = write_crops(h_items, "heron", out / dst / "heron", args, rng)
        nd = write_crops(d_items, "duck", out / dst / "duck", args, rng)
        summary.append(f"  {dst:5s} heron={nh:5d}  duck={nd:5d}")

    if args.other:
        imgs = sorted(p for p in Path(args.other).expanduser().rglob("*")
                      if p.suffix.lower() in IMG_EXTS and not p.name.startswith("._"))
        rng.shuffle(imgs)
        cut = {"train": imgs[: int(len(imgs) * 0.8)],
               "val": imgs[int(len(imgs) * 0.8): int(len(imgs) * 0.9)],
               "test": imgs[int(len(imgs) * 0.9):]}
        for dst, files in cut.items():
            items = []
            for f in files:
                with Image.open(f) as im:
                    W, H = im.size
                items.append((f, (0, 0, W, H), 0))
            n = write_crops(items, "other", out / dst / "other", args, rng)
            summary.append(f"  {dst:5s} other={n}")
    print("species_cls crops:\n" + "\n".join(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--herons", required=True, help="heron dataset (zip or folder)")
    ap.add_argument("--ducks", required=True, help="duck dataset (zip or folder)")
    ap.add_argument("--other", default=None, help="optional folder of photos of OTHER birds (gulls, egrets...)")
    ap.add_argument("--out", default="prepared")
    ap.add_argument("--heron-variants", type=int, default=3, help="degraded copies per heron crop in train")
    ap.add_argument("--duck-ratio", type=float, default=1.0, help="ducks per heron crop (1.0 = balanced)")
    ap.add_argument("--res-min", type=int, default=20, help="min effective crop resolution (px)")
    ap.add_argument("--res-max", type=int, default=64, help="max effective crop resolution (px)")
    ap.add_argument("--min-box", type=int, default=12, help="skip boxes smaller than this (px)")
    ap.add_argument("--size", type=int, default=224, help="saved crop size")
    ap.add_argument("--grayscale", action="store_true",
                    help="strip colour from crops (use if the classifier keys on background colour); "
                         "then run the demo with --grayscale too")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    herons = resolve_root(args.herons, out)
    ducks = resolve_root(args.ducks, out)

    build_detector(herons, ducks, out / "bird_det")
    build_classifier(herons, ducks, out / "species_cls", args)
    print(f"\nDone. Output in {out}")


if __name__ == "__main__":
    main()
