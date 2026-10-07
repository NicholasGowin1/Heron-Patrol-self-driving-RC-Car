"""
Build a YOLO-World bird detector with a fixed vocabulary (run ONCE, needs internet).

    python make_world_model.py
    python make_world_model.py --birds heron duck gull egret goose --export-ncnn

Creates world_birds.pt (and world_birds_ncnn_model/ with --export-ncnn).
yolo_demo.py picks world_birds.pt up automatically when bird_det.pt isn't present,
or use:  python yolo_demo.py --detector world_birds.pt --conf 0.15

Why background words are included: YOLO-World must label every box with SOMETHING from
its vocabulary. Giving it "tree", "rock", "person"... lets it say "not a bird" instead of
forcing a wrong "heron". The demo hides those labels.
"""
import argparse

BACKGROUND = ["person", "tree", "rock", "fence", "car", "plant", "branch", "sign",
              "building", "boat", "dog", "cat"]

ap = argparse.ArgumentParser()
ap.add_argument("--birds", nargs="+", default=["heron", "duck", "gull", "egret", "goose"],
                help="bird words to detect")
ap.add_argument("--background", nargs="*", default=BACKGROUND, help="non-bird words")
ap.add_argument("--base", default="yolov8s-worldv2.pt", help="YOLO-World weights to start from")
ap.add_argument("--out", default="world_birds.pt")
ap.add_argument("--export-ncnn", action="store_true", help="also export a faster NCNN version (Pi)")
ap.add_argument("--imgsz", type=int, default=320, help="NCNN export size; use the same --imgsz in the demo")
args = ap.parse_args()

from ultralytics import YOLO

model = YOLO(args.base)
model.set_classes(args.birds + args.background)   # needs the CLIP text encoder (auto-downloads)
model.save(args.out)
print("saved", args.out, "with classes:", args.birds + args.background)

if args.export_ncnn:
    try:
        YOLO(args.out).export(format="ncnn", imgsz=args.imgsz)
        print("exported NCNN model (look for the *_ncnn_model folder)")
    except Exception as e:
        print("NCNN export failed, the .pt file still works:", e)
