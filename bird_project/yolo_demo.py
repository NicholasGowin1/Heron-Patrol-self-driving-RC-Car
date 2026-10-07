"""
Two-stage bird detection:  bird detector -> species classifier -> smoothed label.

    # Raspberry Pi camera
    python yolo_demo.py --detector bird_det.pt --classifier species.pt

    # USB / laptop webcam (0) or a video file, handy for testing off the Pi
    python yolo_demo.py --detector bird_det.pt --classifier species.pt --source 0
    python yolo_demo.py --detector bird_det.pt --classifier species.pt --source clip.mp4

How it decides a label:
  1. The detector finds anything bird-shaped (class "bird").
  2. Each bird is cropped and the classifier names the species. If it isn't at
     least --species-conf sure, that frame's vote is just "bird".
  3. A species is only shown once it wins --min-votes of the last --window frames
     for that bird, with a lead of at least 2 votes over any other species.
     Otherwise it stays "bird" (the safe parent label).

Press q to quit.
"""
import argparse
import csv
import time
from datetime import datetime
from collections import Counter, deque
from pathlib import Path

PAD = 0.08          # context around each box; must match prepare_data.py
PARENT = "bird"     # fallback label when the species isn't certain
SCRIPT_DIR = Path(__file__).resolve().parent  # weights are looked for here too
# Labels a YOLO-World model may output that are NOT birds; they are dropped from the display.
# (make_world_model.py puts these in the vocabulary so they soak up false "heron" hits.)
DEFAULT_IGNORE = "person,tree,rock,fence,car,plant,branch,sign,building,boat,dog,cat"


# --------------------------------------------------------------------------
# Helpers (no model / camera needed, so they're easy to test)
# --------------------------------------------------------------------------
def square_crop(frame, box, pad=PAD):
    """Square region centred on the box (plus padding), kept inside the frame."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    side = min(max(x2 - x1, y2 - y1) * (1 + 2 * pad), W, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(round(min(max(cx - side / 2, 0), W - side)))
    top = int(round(min(max(cy - side / 2, 0), H - side)))
    s = int(round(side))
    return frame[top:top + s, left:left + s]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Smoother:
    """Follows each bird across frames and votes on its species."""

    def __init__(self, window=5, min_votes=3, margin=2, iou_thr=0.3, max_missed=8):
        self.window, self.min_votes, self.margin = window, min_votes, margin
        self.iou_thr, self.max_missed = iou_thr, max_missed
        self.tracks = []  # each: {"box", "hist", "missed"}

    def update(self, boxes, labels):
        """boxes/labels for this frame -> final labels, same order."""
        pairs = sorted(
            ((iou(t["box"], b), ti, bi)
             for ti, t in enumerate(self.tracks) for bi, b in enumerate(boxes)),
            reverse=True)
        used_t, used_b, match = set(), set(), {}
        for score, ti, bi in pairs:
            if score < self.iou_thr:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            match[bi] = ti
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["missed"] += 1
        out = []
        for bi, (box, label) in enumerate(zip(boxes, labels)):
            if bi in match:
                t = self.tracks[match[bi]]
                t["box"], t["missed"] = box, 0
            else:
                t = {"box": box, "hist": deque(maxlen=self.window), "missed": 0}
                self.tracks.append(t)
            t["hist"].append(label)
            top = Counter(l for l in t["hist"] if l != PARENT).most_common(2)
            lead = top[0][1] - (top[1][1] if len(top) > 1 else 0) if top else 0
            # a species needs enough votes AND a clear lead, so heron/duck can't flip-flop
            out.append(top[0][0] if top and top[0][1] >= self.min_votes and lead >= self.margin else PARENT)
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
        return out


def save_capture(cap_dir, raw, annotated, boxes, labels, confs, cv2):
    """Save the untouched frame (for training) and the boxed frame (for review), plus a log row."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best = max(range(len(confs)), key=lambda i: confs[i])
    name = f"{ts}_{labels[best]}_{confs[best]:.2f}.jpg"
    (cap_dir / "raw").mkdir(parents=True, exist_ok=True)
    (cap_dir / "annotated").mkdir(parents=True, exist_ok=True)
    ok1 = cv2.imwrite(str(cap_dir / "raw" / name), raw)
    ok2 = cv2.imwrite(str(cap_dir / "annotated" / name), annotated)
    if not (ok1 and ok2):
        print(f"WARNING: could not write to {cap_dir}")
        return
    log = cap_dir / "log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "file", "label", "det_conf", "x1", "y1", "x2", "y2"])
        for b, l, c in zip(boxes, labels, confs):
            w.writerow([ts, name, l, f"{c:.3f}", *[int(v) for v in b]])
    print(f"[{ts}] captured {name}  ({len(boxes)} detection(s))")


# --------------------------------------------------------------------------
# Camera / video source
# --------------------------------------------------------------------------
def open_source(args, cv2):
    if args.source is None:
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("picamera2 not found (not on a Pi), using webcam 0 instead.")
            args.source = "0"
    if args.source is None:
        cam = Picamera2()
        cam.preview_configuration.main.size = (args.cam_size, args.cam_size)
        cam.preview_configuration.main.format = "RGB888"  # arrives as BGR, what OpenCV expects
        cam.preview_configuration.align()
        cam.configure("preview")
        cam.start()
        return cam.capture_array, cam.stop
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    def grab():
        ok, frame = cap.read()
        return frame if ok else None

    return grab, cap.release


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
COLORS = {"heron": (0, 200, 0), "duck": (0, 165, 255), PARENT: (190, 190, 190)}  # BGR


def run(args):
    import cv2
    from ultralytics import YOLO

    def find(name):
        for base in (Path.cwd(), SCRIPT_DIR):
            if (base / name).exists():
                return str(base / name)
        return None

    # No arguments needed: use the trained two-stage models if present,
    # otherwise fall back to the old heron model (weights.pt).
    if args.detector is None:
        args.detector = find("bird_det.pt") or find("world_birds.pt") or find("weights.pt")
    else:
        args.detector = find(args.detector)
    if args.classifier is None:
        args.classifier = find("species.pt")
    else:
        args.classifier = find(args.classifier)
    if not args.detector:
        raise SystemExit("No detector weights found. Put bird_det.pt (or weights.pt) next to yolo_demo.py.")
    print(f"Detector:   {args.detector}")
    print(f"Classifier: {args.classifier or 'none (detector-only)'}")

    detector = YOLO(args.detector)
    ignore = {x.strip().lower() for x in args.ignore.split(",") if x.strip()}
    classifier = YOLO(args.classifier) if args.classifier else None
    smoother = Smoother(args.window, args.min_votes)
    grab, close = open_source(args, cv2)
    font, fps, last = cv2.FONT_HERSHEY_SIMPLEX, 0.0, time.perf_counter()
    got_frame = False
    cap_dir = Path(args.capture_dir).expanduser() if args.capture_dir else SCRIPT_DIR / "photos"
    last_capture = float("-inf")
    if args.cooldown >= 0:
        print(f"Saving a photo when a bird is detected (then waiting {args.cooldown:.0f}s) to: {cap_dir}")

    while True:
        frame = grab()
        if frame is None:
            if not got_frame:
                print("Could not read any frames. Is a webcam connected / is the video path right? "
                      "Try --source 1, or --source path/to/clip.mp4")
            break
        got_frame = True

        res = detector(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
        confs = [float(c) for c in res.boxes.conf.cpu().numpy()]
        cls_ids = [int(c) for c in res.boxes.cls.cpu().numpy()]
        keep = [i for i, c in enumerate(cls_ids) if str(detector.names[c]).lower() not in ignore]
        boxes = [boxes[i] for i in keep]
        confs = [confs[i] for i in keep]
        cls_ids = [cls_ids[i] for i in keep]

        labels = []
        for i, box in enumerate(boxes):
            if classifier is None:  # detector-only fallback: use its own class names
                labels.append(detector.names[cls_ids[i]])
                continue
            crop = square_crop(frame, box)
            label = PARENT
            if crop.size:
                if args.grayscale:
                    crop = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                probs = classifier(crop, imgsz=args.cls_imgsz, verbose=False)[0].probs
                if float(probs.top1conf) >= args.species_conf:
                    label = classifier.names[int(probs.top1)]
            labels.append(label)

        final = smoother.update(boxes, labels)

        do_capture = (args.cooldown >= 0 and boxes
                      and time.monotonic() - last_capture >= args.cooldown)
        raw = frame.copy() if do_capture else None

        for (x1, y1, x2, y2), label, conf in zip(boxes, final, confs):
            color = COLORS.get(label, (255, 255, 0))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), max(int(y1) - 6, 14)),
                        font, 0.6, color, 2, cv2.LINE_AA)

        counts = Counter(final)
        summary = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no birds"
        cv2.putText(frame, summary, (10, 28), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
        last = now
        text = f"FPS: {fps:.1f}"
        tw = cv2.getTextSize(text, font, 1, 2)[0][0]
        cv2.putText(frame, text, (frame.shape[1] - tw - 10, 36), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if do_capture:
            save_capture(cap_dir, raw, frame, boxes, final, confs, cv2)
            last_capture = time.monotonic()

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Two-stage bird -> species demo")
    ap.add_argument("--detector", default=None, help="stage 1 weights (default: bird_det.pt, else weights.pt)")
    ap.add_argument("--classifier", default=None, help="stage 2 weights (default: species.pt if it exists)")
    ap.add_argument("--source", default=None, help="0 for a webcam, or a video path; default: Pi camera if available, else webcam 0")
    ap.add_argument("--imgsz", type=int, default=640, help="detector input size")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--cls-imgsz", type=int, default=224, help="classifier input size")
    ap.add_argument("--species-conf", type=float, default=0.70, help="min classifier confidence to vote for a species")
    ap.add_argument("--window", type=int, default=5, help="frames of history per bird")
    ap.add_argument("--min-votes", type=int, default=3, help="votes needed within the window to show a species")
    ap.add_argument("--ignore", default=DEFAULT_IGNORE,
                    help="comma-separated detector labels to hide (non-bird classes from a YOLO-World model)")
    ap.add_argument("--cooldown", type=float, default=300,
                    help="seconds to wait between saved photos (default 300); -1 turns saving off")
    ap.add_argument("--capture-dir", default=None, help="where photos go (default: photos/ next to the script)")
    ap.add_argument("--grayscale", action="store_true", help="only if you built the crops with --grayscale")
    ap.add_argument("--cam-size", type=int, default=1280, help="Pi camera frame size")
    run(ap.parse_args())    
