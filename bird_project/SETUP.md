# Setup guide: heron / duck bird detector

How to set up the Python environment, get the data, and run the project on a Raspberry Pi.
Commands assume the project lives in `~/Desktop/yolo object detect/bird_project`. Change the path if yours is different.

---

## 1. Data sources

Both datasets come from Roboflow (Colorado workspace `nigo6803-colorado-edu`):

| Dataset | Link |
|---|---|
| Herons (21k images) | https://app.roboflow.com/nigo6803-colorado-edu/herons-21k-dyhlc/1 |
| Ducks | https://app.roboflow.com/nigo6803-colorado-edu/ducks-civ2o-vwity/1 |

To download each one:

1. Log in to Roboflow and open the link (you need access to the workspace).
2. Click **Download Dataset**.
3. Choose the **YOLO26** format (the one the zip names `..._yolo26.zip` come from).
4. Choose **Download zip to computer**.

Keep the two zips as they are. You do NOT need to unzip them, because `prepare_data.py` extracts them itself.
Their names must contain `heron` and `duck` (for example `herons-21k.v1i.yolo26.zip` and `Ducks.v1i.yolo26.zip`).

---

## 2. Project files

Everything lives in one folder:

```
bird_project/
├── yolo_demo.py          live demo
├── make_world_model.py   builds the YOLO-World detector
├── prepare_data.py       builds training data from the two zips
├── train_colab.ipynb     trains the two-stage models on Google Colab
├── weights.pt            old heron-only model (optional stopgap)
├── photos/               created automatically; detection snapshots go here
├── venv/                 Python environment (created in step 3)
├── .gitignore
└── SETUP.md
```

---

## 3. Python environment (Raspberry Pi)

The Pi needs an internet connection for these steps.

### 3.1 System packages

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-venv
```

### 3.2 Virtual environment

`--system-site-packages` lets the venv see the camera library (`picamera2`) and OpenCV installed above.

```bash
cd "/home/arpi/Desktop/yolo object detect/bird_project"
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install ultralytics
```

This is a large download (it pulls in PyTorch), so it takes a while.

### 3.3 Check everything is visible

```bash
python -c "import cv2, ultralytics; from picamera2 import Picamera2; print('all good')"
```

### 3.4 Check the camera

```bash
rpicam-hello -t 5000
```

This should show a 5-second preview. On older Pi OS versions the command is `libcamera-hello`.
If there is no picture, check that the camera ribbon cable is seated properly.

### Each time you open a new terminal

```bash
cd "/home/arpi/Desktop/yolo object detect/bird_project"
source venv/bin/activate
```

You should see `(venv)` at the start of the prompt.

---

## 4. Option A: YOLO-World detector (no training needed)

YOLO-World detects objects from a list of words, so it needs no training. It also handles "not a bird" much better than the old heron-only model.

### 4.1 Build the model (once, needs internet)

```bash
python make_world_model.py
```

This automatically downloads:
- `yolov8s-worldv2.pt`, the YOLO-World base model
- the CLIP text model (a few hundred MB); it may also install a small CLIP package

and creates `world_birds.pt`.

Options:

```bash
# choose different bird words
python make_world_model.py --birds heron duck egret

# add non-bird words to cut false alarms (e.g. a fence post that looks like a heron)
python make_world_model.py --background person tree rock fence post
```

### 4.2 Optional: export to ONNX for speed

NCNN export does NOT work for YOLO-World (the model crashes on load). ONNX does export and is normally faster than plain PyTorch.

```bash
pip install onnxruntime
python -c "from ultralytics import YOLO; YOLO('world_birds.pt').export(format='onnx', imgsz=320)"
```

### 4.3 Run it

```bash
python yolo_demo.py --detector world_birds.pt --conf 0.15 --cam-size 640 --imgsz 320
# or, if you exported ONNX:
python yolo_demo.py --detector world_birds.onnx --conf 0.15 --cam-size 640 --imgsz 320
```

`--imgsz` must match the size you exported at. Press `q` in the window to quit.

---

## 5. Option B: train the two-stage models (more accurate, needs the data)

A detector finds any bird, then a classifier names it heron or duck. Training needs a GPU, so use Google Colab, not the Pi.

### 5.1 Train on Google Colab (easiest)

1. Go to https://colab.research.google.com and choose **Upload**, then select `train_colab.ipynb`.
2. **Runtime > Change runtime type > T4 GPU**.
3. Run the cells in order. When asked to upload files, select:
   - `prepare_data.py`
   - the herons zip
   - the ducks zip
4. At the end, `bird_det.pt` and `species.pt` download to your Downloads folder.

Move them into the project folder:

```bash
mv ~/Downloads/bird_det.pt ~/Downloads/species.pt "/home/arpi/Desktop/yolo object detect/bird_project/"
```

### 5.2 Or train locally (needs an NVIDIA GPU)

```bash
pip install ultralytics
python prepare_data.py --herons herons-21k.v1i.yolo26.zip --ducks Ducks.v1i.yolo26.zip
yolo detect train data=prepared/bird_det/data.yaml model=yolo26n.pt epochs=100 imgsz=640 scale=0.9 project=runs name=bird_det
yolo classify train data=prepared/species_cls model=yolo26n-cls.pt epochs=40 imgsz=224 project=runs name=species
find runs -name best.pt
```

Copy the two `best.pt` files into the project folder as `bird_det.pt` and `species.pt`.

Optional `prepare_data.py` flags:
- `--other <folder>` adds an "other" class from photos of other birds (or backyard patches)
- `--grayscale` strips colour (if the classifier keys on background colour, or you use infrared at night). Then also run the demo with `--grayscale`.

### 5.3 Run it

```bash
python yolo_demo.py --detector bird_det.pt --classifier species.pt
```

### 5.4 Optional: export to NCNN for speed on the Pi

This works for the trained nano models (unlike YOLO-World):

```bash
yolo export model=bird_det.pt format=ncnn imgsz=320
yolo export model=species.pt format=ncnn imgsz=224
python yolo_demo.py --detector bird_det_ncnn_model --classifier species_ncnn_model --imgsz 320 --cls-imgsz 224
```

---

## 6. Running the demo

With no arguments, the demo picks weights in this order: `bird_det.pt`, then `world_birds.pt`, then `weights.pt`. It uses `species.pt` as the classifier if present.

```bash
python yolo_demo.py
```

On a Pi it uses the Pi camera. On other computers it uses webcam 0.

Useful flags:

| Flag | What it does |
|---|---|
| `--source clip.mp4` | run on a video file (or `0` for a webcam) |
| `--conf 0.15` | detector confidence threshold (lower finds more, raise to cut false alarms) |
| `--imgsz 320` | detector size (smaller is faster) |
| `--cam-size 640` | Pi camera frame size (smaller is faster) |
| `--min-votes 4` | stricter smoothing to reduce flicker |
| `--cooldown 300` | seconds between saved photos (`-1` turns saving off) |
| `--capture-dir <path>` | save photos somewhere other than `photos/` |

### Saved photos

When a bird is detected, a photo is saved to `photos/`, then it waits 300 seconds before saving another:

- `photos/raw/` untouched frames (use these as training images or backgrounds)
- `photos/annotated/` the same frames with boxes drawn, for reviewing accuracy
- `photos/log.csv` time, label, confidence and box for every detection

Frames with false alarms are the most useful negatives to collect. Add them to training as backgrounds.

---

## 7. Testing without a heron

Herons only visit for a couple of weeks a year, so test with stand-ins:

```bash
python yolo_demo.py --source some_heron_video.mp4
```

You can also point the camera at a screen playing a heron video.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'cv2'` | the venv isn't active, or `python3-opencv` isn't installed (step 3) |
| `No module named 'picamera2'` | recreate the venv with `--system-site-packages` |
| `Temporary failure in name resolution` | no internet; check Wi-Fi, then retry |
| `--detector` is required | you have an old `yolo_demo.py`; replace it with the latest |
| Segmentation fault loading a YOLO-World NCNN model | NCNN isn't supported for YOLO-World; use `.pt` or ONNX (step 4.2) |
| Frame rate too low | smaller `--cam-size` and `--imgsz`; export to ONNX or NCNN |
| Every bird or object called a heron | use the YOLO-World model (section 4), or train with backgrounds (section 5) |

---

## 9. Git

The `.gitignore` excludes `venv/`, `photos/`, model weights (`*.pt`, `*.onnx`, `*_ncnn_model/`), the dataset zips, `prepared/`, `runs/` and video files, so nothing over GitHub's 100 MB limit gets uploaded.

To rebuild the project on a new machine, clone the repo, then redo steps 1 and 3, plus step 4 (`make_world_model.py`) or step 5 (training), since the weights are not stored in git.
