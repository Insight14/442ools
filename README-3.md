# FootballVision-AI — From Scratch

Building this yourself, phase by phase, rather than cloning an existing repo.
Each phase produces something you can actually run and see before moving on.

## Setup

```bash
cd footballvision-scratch
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of torch *before* the
requirements above (otherwise `ultralytics` will pull the CPU-only version):

```bash
# check your CUDA version first: nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Phase 1 (this step): Detection + Tracking, placeholder model

Drop any football clip into `input_videos/` (a 10-20 second clip is plenty
to iterate on) and run:

```bash
python src/detect_track.py --source input_videos/clip.mp4 --output output_videos/tracked.mp4
```

What this proves:
- Frame-by-frame YOLO inference works
- Detections flow correctly into ByteTrack
- Tracker IDs stay consistent as players move across frames
- Ellipse-style annotation renders (matching the broadcast look, not boxes)

What it does NOT do yet (by design):
- No goalkeeper/referee distinction (COCO only knows "person")
- Ball detection will be spotty (COCO's "sports ball" wasn't trained for
  small fast-moving broadcast footage)
- No team classification, no pitch calibration, no speed/distance, no
  possession -- all later phases

Watch the output video. If tracker IDs are flickering or switching between
players a lot, note it now -- ByteTrack tuning (confidence thresholds,
track buffer) is something we'll revisit once real classes are in play.

## Next phases (not started yet)

- **Phase 2 — Data labeling.** Label your own player/goalkeeper/referee/ball
  dataset (CVAT or Label Studio), or pull an existing one from Roboflow
  Universe to fine-tune on. This is the actual bottleneck of the whole
  project -- budget real time here.
- **Phase 3 — Fine-tune YOLO** on your labeled data to replace the
  COCO placeholder with real football classes.
- **Phase 4 — Dedicated ball model**, trained/fine-tuned separately at
  higher resolution since it's a much smaller, faster-moving target.
- **Phase 5 — Team classification** via jersey color clustering.
- **Phase 6 — Pitch keypoint detection + homography** (pixels -> metres).
- **Phase 7 — Speed/distance/possession/heatmaps** analytics layer.
- **Phase 8 — Final rendering** (overlays, CSV export, match report).

We'll build these one at a time, testing each before adding the next.

## Train and run the custom detector

After adding or changing labels, rebuild the dataset split and retrain. The
training script validates the image/label folders and always refreshes the
canonical checkpoint at `runs/detect/train/weights/best.pt`:

```bash
source venv/bin/activate
python dataset/split_by_clip.py --labeled-dir data/labeled --val-clips olise_clip
python src/train_detector.py --epochs 50
python src/detect_track.py --source input_videos/olise_clip.mov --output output_videos/tracked_custom.mp4
```

The detector now automatically uses the trained checkpoint when it exists.
Use `--model yolov8n.pt` only when you intentionally want the original COCO
baseline. The training split must contain images and matching `.txt` labels;
rerunning `split_by_clip.py` is required after uploading new labeled frames.

## Prediction layer

The detector can export pitch-space tracks and an interpretable baseline
prediction stream when a homography is supplied:

```bash
python src/detect_track.py \
  --source input_videos/olise_clip.mov \
  --output output_videos/tracked_predictions.mp4 \
  --homography pitch_calibration/homography_olise_clip.json \
  --events-output output_videos/olise_predictions.jsonl
```

The JSONL contains player positions in metres, ball position, possession,
pass/shot suggestions, completed-pass events, and a coarse play label. These
are transparent heuristics for a baseline and training-data generator, not
match-ready predictions.

When `data_quality` is `ball_missing`, do not interpret the play label. The
current custom detector has insufficient ball confidence on the Olise clip,
so the next model task is a dedicated ball detector: label more small-ball
instances, train at `imgsz=1280` or higher, validate ball recall, and only
then enable possession and pass/shot evaluation. Player tracking and pitch
coordinates can be developed in parallel.
