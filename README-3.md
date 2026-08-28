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
