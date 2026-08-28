"""
Phase 1 + Phase 5 (pulled forward): Detection + Tracking + Team-Aware
Stabilization (placeholder detector)
----------------------------------------------------------------------
Goal of this script: prove the detect -> track -> draw pipeline works
end-to-end BEFORE you invest time training a custom football model.

We use a pretrained YOLOv8 model trained on COCO. COCO has a "person" class
and a "sports ball" class, which is good enough to validate your pipeline
on a real football clip. It will NOT distinguish player/goalkeeper/referee,
and ball detection will be lower-recall than a dedicated model. That's
expected -- fixing detection quality is Phase 2/3/4.

On top of raw ByteTrack, this script now layers:
  - Jersey-color team classification (team_assigner/)
  - A team + position based ID stabilizer (src/track_stabilizer.py) that
    re-stitches fragmented ByteTrack IDs when a new raw ID appears close to
    where a same-team player was just lost.

Usage:
    python src/detect_track.py --source input_videos/clip.mp4 --output output_videos/tracked.mp4
"""

import argparse
import sys
import os
import cv2
from ultralytics import YOLO
import supervision as sv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from team_assigner.team_assigner import TeamAssigner
from src.track_stabilizer import TrackStabilizer


# COCO class ids we care about for now
PERSON_CLASS_ID = 0
SPORTS_BALL_CLASS_ID = 32

# Display colors: index 0 = team 1, index 1 = team 2, index 2 = fallback
# (ball, or a player not yet classified during team-clustering warmup)
TEAM_PALETTE = sv.ColorPalette(colors=[
    sv.Color(255, 87, 51),   # team 1 -- orange
    sv.Color(51, 153, 255),  # team 2 -- blue
    sv.Color(180, 180, 180), # fallback -- gray
])
TEAM1_IDX, TEAM2_IDX, FALLBACK_IDX = 0, 1, 2

TEAM_BOOTSTRAP_MIN_SAMPLES = 15   # jersey-color samples collected before fitting the 2-team clusters
STABILIZER_MATCH_DISTANCE_PX = 90  # how close (in pixels) a new detection must be to a recently-lost same-team track to be merged
STABILIZER_MAX_FRAME_GAP = 45      # how many frames a lost track stays "eligible" for re-matching (~1.5s at ~30fps)


def run(source_path: str, output_path: str, model_name: str = "yolov8n.pt", conf: float = 0.45):
    import numpy as np

    # Loads (and auto-downloads on first run) a pretrained COCO model.
    model = YOLO(model_name)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ByteTrack tracker from the `supervision` library -- assigns a raw
    # persistent tracker_id to each detection across frames.
    #
    # NOTE: an earlier version of this script loosened these thresholds to
    # try to reduce ID churn. Testing showed that made things WORSE -- lower
    # activation thresholds let noisier, weaker detections into tracking,
    # and noisy detections churn through new IDs even faster. Reverted to
    # library defaults here. Filtering weak detections BEFORE tracking
    # (see `conf` below) is what actually helped.
    tracker = sv.ByteTrack(frame_rate=int(fps))

    # Team classifier (jersey-color clustering) and the ID stabilizer that
    # uses it to re-stitch fragmented ByteTrack IDs -- see team_assigner/
    # and src/track_stabilizer.py for the full explanation of the approach.
    team_assigner = TeamAssigner()
    stabilizer = TrackStabilizer(
        match_distance_px=STABILIZER_MATCH_DISTANCE_PX,
        max_frame_gap=STABILIZER_MAX_FRAME_GAP,
    )

    box_annotator = sv.EllipseAnnotator(color=TEAM_PALETTE)
    label_annotator = sv.LabelAnnotator(color=TEAM_PALETTE)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    raw_id_seen = set()       # for reporting: distinct raw ByteTrack ids seen
    display_id_seen = set()   # for reporting: distinct stabilized display ids seen

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run detection for people at the main confidence threshold, and
        # the ball separately at a lower threshold (see Phase 1 notes).
        person_results = model(frame, conf=conf, classes=[PERSON_CLASS_ID], verbose=False)[0]
        ball_results = model(frame, conf=0.10, classes=[SPORTS_BALL_CLASS_ID], verbose=False)[0]

        detections = sv.Detections.merge([
            sv.Detections.from_ultralytics(person_results),
            sv.Detections.from_ultralytics(ball_results),
        ])

        # Feed detections into ByteTrack to get raw persistent IDs
        detections = tracker.update_with_detections(detections)

        labels = []
        color_lookup = []  # per-detection index into TEAM_PALETTE

        for i in range(len(detections)):
            raw_id = int(detections.tracker_id[i])
            class_id = int(detections.class_id[i])
            bbox = detections.xyxy[i]
            is_person = class_id == PERSON_CLASS_ID

            raw_id_seen.add(raw_id)

            if is_person:
                # During warmup, keep collecting jersey-color samples until
                # we have enough to fit the two team clusters.
                if not team_assigner.is_ready:
                    team_assigner.add_bootstrap_sample(frame, bbox)
                    team_assigner.try_fit(min_samples=TEAM_BOOTSTRAP_MIN_SAMPLES)

                team = team_assigner.predict_team(frame, bbox)  # 1, 2, or None if not ready yet

                # Foot position (bottom-center of the box) is a better
                # position signal for re-matching than the box center,
                # since it changes less under partial occlusion.
                x1, y1, x2, y2 = bbox
                foot_pos = ((x1 + x2) / 2, y2)

                if team is not None:
                    display_id = stabilizer.update(raw_id, team, foot_pos, frame_idx)
                    palette_idx = TEAM1_IDX if team == 1 else TEAM2_IDX
                else:
                    display_id = raw_id  # not yet classified -- show raw id for now
                    palette_idx = FALLBACK_IDX

                display_id_seen.add(display_id)
                labels.append(f"#{display_id}")
                color_lookup.append(palette_idx)
            else:
                labels.append("ball")
                color_lookup.append(FALLBACK_IDX)

        color_lookup_arr = np.array(color_lookup, dtype=int) if color_lookup else np.array([], dtype=int)

        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections, custom_color_lookup=color_lookup_arr)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels, custom_color_lookup=color_lookup_arr)

        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames... (raw ids so far: {len(raw_id_seen)}, stabilized ids so far: {len(display_id_seen)})")

    cap.release()
    writer.release()
    print(f"Done. Wrote {frame_idx} frames to {output_path}")
    print(f"Total distinct raw ByteTrack ids: {len(raw_id_seen)}")
    print(f"Total distinct stabilized display ids: {len(display_id_seen)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to input video clip")
    parser.add_argument("--output", required=True, help="Path to write annotated output video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model to use (default: pretrained yolov8n)")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold (higher = fewer, cleaner detections)")
    args = parser.parse_args()

    run(args.source, args.output, args.model, args.conf)