"""
Phase 1 + Phase 3 + Phase 5: Detection + Tracking + Team-Aware Stabilization
------------------------------------------------------------------------------
This script now auto-detects whether it's been given:
  (a) your custom-trained 4-class football model (player/goalkeeper/referee/
      ball), or
  (b) the original pretrained COCO placeholder (yolov8n.pt),
by inspecting the loaded model's class names, and adjusts detection +
labeling accordingly. Old commands using yolov8n.pt still work exactly as
before -- this is additive, not a breaking change.

With the custom model, two real improvements become possible now that we
have a genuine `referee` class instead of lumping everyone into "person":
  - Referees are excluded from jersey-color team clustering entirely --
    previously a referee's kit color could pollute the 2-team k-means fit.
    They now get their own distinct label/color instead.
  - Goalkeepers get a "GK" prefix in their label. Note: team-color
    assignment for goalkeepers is still done via the same 2-cluster jersey
    model as outfield players (a known limitation -- see team_assigner.py),
    so a keeper's team color can still be wrong even though their ROLE is
    now correctly detected.

On top of raw ByteTrack, this script layers:
  - Jersey-color team classification (team_assigner/)
  - A team + position based ID stabilizer (src/track_stabilizer.py) that
    re-stitches fragmented ByteTrack IDs when a new raw ID appears close to
    where a same-team player was just lost.

Usage:
    python src/detect_track.py --source input_videos/clip.mp4 --output output_videos/tracked.mp4 --model runs/detect/train/weights/best.pt
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


# COCO fallback class ids (used only if the loaded model isn't our custom one)
COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32

CUSTOM_CLASS_NAMES = {"player", "goalkeeper", "referee", "ball"}

# Display colors: team1, team2, referee, fallback (ball / not-yet-classified)
TEAM_PALETTE = sv.ColorPalette(colors=[
    sv.Color(255, 87, 51),    # team 1 -- orange
    sv.Color(51, 153, 255),   # team 2 -- blue
    sv.Color(255, 215, 0),    # referee -- yellow
    sv.Color(180, 180, 180),  # fallback -- gray (ball, or unclassified)
])
TEAM1_IDX, TEAM2_IDX, REFEREE_IDX, FALLBACK_IDX = 0, 1, 2, 3

TEAM_BOOTSTRAP_MIN_SAMPLES = 15   # jersey-color samples collected before fitting the 2-team clusters
STABILIZER_MATCH_DISTANCE_PX = 90  # how close (in pixels) a new detection must be to a recently-lost same-team track to be merged
STABILIZER_MAX_FRAME_GAP = 45      # how many frames a lost track stays "eligible" for re-matching (~1.5s at ~30fps)


def resolve_class_ids(model) -> dict:
    """Inspect the loaded model's class names and return a dict of role ->
    class_id (or None if that role doesn't exist in this model), plus a
    flag for whether this is our custom football model or the COCO
    fallback."""
    names = model.names  # dict: class_id -> name
    name_to_id = {v: k for k, v in names.items()}

    if CUSTOM_CLASS_NAMES.issubset(set(names.values())):
        return {
            "is_custom": True,
            "player": name_to_id["player"],
            "goalkeeper": name_to_id["goalkeeper"],
            "referee": name_to_id["referee"],
            "ball": name_to_id["ball"],
        }
    else:
        return {
            "is_custom": False,
            "player": COCO_PERSON_CLASS_ID,
            "goalkeeper": None,
            "referee": None,
            "ball": COCO_SPORTS_BALL_CLASS_ID,
        }


def run(source_path: str, output_path: str, model_name: str = "yolov8n.pt", conf: float = 0.45):
    import numpy as np

    # Loads your custom-trained weights, or auto-downloads the pretrained
    # COCO model on first run if you pass the default yolov8n.pt.
    model = YOLO(model_name)
    class_ids = resolve_class_ids(model)
    if class_ids["is_custom"]:
        print(f"Loaded custom football model -- classes: {model.names}")
    else:
        print(f"Loaded COCO fallback model ({model_name}) -- no goalkeeper/referee distinction available.")

    person_like_classes = [c for c in (class_ids["player"], class_ids["goalkeeper"], class_ids["referee"]) if c is not None]
    ball_class = class_ids["ball"]

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

        # Run detection for people/goalkeeper/referee at the main confidence
        # threshold, and the ball separately at a lower threshold -- the
        # ball is a much smaller, harder target regardless of which model
        # is loaded, so it's held to a more lenient bar.
        person_results = model(frame, conf=conf, classes=person_like_classes, verbose=False)[0]
        ball_results = model(frame, conf=0.10, classes=[ball_class], verbose=False)[0]

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

            is_referee = class_id == class_ids["referee"]
            is_goalkeeper = class_id == class_ids["goalkeeper"]
            is_ball = class_id == class_ids["ball"]
            # Team-eligible: player or goalkeeper (or generic "person" in
            # COCO fallback mode, where goalkeeper/referee don't exist as
            # separate classes). Referees are explicitly excluded -- see
            # module docstring for why.
            is_team_eligible = not is_referee and not is_ball

            raw_id_seen.add(raw_id)

            if is_team_eligible:
                # During warmup, keep collecting jersey-color samples until
                # we have enough to fit the two team clusters. Referees
                # never contribute samples here, which keeps the 2-team
                # color clusters cleaner when using the custom model.
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
                prefix = "GK " if is_goalkeeper else ""
                labels.append(f"{prefix}#{display_id}")
                color_lookup.append(palette_idx)
            elif is_referee:
                labels.append(f"ref #{raw_id}")
                color_lookup.append(REFEREE_IDX)
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