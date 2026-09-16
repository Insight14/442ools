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
import json
import sys
import os
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from team_assigner.team_assigner import TeamAssigner
from src.track_stabilizer import TrackStabilizer
from src.play_predictor import PlayerState, PlayPredictor
from pitch_calibration.pitch_transformer import PitchTransformer


# COCO fallback class ids (used only if the loaded model isn't our custom one)
COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32

CUSTOM_CLASS_NAMES = {"player", "goalkeeper", "referee", "ball"}

# Display colors: team1, team2, referee, fallback, goalkeeper.
TEAM_PALETTE = sv.ColorPalette(colors=[
    sv.Color(255, 87, 51),    # team 1 -- orange
    sv.Color(51, 153, 255),   # team 2 -- blue
    sv.Color(255, 215, 0),    # referee -- yellow
    sv.Color(180, 180, 180),  # fallback -- gray (ball, or unclassified)
    sv.Color(191, 64, 224),   # goalkeeper -- purple
])
TEAM1_IDX, TEAM2_IDX, REFEREE_IDX, FALLBACK_IDX, GOALKEEPER_IDX = 0, 1, 2, 3, 4

# OpenCV uses BGR tuples while the supervision palette uses RGB colors.
RING_COLORS_BGR = [
    (51, 87, 255),    # team 1 -- orange
    (255, 153, 51),   # team 2 -- blue
    (0, 215, 255),    # referee -- yellow
    (180, 180, 180),  # fallback -- gray
    (224, 64, 191),   # goalkeeper -- purple
]
PREDICTION_PANEL_WIDTH = 360


def draw_tech_marker(frame, center, axes, color):
    """Draw a restrained segmented ground marker around a tracked object."""
    center = (int(center[0]), int(center[1]))
    axes = (int(axes[0]), int(axes[1]))
    outer_axes = (axes[0] + 2, axes[1] + 1)
    cv2.ellipse(frame, center, outer_axes, 0, 0, 360, color, 1, cv2.LINE_AA)
    for start_angle, end_angle, thickness in ((12, 102, 3), (132, 214, 2), (246, 326, 3)):
        cv2.ellipse(frame, center, axes, 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, center, (max(2, axes[0] - 3), max(2, axes[1] - 2)), 0, 168, 206, color, 1, cv2.LINE_AA)


def draw_glowing_ball(frame, center, radius, trail):
    """Draw a bright ball core with a short fading motion trail."""
    glow_color = (40, 150, 255)  # warm orange in BGR
    glow_layer = np.zeros_like(frame)
    for index, point in enumerate(trail):
        age_ratio = (index + 1) / max(1, len(trail))
        point_radius = max(2, int(radius * (0.45 + age_ratio * 0.35)))
        cv2.circle(glow_layer, (int(point[0]), int(point[1])), point_radius, glow_color, -1, cv2.LINE_AA)
    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=max(3.0, radius * 1.8))
    cv2.addWeighted(glow_layer, 0.28, frame, 0.72, 0, frame)

    for start, end in zip(trail, list(trail)[1:]):
        cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), glow_color, max(1, radius // 2), cv2.LINE_AA)
    ball_center = (int(center[0]), int(center[1]))
    cv2.circle(frame, ball_center, max(3, radius + 2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, ball_center, max(2, radius), (230, 245, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, ball_center, max(1, radius // 2), (255, 255, 255), -1, cv2.LINE_AA)


def draw_prediction_panel(frame, prediction, player_pixels):
    """Add a tactical sidebar and projected pass/shot suggestion paths."""
    panel = np.zeros((frame.shape[0], PREDICTION_PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (12, 20, 31)
    canvas = np.concatenate([frame, panel], axis=1)
    panel_x = frame.shape[1]
    cyan = (255, 210, 80)
    white = (235, 242, 248)
    muted = (145, 165, 180)
    orange = (70, 150, 255)
    green = (100, 220, 130)

    cv2.line(canvas, (panel_x, 0), (panel_x, canvas.shape[0]), (45, 75, 95), 2)
    cv2.putText(canvas, "LIVE PLAY INTELLIGENCE", (panel_x + 22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, cyan, 2, cv2.LINE_AA)
    cv2.putText(canvas, "442OOLS / TACTICAL FEED", (panel_x + 22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    cv2.line(canvas, (panel_x + 22, 88), (panel_x + PREDICTION_PANEL_WIDTH - 22, 88), (45, 75, 95), 1)

    play = prediction.get("play", "unknown").replace("_", " ").upper()
    quality = prediction.get("data_quality", "unknown").replace("_", " ").upper()
    possession = prediction.get("possession_track_id")
    possession_text = f"TRACK #{possession}" if possession is not None else "UNCONFIRMED"
    cv2.putText(canvas, "CURRENT PLAY", (panel_x + 22, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, play, (panel_x + 22, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.9, white if quality == "USABLE" else orange, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"POSSESSION  {possession_text}", (panel_x + 22, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.47, white, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"DATA QUALITY  {quality}", (panel_x + 22, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.47, green if quality == "USABLE" else orange, 1, cv2.LINE_AA)

    cv2.putText(canvas, "NEXT ACTIONS", (panel_x + 22, 264), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cyan, 1, cv2.LINE_AA)
    suggestions = prediction.get("suggestions", [])
    if not suggestions:
        cv2.putText(canvas, "No confident action", (panel_x + 22, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, muted, 1, cv2.LINE_AA)
    for index, suggestion in enumerate(suggestions[:4]):
        y = 304 + index * 60
        kind = suggestion["type"].upper()
        confidence = int(round(suggestion.get("confidence", 0) * 100))
        if kind == "PASS":
            detail = f"#{suggestion['from_track_id']}  ->  #{suggestion['to_track_id']}"
        else:
            detail = f"TRACK #{suggestion.get('from_track_id', '?')}  /  {kind}"
        cv2.putText(canvas, f"{kind}  {confidence}%", (panel_x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, white, 2, cv2.LINE_AA)
        cv2.putText(canvas, detail, (panel_x + 22, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, muted, 1, cv2.LINE_AA)

    for event in prediction.get("events", [])[:2]:
        y = 570
        cv2.putText(canvas, f"EVENT  {event['type'].replace('_', ' ').upper()}", (panel_x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, cyan, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"confidence {int(event.get('confidence', 0) * 100)}%", (panel_x + 22, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, muted, 1, cv2.LINE_AA)

    for suggestion in suggestions[:4]:
        if suggestion["type"] != "pass":
            continue
        source = player_pixels.get(suggestion.get("from_track_id"))
        target = player_pixels.get(suggestion.get("to_track_id"))
        if source is None or target is None:
            continue
        start = (int(source[0]), int(source[1]))
        end = (int(target[0]), int(target[1]))
        cv2.line(canvas, start, end, (60, 190, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, end, 7, (60, 190, 255), -1, cv2.LINE_AA)

    return canvas

TEAM_BOOTSTRAP_MIN_SAMPLES = 15   # jersey-color samples collected before fitting the 2-team clusters
STABILIZER_MATCH_DISTANCE_PX = 90  # how close (in pixels) a new detection must be to a recently-lost same-team track to be merged
STABILIZER_MAX_FRAME_GAP = 45      # how many frames a lost track stays "eligible" for re-matching (~1.5s at ~30fps)


def default_model_path() -> str:
    """Prefer trained weights, while keeping a fresh checkout runnable."""
    trained_model = os.path.join("runs", "detect", "train", "weights", "best.pt")
    return trained_model if os.path.isfile(trained_model) else "yolov8n.pt"


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


def run(source_path: str, output_path: str, model_name: str = "yolov8n.pt", conf: float = 0.45,
    homography_path: str | None = None, events_output: str | None = None,
    attacking_direction: int = 1, ball_conf: float = 0.05, ball_imgsz: int = 1280):
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
    transformer = PitchTransformer(homography_path) if homography_path else None
    predictor = PlayPredictor(attacking_direction=attacking_direction) if transformer else None
    events_file = open(events_output, "w") if events_output else None

    label_annotator = sv.LabelAnnotator(color=TEAM_PALETTE)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_width = width + PREDICTION_PANEL_WIDTH if predictor else width
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))

    frame_idx = 0
    ball_trail = deque(maxlen=10)
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
        ball_results = model(frame, conf=ball_conf, imgsz=ball_imgsz, classes=[ball_class], verbose=False)[0]

        detections = sv.Detections.merge([
            sv.Detections.from_ultralytics(person_results),
            sv.Detections.from_ultralytics(ball_results),
        ])

        # Feed detections into ByteTrack to get raw persistent IDs
        detections = tracker.update_with_detections(detections)

        labels = []
        color_lookup = []  # per-detection index into TEAM_PALETTE
        ring_specs = []
        ball_render_specs = []
        player_pixels = {}
        players = []
        ball_position = None

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
                    palette_idx = GOALKEEPER_IDX if is_goalkeeper else (TEAM1_IDX if team == 1 else TEAM2_IDX)
                else:
                    display_id = raw_id  # not yet classified -- show raw id for now
                    palette_idx = GOALKEEPER_IDX if is_goalkeeper else FALLBACK_IDX

                display_id_seen.add(display_id)
                player_pixels[display_id] = foot_pos
                ring_width = max(14, min(38, int((x2 - x1) * 0.52)))
                ring_height = max(6, min(11, int(ring_width * 0.28)))
                ring_center = (foot_pos[0], foot_pos[1] - ring_height)
                ring_specs.append((ring_center, (ring_width, ring_height), palette_idx))
                if predictor and team is not None:
                    player_position, reliable = transformer.pixel_to_pitch_checked(foot_pos)
                    if reliable:
                        players.append(PlayerState(display_id, team, "goalkeeper" if is_goalkeeper else "player", player_position))
                prefix = "GK " if is_goalkeeper else ""
                labels.append(f"{prefix}#{display_id}")
                color_lookup.append(palette_idx)
            elif is_referee:
                labels.append(f"ref #{raw_id}")
                color_lookup.append(REFEREE_IDX)
                x1, y1, x2, y2 = bbox
                foot_pos = ((x1 + x2) / 2, y2)
                ring_width = max(14, min(38, int((x2 - x1) * 0.52)))
                ring_height = max(6, min(11, int(ring_width * 0.28)))
                ring_center = (foot_pos[0], foot_pos[1] - ring_height)
                ring_specs.append((ring_center, (ring_width, ring_height), REFEREE_IDX))
            else:
                labels.append("ball")
                color_lookup.append(FALLBACK_IDX)
                x1, y1, x2, y2 = bbox
                ball_pixel = ((x1 + x2) / 2, (y1 + y2) / 2)
                ball_radius = max(4, min(10, int(max(x2 - x1, y2 - y1) * 0.8)))
                ball_render_specs.append((ball_pixel, ball_radius))
                ball_trail.append(ball_pixel)
                if predictor:
                    ball_position, reliable = transformer.pixel_to_pitch_checked(ball_pixel)
                    if not reliable:
                        ball_position = None

        color_lookup_arr = np.array(color_lookup, dtype=int) if color_lookup else np.array([], dtype=int)

        annotated = frame.copy()
        for center, axes, palette_idx in ring_specs:
            draw_tech_marker(annotated, center, axes, RING_COLORS_BGR[palette_idx])
        for ball_center, ball_radius in ball_render_specs:
            draw_glowing_ball(annotated, ball_center, ball_radius, ball_trail)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels, custom_color_lookup=color_lookup_arr)
        rendered = annotated
        if predictor:
            prediction = predictor.update(frame_idx, frame_idx / fps, players, ball_position)
            rendered = draw_prediction_panel(annotated, prediction, player_pixels)
            if events_file:
                events_file.write(json.dumps({
                    "players": [player.__dict__ for player in players],
                    "ball_position": ball_position,
                    **prediction,
                }) + "\n")

        writer.write(rendered)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames... (raw ids so far: {len(raw_id_seen)}, stabilized ids so far: {len(display_id_seen)})")

    cap.release()
    writer.release()
    if events_file:
        events_file.close()
    print(f"Done. Wrote {frame_idx} frames to {output_path}")
    print(f"Total distinct raw ByteTrack ids: {len(raw_id_seen)}")
    print(f"Total distinct stabilized display ids: {len(display_id_seen)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to input video clip")
    parser.add_argument("--output", required=True, help="Path to write annotated output video")
    parser.add_argument("--model", default=default_model_path(), help="YOLO model to use (defaults to trained best.pt when available)")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold (higher = fewer, cleaner detections)")
    parser.add_argument("--homography", help="Pitch homography JSON; enables pitch coordinates and play predictions")
    parser.add_argument("--events-output", help="JSONL file for tracks, possession, events, and suggestions")
    parser.add_argument("--attacking-direction", type=int, choices=(-1, 1), default=1, help="Team 1 attacks toward x=105 (1) or x=0 (-1)")
    parser.add_argument("--ball-conf", type=float, default=0.05, help="Ball confidence threshold")
    parser.add_argument("--ball-imgsz", type=int, default=1280, help="Inference size for the small ball")
    args = parser.parse_args()

    run(args.source, args.output, args.model, args.conf, args.homography, args.events_output, args.attacking_direction, args.ball_conf, args.ball_imgsz)