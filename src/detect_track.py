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
import math
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
    """Draw a vibrant glowing ball core with an organic tapered fluid comet motion trail."""
    if not trail:
        ball_center = (int(center[0]), int(center[1]))
        cv2.circle(frame, ball_center, max(3, radius), (255, 255, 255), -1, cv2.LINE_AA)
        return

    trail_list = list(trail)
    n_points = len(trail_list)
    glow_color_bgr = (40, 160, 255)  # warm electric orange-cyan halo

    # 1. Multi-pass smooth organic tapered trail
    if n_points >= 2:
        overlay = np.zeros_like(frame)
        for i in range(n_points - 1):
            p1 = trail_list[i]
            p2 = trail_list[i + 1]
            t = (i + 1) / float(n_points)  # 0 at oldest, 1 at ball head
            
            # Organic non-linear teardrop taper: starts ultra-thin, blossoms into rounded head
            thickness = max(1, int((t ** 1.8) * radius * 1.5))
            alpha_step = max(0.08, min(0.85, t ** 1.4))

            # Outer soft glow line
            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), glow_color_bgr, thickness + 4, cv2.LINE_AA)
            # Inner bright filament
            cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255), max(1, thickness // 2), cv2.LINE_AA)

        # Soft Gaussian blur on the trail layer for natural luminescence
        blurred_trail = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(2.0, radius * 1.2))
        cv2.addWeighted(blurred_trail, 0.45, frame, 0.55, 0, frame)

    # 2. Ball Core with pristine multi-ring neon halo
    ball_center = (int(center[0]), int(center[1]))
    cv2.circle(frame, ball_center, max(4, radius + 3), glow_color_bgr, 1, cv2.LINE_AA)
    cv2.circle(frame, ball_center, max(2, radius), (240, 250, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, ball_center, max(1, radius // 2), (255, 255, 255), -1, cv2.LINE_AA)


def draw_tactical_vector(canvas, start_pt, end_pt, color_bgr, alpha=0.9, subtype="ground", prob=None, label=""):
    """Draw tactical trajectory with pass-type styling (curves for crosses, dashes/tags for through passes)."""
    x1, y1 = int(start_pt[0]), int(start_pt[1])
    x2, y2 = int(end_pt[0]), int(end_pt[1])

    glow_canvas = canvas.copy()

    if subtype in ("lob", "cross"):
        # Draw arched curve trajectory for aerial lofted balls
        mid_x = (x1 + x2) // 2
        mid_y = min(y1, y2) - max(25, int(math.dist(start_pt, end_pt) * 0.18))
        curve_pts = []
        for t in np.linspace(0, 1, 24):
            bx = int((1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2)
            by = int((1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2)
            curve_pts.append((bx, by))
        
        for i in range(len(curve_pts) - 1):
            cv2.line(glow_canvas, curve_pts[i], curve_pts[i+1], color_bgr, 4, cv2.LINE_AA)
            cv2.line(glow_canvas, curve_pts[i], curve_pts[i+1], (255, 255, 255), 1, cv2.LINE_AA)

        mid_pt = curve_pts[len(curve_pts)//2]
    else:
        # Ground or Through Ball trajectory
        if subtype == "through_ball":
            # Animated style line
            cv2.line(glow_canvas, (x1, y1), (x2, y2), color_bgr, 5, cv2.LINE_AA)
            cv2.line(glow_canvas, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.line(glow_canvas, (x1, y1), (x2, y2), color_bgr, 4, cv2.LINE_AA)
            cv2.line(glow_canvas, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        
        mid_pt = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Target indicator ring
    cv2.circle(glow_canvas, (x2, y2), 7, color_bgr, 2, cv2.LINE_AA)
    cv2.circle(glow_canvas, (x2, y2), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Render floating Success Probability Badge (e.g. 74% / 88%)
    if prob is not None:
        tag_text = f"{prob}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.44
        (tw, th), _ = cv2.getTextSize(tag_text, font, scale, 1)
        bx, by = mid_pt[0] - tw // 2, mid_pt[1] - 8
        cv2.rectangle(glow_canvas, (bx - 5, by - th - 4), (bx + tw + 5, by + 4), (12, 22, 34), -1)
        cv2.rectangle(glow_canvas, (bx - 5, by - th - 4), (bx + tw + 5, by + 4), color_bgr, 1)
        prob_col = (80, 240, 120) if prob >= 65 else ((60, 200, 255) if prob >= 40 else (70, 70, 255))
        cv2.putText(glow_canvas, tag_text, (bx, by - 1), font, scale, prob_col, 1, cv2.LINE_AA)

    cv2.addWeighted(glow_canvas, alpha, canvas, 1.0 - alpha, 0, canvas)


def draw_space_zone_shading(canvas, space_zones, transformer):
    """Highlight high-threat exploitable pitch zones with glowing semi-transparent emerald/cyan polygons."""
    if not space_zones or not transformer:
        return

    overlay = canvas.copy()
    for zone in space_zones:
        cx, cy = zone["center"]
        r = zone["radius_m"]
        
        # Approximate circle with 16 polygon vertices transformed to camera pixels
        poly_pts = []
        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            pixel_pt, rel = transformer.pitch_to_pixel_checked((px, py))
            if rel and 0 <= pixel_pt[0] < canvas.shape[1] - PREDICTION_PANEL_WIDTH and 0 <= pixel_pt[1] < canvas.shape[0]:
                poly_pts.append([int(pixel_pt[0]), int(pixel_pt[1])])

        if len(poly_pts) >= 4:
            pts_arr = np.array(poly_pts, dtype=np.int32)
            # Emerald space tint
            cv2.fillPoly(overlay, [pts_arr], (70, 210, 110))
            cv2.polylines(overlay, [pts_arr], True, (120, 255, 160), 2, cv2.LINE_AA)
            
            # Label
            center_px, rel_c = transformer.pitch_to_pixel_checked((cx, cy))
            if rel_c:
                cpx, cpy = int(center_px[0]), int(center_px[1])
                cv2.putText(overlay, "EXPLOIT SPACE", (cpx - 44, cpy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 255, 240), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.26, canvas, 0.74, 0, canvas)


def draw_prediction_panel(frame, prediction, player_pixels, transformer=None):
    """Add a tactical sidebar, projected pass/shot suggestion paths, and space zone highlights."""
    panel = np.zeros((frame.shape[0], PREDICTION_PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (8, 16, 26)
    canvas = np.concatenate([frame, panel], axis=1)
    panel_x = frame.shape[1]
    cyan = (255, 210, 80)
    white = (240, 248, 255)
    muted = (135, 155, 175)
    green = (80, 230, 110)
    faded_yellow = (70, 210, 240)
    electric_blue = (255, 160, 40)
    purple_accent = (220, 120, 255)

    # 1. Draw Space Zone Highlight Shading on Pitch
    space_zones = prediction.get("space_zones", [])
    if space_zones and transformer:
        draw_space_zone_shading(canvas, space_zones, transformer)

    cv2.line(canvas, (panel_x, 0), (panel_x, canvas.shape[0]), (35, 65, 85), 2)
    cv2.putText(canvas, "LIVE TACTICAL INTELLIGENCE", (panel_x + 22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cyan, 2, cv2.LINE_AA)
    cv2.putText(canvas, "442OOLS / META-VISION", (panel_x + 22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, muted, 1, cv2.LINE_AA)
    cv2.line(canvas, (panel_x + 22, 88), (panel_x + PREDICTION_PANEL_WIDTH - 22, 88), (35, 65, 85), 1)

    play = prediction.get("play_label", "unknown").replace("_", " ").upper()
    quality = prediction.get("data_quality", "unknown").replace("_", " ").upper()
    possession = prediction.get("possession_track_id")
    possession_text = f"TRACK #{possession}" if possession is not None else "UNCONFIRMED"
    
    cv2.putText(canvas, "TACTICAL PHASE", (panel_x + 22, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.40, muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, play, (panel_x + 22, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.76, white if quality == "USABLE" else (70, 150, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"POSSESSION  {possession_text}", (panel_x + 22, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.46, white, 1, cv2.LINE_AA)

    suggestions = prediction.get("suggestions", [])
    
    # Render Overlay Trajectory Lines on Pitch
    for sugg in suggestions:
        s_type = sugg.get("type")
        subtype = sugg.get("subtype", "ground")
        prob = sugg.get("success_prob")
        from_id = sugg.get("from_track_id")
        source = player_pixels.get(from_id)
        if source is None:
            continue

        target_pitch = sugg.get("target_pitch_pos")
        target_pixel = None
        if subtype == "through_ball" and target_pitch and transformer:
            px, rel = transformer.pitch_to_pixel_checked(target_pitch)
            if rel:
                target_pixel = px
        if target_pixel is None:
            target_pixel = player_pixels.get(sugg.get("to_track_id"))

        if target_pixel is None and s_type != "shot":
            continue

        if "pass" in s_type:
            tier = sugg.get("tier", "primary")
            if subtype == "through_ball":
                col = (40, 235, 255) # Electric Cyan / Gold
                draw_tactical_vector(canvas, source, target_pixel, color_bgr=col, alpha=0.94, subtype=subtype, prob=prob, label="THROUGH")
            elif subtype in ("cross", "lob"):
                col = (235, 120, 255) # Electric Pink/Magenta
                draw_tactical_vector(canvas, source, target_pixel, color_bgr=col, alpha=0.90, subtype=subtype, prob=prob, label="CROSS")
            elif tier == "primary":
                col = (60, 240, 100) # Emerald Neon
                draw_tactical_vector(canvas, source, target_pixel, color_bgr=col, alpha=0.92, subtype="ground", prob=prob, label="BEST PASS")
            else:
                col = (50, 195, 230) # Faded Gold
                draw_tactical_vector(canvas, source, target_pixel, color_bgr=col, alpha=0.45, subtype="ground", prob=prob, label="BACKUP")

        elif s_type == "shot":
            xg = sugg.get("score", 0.3)
            shot_alpha = float(clamp(0.40 + xg * 0.85, 0.40, 0.95))
            goal_x = int(canvas.shape[1] - PREDICTION_PANEL_WIDTH - 20) if source[0] > 100 else 20
            goal_pt = (goal_x, int(canvas.shape[0] * 0.5))
            draw_tactical_vector(canvas, source, goal_pt, color_bgr=(255, 160, 40), alpha=shot_alpha, subtype="shot", prob=int(xg*100), label="SHOT")

    # Render Sidebar Suggestion Cards
    cv2.putText(canvas, "RECOMMENDED PLAYS", (panel_x + 22, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cyan, 1, cv2.LINE_AA)
    if not suggestions:
        cv2.putText(canvas, "Scanning passing options...", (panel_x + 22, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.48, muted, 1, cv2.LINE_AA)
    else:
        y_offset = 265
        for s in suggestions[:3]:
            s_type = s.get("type", "").upper()
            subtype = s.get("subtype", "ground")
            prob = s.get("success_prob", 50)
            tier = s.get("tier", "")
            
            if subtype == "through_ball":
                title_col = cyan
                tag = f"[THROUGH BALL ({prob}%)]"
            elif subtype in ("cross", "lob"):
                title_col = purple_accent
                tag = f"[CROSS / LOB ({prob}%)]"
            elif tier == "primary":
                title_col = green
                tag = f"[#1 BEST PASS ({prob}%)]"
            elif "PASS" in s_type:
                title_col = faded_yellow
                tag = f"[BACKUP PASS ({prob}%)]"
            else:
                title_col = electric_blue
                tag = f"[SHOT THREAT ({prob}% xG)]"

            cv2.putText(canvas, f"{tag}  #{s.get('from_track_id','?')} -> #{s.get('to_track_id','GOAL')}", (panel_x + 22, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, title_col, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"xT: +{s.get('expected_threat', 0)}  |  Dist: {s.get('distance_m', s.get('goal_distance_m', 0))}m", (panel_x + 22, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, white, 1, cv2.LINE_AA)
            
            desc = s.get("desc", "")
            if desc:
                cv2.putText(canvas, desc[:36], (panel_x + 22, y_offset + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, muted, 1, cv2.LINE_AA)
            y_offset += 72

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
            rendered = draw_prediction_panel(annotated, prediction, player_pixels, transformer=transformer)
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