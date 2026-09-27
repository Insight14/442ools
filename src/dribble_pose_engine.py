"""
1v1 Dribble & Duel Posture Movement Engine
==========================================
Analyzes close-up 1v1 dribbles, attacker vs defender posture, joint angles,
weight distribution, body feints, and directional burst exit pathways.

- Attacker: Pink skeleton with glowing keypoint markers (#FF4181 / #FF2A85)
- Defender(s): Blue / Cyan skeleton with glowing keypoint markers (#00D2FF / #0099FF)
- Directional take-on burst chevron path (Transparent Pink / Rose)
- Dynamic posture analytics (hip lean, knee flexion, stance width)
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# COCO 17-Keypoint Index Reference:
# 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
# 5: Left Shoulder, 6: Right Shoulder
# 7: Left Elbow, 8: Right Elbow
# 9: Left Wrist, 10: Right Wrist
# 11: Left Hip, 12: Right Hip
# 13: Left Knee, 14: Right Knee
# 15: Left Ankle, 16: Right Ankle

SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_glowing_line(img, pt1, pt2, color, thickness=5, glow_color=None, glow_radius=10):
    """Draws a neon glowing anti-aliased line between two keypoints."""
    if pt1 is None or pt2 is None:
        return
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    
    # Outer glow overlay
    if glow_color is not None:
        overlay = img.copy()
        cv2.line(overlay, (x1, y1), (x2, y2), glow_color, thickness + glow_radius, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        
    # Core sharp line
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

def draw_diamond_keypoint(img, pt, color, border_color=(255, 255, 255), size=11):
    """Draws a crisp diamond keypoint marker matching broadcast tactical aesthetics."""
    if pt is None:
        return
    x, y = int(pt[0]), int(pt[1])
    pts = np.array([
        [x, y - size],
        [x + size, y],
        [x, y + size],
        [x - size, y]
    ], np.int32)
    
    # Fill
    cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
    # Border
    cv2.polylines(img, [pts], True, border_color, 2, cv2.LINE_AA)

def draw_ground_joystick_ring(img, center_pt, angle_rad, radius=65, thickness=4, color=(0, 240, 255), arrow_color=(0, 255, 120), opacity=0.85, speed_scale=1.0, is_wrong_direction=False):
    """
    Draws an EA FC / FIFA style tactical ground ellipse indicator around player feet.
    - Arrow grows/shrinks dynamically based on velocity & displacement speed.
    - Opacity scales dynamically with speed.
    - Arrow turns red if defender is wrong-footed / moving in opposing stance to the dribbler.
    """
    if center_pt is None:
        return
    
    cx, cy = int(center_pt[0]), int(center_pt[1])
    overlay = img.copy()
    
    # Ground perspective ellipse
    axes = (int(radius * 1.1), int(radius * 0.42))
    
    # 1. Base ring arc with glow
    ring_glow_color = (0, 0, 0)
    cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, ring_glow_color, thickness + 6, cv2.LINE_AA)
    cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, color, thickness, cv2.LINE_AA)
    
    # 2. Dynamic Directional Pointer Arrow (responding to joystick/posture shift)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    
    # Dynamic arrow growth / shrink based on velocity & speed_scale
    clamped_speed = max(0.6, min(2.4, speed_scale))
    arrow_len = radius * (0.85 + 0.65 * clamped_speed)
    arrow_width = radius * (0.28 + 0.15 * clamped_speed)
    
    tip_x = int(cx + dx * arrow_len)
    tip_y = int(cy + dy * (arrow_len * 0.42))
    
    base_left_x = int(cx + (dx * radius * 0.6) - (dy * arrow_width))
    base_left_y = int(cy + (dy * (radius * 0.6) * 0.42) + (dx * arrow_width * 0.5))
    
    base_right_x = int(cx + (dx * radius * 0.6) + (dy * arrow_width))
    base_right_y = int(cy + (dy * (radius * 0.6) * 0.42) - (dx * arrow_width * 0.5))
    
    notch_x = int(cx + dx * radius * 0.72)
    notch_y = int(cy + dy * radius * 0.72 * 0.42)
    
    arrow_pts = np.array([
        [tip_x, tip_y],
        [base_left_x, base_left_y],
        [notch_x, notch_y],
        [base_right_x, base_right_y]
    ], np.int32)
    
    # Color logic: Crimson Red if wrong-footed / wrong direction, otherwise designated arrow_color (Green/Lime)
    effective_arrow_color = (40, 50, 255) if is_wrong_direction else arrow_color
    border_color = (255, 200, 200) if is_wrong_direction else (255, 255, 255)
    
    # Fill arrow
    cv2.fillConvexPoly(overlay, arrow_pts, effective_arrow_color, cv2.LINE_AA)
    cv2.polylines(overlay, [arrow_pts], True, border_color, 2, cv2.LINE_AA)
    
    # Blend with dynamic opacity based on speed & displacement
    clamped_opacity = max(0.35, min(0.95, opacity))
    cv2.addWeighted(overlay, clamped_opacity, img, 1.0 - clamped_opacity, 0, img)

class DribbleDuelAnalyzer:
    def __init__(self, pose_model_name="yolov8m-pose.pt", conf_thresh=0.4):
        print(f"Loading Pose Model: {pose_model_name}...")
        self.pose_model = YOLO(pose_model_name)
        self.conf_thresh = conf_thresh
        
        # Position and Angle Tracking for Speed & Velocity calculation
        self.attacker_prev_pos = None
        self.defender_prev_pos = None
        self.attacker_smoothed_angle = None
        self.defender_smoothed_angle = None
        self.attacker_speed = 0.5
        self.defender_speed = 0.5

    def compute_player_vector(self, kpts, prev_pos=None, is_defender=False, opponent_center=None):
        """
        Computes the instantaneous velocity + posture direction vector for any player:
        1. Torso Lean (shoulders -> hips)
        2. Displacement velocity vector from previous frame
        3. Stance / foot push-off vector
        """
        shoulders = self.get_midpoint(kpts[5], kpts[6])
        hips = self.get_midpoint(kpts[11], kpts[12])
        left_ankle = kpts[15]
        right_ankle = kpts[16]
        
        vec_x, vec_y = 0.0, 0.0
        weight_sum = 0.0
        speed_displacement = 0.0
        
        # A. Center Position & Frame Displacement
        current_center = hips if hips is not None else shoulders
        if current_center is not None and prev_pos is not None:
            disp_x = current_center[0] - prev_pos[0]
            disp_y = current_center[1] - prev_pos[1]
            speed_displacement = math.hypot(disp_x, disp_y)
            # Add velocity to vector
            vec_x += disp_x * 4.0
            vec_y += disp_y * 4.0
            weight_sum += 4.0

        # B. Torso Lean (Joystick Angle Shift before movement)
        if shoulders is not None and hips is not None:
            torso_dx = shoulders[0] - hips[0]
            torso_dy = shoulders[1] - hips[1]
            vec_x += torso_dx * 3.0
            vec_y += torso_dy * 1.5
            weight_sum += 3.0
            
        # C. Stance / Footing
        if left_ankle[2] > 0.3 and right_ankle[2] > 0.3 and hips is not None:
            feet_mid_x = (left_ankle[0] + right_ankle[0]) / 2.0
            hip_offset_x = hips[0] - feet_mid_x
            vec_x += hip_offset_x * 2.5
            weight_sum += 2.5

        if weight_sum > 0 and (abs(vec_x) > 0.1 or abs(vec_y) > 0.1):
            target_angle = math.atan2(vec_y * 0.6, vec_x)
        else:
            target_angle = 0.0 if not is_defender else math.pi

        # Dynamic Speed / Intensity Score (combining displacement & posture tilt)
        speed_score = min(2.2, max(0.5, (speed_displacement / 12.0) + (abs(vec_x) / 30.0)))
        opacity_score = min(0.95, max(0.35, 0.45 + (speed_score * 0.28)))
        
        return target_angle, opacity_score, speed_score, current_center

    def is_defender_wrong_footed(self, att_angle, def_angle, att_center, def_center):
        """
        Determines if the defender is wrong-footed / moving in the wrong direction relative to the dribbler.
        If the dribbler is bursting right/forward and defender is leaning or stepping in the opposite direction,
        return True (Turns Red).
        """
        if att_angle is None or def_angle is None:
            return False
            
        att_dx = math.cos(att_angle)
        def_dx = math.cos(def_angle)
        
        # When defender is facing the attacker to jockey, their response vector should follow the dribbler's lateral cut
        # If the attacker cuts right (att_dx > 0) but defender commits left (def_dx < -0.2), defender is wrong-footed
        dot_product = math.cos(att_angle - def_angle)
        
        # If dot product indicates opposing lateral commitments or severe mismatch (> 100 degrees mismatch in lateral coverage)
        lateral_mismatch = (att_dx * def_dx < -0.15)
        
        return lateral_mismatch or (dot_product < -0.35)

    def process_video(self, video_path, output_path, max_frames=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        print(f"Analyzing 1v1 duel in {video_path.name} ({width}x{height} @ {fps:.1f} fps)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break
                
            # 1. Run YOLO-Pose Inference
            results = self.pose_model(frame, conf=self.conf_thresh, verbose=False)[0]
            
            # Extract keypoints and boxes
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            keypoints_data = results.keypoints.data.cpu().numpy() if results.keypoints is not None else []
            
            # Sort players by box area (largest players are the foreground 1v1 duelists)
            players = []
            for i in range(len(boxes)):
                box = boxes[i]
                kpts = keypoints_data[i] # Shape (17, 3) [x, y, conf]
                area = (box[2] - box[0]) * (box[3] - box[1])
                center_x = (box[0] + box[2]) / 2.0
                center_y = (box[1] + box[3]) / 2.0
                players.append({
                    "box": box,
                    "kpts": kpts,
                    "area": area,
                    "center": (center_x, center_y),
                    "conf": float(results.boxes.conf[i].cpu().numpy())
                })
                
            players = sorted(players, key=lambda p: p["area"], reverse=True)
            
            # Take top 2 foreground players for the 1v1 duel
            attacker_info = None
            defender_info = None
            
            if len(players) >= 1:
                if len(players) == 1:
                    attacker_info = players[0]
                else:
                    p1 = players[0]
                    p2 = players[1]
                    
                    p1_bottom = p1["box"][3]
                    p2_bottom = p2["box"][3]
                    
                    if p1_bottom >= p2_bottom:
                        attacker_info = p1
                        defender_info = p2
                    else:
                        attacker_info = p2
                        defender_info = p1
            
            annotated_frame = frame.copy()
            
            att_angle, att_opacity, att_speed = 0.0, 0.5, 0.5
            att_feet_center = None
            def_angle, def_opacity, def_speed = math.pi, 0.5, 0.5
            def_feet_center = None
            
            # 1. Compute Attacker Vector & Feet Center
            if attacker_info is not None:
                kpts = attacker_info["kpts"]
                left_ankle = kpts[15]
                right_ankle = kpts[16]
                if left_ankle[2] > 0.25 and right_ankle[2] > 0.25:
                    att_feet_center = ((left_ankle[0] + right_ankle[0]) / 2.0, max(left_ankle[1], right_ankle[1]) + 8)
                else:
                    att_feet_center = (attacker_info["center"][0], attacker_info["box"][3] - 8)
                    
                target_ang, opac, spd, curr_pos = self.compute_player_vector(
                    kpts, 
                    prev_pos=self.attacker_prev_pos, 
                    is_defender=False
                )
                self.attacker_prev_pos = curr_pos
                
                # Smooth angle
                if self.attacker_smoothed_angle is None:
                    self.attacker_smoothed_angle = target_ang
                else:
                    diff = (target_ang - self.attacker_smoothed_angle + math.pi) % (2 * math.pi) - math.pi
                    self.attacker_smoothed_angle += diff * 0.45
                    
                att_angle = self.attacker_smoothed_angle
                att_opacity = opac
                att_speed = spd
                
            # 2. Compute Defender Vector & Feet Center
            if defender_info is not None:
                kpts = defender_info["kpts"]
                left_ankle = kpts[15]
                right_ankle = kpts[16]
                if left_ankle[2] > 0.25 and right_ankle[2] > 0.25:
                    def_feet_center = ((left_ankle[0] + right_ankle[0]) / 2.0, max(left_ankle[1], right_ankle[1]) + 8)
                else:
                    def_feet_center = (defender_info["center"][0], defender_info["box"][3] - 8)
                    
                target_ang, opac, spd, curr_pos = self.compute_player_vector(
                    kpts, 
                    prev_pos=self.defender_prev_pos, 
                    is_defender=True
                )
                self.defender_prev_pos = curr_pos
                
                # Smooth defender angle
                if self.defender_smoothed_angle is None:
                    self.defender_smoothed_angle = target_ang
                else:
                    diff = (target_ang - self.defender_smoothed_angle + math.pi) % (2 * math.pi) - math.pi
                    self.defender_smoothed_angle += diff * 0.45
                    
                def_angle = self.defender_smoothed_angle
                def_opacity = opac
                def_speed = spd

            # 3. Check if defender is wrong-footed relative to dribbler
            is_def_wrong_footed = False
            if attacker_info is not None and defender_info is not None:
                is_def_wrong_footed = self.is_defender_wrong_footed(
                    att_angle, 
                    def_angle, 
                    attacker_info["center"], 
                    defender_info["center"]
                )

            # 4. Draw Defender Ground Ring & Skeleton
            if defender_info is not None and def_feet_center is not None:
                # Defender Ground Indicator: Cyan Ring + Green Arrow (or RED if wrong-footed)
                draw_ground_joystick_ring(
                    annotated_frame,
                    center_pt=def_feet_center,
                    angle_rad=def_angle,
                    radius=int(min(width, height) * 0.052),
                    thickness=3,
                    color=(255, 190, 0),        # Cyan Ring
                    arrow_color=(0, 255, 120),    # Green (or Red if wrong footed)
                    opacity=def_opacity,
                    speed_scale=def_speed,
                    is_wrong_direction=is_def_wrong_footed
                )
                
                self.render_player_skeleton(
                    annotated_frame, 
                    defender_info["kpts"],
                    bone_color=(255, 180, 0),       # Cyan-Blue BGR
                    glow_color=(255, 120, 0),
                    joint_color=(255, 220, 50),
                    role="DEFENDER"
                )
                
            # 5. Draw Attacker Ground Ring & Skeleton
            if attacker_info is not None and att_feet_center is not None:
                # Attacker Ground Indicator: Neon Pink Ring + Electric Lime Arrow (grows with burst speed)
                draw_ground_joystick_ring(
                    annotated_frame,
                    center_pt=att_feet_center,
                    angle_rad=att_angle,
                    radius=int(min(width, height) * 0.055),
                    thickness=3,
                    color=(220, 50, 255),       # Neon Pink Arc
                    arrow_color=(0, 255, 140),   # Electric Lime Arrow
                    opacity=att_opacity,
                    speed_scale=att_speed,
                    is_wrong_direction=False
                )

                self.render_player_skeleton(
                    annotated_frame, 
                    attacker_info["kpts"],
                    bone_color=(190, 40, 255),      # Neon Pink BGR
                    glow_color=(140, 20, 230),
                    joint_color=(220, 100, 255),
                    role="ATTACKER"
                )
                
            # 6. Render Tactical Telemetry Overlay
            self.render_hud_overlay(annotated_frame, attacker_info, defender_info, frame_idx, total_frames, is_def_wrong_footed)
            
            out.write(annotated_frame)
            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}/{total_frames} ({frame_idx*100/total_frames:.1f}%)")
                
        cap.release()
        out.release()
        print(f"1v1 Duel Analysis Complete -> {output_path}")
        return output_path

    def get_midpoint(self, pt1, pt2, min_conf=0.25):
        if pt1[2] < min_conf or pt2[2] < min_conf:
            return None
        return ((pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0)

    def render_player_skeleton(self, img, kpts, bone_color, glow_color, joint_color, role="ATTACKER"):
        """Draws glowing skeleton connections and diamond joint vertices."""
        # 1. Bones
        for i1, i2 in SKELETON_CONNECTIONS:
            p1 = kpts[i1]
            p2 = kpts[i2]
            if p1[2] >= self.conf_thresh and p2[2] >= self.conf_thresh:
                draw_glowing_line(
                    img, 
                    (p1[0], p1[1]), 
                    (p2[0], p2[1]), 
                    color=bone_color, 
                    thickness=3, 
                    glow_color=glow_color, 
                    glow_radius=6
                )
                
        # 2. Keypoints (Diamond markers)
        for i in range(len(kpts)):
            pt = kpts[i]
            if pt[2] >= self.conf_thresh:
                # Larger markers for primary joints (head, shoulders, hips, knees, feet)
                size = 7 if i in [0, 5, 6, 11, 12, 13, 14, 15, 16] else 5
                draw_diamond_keypoint(
                    img, 
                    (pt[0], pt[1]), 
                    color=joint_color, 
                    border_color=(255, 255, 255), 
                    size=size
                )

    def render_hud_overlay(self, img, attacker, defender, frame_idx, total_frames, is_wrong_footed=False):
        """Draws dynamic sports intelligence telemetry (angles, stance width, duel balance)."""
        h, w = img.shape[:2]
        
        # Top Left Badge
        badge_w, badge_h = 360, 75
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (20 + badge_w, 20 + badge_h), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (20, 20), (20 + badge_w, 20 + badge_h), (0, 240, 255), 1)
        
        cv2.putText(img, "META-VISION // 1v1 DUEL TRACKER", (32, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)
        
        if is_wrong_footed:
            status_text = "DEFENDER: WRONG-FOOTED [STANCE MISMATCH]"
            status_color = (40, 60, 255) # Red
        else:
            status_text = "DEFENDER: BALANCED JOCKEY [TRACKING DRIBBLER]"
            status_color = (0, 255, 120) # Green
            
        cv2.putText(img, status_text, (32, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)
        
        if attacker is not None:
            cv2.putText(img, "ATTACKER: INSTANT POSTURE VECTOR ACTIVE", (32, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 120, 240), 1, cv2.LINE_AA)

    def calculate_angle(self, a, b, c):
        """Returns the angle at joint b in degrees."""
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1v1 Dribble & Duel Posture Movement Engine")
    parser.add_argument("--video", type=str, default="input_videos/nicowill_stones_1v1.mov", help="Input 1v1 video path")
    parser.add_argument("--output", type=str, default="output_videos/nicowill_stones_1v1_dribble_analyzed.mp4", help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()
    
    analyzer = DribbleDuelAnalyzer()
    analyzer.process_video(Path(args.video), Path(args.output), max_frames=args.max_frames)
