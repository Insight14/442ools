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

def draw_ground_joystick_ring(img, center_pt, angle_rad, radius=65, thickness=4, color=(0, 240, 255), arrow_color=(0, 255, 120), opacity=0.85, speed_intensity=1.0):
    """
    Draws an EA FC / FIFA style tactical ground ellipse indicator around player feet
    with an instantaneous directional arrow pointer responding right as weight shifts.
    """
    if center_pt is None:
        return
    
    cx, cy = int(center_pt[0]), int(center_pt[1])
    overlay = img.copy()
    
    # Ground perspective ellipse
    axes = (int(radius * 1.1), int(radius * 0.42))
    
    # 1. Base ring arc with glow
    cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, (0, 0, 0), thickness + 6, cv2.LINE_AA)
    cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, color, thickness, cv2.LINE_AA)
    
    # 2. Dynamic Directional Pointer Arrow (responding to joystick/posture shift)
    # Convert angle to ellipse perimeter coordinate
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    
    arrow_len = radius * (1.2 + 0.4 * speed_intensity)
    tip_x = int(cx + dx * arrow_len)
    tip_y = int(cy + dy * (arrow_len * 0.42))
    
    base_left_x = int(cx + (dx * radius * 0.6) - (dy * radius * 0.3))
    base_left_y = int(cy + (dy * (radius * 0.6) * 0.42) + (dx * radius * 0.15))
    
    base_right_x = int(cx + (dx * radius * 0.6) + (dy * radius * 0.3))
    base_right_y = int(cy + (dy * (radius * 0.6) * 0.42) - (dx * radius * 0.15))
    
    arrow_pts = np.array([
        [tip_x, tip_y],
        [base_left_x, base_left_y],
        [int(cx + dx * radius * 0.75), int(cy + dy * radius * 0.75 * 0.42)],
        [base_right_x, base_right_y]
    ], np.int32)
    
    # Fill arrow with vivid responsive neon color
    cv2.fillConvexPoly(overlay, arrow_pts, arrow_color, cv2.LINE_AA)
    cv2.polylines(overlay, [arrow_pts], True, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Blend with dynamic opacity based on conviction/speed
    clamped_opacity = max(0.2, min(0.95, opacity))
    cv2.addWeighted(overlay, clamped_opacity, img, 1.0 - clamped_opacity, 0, img)

class DribbleDuelAnalyzer:
    def __init__(self, pose_model_name="yolov8m-pose.pt", conf_thresh=0.4):
        print(f"Loading Pose Model: {pose_model_name}...")
        self.pose_model = YOLO(pose_model_name)
        self.conf_thresh = conf_thresh
        
        # Tracking history for predictive velocity & instantaneous posture vector
        self.attacker_history = deque(maxlen=15)
        self.defender_history = deque(maxlen=15)
        self.smoothed_angle = None
        self.smoothed_opacity = 0.5

    def compute_posture_direction(self, attacker_kpts, defender_center=None):
        """
        Computes the instantaneous intended direction vector using:
        1. Torso Lean (shoulders -> hips) - detects body drop/feint before step
        2. Knee flexion & lead foot plant (ankles)
        3. Head/facing vector
        4. Defender stance leverage / open exit corridor
        """
        # Torso vector
        shoulders = self.get_midpoint(attacker_kpts[5], attacker_kpts[6])
        hips = self.get_midpoint(attacker_kpts[11], attacker_kpts[12])
        left_ankle = attacker_kpts[15]
        right_ankle = attacker_kpts[16]
        
        vec_x, vec_y = 0.0, 0.0
        weight_sum = 0.0
        intensity = 0.5
        
        # A. Torso / Shoulder Drop Lean (Highest predictive indicator - joystick tilt)
        if shoulders is not None and hips is not None:
            torso_dx = shoulders[0] - hips[0] # Positive = leaning right
            torso_dy = shoulders[1] - hips[1]
            vec_x += torso_dx * 3.5
            vec_y += torso_dy * 1.2
            weight_sum += 3.5
            intensity = min(1.5, abs(torso_dx) / 35.0 + 0.4)
            
        # B. Plant Foot vs Push-Off Foot Vector
        if left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
            # Distance between feet
            step_dx = right_ankle[0] - left_ankle[0]
            # If hips are shifted forward relative to feet
            if hips is not None:
                feet_mid_x = (left_ankle[0] + right_ankle[0]) / 2.0
                hip_offset_x = hips[0] - feet_mid_x
                vec_x += hip_offset_x * 2.0
                weight_sum += 2.0
                
        # C. Defender Stance Opposition Vector
        if defender_center is not None and hips is not None:
            # Space channel away from defender's core
            def_rel_x = hips[0] - defender_center[0]
            vec_x += np.sign(def_rel_x) * 15.0
            weight_sum += 1.0

        if weight_sum > 0:
            target_angle = math.atan2(vec_y * 0.5, vec_x)
        else:
            target_angle = 0.0 # Default forward/right

        # Smooth angle transitions to eliminate jitter while keeping instant response
        if self.smoothed_angle is None:
            self.smoothed_angle = target_angle
        else:
            # Angular interpolation
            diff = (target_angle - self.smoothed_angle + math.pi) % (2 * math.pi) - math.pi
            self.smoothed_angle += diff * 0.45
            
        target_opacity = min(0.95, max(0.3, intensity * 0.85))
        self.smoothed_opacity = self.smoothed_opacity * 0.6 + target_opacity * 0.4
        
        return self.smoothed_angle, self.smoothed_opacity, intensity

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
            
            # 2. Draw Defender Skeleton (Blue / Cyan Theme)
            if defender_info is not None:
                self.render_player_skeleton(
                    annotated_frame, 
                    defender_info["kpts"],
                    bone_color=(255, 180, 0),       # Cyan-Blue BGR
                    glow_color=(255, 120, 0),
                    joint_color=(255, 220, 50),
                    role="DEFENDER"
                )
                
            # 3. Draw Attacker Skeleton (Pink / Rose Theme) & FIFA-Style Directional Ring
            if attacker_info is not None:
                kpts = attacker_info["kpts"]
                
                # Ground center anchor (midpoint of feet or bottom of bounding box)
                left_ankle = kpts[15]
                right_ankle = kpts[16]
                if left_ankle[2] > 0.25 and right_ankle[2] > 0.25:
                    feet_center = ((left_ankle[0] + right_ankle[0]) / 2.0, max(left_ankle[1], right_ankle[1]) + 8)
                else:
                    feet_center = (attacker_info["center"][0], attacker_info["box"][3] - 10)
                
                def_center = defender_info["center"] if defender_info is not None else None
                angle, opacity, intensity = self.compute_posture_direction(kpts, def_center)
                
                # Draw FIFA/FC Mobile style responsive ground ring + joystick vector arrow
                draw_ground_joystick_ring(
                    annotated_frame,
                    center_pt=feet_center,
                    angle_rad=angle,
                    radius=int(min(width, height) * 0.055),
                    thickness=3,
                    color=(220, 50, 255),       # Neon Pink Arc
                    arrow_color=(0, 255, 140),   # Electric Lime Arrow (instant feedback)
                    opacity=opacity,
                    speed_intensity=intensity
                )

                self.render_player_skeleton(
                    annotated_frame, 
                    attacker_info["kpts"],
                    bone_color=(190, 40, 255),      # Neon Pink BGR
                    glow_color=(140, 20, 230),
                    joint_color=(220, 100, 255),
                    role="ATTACKER"
                )
                
            # 4. Render Tactical Telemetry Overlay
            self.render_hud_overlay(annotated_frame, attacker_info, defender_info, frame_idx, total_frames)
            
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

    def render_hud_overlay(self, img, attacker, defender, frame_idx, total_frames):
        """Draws dynamic sports intelligence telemetry (angles, stance width, duel balance)."""
        h, w = img.shape[:2]
        
        # Top Left Badge
        badge_w, badge_h = 320, 68
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (20 + badge_w, 20 + badge_h), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (20, 20), (20 + badge_w, 20 + badge_h), (0, 240, 255), 1)
        
        cv2.putText(img, "META-VISION // 1v1 DUEL TRACKER", (32, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)
        
        # Calculate dynamic angles if attacker exists
        if attacker is not None:
            kpts = attacker["kpts"]
            # Knee bend angle
            if kpts[11][2] > 0.3 and kpts[13][2] > 0.3 and kpts[15][2] > 0.3:
                knee_angle = self.calculate_angle(kpts[11], kpts[13], kpts[15])
                text_stat = f"ATTACKER KNEE FLEX: {int(knee_angle)} deg | POSTURE: LOW AGILITY"
            else:
                text_stat = "ATTACKER: STEP-OVER ACCELERATION"
            cv2.putText(img, text_stat, (32, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 100, 220), 1, cv2.LINE_AA)

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
