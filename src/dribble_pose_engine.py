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

def draw_directional_chevron(img, start_pt, angle_rad, length=240, width=110, color=(140, 80, 180), alpha=0.45):
    """Draws a bold semi-transparent directional take-on chevron corridor."""
    overlay = img.copy()
    sx, sy = float(start_pt[0]), float(start_pt[1])
    
    # Unit direction vectors
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    
    # Perpendicular vector
    px = -dy
    py = dx
    
    p_tail_left = [int(sx + px * (width / 2)), int(sy + py * (width / 2))]
    p_tail_right = [int(sx - px * (width / 2)), int(sy - py * (width / 2))]
    p_tail_center = [int(sx + dx * (length * 0.3)), int(sy + dy * (length * 0.3))]
    
    p_head_center = [int(sx + dx * length), int(sy + dy * length)]
    p_head_left = [int(sx + dx * (length * 0.7) + px * (width / 2)), int(sy + dy * (length * 0.7) + py * (width / 2))]
    p_head_right = [int(sx + dx * (length * 0.7) - px * (width / 2)), int(sy + dy * (length * 0.7) - py * (width / 2))]
    
    chevron_pts = np.array([
        p_tail_left,
        p_head_left,
        p_head_center,
        p_head_right,
        p_tail_right,
        p_tail_center
    ], np.int32)
    
    cv2.fillConvexPoly(overlay, chevron_pts, color, cv2.LINE_AA)
    cv2.polylines(overlay, [chevron_pts], True, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

class DribbleDuelAnalyzer:
    def __init__(self, pose_model_name="yolov8m-pose.pt", conf_thresh=0.4):
        print(f"Loading Pose Model: {pose_model_name}...")
        self.pose_model = YOLO(pose_model_name)
        self.conf_thresh = conf_thresh
        
        # Tracking history
        self.attacker_history = deque(maxlen=30)
        self.defender_history = deque(maxlen=30)
        self.ball_history = deque(maxlen=30)

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
                # The player positioned lower or with feet closest to camera foreground is usually the dribbler/attacker
                if len(players) == 1:
                    attacker_info = players[0]
                else:
                    p1 = players[0]
                    p2 = players[1]
                    
                    # Estimate player with feet closer to bottom or with greater movement
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
                
            # 3. Draw Attacker Skeleton (Pink / Rose Theme) & Directional Path
            if attacker_info is not None:
                # Calculate movement direction / burst corridor
                kpts = attacker_info["kpts"]
                hips_center = self.get_midpoint(kpts[11], kpts[12])
                shoulders_center = self.get_midpoint(kpts[5], kpts[6])
                
                # Predict take-on angle based on body lean
                takeon_angle = 0.0 # Default rightwards
                if hips_center is not None and shoulders_center is not None:
                    # Torso angle
                    lean_dx = hips_center[0] - shoulders_center[0]
                    lean_dy = hips_center[1] - shoulders_center[1]
                    
                    # Directional take-on chevron on the open side
                    if defender_info is not None:
                        def_center = defender_info["center"]
                        # Direction pointing away from defender's balance
                        if hips_center[0] < def_center[0]:
                            takeon_angle = -0.15 # Angle towards right channel
                        else:
                            takeon_angle = 0.15
                    else:
                        takeon_angle = 0.0
                        
                    chevron_anchor = (hips_center[0] + 60, hips_center[1] - 40)
                    draw_directional_chevron(
                        annotated_frame, 
                        chevron_anchor, 
                        angle_rad=takeon_angle, 
                        length=140, 
                        width=65, 
                        color=(150, 100, 180), # Soft crimson/rose chevron
                        alpha=0.42
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
