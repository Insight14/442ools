"""Generate a high-production cinematic intro video explaining Meta-Vision / 442OOLS.

Features:
- 1080p 60fps high-fidelity rendering
- Cybernetic glowing grid, particle constellations, dynamic radar circles
- Title cards, architecture highlights, CLI commands, and feature breakdown
- Smooth alpha crossfades, sliding badges, and glowing HUD telemetries
- Ready for video editing (overlays, transitions, demo cuts)
"""

import math
import os
import cv2
import numpy as np

WIDTH = 1920
HEIGHT = 1080
FPS = 30
DURATION_SEC = 14
TOTAL_FRAMES = FPS * DURATION_SEC

os.makedirs("output_videos", exist_ok=True)
OUT_PATH = "output_videos/project_intro_demo.mp4"

# Color Palette (BGR)
BG_DARK = (10, 14, 20)
CYAN_GLOW = (255, 235, 0)
NEON_GREEN = (100, 240, 0)
HOT_PINK = (220, 40, 255)
GOLD = (0, 215, 255)
WHITE = (245, 250, 255)
MUTED = (160, 140, 120)
BORDER_COL = (60, 45, 30)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

def draw_background_grid(frame, t):
    """Draw animated perspective grid with pulsing scanline."""
    h, w = frame.shape[:2]
    # Subtle dark gradient
    frame[:] = BG_DARK
    
    # Animated vertical and horizontal grid lines
    grid_spacing = 80
    offset_x = int((t * 20) % grid_spacing)
    offset_y = int((t * 20) % grid_spacing)
    
    overlay = frame.copy()
    for x in range(offset_x, w, grid_spacing):
        cv2.line(overlay, (x, 0), (x, h), (35, 25, 18), 1)
    for y in range(offset_y, h, grid_spacing):
        cv2.line(overlay, (0, y), (w, y), (35, 25, 18), 1)
        
    # Scanning laser line
    scan_y = int((math.sin(t * 1.5) * 0.5 + 0.5) * h)
    cv2.line(overlay, (0, scan_y), (w, scan_y), (180, 140, 0), 2)
    cv2.GaussianBlur(overlay, (0, 0), sigmaX=3.0)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

def draw_hud_corners(frame):
    """Draw tactical HUD boundary corners."""
    h, w = frame.shape[:2]
    margin = 50
    len_corner = 40
    color = (180, 140, 0)
    
    # Top-Left
    cv2.line(frame, (margin, margin), (margin + len_corner, margin), color, 2)
    cv2.line(frame, (margin, margin), (margin, margin + len_corner), color, 2)
    # Top-Right
    cv2.line(frame, (w - margin, margin), (w - margin - len_corner, margin), color, 2)
    cv2.line(frame, (w - margin, margin), (w - margin, margin + len_corner), color, 2)
    # Bottom-Left
    cv2.line(frame, (margin, h - margin), (margin + len_corner, h - margin), color, 2)
    cv2.line(frame, (margin, h - margin), (margin, h - margin - len_corner), color, 2)
    # Bottom-Right
    cv2.line(frame, (w - margin, h - margin), (w - margin - len_corner, h - margin), color, 2)
    cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - len_corner), color, 2)

    # Top Header Tag
    cv2.putText(frame, "META-VISION // 442OOLS TACTICAL AI STUDIO", (margin + 20, margin + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN_GLOW, 2, cv2.LINE_AA)
    cv2.putText(frame, "SPATIAL & BIOMECHANICAL VISION ENGINE v2.4", (w - margin - 460, margin + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, MUTED, 1, cv2.LINE_AA)

def render_scene_1(frame, t, local_t):
    """Scene 1 (0-3.5s): Title & Vision Hook."""
    alpha = min(1.0, local_t * 1.5)
    
    center_x, center_y = WIDTH // 2, HEIGHT // 2 - 40
    
    # Pulsing Radar Target in center
    radius = int(80 + math.sin(local_t * 4) * 15)
    cv2.circle(frame, (center_x, center_y), radius, CYAN_GLOW, 2, cv2.LINE_AA)
    cv2.circle(frame, (center_x, center_y), int(radius * 0.6), (100, 200, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (center_x, center_y), 6, NEON_GREEN, -1, cv2.LINE_AA)

    # Title Banner
    cv2.putText(frame, "META-VISION", (center_x - 300, center_y + 160), cv2.FONT_HERSHEY_DUPLEX, 2.0, WHITE, 3, cv2.LINE_AA)
    cv2.putText(frame, "442OOLS INTELLIGENCE", (center_x - 330, center_y + 225), cv2.FONT_HERSHEY_DUPLEX, 1.4, CYAN_GLOW, 2, cv2.LINE_AA)
    cv2.putText(frame, "Next-Gen Computer Vision for Match Analytics & Biomechanical Dribble Tracking", (center_x - 490, center_y + 285), cv2.FONT_HERSHEY_SIMPLEX, 0.75, MUTED, 2, cv2.LINE_AA)

def render_scene_2(frame, t, local_t):
    """Scene 2 (3.5-7.5s): Core Architecture & Capabilities."""
    cv2.putText(frame, "SYSTEM CAPABILITIES & ARCHITECTURE", (WIDTH // 2 - 320, 180), cv2.FONT_HERSHEY_DUPLEX, 1.1, CYAN_GLOW, 2, cv2.LINE_AA)
    cv2.line(frame, (WIDTH // 2 - 340, 205), (WIDTH // 2 + 340, 205), BORDER_COL, 2)

    cards = [
        ("01. DETECTION & TRACKING", "Custom YOLOv8 (Player/GK/Ref/Ball)\n+ Team-Aware ByteTrack Stabilization", NEON_GREEN, 160),
        ("02. TACTICAL DECISION ENGINE", "Expected Threat (xT) Grid\n+ Through Pass, Lob & Cross Physics\n+ Goalkeeper Sweeping Agility Risk", CYAN_GLOW, 560),
        ("03. 1v1 BIOMECHANICAL POSE", "YOLOv8-Pose Multi-Skeleton\n+ FC/FIFA Posture Joystick Indicator\n+ Defender Wrong-Footing Detection", HOT_PINK, 960),
    ]

    for title, desc, col, start_x in cards:
        card_w, card_h = 360, 420
        card_y = 280
        
        # Glow Card Box
        cv2.rectangle(frame, (start_x, card_y), (start_x + card_w, card_y + card_h), (20, 28, 38), -1)
        cv2.rectangle(frame, (start_x, card_y), (start_x + card_w, card_y + card_h), col, 2)
        cv2.rectangle(frame, (start_x, card_y), (start_x + card_w, card_y + 55), (30, 40, 55), -1)
        
        cv2.putText(frame, title, (start_x + 18, card_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)
        
        lines = desc.split("\n")
        line_y = card_y + 110
        for l in lines:
            cv2.putText(frame, l, (start_x + 20, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED if not l.startswith("+") else col, 1, cv2.LINE_AA)
            line_y += 45

def render_scene_3(frame, t, local_t):
    """Scene 3 (7.5-11.0s): How To Run CLI & Web Studio."""
    cv2.putText(frame, "QUICKSTART: HOW TO RUN", (WIDTH // 2 - 230, 180), cv2.FONT_HERSHEY_DUPLEX, 1.1, CYAN_GLOW, 2, cv2.LINE_AA)
    cv2.line(frame, (WIDTH // 2 - 260, 205), (WIDTH // 2 + 260, 205), BORDER_COL, 2)

    # Terminal Box
    term_x, term_y = 260, 260
    term_w, term_h = 1400, 480
    
    cv2.rectangle(frame, (term_x, term_y), (term_x + term_w, term_y + term_h), (14, 18, 25), -1)
    cv2.rectangle(frame, (term_x, term_y), (term_x + term_w, term_y + term_h), (70, 90, 120), 2)
    # Header bar
    cv2.rectangle(frame, (term_x, term_y), (term_x + term_w, term_y + 40), (25, 32, 45), -1)
    cv2.putText(frame, "TERMINAL // BASH", (term_x + 25, term_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED, 1, cv2.LINE_AA)

    commands = [
        ("1. Launch Web Dashboard & API:", "python server.py", CYAN_GLOW),
        ("2. Run Match Tactical Intelligence Engine:", "python src/detect_track.py --source input_videos/kdb_clip.mov --output output_videos/tracked.mp4", NEON_GREEN),
        ("3. Run 1v1 Dribble Biomechanics & Posture:", "python src/dribble_pose_engine.py --video input_videos/mbappe_sociedad.mov", HOT_PINK),
    ]

    y = term_y + 90
    for label, cmd, col in commands:
        cv2.putText(frame, label, (term_x + 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2, cv2.LINE_AA)
        y += 35
        # Code background badge
        cv2.rectangle(frame, (term_x + 40, y - 24), (term_x + term_w - 40, y + 16), (22, 28, 38), -1)
        cv2.putText(frame, f"$ {cmd}", (term_x + 55, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)
        y += 75

def render_scene_4(frame, t, local_t):
    """Scene 4 (11.0-14.0s): Outro / Ready for Demo."""
    center_x, center_y = WIDTH // 2, HEIGHT // 2 - 40
    
    cv2.putText(frame, "READY FOR ANALYSIS", (center_x - 270, center_y), cv2.FONT_HERSHEY_DUPLEX, 1.6, NEON_GREEN, 3, cv2.LINE_AA)
    cv2.putText(frame, "LIVE DEMO FEED & PLAY-BY-PLAY BREAKDOWN", (center_x - 360, center_y + 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "Insert match clips & dribble breakdowns below", (center_x - 260, center_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, MUTED, 1, cv2.LINE_AA)

print(f"Rendering {TOTAL_FRAMES} frames intro video to {OUT_PATH}...")

for frame_idx in range(TOTAL_FRAMES):
    t = frame_idx / FPS
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    draw_background_grid(frame, t)
    draw_hud_corners(frame)
    
    if t < 3.5:
        render_scene_1(frame, t, t)
    elif t < 7.5:
        render_scene_2(frame, t, t - 3.5)
    elif t < 11.0:
        render_scene_3(frame, t, t - 7.5)
    else:
        render_scene_4(frame, t, t - 11.0)
        
    writer.write(frame)
    if frame_idx % 60 == 0:
        print(f"Rendered {frame_idx}/{TOTAL_FRAMES} frames...")

writer.release()
print(f"Successfully generated {OUT_PATH}!")
