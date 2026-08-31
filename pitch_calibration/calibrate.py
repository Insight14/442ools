"""
Interactive pitch calibration.
---------------------------------------------------
Click known pitch landmarks (corners, penalty box edges, center spot, etc.)
on a single reference frame from your clip, and this computes + saves the
homography matrix that converts pixel coordinates to real-world pitch
coordinates (in metres).

This is a ONE-TIME calibration per fixed camera shot. If your clip has the
camera panning/zooming significantly, this single homography will drift out
of accuracy over the clip -- that's a known limitation (see README Phase 6
notes on camera movement estimation, not yet built).

Workflow:
    1. Run this script, pointing it at your clip. It opens the first frame
       in a window.
    2. Look at the frame and identify which pitch landmarks are visible
       (corners, penalty box lines, center circle, etc.)
    3. Click a landmark on the image, then type its name in the terminal
       (autocomplete-style: partial names are matched if unambiguous).
    4. Repeat for at least 4 landmarks (more = better, aim for 6+ spread
       across the frame, not clustered in one area).
    5. Press 'q' in the image window when done. The homography is computed
       and saved to pitch_calibration/homography_<clipname>.json

Usage:
    python pitch_calibration/calibrate.py --source input_videos/clip.mp4 --frame 0
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pitch_calibration.pitch_reference import PITCH_LANDMARKS


def match_landmark_name(user_input: str):
    """Match partial/case-insensitive user input to a landmark name.
    Returns the matched name, or None if no unambiguous match."""
    user_input = user_input.strip().lower()
    exact = [n for n in PITCH_LANDMARKS if n == user_input]
    if exact:
        return exact[0]
    partial = [n for n in PITCH_LANDMARKS if user_input in n]
    if len(partial) == 1:
        return partial[0]
    return None


def run(source_path: str, frame_index: int, output_dir: str):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {source_path}")

    clicked_points = []  # list of (pixel_x, pixel_y)
    display = frame.copy()

    print("\nAvailable landmark names:")
    for name in sorted(PITCH_LANDMARKS):
        print(f"  {name}")
    print("\nClick a point on the image window, then type its landmark name here.")
    print("Type 'done' instead of a name to finish early. Need at least 4 points.\n")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Calibration frame - click landmarks", display)

    cv2.namedWindow("Calibration frame - click landmarks")
    cv2.setMouseCallback("Calibration frame - click landmarks", on_click)
    cv2.imshow("Calibration frame - click landmarks", display)

    pixel_points = []
    world_points = []
    used_names = set()

    while True:
        cv2.waitKey(200)  # let the window refresh while we wait on terminal input
        if len(clicked_points) > len(pixel_points):
            px, py = clicked_points[-1]
            name = None
            while name is None:
                raw = input(f"Point at ({px},{py}) -- landmark name (or 'done'): ")
                if raw.strip().lower() == "done":
                    break
                name = match_landmark_name(raw)
                if name is None:
                    print("  Not recognized or ambiguous -- try again (see list above).")
                elif name in used_names:
                    print(f"  '{name}' already used for another point -- pick a different landmark.")
                    name = None

            if raw.strip().lower() == "done":
                break

            used_names.add(name)
            pixel_points.append([px, py])
            world_points.append(list(PITCH_LANDMARKS[name]))
            print(f"  -> recorded '{name}'  ({len(pixel_points)} points so far)\n")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(pixel_points) < 4:
        print(f"\nOnly {len(pixel_points)} points recorded -- need at least 4 for a homography. Aborting without saving.")
        return

    pixel_arr = np.array(pixel_points, dtype=np.float32)
    world_arr = np.array(world_points, dtype=np.float32)

    homography, mask = cv2.findHomography(pixel_arr, world_arr, method=cv2.RANSAC)
    if homography is None:
        print("\ncv2.findHomography failed to compute a solution -- check your points aren't collinear or mis-clicked.")
        return

    # Reprojection error: how far off is each point when mapped through the
    # computed homography, vs. where it should land. Good sanity check for
    # whether the calibration is actually trustworthy.
    reprojected = cv2.perspectiveTransform(pixel_arr.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(reprojected - world_arr, axis=1)
    print(f"\nReprojection error per point (metres): {[round(e, 3) for e in errors]}")
    print(f"Mean reprojection error: {errors.mean():.3f} m")
    if errors.mean() > 2.0:
        print("WARNING: mean error is quite high (>2m). Double check your clicks were accurate "
              "and the landmark names you typed match where you actually clicked.")

    os.makedirs(output_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(source_path))[0]
    out_path = os.path.join(output_dir, f"homography_{clip_name}.json")
    with open(out_path, "w") as f:
        json.dump({
            "source_clip": source_path,
            "frame_index": frame_index,
            "homography": homography.tolist(),
            "pixel_points": pixel_points,
            "world_points": world_points,
            "mean_reprojection_error_m": float(errors.mean()),
        }, f, indent=2)

    print(f"\nSaved calibration to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the video clip to calibrate")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use as the calibration reference (default: first frame)")
    parser.add_argument("--out", default="pitch_calibration", help="Output directory for the saved homography file")
    args = parser.parse_args()

    run(args.source, args.frame, args.out)