"""
Phase 2: Extract labeling candidate frames from source clips.
---------------------------------------------------------------
Goal: turn raw video clips into a curated set of still frames worth
labeling, filtering out frames that waste labeling time -- blurry
(motion-blurred) frames and near-duplicates of a frame we already kept.

This does NOT decide what's "interesting" football-wise (a corner kick vs.
open play) -- that judgment is still yours. It just stops you from
manually skipping through near-identical or unusably blurry frames.

Blur filtering uses a PERCENTILE cutoff, not a fixed score threshold.
An earlier version of this script used a fixed threshold (80) and it was
wrong -- tested against real broadcast footage, the median sharpness score
was ~43, so a fixed 80 rejected the majority of normal frames, not just
genuinely blurry ones. Laplacian-variance "sharpness" scores aren't
comparable across different footage/resolutions/compression, so a fixed
number doesn't generalize. Dropping the bottom N% of each clip's own score
distribution does.

Usage:
    python src/extract_labeling_frames.py \
        --source input_videos/clip1.mp4 input_videos/clip2.mp4 \
        --out data/candidates \
        --every 15 \
        --blur-drop-pct 15 \
        --dup-threshold 0.90
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def blur_score(gray_frame: np.ndarray) -> float:
    """Higher = sharper. Uses variance of the Laplacian -- a standard,
    cheap blur-detection heuristic. Not comparable across different clips/
    resolutions in absolute terms -- see module docstring."""
    return cv2.Laplacian(gray_frame, cv2.CV_64F).var()


def frame_similarity(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Returns a 0-1 similarity score between two grayscale frames using
    normalized cross-correlation on a downsized version (fast, good enough
    to catch near-duplicates -- this is NOT meant to be a rigorous
    perceptual hash)."""
    small_a = cv2.resize(gray_a, (64, 64))
    small_b = cv2.resize(gray_b, (64, 64))
    a = small_a.astype(np.float32).flatten()
    b = small_b.astype(np.float32).flatten()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-6
    return float(np.dot(a, b) / denom)


def extract_from_clip(source_path: str, out_dir: str, every: int, blur_drop_pct: float, dup_threshold: float):
    clip_name = Path(source_path).stem
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    # Pass 1: sample candidate frames, drop near-duplicates immediately
    # (cheap, sequential check), but hold blur scoring until we've seen
    # the whole clip's distribution.
    candidates = []  # list of (frame_idx, frame_bgr, blur_score)
    frame_idx = 0
    last_kept_gray = None
    skipped_dup = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if last_kept_gray is not None and frame_similarity(gray, last_kept_gray) >= dup_threshold:
                skipped_dup += 1
            else:
                candidates.append((frame_idx, frame, blur_score(gray)))
                last_kept_gray = gray

        frame_idx += 1
    cap.release()

    if not candidates:
        print(f"[{clip_name}] no candidate frames survived duplicate filtering.")
        return

    # Pass 2: drop the least-sharp N% of THIS clip's own candidates,
    # rather than comparing against an arbitrary fixed number.
    scores = np.array([c[2] for c in candidates])
    cutoff = np.percentile(scores, blur_drop_pct)

    saved = 0
    skipped_blur = 0
    for frame_idx, frame, score in candidates:
        if score < cutoff:
            skipped_blur += 1
            continue
        out_name = f"{clip_name}_f{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    print(f"[{clip_name}] frames read: {frame_idx} | candidates: {len(candidates)} | "
          f"saved: {saved} | skipped (blurriest {blur_drop_pct:.0f}%): {skipped_blur} | "
          f"skipped (near-duplicate): {skipped_dup} | sharpness range: {scores.min():.1f}-{scores.max():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", required=True, help="One or more source video paths")
    parser.add_argument("--out", default="data/candidates", help="Output folder for candidate frames")
    parser.add_argument("--every", type=int, default=15, help="Sample every Nth frame before filtering (e.g. 15 at ~30fps = ~2 candidates/sec)")
    parser.add_argument("--blur-drop-pct", type=float, default=15.0, help="Drop the blurriest N%% of each clip's own candidate frames (0-100)")
    parser.add_argument("--dup-threshold", type=float, default=0.90, help="Similarity above which a frame is treated as a near-duplicate of the last kept frame (0-1)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for source in args.source:
        extract_from_clip(source, args.out, args.every, args.blur_drop_pct, args.dup_threshold)

    print(f"\nDone. Candidate frames written to: {args.out}")
    print("Next: upload this folder to CVAT/Label Studio and start labeling.")