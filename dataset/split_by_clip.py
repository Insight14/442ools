"""
Split labeled image/label pairs into train/val by SOURCE CLIP.
------------------------------------------------------------------
Why not a random split: consecutive frames from the same clip look very
similar. If near-duplicate frames end up on both sides of a random split,
validation accuracy looks better than the model actually is, because it's
partly "remembering" near-identical training frames. Splitting whole clips
into val (never seen in train) gives an honest signal.

Assumes filenames from src/extract_labeling_frames.py, i.e.
"<clipname>_f<frame_idx>.jpg" with a matching "<clipname>_f<frame_idx>.txt"
label file (standard CVAT/Label Studio YOLO export naming).

Usage:
    # First, see what clips you actually have:
    python dataset/split_by_clip.py --labeled-dir data/labeled --list-clips

    # Then assign specific clips to validation, rest go to train:
    python dataset/split_by_clip.py --labeled-dir data/labeled \
        --val-clips worldcup_final_2022 another_clip_name
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter


def clip_name_from_filename(filename: str) -> str:
    # "<clipname>_f000123.jpg" -> "<clipname>"
    stem = Path(filename).stem
    if "_f" in stem:
        return stem.rsplit("_f", 1)[0]
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", required=True, help="Folder containing labeled image+.txt pairs (flat, not yet split)")
    parser.add_argument("--dataset-dir", default="dataset", help="Target dataset/ folder with images/{train,val} and labels/{train,val}")
    parser.add_argument("--val-clips", nargs="*", default=[], help="Clip names to send to validation; all other clips go to train")
    parser.add_argument("--list-clips", action="store_true", help="Just print clip names found and their frame counts, then exit")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    images = sorted([p for p in labeled_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    if not images:
        print(f"No images found in {labeled_dir}")
        return

    clip_counts = Counter(clip_name_from_filename(p.name) for p in images)

    if args.list_clips:
        print(f"Found {len(images)} labeled images across {len(clip_counts)} clip(s):")
        for clip, count in clip_counts.most_common():
            print(f"  {clip}: {count} frames")
        print("\nRe-run with --val-clips <name1> <name2> ... to perform the split.")
        return

    val_clips = set(args.val_clips)
    unknown = val_clips - set(clip_counts)
    if unknown:
        print(f"Warning: these --val-clips weren't found in {labeled_dir}: {sorted(unknown)}")

    dataset_dir = Path(args.dataset_dir)
    counts = {"train": 0, "val": 0, "missing_label": 0}

    for img_path in images:
        clip = clip_name_from_filename(img_path.name)
        split = "val" if clip in val_clips else "train"

        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            counts["missing_label"] += 1
            print(f"  Skipping {img_path.name} -- no matching label file")
            continue

        img_out = dataset_dir / "images" / split / img_path.name
        label_out = dataset_dir / "labels" / split / label_path.name
        img_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, img_out)
        shutil.copy2(label_path, label_out)
        counts[split] += 1

    print(f"\nDone. train: {counts['train']} | val: {counts['val']} | skipped (no label): {counts['missing_label']}")


if __name__ == "__main__":
    main()