#!/usr/bin/env python3
"""Build a UnifiedDataset metadata CSV (columns: image,prompt) from COCO captions.

Paths are written relative to --coco_root (the dataset_base_path), e.g.
``train2017/000000391895.jpg``. One (first) caption per image, deterministic
subsample.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco_root", type=Path, required=True)
    p.add_argument("--split", type=str, default="train2017")
    p.add_argument("--num_images", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    captions = json.loads(
        (args.coco_root / "annotations" / f"captions_{args.split}.json").read_text()
    )
    file_by_image = {img["id"]: img["file_name"] for img in captions["images"]}
    caption_by_image: dict[int, str] = {}
    for ann in captions["annotations"]:
        if ann["image_id"] not in caption_by_image:
            text = " ".join(str(ann.get("caption", "")).split())
            if text:
                caption_by_image[ann["image_id"]] = text

    ids = sorted(set(caption_by_image) & set(file_by_image))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    ids = ids[: args.num_images]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prompt"])
        for image_id in ids:
            writer.writerow([f"{args.split}/{file_by_image[image_id]}",
                             caption_by_image[image_id]])
    print(f"[metadata] wrote {len(ids)} rows ({args.split}) to {args.out}")


if __name__ == "__main__":
    main()
