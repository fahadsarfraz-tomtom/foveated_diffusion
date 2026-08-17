#!/usr/bin/env python3
"""Build a prompts CSV (column ``prompt``) from COCO captions.

Prefers val2017 captions; falls back to train2017 when the val annotations are
not on disk. One caption per image (the first listed), deterministic sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val2017")
    parser.add_argument("--fallback_split", type=str, default="train2017")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_captions(coco_root: Path, split: str) -> list[str]:
    path = coco_root / "annotations" / f"captions_{split}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    first_by_image: dict[int, str] = {}
    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        if image_id not in first_by_image:
            caption = " ".join(str(ann.get("caption", "")).split())
            if caption:
                first_by_image[image_id] = caption
    return [first_by_image[k] for k in sorted(first_by_image)]


def main():
    args = parse_args()
    captions = load_captions(args.coco_root, args.split)
    used_split = args.split
    if not captions:
        captions = load_captions(args.coco_root, args.fallback_split)
        used_split = args.fallback_split
    if not captions:
        raise SystemExit(
            f"no captions found under {args.coco_root}/annotations for "
            f"{args.split} or {args.fallback_split}"
        )

    rng = random.Random(args.seed)
    if len(captions) > args.num_prompts:
        captions = rng.sample(captions, args.num_prompts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt"])
        for caption in captions:
            writer.writerow([caption])
    print(f"[prompts] wrote {len(captions)} prompts from COCO {used_split} to {args.out}")


if __name__ == "__main__":
    main()
