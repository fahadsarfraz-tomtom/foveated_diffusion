#!/usr/bin/env python3
"""Download COCO and Visual Genome for foveation-policy supervision.

Default layout:

    data/foveation/
      coco/
        train2017/
        val2017/
        annotations/
      visual_genome/
        VG_100K/
        VG_100K_2/
        image_data.json
        region_descriptions.json
        objects_v1_2.json
        ...
      _archives/

The files are large. COCO train+val+annotations is about 20GB compressed;
Visual Genome images+selected annotations are larger than that again.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from urllib.request import urlretrieve
import zipfile


COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ),
}

VG_IMAGE_URLS = {
    "images.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "images2.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
}

VG_ANNOTATION_URLS = {
    "image_data.json.zip": (
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
    ),
    "region_descriptions.json.zip": (
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"
    ),
    "objects_v1_2.json.zip": (
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects_v1_2.json.zip"
    ),
    "relationships_v1_2.json.zip": (
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships_v1_2.json.zip"
    ),
    "attributes.json.zip": (
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip"
    ),
    "scene_graphs.json.zip": (
        "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/scene_graphs.json.zip"
    ),
}


def _progress(name: str):
    started = time.time()

    def hook(blocks: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = blocks * block_size
        pct = min(downloaded / total_size, 1.0) * 100
        elapsed = max(time.time() - started, 1e-6)
        mb = downloaded / (1024 ** 2)
        speed = mb / elapsed
        print(f"\r{name}: {pct:5.1f}%  {mb:,.1f} MiB  {speed:,.1f} MiB/s", end="")

    return hook


def download(url: str, path: Path, skip_existing: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and path.exists() and path.stat().st_size > 0:
        print(f"[skip] {path}")
        return
    print(f"[download] {url}")
    urlretrieve(url, path, _progress(path.name))
    print()


def extract_zip(path: Path, dest: Path, skip_existing: bool = True):
    marker = dest / f".extracted_{path.stem}"
    if skip_existing and marker.exists():
        print(f"[skip extract] {path.name}")
        return
    print(f"[extract] {path} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def setup_coco(root: Path, archive_dir: Path, no_extract: bool, skip_existing: bool):
    coco_dir = root / "coco"
    coco_archive = archive_dir / "coco"
    for name, url in COCO_URLS.items():
        zip_path = coco_archive / name
        download(url, zip_path, skip_existing=skip_existing)
        if not no_extract:
            extract_zip(zip_path, coco_dir, skip_existing=skip_existing)


def setup_visual_genome(
    root: Path,
    archive_dir: Path,
    no_extract: bool,
    skip_existing: bool,
    skip_images: bool,
):
    vg_dir = root / "visual_genome"
    vg_archive = archive_dir / "visual_genome"
    urls = dict(VG_ANNOTATION_URLS)
    if not skip_images:
        urls = {**VG_IMAGE_URLS, **urls}
    for name, url in urls.items():
        zip_path = vg_archive / name
        download(url, zip_path, skip_existing=skip_existing)
        if not no_extract:
            extract_zip(zip_path, vg_dir, skip_existing=skip_existing)


def write_manifest(root: Path, datasets: list[str], skip_vg_images: bool):
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "datasets": datasets,
        "coco_urls": COCO_URLS if "coco" in datasets else {},
        "visual_genome_image_urls": (
            {} if skip_vg_images or "visual_genome" not in datasets else VG_IMAGE_URLS
        ),
        "visual_genome_annotation_urls": (
            VG_ANNOTATION_URLS if "visual_genome" in datasets else {}
        ),
    }
    path = root / "dataset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/foveation"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["coco", "visual_genome"],
        choices=["coco", "visual_genome"],
    )
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Redownload/re-extract existing files.")
    parser.add_argument(
        "--skip-vg-images",
        action="store_true",
        help="Download Visual Genome annotations only. Useful before the Phase 3 trainer exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    archive_dir = (args.archive_dir.expanduser().resolve() if args.archive_dir else root / "_archives")
    skip_existing = not args.overwrite

    root.mkdir(parents=True, exist_ok=True)
    if "coco" in args.datasets:
        setup_coco(root, archive_dir, args.no_extract, skip_existing)
    if "visual_genome" in args.datasets:
        setup_visual_genome(
            root,
            archive_dir,
            args.no_extract,
            skip_existing,
            skip_images=args.skip_vg_images,
        )
    write_manifest(root, args.datasets, args.skip_vg_images)
    print("[done]")


if __name__ == "__main__":
    main()
