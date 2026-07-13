"""LVIS supervision utilities for FPM pretraining.

LVIS reuses COCO images but provides a much larger long-tail object vocabulary.
For the foveation module this gives us direct object-center/radius supervision
without needing natural captions: each image is paired with a synthetic prompt
constructed from the selected LVIS categories.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .coco_fpm import CocoTarget, token_ids_from_caption


LVIS_FREQUENCY_WEIGHT = {
    "r": 1.45,
    "c": 1.20,
    "f": 1.00,
}


def _category_display_name(category: dict[str, Any]) -> str:
    name = category.get("name")
    if name:
        return str(name).replace("_", " ")
    synonyms = category.get("synonyms") or []
    if synonyms:
        return str(synonyms[0]).replace("_", " ")
    return f"category {category.get('id', 'unknown')}"


def _image_file_name(image: dict[str, Any]) -> str:
    file_name = image.get("file_name")
    if file_name:
        return str(file_name)
    coco_url = image.get("coco_url")
    if coco_url:
        return str(coco_url).rstrip("/").split("/")[-1]
    image_id = int(image["id"])
    return f"{image_id:012d}.jpg"


class LvisFoveationDataset(Dataset):
    """LVIS instance annotations converted into foveation slots."""

    def __init__(
        self,
        lvis_root: str | Path,
        coco_image_root: str | Path,
        split: str = "train2017",
        ann_file: str | None = None,
        image_size: int = 256,
        max_objects: int = 3,
        max_tokens: int = 32,
        vocab_size: int = 8192,
        prompt_template: str = "a photo containing {categories}",
        min_box_area: float = 16.0,
        radius_margin: float = 1.35,
        radius_min: float = 0.03,
        radius_max: float = 0.60,
        rare_category_bonus: float = 0.15,
        max_samples: int | None = None,
    ):
        self.lvis_root = Path(lvis_root)
        self.coco_image_root = Path(coco_image_root)
        self.split = split
        self.image_dir = self.coco_image_root / split
        self.image_size = image_size
        self.max_objects = max_objects
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.prompt_template = prompt_template
        self.min_box_area = min_box_area
        self.radius_margin = radius_margin
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.rare_category_bonus = rare_category_bonus

        split_key = split.replace("2017", "")
        annotation_name = ann_file or f"lvis_v1_{split_key}.json"
        annotation_path = self.lvis_root / "annotations" / annotation_name
        if not annotation_path.exists():
            annotation_path = self.lvis_root / annotation_name
        if not annotation_path.exists():
            raise FileNotFoundError(f"missing LVIS annotation file: {annotation_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"missing COCO image directory for LVIS: {self.image_dir}")

        with open(annotation_path) as f:
            data = json.load(f)

        self.categories = {c["id"]: c for c in data["categories"]}
        self.category_names = {
            category_id: _category_display_name(category)
            for category_id, category in self.categories.items()
        }
        self.category_frequencies = {
            category_id: str(category.get("frequency", ""))
            for category_id, category in self.categories.items()
        }
        self.images = {img["id"]: img for img in data["images"]}

        self.anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            x, y, w, h = ann["bbox"]
            if ann.get("iscrowd", 0) or w * h < self.min_box_area:
                continue
            if ann["image_id"] not in self.images:
                continue
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        samples = []
        for image_id, image in self.images.items():
            if image_id not in self.anns_by_image:
                continue
            if not (self.image_dir / _image_file_name(image)).exists():
                continue
            samples.append(image_id)
            if max_samples is not None and len(samples) >= max_samples:
                break
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _select_targets(self, image_id: int) -> list[CocoTarget]:
        image = self.images[image_id]
        width, height = float(image["width"]), float(image["height"])
        targets = []
        for ann in self.anns_by_image[image_id]:
            category_id = ann["category_id"]
            category = self.category_names.get(category_id, f"category {category_id}")
            frequency = self.category_frequencies.get(category_id, "")
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            area_norm = max((bw * bh) / max(width * height, 1.0), 1e-8)
            cx = (x + 0.5 * bw) / width
            cy = (y + 0.5 * bh) / height
            radius = 0.5 * math.sqrt((bw / width) ** 2 + (bh / height) ** 2)
            radius = float(np.clip(radius * self.radius_margin, self.radius_min, self.radius_max))
            small_bonus = 0.25 * (1.0 - min(area_norm / 0.20, 1.0))
            frequency_weight = LVIS_FREQUENCY_WEIGHT.get(frequency, 1.0)
            rare_bonus = self.rare_category_bonus if frequency == "r" else 0.0
            score = area_norm * frequency_weight + small_bonus + rare_bonus
            targets.append(
                CocoTarget(
                    center=(float(cx), float(cy)),
                    radius=radius,
                    category=category,
                    bbox_xywh=(x, y, bw, bh),
                    score=float(score),
                )
            )
        return sorted(targets, key=lambda item: item.score, reverse=True)[: self.max_objects]

    def _prompt_from_categories(self, categories: list[str]) -> str:
        if not categories:
            return "a photo"
        unique_categories = list(dict.fromkeys(categories))
        category_text = ", ".join(unique_categories)
        return self.prompt_template.format(
            categories=category_text,
            category=unique_categories[0],
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.samples[index]
        image_info = self.images[image_id]
        image_path = self.image_dir / _image_file_name(image_info)
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        image_np = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        centers = torch.zeros(self.max_objects, 2, dtype=torch.float32)
        radii = torch.zeros(self.max_objects, dtype=torch.float32)
        valid = torch.zeros(self.max_objects, dtype=torch.float32)
        categories = []
        targets = self._select_targets(image_id)
        for target_idx, target in enumerate(targets):
            centers[target_idx] = torch.tensor(target.center, dtype=torch.float32)
            radii[target_idx] = float(target.radius)
            valid[target_idx] = 1.0
            categories.append(target.category)

        prompt = self._prompt_from_categories(categories)
        return {
            "image": image_tensor,
            "token_ids": token_ids_from_caption(prompt, self.vocab_size, self.max_tokens),
            "target_centers": centers,
            "target_radii": radii,
            "target_valid": valid,
            "prompt": prompt,
            "image_id": image_id,
            "categories": categories,
        }


def lvis_foveation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "token_ids": torch.stack([item["token_ids"] for item in batch]),
        "target_centers": torch.stack([item["target_centers"] for item in batch]),
        "target_radii": torch.stack([item["target_radii"] for item in batch]),
        "target_valid": torch.stack([item["target_valid"] for item in batch]),
        "prompt": [item["prompt"] for item in batch],
        "image_id": torch.tensor([item["image_id"] for item in batch], dtype=torch.long),
        "categories": [item["categories"] for item in batch],
    }
