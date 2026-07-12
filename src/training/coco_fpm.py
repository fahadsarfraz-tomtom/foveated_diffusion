"""COCO supervision utilities for FPM pretraining.

This phase trains the foveation module to predict object count, centers, radii,
and a Gaussian foveal map from image+caption supervision. It is deliberately
separate from full FLUX LoRA training so we can validate the supervision signal
before plugging predicted masks into Chao's mixed-resolution denoising loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


COCO_ALIASES = {
    "person": {"person", "people", "man", "woman", "boy", "girl", "child", "children"},
    "airplane": {"airplane", "plane", "jet"},
    "bicycle": {"bicycle", "bike"},
    "motorcycle": {"motorcycle", "motorbike"},
    "sports ball": {"sports ball", "ball"},
    "cell phone": {"cell phone", "phone", "mobile phone"},
    "couch": {"couch", "sofa"},
    "tv": {"tv", "television", "screen"},
    "potted plant": {"potted plant", "plant"},
    "dining table": {"dining table", "table"},
    "traffic light": {"traffic light", "stoplight"},
    "fire hydrant": {"fire hydrant", "hydrant"},
    "stop sign": {"stop sign", "sign"},
    "parking meter": {"parking meter", "meter"},
    "wine glass": {"wine glass", "glass"},
    "hot dog": {"hot dog", "hotdog"},
    "teddy bear": {"teddy bear", "bear"},
    "hair drier": {"hair drier", "hair dryer", "dryer"},
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_-]*", text.lower())


def _normalize_text(text: str) -> str:
    return " ".join(_tokenize(text.replace("-", " ")))


def _pluralize(token: str) -> str:
    if token.endswith("y"):
        return token[:-1] + "ies"
    if token.endswith(("s", "x", "ch", "sh")):
        return token + "es"
    return token + "s"


def category_aliases(category_name: str) -> set[str]:
    aliases = set(COCO_ALIASES.get(category_name, set()))
    aliases.add(category_name)
    aliases.add(category_name.replace(" ", "-"))
    if " " not in category_name:
        aliases.add(_pluralize(category_name))
    return {_normalize_text(alias) for alias in aliases}


def caption_mentions_category(caption: str, category_name: str) -> bool:
    caption_norm = f" {_normalize_text(caption)} "
    for alias in category_aliases(category_name):
        if f" {alias} " in caption_norm:
            return True
    return False


def token_ids_from_caption(caption: str, vocab_size: int, max_tokens: int) -> torch.Tensor:
    ids = torch.zeros(max_tokens, dtype=torch.long)
    for idx, token in enumerate(_tokenize(caption)[:max_tokens]):
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        ids[idx] = int(digest[:8], 16) % (vocab_size - 1) + 1
    return ids


class HashTextEmbedder(nn.Module):
    """Small trainable text embedder for supervised COCO bootstrap training."""

    def __init__(self, vocab_size: int = 8192, text_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


@dataclass
class CocoTarget:
    center: tuple[float, float]
    radius: float
    category: str
    bbox_xywh: tuple[float, float, float, float]
    score: float


class CocoFoveationDataset(Dataset):
    """COCO captions + instance annotations converted into foveation slots."""

    def __init__(
        self,
        coco_root: str | Path,
        split: str = "train2017",
        image_size: int = 256,
        max_objects: int = 3,
        max_tokens: int = 32,
        vocab_size: int = 8192,
        caption_mode: str = "all",
        fallback_all_objects: bool = True,
        min_box_area: float = 16.0,
        radius_margin: float = 1.35,
        radius_min: float = 0.03,
        radius_max: float = 0.60,
        max_samples: int | None = None,
    ):
        self.coco_root = Path(coco_root)
        self.split = split
        self.image_dir = self.coco_root / split
        self.annotation_dir = self.coco_root / "annotations"
        self.image_size = image_size
        self.max_objects = max_objects
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.caption_mode = caption_mode
        self.fallback_all_objects = fallback_all_objects
        self.min_box_area = min_box_area
        self.radius_margin = radius_margin
        self.radius_min = radius_min
        self.radius_max = radius_max

        instances_path = self.annotation_dir / f"instances_{split}.json"
        captions_path = self.annotation_dir / f"captions_{split}.json"
        if not instances_path.exists():
            raise FileNotFoundError(f"missing COCO instances file: {instances_path}")
        if not captions_path.exists():
            raise FileNotFoundError(f"missing COCO captions file: {captions_path}")

        with open(instances_path) as f:
            instances = json.load(f)
        with open(captions_path) as f:
            captions = json.load(f)
        self.categories = {c["id"]: c["name"] for c in instances["categories"]}
        self.images = {img["id"]: img for img in instances["images"]}

        self.anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in instances["annotations"]:
            x, y, w, h = ann["bbox"]
            if ann.get("iscrowd", 0) or w * h < self.min_box_area:
                continue
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        captions_by_image: dict[int, list[str]] = {}
        for ann in captions["annotations"]:
            captions_by_image.setdefault(ann["image_id"], []).append(ann["caption"])

        samples = []
        for image_id, image in self.images.items():
            if image_id not in self.anns_by_image or image_id not in captions_by_image:
                continue
            image_path = self.image_dir / image["file_name"]
            if not image_path.exists():
                continue
            image_captions = captions_by_image[image_id]
            if caption_mode == "first":
                image_captions = image_captions[:1]
            elif caption_mode != "all":
                raise ValueError(f"unknown caption_mode: {caption_mode}")
            for caption in image_captions:
                samples.append((image_id, caption))
                if max_samples is not None and len(samples) >= max_samples:
                    break
            if max_samples is not None and len(samples) >= max_samples:
                break
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _select_targets(self, image_id: int, caption: str) -> list[CocoTarget]:
        image = self.images[image_id]
        width, height = float(image["width"]), float(image["height"])
        anns = self.anns_by_image[image_id]
        targets = []
        fallback = []
        for ann in anns:
            category = self.categories[ann["category_id"]]
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            area_norm = max((bw * bh) / max(width * height, 1.0), 1e-8)
            cx = (x + 0.5 * bw) / width
            cy = (y + 0.5 * bh) / height
            radius = 0.5 * math.sqrt((bw / width) ** 2 + (bh / height) ** 2)
            radius = float(np.clip(radius * self.radius_margin, self.radius_min, self.radius_max))
            small_bonus = 0.25 * (1.0 - min(area_norm / 0.20, 1.0))
            target = CocoTarget(
                center=(float(cx), float(cy)),
                radius=radius,
                category=category,
                bbox_xywh=(x, y, bw, bh),
                score=float(area_norm),
            )
            if caption_mentions_category(caption, category):
                target.score = 2.0 + small_bonus
                targets.append(target)
            else:
                fallback.append(target)
        if not targets and self.fallback_all_objects:
            targets = sorted(fallback, key=lambda item: item.score, reverse=True)
        else:
            targets = sorted(targets, key=lambda item: item.score, reverse=True)
        return targets[: self.max_objects]

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id, caption = self.samples[index]
        image_info = self.images[image_id]
        image_path = self.image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        image_np = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        centers = torch.zeros(self.max_objects, 2, dtype=torch.float32)
        radii = torch.zeros(self.max_objects, dtype=torch.float32)
        valid = torch.zeros(self.max_objects, dtype=torch.float32)
        categories = []
        targets = self._select_targets(image_id, caption)
        for target_idx, target in enumerate(targets):
            centers[target_idx] = torch.tensor(target.center, dtype=torch.float32)
            radii[target_idx] = float(target.radius)
            valid[target_idx] = 1.0
            categories.append(target.category)

        return {
            "image": image_tensor,
            "token_ids": token_ids_from_caption(caption, self.vocab_size, self.max_tokens),
            "target_centers": centers,
            "target_radii": radii,
            "target_valid": valid,
            "prompt": caption,
            "image_id": image_id,
            "categories": categories,
        }


def coco_foveation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
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


def target_gaussian_map(
    centers: torch.Tensor,
    radii: torch.Tensor,
    valid: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build normalized target maps [B, H, W] from padded object slots."""
    batch, num_slots, _ = centers.shape
    yy = torch.linspace(0, 1, height, device=centers.device, dtype=centers.dtype).view(1, 1, height, 1)
    xx = torch.linspace(0, 1, width, device=centers.device, dtype=centers.dtype).view(1, 1, 1, width)
    cx = centers[..., 0].view(batch, num_slots, 1, 1)
    cy = centers[..., 1].view(batch, num_slots, 1, 1)
    r = radii.view(batch, num_slots, 1, 1).clamp_min(1e-4)
    active = valid.view(batch, num_slots, 1, 1)
    weight = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)) * active
    weight = weight.sum(dim=1)
    return weight / weight.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def _best_assignment(cost: torch.Tensor, num_targets: int) -> list[tuple[int, int]]:
    if num_targets == 0:
        return []
    num_pred = cost.shape[0]
    best_score = None
    best_perm = None
    for pred_perm in itertools.permutations(range(num_pred), num_targets):
        score = sum(float(cost[pred_idx, tgt_idx].item()) for tgt_idx, pred_idx in enumerate(pred_perm))
        if best_score is None or score < best_score:
            best_score = score
            best_perm = pred_perm
    return [(pred_idx, tgt_idx) for tgt_idx, pred_idx in enumerate(best_perm or [])]


def fpm_supervision_loss(
    cx: torch.Tensor,
    cy: torch.Tensor,
    radii: torch.Tensor,
    object_logits: torch.Tensor,
    weight_map: torch.Tensor,
    target_centers: torch.Tensor,
    target_radii: torch.Tensor,
    target_valid: torch.Tensor,
    target_map: torch.Tensor,
    lambda_center: float = 2.0,
    lambda_radius: float = 0.5,
    lambda_object: float = 1.0,
    lambda_map: float = 1.0,
    lambda_budget: float = 0.1,
    lambda_repulsion: float = 0.01,
    lambda_area: float = 0.0,
    target_budget_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DETR-style slot matching plus map/budget supervision for FPM."""
    batch, num_slots = cx.shape
    pred_centers = torch.stack([cx, cy], dim=-1)
    object_targets = torch.zeros_like(object_logits)
    center_terms = []
    radius_terms = []
    matched_count = 0
    center_error_sum = 0.0

    for batch_idx in range(batch):
        num_targets = int(target_valid[batch_idx].sum().item())
        if num_targets <= 0:
            continue
        tgt_centers = target_centers[batch_idx, :num_targets]
        tgt_radii = target_radii[batch_idx, :num_targets]
        center_cost = torch.cdist(pred_centers[batch_idx], tgt_centers, p=1)
        radius_cost = torch.abs(radii[batch_idx].unsqueeze(1) - tgt_radii.unsqueeze(0))
        matches = _best_assignment(center_cost + 0.5 * radius_cost, num_targets)
        for pred_idx, tgt_idx in matches:
            object_targets[batch_idx, pred_idx] = 1.0
            center_loss = F.smooth_l1_loss(
                pred_centers[batch_idx, pred_idx],
                tgt_centers[tgt_idx],
                reduction="sum",
            )
            radius_loss = F.smooth_l1_loss(
                radii[batch_idx, pred_idx],
                tgt_radii[tgt_idx],
                reduction="sum",
            )
            center_terms.append(center_loss)
            radius_terms.append(radius_loss)
            center_error_sum += float(torch.abs(pred_centers[batch_idx, pred_idx] - tgt_centers[tgt_idx]).mean().item())
            matched_count += 1

    if center_terms:
        center_loss = torch.stack(center_terms).mean()
        radius_loss = torch.stack(radius_terms).mean()
    else:
        center_loss = object_logits.new_tensor(0.0)
        radius_loss = object_logits.new_tensor(0.0)

    object_loss = F.binary_cross_entropy_with_logits(object_logits, object_targets)
    map_mse = F.mse_loss(weight_map, target_map)
    inter = (weight_map * target_map).sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * inter + 1e-6) / (
        weight_map.sum(dim=(-2, -1)) + target_map.sum(dim=(-2, -1)) + 1e-6
    )
    map_loss = map_mse + dice.mean()
    budget_loss = F.mse_loss(weight_map.mean(dim=(-2, -1)), target_map.mean(dim=(-2, -1)))
    target_budget = target_map.mean(dim=(-2, -1)) * target_budget_scale
    area_overrun = torch.relu(weight_map.mean(dim=(-2, -1)) - target_budget)
    area_loss = (area_overrun ** 2).mean()

    repulsion_loss = object_logits.new_tensor(0.0)
    if num_slots > 1:
        distances = torch.cdist(pred_centers, pred_centers, p=2)
        eye = torch.eye(num_slots, device=distances.device, dtype=torch.bool).unsqueeze(0)
        pair_energy = torch.exp(-(distances ** 2) / (2 * 0.08 ** 2)).masked_fill(eye, 0.0)
        repulsion_loss = pair_energy.mean()

    total = (
        lambda_center * center_loss
        + lambda_radius * radius_loss
        + lambda_object * object_loss
        + lambda_map * map_loss
        + lambda_budget * budget_loss
        + lambda_repulsion * repulsion_loss
        + lambda_area * area_loss
    )
    metrics = {
        "loss": float(total.detach().item()),
        "loss_center": float(center_loss.detach().item()),
        "loss_radius": float(radius_loss.detach().item()),
        "loss_object": float(object_loss.detach().item()),
        "loss_map": float(map_loss.detach().item()),
        "loss_budget": float(budget_loss.detach().item()),
        "loss_repulsion": float(repulsion_loss.detach().item()),
        "loss_area": float(area_loss.detach().item()),
        "pred_area": float(weight_map.mean().detach().item()),
        "target_area": float(target_map.mean().detach().item()),
        "matched_slots": float(matched_count),
        "center_l1": center_error_sum / max(matched_count, 1),
        "target_count": float(target_valid.sum().item() / max(batch, 1)),
        "pred_count": float((torch.sigmoid(object_logits) > 0.5).float().sum().item() / max(batch, 1)),
    }
    return total, metrics
