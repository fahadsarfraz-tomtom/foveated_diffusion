"""Adaptive foveation policy layer for Chao-style mixed-resolution diffusion.

This file provides the non-invasive adapter between our proposal components and
Chao's implementation:

- LSCA/FPM emits where to look (represented here by deterministic proxy policies
  until a learned checkpoint is trained).
- NaFo emits the budget to use.
- TextFov proxy uses spatial prompt tokens to seed mask centers.
- Video path planning emits `(centers, radii)` consumable by `build_state`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
import re
from typing import Any

import torch

from .shapes import create_foveation_mask, create_foveation_mask_full_res


SPATIAL_TOKEN_HINTS = {
    "animal", "baby", "ball", "bird", "boat", "book", "bottle", "building",
    "bus", "car", "cat", "chair", "child", "clock", "dog", "dress", "eye",
    "eyes", "face", "flower", "food", "hand", "horse", "house", "logo",
    "man", "person", "phone", "portrait", "sign", "table", "text", "tower",
    "tree", "woman",
}


@dataclass
class AdaptiveFoveationConfig:
    """Configuration for a prompt-conditioned foveation policy."""

    policy: str = "prompt_hash"
    num_fixations: int = 3
    center_range: float = 0.35
    mask_shape: str = "circular"
    fixed_beta: float = 0.25
    beta_mode: str = "fixed"
    beta_min: float = 0.20
    beta_max: float = 0.85
    beta_schedule: str = "cosine"
    fixed_radius: float | None = None


@dataclass
class FoveationPlan:
    """A Chao-compatible image foveation plan."""

    token_mask: torch.Tensor
    full_res_mask: torch.Tensor
    centers: list[tuple[float, float]]
    radii: list[float]
    beta: float
    hr_fraction: float
    token_ratio: float
    policy: str
    metadata: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        data = {
            "centers": [[float(x), float(y)] for x, y in self.centers],
            "radii": [float(r) for r in self.radii],
            "beta": float(self.beta),
            "hr_fraction": float(self.hr_fraction),
            "token_ratio": float(self.token_ratio),
            "policy": self.policy,
        }
        data.update(self.metadata)
        return data


def nafo_beta(
    step_index: int,
    num_steps: int,
    beta_min: float = 0.20,
    beta_max: float = 0.85,
    schedule: str = "cosine",
) -> float:
    """Noise-adaptive foveal budget.

    `step_index=0` is the high-noise beginning of sampling and receives
    `beta_max`; the final step receives `beta_min`.
    """
    if num_steps <= 1:
        progress = 0.0
    else:
        progress = 1.0 - float(step_index) / float(num_steps - 1)
    progress = min(max(progress, 0.0), 1.0)

    if schedule == "linear":
        factor = progress
    elif schedule == "stepped":
        if progress > 0.6:
            factor = 1.0
        elif progress > 0.3:
            return 0.5 * (beta_min + beta_max)
        else:
            factor = 0.0
    elif schedule == "cosine":
        factor = 0.5 * (1.0 - math.cos(progress * math.pi))
    else:
        raise ValueError(f"unknown NaFo schedule: {schedule}")
    return beta_min + (beta_max - beta_min) * factor


def resolve_static_beta(config: AdaptiveFoveationConfig, num_steps: int) -> float:
    """Project a NaFo schedule to the static mask interface used by Chao image inference."""
    if config.beta_mode == "fixed":
        return config.fixed_beta
    if config.beta_mode == "nafo_early":
        return nafo_beta(0, num_steps, config.beta_min, config.beta_max, config.beta_schedule)
    if config.beta_mode == "nafo_late":
        return nafo_beta(num_steps - 1, num_steps, config.beta_min, config.beta_max, config.beta_schedule)
    if config.beta_mode == "nafo_mean":
        values = [
            nafo_beta(i, num_steps, config.beta_min, config.beta_max, config.beta_schedule)
            for i in range(max(num_steps, 1))
        ]
        return float(sum(values) / len(values))
    raise ValueError(f"unknown beta_mode: {config.beta_mode}")


def token_ratio_from_mask(mask: torch.Tensor, lr_factor: int = 2) -> float:
    """Effective mixed-resolution token ratio under Chao's packing rule."""
    total = float(mask.numel())
    hr = float(mask.float().sum().item())
    lr = (total - hr) / float(lr_factor ** 2)
    return (hr + lr) / total


def _stable_unit_interval(key: str, offset: int) -> float:
    digest = hashlib.sha256(f"{key}|{offset}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2 ** 64 - 1)


def _stable_center(key: str, offset: int, center_range: float) -> tuple[float, float]:
    ux = _stable_unit_interval(key, 2 * offset)
    uy = _stable_unit_interval(key, 2 * offset + 1)
    return ((2 * ux - 1) * center_range, (2 * uy - 1) * center_range)


def _prompt_tokens(prompt: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]*", prompt.lower())


def _spatial_token_score(token: str) -> float:
    if token in SPATIAL_TOKEN_HINTS:
        return 1.0
    if token.endswith(("er", "or", "ist")):
        return 0.55
    if len(token) >= 6:
        return 0.35
    return 0.05


def _centers_from_prompt(prompt: str, config: AdaptiveFoveationConfig) -> list[tuple[float, float]]:
    if config.policy == "center":
        return [(0.0, 0.0)]

    if config.policy == "textfov_proxy":
        scored = sorted(
            ((tok, _spatial_token_score(tok)) for tok in _prompt_tokens(prompt)),
            key=lambda item: item[1],
            reverse=True,
        )
        tokens = [tok for tok, score in scored if score >= 0.35]
        centers = [
            _stable_center(f"{prompt}|{tok}", idx, config.center_range)
            for idx, tok in enumerate(tokens[: config.num_fixations])
        ]
        while len(centers) < config.num_fixations:
            centers.append(_stable_center(prompt, len(centers), config.center_range))
        return centers

    if config.policy in ("prompt_hash", "lsca_proxy"):
        return [
            _stable_center(prompt, idx, config.center_range)
            for idx in range(max(1, config.num_fixations))
        ]

    raise ValueError(f"unknown adaptive policy: {config.policy}")


def _combine_token_masks(
    height: int,
    width: int,
    centers: list[tuple[float, float]],
    radius: float,
    shape: str,
    device: torch.device,
    lr_factor: int,
) -> torch.Tensor:
    mask = None
    for center in centers:
        m = create_foveation_mask(height, width, center, radius, shape, device, lr_factor=lr_factor)
        mask = m if mask is None else torch.maximum(mask, m)
    return mask


def _combine_full_res_masks(
    height: int,
    width: int,
    centers: list[tuple[float, float]],
    radius: float,
    shape: str,
    device: torch.device,
) -> torch.Tensor:
    mask = None
    for center in centers:
        m = create_foveation_mask_full_res(height, width, center, radius, shape, device)
        mask = m if mask is None else torch.maximum(mask, m)
    return mask


def _radius_for_budget(
    height: int,
    width: int,
    centers: list[tuple[float, float]],
    beta: float,
    shape: str,
    device: torch.device,
    lr_factor: int,
) -> float:
    """Find a common radius whose HR mask fraction is close to `beta`."""
    beta = min(max(float(beta), 0.001), 0.999)
    lo, hi = 0.01, 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        mask = _combine_token_masks(height, width, centers, mid, shape, device, lr_factor)
        frac = float(mask.float().mean().item())
        if frac < beta:
            lo = mid
        else:
            hi = mid
    return hi


class AdaptiveFoveationPolicy:
    """Builds Chao-compatible masks/paths from our adaptive policy components."""

    def __init__(self, config: AdaptiveFoveationConfig):
        self.config = config

    def plan_image(
        self,
        prompt: str,
        height: int,
        width: int,
        device: torch.device,
        num_inference_steps: int,
        lr_factor: int = 2,
    ) -> FoveationPlan:
        centers = _centers_from_prompt(prompt, self.config)
        beta = resolve_static_beta(self.config, num_inference_steps)
        if self.config.fixed_radius is not None:
            radius = float(self.config.fixed_radius)
        else:
            radius = _radius_for_budget(
                height, width, centers, beta, self.config.mask_shape, device, lr_factor
            )

        token_mask = _combine_token_masks(
            height, width, centers, radius, self.config.mask_shape, device, lr_factor
        )
        full_res_mask = _combine_full_res_masks(
            height, width, centers, radius, self.config.mask_shape, device
        )
        hr_fraction = float(token_mask.float().mean().item())
        token_ratio = token_ratio_from_mask(token_mask, lr_factor=lr_factor)
        metadata = {
            "config": asdict(self.config),
            "num_inference_steps": int(num_inference_steps),
            "lr_factor": int(lr_factor),
        }
        return FoveationPlan(
            token_mask=token_mask,
            full_res_mask=full_res_mask,
            centers=centers,
            radii=[radius for _ in centers],
            beta=beta,
            hr_fraction=hr_fraction,
            token_ratio=token_ratio,
            policy=self.config.policy,
            metadata=metadata,
        )

    def plan_video(
        self,
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        lr_factor: int = 2,
    ) -> tuple[list[tuple[float, float]], list[float], dict[str, Any]]:
        latent_length = (num_frames - 1) // 4 + 1
        anchors = _centers_from_prompt(prompt, self.config)
        if len(anchors) == 1:
            centers = anchors * latent_length
        else:
            centers = []
            for i in range(latent_length):
                pos = i / max(latent_length - 1, 1)
                scaled = pos * (len(anchors) - 1)
                left = min(int(math.floor(scaled)), len(anchors) - 2)
                frac = scaled - left
                x0, y0 = anchors[left]
                x1, y1 = anchors[left + 1]
                centers.append((x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)))

        beta = resolve_static_beta(self.config, max(latent_length, 1))
        if self.config.fixed_radius is not None:
            radius = float(self.config.fixed_radius)
        else:
            radius = _radius_for_budget(
                height, width, centers[:1], beta, self.config.mask_shape, device, lr_factor
            )
        radii = [radius for _ in range(latent_length)]
        metadata = {
            "config": asdict(self.config),
            "latent_length": latent_length,
            "beta": float(beta),
            "radius": float(radius),
            "anchors": [[float(x), float(y)] for x, y in anchors],
        }
        return centers, radii, metadata


def adaptive_config_from_args(args) -> AdaptiveFoveationConfig:
    """Build `AdaptiveFoveationConfig` from image/video inference args."""
    return AdaptiveFoveationConfig(
        policy=getattr(args, "adaptive_policy", "prompt_hash"),
        num_fixations=getattr(args, "adaptive_num_fixations", 3),
        center_range=getattr(args, "adaptive_center_range", 0.35),
        mask_shape=getattr(args, "mask_shape", "circular"),
        fixed_beta=getattr(args, "adaptive_fixed_beta", 0.25),
        beta_mode=getattr(args, "adaptive_beta_mode", "fixed"),
        beta_min=getattr(args, "adaptive_beta_min", 0.20),
        beta_max=getattr(args, "adaptive_beta_max", 0.85),
        beta_schedule=getattr(args, "adaptive_beta_schedule", "cosine"),
        fixed_radius=getattr(args, "adaptive_radius", None),
    )
