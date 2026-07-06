"""Trainable foveation policy modules.

These modules are intentionally small and independent of the FLUX/Wan pipeline
internals. They provide the learnable part of our method; Chao's mixed-resolution
pipeline remains the execution substrate that consumes the masks/paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def gaussian_weight_map(
    cx: torch.Tensor,
    cy: torch.Tensor,
    r: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build a normalized K-Gaussian foveal map.

    Args:
        cx, cy, r: tensors shaped [B, K]. Coordinates are normalized to [0, 1].
        height, width: output grid size.

    Returns:
        Tensor [B, height, width] with values in [0, 1].
    """
    batch, num_fix = cx.shape
    yy = torch.linspace(0, 1, height, device=cx.device, dtype=cx.dtype).view(1, 1, height, 1)
    xx = torch.linspace(0, 1, width, device=cx.device, dtype=cx.dtype).view(1, 1, 1, width)

    cx = cx.view(batch, num_fix, 1, 1)
    cy = cy.view(batch, num_fix, 1, 1)
    r = r.view(batch, num_fix, 1, 1).clamp_min(1e-4)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    weight = torch.exp(-dist2 / (2 * r ** 2)).sum(dim=1)
    return weight / weight.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


class SpatialTokenScorer(nn.Module):
    """Scores text tokens by whether they should command spatial compute."""

    def __init__(self, text_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Return token scores [B, T] in [0, 1]."""
        return self.net(text_embeddings).squeeze(-1)


class FovealPredictionModule(nn.Module):
    """Predicts K Gaussian fixation centers/radii from noisy latents + text.

    This is the LSCA/FPM component from our proposal. It predicts normalized
    [0, 1] centers internally; adapter code can convert them to Chao's
    [-0.5, 0.5] coordinate convention when emitting masks/paths.
    """

    def __init__(
        self,
        latent_channels: int,
        text_dim: int,
        num_fixations: int = 3,
        hidden_dim: int = 128,
        r_init: float = 0.20,
        r_min: float = 0.03,
        r_max: float = 0.60,
    ):
        super().__init__()
        self.num_fixations = num_fixations
        self.r_min = r_min
        self.r_max = r_max

        self.spatial_enc = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.text_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fix_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_fixations * 3),
        )
        self.object_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_fixations),
        )

        nn.init.zeros_(self.fix_head[-1].weight)
        with torch.no_grad():
            bias = self.fix_head[-1].bias.view(num_fixations, 3)
            bias.zero_()
            bias[:, 2] = math.log(r_init)
        nn.init.zeros_(self.object_head[-1].weight)
        nn.init.constant_(self.object_head[-1].bias, -2.0)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def predict_slots(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        out_height: int | None = None,
        out_width: int | None = None,
    ):
        """Return slot predictions: (cx, cy, r, object_logits, weight_map).

        `object_logits` supervises the number of active foveal objects while
        preserving the original Gaussian map interface for downstream adapters.
        """
        batch = latents.shape[0]
        feats = self.spatial_enc(latents)
        grid_h, grid_w = feats.shape[-2:]
        feats = feats.flatten(2).transpose(1, 2)

        text = self.text_proj(text_embeddings)
        feats, _ = self.text_attn(feats, text, text)

        if timesteps is not None:
            t_emb = self.timestep_embedding(timesteps, feats.shape[-1]).to(feats.dtype)
            feats = feats + self.time_proj(t_emb).unsqueeze(1)

        pooled = feats.mean(dim=1)
        params = self.fix_head(pooled).view(batch, self.num_fixations, 3)
        cx = torch.sigmoid(params[..., 0])
        cy = torch.sigmoid(params[..., 1])
        r = torch.exp(params[..., 2]).clamp(self.r_min, self.r_max)
        object_logits = self.object_head(pooled)

        out_height = out_height or grid_h
        out_width = out_width or grid_w
        weight = gaussian_weight_map(cx, cy, r, out_height, out_width)
        return cx, cy, r, object_logits, weight

    def forward(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        out_height: int | None = None,
        out_width: int | None = None,
    ):
        """Return (cx, cy, r, weight_map).

        `cx`, `cy`, and `r` are shaped [B, K]. `weight_map` is produced at the
        latent-token grid if `out_height/out_width` are omitted.
        """
        cx, cy, r, _object_logits, weight = self.predict_slots(
            latents,
            text_embeddings,
            timesteps=timesteps,
            out_height=out_height,
            out_width=out_width,
        )
        return cx, cy, r, weight
