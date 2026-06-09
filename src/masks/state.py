"""Foveation state for the Wan video pipeline.

`build_state` is the central converter from a per-frame foveation
path (centers + radii) to the `FoveationState` consumed by the foveated Wan
DiT and pipeline. State fields:

    hr_grid_mask         [1, 1, T_lat, h*2, w*2]   HR-grid mask, 1 = HR token
    lr_grid_mask         [1, 1, T_lat, h, w]       LR-grid mask, 1 = HR block
    packed_hr_positions  [L] bool                   True at HR-token positions
                                                    in the packed sequence
    packed_key_positions [L] bool                   key-subsampling mask for
                                                    CRPA attention
    latent_pixel_mask    [1, 1, T_lat, H_lat, W_lat]  full-latent-res mask
    decode_blend_mask    [1, 1, T, H, W]           Gaussian-blurred mask for
                                                    decode-time HR/LR blending

The builder is pipeline-independent — it takes the relevant division factors
as explicit args rather than a pipe handle, so it can be exercised in unit
tests without loading a model.
"""

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from .paths import circle_mask


class FoveationState(NamedTuple):
    """All mask tensors the foveated Wan DiT + pipeline need at one timestep.

    Named for readability over the legacy 6-tuple; field order matches the
    `video-gen` ordering so positional unpacking still works if needed.
    """
    hr_grid_mask: torch.Tensor
    lr_grid_mask: torch.Tensor
    packed_hr_positions: torch.Tensor
    packed_key_positions: torch.Tensor
    latent_pixel_mask: torch.Tensor
    decode_blend_mask: torch.Tensor


def pack_resolution_masks(hr_grid_mask: torch.Tensor, device: torch.device):
    """Pack a [1, 1, T, h*2, w*2] HR-grid mask into per-token bool masks.

    Returns `(packed_hr_positions, packed_key_positions)`:

      - `packed_hr_positions[i]` is True iff sequence position `i` corresponds
        to an HR token in the packed mixed-resolution sequence (HR blocks
        contribute 4 tokens; LR blocks contribute 1).
      - `packed_key_positions` is the key-subsampling mask used by CRPA
        attention: True at the LR token and at the top-left of every HR block.
    """
    _, _, frame, height, width = hr_grid_mask.shape
    h_2, w_2 = height // 2, width // 2

    mask_blocks = (
        hr_grid_mask.view(frame, h_2, 2, w_2, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(frame, h_2, w_2, 4)
    )
    is_hr_block = mask_blocks.sum(dim=-1) > 0
    is_lr_block = ~is_hr_block

    res_grid = torch.ones(frame, height, width, device=device)
    res_blocks = (
        res_grid.view(frame, h_2, 2, w_2, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(frame, h_2, w_2, 4)
    )
    out_res = res_blocks.clone()
    out_res[is_lr_block, :] = 0.0

    valid = torch.ones(frame, h_2, w_2, 4, device=device, dtype=torch.bool)
    valid[is_lr_block, 1:] = False
    flat_valid = valid.reshape(-1)

    packed_hr_positions = out_res.reshape(-1)[flat_valid].bool()

    tl_blocks = torch.zeros_like(res_blocks)
    tl_blocks[..., 0] = 1.0
    tl_blocks[is_lr_block, :] = 1.0
    packed_key_positions = tl_blocks.reshape(-1)[flat_valid].bool()
    return packed_hr_positions, packed_key_positions


def _gaussian_blur_2d(mask, sigma, device, dtype):
    k = max(3, int(math.ceil(3 * sigma)) * 2 + 1)
    if k % 2 == 0:
        k += 1
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2 + 1e-6))
    g = g / g.sum()
    pad = k // 2
    out = F.conv2d(mask, g.view(1, 1, k, 1), padding=(pad, 0))
    out = F.conv2d(out, g.view(1, 1, 1, k), padding=(0, pad))
    return out


def build_decode_blend_mask(
    centers, radii, num_frames, height, width, sigma, device,
    dtype=torch.float32,
):
    """[1, 1, T, H, W] per-frame Gaussian-blurred mask for decode-time HR/LR blending.

    Centers and radii are linearly interpolated from the latent-frame
    keyframes to all `num_frames` decode frames.
    """
    n_keys = len(centers)
    t_keys = np.linspace(0, 1, n_keys)
    t_frames = np.linspace(0, 1, num_frames)
    cx = np.interp(t_frames, t_keys, [c[0] for c in centers])
    cy = np.interp(t_frames, t_keys, [c[1] for c in centers])
    r  = np.interp(t_frames, t_keys, radii)
    masks = []
    for i in range(num_frames):
        m = circle_mask(height, width, (float(cx[i]), float(cy[i])), float(r[i]), device, dtype)
        m = _gaussian_blur_2d(m.unsqueeze(0).unsqueeze(0), sigma, device, dtype)
        masks.append(m.squeeze(0))  # [1, H, W]
    return torch.stack(masks, dim=1).unsqueeze(0)  # [1, 1, T, H, W]


def build_state(
    centers, radii, height, width, num_frames,
    device, dtype=torch.float32,
    height_division_factor: int = 16,
    width_division_factor: int = 16,
    time_division_factor: int = 4,
    vae_upsampling_factor: int = 8,
):
    """Convert a per-latent-frame foveation path to a `FoveationState`.

    `len(centers) == len(radii)` must equal the latent temporal length,
    `(num_frames - 1) // time_division_factor + 1`. The training-time random
    sampler and the inference-time spline sampler both produce paths at this
    resolution.
    """
    h = height // height_division_factor // 2
    w = width // width_division_factor // 2
    length = (num_frames - 1) // time_division_factor + 1
    if len(centers) != length or len(radii) != length:
        raise ValueError(
            f"expected path of length {length}, got centers={len(centers)} radii={len(radii)}"
        )

    masks_lr = []
    for i in range(length):
        m = circle_mask(h, w, centers[i], radii[i], device, dtype)
        masks_lr.append(m.unsqueeze(0).unsqueeze(0).unsqueeze(0))
    lr_grid_mask = torch.cat(masks_lr, dim=2)                  # [1,1,T,h,w]
    hr_grid_mask = F.interpolate(
        lr_grid_mask, size=(length, h * 2, w * 2), mode="nearest",
    )                                                           # [1,1,T,h*2,w*2]
    latent_pixel_mask = F.interpolate(
        hr_grid_mask,
        size=(length, height // vae_upsampling_factor, width // vae_upsampling_factor),
        mode="nearest",
    )

    packed_hr_positions, packed_key_positions = pack_resolution_masks(hr_grid_mask, device)

    # Invariants from video-gen — confirm the token streams are consistent.
    assert packed_hr_positions.sum() == hr_grid_mask.sum()
    assert packed_hr_positions.shape[0] == (hr_grid_mask.sum() + (1 - lr_grid_mask).sum())
    assert (latent_pixel_mask.sum() + (1 - hr_grid_mask).sum()) \
        == (hr_grid_mask.sum() + (1 - lr_grid_mask).sum()) * 4

    sigma = vae_upsampling_factor * 2
    decode_blend_mask = build_decode_blend_mask(
        centers, radii, num_frames, height, width, sigma, device, dtype,
    )

    return FoveationState(
        hr_grid_mask=hr_grid_mask,
        lr_grid_mask=lr_grid_mask,
        packed_hr_positions=packed_hr_positions,
        packed_key_positions=packed_key_positions,
        latent_pixel_mask=latent_pixel_mask,
        decode_blend_mask=decode_blend_mask,
    )
