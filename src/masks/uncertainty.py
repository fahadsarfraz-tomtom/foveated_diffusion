"""Denoising-uncertainty foveation signals (FGD-019 / D6).

Monte-Carlo denoising uncertainty from a frozen FLUX2 pipeline: noise a real
image's latents at a chosen timestep with K independent draws, run one model
forward per draw, reconstruct the one-step clean estimate x1_hat, and take the
per-token variance across draws. Patch Forcing (Schusterbauer et al., CVPR
2026) validated this MC variance as a proxy for the learned per-patch
difficulty head, so the map can be produced with zero training.

The scheduler/model conventions (what add_noise mixes, what the network
regresses) are probed numerically instead of assumed: both are affine in
(clean, noise), so unit probes recover the coefficients exactly and the
one-step inversion works for any linear flow/diffusion convention.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .adaptive import FoveationPlan, token_ratio_from_mask


# ---------------------------------------------------------------------------
# Scheduler/model convention probing
# ---------------------------------------------------------------------------

def resolve_affine_coeffs(scheduler, timestep) -> tuple[float, float, float, float]:
    """Return (a, b, c, d) with x_t = a*x1 + b*x0 and target = c*x1 + d*x0.

    x1 = clean latents, x0 = noise. Probed with unit tensors so any linear
    scheduler convention is handled without assumptions.
    """
    ones = torch.ones(1, 4, 2)
    zeros = torch.zeros(1, 4, 2)
    ts = timestep.detach().float().cpu().reshape(-1)[:1]
    a = float(scheduler.add_noise(ones, zeros, ts).float().mean())
    b = float(scheduler.add_noise(zeros, ones, ts).float().mean())
    c = float(scheduler.training_target(ones, zeros, ts).float().mean())
    d = float(scheduler.training_target(zeros, ones, ts).float().mean())
    return a, b, c, d


def solve_x1(x_t: torch.Tensor, pred: torch.Tensor, coeffs) -> torch.Tensor:
    """Invert the affine system for the one-step clean estimate x1_hat.

    From x_t = a*x1 + b*x0 and pred ~= c*x1 + d*x0:
        x1_hat = (d*x_t - b*pred) / (d*a - b*c)
    Falls back to the raw prediction when the system is degenerate (then the
    model regresses x1 directly, or the variance of `pred` is the signal).
    """
    a, b, c, d = coeffs
    det = d * a - b * c
    if abs(det) < 1e-6:
        return pred
    return (d * x_t - b * pred) / det


# ---------------------------------------------------------------------------
# Pipeline plumbing for a single (batched) forward pass
# ---------------------------------------------------------------------------

def prepare_conditioning(pipe, prompt: str, height: int, width: int, seed: int = 0) -> dict:
    """Run the pipeline units with no input image to obtain text conditioning,
    full-resolution image ids, and the rest of the model_fn inputs."""
    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": ""}
    inputs_shared = {
        "cfg_scale": 1.0,
        "embedded_guidance": 1.0,
        "input_image": None,
        "denoising_strength": 1.0,
        "height": height,
        "width": width,
        "seed": seed,
        "rand_device": str(pipe.device),
        "num_inference_steps": 1,
        "foveation_mask": None,
        "soft_foveation_blend": False,
        "lr_downsample_factor": 2,
        "prediction_type": "clean",
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega,
        )
    merged = dict(inputs_shared)
    merged.update(inputs_posi)
    return merged


def encode_image_latents(pipe, image) -> torch.Tensor:
    """VAE-encode a PIL image to sequence latents [1, L, C] (the training layout)."""
    pipe.load_models_to_device(["vae"])
    tensor = pipe.preprocess_image(image)
    latents = pipe.vae.encode(tensor)
    return latents.flatten(2).transpose(1, 2).contiguous()


def _expand_batch(inputs: dict, k: int) -> dict:
    """Expand every leading-1 tensor in `inputs` to batch size k."""
    out = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == 1:
            out[key] = value.expand(k, *value.shape[1:])
        else:
            out[key] = value
    return out


@torch.no_grad()
def mc_uncertainty_map(
    pipe,
    image,
    prompt: str,
    timestep_frac: float = 0.5,
    num_samples: int = 8,
    seed: int = 0,
    conditioning: Optional[dict] = None,
) -> tuple[torch.Tensor, dict]:
    """MC denoising-uncertainty map for one image.

    Returns (map [h, w] float32 on CPU, extras). h = height//16 token grid.
    `conditioning` can be passed to reuse unit outputs across timestep fracs.
    """
    width, height = image.size
    h, w = height // 16, width // 16

    try:
        pipe.scheduler.set_timesteps(1000, training=True, dynamic_shift_len=h * w)
    except TypeError:
        pipe.scheduler.set_timesteps(1000, training=True)
    timesteps = pipe.scheduler.timesteps
    t_idx = min(int(timestep_frac * (len(timesteps) - 1)), len(timesteps) - 1)
    timestep = timesteps[t_idx].reshape(1).to(dtype=pipe.torch_dtype, device=pipe.device)

    if conditioning is None:
        conditioning = prepare_conditioning(pipe, prompt, height, width, seed=seed)

    clean = encode_image_latents(pipe, image).to(dtype=pipe.torch_dtype, device=pipe.device)
    if clean.shape[1] != h * w:
        raise ValueError(
            f"latent length {clean.shape[1]} != token grid {h}x{w} — "
            "image size must be a multiple of 16"
        )

    coeffs = resolve_affine_coeffs(pipe.scheduler, timesteps[t_idx])

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (num_samples, *clean.shape[1:]), generator=generator, dtype=torch.float32,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    x_t = pipe.scheduler.add_noise(
        clean.expand(num_samples, -1, -1), noise, timestep.expand(num_samples),
    )

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    inputs = _expand_batch(conditioning, num_samples)
    inputs["latents"] = x_t
    inputs["resolution_mask"] = None
    inputs["resolution_mask_top_left"] = None

    pred = pipe.model_fn(**models, **inputs, timestep=timestep.expand(num_samples))
    pred = pred[:, : clean.shape[1], :]

    x1_hat = solve_x1(x_t.float(), pred.float(), coeffs)
    variance = x1_hat.var(dim=0, unbiased=True).mean(dim=-1)  # [L]
    umap = variance.reshape(h, w).float().cpu()

    extras = {
        "timestep": float(timesteps[t_idx]),
        "timestep_index": int(t_idx),
        "timestep_frac": timestep_frac,
        "coeffs": coeffs,
        "num_samples": num_samples,
        "x1_noise_latent": x_t[:1].float().cpu(),  # one noisy latent, reusable by FPM probes
        "conditioning": conditioning,
    }
    return umap, extras


# ---------------------------------------------------------------------------
# Maps -> masks / centers / plans
# ---------------------------------------------------------------------------

def dense_mask_from_map(umap: torch.Tensor, beta: float) -> torch.Tensor:
    """Binary token mask selecting the top-`beta` fraction of the map."""
    beta = min(max(float(beta), 1e-4), 1.0)
    flat = umap.flatten().float()
    k = max(1, int(round(beta * flat.numel())))
    threshold = torch.topk(flat, k).values.min()
    return (umap >= threshold).float()


def upsample_mask_full_res(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(
        mask[None, None].float(), size=(height, width), mode="nearest",
    )[0, 0]


def centers_from_map(
    umap: torch.Tensor, num_centers: int = 3, min_dist_frac: float = 0.2,
) -> list[tuple[float, float]]:
    """Greedy peak picking -> centers in Chao's [-0.5, 0.5] (x, y) frame."""
    h, w = umap.shape
    work = umap.clone().float()
    min_dist = max(1.0, min_dist_frac * math.sqrt(h * w))
    centers = []
    for _ in range(num_centers):
        idx = int(torch.argmax(work).item())
        cy, cx = divmod(idx, w)
        centers.append((cx / max(w - 1, 1) - 0.5, cy / max(h - 1, 1) - 0.5))
        yy = torch.arange(h, dtype=torch.float32)[:, None]
        xx = torch.arange(w, dtype=torch.float32)[None, :]
        work[((yy - cy) ** 2 + (xx - cx) ** 2) <= min_dist ** 2] = -float("inf")
        if torch.isinf(work).all():
            break
    return centers


def plan_from_dense_map(
    umap: torch.Tensor,
    beta: float,
    height: int,
    width: int,
    device,
    lr_factor: int = 2,
    policy_name: str = "uncertainty",
    metadata_extra: Optional[dict] = None,
) -> FoveationPlan:
    """Build a Chao-compatible FoveationPlan from a dense uncertainty map."""
    token_mask = dense_mask_from_map(umap, beta).to(device)
    full_res_mask = upsample_mask_full_res(token_mask, height, width).to(device)
    metadata = {
        "policy": policy_name,
        "map_min": float(umap.min()),
        "map_max": float(umap.max()),
        "map_mean": float(umap.mean()),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return FoveationPlan(
        token_mask=token_mask,
        full_res_mask=full_res_mask,
        centers=[],
        radii=[],
        beta=float(beta),
        hr_fraction=float(token_mask.float().mean()),
        token_ratio=token_ratio_from_mask(token_mask, lr_factor=lr_factor),
        policy=policy_name,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Comparison metrics (probe)
# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without the scipy dependency."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def iou_top_beta(a: np.ndarray, b: np.ndarray, beta: float = 0.25) -> float:
    """IoU of the top-`beta` regions of two maps."""
    a = torch.from_numpy(np.asarray(a, dtype=np.float32))
    b = torch.from_numpy(np.asarray(b, dtype=np.float32))
    ma = dense_mask_from_map(a, beta).bool()
    mb = dense_mask_from_map(b, beta).bool()
    union = (ma | mb).sum().item()
    return float((ma & mb).sum().item() / union) if union else 0.0
