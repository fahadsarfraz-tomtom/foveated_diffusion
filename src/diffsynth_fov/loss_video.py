"""Foveated flow-matching SFT loss for the Wan video pipeline.

Mirrors the upstream ``diffsynth.diffusion.FlowMatchSFTLoss`` structure but
runs the dual-track foveation forward and applies a region-split MSE:

  - HR track: MSE over the high-resolution noise prediction, masked to the
    foveal region (``foveation_state.latent_pixel_mask``).
  - LR track: MSE over the low-resolution noise prediction, masked to the
    peripheral region (``1 - foveation_state.hr_grid_mask``).

The default sampler is ``random_path`` (linear interpolation between two
sampled endpoints) — NOT per-frame independent random, per §4.1 of the paper.
This is the bug being fixed relative to ``video-gen/diffsynth/diffusion/loss.py``,
which defaulted to ``'random'``.

Saliency training (``foveation_from_video=True`` on the
``WanFoveatedInputVideoEmbedder``) populates ``inputs["foveation_state"]``
before the loss runs; in that case this function skips its own sampler.
"""

import torch
import torch.nn.functional as F

from diffsynth.diffusion.base_pipeline import BasePipeline

from ..masks.paths import sample_random_path
from ..masks.state import build_state


def FoveatedWanFlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    """Foveated flow-matching SFT objective for the Wan video pipeline.

    Expects ``inputs`` to contain (populated by ``WanFoveatedInputVideoEmbedder``):
        input_latents       : HR clean latents [B, C, T_lat, H_lat, W_lat]
        input_latents_lr    : LR clean latents [B, C, T_lat, H_lat/2, W_lat/2]
        height, width       : full pixel dimensions
        num_frames          : decode frame count
        foveation_state     : optional pre-built FoveationState (saliency path)

    Returns scalar loss tensor scaled by the scheduler's training weight.
    """
    n_steps = len(pipe.scheduler.timesteps)
    max_b = int(inputs.get("max_timestep_boundary", 1.0) * n_steps)
    min_b = int(inputs.get("min_timestep_boundary", 0.0) * n_steps)

    timestep_id = torch.randint(min_b, max_b, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(
        dtype=pipe.torch_dtype, device=pipe.device,
    )

    # HR track: noise + flow-matching target.
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    target_hr = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    # LR track: pooled noise with sqrt(N=4)=2 variance correction.
    noise_lr = F.interpolate(noise, scale_factor=(1, 0.5, 0.5), mode="trilinear") * 2
    inputs["latents_lr"] = pipe.scheduler.add_noise(
        inputs["input_latents_lr"], noise_lr, timestep,
    )
    target_lr = pipe.scheduler.training_target(
        inputs["input_latents_lr"], noise_lr, timestep,
    )

    # Foveation foveation_state: use the one the InputVideoEmbedder produced (saliency
    # path), otherwise sample a random_path for this step.
    foveation_state = inputs.get("foveation_state", None)
    if foveation_state is None:
        latent_length = (inputs["num_frames"] - 1) // 4 + 1
        centers, radii = sample_random_path(latent_length, device=pipe.device)
        foveation_state = build_state(
            centers, radii, inputs["height"], inputs["width"], inputs["num_frames"],
            device=pipe.device,
        )
        inputs["foveation_state"] = foveation_state

    # Foveated forward — returns (pred_hr, pred_lr).
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred_hr, noise_pred_lr = pipe.model_fn(
        **models, **inputs, timestep=timestep,
    )

    # Region-split MSE: HR loss only inside the foveal region, LR loss only
    # outside it. Normalisation per-region keeps the two terms comparable when
    # the foveal area varies (e.g. across random_path samples).
    mask_hr = foveation_state.latent_pixel_mask.to(dtype=noise_pred_hr.dtype, device=noise_pred_hr.device)
    mask_lr_grid = foveation_state.hr_grid_mask.to(dtype=noise_pred_lr.dtype, device=noise_pred_lr.device)

    loss_hr = F.mse_loss(noise_pred_hr.float(), target_hr.float(), reduction="none")
    loss_lr = F.mse_loss(noise_pred_lr.float(), target_lr.float(), reduction="none")

    loss = (loss_hr * mask_hr).sum() / mask_hr.sum().clamp_min(1) + \
           (loss_lr * (1 - mask_lr_grid)).sum() / (1 - mask_lr_grid).sum().clamp_min(1)
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
