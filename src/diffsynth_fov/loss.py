"""Foveated flow-matching SFT loss.

Ported from the fork's `diffsynth/diffusion/loss.py`.
"""

import torch

from diffsynth.diffusion.base_pipeline import BasePipeline


def _create_random_foveation_mask(h, w, device, dtype=torch.float32,
                                  center_range=(-0.3, 0.3), r_range=(0.2, 0.5)):
    """Random circular foveation mask, shape [h, w], 1 = HR, 0 = LR."""
    cx = (torch.rand(1, device=device).item() * (center_range[1] - center_range[0])
          + center_range[0] + 0.5) * w
    cy = (torch.rand(1, device=device).item() * (center_range[1] - center_range[0])
          + center_range[0] + 0.5) * h
    r = torch.rand(1, device=device).item() * (r_range[1] - r_range[0]) + r_range[0]
    diagonal = (h ** 2 + w ** 2) ** 0.5
    radius_px = r * (diagonal / 2.0)
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return (dist_sq <= radius_px ** 2).to(device)


def _create_fixed_foveation_mask(h, w, device, dtype=torch.float32,
                                 center=(0.0, 0.0), r=0.5):
    """Fixed circular foveation mask with same geometry as the random variant."""
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h
    diagonal = (h ** 2 + w ** 2) ** 0.5
    radius_px = r * (diagonal / 2.0)
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return (dist_sq <= radius_px ** 2).to(device)


def _resolve_foveation_mask(pipe: BasePipeline, inputs: dict, h: int, w: int):
    """Resolve a token-grid foveation mask based on `foveated_training_mode`.

    Modes:
      - `fixed`: centered, r=0.5
      - `random`: sampled each step
      - `saliency` / `bbox`: use `inputs["foveation_mask"]` (e.g. precomputed from a
        saliency map or bounding boxes)
    """
    mode = inputs.get("foveated_training_mode", "random")
    if mode == "fixed":
        return _create_fixed_foveation_mask(h, w, pipe.device, pipe.torch_dtype,
                                            center=(0.0, 0.0), r=0.5)
    if mode == "random":
        return _create_random_foveation_mask(h, w, pipe.device, pipe.torch_dtype)
    if mode in ("saliency", "bbox"):
        mask = inputs.get("foveation_mask")
        if mask is None:
            raise ValueError(
                f"foveated_training_mode='{mode}' but inputs['foveation_mask'] is None."
            )
        mask = mask.to(device=pipe.device, dtype=pipe.torch_dtype)
        while mask.dim() > 2:
            mask = mask.squeeze(0)
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest",
            ).squeeze(0).squeeze(0)
        return mask
    return _create_random_foveation_mask(h, w, pipe.device, pipe.torch_dtype)


def FoveatedFlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    """Foveated Flow Matching SFT objective.

    For each step, samples a random foveation mask, constructs a mixed-resolution
    training target (HR tokens inside the foveal region, LR tokens averaged into
    one token per `lr_factor x lr_factor` block in the periphery), and computes
    MSE against the foveated noise prediction returned by
    `pipe.foveated_training_forward(...)`.
    """
    max_b = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_b = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_b, max_b, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype,
                                                        device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    noise_downsampled = torch.randn_like(inputs["input_latents_downsampled"])
    inputs["latents_downsampled"] = pipe.scheduler.add_noise(
        inputs["input_latents_downsampled"], noise_downsampled, timestep,
    )
    training_target_downsampled = pipe.scheduler.training_target(
        inputs["input_latents_downsampled"], noise_downsampled, timestep,
    )

    has_foveated_forward = hasattr(pipe, "foveated_training_forward") and callable(
        getattr(pipe, "foveated_training_forward", None)
    )

    if has_foveated_forward:
        height = inputs["height"]
        width = inputs["width"]
        batch_size, _, channels = inputs["latents"].shape
        h, w = height // 16, width // 16
        foveation_mask = _resolve_foveation_mask(pipe, inputs, h, w)

        lr_factor = inputs.get("lr_downsample_factor", 2)
        n_per_block = lr_factor * lr_factor
        h_d, w_d = h // lr_factor, w // lr_factor

        # Build the mixed-resolution training target.
        training_target_merged = training_target.view(batch_size, h, w, channels)
        training_target_downsampled = training_target_downsampled.view(
            batch_size, h_d, w_d, channels,
        )
        training_target_merged = training_target_merged.view(
            batch_size, h_d, lr_factor, w_d, lr_factor, channels,
        )
        training_target_merged = training_target_merged.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size, h_d, w_d, n_per_block, channels,
        )

        mask_blocks = foveation_mask.view(h_d, lr_factor, w_d, lr_factor)
        mask_blocks = mask_blocks.permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)
        is_high_res_block = mask_blocks.sum(dim=-1) > 0
        is_low_res_block = ~is_high_res_block

        training_target_merged[:, is_low_res_block, 0, :] = (
            training_target_downsampled[:, is_low_res_block, :].to(training_target_merged.dtype)
        )
        valid_tokens = torch.ones(
            h_d, w_d, n_per_block, device=training_target_merged.device, dtype=torch.bool,
        )
        valid_tokens[is_low_res_block, 1:] = False

        training_target_merged = training_target_merged.view(batch_size, -1, channels)
        valid_tokens = valid_tokens.view(-1)
        training_target = training_target_merged[:, valid_tokens, :]

        inputs["foveation_mask"] = foveation_mask
        prediction_type = inputs.get("prediction_type", "clean")
        noise_pred = pipe.foveated_training_forward(
            inputs, timestep, timestep_id, prediction_type,
            lr_downsample_factor=lr_factor,
        )
    else:
        # Fall back to standard flow matching if the pipeline doesn't support foveation.
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss
