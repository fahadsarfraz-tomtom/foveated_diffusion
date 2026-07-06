"""Experiment runners: maps `--experiment` value to a callable.

All runners take (pipe, args, output_dir) and write images / metadata under
`output_dir`. See README for the per-experiment output layout.
"""

import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from diffsynth.core import load_state_dict

from ..masks import (
    AdaptiveFoveationPolicy,
    adaptive_config_from_args,
    create_foveation_mask,
    create_foveation_mask_full_res,
    gaussian_blur_mask_2d,
    generate_foveation_trajectory_masks,
)
from .visualize import (
    create_tokenization_mask_vis,
    draw_foveation_outline,
    load_prompt_dataset,
)


# ---------------------------------------------------------------------------
# Common pipeline call
# ---------------------------------------------------------------------------

def _pipe_call(
    pipe,
    args,
    prompt: str,
    foveation_mask,
    full_res_foveation_mask=None,
    decode_mode: str = None,
    extra_kwargs: dict = None,
):
    """Single call to the foveated FLUX2 pipeline. Returns a PIL.Image."""
    kwargs = dict(
        prompt=prompt,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device="cuda",
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.guidance_scale,
        foveation_mask=foveation_mask,
        decode_mode=decode_mode if decode_mode is not None else args.decode_mode,
        prediction_type=args.prediction_type,
        soft_foveation_blend=getattr(args, "soft_foveation_blend", False),
        lr_downsample_factor=getattr(args, "lr_downsample_factor", 2),
    )
    if full_res_foveation_mask is not None:
        kwargs["full_res_foveation_mask"] = full_res_foveation_mask
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return pipe(**kwargs)


# ---------------------------------------------------------------------------
# Single-prompt generation: high_res / naive_mixed_res / ours
# ---------------------------------------------------------------------------

def _run_high_res_one(pipe, args, output_dir, prompt, prompt_idx, foveation_mask=None, full_res_foveation_mask=None):
    print(f"[high_res] prompt {prompt_idx:010d}: {prompt}")
    image = _pipe_call(pipe, args, prompt, foveation_mask=None, decode_mode="direct")
    image.save(os.path.join(output_dir, f"img_{prompt_idx:010d}.png"))


def _run_naive_mixed_res_one(pipe, args, output_dir, prompt, prompt_idx, foveation_mask, full_res_foveation_mask=None):
    print(f"[naive_mixed_res] prompt {prompt_idx:010d}: {prompt}")
    image = _pipe_call(pipe, args, prompt, foveation_mask, full_res_foveation_mask)
    image.save(os.path.join(output_dir, f"img_{prompt_idx:010d}.png"))


def _run_ours_one(pipe, args, output_dir, prompt, prompt_idx, foveation_mask, full_res_foveation_mask=None):
    print(f"[ours] prompt {prompt_idx:010d}: {prompt}")
    image = _pipe_call(pipe, args, prompt, foveation_mask, full_res_foveation_mask)
    image.save(os.path.join(output_dir, f"img_{prompt_idx:010d}.png"))


SINGLE_PROMPT_FUNCS = {
    "high_res": _run_high_res_one,
    "naive_mixed_res": _run_naive_mixed_res_one,
    "ours": _run_ours_one,
}


def _run_full_eval(single_prompt_func, pipe, args, output_dir, prompts):
    """Iterate over prompts; for naive/ours apply a default centered foveation mask."""
    default_mask = None
    full_res_mask = None
    if single_prompt_func is not _run_high_res_one:
        mask_shape = getattr(args, "full_eval_mask", "square")
        r = getattr(args, "mask_radius", 0.5)
        lr_factor = getattr(args, "lr_downsample_factor", 2)
        default_mask = create_foveation_mask(
            args.height, args.width, (0, 0), r, mask_shape, pipe.device, lr_factor=lr_factor,
        )
        full_res_mask = create_foveation_mask_full_res(
            args.height, args.width, (0, 0), r, mask_shape, pipe.device,
        )

    prompt_list, image_names = [], []
    subset_idx = getattr(args, "subset_idx", 0)
    num_subsets = max(1, getattr(args, "num_subsets", 1))
    for global_idx, prompt in enumerate(prompts):
        if args.num_prompts is not None and global_idx >= args.num_prompts:
            break
        if (global_idx % num_subsets) != subset_idx:
            continue
        single_prompt_func(
            pipe, args, output_dir, prompt, global_idx,
            foveation_mask=default_mask, full_res_foveation_mask=full_res_mask,
        )
        prompt_list.append(prompt)
        image_names.append(f"img_{global_idx:010d}.png")

    pd.DataFrame({"image": image_names, "prompt": prompt_list}).to_csv(
        os.path.join(output_dir, f"metadata_{subset_idx:05d}.csv"), index=False,
    )


def _resolve_prompts(args):
    if args.full_eval:
        assert args.prompt_dataset_path is not None, "--prompt_dataset_path required for --full_eval"
        prompts = load_prompt_dataset(args.prompt_dataset_path)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_dataset_path}")
        return prompts
    return [args.prompt]


def run_single_prompt_experiment(pipe, args, output_dir):
    """Dispatcher for high_res / naive_mixed_res / ours (each uses _run_full_eval)."""
    prompts = _resolve_prompts(args)
    _run_full_eval(SINGLE_PROMPT_FUNCS[args.experiment], pipe, args, output_dir, prompts)


def run_adaptive_policy_experiment(pipe, args, output_dir):
    """Run our adaptive policy on top of Chao's static image foveation interface.

    The current FLUX image pipeline accepts one mask per generation and packs the
    mixed-resolution tokens before denoising. For NaFo we therefore project the
    time-varying budget to a single beta via ``--adaptive_beta_mode``.
    """
    prompts = _resolve_prompts(args)
    policy = AdaptiveFoveationPolicy(adaptive_config_from_args(args))
    lr_factor = getattr(args, "lr_downsample_factor", 2)
    subset_idx = getattr(args, "subset_idx", 0)
    num_subsets = max(1, getattr(args, "num_subsets", 1))

    mask_dir = os.path.join(output_dir, "adaptive_masks")
    os.makedirs(mask_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"adaptive_metadata_{subset_idx:05d}.jsonl")
    csv_rows = []

    with open(jsonl_path, "w") as meta_f:
        for global_idx, prompt in enumerate(prompts):
            if args.num_prompts is not None and global_idx >= args.num_prompts:
                break
            if (global_idx % num_subsets) != subset_idx:
                continue

            plan = policy.plan_image(
                prompt=prompt,
                height=args.height,
                width=args.width,
                device=pipe.device,
                num_inference_steps=args.num_inference_steps,
                lr_factor=lr_factor,
            )
            image_name = f"img_{global_idx:010d}.png"
            mask_name = f"mask_{global_idx:010d}.png"
            mask_path = os.path.join(mask_dir, mask_name)
            _save_mask_image(plan.full_res_mask, pipe, args, mask_path)

            print(
                f"[ours_adaptive] prompt {global_idx:010d}: "
                f"policy={plan.policy} beta={plan.beta:.3f} "
                f"HR={plan.hr_fraction:.3f} tokens={plan.token_ratio:.3f}"
            )
            image = _pipe_call(
                pipe,
                args,
                prompt,
                plan.token_mask,
                full_res_foveation_mask=plan.full_res_mask,
            )
            image.save(os.path.join(output_dir, image_name))

            metadata = plan.to_jsonable()
            metadata.update({
                "image": image_name,
                "prompt": prompt,
                "mask": os.path.relpath(mask_path, output_dir),
            })
            meta_f.write(json.dumps(metadata) + "\n")
            csv_rows.append({
                "image": image_name,
                "prompt": prompt,
                "mask": os.path.relpath(mask_path, output_dir),
                "policy": plan.policy,
                "beta": plan.beta,
                "hr_fraction": plan.hr_fraction,
                "token_ratio": plan.token_ratio,
                "centers": json.dumps(metadata["centers"]),
                "radii": json.dumps(metadata["radii"]),
            })

    pd.DataFrame(csv_rows).to_csv(
        os.path.join(output_dir, f"metadata_{subset_idx:05d}.csv"), index=False,
    )


# ---------------------------------------------------------------------------
# Circular trajectory (single prompt, mask center orbits)
# ---------------------------------------------------------------------------

def run_circular_traj(pipe, args, output_dir):
    print(f"[circular_traj] {args.num_frames} frames, orbit={args.orbit_radius}, r={args.mask_radius}")
    lr_factor = getattr(args, "lr_downsample_factor", 2)
    for i in range(args.num_frames):
        angle = 2 * math.pi * (i / args.num_frames)
        center = (args.orbit_radius * math.cos(angle), args.orbit_radius * math.sin(angle))
        foveation_mask = create_foveation_mask(
            args.height, args.width, center, args.mask_radius, args.mask_shape, pipe.device, lr_factor,
        )
        full_res_mask = create_foveation_mask_full_res(
            args.height, args.width, center, args.mask_radius, args.mask_shape, pipe.device,
        )
        _save_mask_image(full_res_mask, pipe, args, os.path.join(output_dir, f"foveation_mask_{i:03d}.png"))

        torch.cuda.empty_cache()
        t0 = time.time()
        image = _pipe_call(pipe, args, args.prompt, foveation_mask, full_res_mask)
        print(f"  step {i:02d}/{args.num_frames}  angle={math.degrees(angle):.1f}deg  time={time.time() - t0:.2f}s")
        image.save(os.path.join(output_dir, f"img_{i:03d}.png"))


# ---------------------------------------------------------------------------
# Vary radius (single prompt, sweep mask radius from 0.1 -> 1.0)
# ---------------------------------------------------------------------------

def run_vary_radius(pipe, args, output_dir):
    print("[vary_radius]")
    lr_factor = getattr(args, "lr_downsample_factor", 2)
    for r_idx, r in enumerate(np.linspace(0.1, 1.0, args.num_frames)):
        tag = f"{r_idx:02d}"
        foveation_mask = create_foveation_mask(
            args.height, args.width, (0, 0), r, args.mask_shape, pipe.device, lr_factor,
        )
        full_res_mask = create_foveation_mask_full_res(
            args.height, args.width, (0, 0), r, args.mask_shape, pipe.device,
        )
        _save_mask_image(full_res_mask, pipe, args, os.path.join(output_dir, f"foveation_mask_{tag}.png"))

        torch.cuda.empty_cache()
        t0 = time.time()
        image = _pipe_call(pipe, args, args.prompt, foveation_mask, full_res_mask)
        print(f"  r={r:.2f}  time={time.time() - t0:.2f}s")
        image.save(os.path.join(output_dir, f"img_{tag}.png"))


# ---------------------------------------------------------------------------
# Runtime benchmark
# ---------------------------------------------------------------------------

def run_runtime_experiments(pipe, args, output_dir):
    print("[runtime] benchmarking foveation radius vs runtime")
    lr_factor = getattr(args, "lr_downsample_factor", 2)
    args.mask_shape = "square"

    r_list = np.arange(0, 64 // lr_factor + 1)[::-1]
    r_list = np.concatenate([[r_list[0]], r_list])

    runtime_list, token_ratio_list = [], []
    for r_idx, r in enumerate(r_list):
        # Build a square HR region with side `r` in low-res latent units, then upsample to token grid.
        foveation_mask = torch.zeros(
            args.height // 16 // lr_factor, args.width // 16 // lr_factor,
            device=pipe.device, dtype=torch.float32,
        )
        foveation_mask[:r, :r] = 1.0
        foveation_mask = F.interpolate(
            foveation_mask.unsqueeze(0).unsqueeze(0),
            size=(args.height // 16, args.width // 16),
            mode="nearest",
        ).squeeze(0).squeeze(0)

        total_orig_tokens = foveation_mask.shape[-2] * foveation_mask.shape[-1]
        num_hr = foveation_mask.sum()
        num_lr = (total_orig_tokens - num_hr) // (lr_factor ** 2)
        token_ratio = float(((num_hr + num_lr) / total_orig_tokens).item())
        token_ratio_list.append(token_ratio)
        print(f"  r={r}  HR={int(num_hr)}  LR={int(num_lr)}  ratio={token_ratio:.3f}")

        num_repeats = 3
        runtimes, saved_image = [], None
        for rep in range(num_repeats):
            torch.cuda.empty_cache()
            image = _pipe_call(pipe, args, args.prompt, foveation_mask)
            runtimes.append(pipe.vit_timing)
            if rep == 0:
                saved_image = image
        avg = float(np.mean(runtimes))
        print(f"  -> avg vit_time={avg:.2f}s")
        runtime_list.append(avg)
        if saved_image is not None:
            saved_image.save(os.path.join(output_dir, f"img_{r_idx:02d}.png"))

    np.savez(
        os.path.join(output_dir, "runtime_and_radius_list.npz"),
        runtime_list=np.array(runtime_list),
        r_list=np.array(r_list),
        token_ratio_list=np.array(token_ratio_list),
    )


# ---------------------------------------------------------------------------
# Foveation trajectory grid (N prompts x M masks)
# ---------------------------------------------------------------------------

def run_foveation_trajectory_grid(pipe, args, output_dir):
    """N prompts (rows) x M trajectory masks (cols). Same trajectory for every prompt
    except `random_circular`, which samples a fresh mask per prompt.
    """
    import imageio

    if args.prompt_dataset_path is not None:
        all_prompts = load_prompt_dataset(args.prompt_dataset_path)
    else:
        all_prompts = [args.prompt]
        args.num_prompts = 1

    prompt_ids = getattr(args, "prompt_ids", None)
    if prompt_ids is not None:
        if not prompt_ids:
            raise ValueError("--prompt_ids must contain at least one index")
        max_idx = len(all_prompts) - 1
        bad = [i for i in prompt_ids if i < 0 or i > max_idx]
        if bad:
            raise ValueError(f"--prompt_ids contains invalid indices {bad}; valid range [0, {max_idx}]")
        prompts = [all_prompts[i] for i in prompt_ids]
    else:
        assert args.num_prompts is not None and args.num_prompts > 0, "--num_prompts required when --prompt_ids not given"
        prompts = all_prompts[: args.num_prompts]
    print(f"[foveation_trajectory_grid] {len(prompts)} prompts")

    outline_width_frac = getattr(args, "outline_width_frac", 0.01)
    outline_color = getattr(args, "outline_color", (255, 0, 0))
    if isinstance(outline_color, str):
        outline_color = tuple(int(x) for x in outline_color.split(","))

    height, width, device = args.height, args.width, pipe.device
    lr_factor = getattr(args, "lr_downsample_factor", 2)
    traj_type = getattr(args, "foveation_trajectory_type", "circular")
    per_prompt_random_mask = (traj_type == "random_circular")

    if not per_prompt_random_mask:
        foveation_masks, full_res_foveation_masks = generate_foveation_trajectory_masks(
            height, width, args, device, lr_factor=lr_factor,
        )
        num_cols = len(foveation_masks)
        print(f"  trajectory={traj_type}, {num_cols} masks (shared across prompts)")

        tok_dir = os.path.join(output_dir, "tokenization_masks")
        os.makedirs(tok_dir, exist_ok=True)
        for col, m in enumerate(foveation_masks):
            tok_vis = create_tokenization_mask_vis(m, height, width, lr_factor=lr_factor)
            imageio.imwrite(os.path.join(tok_dir, f"tokenization_mask_{col:04d}.png"), tok_vis)
    else:
        rng = np.random.default_rng(getattr(args, "seed", 0))
        print(f"  trajectory={traj_type}, 1 fresh mask per prompt")

    for prompt_idx, prompt in enumerate(prompts):
        folder_id = prompt_ids[prompt_idx] if prompt_ids is not None else prompt_idx
        prompt_dir = os.path.join(output_dir, f"{folder_id:03d}")
        os.makedirs(prompt_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        if per_prompt_random_mask:
            orbit_radius = getattr(args, "orbit_radius", 0.25)
            mask_radius = getattr(args, "mask_radius", 0.30)
            angle = float(rng.uniform(0, 2 * math.pi))
            center = (orbit_radius * math.cos(angle), orbit_radius * math.sin(angle))
            shape = getattr(args, "mask_shape", "circular")
            foveation_masks = [create_foveation_mask(height, width, center, mask_radius, shape, device, lr_factor)]
            full_res_foveation_masks = [create_foveation_mask_full_res(height, width, center, mask_radius, shape, device)]

        for col, foveation_mask in enumerate(foveation_masks):
            foveation_mask_upsampled = F.interpolate(
                foveation_mask.unsqueeze(0).unsqueeze(0),
                size=(height, width), mode="nearest",
            ).squeeze(0).squeeze(0)

            torch.cuda.empty_cache()
            image = _pipe_call(
                pipe, args, prompt,
                foveation_mask, full_res_foveation_mask=foveation_mask_upsampled,
            )

            image_np = np.array(image)
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            if getattr(args, "foveation_outline", True):
                draw_foveation_outline(
                    image_np, full_res_foveation_masks[col], height, width,
                    outline_width_frac=outline_width_frac, color=outline_color,
                )

            imageio.imwrite(os.path.join(prompt_dir, f"img_{col:04d}.png"), image_np)
            _save_mask_image(
                full_res_foveation_masks[col], pipe, args,
                os.path.join(prompt_dir, f"mask_{col:04d}.png"),
            )
            print(f"  saved {folder_id:03d}/img_{col:04d}.png")


# ---------------------------------------------------------------------------
# User study (triplets: high_res, naive, ours per prompt)
# ---------------------------------------------------------------------------

def run_user_study(pipe, args, output_dir):
    """For each sampled prompt, produce (high_res, naive, ours) with the same seed and
    a random foveation mask. LoRA / DiT checkpoint is loaded only for the 'ours' pass.
    """
    import imageio

    assert args.prompt_dataset_path, "--prompt_dataset_path required for user_study"
    num_prompts = getattr(args, "num_prompts", None) or 5
    all_prompts = load_prompt_dataset(args.prompt_dataset_path)
    k = min(num_prompts, len(all_prompts))
    prompts = all_prompts if num_prompts == len(all_prompts) else random.sample(all_prompts, k)
    print(f"[user_study] {len(prompts)} prompts  seed={args.seed}")

    height, width, device = args.height, args.width, pipe.device
    lr_factor = getattr(args, "lr_downsample_factor", 2)
    outline_width_frac = getattr(args, "outline_width_frac", 0.01)
    outline_color = getattr(args, "outline_color", (255, 0, 0))
    if isinstance(outline_color, str):
        outline_color = tuple(int(x) for x in outline_color.split(","))

    pipe.clear_lora(verbose=0)
    items = []

    for idx, prompt in enumerate(prompts):
        subdir = os.path.join(output_dir, f"{idx:05d}")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "prompt.txt"), "w") as f:
            f.write(prompt)

        center = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
        r, shape = 0.33, "circular"
        foveation_mask = create_foveation_mask(height, width, center, r, shape, device, lr_factor)
        full_res_mask = create_foveation_mask_full_res(height, width, center, r, shape, device)
        items.append((subdir, prompt, foveation_mask, full_res_mask, center))

        # 1) high-res baseline
        _run_high_res_one(pipe, args, subdir, prompt, 0)
        os.rename(os.path.join(subdir, "img_0000000000.png"), os.path.join(subdir, "img_high_res.png"))

        # 2) naive mixed-resolution (base DiT)
        _run_naive_mixed_res_one(pipe, args, subdir, prompt, 0, foveation_mask, full_res_mask)
        os.rename(os.path.join(subdir, "img_0000000000.png"), os.path.join(subdir, "img_naive.png"))
        if getattr(args, "foveation_outline", True):
            _overlay_outline(os.path.join(subdir, "img_naive.png"), full_res_mask,
                             height, width, outline_width_frac, outline_color)

        _save_mask_image(full_res_mask, pipe, args, os.path.join(subdir, "mask.png"))
        _save_fixation_dot(center, height, width, os.path.join(subdir, "fixation_point.png"))
        print(f"  saved {idx:05d}/ (high_res, naive, mask, fixation)")

    # Swap to LoRA / DiT checkpoint for the "ours" pass
    if args.lora_checkpoint is not None:
        pipe.load_lora(pipe.dit, args.lora_checkpoint)
        print(f"Loaded LoRA checkpoint from {args.lora_checkpoint}")
    if args.dit_checkpoint is not None:
        state_dict = load_state_dict(args.dit_checkpoint, torch_dtype=torch.bfloat16)
        pipe.dit.load_state_dict(state_dict)
        print(f"Loaded DiT checkpoint from {args.dit_checkpoint}")

    for idx, (subdir, prompt, foveation_mask, full_res_mask, _) in enumerate(items):
        _run_ours_one(pipe, args, subdir, prompt, 0, foveation_mask, full_res_mask)
        os.rename(os.path.join(subdir, "img_0000000000.png"), os.path.join(subdir, "img_ours.png"))
        if getattr(args, "foveation_outline", True):
            _overlay_outline(os.path.join(subdir, "img_ours.png"), full_res_mask,
                             height, width, outline_width_frac, outline_color)
        print(f"  saved {idx:05d}/img_ours.png")

    if args.lora_checkpoint is not None:
        pipe.clear_lora(verbose=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_mask_image(full_res_mask, pipe, args, path: str):
    """Save a full-resolution mask, optionally Gaussian-blurred to match the soft blend."""
    mask = full_res_mask.unsqueeze(0).unsqueeze(0).float().to(pipe.device)
    if getattr(args, "soft_foveation_blend", False):
        upscale = args.height / (args.height // 16)
        mask = gaussian_blur_mask_2d(mask, upscale, device=pipe.device, dtype=mask.dtype)
    if path.endswith(".png") and ("foveation_mask" in path or "mask_" in path or path.endswith("mask.png")):
        # Use uint8 PNG via numpy for user-study / trajectory layouts
        import imageio
        arr = np.clip(mask.squeeze().cpu().numpy(), 0.0, 1.0)
        imageio.imwrite(path, (arr * 255).astype(np.uint8))
    else:
        save_image(mask, path)


def _overlay_outline(image_path, full_res_mask, height, width, outline_width_frac, outline_color):
    import imageio
    img = imageio.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    draw_foveation_outline(img, full_res_mask, height, width, outline_width_frac, outline_color)
    imageio.imwrite(image_path, img)


def _save_fixation_dot(center, height, width, path: str):
    import imageio
    cx_px = (center[0] + 0.5) * width
    cy_px = (center[1] + 0.5) * height
    dot_radius = max(2, int(0.01 * min(width, height)))
    y = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    in_circle = (x - cx_px) ** 2 + (y - cy_px) ** 2 <= dot_radius ** 2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[in_circle] = [255, 0, 0]
    imageio.imwrite(path, img)


EXPERIMENTS = {
    "high_res": run_single_prompt_experiment,
    "naive_mixed_res": run_single_prompt_experiment,
    "ours": run_single_prompt_experiment,
    "ours_adaptive": run_adaptive_policy_experiment,
    "circular_traj": run_circular_traj,
    "vary_radius": run_vary_radius,
    "runtime": run_runtime_experiments,
    "foveation_trajectory_grid": run_foveation_trajectory_grid,
    "user_study": run_user_study,
}
