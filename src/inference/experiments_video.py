"""Video inference experiment runners for the foveated Wan2.1 pipeline.

Three experiments, all taking ``(pipe, args, output_dir)``:

  - ``high_res``: vanilla Wan baseline (``foveation_state=None``). Useful as
    a reference for the foveated runs.
  - ``naive``:    spline FoveationState, no LoRA loaded. Exercises CRPA +
    dual-track + merge-decode and shows the paper's Fig. 8 failure mode
    (scale mismatches at the HR/LR boundary).
  - ``ours``:     spline FoveationState + LoRA loaded. The release headline
    result; ``--lora_checkpoint`` defaults to
    ``checkpoints/wan_random_path_lora.safetensors``. To use a
    saliency-trained LoRA instead, pass it via ``--lora_checkpoint`` — the
    inference code path is identical; saliency is purely a training-time
    mask-source choice (see ``--foveation_from_video`` on ``train.py``).

Foveated experiments save two MP4s: the raw output and a circle-overlay
version (paper-figure style) drawn via
``src.inference.visualize.draw_foveation_circle_on_frames``.
"""

import json
import os
import time

import torch

from diffsynth.utils.data import save_video

from ..masks import AdaptiveFoveationPolicy, adaptive_config_from_args
from ..masks.paths import sample_random_path, sample_spline_path
from ..masks.state import build_state
from .args import DEFAULT_VIDEO_NEGATIVE_PROMPT
from .visualize import draw_foveation_circle_on_frames


def _negative_prompt(args):
    return getattr(args, "negative_prompt", None) or DEFAULT_VIDEO_NEGATIVE_PROMPT


def _build_foveation_state_for_args(args, device, prompt=None):
    """Build a (FoveationState, centers, radii, metadata) tuple from CLI args.

    Uses ``args.foveation_trajectory`` to pick the path sampler. Returns
    centers and radii too so the circle-overlay can be drawn later.
    """
    latent_length = (args.num_frames - 1) // 4 + 1
    if args.foveation_trajectory == "random_path":
        centers, radii = sample_random_path(latent_length, device=device)
        metadata = {"trajectory": "random_path"}
    elif args.foveation_trajectory == "adaptive":
        policy = AdaptiveFoveationPolicy(adaptive_config_from_args(args))
        centers, radii, metadata = policy.plan_video(
            prompt or "",
            args.height,
            args.width,
            args.num_frames,
            device,
            lr_factor=getattr(args, "lr_downsample_factor", 2),
        )
        metadata["trajectory"] = "adaptive"
    else:  # default: spline
        centers, radii = sample_spline_path(latent_length)
        metadata = {"trajectory": "spline"}
    foveation_state = build_state(
        centers, radii, args.height, args.width, args.num_frames, device=device,
    )
    metadata["token_ratio"] = float(
        foveation_state.packed_hr_positions.numel()
        / (((args.num_frames - 1) // 4 + 1) * (args.height // 16) * (args.width // 16))
    )
    return foveation_state, centers, radii, metadata


def _resolve_prompt(args):
    """For v1, single-prompt mode. CSV-driven multi-prompt is a future TODO."""
    return args.prompt


def _generate(pipe, args, prompt, foveation_state):
    torch.manual_seed(args.seed)
    return pipe(
        prompt=prompt, negative_prompt=_negative_prompt(args),
        height=args.height, width=args.width, num_frames=args.num_frames,
        cfg_scale=args.cfg_scale, num_inference_steps=args.num_inference_steps,
        seed=args.seed, tiled=True,
        foveation_state=foveation_state,
    )


def _save_with_overlay(args, video, output_dir, basename, centers=None, radii=None):
    out_path = os.path.join(output_dir, f"{basename}.mp4")
    save_video(video, out_path, fps=args.fps, quality=args.quality)
    print(f"[video_experiments] saved -> {out_path}")
    if centers is not None and radii is not None:
        video_circle = draw_foveation_circle_on_frames(video, centers, radii, thickness=2)
        circle_path = os.path.join(output_dir, f"{basename}_with_circle.mp4")
        save_video(video_circle, circle_path, fps=args.fps, quality=args.quality)
        print(f"[video_experiments] saved -> {circle_path}")


def _save_foveation_metadata(output_dir, basename, prompt, centers, radii, metadata):
    path = os.path.join(output_dir, f"{basename}_foveation.json")
    payload = {
        "prompt": prompt,
        "centers": [[float(x), float(y)] for x, y in centers],
        "radii": [float(r) for r in radii],
        "metadata": metadata,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[video_experiments] saved -> {path}")


def run_high_res(pipe, args, output_dir):
    """Vanilla Wan baseline — no foveation. The foveated DiT falls through
    to the standard rope_apply path when ``foveation_state=None``."""
    prompt = _resolve_prompt(args)
    print(f"[high_res] {args.height}x{args.width}x{args.num_frames}, {args.num_inference_steps} steps")
    t0 = time.time()
    video = _generate(pipe, args, prompt, foveation_state=None)
    print(f"[high_res] gen done in {time.time() - t0:.1f}s")
    _save_with_overlay(args, video, output_dir, "high_res")


def run_naive(pipe, args, output_dir):
    """Naive baseline — spline FoveationState, no LoRA. Shows HR/LR seams."""
    prompt = _resolve_prompt(args)
    foveation_state, centers, radii, metadata = _build_foveation_state_for_args(args, pipe.device, prompt)
    print(f"[naive] FoveationState built ({args.foveation_trajectory}), token_ratio="
          f"{metadata['token_ratio']:.3f}")
    t0 = time.time()
    video = _generate(pipe, args, prompt, foveation_state=foveation_state)
    print(f"[naive] gen done in {time.time() - t0:.1f}s")
    _save_with_overlay(args, video, output_dir, "naive", centers=centers, radii=radii)
    _save_foveation_metadata(output_dir, "naive", prompt, centers, radii, metadata)


def run_ours(pipe, args, output_dir):
    """Foveated headline — spline FoveationState + LoRA. LoRA load is handled
    by the pipeline loader; this runner just generates."""
    prompt = _resolve_prompt(args)
    foveation_state, centers, radii, metadata = _build_foveation_state_for_args(args, pipe.device, prompt)
    print(f"[ours] FoveationState built ({args.foveation_trajectory}), token_ratio={metadata['token_ratio']:.3f}")
    t0 = time.time()
    video = _generate(pipe, args, prompt, foveation_state=foveation_state)
    print(f"[ours] gen done in {time.time() - t0:.1f}s")
    _save_with_overlay(args, video, output_dir, "ours", centers=centers, radii=radii)
    _save_foveation_metadata(output_dir, "ours", prompt, centers, radii, metadata)


def run_ours_adaptive(pipe, args, output_dir):
    """Our adaptive foveation policy on top of Chao's Wan video foveation state."""
    prompt = _resolve_prompt(args)
    original_trajectory = args.foveation_trajectory
    args.foveation_trajectory = "adaptive"
    try:
        foveation_state, centers, radii, metadata = _build_foveation_state_for_args(args, pipe.device, prompt)
    finally:
        args.foveation_trajectory = original_trajectory
    print(f"[ours_adaptive] FoveationState built, token_ratio={metadata['token_ratio']:.3f}")
    t0 = time.time()
    video = _generate(pipe, args, prompt, foveation_state=foveation_state)
    print(f"[ours_adaptive] gen done in {time.time() - t0:.1f}s")
    _save_with_overlay(args, video, output_dir, "ours_adaptive", centers=centers, radii=radii)
    _save_foveation_metadata(output_dir, "ours_adaptive", prompt, centers, radii, metadata)


VIDEO_EXPERIMENTS = {
    "high_res": run_high_res,
    "naive": run_naive,
    "ours": run_ours,
    "ours_adaptive": run_ours_adaptive,
}
