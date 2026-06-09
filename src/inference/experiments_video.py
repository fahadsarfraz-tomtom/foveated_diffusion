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

import os
import time

import torch

from diffsynth.utils.data import save_video

from ..masks.paths import sample_random_path, sample_spline_path
from ..masks.state import build_state
from .args import DEFAULT_VIDEO_NEGATIVE_PROMPT
from .visualize import draw_foveation_circle_on_frames


def _negative_prompt(args):
    return getattr(args, "negative_prompt", None) or DEFAULT_VIDEO_NEGATIVE_PROMPT


def _build_foveation_state_for_args(args, device):
    """Build a (FoveationState, centers, radii) triple from CLI args.

    Uses ``args.foveation_trajectory`` to pick the path sampler. Returns
    centers and radii too so the circle-overlay can be drawn later.
    """
    latent_length = (args.num_frames - 1) // 4 + 1
    if args.foveation_trajectory == "random_path":
        centers, radii = sample_random_path(latent_length, device=device)
    else:  # default: spline
        centers, radii = sample_spline_path(latent_length)
    foveation_state = build_state(
        centers, radii, args.height, args.width, args.num_frames, device=device,
    )
    return foveation_state, centers, radii


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
    foveation_state, centers, radii = _build_foveation_state_for_args(args, pipe.device)
    print(f"[naive] FoveationState built ({args.foveation_trajectory}), token_ratio="
          f"{(foveation_state.packed_hr_positions.numel()) / (((args.num_frames-1)//4+1) * (args.height//16) * (args.width//16)):.3f}")
    t0 = time.time()
    video = _generate(pipe, args, prompt, foveation_state=foveation_state)
    print(f"[naive] gen done in {time.time() - t0:.1f}s")
    _save_with_overlay(args, video, output_dir, "naive", centers=centers, radii=radii)


def run_ours(pipe, args, output_dir):
    """Foveated headline — spline FoveationState + LoRA. LoRA load is handled
    by the pipeline loader; this runner just generates."""
    prompt = _resolve_prompt(args)
    foveation_state, centers, radii = _build_foveation_state_for_args(args, pipe.device)
    print(f"[ours] FoveationState built ({args.foveation_trajectory})")
    t0 = time.time()
    video = _generate(pipe, args, prompt, foveation_state=foveation_state)
    print(f"[ours] gen done in {time.time() - t0:.1f}s")
    _save_with_overlay(args, video, output_dir, "ours", centers=centers, radii=radii)


VIDEO_EXPERIMENTS = {
    "high_res": run_high_res,
    "naive": run_naive,
    "ours": run_ours,
}
