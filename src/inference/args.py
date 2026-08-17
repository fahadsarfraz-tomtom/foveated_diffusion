"""argparse for inference.py — supports both --pipeline image (FLUX2) and --pipeline video (Wan2.1)."""

import argparse


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, str):
        return s.lower() in ("true", "1", "yes")
    if isinstance(s, (int, float)):
        return bool(s)
    raise argparse.ArgumentTypeError(f"Expected true/false, got {s!r}")


DEFAULT_IMAGE_PROMPT = (
    "Documentary-style imagery: a lively little dog stands still on a lush green lawn, "
    "filling the entire frame. The dog has brownish-yellow fur, with both ears perked up, "
    "and an expression that is focused and cheerful."
)

# Canonical Wan2.1-T2V-1.3B demo prompt (matches DiffSynth-Studio's example).
DEFAULT_VIDEO_PROMPT = (
    "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，"
    "两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。"
    "背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。"
    "透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
)

# Canonical Wan2.1-T2V-1.3B negative prompt. Used by inference experiments and
# by the training-time sample-video logger; not user-configurable on either side.
DEFAULT_VIDEO_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Foveated diffusion inference (image / video)")
    p.add_argument("--pipeline", type=str, default="image", choices=["image", "video"],
                   help="Which pipeline to run. Selects the experiment registry + loader.")

    # Model
    p.add_argument("--model_id", type=str, default=None,
                   help="HuggingFace model ID. Defaults: 'black-forest-labs/FLUX.2-klein-base-4B' "
                        "for --pipeline image, 'Wan-AI/Wan2.1-T2V-1.3B' for --pipeline video.")
    p.add_argument("--lora_checkpoint", default=None, type=str,
                   help="Path to LoRA .safetensors. Takes precedence over --lora_mode. "
                        "For video --experiment ours, defaults to "
                        "checkpoints/wan_random_path_lora.safetensors (vendored).")
    p.add_argument("--lora_mode", default=None, choices=["random", "saliency", "bbox"],
                   help="(image) Auto-download a pre-trained image LoRA from "
                        "bchao1/foveated-diffusion on Hugging Face. "
                        "Ignored when --lora_checkpoint is set. "
                        "Choices: random (fov_random.safetensors), "
                        "saliency (fov_saliency.safetensors), "
                        "bbox (fov_bbox.safetensors).")
    p.add_argument("--dit_checkpoint", default=None, type=str, help="Path to full DiT checkpoint")

    # Generation
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--prompt", type=str, default=None,
                   help="Single-prompt input. If omitted, defaults to "
                        "DEFAULT_IMAGE_PROMPT for --pipeline image or "
                        "DEFAULT_VIDEO_PROMPT (canonical Wan demo) for --pipeline video.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decode_mode", type=str, default="direct", choices=["direct", "merge"])
    p.add_argument("--prediction_type", type=str, default="clean", choices=["clean", "flow", "refiner"])
    p.add_argument("--soft_foveation_blend", type=str_to_bool, default=False,
                   help="Use a Gaussian-falloff foveation mask boundary in merge decode")
    p.add_argument("--lr_downsample_factor", type=int, default=2, choices=[2, 4],
                   help="Spatial downsampling factor for LR periphery (2 = 4x fewer LR tokens, 4 = 16x)")

    # Output
    p.add_argument("--output_dir", type=str, default="./outputs/flux2_foveated")

    # Experiment selection (combined image + video — dispatch checks --pipeline first)
    p.add_argument("--experiment", type=str, default="ours",
                   choices=[
                       # image experiments
                       "high_res", "naive_mixed_res", "ours", "ours_adaptive",
                       "mask_source_comparison",
                       "circular_traj", "vary_radius",
                       "runtime", "foveation_trajectory_grid",
                       "user_study",
                       # video experiments (--pipeline video)
                       "naive",
                   ])

    # Full / distributed eval
    p.add_argument("--full_eval", action="store_true", default=False)
    p.add_argument("--subset_idx", type=int, default=0)
    p.add_argument("--num_subsets", type=int, default=1)
    p.add_argument("--full_eval_mask", type=str, default="square",
                   choices=["square", "checkerboard", "circular"])
    p.add_argument("--prompt_dataset_path", type=str, help="Path to CSV with a 'prompt' column")
    p.add_argument("--num_prompts", type=int, default=None)

    # Foveation trajectory grid
    p.add_argument("--num_cols", type=int, default=4)
    p.add_argument("--foveation_trajectory_type", type=str, default="circular",
                   choices=["radius", "circular", "random_circular", "polygons",
                            "multi_circle", "grid", "spiral"])
    p.add_argument("--grid_rows", type=int, default=3)
    p.add_argument("--grid_cols", type=int, default=3)
    p.add_argument("--outline_width_frac", type=float, default=0.005)
    p.add_argument("--outline_color", type=str, default="255,0,0")
    p.add_argument("--foveation_outline", type=str_to_bool, default=False)
    p.add_argument("--prompt_ids", type=int, nargs="+", default=None,
                   help="Explicit CSV row indices for foveation_trajectory_grid")

    # Circular / vary_radius experiment knobs
    p.add_argument("--num_frames", type=int, default=81,
                   help="(image) Number of mask positions for trajectory experiments. "
                        "(video) Number of output video frames. 81 is Wan2.1-T2V-1.3B's canonical length.")
    p.add_argument("--orbit_radius", type=float, default=0.25)
    p.add_argument("--mask_radius", type=float, default=0.30,
                   help="Foveation radius (circular) or side ratio (square)")
    p.add_argument("--mask_shape", type=str, default="circular",
                   choices=["circular", "square"])

    # Video-pipeline specific
    p.add_argument("--foveation_trajectory", type=str, default="spline",
                   choices=["spline", "random_path", "adaptive"],
                   help="(video) Foveation-path sampler at inference. `spline` (default) is "
                        "the deterministic default keypoints; `random_path` matches the "
                        "training-time sampler; `adaptive` uses the prompt-conditioned policy.")
    p.add_argument("--negative_prompt", type=str, default=None,
                   help="(video) Negative prompt. Defaults to the canonical Wan Chinese negative.")
    p.add_argument("--cfg_scale", type=float, default=5.0,
                   help="(video) CFG scale. Wan default is 5.0; image-side reads guidance_scale instead.")
    p.add_argument("--fps", type=int, default=15, help="(video) Output FPS.")
    p.add_argument("--quality", type=int, default=5, help="(video) Output MP4 quality 1-10.")

    # Adaptive foveation policy (ours)
    p.add_argument("--adaptive_policy", type=str, default="prompt_hash",
                   choices=["center", "prompt_hash", "lsca_proxy", "textfov_proxy", "fpm"],
                   help="Adaptive policy source. `fpm` uses a trained FPM checkpoint "
                        "(--fpm_checkpoint); `lsca_proxy` is its deterministic stand-in; "
                        "`textfov_proxy` uses spatial prompt-token heuristics.")
    p.add_argument("--fpm_checkpoint", type=str, default=None,
                   help="Path to a trained FPM checkpoint (train_fpm_coco[_latent].py "
                        "save_checkpoint format). Required for --adaptive_policy fpm and "
                        "for the fpm arm of mask_source_comparison.")
    p.add_argument("--fpm_objectness_threshold", type=float, default=0.5,
                   help="Objectness gate for FPM slots; slots below it are dropped "
                        "(the strongest slot is always kept).")
    p.add_argument("--fpm_timestep_id", type=int, default=0,
                   help="Scheduler timestep index the FPM is conditioned on in prior "
                        "mode (0 = noisiest, matching training's timestep_id indexing).")

    # Matched-budget mask-source comparison (FGD-018)
    p.add_argument("--comparison_arms", type=str, nargs="+",
                   default=["center", "random", "saliency", "fpm"],
                   help="Foveated arms for mask_source_comparison. Each arm uses the "
                        "same HR token budget; only the mask source differs.")
    p.add_argument("--comparison_beta", type=float, default=0.25,
                   help="Shared HR token fraction (beta) for all comparison arms.")
    p.add_argument("--adaptive_num_fixations", type=int, default=3,
                   help="Number of fixation centers for adaptive image masks.")
    p.add_argument("--adaptive_center_range", type=float, default=0.35,
                   help="Max absolute center coordinate in Chao's [-0.5, 0.5] frame.")
    p.add_argument("--adaptive_fixed_beta", type=float, default=0.25,
                   help="Target HR token fraction when --adaptive_beta_mode=fixed.")
    p.add_argument("--adaptive_beta_mode", type=str, default="fixed",
                   choices=["fixed", "nafo_mean", "nafo_early", "nafo_late"],
                   help="How to project NaFo beta(t) to Chao's current static image mask interface.")
    p.add_argument("--adaptive_beta_min", type=float, default=0.20,
                   help="NaFo late-step/high-detail HR token budget.")
    p.add_argument("--adaptive_beta_max", type=float, default=0.85,
                   help="NaFo early-step/global-structure HR token budget.")
    p.add_argument("--adaptive_beta_schedule", type=str, default="cosine",
                   choices=["linear", "cosine", "stepped"],
                   help="NaFo schedule shape.")
    p.add_argument("--adaptive_radius", type=float, default=None,
                   help="Optional fixed radius override. If omitted, radius is solved from beta.")

    return p
