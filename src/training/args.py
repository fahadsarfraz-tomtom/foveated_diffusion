"""argparse for train.py — supports both --pipeline image (FLUX2) and --pipeline video (Wan2.1)."""

import argparse

from diffsynth.diffusion import add_general_config, add_video_size_config


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    if isinstance(s, str):
        return s.lower() in ("true", "1", "yes")
    if isinstance(s, (int, float)):
        return bool(s)
    raise argparse.ArgumentTypeError(f"Expected true/false, got {s!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Foveated diffusion training (image / video)")
    p.add_argument("--pipeline", type=str, default="image", choices=["image", "video"],
                   help="Which pipeline to train. Selects the module + size-config schema.")
    p = add_general_config(p)
    # Video size config is a strict superset of image (adds --num_frames);
    # `--num_frames` is just unused when --pipeline=image.
    p = add_video_size_config(p)
    p.add_argument("--tokenizer_path", type=str, default=None)

    # WandB / validation (image-only validation viz for now; video has no logger yet)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="foveated-diffusion")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--validation_prompts", type=str, nargs="+", default=None)
    p.add_argument("--validation_steps", type=int, default=500)
    p.add_argument("--validation_height", type=int, default=1024)
    p.add_argument("--validation_width", type=int, default=1024)
    p.add_argument("--num_validation_images", type=int, default=1)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_training_steps", type=int, default=None,
                   help="Stop training after this many steps. None = full num_epochs.")

    # Image-pipeline specific
    p.add_argument("--prediction_type", type=str, default="clean",
                   choices=["clean", "flow", "refiner"],
                   help="(image) Flow-match parameterisation.")
    p.add_argument("--decode_mode", type=str, default="direct", choices=["direct", "merge"],
                   help="(image) Validation decode mode.")
    p.add_argument("--is_foveated_pipeline", type=str_to_bool, default=True,
                   help="(image) Use the foveated FLUX2 pipeline.")
    p.add_argument("--lr_downsample_factor", type=int, default=2, choices=[2, 4],
                   help="(image) LR periphery downsample factor.")
    p.add_argument(
        "--foveated_training_mode", type=str, default="random",
        choices=["fixed", "random", "saliency", "bbox"],
        help="(image) Foveation-mask sampler during training.",
    )

    # Video-pipeline specific
    p.add_argument("--foveation_from_video", action="store_true",
                   help="(video) Predict foveation path from input video via DeepGaze saliency. "
                        "Default sampler when off is `random_path` (linear-interp endpoints) "
                        "per the paper's video training distribution.")
    p.add_argument("--max_timestep_boundary", type=float, default=1.0,
                   help="(video) Upper bound on sampled timestep id (fraction of total).")
    p.add_argument("--min_timestep_boundary", type=float, default=0.0,
                   help="(video) Lower bound on sampled timestep id (fraction of total).")

    # Video sample-video logging (rendered to wandb at --log_video_steps cadence)
    p.add_argument("--log_video_steps", type=int, default=1000,
                   help="(video) Generate + log a validation video every N training steps.")
    p.add_argument("--sample_prompt", type=str, default=None,
                   help="(video) Prompt for the sample-video logger. Default: canonical Wan demo.")
    p.add_argument("--sample_foveation_trajectory", type=str, default="spline",
                   choices=["spline", "random_path"],
                   help="(video) Foveation path used for sample-video gen. spline = deterministic; "
                        "random_path = a fresh sample each time (less comparable across snapshots).")
    p.add_argument("--sample_num_inference_steps", type=int, default=30,
                   help="(video) Denoising steps for sample-video gen (reduced from inference's 50 "
                        "to keep validation cheap).")
    p.add_argument("--sample_seed", type=int, default=42,
                   help="(video) Fixed seed for sample-video gen so cross-snapshot diffs reflect "
                        "LoRA progress only.")
    p.add_argument("--sample_cfg_scale", type=float, default=5.0,
                   help="(video) CFG scale for sample-video gen.")
    return p
