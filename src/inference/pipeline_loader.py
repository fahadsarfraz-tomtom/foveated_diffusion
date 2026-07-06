"""Load the foveated FLUX2 (image) or Wan2.1 (video) pipeline plus optional LoRA / DiT."""

import os

import torch

from diffsynth.core import load_state_dict
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

from ..diffsynth_fov import (
    Flux2FoveatedImagePipeline,
    WanFoveatedVideoPipeline,
    default_wan_model_configs,
    default_wan_tokenizer_config,
)


HF_REPO = "bchao1/foveated-diffusion"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOCAL_WAN_LORA = os.path.join(_REPO_ROOT, "checkpoints", "wan_random_path_lora.safetensors")

_IMAGE_LORA_HF_FILES = {
    "random": "image/fov_random.safetensors",
    "saliency": "image/fov_saliency.safetensors",
    "bbox": "image/fov_bbox.safetensors",
}

_VIDEO_LORA_HF_FILES = {
    "random": "video/fov_random.safetensors",
}


def _resolve_default_wan_lora() -> str:
    if os.path.exists(_LOCAL_WAN_LORA):
        return _LOCAL_WAN_LORA
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=_VIDEO_LORA_HF_FILES["random"])


def _resolve_image_lora(lora_mode: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=_IMAGE_LORA_HF_FILES[lora_mode])


def _resolve_video_lora(lora_mode: str) -> str:
    if lora_mode not in _VIDEO_LORA_HF_FILES:
        raise ValueError(
            f"--lora_mode {lora_mode!r} is not available for video. "
            f"Available: {list(_VIDEO_LORA_HF_FILES)}"
        )
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, filename=_VIDEO_LORA_HF_FILES[lora_mode])


def load_pipeline(args, use_foveated_pipeline: bool = True):
    """Load FLUX2 image pipeline (foveated by default) + optional LoRA / DiT swap.

    For the user-study experiment LoRA / DiT swaps happen later (inside the runner)
    so the baseline + naive passes can use the base DiT.
    """
    print("Loading FLUX2 foveated pipeline...")
    pipeline_class = Flux2FoveatedImagePipeline if use_foveated_pipeline else Flux2ImagePipeline
    model_id = args.model_id or "black-forest-labs/FLUX.2-klein-base-4B"
    pipe = pipeline_class.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=model_id, origin_file_pattern="tokenizer/"),
    )

    defer_load = (args.experiment == "user_study")

    lora_mode = getattr(args, "lora_mode", None)
    if args.lora_checkpoint is not None:
        lora_path = args.lora_checkpoint
    elif lora_mode is not None:
        lora_path = _resolve_image_lora(lora_mode)
        print(f"Auto-downloaded image LoRA ({lora_mode}): {lora_path}")
    else:
        lora_path = None

    if lora_path is not None and not defer_load:
        pipe.load_lora(pipe.dit, lora_path)
        print(f"Loaded LoRA: {lora_path}")
    elif lora_path is not None and defer_load:
        print("User study: LoRA will be loaded only for 'ours' runs")

    if args.dit_checkpoint is not None and not defer_load:
        state_dict = load_state_dict(args.dit_checkpoint, torch_dtype=torch.bfloat16)
        pipe.dit.load_state_dict(state_dict)
        print(f"Loaded DiT checkpoint from {args.dit_checkpoint}")
    elif args.dit_checkpoint is not None and defer_load:
        print("User study: DiT will be loaded only for 'ours' runs")

    return pipe


def load_video_pipeline(args):
    """Load WanFoveatedVideoPipeline + optional LoRA.

    LoRA loading rules per experiment:
      - high_res / naive: never load a LoRA (vanilla pipeline).
      - ours: load --lora_checkpoint, falling back to the default Wan LoRA
        (local ``checkpoints/`` copy if present, otherwise auto-downloaded
        from Hugging Face — see ``_resolve_default_wan_lora``).
    """
    print("Loading WanFoveatedVideoPipeline...")
    model_id = args.model_id or "Wan-AI/Wan2.1-T2V-1.3B"
    pipe = WanFoveatedVideoPipeline.from_pretrained_foveated(
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_id=model_id,
    )

    lora_mode = getattr(args, "lora_mode", None)
    needs_lora = args.experiment in ("ours", "ours_adaptive")
    if needs_lora:
        if args.lora_checkpoint is not None:
            lora_path = args.lora_checkpoint
        elif lora_mode is not None:
            lora_path = _resolve_video_lora(lora_mode)
            print(f"Auto-downloaded video LoRA ({lora_mode}): {lora_path}")
        else:
            lora_path = _resolve_default_wan_lora()
        pipe.load_lora(pipe.dit, lora_path, alpha=1)
        print(f"Loaded LoRA: {lora_path}")
    elif args.lora_checkpoint is not None or lora_mode is not None:
        # User passed a LoRA for a non-LoRA experiment (high_res / naive).
        # Honor it — useful for sanity comparisons.
        if args.lora_checkpoint is not None:
            lora_path = args.lora_checkpoint
        else:
            lora_path = _resolve_video_lora(lora_mode)
            print(f"Auto-downloaded video LoRA ({lora_mode}): {lora_path}")
        pipe.load_lora(pipe.dit, lora_path, alpha=1)
        print(f"Loaded LoRA (experiment={args.experiment!r}): {lora_path}")

    return pipe
