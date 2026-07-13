#!/usr/bin/env python3
"""Supervised COCO FPM training on noisy FLUX VAE latents.

This is the inference-aligned follow-up to ``train_fpm_coco.py``. The RGB
trainer validates the object-slot supervision cheaply; this trainer replaces the
downsampled RGB proxy with FLUX VAE latents and samples noisy ``z_t`` inputs
from the same flow-match scheduler used by Chao's FLUX2 pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from src.masks import FovealPredictionModule
from src.training.coco_fpm import (
    CocoFoveationDataset,
    HashTextEmbedder,
    coco_foveation_collate,
    fpm_supervision_loss,
    target_gaussian_map,
)
from train_fpm_coco import _init_wandb, _log_wandb_examples, save_checkpoint


@dataclass
class FluxVaeBundle:
    vae: torch.nn.Module
    scheduler: object
    device: torch.device
    torch_dtype: torch.dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", type=Path, default=Path("data/foveation/coco"))
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument("--output_dir", type=Path, default=Path("models/fpm_coco_latent"))
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--model_cache_dir", type=Path, default=None)
    parser.add_argument("--download_source", type=str, default="huggingface", choices=["huggingface", "modelscope"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--mask_size", type=int, default=None)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_fixations", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--text_dim", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--caption_mode", type=str, default="all", choices=["all", "first"])
    parser.add_argument("--no_fallback_all_objects", action="store_true")
    parser.add_argument("--target_radius_margin", type=float, default=1.0)
    parser.add_argument("--target_radius_min", type=float, default=0.03)
    parser.add_argument("--target_radius_max", type=float, default=0.45)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--lambda_center", type=float, default=2.0)
    parser.add_argument("--lambda_radius", type=float, default=1.0)
    parser.add_argument("--lambda_object", type=float, default=1.0)
    parser.add_argument("--lambda_map", type=float, default=1.0)
    parser.add_argument("--lambda_budget", type=float, default=0.5)
    parser.add_argument("--lambda_repulsion", type=float, default=0.05)
    parser.add_argument("--lambda_area", type=float, default=0.0)
    parser.add_argument("--target_budget_scale", type=float, default=1.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="foveation-diffusion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_log_steps", type=int, default=None)
    parser.add_argument("--wandb_image_steps", type=int, default=250)
    parser.add_argument("--wandb_num_images", type=int, default=8)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def load_flux_vae_pipeline(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    model_cache_dir: str | Path | None = None,
    download_source: str = "huggingface",
) -> FluxVaeBundle:
    """Load only the DiffSynth FLUX2 VAE plus the matching flow scheduler."""
    try:
        from diffsynth.core import ModelConfig
        from diffsynth.diffusion import FlowMatchScheduler
        from diffsynth.models.model_loader import ModelPool
    except ImportError as exc:
        raise ImportError(
            "Latent FPM training requires DiffSynth-Studio. Install/provide the "
            "`diffsynth` package, or run this script in the SPIKE foveation image."
        ) from exc

    model_config = ModelConfig(
        model_id=model_id,
        origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
        download_source=download_source,
        local_model_path=str(model_cache_dir) if model_cache_dir is not None else None,
        onload_device=device,
        onload_dtype=dtype,
        preparing_device=device,
        preparing_dtype=dtype,
        computation_device=device,
        computation_dtype=dtype,
    )
    model_config.download_if_necessary()
    model_pool = ModelPool()
    vram_config = model_pool.default_vram_config()
    for key, value in model_config.vram_config().items():
        if value is not None:
            vram_config[key] = value
    model_pool.auto_load_model(model_config.path, vram_config=vram_config)
    vae = model_pool.fetch_model("flux2_vae")
    if vae is None:
        raise RuntimeError(f"failed to load FLUX VAE for model_id={model_id}")
    scheduler = FlowMatchScheduler("FLUX.2")
    scheduler.set_timesteps(1000, training=True)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return FluxVaeBundle(vae=vae, scheduler=scheduler, device=device, torch_dtype=dtype)


def spatial_to_sequence(latents: torch.Tensor) -> torch.Tensor:
    """Convert spatial latents [B, C, H, W] to DiffSynth sequence [B, H*W, C]."""
    return latents.flatten(2).transpose(1, 2).contiguous()


def sequence_to_spatial(sequence: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert DiffSynth sequence [B, H*W, C] back to spatial [B, C, H, W]."""
    batch, tokens, channels = sequence.shape
    if tokens != height * width:
        raise ValueError(f"sequence has {tokens} tokens, expected {height * width}")
    return sequence.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


def add_scheduler_noise_to_spatial_latents(
    scheduler,
    clean_latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Apply a DiffSynth scheduler to spatial latents via the sequence format."""
    batch, _channels, height, width = clean_latents.shape
    noise = torch.randn_like(clean_latents)
    clean_sequence = spatial_to_sequence(clean_latents)
    noise_sequence = spatial_to_sequence(noise)
    noisy_sequence = scheduler.add_noise(clean_sequence, noise_sequence, timestep)
    return sequence_to_spatial(noisy_sequence, height, width)


@torch.no_grad()
def encode_images(pipe, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Encode COCO images in [-1, 1] to frozen FLUX VAE spatial latents."""
    images = images.to(device=pipe.device, dtype=dtype, non_blocking=True)
    latents = pipe.vae.encode(images)
    if latents.dim() != 4:
        raise RuntimeError(f"expected VAE spatial latents [B,C,H,W], got {tuple(latents.shape)}")
    return latents.float()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    vae_dtype = _dtype_from_name(args.vae_dtype)
    pipe = load_flux_vae_pipeline(
        args.model_id,
        device=device,
        dtype=vae_dtype,
        model_cache_dir=args.model_cache_dir,
        download_source=args.download_source,
    )
    pipe.scheduler.set_timesteps(args.num_train_timesteps, training=True)

    dataset = CocoFoveationDataset(
        coco_root=args.coco_root,
        split=args.split,
        image_size=args.image_size,
        max_objects=args.num_fixations,
        max_tokens=args.max_tokens,
        vocab_size=args.vocab_size,
        caption_mode=args.caption_mode,
        fallback_all_objects=not args.no_fallback_all_objects,
        radius_margin=args.target_radius_margin,
        radius_min=args.target_radius_min,
        radius_max=args.target_radius_max,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"no COCO samples found under {args.coco_root}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
        collate_fn=coco_foveation_collate,
    )

    first = next(iter(loader))
    first_images = first["image"].to(device=device, non_blocking=True)
    first_latents = encode_images(pipe, first_images, vae_dtype)
    latent_channels = first_latents.shape[1]
    latent_h, latent_w = first_latents.shape[-2:]
    mask_size = args.mask_size or max(min(latent_h, latent_w) // 4, 1)

    fpm = FovealPredictionModule(
        latent_channels=latent_channels,
        text_dim=args.text_dim,
        num_fixations=args.num_fixations,
        hidden_dim=args.hidden_dim,
    ).to(device)
    text_embedder = HashTextEmbedder(vocab_size=args.vocab_size, text_dim=args.text_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(fpm.parameters()) + list(text_embedder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "latent_channels": latent_channels,
        "latent_height": latent_h,
        "latent_width": latent_w,
        "mask_size_resolved": mask_size,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    print(f"[dataset] {len(dataset):,} samples from {args.coco_root}/{args.split}")
    print(f"[train] device={device} vae_dtype={vae_dtype} latent_shape={(latent_channels, latent_h, latent_w)} mask_size={mask_size}")
    wandb = _init_wandb(args)
    wandb_log_steps = args.wandb_log_steps or args.log_steps

    pending_batch = first
    step = 0
    last_log = time.time()
    metric_sums = {}
    while step < args.max_steps:
        for batch in loader:
            if pending_batch is not None:
                batch = pending_batch
                pending_batch = None
            step += 1
            image = batch["image"].to(device=device, non_blocking=True)
            clean_latents = encode_images(pipe, image, vae_dtype)
            scheduler_timesteps = pipe.scheduler.timesteps
            timestep_id = torch.randint(0, len(scheduler_timesteps), (1,), device=scheduler_timesteps.device)
            timestep = scheduler_timesteps[timestep_id].to(dtype=vae_dtype, device=device)
            noisy_latents = add_scheduler_noise_to_spatial_latents(pipe.scheduler, clean_latents, timestep).float()
            fpm_timesteps = timestep_id.expand(image.shape[0]).to(device=device)

            token_ids = batch["token_ids"].to(device=device, non_blocking=True)
            text_embeddings = text_embedder(token_ids)

            target_centers = batch["target_centers"].to(device=device, non_blocking=True)
            target_radii = batch["target_radii"].to(device=device, non_blocking=True)
            target_valid = batch["target_valid"].to(device=device, non_blocking=True)
            target_map = target_gaussian_map(
                target_centers,
                target_radii,
                target_valid,
                height=mask_size,
                width=mask_size,
            )

            cx, cy, radii, object_logits, weight_map = fpm.predict_slots(
                noisy_latents,
                text_embeddings,
                timesteps=fpm_timesteps,
                out_height=mask_size,
                out_width=mask_size,
            )
            loss, metrics = fpm_supervision_loss(
                cx,
                cy,
                radii,
                object_logits,
                weight_map,
                target_centers,
                target_radii,
                target_valid,
                target_map,
                lambda_center=args.lambda_center,
                lambda_radius=args.lambda_radius,
                lambda_object=args.lambda_object,
                lambda_map=args.lambda_map,
                lambda_budget=args.lambda_budget,
                lambda_repulsion=args.lambda_repulsion,
                lambda_area=args.lambda_area,
                target_budget_scale=args.target_budget_scale,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fpm.parameters()) + list(text_embedder.parameters()), 1.0)
            optimizer.step()

            metrics["latent_timestep_id"] = float(timestep_id.item())
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

            if step % args.log_steps == 0:
                dt = max(time.time() - last_log, 1e-6)
                averaged = {k: v / args.log_steps for k, v in metric_sums.items()}
                msg = " ".join(f"{k}={v:.4f}" for k, v in sorted(averaged.items()))
                print(f"[step {step:06d}] {msg}  {args.log_steps / dt:.2f} steps/s")
                if wandb is not None and step % wandb_log_steps == 0:
                    wandb.log({f"train/{k}": v for k, v in averaged.items()}, step=step)
                    wandb.log({"train/steps_per_second": args.log_steps / dt}, step=step)
                metric_sums.clear()
                last_log = time.time()

            _log_wandb_examples(wandb, step, batch, cx, cy, radii, object_logits, args)

            if step % args.save_steps == 0:
                save_checkpoint(args.output_dir / f"step_{step:06d}.pt", fpm, text_embedder, optimizer, step, args)

            if step >= args.max_steps:
                break

    save_checkpoint(args.output_dir / "final.pt", fpm, text_embedder, optimizer, step, args)
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
