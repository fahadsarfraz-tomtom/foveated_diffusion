#!/usr/bin/env python3
"""Supervised COCO bootstrap training for the Foveal Prediction Module.

This is Phase 1 for learned foveation supervision: train FPM to infer object
count, centers, radii, and a foveal weight map from COCO image-caption pairs.
The trainer intentionally uses a lightweight RGB proxy input; the checkpoint is
for validating the policy objective before wiring FPM into full FLUX latents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.masks import FovealPredictionModule
from src.training.coco_fpm import (
    CocoFoveationDataset,
    HashTextEmbedder,
    coco_foveation_collate,
    fpm_supervision_loss,
    target_gaussian_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", type=Path, default=Path("data/foveation/coco"))
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument("--output_dir", type=Path, default=Path("models/fpm_coco"))
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--mask_size", type=int, default=None)
    parser.add_argument("--num_fixations", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--text_dim", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--caption_mode", type=str, default="all", choices=["all", "first"])
    parser.add_argument("--no_fallback_all_objects", action="store_true")
    parser.add_argument("--target_radius_margin", type=float, default=1.35)
    parser.add_argument("--target_radius_min", type=float, default=0.03)
    parser.add_argument("--target_radius_max", type=float, default=0.60)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda_center", type=float, default=2.0)
    parser.add_argument("--lambda_radius", type=float, default=0.5)
    parser.add_argument("--lambda_object", type=float, default=1.0)
    parser.add_argument("--lambda_map", type=float, default=1.0)
    parser.add_argument("--lambda_budget", type=float, default=0.1)
    parser.add_argument("--lambda_repulsion", type=float, default=0.01)
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


def save_checkpoint(path: Path, fpm, text_embedder, optimizer, step: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "fpm": fpm.state_dict(),
            "text_embedder": text_embedder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )
    print(f"[checkpoint] {path}")


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(-1, 1)
    image = ((image + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def _draw_foveation_overlay(
    image: torch.Tensor,
    target_centers: torch.Tensor,
    target_radii: torch.Tensor,
    target_valid: torch.Tensor,
    pred_cx: torch.Tensor,
    pred_cy: torch.Tensor,
    pred_radii: torch.Tensor,
    pred_logits: torch.Tensor,
    prompt: str,
    categories: list[str],
) -> Image.Image:
    pil = _tensor_to_pil(image).convert("RGB")
    draw = ImageDraw.Draw(pil)
    width, height = pil.size

    def circle(cx: float, cy: float, radius: float, color: tuple[int, int, int], width_px: int):
        px = float(cx) * width
        py = float(cy) * height
        pr = max(float(radius) * max(width, height), 2.0)
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), outline=color, width=width_px)
        draw.line((px - 5, py, px + 5, py), fill=color, width=width_px)
        draw.line((px, py - 5, px, py + 5), fill=color, width=width_px)

    for idx in range(target_centers.shape[0]):
        if float(target_valid[idx]) > 0.5:
            circle(
                float(target_centers[idx, 0]),
                float(target_centers[idx, 1]),
                float(target_radii[idx]),
                (30, 220, 80),
                4,
            )

    objectness = torch.sigmoid(pred_logits.detach().cpu())
    for idx in range(pred_cx.shape[0]):
        score = float(objectness[idx])
        if score < 0.15:
            continue
        color = (255, 70, 70) if score >= 0.5 else (255, 180, 40)
        circle(float(pred_cx[idx]), float(pred_cy[idx]), float(pred_radii[idx]), color, 3)
        px = int(float(pred_cx[idx]) * width)
        py = int(float(pred_cy[idx]) * height)
        draw.text((px + 6, py + 6), f"p{idx}:{score:.2f}", fill=color)

    label = "target=green pred=red/orange"
    if categories:
        label += f" | target: {', '.join(categories)}"
    label += f" | {prompt[:140]}"
    text_box_h = 38
    draw.rectangle((0, 0, width, text_box_h), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return pil


def _init_wandb(args):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("W&B logging requested but `wandb` is not installed.") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        config=vars(args),
    )
    url = run.get_url()
    if url:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "wandb_url.txt").write_text(url + "\n")
        print(f"[wandb] {url}")
    return wandb


def _log_wandb_examples(
    wandb,
    step: int,
    batch,
    cx: torch.Tensor,
    cy: torch.Tensor,
    radii: torch.Tensor,
    object_logits: torch.Tensor,
    args,
):
    if wandb is None or args.wandb_image_steps <= 0 or step % args.wandb_image_steps != 0:
        return
    count = min(args.wandb_num_images, batch["image"].shape[0])
    images = []
    for idx in range(count):
        overlay = _draw_foveation_overlay(
            batch["image"][idx],
            batch["target_centers"][idx],
            batch["target_radii"][idx],
            batch["target_valid"][idx],
            cx[idx].detach().cpu(),
            cy[idx].detach().cpu(),
            radii[idx].detach().cpu(),
            object_logits[idx].detach().cpu(),
            batch["prompt"][idx],
            batch["categories"][idx],
        )
        caption = f"id={int(batch['image_id'][idx])} | {batch['prompt'][idx]}"
        images.append(wandb.Image(overlay, caption=caption))
    wandb.log({"train/foveation_overlays": images}, step=step)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mask_size = args.mask_size or max(args.latent_size // 4, 1)
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

    device = torch.device(args.device)
    fpm = FovealPredictionModule(
        latent_channels=3,
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
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    print(f"[dataset] {len(dataset):,} samples from {args.coco_root}/{args.split}")
    print(f"[train] device={device} latent_size={args.latent_size} mask_size={mask_size}")
    wandb = _init_wandb(args)
    wandb_log_steps = args.wandb_log_steps or args.log_steps

    step = 0
    last_log = time.time()
    metric_sums = {}
    while step < args.max_steps:
        for batch in loader:
            step += 1
            image = batch["image"].to(device=device, non_blocking=True)
            proxy_latents = F.interpolate(
                image,
                size=(args.latent_size, args.latent_size),
                mode="bilinear",
                align_corners=False,
            )
            token_ids = batch["token_ids"].to(device=device, non_blocking=True)
            text_embeddings = text_embedder(token_ids)
            timesteps = torch.randint(0, 1000, (image.shape[0],), device=device)

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
                proxy_latents,
                text_embeddings,
                timesteps=timesteps,
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
