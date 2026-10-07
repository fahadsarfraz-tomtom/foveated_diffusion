#!/usr/bin/env python3
"""FGD-019 Phase 0 — premise probe for uncertainty-guided foveation.

Hypothesis under test: the regions worth foveating are the regions where the
frozen generator's denoising uncertainty is high. For N COCO val images this
script computes MC-variance uncertainty maps from the frozen FLUX2 pipeline
and compares them against three reference signals at the token grid:

  - DeepGaze IIE saliency (the external-saliency baseline FGD-018 uses)
  - COCO ground-truth instance boxes (object regions)
  - the trained FPM's predicted weight map (fgd017 checkpoint, content mode)

Reports pairwise Spearman rank correlations and IoU of top-beta regions per
timestep fraction, plus qualitative panels. No training anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules["sageattention"] = None

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco_root", type=Path, required=True)
    p.add_argument("--split", type=str, default="val2017")
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--image_size", type=int, default=512,
                   help="Square resize; must be a multiple of 16.")
    p.add_argument("--timestep_fracs", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--mc_samples", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.25, help="Top-beta for IoU.")
    p.add_argument("--fpm_checkpoint", type=str, default=None)
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--model_cache_dir", type=str, default=None)
    p.add_argument("--download_source", type=str, default="huggingface")
    p.add_argument("--num_panels", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="foveation-diffusion")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def load_coco_samples(coco_root: Path, split: str, num_images: int, seed: int):
    """Return [(image_path, caption, boxes_xywh, (w, h))] for images with >=1 box."""
    captions = json.loads((coco_root / "annotations" / f"captions_{split}.json").read_text())
    instances = json.loads((coco_root / "annotations" / f"instances_{split}.json").read_text())

    caption_by_image = {}
    for ann in captions["annotations"]:
        caption_by_image.setdefault(ann["image_id"], ann["caption"])
    boxes_by_image = {}
    for ann in instances["annotations"]:
        boxes_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])
    meta_by_image = {img["id"]: img for img in instances["images"]}

    ids = sorted(set(caption_by_image) & set(boxes_by_image))
    random.Random(seed).shuffle(ids)
    samples = []
    for image_id in ids[: num_images]:
        meta = meta_by_image[image_id]
        samples.append((
            coco_root / split / meta["file_name"],
            " ".join(caption_by_image[image_id].split()),
            boxes_by_image[image_id],
            (meta["width"], meta["height"]),
        ))
    return samples


def boxes_to_grid_mask(boxes, orig_wh, grid_hw) -> np.ndarray:
    w0, h0 = orig_wh
    gh, gw = grid_hw
    mask = np.zeros((gh, gw), dtype=np.float32)
    for x, y, bw, bh in boxes:
        x0 = int(np.clip(x / w0 * gw, 0, gw - 1))
        y0 = int(np.clip(y / h0 * gh, 0, gh - 1))
        x1 = int(np.clip((x + bw) / w0 * gw, x0 + 1, gw))
        y1 = int(np.clip((y + bh) / h0 * gh, y0 + 1, gh))
        mask[y0:y1, x0:x1] = 1.0
    return mask


def deepgaze_density_map(model, pil: Image.Image, device, grid_hw) -> np.ndarray:
    arr = np.array(pil).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    centerbias = torch.zeros(1, pil.size[1], pil.size[0], device=device)
    with torch.no_grad():
        log_density = model(tensor, centerbias)
    density = torch.exp(log_density[0, 0]).float().cpu()[None, None]
    return F.interpolate(density, size=grid_hw, mode="bilinear", align_corners=False)[0, 0].numpy()


def to_panel(maps: dict, pil: Image.Image, grid_hw) -> Image.Image:
    """Side-by-side grayscale panel: image thumbnail + normalized maps."""
    gh, gw = grid_hw
    tile = 128
    thumb = pil.resize((tile, tile))
    tiles = [thumb]
    for name, m in maps.items():
        m = np.asarray(m, dtype=np.float32)
        rng = m.max() - m.min()
        m = (m - m.min()) / rng if rng > 0 else m * 0
        img = Image.fromarray((m * 255).astype(np.uint8)).resize((tile, tile), Image.NEAREST)
        tiles.append(img.convert("RGB"))
    panel = Image.new("RGB", (tile * len(tiles), tile))
    for i, t in enumerate(tiles):
        panel.paste(t, (i * tile, 0))
    return panel


def main():
    args = parse_args()
    assert args.image_size % 16 == 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from src.inference.pipeline_loader import load_pipeline
    from src.masks.saliency import load_deepgaze_model
    from src.masks.uncertainty import (
        iou_top_beta, mc_uncertainty_map, prepare_conditioning, spearman,
    )

    pipe_args = SimpleNamespace(
        model_id=args.model_id, lora_checkpoint=None, lora_mode=None,
        dit_checkpoint=None, experiment="uncertainty_probe",
        model_cache_dir=args.model_cache_dir, download_source=args.download_source,
    )
    pipe = load_pipeline(pipe_args)

    deepgaze = load_deepgaze_model(pipe.device)

    fpm_policy = None
    if args.fpm_checkpoint:
        from src.masks.fpm_policy import FpmMaskPolicy
        fpm_policy = FpmMaskPolicy.load(args.fpm_checkpoint, device=pipe.device)

    samples = load_coco_samples(args.coco_root, args.split, args.num_images, args.seed)
    print(f"[probe] {len(samples)} COCO {args.split} images, size={args.image_size}, "
          f"K={args.mc_samples}, t_fracs={args.timestep_fracs}")

    grid = (args.image_size // 16, args.image_size // 16)
    rows = []
    for idx, (path, caption, boxes, orig_wh) in enumerate(samples):
        pil = Image.open(path).convert("RGB").resize(
            (args.image_size, args.image_size), Image.BICUBIC,
        )
        sal = deepgaze_density_map(deepgaze, pil, pipe.device, grid)
        box_mask = boxes_to_grid_mask(boxes, orig_wh, grid)

        conditioning = None
        for t_frac in args.timestep_fracs:
            umap, extras = mc_uncertainty_map(
                pipe, pil, caption, timestep_frac=t_frac,
                num_samples=args.mc_samples, seed=args.seed + idx,
                conditioning=conditioning,
            )
            conditioning = extras["conditioning"]
            unc = umap.numpy()

            maps = {"uncertainty": unc, "saliency": sal, "boxes": box_mask}
            if fpm_policy is not None:
                noisy = extras["x1_noise_latent"].to(pipe.device)
                noisy_spatial = noisy.transpose(1, 2).reshape(
                    1, noisy.shape[-1], grid[0], grid[1],
                )
                fpm_map = fpm_policy.predict_weight_map_from_latents(
                    noisy_spatial, caption, extras["timestep_index"],
                    out_height=grid[0], out_width=grid[1],
                ).float().cpu().numpy()
                maps["fpm"] = fpm_map

            row = {"image": str(path.name), "prompt": caption, "t_frac": t_frac}
            names = list(maps)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    key = f"{names[i]}_vs_{names[j]}"
                    row[f"spearman_{key}"] = spearman(maps[names[i]], maps[names[j]])
                    row[f"iou{args.beta}_{key}"] = iou_top_beta(
                        maps[names[i]], maps[names[j]], args.beta,
                    )
            rows.append(row)

            if idx < args.num_panels and t_frac == args.timestep_fracs[len(args.timestep_fracs) // 2]:
                to_panel(maps, pil, grid).save(
                    args.out_dir / f"panel_{idx:03d}_t{t_frac:.1f}.png"
                )
        print(f"[probe] {idx + 1}/{len(samples)} done: {path.name}")

    import csv
    metric_keys = [k for k in rows[0] if k.startswith(("spearman_", "iou"))]
    with open(args.out_dir / "probe_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"num_images": len(samples), "mc_samples": args.mc_samples,
               "image_size": args.image_size, "beta": args.beta, "by_t_frac": {}}
    for t_frac in args.timestep_fracs:
        sub = [r for r in rows if r["t_frac"] == t_frac]
        summary["by_t_frac"][str(t_frac)] = {
            k: {"mean": statistics.fmean(r[k] for r in sub),
                "std": statistics.pstdev([r[k] for r in sub]) if len(sub) > 1 else 0.0}
            for k in metric_keys if all(k in r for r in sub)
        }
    (args.out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.use_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=args.wandb_run_name or "fgd019-uncertainty-probe",
                         job_type="probe", config=vars(args) | {"out_dir": str(args.out_dir)})
        flat = {}
        for t_frac, metrics in summary["by_t_frac"].items():
            for k, v in metrics.items():
                flat[f"t{t_frac}/{k}"] = v["mean"]
        run.log(flat)
        for png in sorted(args.out_dir.glob("panel_*.png")):
            run.log({f"panels/{png.stem}": wandb.Image(str(png))})
        run.finish()


if __name__ == "__main__":
    main()
