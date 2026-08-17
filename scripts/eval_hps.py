#!/usr/bin/env python3
"""Score mask_source_comparison outputs with HPSv2.1 and summarize per-arm quality.

Reads each arm directory's ``metadata_*.csv`` (columns: image, prompt, ...),
scores every image with HPSv2.1, and writes:

- ``hps_scores.csv``  — one row per (arm, image) with the score
- ``hps_summary.json`` — per-arm n/mean/std plus paired per-prompt deltas
  against the ``center`` arm (and against ``saliency`` when present)

Per this project's evaluation protocol, HPSv2.1 is the primary metric; FID and
CLIP-IQA are known-unreliable in the mixed-resolution regime and are not
computed here.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                        help="mask_source_comparison output dir (contains one subdir per arm)")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Arm subdirs to score. Default: every subdir with a metadata csv.")
    parser.add_argument("--hps_version", default="v2.1")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="foveation-diffusion")
    parser.add_argument("--wandb_run_name", default=None)
    return parser.parse_args()


def discover_arms(run_dir: str) -> list[str]:
    arms = []
    for entry in sorted(os.listdir(run_dir)):
        arm_dir = os.path.join(run_dir, entry)
        if os.path.isdir(arm_dir) and glob.glob(os.path.join(arm_dir, "metadata_*.csv")):
            arms.append(entry)
    return arms


def load_arm_rows(run_dir: str, arm: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(run_dir, arm, "metadata_*.csv")))]
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = arm
    return df


def main():
    args = parse_args()
    import hpsv2  # deferred: heavy import, downloads the HPS checkpoint on first use

    arms = args.arms or discover_arms(args.run_dir)
    if not arms:
        raise SystemExit(f"no arm directories with metadata found under {args.run_dir}")
    print(f"[eval_hps] scoring arms: {arms}")

    score_rows = []
    per_arm_scores: dict[str, dict[str, float]] = {}
    for arm in arms:
        df = load_arm_rows(args.run_dir, arm)
        arm_scores = {}
        for _, row in df.iterrows():
            image_path = os.path.join(args.run_dir, arm, row["image"])
            if not os.path.exists(image_path):
                print(f"[eval_hps] MISSING {image_path} — skipped")
                continue
            result = hpsv2.score(image_path, str(row["prompt"]), hps_version=args.hps_version)
            score = float(result[0]) if isinstance(result, (list, tuple)) else float(result)
            arm_scores[row["image"]] = score
            score_rows.append({
                "arm": arm, "image": row["image"], "prompt": row["prompt"], "hps": score,
            })
        per_arm_scores[arm] = arm_scores
        print(f"[eval_hps] {arm}: {len(arm_scores)} images scored")

    pd.DataFrame(score_rows).to_csv(os.path.join(args.run_dir, "hps_scores.csv"), index=False)

    summary: dict = {"hps_version": args.hps_version, "arms": {}}
    for arm, scores in per_arm_scores.items():
        values = list(scores.values())
        summary["arms"][arm] = {
            "n": len(values),
            "mean": statistics.fmean(values) if values else math.nan,
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    for reference in ("center", "saliency"):
        if reference not in per_arm_scores:
            continue
        ref_scores = per_arm_scores[reference]
        deltas_block = {}
        for arm, scores in per_arm_scores.items():
            if arm == reference:
                continue
            paired = [scores[k] - ref_scores[k] for k in scores if k in ref_scores]
            if not paired:
                continue
            deltas_block[arm] = {
                "n_pairs": len(paired),
                "mean_delta": statistics.fmean(paired),
                "std_delta": statistics.pstdev(paired) if len(paired) > 1 else 0.0,
                "win_rate": sum(d > 0 for d in paired) / len(paired),
            }
        summary[f"paired_vs_{reference}"] = deltas_block

    with open(os.path.join(args.run_dir, "hps_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if args.use_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"hps-eval-{os.path.basename(os.path.normpath(args.run_dir))}",
            job_type="eval",
            config={"run_dir": args.run_dir, "hps_version": args.hps_version},
        )
        flat = {f"hps/{arm}/mean": s["mean"] for arm, s in summary["arms"].items()}
        flat.update({f"hps/{arm}/n": s["n"] for arm, s in summary["arms"].items()})
        for ref_key in ("paired_vs_center", "paired_vs_saliency"):
            for arm, s in summary.get(ref_key, {}).items():
                flat[f"{ref_key}/{arm}/mean_delta"] = s["mean_delta"]
                flat[f"{ref_key}/{arm}/win_rate"] = s["win_rate"]
        run.log(flat)
        run.finish()


if __name__ == "__main__":
    main()
