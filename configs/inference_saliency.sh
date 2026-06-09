#!/usr/bin/env bash
# Saliency-guided foveation: random_circular mask per prompt, saliency-trained LoRA.
# Flag set mirrors tests/test_saliency.sh.
set -e

LORA_SALIENCY=/path/to/saliency_lora/step-11000.safetensors
PROMPTS=/path/to/saliency_prompts.csv

python inference.py \
  --experiment foveation_trajectory_grid \
  --foveation_trajectory_type random_circular \
  --decode_mode merge \
  --soft_foveation_blend true \
  --foveation_outline false \
  --outline_width_frac 0.000 \
  --num_cols 1 \
  --orbit_radius 0.3 \
  --mask_radius 0.3 \
  --lora_checkpoint "$LORA_SALIENCY" \
  --lr_downsample_factor 2 \
  --prompt_dataset_path "$PROMPTS" \
  --num_prompts 100 \
  --seed 42 \
  --output_dir ./outputs/saliency_eval
