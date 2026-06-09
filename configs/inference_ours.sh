#!/usr/bin/env bash
# Run our foveated diffusion method on a CSV of prompts.
# Flag set mirrors tests/test_ours.sh.
set -e

LORA=/path/to/foveated_lora/step-5000.safetensors
PROMPTS=/path/to/prompts.csv

python inference.py \
  --experiment ours \
  --decode_mode merge \
  --soft_foveation_blend true \
  --foveation_outline true \
  --full_eval \
  --full_eval_mask square \
  --mask_radius 0.5 \
  --lora_checkpoint "$LORA" \
  --lr_downsample_factor 2 \
  --prompt_dataset_path "$PROMPTS" \
  --num_prompts 100 \
  --seed 42 \
  --output_dir ./outputs/ours_eval
