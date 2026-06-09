#!/usr/bin/env bash
# Generate baselines (high-res + naive mixed-resolution) for comparison.
# Flag set mirrors tests/test_baseline.sh + tests/test_naive.sh.
set -e

LORA_NO_FOV=/path/to/no_fov_lora/step-15000.safetensors
PROMPTS=/path/to/prompts.csv

# Full-resolution baseline (no foveation).
python inference.py \
  --experiment high_res \
  --decode_mode direct \
  --lora_checkpoint "$LORA_NO_FOV" \
  --full_eval \
  --prompt_dataset_path "$PROMPTS" \
  --num_prompts 100 \
  --seed 42 \
  --output_dir ./outputs/baseline_high_res

# Naive mixed-resolution (no learned components).
python inference.py \
  --experiment naive_mixed_res \
  --decode_mode merge \
  --soft_foveation_blend true \
  --foveation_outline false \
  --lora_checkpoint "$LORA_NO_FOV" \
  --full_eval \
  --full_eval_mask square \
  --mask_radius 0.5 \
  --prompt_dataset_path "$PROMPTS" \
  --num_prompts 100 \
  --seed 42 \
  --output_dir ./outputs/baseline_naive
