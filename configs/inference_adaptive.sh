#!/usr/bin/env bash
# Run our adaptive foveation policy on top of Chao's image pipeline.
set -e

LORA="${LORA:-/path/to/foveated_lora/step-5000.safetensors}"
PROMPTS="${PROMPTS:-/path/to/prompts.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/ours_adaptive_eval}"

python inference.py \
  --pipeline image \
  --experiment ours_adaptive \
  --decode_mode merge \
  --soft_foveation_blend true \
  --foveation_outline true \
  --full_eval \
  --lora_checkpoint "$LORA" \
  --lr_downsample_factor 2 \
  --adaptive_policy textfov_proxy \
  --adaptive_num_fixations 3 \
  --adaptive_beta_mode nafo_mean \
  --adaptive_beta_min 0.20 \
  --adaptive_beta_max 0.85 \
  --prompt_dataset_path "$PROMPTS" \
  --num_prompts 100 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR"
