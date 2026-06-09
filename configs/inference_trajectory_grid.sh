#!/usr/bin/env bash
# Generate a grid of images: N prompts (rows) x M foveation masks (cols).
# Flag set mirrors tests/test_trajectory.sh.
set -e

LORA=/path/to/foveated_lora/step-5000.safetensors
PROMPTS=/path/to/prompts.csv
TRAJECTORY="${TRAJECTORY:-spiral}"   # circular | spiral | polygons | multi_circle | grid | radius | random_circular

python inference.py \
  --experiment foveation_trajectory_grid \
  --foveation_trajectory_type "$TRAJECTORY" \
  --decode_mode merge \
  --soft_foveation_blend true \
  --foveation_outline false \
  --outline_width_frac 0.000 \
  --num_cols 4 \
  --orbit_radius 0.3 \
  --mask_radius 0.3 \
  --lora_checkpoint "$LORA" \
  --lr_downsample_factor 2 \
  --prompt_dataset_path "$PROMPTS" \
  --num_prompts 100 \
  --seed 42 \
  --output_dir "./outputs/trajectory_${TRAJECTORY}"
