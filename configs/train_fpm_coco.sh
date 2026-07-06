#!/usr/bin/env bash
# Phase 1: supervised COCO bootstrap for the Foveal Prediction Module.
set -e

COCO_ROOT="${COCO_ROOT:-./data/foveation/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-./models/fpm_coco}"

python train_fpm_coco.py \
  --coco_root "$COCO_ROOT" \
  --split train2017 \
  --output_dir "$OUTPUT_DIR" \
  --image_size 256 \
  --latent_size 64 \
  --num_fixations 3 \
  --hidden_dim 128 \
  --text_dim 64 \
  --batch_size 32 \
  --max_steps 20000 \
  --learning_rate 3e-4 \
  --save_steps 1000 \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT:-foveation-diffusion}" \
  --wandb_run_name "${WANDB_RUN_NAME:-fpm-coco-local}" \
  --wandb_image_steps "${WANDB_IMAGE_STEPS:-250}"
