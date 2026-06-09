#!/usr/bin/env bash
# Foveated Wan2.1-T2V-1.3B inference.
#
# Single config for all three video experiments; pick via --experiment:
#   high_res - vanilla Wan baseline (no foveation)
#   naive    - spline FoveationState, no LoRA (paper Fig.8 failure mode)
#   ours     - spline FoveationState + LoRA (paper headline result)
#
# When --lora_checkpoint is omitted, `ours` auto-downloads the random_path Wan
# LoRA from huggingface.co/bchao1/foveated-diffusion (video/fov_random_path.safetensors).
# To use a locally-trained LoRA (saliency, bbox, etc.) just pass --lora_checkpoint
# pointing at it; the inference code path is identical regardless of how it was trained.
set -e

EXPERIMENT="${EXPERIMENT:-ours}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/inference_video_${EXPERIMENT}}"

# Prompt defaults to the canonical Wan2.1-T2V-1.3B demo prompt when omitted
# (see DEFAULT_VIDEO_PROMPT in src/inference/args.py). Override via
# `PROMPT="..." bash configs/inference_video.sh`.
PROMPT_ARG=()
if [ -n "${PROMPT:-}" ]; then
  PROMPT_ARG=(--prompt "$PROMPT")
fi

python inference.py \
  --pipeline video \
  --experiment "$EXPERIMENT" \
  "${PROMPT_ARG[@]}" \
  --height 480 --width 832 --num_frames 81 \
  --num_inference_steps 50 \
  --cfg_scale 5.0 \
  --seed 0 \
  --foveation_trajectory spline \
  --output_dir "$OUTPUT_DIR"
