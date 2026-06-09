#!/usr/bin/env bash
# Foveated Wan2.1-T2V-1.3B LoRA fine-tuning — random_path mask sampler (default).
# Matches the paper's video training config (480x832, 81 frames, LoRA r=32).
set -e

exp_name="wan_video_foveated_lora_rank32"

accelerate launch train.py \
  --pipeline video \
  --dataset_base_path /path/to/your/video/dataset \
  --dataset_metadata_path /path/to/your/video/metadata_train.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --tokenizer_path "Wan-AI/Wan2.1-T2V-1.3B:google/umt5-xxl/" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/${exp_name}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --save_steps 1000 \
  --task sft \
  --use_wandb \
  --wandb_project foveated-diffusion-video \
  --log_video_steps 5000 \
  --find_unused_parameters
