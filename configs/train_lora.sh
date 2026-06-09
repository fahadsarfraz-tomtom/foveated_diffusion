#!/usr/bin/env bash
# Foveated FLUX2 LoRA fine-tuning (no token AE).
set -e

exp_name="foveated_lora_rank32"

accelerate launch train.py \
  --dataset_base_path /path/to/your/dataset \
  --dataset_metadata_path /path/to/your/dataset/metadata_train.csv \
  --max_pixels 1048576 \
  --height 1024 \
  --width 1024 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-base-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "black-forest-labs/FLUX.2-klein-base-4B:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/${exp_name}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --save_steps 1000 \
  --task sft \
  --find_unused_parameters \
  --decode_mode "merge" \
  --foveated_training_mode random \
  --lr_downsample_factor 2
