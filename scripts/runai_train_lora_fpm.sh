#!/usr/bin/env bash
# FGD-020: foveated FLUX2 LoRA training with FPM-predicted masks (vs `random` control).
#
# The experiment is the matched pair:
#   MODE=fpm    ./scripts/runai_train_lora_fpm.sh   # our predicted masks
#   MODE=random ./scripts/runai_train_lora_fpm.sh   # location-agnostic control (Chao-style)
# Same data, steps, seed, LoRA config — only the training-mask source differs.
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"

MODE="${MODE:-fpm}"
STEPS="${STEPS:-5000}"
SIZE="${SIZE:-512}"
NUM_IMAGES="${NUM_IMAGES:-20000}"
JOB_NAME="${JOB_NAME:-fgd020-lora-${MODE}-$(date +%s)}"

GIT_REPOSITORY="${GIT_REPOSITORY:-https://github.com/fahadsarfraz-tomtom/foveated_diffusion.git}"
GIT_REV="${GIT_REV:-fahad.sarfraz-codex/adaptive-foveation-policy}"
GIT_SECRET="${GIT_SECRET:-password-github-repo}"
GIT_MOUNT="${GIT_MOUNT:-/git/foveated_diffusion}"

COCO_ROOT="${COCO_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/bucket/foveated_diffusion/outputs/lora_fpm/${JOB_NAME}}"
METADATA_CSV="${METADATA_CSV:-/data/bucket/foveated_diffusion/datasets/foveation/coco_train_metadata_${NUM_IMAGES}.csv}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/data/bucket/foveated_diffusion/models}"
HF_HOME_DIR="${HF_HOME_DIR:-/data/bucket/foveated_diffusion/hf_cache}"
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/data/bucket/foveated_diffusion/deps/DiffSynth-Studio}"
DIFFSYNTH_REPOSITORY="${DIFFSYNTH_REPOSITORY:-https://github.com/modelscope/DiffSynth-Studio.git}"
FPM_CKPT="${FPM_CKPT:-/data/bucket/foveated_diffusion/outputs/fpm_coco_latent/fgd017-coco-latent-5k-1783937370/final.pt}"

WANDB_PROJECT="${WANDB_PROJECT:-foveation-diffusion}"

GPU_DEVICES="${GPU_DEVICES:-1}"
CPU_CORE_REQUEST="${CPU_CORE_REQUEST:-12.0}"
CPU_CORE_LIMIT="${CPU_CORE_LIMIT:-24.0}"
CPU_MEMORY_REQUEST="${CPU_MEMORY_REQUEST:-64G}"
CPU_MEMORY_LIMIT="${CPU_MEMORY_LIMIT:-128G}"

FPM_ARGS=""
if [[ "${MODE}" == "fpm" ]]; then
  FPM_ARGS="--fpm_checkpoint ${FPM_CKPT} --fpm_beta_min 0.15 --fpm_beta_max 0.45"
fi

CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; export WANDB_DIR=/data/bucket/wandb; export WANDB_MODE=online; export HF_HOME=${HF_HOME_DIR}; SRC_FILE=\"\"; for i in \$(seq 1 60); do SRC_FILE=\$(find ${GIT_MOUNT} -maxdepth 5 -type f -name train.py | head -1); test -n \"\$SRC_FILE\" && break; echo \"waiting for git-sync ${GIT_MOUNT} \$i\"; sleep 2; done; test -n \"\$SRC_FILE\" || { echo 'missing train.py in git mount'; exit 66; }; SRC_DIR=\$(dirname \"\$SRC_FILE\"); cd \"\$SRC_DIR\"; test -d \"${COCO_ROOT}/train2017\" || { echo 'missing COCO train2017'; exit 67; }; if [[ \"${MODE}\" == \"fpm\" ]]; then test -f \"${FPM_CKPT}\" || { echo 'missing FPM checkpoint'; exit 68; }; fi; mkdir -p \"${OUTPUT_DIR}\" /data/bucket/wandb \"${HF_HOME_DIR}\" \"${MODEL_CACHE_DIR}\" \"${DIFFSYNTH_ROOT%/*}\"; exec > >(tee -a \"${OUTPUT_DIR}/run.log\") 2>&1; python -m pip install -q -r requirements.txt; if ! python -c 'import diffsynth' >/dev/null 2>&1; then if [ ! -d \"${DIFFSYNTH_ROOT}/.git\" ]; then rm -rf \"${DIFFSYNTH_ROOT}\"; git clone --depth 1 \"${DIFFSYNTH_REPOSITORY}\" \"${DIFFSYNTH_ROOT}\"; fi; python -m pip install -q -e \"${DIFFSYNTH_ROOT}\"; fi; python -c 'import torch, diffsynth'; echo '[deps] ok'; test -f \"${METADATA_CSV}\" || python scripts/make_coco_train_metadata.py --coco_root \"${COCO_ROOT}\" --num_images ${NUM_IMAGES} --seed 0 --out \"${METADATA_CSV}\"; accelerate launch --num_processes 1 --mixed_precision bf16 train.py --pipeline image --dataset_base_path \"${COCO_ROOT}\" --dataset_metadata_path \"${METADATA_CSV}\" --max_pixels \$((${SIZE}*${SIZE})) --height ${SIZE} --width ${SIZE} --dataset_repeat 1 --dataset_num_workers 8 --model_id_with_origin_paths 'black-forest-labs/FLUX.2-klein-base-4B:text_encoder/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors' --tokenizer_path 'black-forest-labs/FLUX.2-klein-base-4B:tokenizer/' --learning_rate 1e-4 --num_epochs 5 --max_training_steps ${STEPS} --save_steps 1000 --remove_prefix_in_ckpt 'pipe.dit.' --output_path \"${OUTPUT_DIR}\" --lora_base_model dit --lora_target_modules 'to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj' --lora_rank 32 --use_gradient_checkpointing --task sft --decode_mode merge --seed 42 --foveated_training_mode ${MODE} ${FPM_ARGS} --lr_downsample_factor 2 --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${JOB_NAME}"

echo "[submit] ${JOB_NAME} (mode=${MODE}, steps=${STEPS}, size=${SIZE})"
runai training submit "${JOB_NAME}" \
  -p "${PROJECT}" \
  -i "${IMAGE}" \
  --preemptibility preemptible \
  --gpu-devices-request "${GPU_DEVICES}" \
  --existing-pvc claimname=foveation,path=/data \
  --git-sync name=foveated-diffusion,repository="${GIT_REPOSITORY}",path="${GIT_MOUNT}",secret="${GIT_SECRET}",rev="${GIT_REV}" \
  --new-pvc claimname=shm,storageclass=exascaler-ephemeral,size=64G,path=/dev/shm,accessmode-rwo,ephemeral \
  --working-dir /data/bucket \
  -e PYTHONUNBUFFERED=1 \
  -e WANDB_DIR=/data/bucket/wandb \
  -e HF_HOME="${HF_HOME_DIR}" \
  --cpu-core-request "${CPU_CORE_REQUEST}" \
  --cpu-core-limit "${CPU_CORE_LIMIT}" \
  --cpu-memory-request "${CPU_MEMORY_REQUEST}" \
  --cpu-memory-limit "${CPU_MEMORY_LIMIT}" \
  --backoff-limit 0 \
  --command -- bash -lc "${CMD}"
