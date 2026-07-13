#!/usr/bin/env bash
# Launch noisy-FLUX-VAE-latent COCO FPM training on SPIKE.
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"
JOB_NAME="${JOB_NAME:-fovdiff-fpm-coco-latent-$(date +%s)}"

GIT_REPOSITORY="${GIT_REPOSITORY:-https://github.com/NeurAI-Lab/foveated_diffusion.git}"
GIT_REV="${GIT_REV:-fahad.sarfraz-codex/adaptive-foveation-policy}"
GIT_SECRET="${GIT_SECRET:-password-github-repo}"
GIT_MOUNT="${GIT_MOUNT:-/git/foveated_diffusion}"
COCO_ROOT="${COCO_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/bucket/foveated_diffusion/outputs/fpm_coco_latent/${JOB_NAME}}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-base-4B}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/data/bucket/foveated_diffusion/models}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-huggingface}"
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/data/bucket/foveated_diffusion/deps/DiffSynth-Studio}"
DIFFSYNTH_REPOSITORY="${DIFFSYNTH_REPOSITORY:-https://github.com/modelscope/DiffSynth-Studio.git}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-foveation-diffusion}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_NAME}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${JOB_NAME}}"
WANDB_IMAGE_STEPS="${WANDB_IMAGE_STEPS:-100}"

GPU_DEVICES="${GPU_DEVICES:-1}"
CPU_CORE_REQUEST="${CPU_CORE_REQUEST:-8.0}"
CPU_CORE_LIMIT="${CPU_CORE_LIMIT:-16.0}"
CPU_MEMORY_REQUEST="${CPU_MEMORY_REQUEST:-48G}"
CPU_MEMORY_LIMIT="${CPU_MEMORY_LIMIT:-96G}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_TRAIN_TIMESTEPS="${NUM_TRAIN_TIMESTEPS:-1000}"
NUM_FIXATIONS="${NUM_FIXATIONS:-3}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
TEXT_DIM="${TEXT_DIM:-64}"
TARGET_RADIUS_MARGIN="${TARGET_RADIUS_MARGIN:-1.0}"
TARGET_RADIUS_MIN="${TARGET_RADIUS_MIN:-0.03}"
TARGET_RADIUS_MAX="${TARGET_RADIUS_MAX:-0.45}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
VAE_DTYPE="${VAE_DTYPE:-bf16}"
LAMBDA_CENTER="${LAMBDA_CENTER:-2.0}"
LAMBDA_RADIUS="${LAMBDA_RADIUS:-1.0}"
LAMBDA_OBJECT="${LAMBDA_OBJECT:-1.0}"
LAMBDA_MAP="${LAMBDA_MAP:-1.0}"
LAMBDA_BUDGET="${LAMBDA_BUDGET:-0.5}"
LAMBDA_REPULSION="${LAMBDA_REPULSION:-0.05}"
LAMBDA_AREA="${LAMBDA_AREA:-0.0}"
TARGET_BUDGET_SCALE="${TARGET_BUDGET_SCALE:-1.0}"

WANDB_ARGS=""
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS="--use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${WANDB_RUN_NAME} --wandb_run_id ${WANDB_RUN_ID} --wandb_image_steps ${WANDB_IMAGE_STEPS}"
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS="${WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
  fi
fi

CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; export WANDB_DIR=/data/bucket/wandb; export WANDB_MODE=online; SRC_FILE=\"\"; for i in \$(seq 1 60); do SRC_FILE=\$(find ${GIT_MOUNT} -maxdepth 5 -type f -name train_fpm_coco_latent.py | head -1); test -n \"\$SRC_FILE\" && break; echo \"waiting for git-sync ${GIT_MOUNT} \$i\"; sleep 2; done; test -n \"\$SRC_FILE\" || { echo \"missing train_fpm_coco_latent.py in ${GIT_MOUNT}; push/check GIT_REPOSITORY and GIT_REV\"; find ${GIT_MOUNT} -maxdepth 4 -type f | head -20 || true; exit 66; }; SRC_DIR=\$(dirname \"\$SRC_FILE\"); cd \"\$SRC_DIR\"; test -d \"${COCO_ROOT}/train2017\" || { echo \"missing COCO train2017 under ${COCO_ROOT}\"; exit 67; }; mkdir -p \"${OUTPUT_DIR}\" /data/bucket/wandb \"${DIFFSYNTH_ROOT%/*}\" \"${MODEL_CACHE_DIR}\"; exec > >(tee -a \"${OUTPUT_DIR}/run.log\") 2>&1; python -m pip install -q -r requirements.txt; if ! python -c \"import diffsynth\" >/dev/null 2>&1; then if [ ! -d \"${DIFFSYNTH_ROOT}/.git\" ]; then rm -rf \"${DIFFSYNTH_ROOT}\"; git clone --depth 1 \"${DIFFSYNTH_REPOSITORY}\" \"${DIFFSYNTH_ROOT}\"; fi; python -m pip install -q -e \"${DIFFSYNTH_ROOT}\"; fi; python -c \"import torch, diffsynth\"; echo \"[deps] torch and diffsynth import ok\"; python train_fpm_coco_latent.py --coco_root \"${COCO_ROOT}\" --split train2017 --output_dir \"${OUTPUT_DIR}\" --model_id \"${MODEL_ID}\" --model_cache_dir \"${MODEL_CACHE_DIR}\" --download_source \"${DOWNLOAD_SOURCE}\" --image_size ${IMAGE_SIZE} --num_train_timesteps ${NUM_TRAIN_TIMESTEPS} --num_fixations ${NUM_FIXATIONS} --hidden_dim ${HIDDEN_DIM} --text_dim ${TEXT_DIM} --target_radius_margin ${TARGET_RADIUS_MARGIN} --target_radius_min ${TARGET_RADIUS_MIN} --target_radius_max ${TARGET_RADIUS_MAX} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --max_steps ${MAX_STEPS} --learning_rate ${LEARNING_RATE} --save_steps ${SAVE_STEPS} --vae_dtype ${VAE_DTYPE} --lambda_center ${LAMBDA_CENTER} --lambda_radius ${LAMBDA_RADIUS} --lambda_object ${LAMBDA_OBJECT} --lambda_map ${LAMBDA_MAP} --lambda_budget ${LAMBDA_BUDGET} --lambda_repulsion ${LAMBDA_REPULSION} --lambda_area ${LAMBDA_AREA} --target_budget_scale ${TARGET_BUDGET_SCALE} ${WANDB_ARGS}"

echo "[submit] ${JOB_NAME}"
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
  -e WANDB_PROJECT="${WANDB_PROJECT}" \
  -e WANDB_RUN_ID="${WANDB_RUN_ID}" \
  -e WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  --cpu-core-request "${CPU_CORE_REQUEST}" \
  --cpu-core-limit "${CPU_CORE_LIMIT}" \
  --cpu-memory-request "${CPU_MEMORY_REQUEST}" \
  --cpu-memory-limit "${CPU_MEMORY_LIMIT}" \
  --backoff-limit 0 \
  --command -- bash -lc "${CMD}"
