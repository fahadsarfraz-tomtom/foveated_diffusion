#!/usr/bin/env bash
# Launch the supervised COCO bootstrap for the Foveal Prediction Module on SPIKE.
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"
JOB_NAME="${JOB_NAME:-fovdiff-fpm-coco-$(date +%s)}"

GIT_REPOSITORY="${GIT_REPOSITORY:-https://github.com/NeurAI-Lab/foveated_diffusion.git}"
GIT_REV="${GIT_REV:-fahad.sarfraz-codex/adaptive-foveation-policy}"
GIT_SECRET="${GIT_SECRET:-password-github-repo}"
GIT_MOUNT="${GIT_MOUNT:-/git/foveated_diffusion}"
COCO_ROOT="${COCO_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/bucket/foveated_diffusion/outputs/fpm_coco/${JOB_NAME}}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-foveation-diffusion}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${JOB_NAME}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${JOB_NAME}}"
WANDB_IMAGE_STEPS="${WANDB_IMAGE_STEPS:-100}"

GPU_DEVICES="${GPU_DEVICES:-1}"
CPU_CORE_REQUEST="${CPU_CORE_REQUEST:-8.0}"
CPU_CORE_LIMIT="${CPU_CORE_LIMIT:-16.0}"
CPU_MEMORY_REQUEST="${CPU_MEMORY_REQUEST:-32G}"
CPU_MEMORY_LIMIT="${CPU_MEMORY_LIMIT:-64G}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
LATENT_SIZE="${LATENT_SIZE:-64}"
NUM_FIXATIONS="${NUM_FIXATIONS:-3}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
TEXT_DIM="${TEXT_DIM:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-20000}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"

WANDB_ARGS=""
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS="--use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${WANDB_RUN_NAME} --wandb_run_id ${WANDB_RUN_ID} --wandb_image_steps ${WANDB_IMAGE_STEPS}"
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS="${WANDB_ARGS} --wandb_entity ${WANDB_ENTITY}"
  fi
fi

CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; export WANDB_DIR=/data/bucket/wandb; export WANDB_MODE=online; SRC_FILE=\"\"; for i in \$(seq 1 60); do SRC_FILE=\$(find ${GIT_MOUNT} -maxdepth 5 -type f -name train_fpm_coco.py | head -1); test -n \"\$SRC_FILE\" && break; echo \"waiting for git-sync ${GIT_MOUNT} \$i\"; sleep 2; done; test -n \"\$SRC_FILE\" || { echo \"missing train_fpm_coco.py in ${GIT_MOUNT}; push/check GIT_REPOSITORY and GIT_REV\"; find ${GIT_MOUNT} -maxdepth 4 -type f | head -20 || true; exit 66; }; SRC_DIR=\$(dirname \"\$SRC_FILE\"); cd \"\$SRC_DIR\"; test -d \"${COCO_ROOT}/train2017\" || { echo \"missing COCO train2017 under ${COCO_ROOT}\"; exit 67; }; mkdir -p \"${OUTPUT_DIR}\" /data/bucket/wandb; python -m pip install -q -r requirements.txt; python train_fpm_coco.py --coco_root \"${COCO_ROOT}\" --split train2017 --output_dir \"${OUTPUT_DIR}\" --image_size ${IMAGE_SIZE} --latent_size ${LATENT_SIZE} --num_fixations ${NUM_FIXATIONS} --hidden_dim ${HIDDEN_DIM} --text_dim ${TEXT_DIM} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --max_steps ${MAX_STEPS} --learning_rate ${LEARNING_RATE} --save_steps ${SAVE_STEPS} ${WANDB_ARGS}"

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
