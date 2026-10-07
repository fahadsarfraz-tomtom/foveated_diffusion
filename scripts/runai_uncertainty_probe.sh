#!/usr/bin/env bash
# FGD-019 Phase 0: MC denoising-uncertainty premise probe on COCO val.
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"
JOB_NAME="${JOB_NAME:-fgd019-unc-probe-$(date +%s)}"

GIT_REPOSITORY="${GIT_REPOSITORY:-https://github.com/fahadsarfraz-tomtom/foveated_diffusion.git}"
GIT_REV="${GIT_REV:-fahad.sarfraz-codex/adaptive-foveation-policy}"
GIT_SECRET="${GIT_SECRET:-password-github-repo}"
GIT_MOUNT="${GIT_MOUNT:-/git/foveated_diffusion}"

COCO_ROOT="${COCO_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/bucket/foveated_diffusion/outputs/uncertainty_probe/${JOB_NAME}}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/data/bucket/foveated_diffusion/models}"
HF_HOME_DIR="${HF_HOME_DIR:-/data/bucket/foveated_diffusion/hf_cache}"
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/data/bucket/foveated_diffusion/deps/DiffSynth-Studio}"
DIFFSYNTH_REPOSITORY="${DIFFSYNTH_REPOSITORY:-https://github.com/modelscope/DiffSynth-Studio.git}"
FPM_CKPT="${FPM_CKPT:-/data/bucket/foveated_diffusion/outputs/fpm_coco_latent/fgd017-coco-latent-5k-1783937370/final.pt}"

NUM_IMAGES="${NUM_IMAGES:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
MC_SAMPLES="${MC_SAMPLES:-8}"
T_FRACS="${T_FRACS:-0.3 0.5 0.7}"
WANDB_PROJECT="${WANDB_PROJECT:-foveation-diffusion}"

CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; export WANDB_DIR=/data/bucket/wandb; export WANDB_MODE=online; export HF_HOME=${HF_HOME_DIR}; SRC_FILE=\"\"; for i in \$(seq 1 60); do SRC_FILE=\$(find ${GIT_MOUNT} -maxdepth 5 -type f -name mc_uncertainty_probe.py | head -1); test -n \"\$SRC_FILE\" && break; echo \"waiting for git-sync \$i\"; sleep 2; done; test -n \"\$SRC_FILE\" || exit 66; SRC_DIR=\$(dirname \$(dirname \"\$SRC_FILE\")); cd \"\$SRC_DIR\"; test -d \"${COCO_ROOT}/val2017\" || { echo 'missing COCO val2017'; exit 67; }; mkdir -p \"${OUTPUT_DIR}\" /data/bucket/wandb \"${HF_HOME_DIR}\" \"${MODEL_CACHE_DIR}\" \"${DIFFSYNTH_ROOT%/*}\"; exec > >(tee -a \"${OUTPUT_DIR}/run.log\") 2>&1; python -m pip install -q -r requirements.txt; python -m pip install -q scipy; python -m pip install -q git+https://github.com/openai/CLIP.git || true; python -m pip install -q git+https://github.com/matthias-k/DeepGaze.git || python -m pip install -q deepgaze-pytorch || echo '[warn] deepgaze install failed'; if ! python -c 'import diffsynth' >/dev/null 2>&1; then if [ ! -d \"${DIFFSYNTH_ROOT}/.git\" ]; then rm -rf \"${DIFFSYNTH_ROOT}\"; git clone --depth 1 \"${DIFFSYNTH_REPOSITORY}\" \"${DIFFSYNTH_ROOT}\"; fi; python -m pip install -q -e \"${DIFFSYNTH_ROOT}\"; fi; python -c 'import torch, diffsynth'; echo '[deps] ok'; python scripts/mc_uncertainty_probe.py --coco_root \"${COCO_ROOT}\" --num_images ${NUM_IMAGES} --image_size ${IMAGE_SIZE} --mc_samples ${MC_SAMPLES} --timestep_fracs ${T_FRACS} --fpm_checkpoint \"${FPM_CKPT}\" --model_cache_dir \"${MODEL_CACHE_DIR}\" --download_source huggingface --out_dir \"${OUTPUT_DIR}\" --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${JOB_NAME}"

echo "[submit] ${JOB_NAME}"
runai training submit "${JOB_NAME}" \
  -p "${PROJECT}" \
  -i "${IMAGE}" \
  --preemptibility preemptible \
  --gpu-devices-request 1 \
  --existing-pvc claimname=foveation,path=/data \
  --git-sync name=foveated-diffusion,repository="${GIT_REPOSITORY}",path="${GIT_MOUNT}",secret="${GIT_SECRET}",rev="${GIT_REV}" \
  --new-pvc claimname=shm,storageclass=exascaler-ephemeral,size=64G,path=/dev/shm,accessmode-rwo,ephemeral \
  --working-dir /data/bucket \
  -e PYTHONUNBUFFERED=1 \
  -e WANDB_DIR=/data/bucket/wandb \
  -e HF_HOME="${HF_HOME_DIR}" \
  --cpu-core-request 8.0 \
  --cpu-core-limit 16.0 \
  --cpu-memory-request 48G \
  --cpu-memory-limit 96G \
  --backoff-limit 0 \
  --command -- bash -lc "${CMD}"
