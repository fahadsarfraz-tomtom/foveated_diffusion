#!/usr/bin/env bash
# Launch FGD-018 (matched-budget mask-source comparison) on SPIKE.
#
# Arms share one HR token budget (BETA); only the fixation source differs:
#   center | random (prompt_hash) | saliency (DeepGaze IIE on high-res refs) | fpm
# plus a high_res reference pass. HPSv2.1 scoring runs in the same job.
#
# Smoke test first:  SMOKE=1 ./scripts/runai_eval_mask_sources.sh
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"

SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  JOB_NAME="${JOB_NAME:-fgd018-mask-cmp-smoke-$(date +%s)}"
  NUM_PROMPTS="${NUM_PROMPTS:-2}"
  NUM_STEPS="${NUM_STEPS:-20}"
else
  JOB_NAME="${JOB_NAME:-fgd018-mask-cmp-$(date +%s)}"
  NUM_PROMPTS="${NUM_PROMPTS:-100}"
  NUM_STEPS="${NUM_STEPS:-50}"
fi

GIT_REPOSITORY="${GIT_REPOSITORY:-https://github.com/fahadsarfraz-tomtom/foveated_diffusion.git}"
GIT_REV="${GIT_REV:-fahad.sarfraz-codex/adaptive-foveation-policy}"
GIT_SECRET="${GIT_SECRET:-password-github-repo}"
GIT_MOUNT="${GIT_MOUNT:-/git/foveated_diffusion}"

COCO_ROOT="${COCO_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/bucket/foveated_diffusion/outputs/mask_source_cmp/${JOB_NAME}}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/data/bucket/foveated_diffusion/models}"
HF_HOME_DIR="${HF_HOME_DIR:-/data/bucket/foveated_diffusion/hf_cache}"
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/data/bucket/foveated_diffusion/deps/DiffSynth-Studio}"
DIFFSYNTH_REPOSITORY="${DIFFSYNTH_REPOSITORY:-https://github.com/modelscope/DiffSynth-Studio.git}"

FPM_CKPT="${FPM_CKPT:-/data/bucket/foveated_diffusion/outputs/fpm_coco_latent/fgd017-coco-latent-5k-1783937370/final.pt}"
LORA_MODE="${LORA_MODE:-random}"     # Chao's released mask-location-agnostic image LoRA
ARMS="${ARMS:-center random saliency fpm}"
BETA="${BETA:-0.25}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
SEED="${SEED:-42}"
PROMPT_SEED="${PROMPT_SEED:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-foveation-diffusion}"

GPU_DEVICES="${GPU_DEVICES:-1}"
CPU_CORE_REQUEST="${CPU_CORE_REQUEST:-8.0}"
CPU_CORE_LIMIT="${CPU_CORE_LIMIT:-16.0}"
CPU_MEMORY_REQUEST="${CPU_MEMORY_REQUEST:-48G}"
CPU_MEMORY_LIMIT="${CPU_MEMORY_LIMIT:-96G}"

CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; export WANDB_DIR=/data/bucket/wandb; export WANDB_MODE=online; export HF_HOME=${HF_HOME_DIR}; SRC_FILE=\"\"; for i in \$(seq 1 60); do SRC_FILE=\$(find ${GIT_MOUNT} -maxdepth 5 -type f -name inference.py | head -1); test -n \"\$SRC_FILE\" && break; echo \"waiting for git-sync ${GIT_MOUNT} \$i\"; sleep 2; done; test -n \"\$SRC_FILE\" || { echo \"missing inference.py in ${GIT_MOUNT}; push/check GIT_REPOSITORY and GIT_REV\"; exit 66; }; SRC_DIR=\$(dirname \"\$SRC_FILE\"); cd \"\$SRC_DIR\"; test -f \"${FPM_CKPT}\" || { echo \"missing FPM checkpoint ${FPM_CKPT}\"; exit 67; }; mkdir -p \"${OUTPUT_DIR}\" /data/bucket/wandb \"${HF_HOME_DIR}\" \"${MODEL_CACHE_DIR}\" \"${DIFFSYNTH_ROOT%/*}\"; exec > >(tee -a \"${OUTPUT_DIR}/run.log\") 2>&1; python -m pip install -q -r requirements.txt; python -m pip install -q deepgaze-pytorch scipy || echo '[warn] deepgaze install failed'; python -m pip install -q hpsv2 || echo '[warn] hpsv2 install failed'; if ! python -c \"import diffsynth\" >/dev/null 2>&1; then if [ ! -d \"${DIFFSYNTH_ROOT}/.git\" ]; then rm -rf \"${DIFFSYNTH_ROOT}\"; git clone --depth 1 \"${DIFFSYNTH_REPOSITORY}\" \"${DIFFSYNTH_ROOT}\"; fi; python -m pip install -q -e \"${DIFFSYNTH_ROOT}\"; fi; python -c \"import torch, diffsynth\"; echo '[deps] ok'; python scripts/make_coco_prompt_csv.py --coco_root \"${COCO_ROOT}\" --num_prompts ${NUM_PROMPTS} --seed ${PROMPT_SEED} --out \"${OUTPUT_DIR}/prompts.csv\"; python inference.py --pipeline image --experiment mask_source_comparison --decode_mode merge --soft_foveation_blend true --lora_mode ${LORA_MODE} --full_eval --prompt_dataset_path \"${OUTPUT_DIR}/prompts.csv\" --num_prompts ${NUM_PROMPTS} --num_inference_steps ${NUM_STEPS} --height ${HEIGHT} --width ${WIDTH} --seed ${SEED} --comparison_beta ${BETA} --comparison_arms ${ARMS} --fpm_checkpoint \"${FPM_CKPT}\" --output_dir \"${OUTPUT_DIR}\"; python scripts/eval_hps.py --run_dir \"${OUTPUT_DIR}\" --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${JOB_NAME}-hps || echo '[warn] HPS eval failed — images are saved, rerun scripts/eval_hps.py later'"

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
  -e HF_HOME="${HF_HOME_DIR}" \
  --cpu-core-request "${CPU_CORE_REQUEST}" \
  --cpu-core-limit "${CPU_CORE_LIMIT}" \
  --cpu-memory-request "${CPU_MEMORY_REQUEST}" \
  --cpu-memory-limit "${CPU_MEMORY_LIMIT}" \
  --backoff-limit 0 \
  --command -- bash -lc "${CMD}"
