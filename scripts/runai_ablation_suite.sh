#!/usr/bin/env bash
# FGD-021 (NaFo + periphery-resolution ablations) and FGD-022 (coverage refinement).
#
#   SUITE=nafo     ./scripts/runai_ablation_suite.sh   # D2: beta-mode grid + D3-lite: lr_factor grid
#   SUITE=coverage ./scripts/runai_ablation_suite.sh   # D5xD6: single vs two-pass coverage refinement
#
# NaFo caveat (recorded): Chao's image path packs tokens once, so beta modes are
# static projections (fixed / nafo_mean / nafo_early / nafo_late), not per-step
# schedules. lr_factor {2,4} ablates the periphery resolution level (2-level
# engine; a true 3-level pyramid needs engine work).
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"
SUITE="${SUITE:-nafo}"
JOB_NAME="${JOB_NAME:-fgd02x-${SUITE}-$(date +%s)}"
case "${SUITE}" in
  nafo) JOB_NAME="${JOB_NAME/fgd02x/fgd021}";;
  coverage) JOB_NAME="${JOB_NAME/fgd02x/fgd022}";;
  *) echo "unknown SUITE=${SUITE}"; exit 2;;
esac

GIT_REPOSITORY="${GIT_REPOSITORY:-https://github.com/fahadsarfraz-tomtom/foveated_diffusion.git}"
GIT_REV="${GIT_REV:-fahad.sarfraz-codex/adaptive-foveation-policy}"
GIT_SECRET="${GIT_SECRET:-password-github-repo}"
GIT_MOUNT="${GIT_MOUNT:-/git/foveated_diffusion}"

COCO_ROOT="${COCO_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation/coco}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/bucket/foveated_diffusion/outputs/ablations/${JOB_NAME}}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/data/bucket/foveated_diffusion/models}"
HF_HOME_DIR="${HF_HOME_DIR:-/data/bucket/foveated_diffusion/hf_cache}"
DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/data/bucket/foveated_diffusion/deps/DiffSynth-Studio}"
DIFFSYNTH_REPOSITORY="${DIFFSYNTH_REPOSITORY:-https://github.com/modelscope/DiffSynth-Studio.git}"
FPM_CKPT="${FPM_CKPT:-/data/bucket/foveated_diffusion/outputs/fpm_coco_latent/fgd017-coco-latent-5k-1783937370/final.pt}"

NUM_PROMPTS="${NUM_PROMPTS:-100}"
STEPS="${STEPS:-50}"
POLICY="${POLICY:-center}"
BETA="${BETA:-0.25}"
WANDB_PROJECT="${WANDB_PROJECT:-foveation-diffusion}"

COMMON="--pipeline image --decode_mode merge --soft_foveation_blend true --lora_mode random --model_cache_dir ${MODEL_CACHE_DIR} --download_source huggingface --full_eval --prompt_dataset_path ${OUTPUT_DIR}/prompts.csv --num_prompts ${NUM_PROMPTS} --height 1024 --width 1024 --seed 42"

if [[ "${SUITE}" == "nafo" ]]; then
  RUN_BLOCK="for CFG in fixed:2 nafo_mean:2 nafo_early:2 nafo_late:2 fixed:4 nafo_mean:4; do BM=\${CFG%%:*}; LF=\${CFG##*:}; SUB=\"${OUTPUT_DIR}/\${BM}_lr\${LF}\"; mkdir -p \"\$SUB\"; python \"\$SRC_DIR/inference.py\" ${COMMON} --experiment ours_adaptive --adaptive_policy ${POLICY} --adaptive_beta_mode \$BM --adaptive_fixed_beta ${BETA} --adaptive_beta_min 0.20 --adaptive_beta_max 0.85 --lr_downsample_factor \$LF --num_inference_steps ${STEPS} --fpm_checkpoint ${FPM_CKPT} --output_dir \"\$SUB\"; done"
else
  RUN_BLOCK="python \"\$SRC_DIR/inference.py\" ${COMMON} --experiment coverage_refine --adaptive_policy ${POLICY} --comparison_beta ${BETA} --coverage_pass1_steps 30 --coverage_pass2_steps 20 --coverage_strength 0.4 --coverage_decay 0.8 --uncertainty_mc_samples 8 --uncertainty_timestep_frac 0.5 --fpm_checkpoint ${FPM_CKPT} --output_dir \"${OUTPUT_DIR}\""
fi

CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; export WANDB_DIR=/data/bucket/wandb; export WANDB_MODE=online; export HF_HOME=${HF_HOME_DIR}; export PATH=\"\$HOME/.local/bin:\$PATH\"; SRC_FILE=\"\"; for i in \$(seq 1 60); do SRC_FILE=\$(find ${GIT_MOUNT} -maxdepth 5 -type f -name inference.py | head -1); test -n \"\$SRC_FILE\" && break; sleep 2; done; test -n \"\$SRC_FILE\" || exit 66; SRC_DIR=\$(dirname \"\$SRC_FILE\"); mkdir -p \"${OUTPUT_DIR}\" /data/bucket/wandb; exec > >(tee -a \"${OUTPUT_DIR}/run.log\") 2>&1; cd \"\$SRC_DIR\"; python -m pip install -q -r requirements.txt; python -m pip install -q scipy; python -m pip install -q git+https://github.com/openai/CLIP.git || true; python -m pip install -q git+https://github.com/matthias-k/DeepGaze.git || python -m pip install -q deepgaze-pytorch || true; python -m pip install -q hpsv2 || true; HPS_FACTORY=\$(python -c 'import hpsv2, os; print(os.path.join(os.path.dirname(hpsv2.__file__), \"src\", \"open_clip\", \"factory.py\"))' 2>/dev/null || true); test -n \"\$HPS_FACTORY\" && sed -i '/from turtle import forward/d' \"\$HPS_FACTORY\" || true; if ! python -c 'import diffsynth' >/dev/null 2>&1; then if [ ! -d \"${DIFFSYNTH_ROOT}/.git\" ]; then rm -rf \"${DIFFSYNTH_ROOT}\"; git clone --depth 1 \"${DIFFSYNTH_REPOSITORY}\" \"${DIFFSYNTH_ROOT}\"; fi; python -m pip install -q -e \"${DIFFSYNTH_ROOT}\"; fi; python -c 'import torch, diffsynth'; echo '[deps] ok'; test -f \"${OUTPUT_DIR}/prompts.csv\" || python \"\$SRC_DIR/scripts/make_coco_prompt_csv.py\" --coco_root \"${COCO_ROOT}\" --num_prompts ${NUM_PROMPTS} --seed 0 --out \"${OUTPUT_DIR}/prompts.csv\"; ${RUN_BLOCK}; python \"\$SRC_DIR/scripts/eval_hps.py\" --run_dir \"${OUTPUT_DIR}\" --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${JOB_NAME}-hps || echo '[warn] HPS eval failed — rerun scripts/eval_hps.py later'"

echo "[submit] ${JOB_NAME} (suite=${SUITE})"
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
