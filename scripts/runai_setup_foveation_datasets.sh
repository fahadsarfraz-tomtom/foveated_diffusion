#!/usr/bin/env bash
# Seed COCO and Visual Genome on the SPIKE foveation PVC with a CPU-only job.
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"
DATASET_ROOT="${DATASET_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation}"
JOB_NAME="${JOB_NAME:-fovdiff-datasets-$(date +%s)}"
SKIP_VG_IMAGES="${SKIP_VG_IMAGES:-0}"

CPU_CORE_REQUEST="${CPU_CORE_REQUEST:-4.0}"
CPU_CORE_LIMIT="${CPU_CORE_LIMIT:-8.0}"
CPU_MEMORY_REQUEST="${CPU_MEMORY_REQUEST:-16G}"
CPU_MEMORY_LIMIT="${CPU_MEMORY_LIMIT:-32G}"

vg_image_steps=""
if [[ "${SKIP_VG_IMAGES}" != "1" ]]; then
  vg_image_steps='dl https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip "$ARCH/visual_genome/images.zip"; ex "$ARCH/visual_genome/images.zip" "$ROOT/visual_genome"; dl https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip "$ARCH/visual_genome/images2.zip"; ex "$ARCH/visual_genome/images2.zip" "$ROOT/visual_genome";'
fi

CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; ROOT=${DATASET_ROOT}; ARCH=\"\$ROOT/_archives\"; mkdir -p \"\$ROOT/coco\" \"\$ROOT/visual_genome\" \"\$ARCH/coco\" \"\$ARCH/visual_genome\" /data/bucket/foveated_diffusion/logs; echo \"dataset root=\$ROOT\"; df -h /data /data/bucket || true; dl(){ url=\"\$1\"; out=\"\$2\"; mkdir -p \"\$(dirname \"\$out\")\"; if [ -s \"\$out\" ]; then echo \"[skip download] \$out\"; else echo \"[download] \$url -> \$out\"; curl -fL --retry 20 --retry-delay 30 -C - -o \"\$out\" \"\$url\"; fi; }; ex(){ zip=\"\$1\"; dest=\"\$2\"; marker=\"\$dest/.extracted_\$(basename \"\$zip\")\"; if [ -e \"\$marker\" ]; then echo \"[skip extract] \$zip\"; else echo \"[extract] \$zip -> \$dest\"; unzip -n \"\$zip\" -d \"\$dest\"; date > \"\$marker\"; fi; du -sh \"\$dest\" || true; }; dl http://images.cocodataset.org/zips/train2017.zip \"\$ARCH/coco/train2017.zip\"; ex \"\$ARCH/coco/train2017.zip\" \"\$ROOT/coco\"; dl http://images.cocodataset.org/zips/val2017.zip \"\$ARCH/coco/val2017.zip\"; ex \"\$ARCH/coco/val2017.zip\" \"\$ROOT/coco\"; dl http://images.cocodataset.org/annotations/annotations_trainval2017.zip \"\$ARCH/coco/annotations_trainval2017.zip\"; ex \"\$ARCH/coco/annotations_trainval2017.zip\" \"\$ROOT/coco\"; ${vg_image_steps} for f in image_data.json.zip region_descriptions.json.zip objects_v1_2.json.zip relationships_v1_2.json.zip attributes.json.zip scene_graphs.json.zip; do dl \"https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/\$f\" \"\$ARCH/visual_genome/\$f\"; ex \"\$ARCH/visual_genome/\$f\" \"\$ROOT/visual_genome\"; done; { echo \"created_at=\$(date -Iseconds)\"; echo \"root=\$ROOT\"; echo \"coco_root=\$ROOT/coco\"; echo \"visual_genome_root=\$ROOT/visual_genome\"; echo \"skip_vg_images=${SKIP_VG_IMAGES}\"; } > \"\$ROOT/dataset_manifest.txt\"; echo \"[done]\"; find \"\$ROOT\" -maxdepth 2 -type d | sort; du -sh \"\$ROOT\" \"\$ROOT/coco\" \"\$ROOT/visual_genome\" \"\$ARCH\" || true"

echo "[submit] ${JOB_NAME}"
runai training submit "${JOB_NAME}" \
  -p "${PROJECT}" \
  -i "${IMAGE}" \
  --preemptibility preemptible \
  --existing-pvc claimname=foveation,path=/data \
  --working-dir /data/bucket \
  -e PYTHONUNBUFFERED=1 \
  --cpu-core-request "${CPU_CORE_REQUEST}" \
  --cpu-core-limit "${CPU_CORE_LIMIT}" \
  --cpu-memory-request "${CPU_MEMORY_REQUEST}" \
  --cpu-memory-limit "${CPU_MEMORY_LIMIT}" \
  --backoff-limit 0 \
  --command -- bash -lc "${CMD}"
