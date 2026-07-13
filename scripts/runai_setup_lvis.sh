#!/usr/bin/env bash
# Seed LVIS annotations on the SPIKE foveation PVC with a CPU-only job.
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"
DATASET_ROOT="${DATASET_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation}"
JOB_NAME="${JOB_NAME:-fovdiff-lvis-$(date +%s)}"

CPU_CORE_REQUEST="${CPU_CORE_REQUEST:-2.0}"
CPU_CORE_LIMIT="${CPU_CORE_LIMIT:-4.0}"
CPU_MEMORY_REQUEST="${CPU_MEMORY_REQUEST:-8G}"
CPU_MEMORY_LIMIT="${CPU_MEMORY_LIMIT:-16G}"

CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; ROOT=${DATASET_ROOT}; ARCH=\"\$ROOT/_archives/lvis\"; LVIS=\"\$ROOT/lvis/annotations\"; mkdir -p \"\$ARCH\" \"\$LVIS\" /data/bucket/foveated_diffusion/logs; echo \"dataset root=\$ROOT\"; df -h /data /data/bucket || true; dl(){ url=\"\$1\"; out=\"\$2\"; mkdir -p \"\$(dirname \"\$out\")\"; if [ -s \"\$out\" ]; then echo \"[skip download] \$out\"; else echo \"[download] \$url -> \$out\"; curl -fL --retry 20 --retry-delay 30 -C - -o \"\$out\" \"\$url\"; fi; }; ex(){ zip=\"\$1\"; dest=\"\$2\"; marker=\"\$dest/.extracted_\$(basename \"\$zip\")\"; if [ -e \"\$marker\" ]; then echo \"[skip extract] \$zip\"; else echo \"[extract] \$zip -> \$dest\"; unzip -n \"\$zip\" -d \"\$dest\"; date > \"\$marker\"; fi; du -sh \"\$dest\" || true; }; for f in lvis_v1_train.json.zip lvis_v1_val.json.zip lvis_v1_image_info_test_dev.json.zip; do dl \"https://dl.fbaipublicfiles.com/LVIS/\$f\" \"\$ARCH/\$f\"; ex \"\$ARCH/\$f\" \"\$LVIS\"; done; python - <<'PY'
import json
from pathlib import Path
root = Path('${DATASET_ROOT}')
lvis = root / 'lvis' / 'annotations'
coco = root / 'coco'
checks = {}
for name in ['lvis_v1_train.json', 'lvis_v1_val.json', 'lvis_v1_image_info_test_dev.json']:
    path = lvis / name
    if not path.is_file():
        raise SystemExit(f'missing {path}')
    with path.open() as f:
        data = json.load(f)
    checks[name] = {
        'images': len(data.get('images', [])),
        'annotations': len(data.get('annotations', [])),
        'categories': len(data.get('categories', [])),
    }
if checks['lvis_v1_train.json']['categories'] < 1000:
    raise SystemExit('LVIS train categories unexpectedly low')
if not (coco / 'train2017').is_dir():
    raise SystemExit(f'missing COCO train images required by LVIS: {coco / \"train2017\"}')
if not (coco / 'val2017').is_dir():
    raise SystemExit(f'missing COCO val images required by LVIS: {coco / \"val2017\"}')
report_dir = root / 'verification'
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / '${JOB_NAME}.json'
report_path.write_text(json.dumps({'root': str(root), 'lvis_checks': checks}, indent=2) + '\\n')
print(json.dumps(checks, indent=2), flush=True)
print(f'[REPORT] {report_path}', flush=True)
print('[DONE] LVIS verification passed', flush=True)
PY"

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
