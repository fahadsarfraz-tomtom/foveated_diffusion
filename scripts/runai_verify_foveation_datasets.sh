#!/usr/bin/env bash
# Verify COCO and Visual Genome on the SPIKE foveation PVC with a CPU-only job.
set -euo pipefail

PROJECT="${PROJECT:-foveation}"
IMAGE="${IMAGE:-harbor.spike.tue.nl/foveation/trm_foveation_b200_w_claude:0.09}"
DATASET_ROOT="${DATASET_ROOT:-/data/bucket/foveated_diffusion/datasets/foveation}"
JOB_NAME="${JOB_NAME:-fovdiff-dataset-verify-$(date +%s)}"

CPU_CORE_REQUEST="${CPU_CORE_REQUEST:-2.0}"
CPU_CORE_LIMIT="${CPU_CORE_LIMIT:-4.0}"
CPU_MEMORY_REQUEST="${CPU_MEMORY_REQUEST:-8G}"
CPU_MEMORY_LIMIT="${CPU_MEMORY_LIMIT:-16G}"

CMD="set -euo pipefail; python - <<'PY'
import json
import os
from pathlib import Path
import random
import subprocess
from PIL import Image

root = Path('${DATASET_ROOT}')
report_dir = root / 'verification'
report_dir.mkdir(parents=True, exist_ok=True)
report = {'root': str(root), 'checks': {}}

def require(name, condition, detail):
    report['checks'][name] = {'ok': bool(condition), 'detail': detail}
    if not condition:
        raise SystemExit(f'[FAIL] {name}: {detail}')
    print(f'[OK] {name}: {detail}', flush=True)

def count_jpgs(path):
    return sum(1 for _ in path.glob('*.jpg'))

coco = root / 'coco'
vg = root / 'visual_genome'
archives = root / '_archives'

require('root_exists', root.exists(), str(root))
require('coco_train_dir', (coco / 'train2017').is_dir(), str(coco / 'train2017'))
require('coco_val_dir', (coco / 'val2017').is_dir(), str(coco / 'val2017'))
require('coco_annotations_dir', (coco / 'annotations').is_dir(), str(coco / 'annotations'))
require('vg_dir', vg.is_dir(), str(vg))

train_count = count_jpgs(coco / 'train2017')
val_count = count_jpgs(coco / 'val2017')
require('coco_train_count', train_count == 118287, train_count)
require('coco_val_count', val_count == 5000, val_count)

for name in ['instances_train2017.json', 'captions_train2017.json', 'instances_val2017.json', 'captions_val2017.json']:
    path = coco / 'annotations' / name
    require(f'coco_json_exists_{name}', path.is_file(), str(path))
    with path.open() as f:
        data = json.load(f)
    require(f'coco_json_parse_{name}', 'images' in data and 'annotations' in data, {'images': len(data.get('images', [])), 'annotations': len(data.get('annotations', []))})

vg_image_dirs = [p for p in [vg / 'VG_100K', vg / 'VG_100K_2'] if p.is_dir()]
vg_image_count = sum(count_jpgs(p) for p in vg_image_dirs)
require('vg_image_dirs', len(vg_image_dirs) >= 1, [str(p) for p in vg_image_dirs])
require('vg_image_count_min', vg_image_count >= 100000, vg_image_count)

for name in ['image_data.json', 'region_descriptions.json', 'objects.json', 'relationships.json', 'attributes.json', 'scene_graphs.json']:
    path = vg / name
    if not path.exists() and name == 'objects.json':
        path = vg / 'objects_v1_2.json'
    if not path.exists() and name == 'relationships.json':
        path = vg / 'relationships_v1_2.json'
    require(f'vg_json_exists_{name}', path.is_file(), str(path))
    with path.open() as f:
        data = json.load(f)
    require(f'vg_json_parse_{path.name}', isinstance(data, list) and len(data) > 1000, len(data) if isinstance(data, list) else type(data).__name__)

archive_files = [
    archives / 'coco' / 'train2017.zip',
    archives / 'coco' / 'val2017.zip',
    archives / 'coco' / 'annotations_trainval2017.zip',
    archives / 'visual_genome' / 'images.zip',
    archives / 'visual_genome' / 'images2.zip',
    archives / 'visual_genome' / 'image_data.json.zip',
    archives / 'visual_genome' / 'region_descriptions.json.zip',
    archives / 'visual_genome' / 'objects_v1_2.json.zip',
    archives / 'visual_genome' / 'relationships_v1_2.json.zip',
    archives / 'visual_genome' / 'attributes.json.zip',
    archives / 'visual_genome' / 'scene_graphs.json.zip',
]
for archive in archive_files:
    require(f'archive_exists_{archive.name}', archive.is_file() and archive.stat().st_size > 0, str(archive))
    subprocess.run(['unzip', '-tq', str(archive)], check=True)
    print(f'[OK] archive_integrity_{archive.name}', flush=True)

sample_images = list((coco / 'val2017').glob('*.jpg'))[:20]
for p in random.sample(sample_images, min(5, len(sample_images))):
    with Image.open(p) as img:
        img.verify()
require('pil_decode_coco_samples', True, len(sample_images))

report_path = report_dir / '${JOB_NAME}.json'
report_path.write_text(json.dumps(report, indent=2) + '\\n')
print(f'[REPORT] {report_path}', flush=True)
print('[DONE] dataset verification passed', flush=True)
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
