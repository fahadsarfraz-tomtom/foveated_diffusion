import json

import numpy as np
from PIL import Image

from src.training.lvis_fpm import LvisFoveationDataset


def test_lvis_foveation_dataset_builds_synthetic_prompt_and_targets(tmp_path):
    root = tmp_path / "data"
    lvis_ann = root / "lvis" / "annotations"
    image_dir = root / "coco" / "train2017"
    lvis_ann.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    image = np.zeros((80, 100, 3), dtype=np.uint8)
    Image.fromarray(image).save(image_dir / "000000000001.jpg")

    lvis = {
        "images": [
            {
                "id": 1,
                "coco_url": "http://images.cocodataset.org/train2017/000000000001.jpg",
                "width": 100,
                "height": 80,
            }
        ],
        "categories": [
            {"id": 1, "name": "common_object", "frequency": "f"},
            {"id": 2, "name": "rare_object", "frequency": "r"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [5, 5, 50, 40], "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [70, 50, 10, 10], "iscrowd": 0},
        ],
    }
    (lvis_ann / "lvis_v1_train.json").write_text(json.dumps(lvis))

    dataset = LvisFoveationDataset(
        lvis_root=root / "lvis",
        coco_image_root=root / "coco",
        split="train2017",
        image_size=32,
        max_objects=2,
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 32, 32)
    assert sample["target_valid"].sum().item() == 2
    assert sample["categories"][0] == "rare object"
    assert "rare object" in sample["prompt"]
