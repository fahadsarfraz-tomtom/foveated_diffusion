import json

import numpy as np
from PIL import Image

from src.training.coco_fpm import CocoFoveationDataset, caption_mentions_category


def test_caption_alias_matches_coco_sports_ball():
    assert caption_mentions_category("a dog chasing a small ball", "sports ball")


def test_coco_foveation_dataset_builds_prompt_relevant_target(tmp_path):
    coco_root = tmp_path / "coco"
    image_dir = coco_root / "train2017"
    ann_dir = coco_root / "annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    Image.fromarray(image).save(image_dir / "000000000001.jpg")

    instances = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 64}],
        "categories": [
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "sports ball"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [2, 2, 30, 40], "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [44, 42, 8, 8], "iscrowd": 0},
        ],
    }
    captions = {
        "images": instances["images"],
        "annotations": [{"id": 1, "image_id": 1, "caption": "a dog playing with a ball"}],
    }
    (ann_dir / "instances_train2017.json").write_text(json.dumps(instances))
    (ann_dir / "captions_train2017.json").write_text(json.dumps(captions))

    dataset = CocoFoveationDataset(coco_root, split="train2017", image_size=32, max_objects=3)
    sample = dataset[0]

    assert sample["image"].shape == (3, 32, 32)
    assert sample["target_valid"].sum().item() == 2
    assert sample["categories"] == ["sports ball", "dog"]
