try:
    from .args import build_parser, str_to_bool
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("diffsynth"):
        raise
    build_parser = None
    str_to_bool = None

from .coco_fpm import (
    CocoFoveationDataset,
    HashTextEmbedder,
    coco_foveation_collate,
    fpm_supervision_loss,
    target_gaussian_map,
)
from .lvis_fpm import LvisFoveationDataset, lvis_foveation_collate

try:
    from .module import Flux2FoveatedImageTrainingModule
    from .module_video import WanFoveatedVideoTrainingModule
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("diffsynth"):
        raise
    Flux2FoveatedImageTrainingModule = None
    WanFoveatedVideoTrainingModule = None
