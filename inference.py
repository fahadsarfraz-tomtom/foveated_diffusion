"""Foveated Diffusion — inference entry point.

Dispatches on ``--pipeline``:
  - ``image``: FLUX2 foveated experiments from ``src.inference.experiments``.
  - ``video``: Wan2.1 T2V foveated experiments from ``src.inference.experiments_video``.

Per-pipeline experiment registries don't share names; e.g. ``high_res`` and
``ours`` mean different things under each pipeline. The dispatch picks the
right registry off ``args.pipeline`` first.
"""

import os
import random
import sys

# Disable sageattention: in some envs the installed binary is built against a
# different libtorch ABI and crashes at import. We fall back to PyTorch attention.
sys.modules["sageattention"] = None

import numpy as np
import torch

from src.inference import (
    EXPERIMENTS,
    VIDEO_EXPERIMENTS,
    build_parser,
    load_pipeline,
    load_video_pipeline,
)
from src.inference.args import DEFAULT_IMAGE_PROMPT, DEFAULT_VIDEO_PROMPT

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = build_parser().parse_args()
    _set_seed(args.seed)

    # Pipeline-aware prompt default.
    if args.prompt is None:
        args.prompt = DEFAULT_VIDEO_PROMPT if args.pipeline == "video" else DEFAULT_IMAGE_PROMPT

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir: {args.output_dir}")
    print(f"Pipeline: {args.pipeline}   Experiment: {args.experiment}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if args.pipeline == "image":
        if args.experiment not in EXPERIMENTS:
            raise ValueError(
                f"--experiment {args.experiment!r} is not an image experiment. "
                f"Image experiments: {list(EXPERIMENTS)}"
            )
        pipe = load_pipeline(args)
        EXPERIMENTS[args.experiment](pipe, args, args.output_dir)
    else:  # video
        if args.experiment not in VIDEO_EXPERIMENTS:
            raise ValueError(
                f"--experiment {args.experiment!r} is not a video experiment. "
                f"Video experiments: {list(VIDEO_EXPERIMENTS)}"
            )
        pipe = load_video_pipeline(args)
        VIDEO_EXPERIMENTS[args.experiment](pipe, args, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
