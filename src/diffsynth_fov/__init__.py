"""Foveation-specific additions on top of upstream DiffSynth-Studio.

Image (FLUX2):
  - pipeline.py : Flux2FoveatedImagePipeline (mixed-resolution FLUX2 inference / training forward)
  - dit.py      : Flux2DiTFoveated (CRPA attention DiT)
  - loss.py     : FoveatedFlowMatchSFTLoss + helpers
  - logger.py   : WandbModelLogger with foveated validation visualization
  - runner.py   : training launcher with --max_training_steps support

Video (Wan2.1) — filenames mirror the image side with a `_video` suffix:
  - pipeline_video.py : WanFoveatedVideoPipeline + WanFoveatedInputVideoEmbedder + helpers
  - dit_video.py      : convert_to_foveated_wan_dit (CRPA self-attn rebind) + foveated model_fn
  - loss_video.py     : FoveatedWanFlowMatchSFTLoss (region-split MSE, random_path default)
  - logger_video.py   : WanVideoWandbModelLogger (per-step loss + sample-video validation)

Mask state primitives the video pipeline consumes live in `src/masks/`
(`paths`, `state`, `saliency`) — they don't depend on
DiffSynth and stay there.

Image-side classes are lazy-imported via PEP 562 `__getattr__` so the video
side stays importable when only the video deps are installed.
"""

import importlib


_LAZY = {
    # Image (FLUX2)
    "Flux2DiTFoveated":             (".dit",       "Flux2DiTFoveated"),
    "WandbModelLogger":             (".logger",    "WandbModelLogger"),
    "FoveatedFlowMatchSFTLoss":     (".loss",      "FoveatedFlowMatchSFTLoss"),
    "Flux2FoveatedImagePipeline":   (".pipeline",  "Flux2FoveatedImagePipeline"),
    "launch_data_process_task":     (".runner",    "launch_data_process_task"),
    "launch_training_task":         (".runner",    "launch_training_task"),
    # Video (Wan2.1 T2V) — naming mirrors the image side with a _video suffix.
    "convert_to_foveated_wan_dit":   (".dit_video",      "convert_to_foveated_wan_dit"),
    "model_fn_wan_video_foveated":   (".dit_video",      "model_fn_wan_video_foveated"),
    "WanFoveatedVideoPipeline":      (".pipeline_video", "WanFoveatedVideoPipeline"),
    "WanFoveatedInputVideoEmbedder": (".pipeline_video", "WanFoveatedInputVideoEmbedder"),
    "default_wan_model_configs":     (".pipeline_video", "default_wan_model_configs"),
    "default_wan_tokenizer_config":  (".pipeline_video", "default_wan_tokenizer_config"),
    "FoveatedWanFlowMatchSFTLoss":   (".loss_video",     "FoveatedWanFlowMatchSFTLoss"),
    "WanVideoWandbModelLogger":      (".logger_video",   "WanVideoWandbModelLogger"),
}


def __getattr__(name):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
