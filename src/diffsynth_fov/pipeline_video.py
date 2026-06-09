"""Foveated Wan2.1 T2V pipeline (release-side, video port).

The image-side analog is ``pipeline.py`` (``Flux2FoveatedImagePipeline``).
Subclasses upstream ``WanVideoPipeline`` and adds:

  - A foveation-aware ``WanFoveatedInputVideoEmbedder`` unit that produces
    the LR latent track and (optionally) a saliency-derived ``FoveationState``
    when ``foveation_from_video=True``.
  - ``WanFoveatedVideoPipeline.__call__`` that runs the dual denoising loop
    (HR + LR latents stepped together) and the merge decode (blend HR and LR
    VAE-decoded videos using ``foveation_state.decode_blend_mask``).
  - ``from_pretrained_foveated`` classmethod that loads upstream Wan, promotes
    the instance class, and installs the foveated self-attn via
    ``convert_to_foveated_wan_dit`` (lives in ``dit_video.py``).

Only the basic Wan2.1 T2V code path is supported — VACE / S2V / animate /
camera-control / sliding-window / TeaCache paths are intentionally absent.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffsynth.pipelines.wan_video import (
    ModelConfig,
    WanVideoPipeline,
    WanVideoUnit_InputVideoEmbedder,
)
from diffsynth.diffusion.base_pipeline import PipelineUnit

from ..masks.state import FoveationState, build_state
from .dit_video import convert_to_foveated_wan_dit, model_fn_wan_video_foveated


# ---------------------------------------------------------------------------
# Foveation-aware InputVideoEmbedder unit
# ---------------------------------------------------------------------------

class WanFoveatedInputVideoEmbedder(PipelineUnit):
    """Replacement for upstream ``WanVideoUnit_InputVideoEmbedder``.

    Extra behaviour vs upstream:

      - Always populates ``input_latents_lr`` (the LR-track encoded video) when
        ``input_video`` is supplied — required by the foveated loss.
      - If ``foveation_from_video=True``, runs DeepGaze on the input video and
        builds a ``FoveationState`` that downstream consumers read from
        ``inputs['foveation_state']``.

    For pure T2V (input_video is None), this unit is a no-op aside from
    forwarding the initial noise tensor — same as upstream.
    """

    def __init__(self, foveation_from_video: bool = False):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents", "input_latents_lr", "foveation_state"),
            onload_model_names=("vae",),
        )
        self.foveation_from_video = foveation_from_video
        self._deepgaze = None  # lazy-loaded

    def _get_saliency_state(self, pipe, input_video, num_frames, height, width):
        from ..masks.saliency import extract_saliency_path, load_deepgaze_model

        if self._deepgaze is None:
            self._deepgaze = load_deepgaze_model("cpu")
        self._deepgaze = self._deepgaze.to(pipe.device)
        centers, radii = extract_saliency_path(self._deepgaze, input_video, pipe.device)
        self._deepgaze = self._deepgaze.to("cpu")

        # Saliency yields one (c, r) per *input* frame; downsample to latent length.
        import numpy as np
        latent_length = (num_frames - 1) // 4 + 1
        t_in = np.linspace(0, 1, len(centers))
        t_lat = np.linspace(0, 1, latent_length)
        cx = np.interp(t_lat, t_in, [c[0] for c in centers])
        cy = np.interp(t_lat, t_in, [c[1] for c in centers])
        r  = np.interp(t_lat, t_in, radii)
        return build_state(
            list(zip(cx.tolist(), cy.tolist())), r.tolist(),
            height, width, num_frames, device=pipe.device,
        )

    def process(self, pipe, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {
                "latents": noise,
                "input_latents": None,
                "input_latents_lr": None,
                "foveation_state": None,
            }

        pipe.load_models_to_device(self.onload_model_names)
        height, width = input_video[0].size[1], input_video[0].size[0]
        num_frames = len(input_video)

        foveation_state = None
        if self.foveation_from_video:
            foveation_state = self._get_saliency_state(
                pipe, input_video, num_frames, height, width,
            )

        video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        video_lr = F.interpolate(video, scale_factor=(1, 0.5, 0.5), mode="trilinear")
        input_latents_lr = pipe.vae.encode(
            video_lr, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        if pipe.scheduler.training:
            return {
                "latents": noise,
                "input_latents": input_latents,
                "input_latents_lr": input_latents_lr,
                "foveation_state": foveation_state,
            }
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {
            "latents": latents,
            "input_latents": input_latents,
            "input_latents_lr": input_latents_lr,
            "foveation_state": foveation_state,
        }


# ---------------------------------------------------------------------------
# Pipeline subclass + sensible defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
_DEFAULT_MODEL_FILE_PATTERNS = (
    "diffusion_pytorch_model*.safetensors",  # DiT weights
    "models_t5_umt5-xxl-enc-bf16.pth",       # text encoder (auto-redirected to .safetensors by DiffSynth)
    "Wan2.1_VAE.pth",                        # VAE (auto-redirected to .safetensors)
)


def default_wan_model_configs(model_id: str = _DEFAULT_MODEL_ID) -> list:
    """Build the standard 3-component ``ModelConfig`` list for Wan2.1-T2V.

    Used by ``WanFoveatedVideoPipeline.from_pretrained_foveated`` as its
    fallback when the caller doesn't supply ``model_configs`` explicitly,
    and by the inference launcher to build configs from CLI args.
    """
    return [ModelConfig(model_id=model_id, origin_file_pattern=p)
            for p in _DEFAULT_MODEL_FILE_PATTERNS]


def default_wan_tokenizer_config(model_id: str = _DEFAULT_MODEL_ID) -> ModelConfig:
    return ModelConfig(model_id=model_id, origin_file_pattern="google/umt5-xxl/")


class WanFoveatedVideoPipeline(WanVideoPipeline):
    """Wan2.1 T2V pipeline with foveation support.

    Use ``from_pretrained_foveated`` to load (it wraps upstream's loader and
    runs the DiT method swap). Pass a ``FoveationState`` to ``__call__`` to
    enable mixed-resolution generation; pass ``None`` to fall back to vanilla
    full-resolution behaviour.

    Default model is ``Wan-AI/Wan2.1-T2V-1.3B`` — first call downloads the
    weights (~10 GB) into ``~/.cache/huggingface/hub/`` via
    ``huggingface_hub``; subsequent calls hit the cache.
    """

    @classmethod
    def from_pretrained_foveated(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        model_id: str = _DEFAULT_MODEL_ID,
        model_configs: Optional[list] = None,
        tokenizer_config: Optional[ModelConfig] = None,
        foveation_from_video: bool = False,
        **kwargs,
    ):
        """Load Wan, promote to ``cls``, install foveated attention + unit.

        If ``model_configs`` is ``None``, builds the standard 3-component
        config list for ``model_id`` (DiT + text encoder + VAE). If
        ``tokenizer_config`` is ``None``, builds a default pointing at
        ``model_id``'s ``google/umt5-xxl/`` subdir.
        """
        if model_configs is None:
            model_configs = default_wan_model_configs(model_id)
        if tokenizer_config is None:
            tokenizer_config = default_wan_tokenizer_config(model_id)
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype, device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            **kwargs,
        )
        pipe.__class__ = cls

        n_blocks = convert_to_foveated_wan_dit(pipe.dit)
        print(f"[WanFoveatedVideoPipeline] rebound {n_blocks} self-attn forwards to foveated")

        # Replace the InputVideoEmbedder unit with our foveation-aware version.
        for i, unit in enumerate(pipe.units):
            if isinstance(unit, WanVideoUnit_InputVideoEmbedder):
                pipe.units[i] = WanFoveatedInputVideoEmbedder(
                    foveation_from_video=foveation_from_video,
                )
                break

        pipe.foveation = True
        pipe.foveation_from_video = foveation_from_video
        pipe.model_fn = model_fn_wan_video_foveated
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = True,
        tile_size: tuple = (30, 52),
        tile_stride: tuple = (15, 26),
        foveation_state: Optional[FoveationState] = None,
        progress_bar_cmd=tqdm,
    ):
        """Generate a single video.

        When ``foveation_state`` is provided, the pipeline runs the dual
        denoising loop (HR + LR latents stepped together) and the merge
        decode (HR + LR videos blended pixel-wise using
        ``foveation_state.decode_blend_mask``).
        """
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        # Provide all keys the upstream unit chain may touch — most are direct
        # `inputs_shared["..."]` accesses inside `unit_runner` / individual units,
        # so missing keys raise KeyError. Pure T2V means most are None.
        inputs_shared = {
            "input_video": None, "height": height, "width": width, "num_frames": num_frames,
            "seed": seed, "rand_device": rand_device,
            "cfg_scale": cfg_scale, "cfg_merge": False,
            "sigma_shift": sigma_shift,
            "denoising_strength": 1.0,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            # T2V: every other modality is None (vace / s2v / animate / vap / longcat / etc.)
            "input_image": None, "end_image": None,
            "control_video": None, "reference_image": None,
            "camera_control_direction": None, "camera_control_speed": None, "camera_control_origin": None,
            "vace_video": None, "vace_video_mask": None, "vace_reference_image": None, "vace_scale": 1,
            "motion_bucket_id": None, "longcat_video": None,
            "sliding_window_size": None, "sliding_window_stride": None,
            "input_audio": None, "audio_sample_rate": None, "s2v_pose_video": None,
            "audio_embeds": None, "s2v_pose_latents": None, "motion_video": None,
            "animate_pose_video": None, "animate_face_video": None,
            "animate_inpaint_video": None, "animate_mask_video": None,
            "vap_video": None,
        }
        inputs_posi = {
            "prompt": prompt, "positive": True,
            "vap_prompt": None,
            "tea_cache_l1_thresh": None, "tea_cache_model_id": "",
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "prompt": negative_prompt, "negative_prompt": negative_prompt, "positive": False,
            "vap_prompt": None, "negative_vap_prompt": None,
            "tea_cache_l1_thresh": None, "tea_cache_model_id": "",
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega,
            )

        # Foveation: build the LR latent track from the HR noise (variance-corrected).
        latents = inputs_shared["latents"]
        latents_lr = None
        if foveation_state is not None:
            # Variance correction: sqrt(N) with N=4 (a 2x2 spatial pool).
            latents_lr = F.interpolate(latents, scale_factor=(1, 0.5, 0.5), mode="trilinear") * 2
            inputs_shared["latents_lr"] = latents_lr
            inputs_shared["foveation_state"] = foveation_state

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        for progress_id, t in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            t_in = t.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            pred_posi, pred_posi_lr = self.model_fn(
                **models, **inputs_shared, **inputs_posi, timestep=t_in,
            )
            if cfg_scale != 1.0:
                pred_nega, pred_nega_lr = self.model_fn(
                    **models, **inputs_shared, **inputs_nega, timestep=t_in,
                )
                pred = pred_nega + cfg_scale * (pred_posi - pred_nega)
                pred_lr = (pred_nega_lr + cfg_scale * (pred_posi_lr - pred_nega_lr)
                           if pred_posi_lr is not None else None)
            else:
                pred = pred_posi
                pred_lr = pred_posi_lr

            latents = self.scheduler.step(pred, self.scheduler.timesteps[progress_id], latents)
            inputs_shared["latents"] = latents
            if foveation_state is not None:
                latents_lr = self.scheduler.step(
                    pred_lr, self.scheduler.timesteps[progress_id], latents_lr,
                )
                inputs_shared["latents_lr"] = latents_lr

        # Decode.
        self.load_models_to_device(["vae"])
        if foveation_state is None:
            video = self.vae.decode(
                latents, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
        else:
            # Merge decode: HR-region from latents (HR track), LR-region from
            # latents_lr (LR track), upsample LR video to full res, then blend
            # pixel-wise with the Gaussian-smoothed decode_blend_mask.
            latent_pixel = foveation_state.latent_pixel_mask.to(dtype=self.torch_dtype, device=self.device)
            latents_hr_blend = latents * latent_pixel + \
                F.interpolate(latents_lr, scale_factor=(1, 2, 2), mode="nearest") * (1 - latent_pixel)
            video_hr = self.vae.decode(
                latents_hr_blend, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
            latents_lr_blend = F.interpolate(latents, scale_factor=(1, 0.5, 0.5), mode="nearest") \
                * foveation_state.hr_grid_mask.to(latents.dtype).to(latents.device) \
                + latents_lr * (1 - foveation_state.hr_grid_mask.to(latents.dtype).to(latents.device))
            video_lr = self.vae.decode(
                latents_lr_blend, device=self.device,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
            video_lr = F.interpolate(video_lr, size=(num_frames, height, width), mode="trilinear")
            blend = foveation_state.decode_blend_mask.to(dtype=video_hr.dtype, device=video_hr.device)
            video = video_hr * blend + video_lr * (1 - blend)

        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video
