"""Video training logger: per-step loss to stdout + optional WandB + sample-video validation.

Sample-video generation runs every ``log_video_steps`` training steps (default 1000).
Each render:
  - Saves the foveated MP4 to ``{output_path}/samples/step_{N}.mp4``
  - Uploads it as a ``wandb.Video`` when ``use_wandb=True``

Uses the canonical Wan negative prompt (not user-configurable per design — the
release ships a single Wan baseline). The positive prompt, foveation trajectory,
inference-step count, seed, and CFG scale are CLI-tunable via ``--sample_*``.
"""

import os
from typing import Optional

import torch
from accelerate import Accelerator

from diffsynth.diffusion import ModelLogger
from diffsynth.utils.data import save_video

from ..inference.args import DEFAULT_VIDEO_PROMPT, DEFAULT_VIDEO_NEGATIVE_PROMPT
from ..masks.paths import sample_random_path, sample_spline_path
from ..masks.state import build_state

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class WanVideoWandbModelLogger(ModelLogger):
    """Extends upstream ``ModelLogger`` with per-step loss printing + WandB +
    periodic sample-video validation.

    Print-loss is always on so the standard logging path remains useful without
    WandB (e.g. smoke tests). WandB init is gated on ``use_wandb``. Sample-video
    generation runs at the ``log_video_steps`` cadence regardless of WandB; the
    video lands on disk in either case. With WandB it also gets uploaded.
    """

    def __init__(
        self,
        output_path: str,
        remove_prefix_in_ckpt: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "foveated-diffusion-video",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[dict] = None,
        # Sample-video knobs (mirror --sample_* CLI args)
        log_video_steps: int = 1000,
        sample_prompt: Optional[str] = None,
        sample_foveation_trajectory: str = "spline",
        sample_num_inference_steps: int = 30,
        sample_seed: int = 42,
        sample_cfg_scale: float = 5.0,
        sample_height: int = 480,
        sample_width: int = 832,
        sample_num_frames: int = 81,
    ):
        super().__init__(output_path, remove_prefix_in_ckpt)
        self.use_wandb = use_wandb and _WANDB_AVAILABLE
        if use_wandb and not _WANDB_AVAILABLE:
            print("[WanVideoWandbModelLogger] WandB requested but `wandb` not installed; "
                  "disabling. Install with: pip install wandb")
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self._wandb_initialized = False

        # Sample-video config
        self.log_video_steps = log_video_steps
        self.sample_prompt = sample_prompt or DEFAULT_VIDEO_PROMPT
        self.sample_foveation_trajectory = sample_foveation_trajectory
        self.sample_num_inference_steps = sample_num_inference_steps
        self.sample_seed = sample_seed
        self.sample_cfg_scale = sample_cfg_scale
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_num_frames = sample_num_frames
        self._samples_dir = os.path.join(output_path, "samples")

    # ------------------------------------------------------------------
    # WandB plumbing
    # ------------------------------------------------------------------
    def _init_wandb(self, accelerator: Accelerator):
        if self._wandb_initialized or not self.use_wandb:
            return
        if accelerator.is_main_process:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=self.wandb_config,
                resume="allow",
            )
            self._wandb_initialized = True

    # ------------------------------------------------------------------
    # Sample-video generation
    # ------------------------------------------------------------------
    def _build_foveation_state(self, pipe):
        """Build a FoveationState for the configured sample trajectory."""
        latent_length = (self.sample_num_frames - 1) // 4 + 1
        if self.sample_foveation_trajectory == "random_path":
            # NOTE: random_path uses torch.rand. We've already isolated the global
            # RNG state in _generate_sample_video below, so the variant the sample
            # uses won't pollute training's random stream.
            centers, radii = sample_random_path(latent_length, device=pipe.device)
        else:  # spline (default)
            centers, radii = sample_spline_path(latent_length)
        return build_state(
            centers, radii, self.sample_height, self.sample_width, self.sample_num_frames,
            device=pipe.device,
        )

    def _generate_sample_video(self, accelerator: Accelerator, model: torch.nn.Module):
        """Render one sample video and (if WandB) upload it.

        Only rank 0 runs the inference; other ranks wait at a barrier so the
        next training step doesn't start until the sample is done.
        """
        # All ranks must enter the barrier; only rank 0 generates.
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return

        # Reach the underlying pipeline through DDP wrapping
        pipe = accelerator.unwrap_model(model).pipe

        os.makedirs(self._samples_dir, exist_ok=True)
        out_path = os.path.join(self._samples_dir, f"step_{self.num_steps:08d}.mp4")

        # Restore scheduler training state + global RNG after the sample render.
        # Upstream's `pipe(...)` internally invokes
        # `scheduler.set_timesteps(N, training=False)` which overwrites
        # `timesteps`, `sigmas`, `linear_timesteps_weights`, and `training`.
        # Training sets `set_timesteps(1000, training=True)` ONCE at init
        # (training_module.py:275) and never resets — so we must reset here to
        # the same training configuration, otherwise subsequent training steps
        # would sample timesteps from the narrowed inference grid.
        rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(self.sample_seed)
            torch.cuda.empty_cache()

            foveation_state = self._build_foveation_state(pipe)
            print(f"[WanVideoWandbModelLogger] sample-video at step {self.num_steps}: "
                  f"trajectory={self.sample_foveation_trajectory} "
                  f"steps={self.sample_num_inference_steps} seed={self.sample_seed}", flush=True)
            video = pipe(
                prompt=self.sample_prompt,
                negative_prompt=DEFAULT_VIDEO_NEGATIVE_PROMPT,
                height=self.sample_height, width=self.sample_width,
                num_frames=self.sample_num_frames,
                cfg_scale=self.sample_cfg_scale,
                num_inference_steps=self.sample_num_inference_steps,
                seed=self.sample_seed,
                tiled=True,
                foveation_state=foveation_state,
            )
            save_video(video, out_path, fps=15, quality=5)
            print(f"[WanVideoWandbModelLogger] saved {out_path}", flush=True)

            if self._wandb_initialized:
                wandb.log(
                    {"validation/sample_video": wandb.Video(out_path, fps=15, format="mp4")},
                    step=self.num_steps,
                )
        finally:
            # Restore training mode: rebuilds timesteps, sigmas, training
            # weights, and the training flag in one call — matches the init
            # in upstream training_module.py.
            pipe.scheduler.set_timesteps(1000, training=True)
            torch.random.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step-end hook
    # ------------------------------------------------------------------
    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps: Optional[int] = None,
        loss: Optional[float] = None,
        **kwargs,
    ):
        if not self._wandb_initialized:
            self._init_wandb(accelerator)
        # Upstream ModelLogger.on_step_end handles step counter + checkpoint saves;
        # call into it first so our logging sees the post-increment num_steps.
        super().on_step_end(accelerator, model, save_steps, **kwargs)

        if loss is not None and accelerator.is_main_process:
            loss_val = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            print(f"[step {self.num_steps:6d}]  loss = {loss_val:.6f}", flush=True)
            if self._wandb_initialized:
                wandb.log({"train/loss": loss_val}, step=self.num_steps)

        # Sample-video at cadence (skip step 0 — no training has happened yet)
        if (
            self.log_video_steps
            and self.log_video_steps > 0
            and self.num_steps > 0
            and self.num_steps % self.log_video_steps == 0
        ):
            self._generate_sample_video(accelerator, model)
