"""WandB-aware ModelLogger with foveated validation visualizations.

Subclasses upstream `diffsynth.diffusion.ModelLogger` and adds:
  - WandB initialization, loss / metrics / image logging
  - Per-validation-step foveated image generation at four mask positions
  - Optional target / noise-prediction / L2-error visualization at three timesteps

Token-AE specific code from the fork is removed in this release.
"""

import os
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from diffsynth.diffusion import ModelLogger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _create_foveation_mask(h: int, w: int, center: tuple, r: float, device):
    """Circular foveation mask in latent-pixel coords."""
    cx = (center[0] + 0.5) * w
    cy = (center[1] + 0.5) * h
    diagonal = (h ** 2 + w ** 2) ** 0.5
    radius_px = r * (diagonal / 2.0)
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2


def _unpatchify_for_vis(latents: torch.Tensor, latent_height: int, latent_width: int):
    """Convert [B, H*W, C] packed FLUX2 latents to [B, 3, H', W'] for visualization."""
    B, _, C = latents.shape
    latents = latents.reshape(B, latent_height, latent_width, C).permute(0, 3, 1, 2)
    if C % 4 != 0:
        return latents[:, : min(3, C)].float().cpu()
    c_base = C // 4
    latents = latents.reshape(B, c_base, 2, 2, latent_height, latent_width)
    latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(B, c_base, latent_height * 2, latent_width * 2)
    return latents[:, : min(3, c_base)].float().cpu()


def _tensor_to_pil_vis(x: torch.Tensor, vmin: float, vmax: float):
    """Convert tensor [B, 3, H, W] to PIL with a fixed [vmin, vmax] scale."""
    x = x.clamp(vmin, vmax)
    x = (x - vmin) / (vmax - vmin + 1e-8)
    x = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(x[i]) for i in range(x.shape[0])]


class WandbModelLogger(ModelLogger):
    """ModelLogger extended with WandB logging and foveated validation viz."""

    def __init__(
        self,
        output_path: str,
        project_name: str = "diffsynth-training",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        remove_prefix_in_ckpt: Optional[str] = None,
        state_dict_converter: Callable = lambda x: x,
        validation_prompts: Optional[List[str]] = None,
        validation_steps: int = 500,
        log_image_steps: int = 500,
        num_validation_images: int = 4,
        validation_kwargs: Optional[dict] = None,
    ):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.validation_prompts = validation_prompts or []
        self.validation_steps = validation_steps
        self.log_image_steps = log_image_steps
        self.num_validation_images = num_validation_images
        self.validation_kwargs = validation_kwargs or {}
        self._wandb_initialized = False

    # -----------------------------------------------------------------
    # WandB plumbing
    # -----------------------------------------------------------------

    def init_wandb(self, accelerator: Accelerator):
        if self._wandb_initialized:
            return
        if accelerator.is_main_process:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                resume="allow",
            )
            self._wandb_initialized = True

    def log_loss(self, loss: float, step: Optional[int] = None):
        if not self._wandb_initialized:
            return
        wandb.log({"train/loss": loss}, step=step if step is not None else self.num_steps)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        if not self._wandb_initialized:
            return
        wandb.log(metrics, step=step if step is not None else self.num_steps)

    def log_images(self, images: List[Any], captions: Optional[List[str]] = None,
                   key: str = "validation"):
        if not self._wandb_initialized:
            return
        if captions is None:
            captions = [f"Image {i}" for i in range(len(images))]
        wandb_images = [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]
        wandb.log({key: wandb_images}, step=self.num_steps)

    # -----------------------------------------------------------------
    # Foveation-aware validation
    # -----------------------------------------------------------------

    def _is_foveated_pipeline(self, pipe):
        return (
            hasattr(pipe, "is_foveated_pipeline")
            and pipe.is_foveated_pipeline
        )

    def _log_foveation_masks_once(self, pipe, height: int, width: int):
        """Visualize four foveation masks (left/right/top/bottom) at fixed radius."""
        h, w = height // 16, width // 16
        device = getattr(pipe, "device", "cuda")
        positions = [("left", (-0.3, 0)), ("right", (0.3, 0)),
                     ("top", (0, -0.3)), ("bottom", (0, 0.3))]
        mask_images = []
        for name, center in positions:
            mask = _create_foveation_mask(h, w, center, r=0.25, device=device)
            mask_np = mask.float().cpu().numpy() * 255
            mask_pil = Image.fromarray(mask_np.astype(np.uint8))
            mask_images.append(wandb.Image(mask_pil, caption=f"foveation_mask_{name}"))
        if mask_images:
            wandb.log({"validation/foveation_masks": mask_images}, step=self.num_steps)

    def _visualize_target_noise_pred(self, pipe, height: int, width: int):
        """At 25%/50%/75% timesteps, visualize training_target, noise_pred, and L2 error."""
        latent_height, latent_width = height // 16, width // 16
        device = getattr(pipe, "device", "cuda")
        dtype = getattr(pipe, "torch_dtype", torch.bfloat16)
        if not hasattr(pipe, "foveated_training_forward"):
            return
        try:
            prompt = self.validation_prompts[0] if self.validation_prompts else ""
            seed = self.validation_kwargs.get("seed", 42)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            pipe.scheduler.set_timesteps(1000, training=True)
            ts = pipe.scheduler.timesteps
            timestep_ids = [int(len(ts) * 0.25), int(len(ts) * 0.50), int(len(ts) * 0.75)]
            labels = ["t0", "t_mid", "t_last"]
            all_target_vis, all_pred_vis, all_l2_error = [], [], []
            inputs_shared_base = {
                "height": height, "width": width, "prompt": prompt, "cfg_scale": 1.0,
                "embedded_guidance": 1.0, "input_image": None, "rand_device": str(device),
                "seed": seed,
                **{k: v for k, v in self.validation_kwargs.items()
                   if k not in ("height", "width")},
            }
            inputs_posi, inputs_nega = {"prompt": prompt}, {"negative_prompt": ""}
            for unit in pipe.units:
                inputs_shared_base, inputs_posi, inputs_nega = pipe.unit_runner(
                    unit, pipe, inputs_shared_base, inputs_posi, inputs_nega,
                )

            input_latents = pipe.fixed_clean_latent
            noise = torch.randn_like(input_latents)
            foveation_mask = _create_foveation_mask(latent_height, latent_width, (0.0, 0.0),
                                                   r=0.25, device=device)

            for tid, label in zip(timestep_ids, labels):
                timestep_id = torch.tensor([tid], device=ts.device)
                timestep = ts[timestep_id].to(dtype=dtype, device=device)
                latents = pipe.scheduler.add_noise(input_latents, noise, timestep)
                training_target = pipe.scheduler.training_target(input_latents, noise, timestep)
                inputs_shared = dict(inputs_shared_base)
                inputs_shared["latents"] = latents
                inputs_shared["input_latents"] = input_latents
                inputs_shared["foveation_mask"] = foveation_mask
                inputs = dict(inputs_shared, **inputs_posi)
                prediction_type = self.validation_kwargs.get("prediction_type", "clean")
                lr_downsample_factor = self.validation_kwargs.get("lr_downsample_factor", 2)
                noise_pred = pipe.foveated_training_forward(
                    inputs, timestep, timestep_id, prediction_type,
                    lr_downsample_factor=lr_downsample_factor,
                )
                target_vis = _unpatchify_for_vis(training_target, latent_height, latent_width)
                pred_vis = _unpatchify_for_vis(noise_pred, latent_height, latent_width)
                l2_error = (target_vis - pred_vis).pow(2).sum(dim=1, keepdim=True).sqrt()
                all_target_vis.append((label, target_vis))
                all_pred_vis.append((label, pred_vis))
                all_l2_error.append((label, l2_error))

            target_images, pred_images = [], []
            for (label, t), (_, p) in zip(all_target_vis, all_pred_vis):
                vmin = float(min(t.min().item(), p.min().item()))
                vmax = float(max(t.max().item(), p.max().item()))
                target_images.append(wandb.Image(_tensor_to_pil_vis(t, vmin, vmax)[0],
                                                 caption=f"target_{label}"))
                pred_images.append(wandb.Image(_tensor_to_pil_vis(p, vmin, vmax)[0],
                                               caption=f"noise_pred_{label}"))
            l2_images = []
            for label, e in all_l2_error:
                e_3ch = e.expand(-1, 3, -1, -1)
                l2_max = float(e_3ch.max().item())
                l2_images.append(wandb.Image(_tensor_to_pil_vis(e_3ch, 0.0, l2_max)[0],
                                             caption=f"l2_error_{label}"))
            wandb.log({
                "validation/target": target_images,
                "validation/noise_pred": pred_images,
                "validation/l2_error": l2_images,
            }, step=self.num_steps)
        except Exception as e:
            print(f"Warning: Target/noise viz failed: {e}")

    def generate_validation_images(self, accelerator: Accelerator, model: torch.nn.Module):
        """Generate validation images (foveated 4-position grid if pipe is foveated)."""
        if not accelerator.is_main_process or not self.validation_prompts:
            return
        unwrapped_model = accelerator.unwrap_model(model)
        if not hasattr(unwrapped_model, "pipe"):
            return
        pipe = unwrapped_model.pipe
        pipe.eval()
        all_images, all_captions = [], []
        use_foveation = self._is_foveated_pipeline(pipe)
        height = self.validation_kwargs.get("height", 1024)
        width = self.validation_kwargs.get("width", 1024)

        with torch.no_grad():
            if use_foveation and hasattr(pipe, "foveated_training_forward"):
                self._visualize_target_noise_pred(pipe, height, width)
            if use_foveation:
                self._log_foveation_masks_once(pipe, height, width)
                positions = [("left", (-0.3, 0)), ("right", (0.3, 0)),
                             ("top", (0, -0.3)), ("bottom", (0, 0.3))]
                h, w = height // 16, width // 16
                device = getattr(pipe, "device", "cuda")
                for prompt in self.validation_prompts:
                    for pos_name, center in positions:
                        try:
                            mask = _create_foveation_mask(h, w, center, r=0.25, device=device)
                            kwargs = {"prompt": prompt, "foveation_mask": mask,
                                      **self.validation_kwargs}
                            images = pipe(**kwargs)
                            img = images[0] if isinstance(images, (list, tuple)) else images
                            cap = f"{prompt[:40]}... [{pos_name}]" if len(prompt) > 40 \
                                  else f"{prompt} [{pos_name}]"
                            all_images.append(img)
                            all_captions.append(cap)
                        except Exception as e:
                            print(f"Warning: Foveated validation failed {pos_name}: {e}")
            else:
                for prompt in self.validation_prompts:
                    try:
                        images = pipe(prompt=prompt, **self.validation_kwargs)
                        if isinstance(images, (list, tuple)):
                            for i, img in enumerate(images[: self.num_validation_images]):
                                cap = f"{prompt[:50]}... [{i}]" if len(prompt) > 50 \
                                      else f"{prompt} [{i}]"
                                all_images.append(img)
                                all_captions.append(cap)
                        else:
                            cap = prompt[:50] + "..." if len(prompt) > 50 else prompt
                            all_images.append(images)
                            all_captions.append(cap)
                    except Exception as e:
                        print(f"Warning: Validation failed: {e}")

        pipe.train()
        if all_images:
            self.log_images(all_images, all_captions, key="validation/generated")

    # -----------------------------------------------------------------
    # ModelLogger hooks
    # -----------------------------------------------------------------

    def on_training_start(self, accelerator: Accelerator, model: torch.nn.Module):
        self.init_wandb(accelerator)
        if accelerator.is_main_process:
            print("Generating baseline validation images (step 0)...")
        self.generate_validation_images(accelerator, model)

    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps: Optional[int] = None,
        loss: Optional[float] = None,
        **kwargs,
    ):
        self.init_wandb(accelerator)
        self.num_steps += 1
        if loss is not None and accelerator.is_main_process:
            self.log_loss(loss)
        if kwargs and accelerator.is_main_process:
            metrics = {f"train/{k}": v for k, v in kwargs.items()
                       if isinstance(v, (int, float))}
            if metrics:
                self.log_metrics(metrics)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        if self.validation_steps > 0 and self.num_steps % self.validation_steps == 0:
            self.generate_validation_images(accelerator, model)

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id: int):
        self.init_wandb(accelerator)
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch_id}, step=self.num_steps)
        super().on_epoch_end(accelerator, model, epoch_id)
        self.generate_validation_images(accelerator, model)

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module,
                        save_steps: Optional[int] = None):
        super().on_training_end(accelerator, model, save_steps)
        self.generate_validation_images(accelerator, model)
        if accelerator.is_main_process and self._wandb_initialized:
            wandb.finish()
