"""
FLUX2 Foveated Image Generation Pipeline for DiffSynth.

This pipeline integrates foveation logic from pipeline_flux2_klein_foveation.py
into the DiffSynth FLUX2 pipeline architecture.

Key features:
- Mixed-resolution latent processing for efficient foveated rendering
- X0 (clean image) prediction for better upsampling quality
- Phase-aligned RoPE for consistent position encoding across resolutions
- CRPA attention for efficient mixed-resolution attention
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import math
import torchvision
import numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import Union, List, Optional, Tuple
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import save_image

from diffsynth.core.device.npu_compatible_device import get_device_type
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput

from transformers import AutoProcessor, AutoTokenizer
from diffsynth.models.flux2_text_encoder import Flux2TextEncoder
from .dit import Flux2DiTFoveated
from diffsynth.models.flux2_vae import Flux2VAE
from diffsynth.models.z_image_text_encoder import ZImageTextEncoder
from scipy.ndimage import zoom, gaussian_filter
import time

# Optional saliency-detection dependencies: only required when
# foveated_training_mode == "saliency". Import lazily so a default
# install does not need them.
try:
    from scipy.datasets import face as scipy_face
except Exception:
    scipy_face = None
try:
    import deepgaze_pytorch
except Exception:
    deepgaze_pytorch = None


def _gaussian_blur_mask_2d(
    mask: torch.Tensor,
    sigma_pixels: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply separable Gaussian blur to a [1, 1, H, W] mask. sigma_pixels controls falloff width."""
    # Kernel size odd, ~3 sigma on each side
    k = max(3, int(math.ceil(3 * sigma_pixels)) * 2 + 1)
    if k % 2 == 0:
        k += 1
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma_pixels ** 2 + 1e-6))
    g = g / g.sum()
    # [1, 1, k, 1] and [1, 1, 1, k] for separable conv
    kernel_h = g.view(1, 1, k, 1)
    kernel_w = g.view(1, 1, 1, k)
    pad = k // 2
    out = F.conv2d(mask, kernel_h, padding=(pad, 0))
    out = F.conv2d(out, kernel_w, padding=(0, pad))
    return out


class Flux2FoveatedImagePipeline(BasePipeline):
    """
    FLUX2 Foveated Image Generation Pipeline.
    
    This pipeline extends the standard FLUX2 pipeline with support for:
    - Foveated rendering with mixed-resolution latents
    - X0 prediction upsampling for better quality
    - CRPA attention for efficient attention computation
    """

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("FLUX.2")
        self.text_encoder: Flux2TextEncoder = None
        self.text_encoder_qwen3: ZImageTextEncoder = None
        self.dit: Flux2DiTFoveated = None
        self.vae: Flux2VAE = None
        self.tokenizer: AutoProcessor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            Flux2FoveatedUnit_ShapeChecker(),
            Flux2FoveatedUnit_PromptEmbedder(),
            Flux2FoveatedUnit_Qwen3PromptEmbedder(),
            Flux2FoveatedUnit_NoiseInitializer(),
            Flux2FoveatedUnit_InputImageEmbedder(),
            Flux2FoveatedUnit_ImageIDs(),
        ]
        
        # fixed latent for tracking validation
        self.fixed_clean_latent = None
        self.detach_z_mix = False

        self.model_fn = model_fn_flux2_foveated
        
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        """
        Load a FLUX2 Foveated Image Pipeline from pretrained weights.
        
        The pipeline will first try to fetch a registered "flux2_dit_foveated" model.
        If not found, it will fetch a standard "flux2_dit" model and convert it to
        the foveated version (same weights, different attention processing).
        """
        # Initialize pipeline
        pipe = Flux2FoveatedImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("flux2_text_encoder")
        pipe.text_encoder_qwen3 = model_pool.fetch_model("z_image_text_encoder")
        
        # Try to fetch foveated model first, fall back to base model
        pipe.dit = model_pool.fetch_model("flux2_dit_foveated")
        if pipe.dit is None:
            # Load base FLUX2 model and convert to foveated version
            base_dit = model_pool.fetch_model("flux2_dit")
            if base_dit is not None:
                pipe.dit = Flux2FoveatedImagePipeline._convert_to_foveated_dit(base_dit)
                print("Converted base FLUX2 DiT to foveated version.")
        
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @staticmethod
    def _convert_to_foveated_dit(base_dit):
        """
        Convert a base FLUX2 DiT to a foveated version.
        
        The foveated model has the exact same architecture as the base model,
        just with CRPA attention processors. We infer config from state dict.
        
        Args:
            base_dit: A loaded Flux2DiT model instance.
            
        Returns:
            A Flux2DiTFoveated model with the same weights.
        """
        # Get the state dict from the base model
        state_dict = base_dit.state_dict()
        
        # Infer configuration from state dict shapes
        # inner_dim from x_embedder.weight: [inner_dim, in_channels]
        inner_dim = state_dict['x_embedder.weight'].shape[0]
        in_channels = state_dict['x_embedder.weight'].shape[1]
        
        # joint_attention_dim from context_embedder.weight: [inner_dim, joint_attention_dim]
        joint_attention_dim = state_dict['context_embedder.weight'].shape[1]
        
        # Count transformer blocks
        num_layers = 0
        num_single_layers = 0
        for key in state_dict.keys():
            if key.startswith('transformer_blocks.') and key.endswith('.attn.to_q.weight'):
                idx = int(key.split('.')[1])
                num_layers = max(num_layers, idx + 1)
            if key.startswith('single_transformer_blocks.') and key.endswith('.attn.to_qkv_mlp_proj.weight'):
                idx = int(key.split('.')[1])
                num_single_layers = max(num_single_layers, idx + 1)
        
        # Infer attention config from to_q weight: [inner_dim, inner_dim]
        # inner_dim = num_heads * head_dim
        # Default head_dim is 128, so num_heads = inner_dim / 128
        attention_head_dim = 128
        num_attention_heads = inner_dim // attention_head_dim
        
        # Check for guidance embeddings
        guidance_embeds = 'time_guidance_embed.guidance_embedder.linear_1.weight' in state_dict
        
        print(f"Inferred config: inner_dim={inner_dim}, in_channels={in_channels}, "
              f"joint_attention_dim={joint_attention_dim}, num_layers={num_layers}, "
              f"num_single_layers={num_single_layers}, num_attention_heads={num_attention_heads}, "
              f"attention_head_dim={attention_head_dim}, guidance_embeds={guidance_embeds}")
        
        # Create foveated model with inferred parameters
        foveated_dit = Flux2DiTFoveated(
            in_channels=in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            guidance_embeds=guidance_embeds,
        )
        
        # Load state dict directly - architectures are identical
        foveated_dit.load_state_dict(state_dict)
        
        # Move to same device and dtype as base model
        device = next(base_dit.parameters()).device
        dtype = next(base_dit.parameters()).dtype
        foveated_dit = foveated_dit.to(device=device, dtype=dtype)
        
        return foveated_dit

    # =====================================================================
    # Patchify / Unpatchify Operations (FLUX2 2x2 patches)
    # =====================================================================

    @staticmethod
    def _patchify_latents(latents):
        """Convert [B, C, H, W] to 2x2 patched format [B, C*4, H/2, W/2]."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        """Convert [B, C*4, H/2, W/2] back to [B, C, H, W]."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    # =====================================================================
    # Foveation-Specific Latent Operations
    # =====================================================================

    @staticmethod
    def _prepare_latent_image_ids_foveation(
        batch_size: int, 
        height: int, 
        width: int, 
        device: torch.device, 
        dtype: torch.dtype, 
        foveation_mask: Optional[torch.Tensor] = None,
        lr_factor: int = 2,
    ):
        """
        Prepare 4D position IDs for image latents with foveation support.
        
        For FLUX2, position IDs have format (T, H, W, L) where:
        - T: Time coordinate (always 0 for image latents)
        - H: Height coordinate
        - W: Width coordinate  
        - L: Layer coordinate (always 0 for image latents)
        
        With foveation:
        - High-res blocks keep all 4 tokens
        - Low-res blocks keep only top-left token
        
        Returns:
            latent_image_ids: [Seq_Len, 4] position IDs
            resolution_mask: [Seq_Len] mask (1=HR, 0=LR)
            resolution_mask_top_left: [Seq_Len] mask for key subsampling
        """
        # Create base coordinate grid (all must have same dtype for cartesian_prod)
        t = torch.zeros(1, dtype=dtype)
        h = torch.arange(height, dtype=dtype)
        w = torch.arange(width, dtype=dtype)
        l = torch.zeros(1, dtype=dtype)
        
        latent_image_ids = torch.cartesian_prod(t, h, w, l).to(device=device)
        
        resolution_mask_grid = torch.ones(height, width, device=device, dtype=dtype)

        if foveation_mask is not None:
            if height % lr_factor != 0 or width % lr_factor != 0:
                raise ValueError(f"Height and width must be divisible by lr_factor ({lr_factor}) for mixed-resolution.")

            n_per_block = lr_factor * lr_factor
            h_d, w_d = height // lr_factor, width // lr_factor

            grid_2d = latent_image_ids.view(height, width, 4)
            grid_blocks = grid_2d.view(h_d, lr_factor, w_d, lr_factor, 4).permute(0, 2, 1, 3, 4).reshape(h_d, w_d, n_per_block, 4)

            mask_blocks = foveation_mask.view(h_d, lr_factor, w_d, lr_factor).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)
            is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
            is_low_res_block = ~is_high_res_block

            low_res_coords = grid_blocks[..., 0, :]
            output_grid = grid_blocks.clone()
            # Flatten before mask indexing — torch 2.5.1 mishandles 2D bool mask + int/slice.
            output_grid.view(h_d * w_d, n_per_block, 4)[is_low_res_block.view(-1), 0, :] = low_res_coords[is_low_res_block]

            res_mask_blocks = resolution_mask_grid.view(h_d, lr_factor, w_d, lr_factor).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)
            output_res_mask = res_mask_blocks.clone()
            output_res_mask[is_low_res_block, :] = 0.0

            valid_tokens = torch.ones(h_d, w_d, n_per_block, device=device, dtype=torch.bool)
            valid_tokens.view(h_d * w_d, n_per_block)[is_low_res_block.view(-1), 1:] = False

            output_grid = output_grid.reshape(-1, 4)
            output_res_mask = output_res_mask.reshape(-1)
            valid_tokens = valid_tokens.reshape(-1)

            tl_mask_blocks = torch.zeros_like(res_mask_blocks)
            tl_mask_blocks[..., 0] = 1.0
            tl_mask_blocks[is_low_res_block, :] = 1.0
            resolution_mask_top_left = tl_mask_blocks.reshape(-1)[valid_tokens]

            latent_image_ids = output_grid[valid_tokens]
            resolution_mask = output_res_mask[valid_tokens]

            return (
                latent_image_ids.to(device=device, dtype=dtype),
                resolution_mask.to(device=device, dtype=dtype),
                resolution_mask_top_left.to(device=device, dtype=dtype)
            )
        else:
            return (
                latent_image_ids.to(device=device, dtype=dtype),
                None,
                None
            )

    @staticmethod
    def _downsample_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        foveation_mask: Optional[torch.Tensor] = None,
        sample_mode: str = "nearest",
        latents_downsampled: Optional[torch.Tensor] = None,
        lr_factor: int = 2,
    ):
        """
        Downsample latents spatially based on foveation mask.

        For low-res blocks, averages/samples lr_factor x lr_factor regions to single tokens.
        High-res blocks keep all lr_factor*lr_factor tokens.
        """
        if foveation_mask is None:
            return latents

        mask = foveation_mask.to(latents.device)
        batch_size, seq_len, channels = latents.shape

        if height * width != seq_len:
            raise ValueError(f"Height ({height}) * Width ({width}) != seq_len ({seq_len})")

        output_latents = latents.clone()

        # Unpack to spatial format for downsampling
        latents_spatial = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents_spatial = latents_spatial.permute(0, 3, 1, 4, 2, 5)
        latents_spatial = latents_spatial.reshape(batch_size, channels // 4, height * 2, width * 2)

        # Downsample (naive interpolation when latents_downsampled isn't precomputed).
        if latents_downsampled is None:
            if sample_mode == "average":
                latents_down = F.interpolate(latents_spatial, scale_factor=1.0 / lr_factor, mode="bilinear") * lr_factor
            elif sample_mode == "top_left":
                latents_down = F.interpolate(latents_spatial, scale_factor=1.0 / lr_factor, mode="nearest")
            else:
                raise ValueError(f"Unknown sample_mode: {sample_mode}")

            latents_down = latents_down.reshape(batch_size, channels // 4, height // lr_factor, 2, width // lr_factor, 2)
            latents_down = latents_down.permute(0, 2, 4, 1, 3, 5)
            low_res_packed = latents_down.reshape(batch_size, height // lr_factor, width // lr_factor, channels)
        else:
            low_res_packed = latents_downsampled.reshape(batch_size, height // lr_factor, width // lr_factor, channels)

        n_per_block = lr_factor * lr_factor
        h_d, w_d = height // lr_factor, width // lr_factor
        output_latents = output_latents.view(batch_size, height, width, channels)
        output_latents = output_latents.view(batch_size, h_d, lr_factor, w_d, lr_factor, channels)
        output_latents = output_latents.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, h_d, w_d, n_per_block, channels)

        mask_blocks = mask.view(h_d, lr_factor, w_d, lr_factor).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)
        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
        is_low_res_block = ~is_high_res_block

        # Flatten the (h_d, w_d) dims before mask indexing — torch 2.5.1 mishandles
        # mixing a 2D bool mask with an integer index in adjacent slots.
        output_latents_flat = output_latents.view(batch_size, h_d * w_d, n_per_block, channels)
        low_res_packed_flat = low_res_packed.view(batch_size, h_d * w_d, channels)
        is_low_res_block_flat = is_low_res_block.view(-1)
        output_latents_flat[:, is_low_res_block_flat, 0, :] = low_res_packed_flat[:, is_low_res_block_flat, :].to(output_latents.dtype)

        valid_tokens = torch.ones(h_d, w_d, n_per_block, device=latents.device, dtype=torch.bool)
        # Flatten before mask indexing — torch 2.5.1 mishandles 2D bool mask + slice.
        valid_tokens_flat = valid_tokens.view(h_d * w_d, n_per_block)
        valid_tokens_flat[is_low_res_block.view(-1), 1:] = False

        output_latents = output_latents.view(batch_size, -1, channels)
        valid_tokens = valid_tokens.view(-1)
        
        return output_latents[:, valid_tokens, :]

    @staticmethod
    def _downsample_remaining_high_res(
        latents: torch.Tensor,
        height: int,
        width: int,
        foveation_mask: torch.Tensor,
        downsample_mode: str = "nearest",
        lr_factor: int = 2,
    ):
        """
        Downsample all blocks to uniform low-res (single token per block).
        """
        batch_size, reduced_len, channels = latents.shape
        device = latents.device
        dtype = latents.dtype

        n_per_block = lr_factor * lr_factor
        h_d, w_d = height // lr_factor, width // lr_factor

        mask = foveation_mask.to(device)
        mask_blocks = mask.view(h_d, lr_factor, w_d, lr_factor).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)

        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
        is_low_res_block = ~is_high_res_block

        layout_mask = torch.ones(h_d, w_d, n_per_block, device=device, dtype=torch.bool)
        # Flatten before mask indexing — torch 2.5.1 mishandles 2D bool mask + slice.
        layout_mask.view(h_d * w_d, n_per_block)[is_low_res_block.view(-1), 1:] = False

        if layout_mask.sum() != reduced_len:
            raise ValueError(f"Latent length {reduced_len} != mask topology {layout_mask.sum()}")

        reconstructed_blocks = torch.zeros(batch_size, h_d * w_d * n_per_block, channels, device=device, dtype=dtype)
        reconstructed_blocks[:, layout_mask.view(-1), :] = latents
        reconstructed_blocks = reconstructed_blocks.view(batch_size, h_d, w_d, n_per_block, channels)

        if is_high_res_block.any():
            high_res_tokens = reconstructed_blocks[:, is_high_res_block, :, :]

            if downsample_mode == "average":
                pooled_val = high_res_tokens.mean(dim=2)
                # Flatten spatial dims before mask indexing — torch 2.5.1 mishandles
                # mixing a 2D bool mask with an integer index in adjacent slots.
                reconstructed_blocks_flat = reconstructed_blocks.view(batch_size, h_d * w_d, n_per_block, channels)
                reconstructed_blocks_flat[:, is_high_res_block.view(-1), 0, :] = pooled_val

        final_grid = reconstructed_blocks[:, :, :, 0, :]
        return final_grid.reshape(batch_size, h_d * w_d, channels)

    @staticmethod
    def _upsample_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        foveation_mask: Optional[torch.Tensor] = None,
        upsample_mode: str = "nearest",
        lr_factor: int = 2,
    ):
        """
        Upsamples reduced latents back to the original spatial grid size.
        """
        if foveation_mask is None:
            return latents

        batch_size, reduced_len, channels = latents.shape
        device = latents.device
        dtype = latents.dtype

        n_per_block = lr_factor * lr_factor
        h_d, w_d = height // lr_factor, width // lr_factor

        mask = foveation_mask.to(device)
        mask_blocks = mask.view(h_d, lr_factor, w_d, lr_factor).permute(0, 2, 1, 3).reshape(h_d, w_d, n_per_block)

        is_high_res_block = (mask_blocks.sum(dim=-1) > 0)
        is_low_res_block = ~is_high_res_block

        layout_mask = torch.ones(h_d, w_d, n_per_block, device=device, dtype=torch.bool)
        # Flatten before mask indexing — torch 2.5.1 mishandles 2D bool mask + slice.
        layout_mask.view(h_d * w_d, n_per_block)[is_low_res_block.view(-1), 1:] = False

        if layout_mask.sum() != reduced_len:
            raise ValueError(f"Latent length {reduced_len} does not match mask topology {layout_mask.sum()}.")

        reconstructed_blocks = torch.zeros(batch_size, h_d * w_d * n_per_block, channels, device=device, dtype=dtype)
        reconstructed_blocks[:, layout_mask.view(-1), :] = latents
        reconstructed_blocks = reconstructed_blocks.view(batch_size, h_d, w_d, n_per_block, channels)

        if upsample_mode == "nearest":
            if is_low_res_block.any():
                # Flatten before mask indexing — torch 2.5.1 mishandles 2D bool mask + slice.
                rb_flat = reconstructed_blocks.view(batch_size, h_d * w_d, n_per_block, channels)
                low_res_block_flat = is_low_res_block.view(-1)
                low_res_vals = rb_flat[:, low_res_block_flat, 0:1, :]
                rb_flat[:, low_res_block_flat, :, :] = low_res_vals.expand(-1, -1, n_per_block, -1)

            final_output = reconstructed_blocks.view(batch_size, h_d, w_d, lr_factor, lr_factor, channels)
            final_output = final_output.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)
        else:
            coarse_grid = reconstructed_blocks[..., 0, :].clone()

            if is_high_res_block.any():
                high_res_means = reconstructed_blocks[:, is_high_res_block, :, :].mean(dim=2)
                coarse_grid[:, is_high_res_block, :] = high_res_means

            coarse_grid = coarse_grid.permute(0, 3, 1, 2)

            fine_grid = F.interpolate(
                coarse_grid,
                size=(height, width),
                mode=upsample_mode,
                align_corners=False,
                antialias=True if upsample_mode != 'nearest' else False
            )

            final_output = fine_grid.permute(0, 2, 3, 1)

            high_res_raster = reconstructed_blocks.view(batch_size, h_d, w_d, lr_factor, lr_factor, channels)
            high_res_raster = high_res_raster.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, height, width, channels)

            pixel_mask = is_high_res_block.repeat_interleave(lr_factor, dim=0).repeat_interleave(lr_factor, dim=1)
            final_output[:, pixel_mask, :] = high_res_raster[:, pixel_mask, :]

        return final_output.reshape(batch_size, height * width, channels)

    def foveated_training_forward(
        self,
        inputs: dict,
        timestep: torch.Tensor,
        timestep_id: int,
        prediction_type: str,
        lr_downsample_factor: int = 2,
    ):
        """Single-step foveated forward for training. Used by FoveatedFlowMatchSFTLoss.

        Downsamples latents with foveation, runs the DiT, and returns the foveated
        noise prediction [B, L, C] matching the mixed-resolution training target.
        """
        foveation_mask = inputs.get("foveation_mask")
        if foveation_mask is None:
            models = {name: getattr(self, name) for name in self.in_iteration_models}
            return self.model_fn(**models, **inputs, timestep=timestep)

        # fixed clean latent for tracking validation
        if self.fixed_clean_latent is None:
            self.fixed_clean_latent = inputs["input_latents"]

        latents = inputs["latents"]
        height = inputs["height"]
        width = inputs["width"]
        latent_height, latent_width = height // 16, width // 16
        batch_size = latents.shape[0]
        foveation_mask = foveation_mask.to(latents.device)

        latents_input = self._downsample_latents(
            latents, latent_height, latent_width,
            foveation_mask=foveation_mask,
            latents_downsampled=inputs["latents_downsampled"],
            lr_factor=lr_downsample_factor,
        )

        latent_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids_foveation(
            batch_size, latent_height, latent_width, self.device, latents.dtype,
            foveation_mask, lr_factor=lr_downsample_factor,
        )
        if latent_ids.ndim == 2:
            latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        inputs["latents"] = latents_input
        inputs["image_ids"] = latent_ids
        inputs["resolution_mask"] = resolution_mask
        inputs["resolution_mask_top_left"] = resolution_mask_top_left

        models = {name: getattr(self, name) for name in self.in_iteration_models}
        return self.model_fn(**models, **inputs, timestep=timestep)

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        progress_bar_cmd=tqdm,
        # Foveation
        foveation_mask: Optional[torch.Tensor] = None,
        decode_mode: str = "direct",
        prediction_type: str = "clean",
        soft_foveation_blend: bool = False,
        full_res_foveation_mask: Optional[torch.Tensor] = None,
        lr_downsample_factor: int = 2,
    ):
        """Generate images with optional foveated rendering.

        Args:
            prompt: Text prompt for generation.
            foveation_mask: [H, W] binary mask (1 = high-res, 0 = low-res).
                Shape should match the token grid (height//16, width//16).
            decode_mode: "direct" (HR-only decode) or "merge" (blend HR + LR decodes).
            soft_foveation_blend: If True, Gaussian-blur the mask boundary before
                blending in merge decode.
        """
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            dynamic_shift_len=height // 16 * width // 16,
        )

        latent_height = height // 16
        latent_width = width // 16

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "foveation_mask": foveation_mask,
            "soft_foveation_blend": soft_foveation_blend,
            "lr_downsample_factor": lr_downsample_factor,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega,
            )

        # Denoise with foveation support
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        latents = inputs_shared["latents"]
        # All downsampling happens once up front; the DiT operates on the mixed-resolution
        # token sequence for every denoising step.
        latents = self._downsample_latents(
            latents, latent_height, latent_width,
            foveation_mask=foveation_mask,
            latents_downsampled=inputs_shared["latents_downsampled"],
            lr_factor=lr_downsample_factor,
        )
        print('decode mode', decode_mode)

        start_t = time.time()
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_val = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            batch_size = latents.shape[0]

            if foveation_mask is not None:
                latents_input = latents
                latent_ids, resolution_mask, resolution_mask_top_left = self._prepare_latent_image_ids_foveation(
                    batch_size, latent_height, latent_width,
                    self.device, latents.dtype, foveation_mask,
                    lr_factor=lr_downsample_factor,
                )
                if latent_ids.ndim == 2:
                    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                latents_input = latents
                resolution_mask = None
                resolution_mask_top_left = None
                latent_ids = inputs_shared["image_ids"]

            inputs_shared["latents"] = latents_input
            inputs_shared["image_ids"] = latent_ids
            inputs_shared["resolution_mask"] = resolution_mask
            inputs_shared["resolution_mask_top_left"] = resolution_mask_top_left

            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models,
                timestep=timestep_val,
                progress_id=progress_id,
            )

            if foveation_mask is None:
                noise_pred = noise_pred[:, :latents.size(1), :]

            latents = self.step(
                self.scheduler, progress_id=progress_id,
                noise_pred=noise_pred, latents=latents,
            )
        vit_end_t = time.time()

        inputs_shared["latents"] = latents

        # Decode
        self.load_models_to_device(['vae'])

        if decode_mode == "direct":
            latents_spatial = rearrange(latents, "B (H W) C -> B C H W", H=latent_height, W=latent_width)
            image = self.vae.decode(latents_spatial)
            image = self.vae_output_to_image(image)
        elif decode_mode == "merge" and foveation_mask is not None:
            # Merge decoding: decode low-res and high-res separately, then blend.
            # Note: DiffSynth VAE expects patched latents [B, 128, H/2, W/2].
            mixed_res_latents = latents  # already mixed-resolution

            # Step 2: Low-res path - downsample mixed-res to uniform low-res grid
            low_res_latents = self._downsample_remaining_high_res(
                mixed_res_latents, latent_height, latent_width, foveation_mask=foveation_mask,
                lr_factor=lr_downsample_factor
            )
            h_d, w_d = latent_height // lr_downsample_factor, latent_width // lr_downsample_factor
            # VAE expects patched format [B, 128, H/2, W/2] - just reshape, don't unpatchify
            low_res_spatial = low_res_latents.view(batch_size, h_d, w_d, -1).permute(0, 3, 1, 2)
            low_res_image = self.vae.decode(low_res_spatial)

            # Step 3: High-res path - upsample mixed-res to uniform high-res grid
            high_res_latents = self._upsample_latents(
                mixed_res_latents, latent_height, latent_width, foveation_mask=foveation_mask,
                lr_factor=lr_downsample_factor
            )
            # VAE expects patched format [B, 128, H/2, W/2] - just reshape, don't unpatchify
            high_res_spatial = high_res_latents.view(
                batch_size, latent_height, latent_width, -1
            ).permute(0, 3, 1, 2)
            high_res_image = self.vae.decode(high_res_spatial)

            # Step 4: Upsample low-res image and merge
            low_res_image = F.interpolate(low_res_image, size=(height, width), mode='bicubic', align_corners=False)
            
            if full_res_foveation_mask is not None:
                high_res_foveation_mask = full_res_foveation_mask[None, None, :, :].float()
            else:
                high_res_foveation_mask = F.interpolate(
                    foveation_mask[None, None, :, :].float(), 
                    size=(height, width), 
                    mode='bicubic'
                )
            high_res_foveation_mask = high_res_foveation_mask.to(latents.dtype).to(latents.device)
                
            if soft_foveation_blend:
                # Soft edge: Gaussian falloff over ~upscale_factor pixels at high-res
                upscale = height / latent_height
                high_res_foveation_mask = _gaussian_blur_mask_2d(
                    high_res_foveation_mask, sigma_pixels=upscale, device=latents.device, dtype=latents.dtype
                )
                high_res_foveation_mask = high_res_foveation_mask.clamp(0.0, 1.0)
            
            merged_image = low_res_image * (1 - high_res_foveation_mask) + high_res_image * high_res_foveation_mask
            image = self.vae_output_to_image(merged_image)
        else:
            # Fallback to direct
            latents_spatial = rearrange(latents, "B (H W) C -> B C H W", H=latent_height, W=latent_width)
            image = self.vae.decode(latents_spatial)
            image = self.vae_output_to_image(image)
        vae_end_t = time.time()

        # individually track time
        self.vit_timing = vit_end_t - start_t
        self.vae_timing = vae_end_t - vit_end_t
            
        self.load_models_to_device([])

        return image


# =====================================================================
# Pipeline Units (adapted from flux2_image.py)
# =====================================================================

class Flux2FoveatedUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: Flux2FoveatedImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class Flux2FoveatedUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )
        self.system_message = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."

    def format_text_input(self, prompts: List[str], system_message: str = None):
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]
        return [
            [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]

    def get_mistral_3_small_prompt_embeds(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        system_message: str = None,
        hidden_states_layers: List[int] = (10, 20, 30),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        messages_batch = self.format_text_input(prompts=prompt, system_message=system_message)
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds
    
    def prepare_text_ids(self, x: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    def encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds = self.get_mistral_3_small_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
                system_message=self.system_message,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def process(self, pipe: Flux2FoveatedImagePipeline, prompt):
        if pipe.text_encoder_qwen3 is not None:
            return {}
        
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, text_ids = self.encode_prompt(
            pipe.text_encoder, pipe.tokenizer, prompt,
            dtype=pipe.torch_dtype, device=pipe.device,
        )
        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids}


class Flux2FoveatedUnit_Qwen3PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder_qwen3",)
        )
        self.hidden_states_layers = (9, 18, 27)

    def get_qwen3_prompt_embeds(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        with torch.inference_mode():
            output = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        out = torch.stack([output.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds

    def prepare_text_ids(self, x: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    def encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds = self.get_qwen3_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def process(self, pipe: Flux2FoveatedImagePipeline, prompt):
        if pipe.text_encoder_qwen3 is None:
            return {}
        
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, text_ids = self.encode_prompt(
            pipe.text_encoder_qwen3, pipe.tokenizer, prompt,
            dtype=pipe.torch_dtype, device=pipe.device,
        )
        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids}


class Flux2FoveatedUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "lr_downsample_factor"),
            output_params=("noise", "downsampled_noise"),
        )

    def process(self, pipe: Flux2FoveatedImagePipeline, height, width, seed, rand_device, lr_downsample_factor=2):
        noise = pipe.generate_noise((1, 128, height//16, width//16), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise_downsampled = pipe.generate_noise((1, 128, height//16//lr_downsample_factor, width//16//lr_downsample_factor), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise = noise.reshape(1, 128, height//16 * width//16).permute(0, 2, 1)
        noise_downsampled = noise_downsampled.reshape(1, 128, height//16//lr_downsample_factor * width//16//lr_downsample_factor).permute(0, 2, 1)
        return {"noise": noise, "noise_downsampled": noise_downsampled}


# Saliency-to-mask constants (same as run_saliency_detection.py / fovetaion_masks_from_video.py)
_INFERENCE_SCALE = 0.5
_SPATIAL_SIGMA = 20.0
_R_MIN, _R_MAX = 0.2, 0.5
_COVERAGE = 0.9
_DENSE_THRESHOLD = 0.3


def bbox_to_foveation_mask(
    height: int,
    width: int,
    bboxes_xyxy,
    sigma: float = 10.0,
):
    """
    Create a full-resolution foveation mask (H, W) in [0, 1] from a list of
    bounding boxes in absolute pixel coordinates (x1, y1, x2, y2).

    This function mirrors bbox_to_foveation_mask in misc/run_bbox_detection.py:
    the mask is 1 inside any bbox, 0 outside, followed by Gaussian smoothing
    and normalization. This yields a smooth "high-acuity" region covering all
    detected objects.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for box in bboxes_xyxy:
        x1, y1, x2, y2 = box
        x1_i = max(0, min(width - 1, int(np.floor(x1))))
        y1_i = max(0, min(height - 1, int(np.floor(y1))))
        x2_i = max(0, min(width, int(np.ceil(x2))))
        y2_i = max(0, min(height, int(np.ceil(y2))))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue
        mask[y1_i:y2_i, x1_i:x2_i] = 1.0

    if sigma > 0:
        mask = gaussian_filter(mask, sigma=sigma)
    max_val = mask.max()
    if max_val > 0:
        mask = mask / max_val
    return mask


class Flux2FoveatedUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "noise_downsampled", "lr_downsample_factor"),
            output_params=("latents", "latents_downsampled", "input_latents", "input_latents_downsampled", "foveation_mask"),
            onload_model_names=("vae",)
        )
        try:
            self.saliency_detector = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
            self.saliency_detector.eval()
            print("Saliency detector loaded")
        except Exception as e:
            print(e)
            self.saliency_detector = None
            print("Saliency detector not loaded")
        self.bbox_detector = None

    def get_foveation_mask_from_image(
        self,
        pipe: "Flux2FoveatedImagePipeline",
        image: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Create a foveation mask from the input image, depending on the
        foveated_training_mode on the pipeline:
        - "saliency": current DeepGaze-based saliency-to-mask logic.
        - "bbox": use bbox_to_foveation_mask (same logic as run_bbox_detection.py)
          on provided bounding boxes, then downsample to latent grid size.
        image: (1, 3, H, W) in [-1, 1]. Returns (H//16, W//16) mask.
        """
        mode = getattr(pipe, "foveated_training_mode", "saliency")

        if mode == "bbox":
            # Run YOLO detector directly on the input image to obtain bboxes,
            # then convert them to a foveation mask using the exact same logic
            # as misc/run_bbox_detection.py.
            if image.dim() == 3:
                image = image.unsqueeze(0)
            device = image.device
            h_orig, w_orig = int(image.shape[2]), int(image.shape[3])

            # Lazy-load YOLO detector with same defaults as run_bbox_detection.py
            if self.bbox_detector is None:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:
                    print(
                        "ultralytics not installed; bbox-based foveation requires "
                        "`pip install ultralytics`. Falling back to no foveation mask."
                    )
                    return None
                # Use same default model and keep device selection at predict-time
                self.bbox_detector = YOLO("yolov8n.pt")
                self.bbox_detector.model.float()

            # Convert input tensor ([-1, 1]) to uint8 RGB numpy array, same scaling
            # convention as saliency path (0-255).
            img_01 = (image[0].float() + 1.0) / 2.0
            img_255 = (img_01.clamp(0, 1) * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            # Device string for YOLO; follow run_bbox_detection default conf params.
            device_str = str(pipe.device) if hasattr(pipe, "device") else "cpu"
            results = self.bbox_detector.predict(
                img_255,
                conf=0.25,
                device=device_str,
                verbose=False,
            )
            if not results:
                # No detections result: use full high-res mask
                mask_np = np.ones((h_orig, w_orig), dtype=np.float32)
            else:
                res = results[0]
                if res.boxes is None or res.boxes.xyxy is None or len(res.boxes) == 0:
                    # No boxes: use full high-res mask
                    mask_np = np.ones((h_orig, w_orig), dtype=np.float32)
                else:
                    boxes_np = res.boxes.xyxy.cpu().numpy()
                    # Full-resolution [H, W] mask using the exact bbox->mask logic.
                    mask_np = bbox_to_foveation_mask(h_orig, w_orig, boxes_np)
            mask_hr = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
            # Downsample by 32, binarize, then upsample 2x so effective factor is 16
            mask_hr_4d = mask_hr.unsqueeze(0).unsqueeze(0)
            h_32, w_32 = h_orig // 32, w_orig // 32
            mask_32 = F.interpolate(mask_hr_4d, size=(h_32, w_32), mode="bilinear", align_corners=False)
            mask_binary_32 = (mask_32.squeeze(0).squeeze(0) > 0.5).to(torch.float32).unsqueeze(0).unsqueeze(0)
            latent_h, latent_w = h_orig // 16, w_orig // 16
            mask_binary = F.interpolate(mask_binary_32, size=(latent_h, latent_w), mode="nearest")
            return mask_binary

        # Default: saliency-based foveation mask
        if self.saliency_detector is None:
            return None
        if image.dim() == 3:
            image = image.unsqueeze(0)
        device = image.device
        self.saliency_detector = self.saliency_detector.to(device)
        h_orig, w_orig = int(image.shape[2]), int(image.shape[3])
        # DeepGaze input: [0, 255] float, same as run_saliency_detection _frame_to_tensor
        img_01 = (image.float() + 1.0) / 2.0
        img_255 = img_01.clamp(0, 1) * 255.0
        h_inf = int(h_orig * _INFERENCE_SCALE)
        w_inf = int(w_orig * _INFERENCE_SCALE)
        img_inf = F.interpolate(img_255, size=(h_inf, w_inf), mode="bilinear", align_corners=False)
        centerbias = torch.zeros(1, h_inf, w_inf, device=device, dtype=torch.float32)
        with torch.no_grad():
            log_density = self.saliency_detector(img_inf, centerbias)
        smap = torch.exp(log_density[0, 0]).cpu().numpy()
        smax = float(smap.max())
        if smax > 0:
            smap = smap / smax
        smoothed = gaussian_filter(smap, sigma=_SPATIAL_SIGMA)
        smax = float(smoothed.max())
        if smax > 0:
            smoothed = smoothed / smax
        # _saliency_to_params (same as run_saliency_detection)
        h, w = smoothed.shape
        eps = 1e-8
        peak = float(smoothed.max())
        if peak < eps:
            cx_norm, cy_norm, r = 0.0, 0.0, _R_MIN
        else:
            thresh = _DENSE_THRESHOLD * peak
            dense_mask = smoothed >= thresh
            dense_smap = smoothed * dense_mask
            dense_total = dense_smap.sum() + eps
            y_coords = np.arange(h, dtype=np.float64)
            x_coords = np.arange(w, dtype=np.float64)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            cx_px = (dense_smap * x_grid).sum() / dense_total
            cy_px = (dense_smap * y_grid).sum() / dense_total
            cx_norm = (cx_px / w) - 0.5
            cy_norm = (cy_px / h) - 0.5
            dist = np.sqrt((x_grid - cx_px) ** 2 + (y_grid - cy_px) ** 2)
            dense_pixels = dense_mask.ravel()
            flat_dist = dist.ravel()[dense_pixels]
            flat_sal = dense_smap.ravel()[dense_pixels]
            sort_idx = np.argsort(flat_dist)
            cumsum = np.cumsum(flat_sal[sort_idx])
            target = _COVERAGE * dense_total
            hit = min(np.searchsorted(cumsum, target), len(flat_dist) - 1)
            radius_px = flat_dist[sort_idx[hit]]
            half_diag = 0.5 * np.sqrt(h ** 2 + w ** 2)
            r = float(np.clip(radius_px / half_diag, _R_MIN, _R_MAX))
        # Binarized circular mask at full image res
        cx = (cx_norm + 0.5) * w_orig
        cy = (cy_norm + 0.5) * h_orig
        diagonal = (h_orig ** 2 + w_orig ** 2) ** 0.5
        radius_px = r * (diagonal / 2.0)
        y = np.arange(h_orig, dtype=np.float64)
        x = np.arange(w_orig, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask_hr = (dist_sq <= radius_px ** 2).astype(np.float32)
        mask_hr = torch.from_numpy(mask_hr).to(device=device, dtype=torch.float32)
        # Downsample by 32, binarize, then upsample 2x so effective factor is 16
        mask_hr_4d = mask_hr.unsqueeze(0).unsqueeze(0)
        h_32, w_32 = h_orig // 32, w_orig // 32
        mask_32 = F.interpolate(mask_hr_4d, size=(h_32, w_32), mode="bilinear", align_corners=False)
        mask_binary_32 = (mask_32.squeeze(0).squeeze(0) > 0.5).to(torch.float32).unsqueeze(0).unsqueeze(0)
        latent_h, latent_w = h_orig // 16, w_orig // 16
        mask_binary = F.interpolate(mask_binary_32, size=(latent_h, latent_w), mode="nearest")
        return mask_binary

    def process(self, pipe: Flux2FoveatedImagePipeline, input_image, noise, noise_downsampled, lr_downsample_factor=2):
        if input_image is None:
            return {
                "latents": noise,
                "latents_downsampled": noise_downsampled,
                "input_latents": None,
                "input_latents_downsampled": None,
                "foveation_mask": None,
            }
        pipe.load_models_to_device(["vae"])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae.encode(image)
        # Decide between saliency-based and bbox-based mask creation.
        foveation_mask = self.get_foveation_mask_from_image(pipe, image)
        #save_image((image + 1) / 2.0, "/home/brianchc/foveated_diffusion/paper_figures/input_image.png")
        #save_image(foveation_mask, "/home/brianchc/foveated_diffusion/paper_figures/foveation_mask.png")
        #exit()
        _, _, latent_height, latent_width = input_latents.shape
        input_latents = rearrange(input_latents, "B C H W -> B (H W) C")

        # create downsample
        image_downsampled = F.interpolate(image, size=(image.shape[-2]//lr_downsample_factor, image.shape[-1]//lr_downsample_factor), mode='bicubic', align_corners=False)
        input_latents_downsampled = pipe.vae.encode(image_downsampled)
        input_latents_downsampled = rearrange(input_latents_downsampled, "B C H W -> B (H W) C") # 4X less tokens


        if pipe.scheduler.training:
            return {
                "latents": noise,
                "latents_downsampled": noise_downsampled,
                "input_latents": input_latents,
                "input_latents_downsampled": input_latents_downsampled,
                "foveation_mask": foveation_mask,
            }
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            latents_downsampled = pipe.scheduler.add_noise(input_latents_downsampled, noise_downsampled, timestep=pipe.scheduler.timesteps[0])
            return {
                "latents": latents,
                "latents_downsampled": latents_downsampled,
                "input_latents": input_latents,
                "input_latents_downsampled": input_latents_downsampled,
                "foveation_mask": foveation_mask,
            }


class Flux2FoveatedUnit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "foveation_mask", "lr_downsample_factor"),
            output_params=("image_ids",),
        )

    def prepare_latent_ids(self, height, width, device, dtype, foveation_mask=None, lr_factor=2):
        return Flux2FoveatedImagePipeline._prepare_latent_image_ids_foveation(
            batch_size=1,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
            foveation_mask=foveation_mask,
            lr_factor=lr_factor,
        )

    def process(self, pipe: Flux2FoveatedImagePipeline, height, width, foveation_mask, lr_downsample_factor=2):
        latent_ids, resolution_mask, resolution_mask_top_left = self.prepare_latent_ids(
            height // 16, width // 16, pipe.device, pipe.torch_dtype, foveation_mask, lr_factor=lr_downsample_factor
        )
        latent_ids = latent_ids.unsqueeze(0).expand(1, -1, -1)
        return {
            "image_ids": latent_ids,
            "resolution_mask": resolution_mask,
            "resolution_mask_top_left": resolution_mask_top_left,
        }


# =====================================================================
# Model Function for Foveated FLUX2
# =====================================================================

def model_fn_flux2_foveated(
    dit: Flux2DiTFoveated,
    latents=None,
    timestep=None,
    embedded_guidance=None,
    prompt_embeds=None,
    text_ids=None,
    image_ids=None,
    resolution_mask=None,
    resolution_mask_top_left=None,
    lr_downsample_factor=2,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    """Model function for foveated FLUX2 generation."""
    image_seq_len = latents.shape[1]
    embedded_guidance = torch.tensor([embedded_guidance], device=latents.device)

    model_output = dit(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=embedded_guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        resolution_mask=resolution_mask,
        resolution_mask_top_left=resolution_mask_top_left,
        lr_factor=lr_downsample_factor,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    
    model_output = model_output[:, :image_seq_len]
    return model_output