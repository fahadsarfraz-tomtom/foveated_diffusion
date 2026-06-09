"""Foveated Wan2.1 DiT: CRPA self-attention rebinding + foveated model_fn.

The image-side analog is ``dit.py`` (``Flux2DiTFoveated`` + CRPA AttnProcessor).
For Wan, upstream's ``SelfAttention.forward`` is a plain method on the attention
module — no processor seam to subclass — so we install foveated attention by
in-place method-rebinding via ``convert_to_foveated_wan_dit``.

Exports:
  - ``convert_to_foveated_wan_dit(dit)``       : in-place block.self_attn.forward swap
  - ``model_fn_wan_video_foveated(...)``       : focused T2V model_fn, dual-track output
  - ``_foveated_self_attn_forward(self, x, freqs)`` : the rebound method
  - ``_build_phase_aligned_rope(...)``         : CRPA RoPE construction (HR + LR phases)
"""

import types
from typing import Optional

import torch
from einops import rearrange

from diffsynth.models.wan_video_dit import (
    WanModel,
    rope_apply,
    sinusoidal_embedding_1d,
)

from ..masks.state import FoveationState


# ---------------------------------------------------------------------------
# DiT method swap: foveated self-attention
# ---------------------------------------------------------------------------

def _foveated_self_attn_forward(self, x, freqs):
    """Drop-in replacement for ``SelfAttention.forward`` that handles foveation.

    When ``freqs`` is the standard precomputed RoPE tensor, behaviour matches
    upstream Wan exactly. When it's a 4-tuple
    ``(freqs_hr, freqs_lr, packed_hr_positions, packed_key_positions)``, we
    run cross-resolution phase-aligned attention (CRPA) — HR queries attend
    to full keys with HR RoPE; LR queries attend to a subsampled key set
    (one per HR block) with LR-aligned RoPE.
    """
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    if isinstance(freqs, tuple):
        freqs_hr, freqs_lr, packed_hr, packed_key = freqs
        out = torch.zeros_like(q)
        if packed_hr.any():
            q_hr = rope_apply(q[:, packed_hr, ...], freqs_hr[packed_hr], self.num_heads)
            k_hr = rope_apply(k, freqs_hr, self.num_heads)
            out[:, packed_hr, ...] = self.attn(q_hr, k_hr, v)
        if (~packed_hr).any():
            q_lr = rope_apply(q[:, ~packed_hr, ...], freqs_lr[~packed_hr], self.num_heads)
            k_lr = rope_apply(k[:, packed_key, ...], freqs_lr[packed_key], self.num_heads)
            out[:, ~packed_hr, ...] = self.attn(q_lr, k_lr, v[:, packed_key, ...])
        x = out
    else:
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
    return self.o(x)


def convert_to_foveated_wan_dit(dit: WanModel) -> int:
    """Rebind every block's ``self_attn.forward`` to the foveated version.

    Returns the number of blocks converted, for logging.
    """
    n = 0
    for block in dit.blocks:
        block.self_attn.forward = types.MethodType(
            _foveated_self_attn_forward, block.self_attn,
        )
        n += 1
    return n


# ---------------------------------------------------------------------------
# Foveated model_fn (focused T2V path)
# ---------------------------------------------------------------------------

def _build_phase_aligned_rope(dit, f, h, w, foveation_state, device):
    """Build the (freq_hr, freq_lr, packed_hr, packed_key) tuple consumed by CRPA."""
    hr_mask_flat = foveation_state.hr_grid_mask.reshape(-1).bool().to(device)
    lr_mask_flat = foveation_state.lr_grid_mask.reshape(-1).bool().to(device)
    packed_hr = foveation_state.packed_hr_positions.to(device)
    packed_key = foveation_state.packed_key_positions.to(device)

    # Standard HR-grid RoPE freqs (matches upstream).
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(device)

    # HR query freqs: copy from full grid at HR positions; LR-block positions
    # get the top-left of each 2x2 HR block (i.e. step-2 subsample).
    freq_hr = torch.zeros([packed_hr.shape[0]] + list(freqs.shape[1:]),
                          device=freqs.device, dtype=freqs.dtype)
    freq_hr[packed_hr] = freqs[hr_mask_flat]
    freq_hr_sub = freqs.reshape(f, h, w, 1, -1)[:, ::2, ::2].reshape(f * h * w // 4, 1, -1)
    freq_hr[~packed_hr] = freq_hr_sub[~lr_mask_flat]

    # LR query freqs: phase-aligned by repeat_interleave so 2x2 HR blocks
    # share a single phase (see paper §3.2 / Wu et al.).
    freqs2 = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h // 2].repeat_interleave(2, dim=0).view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w // 2].repeat_interleave(2, dim=0).view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(device)
    freqs_lr = torch.zeros_like(freq_hr)
    freqs_lr[packed_hr] = freqs2[hr_mask_flat]
    freqs_lr_sub = freqs2.reshape(f, h, w, 1, -1)[:, ::2, ::2].reshape(f * h * w // 4, 1, -1)
    freqs_lr[~packed_hr] = freqs_lr_sub[~lr_mask_flat]

    return (freq_hr, freqs_lr, packed_hr, packed_key)


def model_fn_wan_video_foveated(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    latents_lr: Optional[torch.Tensor] = None,
    foveation_state: Optional[FoveationState] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **_unused,
):
    """Foveated Wan T2V model_fn.

    Returns ``(noise_pred_hr, noise_pred_lr)`` — the LR prediction is None when
    ``foveation_state is None`` (vanilla path), matching the dual-output
    convention the foveated loss + pipeline expect.
    """
    # Timestep + context embeddings (vanilla Wan path).
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Broadcast x / timestep over the batched (posi, nega) context if needed.
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    x = dit.patchify(x)                        # (B, dim, f, h, w)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    if foveation_state is not None:
        # Pack HR + LR tokens into a single mixed-resolution sequence.
        hr_mask_flat = foveation_state.hr_grid_mask.reshape(-1).bool().to(x.device)
        lr_mask_flat = foveation_state.lr_grid_mask.reshape(-1).bool().to(x.device)
        packed_hr = foveation_state.packed_hr_positions.to(x.device)

        x_lr = dit.patchify(latents_lr)
        if x_lr.shape[0] != context.shape[0]:
            x_lr = torch.concat([x_lr] * context.shape[0], dim=0)
        x_lr = rearrange(x_lr, "b c f h w -> b (f h w) c").contiguous()

        x_merged = torch.zeros(
            x.shape[0], packed_hr.shape[0], x.shape[2],
            device=x.device, dtype=x.dtype,
        )
        x_merged[:, packed_hr] = x[:, hr_mask_flat]
        x_merged[:, ~packed_hr] = x_lr[:, ~lr_mask_flat]
        x = x_merged

        freqs = _build_phase_aligned_rope(dit, f, h, w, foveation_state, x.device)
    else:
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # Transformer blocks.
    for block in dit.blocks:
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    block, x, context, t_mod, freqs, use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                block, x, context, t_mod, freqs, use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)

    if foveation_state is not None:
        # Split the packed sequence back into HR + LR token streams.
        n_lr_blocks = f * h * w // 4
        x_lr_out = torch.zeros(x.shape[0], n_lr_blocks, x.shape[2],
                               device=x.device, dtype=x.dtype)
        x_lr_out[:, ~lr_mask_flat] = x[:, ~packed_hr]
        x_lr_out = dit.unpatchify(x_lr_out, (f, h // 2, w // 2))

        x_hr_out = torch.zeros(x.shape[0], f * h * w, x.shape[2],
                               device=x.device, dtype=x.dtype)
        x_hr_out[:, hr_mask_flat] = x[:, packed_hr]
        x_hr_out = dit.unpatchify(x_hr_out, (f, h, w))
        return x_hr_out, x_lr_out

    x = dit.unpatchify(x, (f, h, w))
    return x, None
