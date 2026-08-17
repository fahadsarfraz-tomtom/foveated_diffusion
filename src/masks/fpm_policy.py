"""Inference-side wrapper that turns a trained FPM checkpoint into a mask policy.

Loads checkpoints written by ``train_fpm_coco.py`` / ``train_fpm_coco_latent.py``
(``save_checkpoint``: dict with ``fpm``, ``text_embedder``, ``args``) and exposes
``predict_centers(prompt, device)`` for the adaptive policy layer.

At text-to-image time there is no clean latent to condition on, so the policy
runs the FPM in *prior mode*: a prompt-seeded noise latent at the noisiest
trained timestep id. This matches the training distribution at high noise,
where the FPM must rely on the text pathway.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .fpm import FovealPredictionModule


class FpmMaskPolicy:
    """A trained FPM + text embedder packaged as a fixation-center predictor."""

    def __init__(
        self,
        fpm: FovealPredictionModule,
        text_embedder: torch.nn.Module,
        vocab_size: int,
        max_tokens: int,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
        objectness_threshold: float = 0.5,
        timestep_id: int = 0,
        checkpoint_path: str | None = None,
    ):
        self.fpm = fpm
        self.text_embedder = text_embedder
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.objectness_threshold = objectness_threshold
        self.timestep_id = timestep_id
        self.checkpoint_path = checkpoint_path

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: torch.device | str = "cpu",
        objectness_threshold: float = 0.5,
        timestep_id: int = 0,
    ) -> "FpmMaskPolicy":
        # Lazy import: the tokenizer/embedder live with the trainers, and the
        # masks package must stay importable without the training extras.
        from ..training.coco_fpm import HashTextEmbedder

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {}) or {}
        fpm_state = ckpt["fpm"]

        hidden_dim, latent_channels = fpm_state["spatial_enc.0.weight"].shape[:2]
        num_fixations = int(args.get("num_fixations") or fpm_state["object_head.3.bias"].shape[0])
        text_dim = int(args.get("text_dim") or fpm_state["text_proj.weight"].shape[0])
        vocab_size = int(args.get("vocab_size", 8192))
        max_tokens = int(args.get("max_tokens", 32))

        fpm = FovealPredictionModule(
            latent_channels=int(latent_channels),
            text_dim=text_dim,
            num_fixations=num_fixations,
            hidden_dim=int(hidden_dim),
        )
        fpm.load_state_dict(fpm_state)
        fpm.to(device).eval()

        text_embedder = HashTextEmbedder(vocab_size=vocab_size, text_dim=text_dim)
        text_embedder.load_state_dict(ckpt["text_embedder"])
        text_embedder.to(device).eval()

        latent_height, latent_width = cls._resolve_latent_size(checkpoint_path, args)
        return cls(
            fpm=fpm,
            text_embedder=text_embedder,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            latent_channels=int(latent_channels),
            latent_height=latent_height,
            latent_width=latent_width,
            objectness_threshold=objectness_threshold,
            timestep_id=timestep_id,
            checkpoint_path=str(checkpoint_path),
        )

    @staticmethod
    def _resolve_latent_size(checkpoint_path: str, args: dict) -> tuple[int, int]:
        config_path = Path(checkpoint_path).parent / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                h, w = config.get("latent_height"), config.get("latent_width")
                if h and w:
                    return int(h), int(w)
            except (json.JSONDecodeError, OSError):
                pass
        # FLUX2 VAE downsamples 8x; the RGB-proxy trainer uses image_size // 8 grids too.
        side = max(int(args.get("image_size", 256)) // 8, 8)
        return side, side

    @staticmethod
    def _prompt_seed(prompt: str) -> int:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False)

    @torch.no_grad()
    def predict_centers(
        self,
        prompt: str,
        device: torch.device | str,
    ) -> tuple[list[tuple[float, float]], list[float], dict[str, Any]]:
        """Predict fixation centers for a prompt.

        Returns ``(centers, radii, metadata)`` with centers in Chao's
        [-0.5, 0.5] frame. Slots pass through an objectness gate; when no slot
        clears the threshold the strongest slot is kept so the policy always
        emits at least one fixation.
        """
        from ..training.coco_fpm import token_ids_from_caption

        device = torch.device(device)
        if next(self.fpm.parameters()).device != device:
            self.fpm.to(device)
            self.text_embedder.to(device)

        token_ids = token_ids_from_caption(prompt, self.vocab_size, self.max_tokens)
        token_ids = token_ids.unsqueeze(0).to(device)
        text_embeddings = self.text_embedder(token_ids)

        generator = torch.Generator(device="cpu").manual_seed(self._prompt_seed(prompt))
        latents = torch.randn(
            1, self.latent_channels, self.latent_height, self.latent_width,
            generator=generator,
        ).to(device)
        timesteps = torch.full((1,), float(self.timestep_id), device=device)

        cx, cy, radii, object_logits, _weight = self.fpm.predict_slots(
            latents, text_embeddings, timesteps=timesteps,
        )
        objectness = torch.sigmoid(object_logits)[0]
        kept = (objectness >= self.objectness_threshold).nonzero().flatten().tolist()
        if not kept:
            kept = [int(objectness.argmax().item())]

        centers = [(float(cx[0, k]) - 0.5, float(cy[0, k]) - 0.5) for k in kept]
        pred_radii = [float(radii[0, k]) for k in kept]
        metadata = {
            "fpm_checkpoint": self.checkpoint_path,
            "fpm_timestep_id": self.timestep_id,
            "fpm_objectness": [round(float(o), 4) for o in objectness.tolist()],
            "fpm_kept_slots": kept,
            "fpm_pred_radii": [round(r, 4) for r in pred_radii],
        }
        return centers, pred_radii, metadata
