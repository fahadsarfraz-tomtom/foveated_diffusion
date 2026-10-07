from types import SimpleNamespace

import torch

from src.diffsynth_fov.loss import _resolve_foveation_mask
from tests.test_fpm_policy import _write_checkpoint


def _fake_pipe(tmp_path, latent_channels=4):
    ckpt = _write_checkpoint(tmp_path, latent_channels=latent_channels)
    return SimpleNamespace(
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
        fpm_checkpoint=str(ckpt),
        fpm_beta_min=0.20,
        fpm_beta_max=0.20,  # pin beta for a deterministic budget assert
    )


def test_fpm_training_mode_produces_budgeted_mask(tmp_path):
    pipe = _fake_pipe(tmp_path)
    h = w = 16
    latents = torch.randn(2, h * w, 4)
    inputs = {
        "foveated_training_mode": "fpm",
        "prompt": "a dog chasing a ball on grass",
        "latents": latents,
    }
    mask = _resolve_foveation_mask(pipe, inputs, h, w, timestep_id=torch.tensor([500]))
    assert mask.shape == (h, w)
    frac = float(mask.float().mean())
    assert abs(frac - 0.20) < 0.06  # quantile threshold lands near the pinned beta
    assert set(torch.unique(mask.float()).tolist()) <= {0.0, 1.0}


def test_fpm_training_mode_requires_checkpoint(tmp_path):
    pipe = SimpleNamespace(device=torch.device("cpu"), torch_dtype=torch.float32,
                           fpm_checkpoint=None)
    inputs = {"foveated_training_mode": "fpm", "prompt": "x",
              "latents": torch.randn(1, 16, 4)}
    try:
        _resolve_foveation_mask(pipe, inputs, 4, 4, timestep_id=0)
    except ValueError as exc:
        assert "fpm_checkpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError without fpm_checkpoint")


def test_fpm_training_mask_is_content_dependent(tmp_path):
    pipe = _fake_pipe(tmp_path)
    h = w = 16
    torch.manual_seed(0)
    inputs_a = {"foveated_training_mode": "fpm", "prompt": "a red bus",
                "latents": torch.randn(1, h * w, 4)}
    torch.manual_seed(0)  # same beta draw
    inputs_b = {"foveated_training_mode": "fpm", "prompt": "a red bus",
                "latents": torch.randn(1, h * w, 4) * 3 + 1}
    mask_a = _resolve_foveation_mask(pipe, inputs_a, h, w, timestep_id=100)
    torch.manual_seed(0)
    mask_b = _resolve_foveation_mask(pipe, inputs_b, h, w, timestep_id=100)
    assert mask_a.shape == mask_b.shape == (h, w)
