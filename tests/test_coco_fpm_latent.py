import torch

from train_fpm_coco_latent import (
    add_scheduler_noise_to_spatial_latents,
    sequence_to_spatial,
    spatial_to_sequence,
)


class FakeScheduler:
    def add_noise(self, clean, noise, timestep):
        weight = timestep.float().view(1, 1, 1) / 10.0
        return clean * (1.0 - weight) + noise * weight


def test_spatial_sequence_round_trip():
    latents = torch.randn(2, 5, 4, 3)
    sequence = spatial_to_sequence(latents)

    assert sequence.shape == (2, 12, 5)
    assert torch.equal(sequence_to_spatial(sequence, 4, 3), latents)


def test_scheduler_noise_preserves_spatial_latent_shape(monkeypatch):
    latents = torch.randn(2, 5, 4, 3)

    monkeypatch.setattr(torch, "randn_like", lambda x: torch.ones_like(x))
    noisy = add_scheduler_noise_to_spatial_latents(
        FakeScheduler(),
        latents,
        timestep=torch.tensor([5.0]),
    )

    assert noisy.shape == latents.shape
    assert torch.allclose(noisy, latents * 0.5 + 0.5)
