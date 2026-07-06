import torch

from src.masks.adaptive import (
    AdaptiveFoveationConfig,
    AdaptiveFoveationPolicy,
    nafo_beta,
    token_ratio_from_mask,
)
from src.masks.fpm import FovealPredictionModule, SpatialTokenScorer, gaussian_weight_map


def test_nafo_beta_decreases_from_early_to_late_steps():
    early = nafo_beta(0, 10, beta_min=0.2, beta_max=0.85, schedule="cosine")
    late = nafo_beta(9, 10, beta_min=0.2, beta_max=0.85, schedule="cosine")
    assert early > late
    assert late == 0.2


def test_adaptive_image_plan_has_chao_compatible_masks():
    cfg = AdaptiveFoveationConfig(
        policy="textfov_proxy",
        num_fixations=3,
        fixed_beta=0.25,
        beta_mode="fixed",
    )
    policy = AdaptiveFoveationPolicy(cfg)
    plan = policy.plan_image(
        prompt="a dog with a small ball in the garden",
        height=256,
        width=256,
        device=torch.device("cpu"),
        num_inference_steps=8,
        lr_factor=2,
    )

    assert plan.token_mask.shape == (16, 16)
    assert plan.full_res_mask.shape == (256, 256)
    assert len(plan.centers) == 3
    assert len(plan.radii) == 3
    assert 0.0 < plan.hr_fraction <= 1.0
    assert 0.0 < plan.token_ratio <= 1.0
    assert token_ratio_from_mask(plan.token_mask, lr_factor=2) == plan.token_ratio


def test_adaptive_video_plan_matches_latent_frame_count():
    cfg = AdaptiveFoveationConfig(policy="prompt_hash", num_fixations=4, fixed_beta=0.25)
    policy = AdaptiveFoveationPolicy(cfg)
    centers, radii, metadata = policy.plan_video(
        prompt="a sports car driving past a city sign",
        height=480,
        width=832,
        num_frames=81,
        device=torch.device("cpu"),
        lr_factor=2,
    )

    assert len(centers) == 21
    assert len(radii) == 21
    assert metadata["latent_length"] == 21
    assert metadata["radius"] > 0


def test_trainable_fpm_scaffold_forward_shapes():
    module = FovealPredictionModule(
        latent_channels=4,
        text_dim=32,
        num_fixations=3,
        hidden_dim=32,
    )
    latents = torch.randn(2, 4, 32, 32)
    text_embeddings = torch.randn(2, 5, 32)
    timesteps = torch.tensor([1, 2])

    cx, cy, radius, weight = module(
        latents,
        text_embeddings,
        timesteps=timesteps,
        out_height=16,
        out_width=16,
    )
    cx_slots, cy_slots, radius_slots, object_logits, weight_slots = module.predict_slots(
        latents,
        text_embeddings,
        timesteps=timesteps,
        out_height=16,
        out_width=16,
    )

    assert cx.shape == (2, 3)
    assert cy.shape == (2, 3)
    assert radius.shape == (2, 3)
    assert weight.shape == (2, 16, 16)
    assert cx_slots.shape == (2, 3)
    assert cy_slots.shape == (2, 3)
    assert radius_slots.shape == (2, 3)
    assert object_logits.shape == (2, 3)
    assert weight_slots.shape == (2, 16, 16)
    assert torch.all((cx >= 0) & (cx <= 1))
    assert torch.all((cy >= 0) & (cy <= 1))
    assert torch.all(weight >= 0)


def test_text_token_scorer_and_gaussian_map_shapes():
    scorer = SpatialTokenScorer(text_dim=16, hidden_dim=8)
    scores = scorer(torch.randn(2, 7, 16))
    assert scores.shape == (2, 7)
    assert torch.all((scores >= 0) & (scores <= 1))

    cx = torch.tensor([[0.5, 0.25]])
    cy = torch.tensor([[0.5, 0.75]])
    radius = torch.tensor([[0.2, 0.15]])
    weight = gaussian_weight_map(cx, cy, radius, height=8, width=10)
    assert weight.shape == (1, 8, 10)
    assert torch.isclose(weight.max(), torch.tensor(1.0), atol=1e-6)
