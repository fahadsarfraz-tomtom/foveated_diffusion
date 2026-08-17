import json

import torch

from src.masks.adaptive import AdaptiveFoveationConfig, AdaptiveFoveationPolicy
from src.masks.fpm import FovealPredictionModule
from src.masks.fpm_policy import FpmMaskPolicy
from src.training.coco_fpm import HashTextEmbedder


def _write_checkpoint(tmp_path, num_fixations=3, hidden_dim=32, text_dim=16,
                      latent_channels=4, vocab_size=256, max_tokens=8):
    fpm = FovealPredictionModule(
        latent_channels=latent_channels,
        text_dim=text_dim,
        num_fixations=num_fixations,
        hidden_dim=hidden_dim,
    )
    text_embedder = HashTextEmbedder(vocab_size=vocab_size, text_dim=text_dim)
    ckpt_path = tmp_path / "final.pt"
    torch.save(
        {
            "step": 1,
            "fpm": fpm.state_dict(),
            "text_embedder": text_embedder.state_dict(),
            "optimizer": {},
            "args": {
                "num_fixations": num_fixations,
                "hidden_dim": hidden_dim,
                "text_dim": text_dim,
                "vocab_size": vocab_size,
                "max_tokens": max_tokens,
                "image_size": 256,
            },
        },
        ckpt_path,
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"latent_channels": latent_channels, "latent_height": 16, "latent_width": 16})
    )
    return ckpt_path


def test_fpm_policy_load_and_predict(tmp_path):
    ckpt_path = _write_checkpoint(tmp_path)
    policy = FpmMaskPolicy.load(str(ckpt_path), device="cpu")

    assert policy.latent_channels == 4
    assert (policy.latent_height, policy.latent_width) == (16, 16)

    centers, radii, metadata = policy.predict_centers("a dog with a ball", "cpu")
    assert 1 <= len(centers) <= 3
    assert len(centers) == len(radii)
    for (x, y), r in zip(centers, radii):
        assert -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5
        assert 0.0 < r <= 0.6
    assert len(metadata["fpm_objectness"]) == 3
    assert metadata["fpm_kept_slots"]


def test_fpm_policy_is_deterministic_per_prompt(tmp_path):
    ckpt_path = _write_checkpoint(tmp_path)
    policy = FpmMaskPolicy.load(str(ckpt_path), device="cpu")
    first = policy.predict_centers("a red bus on a street", "cpu")
    second = policy.predict_centers("a red bus on a street", "cpu")
    assert first[0] == second[0]
    assert first[1] == second[1]


def test_adaptive_plan_image_with_fpm_policy(tmp_path):
    ckpt_path = _write_checkpoint(tmp_path)
    cfg = AdaptiveFoveationConfig(
        policy="fpm",
        fixed_beta=0.25,
        beta_mode="fixed",
        fpm_checkpoint=str(ckpt_path),
    )
    policy = AdaptiveFoveationPolicy(cfg)
    plan = policy.plan_image(
        prompt="a cat sitting on a table",
        height=256,
        width=256,
        device=torch.device("cpu"),
        num_inference_steps=8,
        lr_factor=2,
    )
    assert plan.token_mask.shape == (16, 16)
    assert plan.policy == "fpm"
    assert 0.15 <= plan.hr_fraction <= 0.40  # solved radius lands near beta
    assert plan.metadata["fpm_kept_slots"]


def test_adaptive_plan_image_with_centers_override(tmp_path):
    cfg = AdaptiveFoveationConfig(policy="saliency", fixed_beta=0.25, beta_mode="fixed")
    policy = AdaptiveFoveationPolicy(cfg)
    plan = policy.plan_image(
        prompt="anything",
        height=256,
        width=256,
        device=torch.device("cpu"),
        num_inference_steps=8,
        lr_factor=2,
        centers_override=[(0.1, -0.2)],
    )
    assert plan.centers == [(0.1, -0.2)]
    assert 0.15 <= plan.hr_fraction <= 0.40
