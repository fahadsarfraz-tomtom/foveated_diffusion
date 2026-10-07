import numpy as np
import torch

from src.masks.uncertainty import (
    centers_from_map,
    dense_mask_from_map,
    iou_top_beta,
    plan_from_dense_map,
    solve_x1,
    spearman,
)


def test_dense_mask_hits_budget():
    torch.manual_seed(0)
    umap = torch.rand(32, 32)
    for beta in (0.1, 0.25, 0.5):
        mask = dense_mask_from_map(umap, beta)
        frac = float(mask.mean())
        assert abs(frac - beta) < 0.05
        assert mask.shape == umap.shape


def test_dense_mask_selects_highest_values():
    umap = torch.zeros(8, 8)
    umap[2, 3] = 5.0
    umap[6, 1] = 4.0
    mask = dense_mask_from_map(umap, 2 / 64)
    assert mask[2, 3] == 1.0 and mask[6, 1] == 1.0
    assert float(mask.sum()) == 2.0


def test_centers_from_map_picks_separated_peaks():
    umap = torch.zeros(16, 16)
    umap[2, 2] = 3.0
    umap[12, 13] = 2.0
    centers = centers_from_map(umap, num_centers=2, min_dist_frac=0.2)
    assert len(centers) == 2
    for x, y in centers:
        assert -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5
    # first peak near top-left, second near bottom-right (Chao frame)
    assert centers[0][0] < 0 and centers[0][1] < 0
    assert centers[1][0] > 0 and centers[1][1] > 0


def test_plan_from_dense_map_budget_and_ratio():
    torch.manual_seed(1)
    umap = torch.rand(16, 16)
    plan = plan_from_dense_map(umap, beta=0.25, height=256, width=256,
                               device=torch.device("cpu"), lr_factor=2)
    assert plan.token_mask.shape == (16, 16)
    assert plan.full_res_mask.shape == (256, 256)
    assert abs(plan.hr_fraction - 0.25) < 0.05
    assert 0.0 < plan.token_ratio <= 1.0
    assert plan.policy == "uncertainty"
    assert plan.centers == []


def test_solve_x1_inverts_affine_system():
    torch.manual_seed(2)
    x1 = torch.randn(4, 16, 8)
    x0 = torch.randn(4, 16, 8)
    a, b, c, d = 0.4, 0.6, -1.0, 1.0  # x_t mix + flow target (x0 - x1)
    x_t = a * x1 + b * x0
    pred = c * x1 + d * x0
    x1_hat = solve_x1(x_t, pred, (a, b, c, d))
    assert torch.allclose(x1_hat, x1, atol=1e-5)


def test_solve_x1_degenerate_falls_back_to_pred():
    x_t = torch.randn(2, 4, 3)
    pred = torch.randn(2, 4, 3)
    out = solve_x1(x_t, pred, (1.0, 0.0, 1.0, 0.0))  # det == 0 (clean prediction)
    assert torch.equal(out, pred)


def test_spearman_and_iou_sanity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 16))
    assert spearman(a, a) > 0.999
    assert spearman(a, -a) < -0.999
    assert iou_top_beta(a, a, 0.25) == 1.0
    b = rng.normal(size=(16, 16))
    assert 0.0 <= iou_top_beta(a, b, 0.25) <= 1.0
