"""Per-frame foveation paths (center + radius trajectories) for video.

Two path types, both returning `(centers, radii)` lists of equal length —
one entry per latent frame:

- `random_path`: training default. Linear interpolation between two sampled
  endpoints. This is the mask distribution described in §4.1 of the paper
  ("a random foveation path for video") — NOT per-frame independent random.
- `spline`: inference default. Cubic spline through fixed keypoints; defaults
  describe a gentle S-curve at constant radius.

Also exposes `circle_mask`, the binary-circle rasterizer shared by both the
foveation-state builder and the saliency module.
"""

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline


DEFAULT_SPLINE_CENTERS = [
    (-0.2, -0.2), (-0.1, -0.2), (-0.1, -0.1),
    ( 0.0, -0.1), ( 0.0,  0.0), ( 0.0,  0.1),
    ( 0.1,  0.1), ( 0.1,  0.2), ( 0.2,  0.2),
]
DEFAULT_SPLINE_RADII = [0.3] * 9


def circle_mask(h: int, w: int, center, r: float, device, dtype=torch.float32):
    """Binary circular mask of shape (h, w).

    `center` is normalized to [-0.5, 0.5]; `r` is relative to half the diagonal.
    """
    cx = int((center[0] + 0.5) * w)
    cy = int((center[1] + 0.5) * h)
    diag = (h ** 2 + w ** 2) ** 0.5
    rp = r * diag / 2.0
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= rp ** 2).to(dtype).to(device)


def sample_random_path(
    length: int,
    device: torch.device = torch.device("cpu"),
    center_range: tuple = (-0.3, 0.3),
    r_range: tuple = (0.2, 0.5),
):
    """Sample a straight-line path; return per-frame (centers, radii).

    Endpoints are sampled uniformly inside `center_range`x`center_range` for
    (cx, cy) and `r_range` for radii. Linear interpolation gives the per-frame
    values.
    """
    def _u(lo, hi):
        return torch.rand(1, device=device).item() * (hi - lo) + lo

    start = (_u(*center_range), _u(*center_range))
    end   = (_u(*center_range), _u(*center_range))
    sr, er = _u(*r_range), _u(*r_range)

    cx = torch.linspace(start[0], end[0], length).tolist()
    cy = torch.linspace(start[1], end[1], length).tolist()
    r  = torch.linspace(sr, er, length).tolist()
    return list(zip(cx, cy)), r


def sample_spline_path(
    length: int,
    center_list: list = None,
    radius_list: list = None,
):
    """Cubic-spline path through `center_list` / `radius_list`.

    Defaults to the module-level `DEFAULT_SPLINE_*` (gentle S-curve at
    constant radius 0.3).
    """
    centers = center_list if center_list is not None else DEFAULT_SPLINE_CENTERS
    radii   = radius_list if radius_list is not None else DEFAULT_SPLINE_RADII

    n_c, n_r = len(centers), len(radii)
    s_center = min(n_c * 0.01, 0.08)
    s_radius = n_r * 0.005
    k_center = min(1, n_c - 1) if n_c <= 3 else min(3, n_c - 1)
    k_radius = min(3, n_r - 1)

    t_c = np.linspace(0, 1, n_c)
    t_r = np.linspace(0, 1, n_r)
    t   = np.linspace(0, 1, length)

    cx = UnivariateSpline(t_c, [c[0] for c in centers], s=s_center, k=k_center)(t)
    cy = UnivariateSpline(t_c, [c[1] for c in centers], s=s_center, k=k_center)(t)
    r  = UnivariateSpline(t_r, radii, s=s_radius, k=k_radius)(t)
    return list(zip(cx.tolist(), cy.tolist())), r.tolist()
