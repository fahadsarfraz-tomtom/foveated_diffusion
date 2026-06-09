"""Foveation mask geometry: circular / square / checkerboard / polygons / multi-circle.

A token-grid mask has shape (H//16, W//16) with values in {0, 1}; a full-resolution
mask has shape (H, W) with the same geometry. Both are produced for the foveated
pipeline (token-grid mask determines mixed-resolution token layout; full-res mask
optionally drives the merge-decode blend).
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def create_foveation_mask(
    height: int,
    width: int,
    center: Tuple[float, float] = (0.0, 0.0),
    r: float = 0.25,
    shape: str = "circular",
    device: torch.device = None,
    lr_factor: int = 2,
) -> torch.Tensor:
    """Token-grid foveation mask of shape (H//16, W//16).

    Args:
        height, width: full image dimensions.
        center: (x, y) normalized to [-0.5, 0.5].
        r: circular -> radius relative to half the diagonal;
           square   -> side length as a fraction of min(H, W).
        shape: "circular", "square", or "checkerboard".
        lr_factor: low-res periphery downsample factor (2 or 4).
    """
    if device is None:
        device = torch.device("cpu")
    high_h, high_w = height // 16, width // 16
    if shape == "checkerboard":
        i = torch.arange(high_h, device=device, dtype=torch.long)
        j = torch.arange(high_w, device=device, dtype=torch.long)
        ii, jj = torch.meshgrid(i, j, indexing="ij")
        return ((ii + jj) % 2 == 0).float()

    low_h, low_w = high_h // lr_factor, high_w // lr_factor
    y_range = torch.arange(low_h, device=device, dtype=torch.float32)
    x_range = torch.arange(low_w, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
    cx = math.floor((center[0] + 0.5) * low_w - 0.5) + 0.5
    cy = math.floor((center[1] + 0.5) * low_h - 0.5) + 0.5

    if shape == "square":
        side = r * min(low_h, low_w)
        half = side / 2.0
        in_x = (x_grid >= cx - half) & (x_grid <= cx + half)
        in_y = (y_grid >= cy - half) & (y_grid <= cy + half)
        mask_low = (in_x & in_y).unsqueeze(0).unsqueeze(0).float()
    else:
        diagonal = (low_h ** 2 + low_w ** 2) ** 0.5
        radius_px = r * (diagonal / 2.0)
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        mask_low = (dist_sq <= radius_px ** 2).unsqueeze(0).unsqueeze(0).float()

    mask_high = F.interpolate(mask_low, size=(high_h, high_w), mode="nearest")
    return mask_high.squeeze(0).squeeze(0)


def create_foveation_mask_full_res(
    height: int,
    width: int,
    center: Tuple[float, float] = (0.0, 0.0),
    r: float = 0.25,
    shape: str = "circular",
    device: torch.device = None,
) -> torch.Tensor:
    """Same geometry as `create_foveation_mask` but rasterized at full (H, W) resolution."""
    if device is None:
        device = torch.device("cpu")
    y_range = torch.arange(height, device=device, dtype=torch.float32)
    x_range = torch.arange(width, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
    cx = (center[0] + 0.5) * width
    cy = (center[1] + 0.5) * height

    if shape == "checkerboard":
        i = torch.arange(height, device=device, dtype=torch.long)
        j = torch.arange(width, device=device, dtype=torch.long)
        ii, jj = torch.meshgrid(i, j, indexing="ij")
        return ((ii + jj) % 2 == 0).float()
    if shape == "square":
        side = r * min(height, width)
        half = side / 2.0
        in_x = (x_grid >= cx - half) & (x_grid <= cx + half)
        in_y = (y_grid >= cy - half) & (y_grid <= cy + half)
        return (in_x & in_y).float()
    diagonal = (height ** 2 + width ** 2) ** 0.5
    radius_px = r * (diagonal / 2.0)
    dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    return (dist_sq <= radius_px ** 2).float()


def gaussian_blur_mask_2d(mask_4d: torch.Tensor, sigma_pixels: float, device, dtype) -> torch.Tensor:
    """Separable Gaussian blur on a [1, 1, H, W] mask. Matches the pipeline's soft blend."""
    k = max(3, int(math.ceil(3 * sigma_pixels)) * 2 + 1)
    if k % 2 == 0:
        k += 1
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma_pixels ** 2 + 1e-6))
    g = g / g.sum()
    pad = k // 2
    out = F.conv2d(mask_4d, g.view(1, 1, k, 1), padding=(pad, 0))
    out = F.conv2d(out, g.view(1, 1, 1, k), padding=(0, pad))
    return out


# ---------------------------------------------------------------------------
# Polygon (triangle / square / hexagon / star) mask rasterization
# ---------------------------------------------------------------------------

def polygon_vertices(shape_name: str, scale: float, center=(0.0, 0.0)):
    """Polygon vertices in normalized coords [-0.5, 0.5], `scale` = half-diagonal radius."""
    cx, cy = center
    if shape_name == "triangle":
        angles = [2 * math.pi * (i / 3) - math.pi / 2 for i in range(3)]
        return [(cx + scale * math.cos(a), cy + scale * math.sin(a)) for a in angles]
    if shape_name == "square":
        angles = [2 * math.pi * (i / 4) - math.pi / 4 for i in range(4)]
        return [(cx + scale * math.cos(a), cy + scale * math.sin(a)) for a in angles]
    if shape_name == "hexagon":
        angles = [2 * math.pi * (i / 6) - math.pi / 6 for i in range(6)]
        return [(cx + scale * math.cos(a), cy + scale * math.sin(a)) for a in angles]
    if shape_name == "star":
        outer = [2 * math.pi * (i / 5) - math.pi / 2 for i in range(5)]
        inner = [a + math.pi / 5 for a in outer]
        verts = []
        for i in range(5):
            verts.append((cx + scale * math.cos(outer[i]), cy + scale * math.sin(outer[i])))
            verts.append((cx + scale * 0.4 * math.cos(inner[i]), cy + scale * 0.4 * math.sin(inner[i])))
        return verts
    raise ValueError("shape_name must be triangle, square, hexagon, or star")


def _point_in_polygon(x: float, y: float, vertices: list) -> bool:
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def rasterize_polygon_mask(height: int, width: int, vertices_norm: list, device) -> torch.Tensor:
    y_range = torch.arange(height, device=device, dtype=torch.float32)
    x_range = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")
    x_norm = (xx + 0.5) / width - 0.5
    y_norm = (yy + 0.5) / height - 0.5
    try:
        from matplotlib.path import Path
        p = Path(vertices_norm)
        xy = np.stack([x_norm.cpu().numpy().ravel(), y_norm.cpu().numpy().ravel()], axis=1)
        inside = p.contains_points(xy)
        return torch.from_numpy(inside.astype(np.float32)).reshape(height, width).to(device)
    except ImportError:
        mask = torch.zeros(height, width, device=device, dtype=torch.float32)
        for i in range(height):
            for j in range(width):
                if _point_in_polygon(float(x_norm[i, j]), float(y_norm[i, j]), vertices_norm):
                    mask[i, j] = 1.0
        return mask


def polygon_mask_pair(height: int, width: int, shape_name: str, center, scale: float, device):
    """Return (token_grid_mask, full_res_mask) for a polygon, both with same geometry."""
    vertices = polygon_vertices(shape_name, scale, center=center)
    full_res = rasterize_polygon_mask(height, width, vertices, device)
    high_h, high_w = height // 16, width // 16
    token_mask = F.interpolate(
        full_res.unsqueeze(0).unsqueeze(0), size=(high_h, high_w), mode="nearest"
    ).squeeze(0).squeeze(0)
    return token_mask, full_res


# ---------------------------------------------------------------------------
# Multi-circle masks (used by the `multi_circle` trajectory)
# ---------------------------------------------------------------------------

def sample_nonoverlapping_circles(rng, num_circles: int, r_lo: float, r_hi: float, gap: float = 0.02):
    """Sample `num_circles` non-overlapping circles spread across image quadrants.

    Returns a list of (cx_norm, cy_norm, r) with normalized centers in [-0.5, 0.5]
    and r relative to half the diagonal.
    """
    zones = [
        (-0.42, -0.12, -0.42, -0.12),
        (0.12, 0.42, -0.42, -0.12),
        (-0.42, -0.12, 0.12, 0.42),
        (0.12, 0.42, 0.12, 0.42),
    ]
    zone_order = rng.permutation(len(zones))
    chosen_zones = [zone_order[i] for i in range(num_circles)]
    centers = []
    for zi in chosen_zones:
        xmin, xmax, ymin, ymax = zones[zi]
        centers.append((float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax))))
    radii = [float(rng.uniform(r_lo, r_hi)) for _ in range(num_circles)]
    r_to_norm = math.sqrt(2) / 2.0
    for _ in range(30):
        overlap = False
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                cix, ciy = centers[i]
                cjx, cjy = centers[j]
                d = math.sqrt((cix - cjx) ** 2 + (ciy - cjy) ** 2)
                need = (radii[i] + radii[j]) * r_to_norm + gap
                if d < need and d > 1e-6:
                    overlap = True
                    max_sum = (d - gap) / r_to_norm
                    scale = max_sum / (radii[i] + radii[j])
                    radii[i] = min(radii[i], radii[i] * scale)
                    radii[j] = min(radii[j], radii[j] * scale)
                elif d < 1e-6:
                    radii[i] = r_lo
                    radii[j] = r_lo
                    overlap = True
        if not overlap:
            break
    return [(centers[i][0], centers[i][1], max(r_lo, radii[i])) for i in range(num_circles)]


def multi_circle_mask(height: int, width: int, circles: list, device) -> torch.Tensor:
    """Union of multiple circles at full (H, W) resolution."""
    y_range = torch.arange(height, device=device, dtype=torch.float32)
    x_range = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_range, x_range, indexing="ij")
    diagonal = (height ** 2 + width ** 2) ** 0.5
    mask = torch.zeros(height, width, device=device, dtype=torch.float32)
    for (cx_norm, cy_norm, r) in circles:
        cx = (cx_norm + 0.5) * width
        cy = (cy_norm + 0.5) * height
        radius_px = r * (diagonal / 2.0)
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = torch.maximum(mask, (dist_sq <= radius_px ** 2).float())
    return mask
