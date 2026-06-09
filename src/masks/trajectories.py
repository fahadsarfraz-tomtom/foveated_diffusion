"""Generate sequences of foveation masks for the trajectory-grid experiments.

A single trajectory produces `num_cols` (token-grid, full-res) mask pairs and is
reused for every prompt in the grid.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .shapes import (
    create_foveation_mask,
    create_foveation_mask_full_res,
    multi_circle_mask,
    polygon_mask_pair,
    sample_nonoverlapping_circles,
)


def generate_foveation_trajectory_masks(height, width, args, device, lr_factor: int = 2):
    """Build (masks, full_res_masks) for `args.foveation_trajectory_type`.

    Trajectory types: radius / circular / random_circular / polygons / multi_circle / grid / spiral.
    The same trajectory is reused across prompts.
    """
    num_cols = getattr(args, "num_cols", 4)
    orbit_radius = getattr(args, "orbit_radius", 0.25)
    mask_radius = getattr(args, "mask_radius", 0.30)
    traj_type = getattr(args, "foveation_trajectory_type", "circular")
    mask_shape = getattr(args, "mask_shape", "circular")

    masks, full_res_masks = [], []
    rng = np.random.default_rng(getattr(args, "seed", 0))

    if traj_type == "radius":
        for r in np.linspace(0.3, 0.7, num_cols):
            r = float(r)
            masks.append(create_foveation_mask(height, width, (0, 0), r, mask_shape, device, lr_factor))
            full_res_masks.append(create_foveation_mask_full_res(height, width, (0, 0), r, mask_shape, device))

    elif traj_type == "polygons":
        shape_choices = ["triangle", "square", "hexagon", "star"]
        center_range = getattr(args, "polygon_center_range", 0.25)
        scale_lo, scale_hi = 0.3, 0.4
        for col in range(num_cols):
            shape_name = shape_choices[col % len(shape_choices)]
            cx = float(rng.uniform(-center_range, center_range))
            cy = float(rng.uniform(-center_range, center_range))
            scale = float(rng.uniform(scale_lo, scale_hi))
            m, m_full = polygon_mask_pair(height, width, shape_name, (cx, cy), scale, device)
            masks.append(m)
            full_res_masks.append(m_full)

    elif traj_type == "grid":
        grid_rows = getattr(args, "grid_rows", 3)
        grid_cols = getattr(args, "grid_cols", 3)
        # auto-compute center range so a `mask_radius` circle never leaves the frame
        low_h_g = height // 16 // lr_factor
        low_w_g = width // 16 // lr_factor
        diagonal_g = (low_h_g ** 2 + low_w_g ** 2) ** 0.5
        radius_px_g = mask_radius * (diagonal_g / 2.0)
        n_min_x = math.floor(radius_px_g - 0.5)
        n_min_y = math.floor(radius_px_g - 0.5)
        n_max_x = low_w_g - 2 - n_min_x
        n_max_y = low_h_g - 2 - n_min_y
        x_min = (n_min_x + 1) / low_w_g - 0.5
        x_max = (n_max_x + 1) / low_w_g - 0.5
        y_min = (n_min_y + 1) / low_h_g - 0.5
        y_max = (n_max_y + 1) / low_h_g - 0.5
        x_positions = np.linspace(x_min, x_max, grid_cols)
        y_positions = np.linspace(y_min, y_max, grid_rows)
        for row in range(grid_rows):
            for col in range(grid_cols):
                center = (float(x_positions[col]), float(y_positions[row]))
                masks.append(create_foveation_mask(height, width, center, mask_radius, mask_shape, device, lr_factor))
                full_res_masks.append(create_foveation_mask_full_res(height, width, center, mask_radius, mask_shape, device))

    elif traj_type == "spiral":
        # Two-pass spiral: outward (r grows 0 -> 0.5) then inward (r grows 0.5 -> 1).
        # Auto-compute orbit_radius so the circle never leaves the frame.
        low_h_s = height // 16 // lr_factor
        low_w_s = width // 16 // lr_factor
        diagonal_s = math.sqrt(low_h_s ** 2 + low_w_s ** 2)
        min_dim_s = min(low_h_s, low_w_s)
        r_max, r_mid, r_min = 1.0, 0.5, 0.0
        r_mid_norm = r_mid * diagonal_s / (2.0 * min_dim_s)
        orbit_radius = 0.5 - r_mid_norm
        n_first = num_cols // 2
        n_second = num_cols - n_first
        for i in range(n_first):
            t = i / max(n_first - 1, 1)
            orbit = orbit_radius * t
            angle = 2 * math.pi * t
            center = (orbit * math.cos(angle), orbit * math.sin(angle))
            r = r_min + (r_mid - r_min) * t
            masks.append(create_foveation_mask(height, width, center, r, "circular", device, lr_factor))
            full_res_masks.append(create_foveation_mask_full_res(height, width, center, r, "circular", device))
        for i in range(n_second):
            t = i / max(n_second - 1, 1)
            orbit = orbit_radius * (1.0 - t)
            angle = 2 * math.pi + 2 * math.pi * t
            center = (orbit * math.cos(angle), orbit * math.sin(angle))
            r = r_mid + (r_max - r_mid) * t
            masks.append(create_foveation_mask(height, width, center, r, "circular", device, lr_factor))
            full_res_masks.append(create_foveation_mask_full_res(height, width, center, r, "circular", device))

    elif traj_type == "multi_circle":
        num_circles_lo, num_circles_hi = 2, 3
        r_lo, r_hi, gap = 0.3, 0.3, 0.3
        high_h, high_w = height // 16, width // 16
        for _ in range(num_cols):
            num_circles = int(rng.integers(num_circles_lo, num_circles_hi))
            circles = sample_nonoverlapping_circles(rng, num_circles, r_lo, r_hi, gap=gap)
            m_full = multi_circle_mask(height, width, circles, device)
            m = F.interpolate(
                m_full.unsqueeze(0).unsqueeze(0), size=(high_h, high_w), mode="nearest"
            ).squeeze(0).squeeze(0)
            masks.append(m)
            full_res_masks.append(m_full)

    elif traj_type == "random_circular":
        angles = rng.uniform(0, 2 * math.pi, size=num_cols)
        for angle in angles:
            angle = float(angle)
            center = (orbit_radius * math.cos(angle), orbit_radius * math.sin(angle))
            masks.append(create_foveation_mask(height, width, center, mask_radius, mask_shape, device, lr_factor))
            full_res_masks.append(create_foveation_mask_full_res(height, width, center, mask_radius, mask_shape, device))

    else:  # "circular"
        for col in range(num_cols):
            angle = 2 * math.pi * (col / num_cols)
            center = (orbit_radius * math.cos(angle), orbit_radius * math.sin(angle))
            masks.append(create_foveation_mask(height, width, center, mask_radius, mask_shape, device, lr_factor))
            full_res_masks.append(create_foveation_mask_full_res(height, width, center, mask_radius, mask_shape, device))

    return masks, full_res_masks
