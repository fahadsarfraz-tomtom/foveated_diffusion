"""Visualization helpers: tokenization-grid overlay, foveation outline,
foveation-circle video overlay, prompt I/O."""

import os

import numpy as np
import pandas as pd


def load_prompt_dataset(prompt_dataset_path: str):
    """Load prompts from a CSV with a `prompt` column."""
    df = pd.read_csv(prompt_dataset_path)
    return [row["prompt"] for _, row in df.iterrows()]


def create_tokenization_mask_vis(foveation_mask, height: int, width: int, lr_factor: int = 2):
    """Render the HR/LR token grid implied by a token-grid foveation mask.

    HR region (mask=1) is white with 16x16 token outlines; LR region (mask=0)
    is light gray with (16*lr_factor)x(16*lr_factor) outlines.
    """
    mask_np = foveation_mask.detach().cpu().numpy() if hasattr(foveation_mask, "detach") else np.asarray(foveation_mask)
    upsampled = np.repeat(np.repeat(mask_np, 16, axis=0), 16, axis=1)[:height, :width]

    vis = np.full((height, width, 3), 255, dtype=np.uint8)
    if lr_factor > 1:
        vis[upsampled < 0.5] = [225, 225, 225]

    yy = np.arange(height)
    xx = np.arange(width)
    border_px = 4
    hr_grid = np.zeros((height, width), dtype=bool)
    hr_grid[yy % 16 < border_px, :] = True
    hr_grid[:, xx % 16 < border_px] = True

    lr_grid = np.zeros((height, width), dtype=bool)
    lr_grid[yy % (16 * lr_factor) < border_px, :] = True
    lr_grid[:, xx % (16 * lr_factor) < border_px] = True

    hr_region = upsampled > 0.5
    outline = (hr_grid & hr_region) | (lr_grid & ~hr_region)
    vis[outline] = [0, 0, 0]

    vis[yy < border_px, :] = [0, 0, 0]
    vis[yy > height - border_px, :] = [0, 0, 0]
    vis[:, xx < border_px] = [0, 0, 0]
    vis[:, xx > width - border_px] = [0, 0, 0]
    return vis


def draw_foveation_outline(
    image_np,
    mask_token_grid,
    height: int,
    width: int,
    outline_width_frac: float = 0.01,
    color=(255, 0, 0),
):
    """Draw a white outline (with thin black backing) around the foveation region, in-place."""
    try:
        import cv2
    except ImportError:
        return image_np
    mask_np = mask_token_grid.detach().cpu().numpy() if hasattr(mask_token_grid, "detach") else np.asarray(mask_token_grid)
    mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
    mask_high = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_width = max(1, int(outline_width_frac * width))
    black_stroke = max(1, int(0.10 * border_width))
    white_stroke = max(1, int(0.80 * border_width))
    cv2.drawContours(image_np, contours, -1, (255, 255, 255), black_stroke + white_stroke)
    cv2.drawContours(image_np, contours, -1, (255, 255, 255), white_stroke)
    return image_np


def draw_foveation_circle_on_frames(
    frames,
    centers,
    radii,
    color=(255, 255, 255),
    thickness: int = 2,
):
    """Draw a circle on each frame at the per-frame interpolated foveation center / radius.

    Matches the paper-figure overlay style. `frames` is a list of HxWx3 arrays
    or PIL images (length = number of output frames). `centers` and `radii`
    are *per-latent-frame* keyframes; they are linearly interpolated to the
    output frame count and rasterized in pixel space.

    Returns a list of HxWx3 uint8 arrays. Falls back to returning the input
    unchanged if OpenCV is missing.
    """
    try:
        import cv2
    except ImportError:
        return list(frames)

    num_frames = len(frames)
    n_keys = len(centers)
    if n_keys == 0 or num_frames == 0:
        return list(frames)

    t_keys = np.linspace(0, 1, n_keys)
    t_frames = np.linspace(0, 1, num_frames)
    cx_interp = np.interp(t_frames, t_keys, [c[0] for c in centers])
    cy_interp = np.interp(t_frames, t_keys, [c[1] for c in centers])
    r_interp = np.interp(t_frames, t_keys, radii)

    out = []
    for t, frame in enumerate(frames):
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        else:
            frame = np.array(frame)
        if frame.dtype != np.uint8 or frame.max() <= 1.0:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        px_cx = int((cx_interp[t] + 0.5) * w)
        px_cy = int((cy_interp[t] + 0.5) * h)
        diagonal = (h ** 2 + w ** 2) ** 0.5
        px_r = int(round(r_interp[t] * (diagonal / 2.0)))
        cv2.circle(frame, (px_cx, px_cy), px_r, color, thickness)
        out.append(frame)
    return out
