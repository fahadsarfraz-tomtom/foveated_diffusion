"""Saliency-driven foveation path extraction via DeepGaze IIE.

Optional dependency: `deepgaze-pytorch`. The module imports cleanly without it;
`load_deepgaze_model` raises with a clear message on first use if it's missing.

Public API:
    load_deepgaze_model(device) -> torch.nn.Module
    extract_saliency_path(model, frames, device) -> (centers, radii)
"""

import math

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


_INFERENCE_SCALE = 0.5
_TEMPORAL_SIGMA = 0.0
_SPATIAL_SIGMA = 20.0
_R_MIN, _R_MAX = 0.2, 0.5
_COVERAGE = 0.9
_DENSE_THRESHOLD = 0.3


def load_deepgaze_model(device):
    """Lazy DeepGaze loader. Raises ImportError if `deepgaze-pytorch` is missing."""
    try:
        import deepgaze_pytorch
    except ImportError as exc:
        raise ImportError(
            "Saliency-based foveation requires `pip install deepgaze-pytorch`."
        ) from exc
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    model.eval()
    return model


def _frame_to_tensor(pil_img):
    arr = np.array(pil_img).astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _smooth_saliency_volume(smaps):
    volume = np.stack(smaps, axis=0)
    volume = gaussian_filter(volume, sigma=(_TEMPORAL_SIGMA, _SPATIAL_SIGMA, _SPATIAL_SIGMA))
    out = []
    for t in range(volume.shape[0]):
        s = volume[t]
        smax = float(s.max())
        if smax > 0:
            s = s / smax
        out.append(s)
    return out


def _saliency_to_params(smap):
    """Weighted centroid + coverage-radius from a single saliency heatmap."""
    h, w = smap.shape
    eps = 1e-8
    peak = smap.max()
    if peak < eps:
        return (0.0, 0.0), _R_MIN

    thresh = _DENSE_THRESHOLD * peak
    dense_mask = smap >= thresh
    dense_smap = smap * dense_mask
    dense_total = dense_smap.sum() + eps

    y_coords = np.arange(h, dtype=np.float64)
    x_coords = np.arange(w, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    cx_px = (dense_smap * x_grid).sum() / dense_total
    cy_px = (dense_smap * y_grid).sum() / dense_total
    cx_norm = (cx_px / w) - 0.5
    cy_norm = (cy_px / h) - 0.5

    dist = np.sqrt((x_grid - cx_px) ** 2 + (y_grid - cy_px) ** 2)
    dense_pixels = dense_mask.ravel()
    flat_dist = dist.ravel()[dense_pixels]
    flat_sal = dense_smap.ravel()[dense_pixels]
    sort_idx = np.argsort(flat_dist)
    cumsum = np.cumsum(flat_sal[sort_idx])
    target = _COVERAGE * dense_total
    hit = min(np.searchsorted(cumsum, target), len(flat_dist) - 1)
    radius_px = flat_dist[sort_idx[hit]]
    half_diag = 0.5 * math.sqrt(h ** 2 + w ** 2)
    r = float(np.clip(radius_px / half_diag, _R_MIN, _R_MAX))
    return (float(cx_norm), float(cy_norm)), r


def extract_saliency_path(model, frames, device):
    """Run DeepGaze on `frames` (list[PIL.Image]) and return a foveation path.

    Pipeline: half-res inference -> heavy 3D smoothing -> per-frame weighted
    centroid + coverage radius. Returns `(centers, radii)` — two lists of
    length `len(frames)`, ready to be passed to `build_state`.
    """
    if not frames:
        return [], []
    w_orig, h_orig = frames[0].size
    w_inf, h_inf = int(w_orig * _INFERENCE_SCALE), int(h_orig * _INFERENCE_SCALE)
    resized = [f.resize((w_inf, h_inf), Image.LANCZOS) for f in frames]

    all_tensors = torch.cat([_frame_to_tensor(f) for f in resized], dim=0).to(device)
    centerbias = torch.zeros(1, h_inf, w_inf, device=device).expand(all_tensors.shape[0], -1, -1)
    with torch.no_grad():
        log_density = model(all_tensors, centerbias)

    raw_smaps = []
    for j in range(log_density.shape[0]):
        smap = np.exp(log_density[j, 0].cpu().numpy())
        smax = float(smap.max())
        if smax > 0:
            smap = smap / smax
        raw_smaps.append(smap)

    smoothed = _smooth_saliency_volume(raw_smaps)
    centers, radii = [], []
    for smap in smoothed:
        c, r = _saliency_to_params(smap)
        centers.append(c)
        radii.append(r)
    return centers, radii
