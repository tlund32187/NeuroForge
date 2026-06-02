# pyright: basic
"""Render learned A1 features as an image grid — "see what perception learned".

A1's STDP detectors are receptive fields over the retina's ON/OFF contrast.
This renders each feature's field (ON minus OFF) as a tile, per-feature
contrast-normalized, and lays them out in a grid — so the bank's learning is
inspectable (did it find edges / corners / blobs / textures?). It returns a
``uint8`` image tensor; callers save it (e.g. ``torchvision.io.write_png``) or
feed it to a dashboard view. This is the Track-D feature view's data/renderer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from neuroforge.core.torch_utils import require_torch

if TYPE_CHECKING:
    from neuroforge.vision.encoding.stdp_features import STDPFeatureLayer

__all__ = ["render_feature_atlas"]


def render_feature_atlas(
    layer: STDPFeatureLayer, *, scale: int = 8, sep: int = 1, cols: int | None = None,
) -> Any:
    """Lay out a feature layer's receptive fields as a single ``[H, W]`` uint8 image.

    Each tile is one feature's ON-minus-OFF field, upscaled by *scale* and
    contrast-normalized independently so weak and strong features are both visible.
    """
    torch = require_torch()
    weights = layer.weights  # [F, in_ch, patch, patch]
    n_features, channels, patch_h, patch_w = weights.shape
    field = weights[:, 0] - weights[:, 1] if channels >= 2 else weights[:, 0]

    flat = field.reshape(n_features, -1).float()
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    norm = ((flat - lo) / (hi - lo + 1e-6)).reshape(n_features, patch_h, patch_w)
    tile = norm.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)

    tile_h, tile_w = patch_h * scale, patch_w * scale
    ncols = cols or math.ceil(math.sqrt(n_features))
    nrows = math.ceil(n_features / ncols)
    grid_h = nrows * tile_h + (nrows + 1) * sep
    grid_w = ncols * tile_w + (ncols + 1) * sep

    canvas = torch.full((grid_h, grid_w), 0.15, dtype=torch.float32)  # separator grey
    for i in range(n_features):
        row, col = divmod(i, ncols)
        y = sep + row * (tile_h + sep)
        x = sep + col * (tile_w + sep)
        canvas[y : y + tile_h, x : x + tile_w] = tile[i]

    return (canvas.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
