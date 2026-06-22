"""Dense anchor boxes for a single feature stride.

Aerial objects are tiny, so the detector runs on a fine stride (a small
downsampling factor) and uses small anchor sizes. This module produces the
flat anchor grid that the detection head predicts offsets against.
"""

from __future__ import annotations

import torch
from torch import Tensor


def generate_anchors(
    height: int,
    width: int,
    stride: int,
    sizes: tuple[int, ...] = (6, 10, 16),
    aspect_ratios: tuple[float, ...] = (1.0,),
) -> Tensor:
    """Generate anchors tiled over a feature map.

    The feature map has shape ``(height // stride, width // stride)``. At each
    location we place one anchor per ``(size, aspect_ratio)`` combination,
    centred on that location mapped back to image coordinates.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        stride: Downsampling factor of the feature map relative to the image.
        sizes: Anchor side lengths (square root of area) in pixels.
        aspect_ratios: Width to height ratios.

    Returns:
        Tensor of shape ``(num_locations * num_anchors_per_loc, 4)`` in
        ``(x1, y1, x2, y2)`` format. The ordering is row major over feature
        locations, then over the ``(size, ratio)`` combinations.
    """
    feat_h = height // stride
    feat_w = width // stride

    shifts_x = (torch.arange(feat_w, dtype=torch.float32) + 0.5) * stride
    shifts_y = (torch.arange(feat_h, dtype=torch.float32) + 0.5) * stride
    centre_y, centre_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    centres = torch.stack(
        [centre_x.reshape(-1), centre_y.reshape(-1)], dim=1
    )  # (L, 2)

    base_wh = []
    for size in sizes:
        for ratio in aspect_ratios:
            w = size * (ratio ** 0.5)
            h = size / (ratio ** 0.5)
            base_wh.append([w, h])
    base_wh_t = torch.tensor(base_wh, dtype=torch.float32)  # (A, 2)

    num_loc = centres.shape[0]
    num_anchor = base_wh_t.shape[0]

    centres_exp = centres[:, None, :].expand(num_loc, num_anchor, 2)
    wh_exp = base_wh_t[None, :, :].expand(num_loc, num_anchor, 2)

    half = wh_exp * 0.5
    x1y1 = centres_exp - half
    x2y2 = centres_exp + half
    anchors = torch.cat([x1y1, x2y2], dim=2).reshape(-1, 4)
    return anchors


def num_anchors_per_location(
    sizes: tuple[int, ...] = (6, 10, 16),
    aspect_ratios: tuple[float, ...] = (1.0,),
) -> int:
    """Number of anchors placed at each feature map location."""
    return len(sizes) * len(aspect_ratios)
