"""Axis aligned bounding box geometry.

Boxes are represented in the corner format ``(x1, y1, x2, y2)`` with
``x2 >= x1`` and ``y2 >= y1``. All functions accept and return ``torch``
tensors so they compose with the rest of the detector. Coordinates are in
pixel units unless stated otherwise.
"""

from __future__ import annotations

import torch
from torch import Tensor


def box_area(boxes: Tensor) -> Tensor:
    """Return the area of each box.

    Args:
        boxes: Tensor of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.

    Returns:
        Tensor of shape ``(N,)`` with non negative areas.
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise intersection over union between two sets of boxes.

    Args:
        boxes_a: Tensor of shape ``(N, 4)``.
        boxes_b: Tensor of shape ``(M, 4)``.

    Returns:
        Tensor of shape ``(N, M)`` where entry ``(i, j)`` is the IoU of
        ``boxes_a[i]`` with ``boxes_b[j]``. The value is in ``[0, 1]``.
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    area_a = box_area(boxes_a)
    area_b = box_area(boxes_b)

    # Intersection corners, broadcast over the (N, M) grid.
    top_left = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def clip_boxes(boxes: Tensor, height: int, width: int) -> Tensor:
    """Clip boxes so they lie inside an image of size ``(height, width)``."""
    clipped = boxes.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(min=0, max=width)
    clipped[:, 1::2] = clipped[:, 1::2].clamp(min=0, max=height)
    return clipped


def boxes_to_centers(boxes: Tensor) -> Tensor:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(cx, cy, w, h)`` format."""
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack([cx, cy, w, h], dim=1)


def centers_to_boxes(centers: Tensor) -> Tensor:
    """Convert ``(cx, cy, w, h)`` format back to corner boxes."""
    cx, cy, w, h = centers.unbind(dim=1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)
