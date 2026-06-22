"""Decode network outputs into final boxes.

Two steps live here. First, box deltas are applied to anchors to produce
predicted boxes (the inverse of the encoding used during training). Second,
non maximum suppression removes duplicate detections that point at the same
object, which is essential when anchors are dense.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .boxes import box_iou, boxes_to_centers, clip_boxes


# Standard deltas weights. They scale the regression targets so center and
# size offsets contribute comparably to the loss.
DEFAULT_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


def encode_boxes(
    anchors: Tensor,
    target_boxes: Tensor,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> Tensor:
    """Encode target boxes as deltas relative to anchors.

    Args:
        anchors: Tensor of shape ``(N, 4)``.
        target_boxes: Tensor of shape ``(N, 4)`` aligned with ``anchors``.
        weights: Per component scaling ``(wx, wy, ww, wh)``.

    Returns:
        Tensor of shape ``(N, 4)`` of deltas ``(dx, dy, dw, dh)``.
    """
    anc = boxes_to_centers(anchors)
    tgt = boxes_to_centers(target_boxes)
    wx, wy, ww, wh = weights

    dx = wx * (tgt[:, 0] - anc[:, 0]) / anc[:, 2].clamp(min=1e-6)
    dy = wy * (tgt[:, 1] - anc[:, 1]) / anc[:, 3].clamp(min=1e-6)
    dw = ww * torch.log((tgt[:, 2] / anc[:, 2].clamp(min=1e-6)).clamp(min=1e-6))
    dh = wh * torch.log((tgt[:, 3] / anc[:, 3].clamp(min=1e-6)).clamp(min=1e-6))
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(
    anchors: Tensor,
    deltas: Tensor,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> Tensor:
    """Apply predicted deltas to anchors. Inverse of :func:`encode_boxes`."""
    anc = boxes_to_centers(anchors)
    wx, wy, ww, wh = weights

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh

    # Clamp dw, dh so exp does not explode on an untrained network.
    dw = dw.clamp(max=4.0)
    dh = dh.clamp(max=4.0)

    pred_cx = dx * anc[:, 2] + anc[:, 0]
    pred_cy = dy * anc[:, 3] + anc[:, 1]
    pred_w = torch.exp(dw) * anc[:, 2]
    pred_h = torch.exp(dh) * anc[:, 3]

    x1 = pred_cx - pred_w * 0.5
    y1 = pred_cy - pred_h * 0.5
    x2 = pred_cx + pred_w * 0.5
    y2 = pred_cy + pred_h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
    """Greedy non maximum suppression.

    Args:
        boxes: Tensor of shape ``(N, 4)``.
        scores: Tensor of shape ``(N,)``.
        iou_threshold: Boxes overlapping a kept box by more than this are
            suppressed.

    Returns:
        Long tensor of kept indices, ordered by descending score.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i : i + 1], boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def detections_from_outputs(
    cls_logits: Tensor,
    bbox_deltas: Tensor,
    anchors: Tensor,
    image_size: tuple[int, int],
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    max_detections: int = 300,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> dict[str, Tensor]:
    """Turn raw single image outputs into final detections.

    Args:
        cls_logits: Tensor of shape ``(L, num_classes)``.
        bbox_deltas: Tensor of shape ``(L, 4)``.
        anchors: Tensor of shape ``(L, 4)``.
        image_size: ``(height, width)`` used to clip boxes.
        score_threshold: Minimum objectness probability to keep.
        nms_threshold: IoU threshold for NMS.
        max_detections: Cap on detections after NMS.
        weights: Delta weights, must match training.

    Returns:
        Dict with ``boxes`` ``(K, 4)``, ``scores`` ``(K,)`` and ``labels``
        ``(K,)``.
    """
    height, width = image_size
    # Single foreground class assumed for objectness style detection; take the
    # max over classes as the score.
    scores_all = torch.sigmoid(cls_logits)
    scores, labels = scores_all.max(dim=1)
    # Class index 0 in a single class head means the one foreground class; we
    # report labels starting at 1 so 0 stays free for background downstream.
    labels = labels + 1

    decoded = decode_boxes(anchors, bbox_deltas, weights=weights)
    decoded = clip_boxes(decoded, height, width)

    keep_mask = scores >= score_threshold
    decoded = decoded[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]

    if decoded.numel() == 0:
        return {
            "boxes": decoded.new_zeros((0, 4)),
            "scores": scores.new_zeros((0,)),
            "labels": labels.new_zeros((0,), dtype=torch.long),
        }

    keep = nms(decoded, scores, iou_threshold=nms_threshold)
    keep = keep[:max_detections]

    return {
        "boxes": decoded[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }
