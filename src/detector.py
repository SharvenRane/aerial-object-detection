"""High level detector pipeline and a small trainer.

This module wires the model, anchors, target assignment, loss and inference
into a single object. The training loop is intentionally lightweight so it can
fit a handful of synthetic scenes inside a unit test, yet it is a real anchor
assignment and a real focal style classification plus smooth L1 regression
loss, not a stand in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .anchors import generate_anchors
from .boxes import box_iou
from .model import AerialDetector
from .postprocess import DEFAULT_WEIGHTS, detections_from_outputs, encode_boxes


def assign_targets(
    anchors: Tensor,
    gt_boxes: Tensor,
    positive_iou: float = 0.5,
    negative_iou: float = 0.3,
) -> tuple[Tensor, Tensor]:
    """Assign each anchor to a ground truth box or to background.

    Args:
        anchors: Tensor of shape ``(L, 4)``.
        gt_boxes: Tensor of shape ``(G, 4)``.
        positive_iou: Anchors with best IoU at or above this are positive.
        negative_iou: Anchors with best IoU below this are negative. Anchors in
            between are ignored (label ``-1``).

    Returns:
        Tuple ``(labels, matched_gt_idx)``. ``labels`` is a long tensor of
        shape ``(L,)`` with values ``1`` (object), ``0`` (background) or ``-1``
        (ignore). ``matched_gt_idx`` is a long tensor of shape ``(L,)`` giving
        the index of the assigned ground truth for positive anchors (and the
        argmax ground truth otherwise, which is unused).
    """
    num_anchors = anchors.shape[0]
    if gt_boxes.shape[0] == 0:
        labels = torch.zeros((num_anchors,), dtype=torch.long)
        matched = torch.zeros((num_anchors,), dtype=torch.long)
        return labels, matched

    ious = box_iou(anchors, gt_boxes)  # (L, G)
    best_iou, best_gt = ious.max(dim=1)

    labels = torch.full((num_anchors,), -1, dtype=torch.long)
    labels[best_iou < negative_iou] = 0
    labels[best_iou >= positive_iou] = 1

    # Guarantee every ground truth has at least one positive anchor: the anchor
    # with the highest IoU for that ground truth.
    best_anchor_per_gt = ious.argmax(dim=0)  # (G,)
    labels[best_anchor_per_gt] = 1
    # Align matched index for those forced positives too.
    forced_gt = torch.arange(gt_boxes.shape[0])
    best_gt[best_anchor_per_gt] = forced_gt

    return labels, best_gt


def detection_loss(
    cls_logits: Tensor,
    bbox_deltas: Tensor,
    anchors: Tensor,
    gt_boxes: Tensor,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> Tensor:
    """Single image detection loss.

    Classification uses binary cross entropy over positives and negatives.
    Regression uses smooth L1 over positive anchors only.

    Args:
        cls_logits: Tensor of shape ``(L, 1)``.
        bbox_deltas: Tensor of shape ``(L, 4)``.
        anchors: Tensor of shape ``(L, 4)``.
        gt_boxes: Tensor of shape ``(G, 4)``.
        weights: Delta weights used for the regression target encoding.

    Returns:
        Scalar loss tensor.
    """
    labels, matched_gt = assign_targets(anchors, gt_boxes)

    valid = labels >= 0
    cls_target = labels.clamp(min=0).float()[valid].unsqueeze(1)
    cls_pred = cls_logits[valid]
    cls_loss = F.binary_cross_entropy_with_logits(
        cls_pred, cls_target, reduction="mean"
    )

    pos = labels == 1
    if pos.any() and gt_boxes.shape[0] > 0:
        pos_anchors = anchors[pos]
        target_boxes = gt_boxes[matched_gt[pos]]
        reg_target = encode_boxes(pos_anchors, target_boxes, weights=weights)
        reg_pred = bbox_deltas[pos]
        reg_loss = F.smooth_l1_loss(reg_pred, reg_target, reduction="mean")
    else:
        reg_loss = cls_logits.new_zeros(())

    return cls_loss + reg_loss


class Detector:
    """Convenience wrapper around the model and its anchor grid."""

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        sizes: tuple[int, ...] = (6, 10, 16),
        aspect_ratios: tuple[float, ...] = (1.0,),
        width: int = 32,
    ):
        self.image_size = image_size
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.model = AerialDetector(
            width=width, sizes=sizes, aspect_ratios=aspect_ratios
        )
        self.anchors = generate_anchors(
            image_size[0],
            image_size[1],
            stride=self.model.stride,
            sizes=sizes,
            aspect_ratios=aspect_ratios,
        )

    def to(self, device: torch.device | str) -> "Detector":
        self.model.to(device)
        self.anchors = self.anchors.to(device)
        return self

    def train_on(
        self,
        scenes,
        epochs: int = 30,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> list[float]:
        """Fit the model on a list of scenes. Returns per epoch loss."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        history: list[float] = []

        images = torch.stack([s.image for s in scenes])
        gts = [s.boxes for s in scenes]

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(images)
            batch_loss = images.new_zeros(())
            for b in range(images.shape[0]):
                batch_loss = batch_loss + detection_loss(
                    outputs["cls_logits"][b],
                    outputs["bbox_deltas"][b],
                    self.anchors,
                    gts[b],
                )
            batch_loss = batch_loss / images.shape[0]
            batch_loss.backward()
            optimizer.step()
            history.append(float(batch_loss.item()))
            if verbose:
                print(f"epoch {epoch}: loss {batch_loss.item():.4f}")
        return history

    @torch.no_grad()
    def predict(
        self,
        image: Tensor,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> dict[str, Tensor]:
        """Run inference on a single image tensor ``(C, H, W)``."""
        self.model.eval()
        outputs = self.model(image.unsqueeze(0))
        return detections_from_outputs(
            outputs["cls_logits"][0],
            outputs["bbox_deltas"][0],
            self.anchors,
            image_size=self.image_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
