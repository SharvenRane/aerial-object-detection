"""Detection evaluation: IoU matching, recall, precision, average precision.

Matching follows the usual greedy protocol. Predictions are sorted by score.
Each prediction claims the highest IoU unmatched ground truth above the IoU
threshold. A ground truth can only be claimed once. Unmatched predictions are
false positives and unmatched ground truths are false negatives.

The small object recall helper restricts the evaluation to ground truths whose
area falls below a pixel threshold, which is the metric that matters most for
aerial and satellite imagery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .boxes import box_area, box_iou


@dataclass
class MatchResult:
    """Outcome of matching predictions to ground truth for one image.

    Attributes:
        true_positives: Boolean tensor over predictions (score sorted).
        scores: Prediction scores in the same order, descending.
        gt_matched: Boolean tensor over ground truths, True if matched.
        num_gt: Number of ground truth boxes.
    """

    true_positives: Tensor
    scores: Tensor
    gt_matched: Tensor
    num_gt: int


def match_detections(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    gt_boxes: Tensor,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedily match predictions to ground truth boxes by IoU.

    Args:
        pred_boxes: Tensor of shape ``(P, 4)``.
        pred_scores: Tensor of shape ``(P,)``.
        gt_boxes: Tensor of shape ``(G, 4)``.
        iou_threshold: Minimum IoU for a match.

    Returns:
        A :class:`MatchResult`.
    """
    num_pred = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]

    order = torch.argsort(pred_scores, descending=True)
    sorted_scores = pred_scores[order]

    tp = torch.zeros((num_pred,), dtype=torch.bool)
    gt_matched = torch.zeros((num_gt,), dtype=torch.bool)

    if num_pred == 0 or num_gt == 0:
        return MatchResult(
            true_positives=tp,
            scores=sorted_scores,
            gt_matched=gt_matched,
            num_gt=num_gt,
        )

    ious = box_iou(pred_boxes[order], gt_boxes)  # (P, G)

    for rank in range(num_pred):
        row = ious[rank].clone()
        # Block already matched ground truths.
        row[gt_matched] = -1.0
        best_iou, best_gt = row.max(dim=0)
        if best_iou.item() >= iou_threshold:
            tp[rank] = True
            gt_matched[best_gt] = True

    return MatchResult(
        true_positives=tp,
        scores=sorted_scores,
        gt_matched=gt_matched,
        num_gt=num_gt,
    )


def recall(match: MatchResult) -> float:
    """Fraction of ground truths that were matched."""
    if match.num_gt == 0:
        return float("nan")
    return float(match.gt_matched.sum().item()) / match.num_gt


def precision(match: MatchResult) -> float:
    """Fraction of predictions that were true positives."""
    num_pred = match.true_positives.numel()
    if num_pred == 0:
        return float("nan")
    return float(match.true_positives.sum().item()) / num_pred


def small_object_recall(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    gt_boxes: Tensor,
    max_area: float,
    iou_threshold: float = 0.5,
) -> float:
    """Recall restricted to ground truth boxes smaller than ``max_area``.

    The full prediction set is still used for matching (a small object can be
    found by any prediction). Only the recall denominator and the matched
    count are restricted to the small ground truths.

    Args:
        pred_boxes: Tensor of shape ``(P, 4)``.
        pred_scores: Tensor of shape ``(P,)``.
        gt_boxes: Tensor of shape ``(G, 4)``.
        max_area: Area threshold in pixels squared.
        iou_threshold: Minimum IoU for a match.

    Returns:
        Recall over small ground truths, or ``nan`` if there are none.
    """
    if gt_boxes.shape[0] == 0:
        return float("nan")

    areas = box_area(gt_boxes)
    small_mask = areas < max_area
    num_small = int(small_mask.sum().item())
    if num_small == 0:
        return float("nan")

    match = match_detections(
        pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold
    )
    matched_small = match.gt_matched & small_mask
    return float(matched_small.sum().item()) / num_small


def average_precision(
    matches: list[MatchResult],
) -> float:
    """Compute average precision over a set of matched images.

    Uses the all points interpolation of the precision recall curve. All
    predictions across images are pooled and ranked by score.

    Args:
        matches: One :class:`MatchResult` per image.

    Returns:
        Average precision in ``[0, 1]``, or ``nan`` if there are no ground
        truths anywhere.
    """
    total_gt = sum(m.num_gt for m in matches)
    if total_gt == 0:
        return float("nan")

    all_scores = torch.cat([m.scores for m in matches]) if matches else torch.zeros(0)
    all_tp = (
        torch.cat([m.true_positives for m in matches]) if matches else torch.zeros(0)
    )
    if all_scores.numel() == 0:
        return 0.0

    order = torch.argsort(all_scores, descending=True)
    tp = all_tp[order].float()
    fp = 1.0 - tp

    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)

    recalls = cum_tp / total_gt
    precisions = cum_tp / (cum_tp + cum_fp).clamp(min=1e-9)

    # Prepend the (recall=0, precision=1) anchor point.
    recalls = torch.cat([torch.zeros(1), recalls])
    precisions = torch.cat([torch.ones(1), precisions])

    # Make precision monotonically decreasing from the right.
    for i in range(precisions.numel() - 1, 0, -1):
        precisions[i - 1] = torch.max(precisions[i - 1], precisions[i])

    # Integrate precision over the recall steps.
    ap = torch.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return float(ap.item())
