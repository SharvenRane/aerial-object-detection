import math

import torch

from src.evaluate import (
    average_precision,
    match_detections,
    precision,
    recall,
    small_object_recall,
)


def test_perfect_match_recall_and_precision():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    preds = gt.clone()
    scores = torch.tensor([0.9, 0.8])
    m = match_detections(preds, scores, gt, iou_threshold=0.5)
    assert recall(m) == 1.0
    assert precision(m) == 1.0
    assert m.true_positives.sum().item() == 2


def test_each_gt_matched_once():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    # Two predictions on the same object: only one is a true positive.
    preds = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 9.0, 9.0]])
    scores = torch.tensor([0.9, 0.8])
    m = match_detections(preds, scores, gt)
    assert m.true_positives.sum().item() == 1
    assert recall(m) == 1.0
    assert precision(m) == 0.5


def test_no_predictions_zero_recall():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    preds = torch.zeros((0, 4))
    scores = torch.zeros((0,))
    m = match_detections(preds, scores, gt)
    assert recall(m) == 0.0
    assert m.num_gt == 1


def test_low_iou_is_not_a_match():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    preds = torch.tensor([[8.0, 8.0, 18.0, 18.0]])  # small overlap
    scores = torch.tensor([0.9])
    m = match_detections(preds, scores, gt, iou_threshold=0.5)
    assert recall(m) == 0.0
    assert m.true_positives.sum().item() == 0


def test_higher_score_prediction_claims_gt_first():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    # The lower scoring prediction is the exact match; the higher scoring one
    # only partially overlaps. Greedy by score should still match the gt once.
    preds = torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 2.0, 12.0, 12.0]])
    scores = torch.tensor([0.4, 0.9])
    m = match_detections(preds, scores, gt, iou_threshold=0.5)
    assert recall(m) == 1.0


def test_small_object_recall_only_counts_small():
    # One small object (area 16) and one large (area 2500).
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 60.0, 60.0]])
    # Predict only the large one well.
    preds = torch.tensor([[10.0, 10.0, 60.0, 60.0]])
    scores = torch.tensor([0.9])
    r = small_object_recall(preds, scores, gt, max_area=100.0)
    # The small object was missed, so small object recall is 0.
    assert r == 0.0


def test_small_object_recall_detects_small():
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 60.0, 60.0]])
    preds = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    scores = torch.tensor([0.9])
    r = small_object_recall(preds, scores, gt, max_area=100.0)
    assert r == 1.0


def test_small_object_recall_nan_when_no_small():
    gt = torch.tensor([[10.0, 10.0, 60.0, 60.0]])
    preds = gt.clone()
    scores = torch.tensor([0.9])
    r = small_object_recall(preds, scores, gt, max_area=50.0)
    assert math.isnan(r)


def test_average_precision_perfect_is_one():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    preds = gt.clone()
    scores = torch.tensor([0.9, 0.8])
    m = match_detections(preds, scores, gt)
    ap = average_precision([m])
    assert abs(ap - 1.0) < 1e-6


def test_average_precision_in_unit_range():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    preds = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # tp
            [40.0, 40.0, 50.0, 50.0],  # fp
            [20.0, 20.0, 30.0, 30.0],  # tp
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    m = match_detections(preds, scores, gt)
    ap = average_precision([m])
    assert 0.0 <= ap <= 1.0
    assert ap < 1.0  # a false positive ranked above a true positive hurts AP
