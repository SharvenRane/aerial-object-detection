import torch

from src.anchors import generate_anchors
from src.postprocess import (
    decode_boxes,
    detections_from_outputs,
    encode_boxes,
    nms,
)


def test_encode_decode_roundtrip():
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 25.0, 15.0]])
    targets = torch.tensor([[1.0, 2.0, 9.0, 11.0], [6.0, 4.0, 20.0, 18.0]])
    deltas = encode_boxes(anchors, targets)
    recovered = decode_boxes(anchors, deltas)
    assert torch.allclose(recovered, targets, atol=1e-4)


def test_nms_removes_duplicates():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 10.5, 10.5],  # near duplicate of box 0
            [50.0, 50.0, 60.0, 60.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_nms_keeps_distinct_boxes():
    boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [20.0, 20.0, 25.0, 25.0]])
    scores = torch.tensor([0.6, 0.5])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_nms_empty():
    keep = nms(torch.zeros((0, 4)), torch.zeros((0,)))
    assert keep.numel() == 0


def test_detections_threshold_filters_low_scores():
    anchors = generate_anchors(32, 32, stride=4, sizes=(8,))
    n = anchors.shape[0]
    cls_logits = torch.full((n, 1), -5.0)  # sigmoid ~ 0.0067
    deltas = torch.zeros((n, 4))
    out = detections_from_outputs(
        cls_logits, deltas, anchors, image_size=(32, 32), score_threshold=0.5
    )
    assert out["boxes"].shape[0] == 0


def test_detections_returns_boxes_for_confident_anchor():
    anchors = generate_anchors(32, 32, stride=4, sizes=(8,))
    n = anchors.shape[0]
    cls_logits = torch.full((n, 1), -5.0)
    cls_logits[10] = 5.0  # one confident anchor
    deltas = torch.zeros((n, 4))
    out = detections_from_outputs(
        cls_logits, deltas, anchors, image_size=(32, 32), score_threshold=0.5
    )
    assert out["boxes"].shape[0] >= 1
    assert torch.all(out["scores"] >= 0.5)
    # Box should match the anchor it came from since deltas are zero.
    assert torch.allclose(out["boxes"][0], anchors[10], atol=1e-4)


def test_detection_boxes_are_clipped():
    anchors = generate_anchors(32, 32, stride=4, sizes=(8,))
    n = anchors.shape[0]
    cls_logits = torch.full((n, 1), 5.0)
    deltas = torch.zeros((n, 4))
    out = detections_from_outputs(
        cls_logits, deltas, anchors, image_size=(32, 32), score_threshold=0.5
    )
    assert torch.all(out["boxes"][:, 0::2] >= 0)
    assert torch.all(out["boxes"][:, 0::2] <= 32)
    assert torch.all(out["boxes"][:, 1::2] >= 0)
    assert torch.all(out["boxes"][:, 1::2] <= 32)
