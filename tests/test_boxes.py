import torch

from src.boxes import (
    box_area,
    box_iou,
    boxes_to_centers,
    centers_to_boxes,
    clip_boxes,
)


def test_box_area_matches_manual():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 1.0, 5.0]])
    areas = box_area(boxes)
    assert torch.allclose(areas, torch.tensor([6.0, 0.0]))


def test_box_iou_identical_is_one():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou = box_iou(boxes, boxes)
    assert torch.allclose(iou, torch.ones(1, 1))


def test_box_iou_disjoint_is_zero():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[5.0, 5.0, 6.0, 6.0]])
    iou = box_iou(a, b)
    assert iou.item() == 0.0


def test_box_iou_half_overlap():
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])  # area 4
    b = torch.tensor([[1.0, 0.0, 3.0, 2.0]])  # area 4, overlap area 2
    iou = box_iou(a, b).item()
    # union = 4 + 4 - 2 = 6, iou = 2 / 6
    assert abs(iou - (2.0 / 6.0)) < 1e-6


def test_box_iou_in_unit_range():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(7, 2, generator=g)
    a = torch.cat([a, a + torch.rand(7, 2, generator=g)], dim=1)
    b = torch.rand(5, 2, generator=g)
    b = torch.cat([b, b + torch.rand(5, 2, generator=g)], dim=1)
    iou = box_iou(a, b)
    assert iou.shape == (7, 5)
    assert torch.all(iou >= 0) and torch.all(iou <= 1.0 + 1e-6)


def test_empty_inputs():
    empty = torch.zeros((0, 4))
    full = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert box_area(empty).shape == (0,)
    assert box_iou(empty, full).shape == (0, 1)
    assert box_iou(full, empty).shape == (1, 0)


def test_center_roundtrip():
    boxes = torch.tensor([[1.0, 2.0, 5.0, 8.0], [0.0, 0.0, 10.0, 4.0]])
    back = centers_to_boxes(boxes_to_centers(boxes))
    assert torch.allclose(back, boxes)


def test_clip_boxes_stays_in_bounds():
    boxes = torch.tensor([[-5.0, -5.0, 200.0, 200.0]])
    clipped = clip_boxes(boxes, height=100, width=80)
    assert torch.allclose(clipped, torch.tensor([[0.0, 0.0, 80.0, 100.0]]))
