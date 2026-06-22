import torch

from src.anchors import generate_anchors, num_anchors_per_location
from src.boxes import box_area
from src.model import AerialDetector


def test_anchor_count_matches_grid():
    h, w, stride = 64, 64, 4
    sizes = (6, 10, 16)
    anchors = generate_anchors(h, w, stride, sizes=sizes)
    expected = (h // stride) * (w // stride) * len(sizes)
    assert anchors.shape == (expected, 4)


def test_anchors_are_valid_boxes():
    anchors = generate_anchors(64, 48, stride=4)
    assert torch.all(anchors[:, 2] >= anchors[:, 0])
    assert torch.all(anchors[:, 3] >= anchors[:, 1])
    assert torch.all(box_area(anchors) > 0)


def test_anchor_sizes_reflect_request():
    # A single square anchor size should produce anchors of that area.
    anchors = generate_anchors(32, 32, stride=4, sizes=(8,), aspect_ratios=(1.0,))
    areas = box_area(anchors)
    assert torch.allclose(areas, torch.full_like(areas, 64.0), atol=1e-4)


def test_num_anchors_per_location():
    assert num_anchors_per_location((6, 10, 16), (1.0,)) == 3
    assert num_anchors_per_location((6, 10), (0.5, 1.0, 2.0)) == 6


def test_model_output_shapes_align_with_anchors():
    h, w = 64, 64
    sizes = (6, 10, 16)
    model = AerialDetector(width=16, sizes=sizes)
    anchors = generate_anchors(h, w, stride=model.stride, sizes=sizes)
    images = torch.rand(2, 3, h, w)
    out = model(images)
    assert out["cls_logits"].shape == (2, anchors.shape[0], 1)
    assert out["bbox_deltas"].shape == (2, anchors.shape[0], 4)


def test_model_rejects_unbatched_input():
    model = AerialDetector(width=8)
    try:
        model(torch.rand(3, 64, 64))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unbatched input")


def test_model_is_differentiable():
    model = AerialDetector(width=8)
    images = torch.rand(1, 3, 32, 32)
    out = model(images)
    loss = out["cls_logits"].sum() + out["bbox_deltas"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)
