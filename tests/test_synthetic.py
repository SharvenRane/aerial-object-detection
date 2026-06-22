import torch

from src.boxes import box_iou
from src.synthetic import make_dataset, make_scene


def test_scene_shapes_and_ranges():
    scene = make_scene(height=96, width=128, num_objects=6, seed=1)
    assert scene.image.shape == (3, 96, 128)
    assert scene.image.min().item() >= 0.0
    assert scene.image.max().item() <= 1.0
    assert scene.boxes.shape[1] == 4
    assert scene.boxes.shape[0] == scene.labels.shape[0]


def test_scene_is_deterministic():
    a = make_scene(seed=7)
    b = make_scene(seed=7)
    assert torch.allclose(a.image, b.image)
    assert torch.equal(a.boxes, b.boxes)


def test_boxes_inside_image():
    scene = make_scene(height=100, width=100, num_objects=10, seed=3)
    assert torch.all(scene.boxes[:, 0] >= 0)
    assert torch.all(scene.boxes[:, 1] >= 0)
    assert torch.all(scene.boxes[:, 2] <= 100)
    assert torch.all(scene.boxes[:, 3] <= 100)


def test_boxes_have_positive_area():
    scene = make_scene(num_objects=12, seed=5)
    w = scene.boxes[:, 2] - scene.boxes[:, 0]
    h = scene.boxes[:, 3] - scene.boxes[:, 1]
    assert torch.all(w > 0) and torch.all(h > 0)


def test_objects_are_small_relative_to_scene():
    scene = make_scene(height=128, width=128, num_objects=8, max_size=12, seed=2)
    w = scene.boxes[:, 2] - scene.boxes[:, 0]
    h = scene.boxes[:, 3] - scene.boxes[:, 1]
    # Every object covers far less than the whole frame.
    assert torch.all(w <= 12) and torch.all(h <= 12)


def test_ground_truth_boxes_do_not_overlap():
    scene = make_scene(height=128, width=128, num_objects=15, seed=9)
    if scene.boxes.shape[0] >= 2:
        iou = box_iou(scene.boxes, scene.boxes)
        # Zero out the diagonal then assert no pair overlaps.
        n = iou.shape[0]
        iou[torch.arange(n), torch.arange(n)] = 0.0
        assert iou.max().item() == 0.0


def test_object_pixels_are_brighter_than_background():
    scene = make_scene(height=128, width=128, num_objects=5, seed=4)
    if scene.boxes.shape[0] > 0:
        b = scene.boxes[0].long()
        patch = scene.image[:, b[1] : b[3], b[0] : b[2]].mean()
        assert patch.item() > 0.7


def test_make_dataset_size():
    scenes = make_dataset(count=5, seed=0)
    assert len(scenes) == 5
    # Different seeds give different scenes.
    assert not torch.allclose(scenes[0].image, scenes[1].image)
