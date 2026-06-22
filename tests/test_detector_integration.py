import torch

from src.detector import Detector, assign_targets, detection_loss
from src.anchors import generate_anchors
from src.evaluate import match_detections, recall, small_object_recall
from src.synthetic import make_dataset, make_scene


def test_assign_targets_marks_positive_and_negative():
    anchors = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # overlaps the gt strongly
            [100.0, 100.0, 110.0, 110.0],  # far away, background
        ]
    )
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    labels, matched = assign_targets(anchors, gt)
    assert labels[0].item() == 1
    assert labels[1].item() == 0
    assert matched[0].item() == 0


def test_assign_targets_no_gt_is_all_background():
    anchors = generate_anchors(32, 32, stride=4, sizes=(8,))
    labels, _ = assign_targets(anchors, torch.zeros((0, 4)))
    assert torch.all(labels == 0)


def test_every_gt_gets_a_positive_anchor():
    anchors = generate_anchors(64, 64, stride=4, sizes=(6, 10, 16))
    scene = make_scene(height=64, width=64, num_objects=5, seed=11)
    if scene.boxes.shape[0] > 0:
        labels, matched = assign_targets(anchors, scene.boxes)
        positive_gts = set(matched[labels == 1].tolist())
        assert positive_gts == set(range(scene.boxes.shape[0]))


def test_detection_loss_is_finite_and_positive():
    anchors = generate_anchors(64, 64, stride=4, sizes=(6, 10, 16))
    scene = make_scene(height=64, width=64, num_objects=4, seed=1)
    n = anchors.shape[0]
    cls = torch.zeros((n, 1), requires_grad=True)
    reg = torch.zeros((n, 4), requires_grad=True)
    loss = detection_loss(cls, reg, anchors, scene.boxes)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    assert cls.grad is not None


def test_predict_returns_well_formed_detections():
    det = Detector(image_size=(64, 64), sizes=(6, 10, 16), width=16)
    scene = make_scene(height=64, width=64, num_objects=4, seed=2)
    out = det.predict(scene.image, score_threshold=0.0)
    assert set(out.keys()) == {"boxes", "scores", "labels"}
    assert out["boxes"].shape[1] == 4
    assert out["boxes"].shape[0] == out["scores"].shape[0]
    # Boxes lie inside the image.
    assert torch.all(out["boxes"][:, 0::2] >= 0)
    assert torch.all(out["boxes"][:, 0::2] <= 64)


def test_training_reduces_loss():
    torch.manual_seed(0)
    det = Detector(image_size=(64, 64), sizes=(6, 10, 16), width=16)
    scenes = make_dataset(count=4, height=64, width=64, num_objects=5, seed=0)
    history = det.train_on(scenes, epochs=25, lr=2e-3)
    assert history[-1] < history[0]


def test_trained_detector_recalls_small_objects():
    torch.manual_seed(0)
    det = Detector(image_size=(64, 64), sizes=(6, 10, 16), width=16)
    train_scenes = make_dataset(
        count=12, height=64, width=64, num_objects=6, seed=0
    )
    det.train_on(train_scenes, epochs=150, lr=2e-3)

    # Evaluate on held out scenes, pooling recall across them.
    eval_scenes = make_dataset(
        count=4, height=64, width=64, num_objects=6, seed=500
    )
    total_recall = 0.0
    counted = 0
    for scene in eval_scenes:
        out = det.predict(scene.image, score_threshold=0.3, nms_threshold=0.3)
        m = match_detections(
            out["boxes"], out["scores"], scene.boxes, iou_threshold=0.3
        )
        if m.num_gt > 0:
            total_recall += recall(m)
            counted += 1
    mean_recall = total_recall / max(counted, 1)
    # The detector should find a real fraction of the small objects, well
    # above what an untrained network would manage.
    assert mean_recall > 0.4


def test_small_object_recall_helper_runs_on_predictions():
    torch.manual_seed(0)
    det = Detector(image_size=(64, 64), sizes=(6, 10, 16), width=16)
    scenes = make_dataset(count=6, height=64, width=64, num_objects=6, seed=0)
    det.train_on(scenes, epochs=60, lr=2e-3)
    scene = make_scene(height=64, width=64, num_objects=6, seed=900)
    out = det.predict(scene.image, score_threshold=0.3, nms_threshold=0.3)
    r = small_object_recall(
        out["boxes"], out["scores"], scene.boxes, max_area=200.0, iou_threshold=0.3
    )
    # All synthetic objects are small, so this should be a real number in [0,1].
    assert 0.0 <= r <= 1.0
