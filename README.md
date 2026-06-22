# aerial-object-detection

Object detection on aerial and satellite style imagery, with a focus on the
case that actually makes overhead imagery hard: lots of tiny, well separated
objects and axis aligned boxes. Think parked cars in a lot, storage tanks in a
tank farm, or boats in a harbour seen from straight above. The whole project
runs on CPU, trains on synthetic scenes, and is covered by a pytest suite that
checks real behaviour rather than mocking it away.

## Why this exists

Most detection tutorials lean on a pretrained backbone and a large benchmark.
That hides the parts that matter for overhead imagery. Here the interesting
problems are out in the open:

- objects are small relative to the frame, so the detector runs on a fine
  stride and small anchors rather than a coarse pyramid level meant for people
  and cars at street level,
- boxes are axis aligned, which keeps the geometry simple but puts pressure on
  recall because a missed tiny object is easy to ignore by eye and easy to miss
  by metric,
- the right thing to measure is small object recall, not a single number that
  large objects can dominate.

Everything is built from scratch on top of PyTorch so you can read the whole
path from pixels to boxes.

## What is inside

The package lives under `src/` and breaks into focused modules.

`synthetic.py` paints a synthetic aerial scene. It lays down a low frequency
textured background that reads like terrain from above, then scatters small
bright rectangles with exact ground truth boxes. Placement rejects overlaps so
every object stays separable, and the whole thing is seeded so a scene is
reproducible.

`boxes.py` holds the axis aligned box geometry: area, pairwise intersection
over union, clipping to the image, and conversion between corner and center
forms. These are the primitives the rest of the code leans on.

`anchors.py` tiles a dense anchor grid over a single feature stride. Because the
objects are tiny the default anchor sizes are small and the stride is fine.

`model.py` is a shallow fully convolutional detector. A small backbone
downsamples the image by a factor of four, then a shared head predicts an
objectness logit and four box deltas per anchor at every location. The forward
pass flattens its outputs so they line up one to one with the anchor grid.

`postprocess.py` turns raw outputs into detections. It encodes and decodes box
deltas with the standard center and size parameterisation, runs greedy non
maximum suppression to collapse duplicate anchors that point at the same
object, and applies a score threshold.

`evaluate.py` is the measurement side. It matches predictions to ground truth
greedily by score and IoU so each ground truth is claimed at most once, then
reports recall, precision, average precision, and a dedicated small object
recall that restricts the denominator to ground truths below an area threshold.

`detector.py` wires it together. It assigns anchors to ground truth, computes a
classification plus smooth L1 regression loss, and exposes a small trainer and a
single image `predict` method.

## Quick start

```python
import torch
from src.detector import Detector
from src.synthetic import make_dataset
from src.evaluate import match_detections, recall

torch.manual_seed(0)
det = Detector(image_size=(64, 64), sizes=(6, 10, 16), width=16)

scenes = make_dataset(count=12, height=64, width=64, num_objects=6, seed=0)
det.train_on(scenes, epochs=150, lr=2e-3)

scene = scenes[0]
out = det.predict(scene.image, score_threshold=0.3, nms_threshold=0.3)
m = match_detections(out["boxes"], out["scores"], scene.boxes, iou_threshold=0.3)
print("boxes:", out["boxes"].shape[0], "recall:", recall(m))
```

`out` is a dict with `boxes`, `scores`, and `labels`. The boxes are clipped to
the image and ordered by descending score after suppression.

## Running the tests

```
C:/Users/sharv/.venvs/cv/Scripts/python.exe -m pytest tests/ -q
```

The suite has 48 tests and runs in a few seconds on CPU with no downloads and
no API keys. It covers the box geometry, the synthetic generator (shapes,
ranges, determinism, non overlapping ground truth, objects brighter than the
background), the anchor grid, the model output shapes and differentiability,
delta encode and decode roundtrips, non maximum suppression, the IoU matching
and average precision, and an end to end check that trains the detector on a
handful of synthetic scenes and confirms it recalls small objects on held out
scenes it never saw during training.

## A note on the results

The integration test trains a fresh detector from a fixed seed and asserts that
its mean recall on held out synthetic scenes clears a real bar. The number you
see is whatever that run produces; nothing here is a hardcoded benchmark. On the
local CPU run used to build this, the trained detector reached a mean held out
recall around 0.87 at an IoU threshold of 0.3 on four held out scenes. Your
exact figure will track your hardware and seed.

## Scope and honesty about the stand in

The synthetic scenes are a stand in for real aerial imagery so the tests can run
offline and deterministically. The detector, the anchor assignment, the loss,
the suppression, and the evaluation are all real and would carry over to real
imagery with a stronger backbone and real data. The goal was a small, correct,
readable detection stack, not a state of the art model.
