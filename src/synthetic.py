"""Synthetic aerial and satellite scene generator.

The generator paints a textured background that resembles terrain seen from
above, then scatters small bright objects (think parked cars, storage tanks,
or boats) onto it. Each object is a filled rectangle with a known box, so the
ground truth is exact. Objects are deliberately small relative to the scene,
which is the regime that makes aerial detection hard.

Everything is deterministic given a seed so tests are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Scene:
    """A single synthetic scene.

    Attributes:
        image: Float tensor of shape ``(C, H, W)`` in ``[0, 1]``.
        boxes: Tensor of shape ``(N, 4)`` ground truth boxes.
        labels: Long tensor of shape ``(N,)`` class indices (all ``1`` here,
            ``0`` is reserved for background).
    """

    image: Tensor
    boxes: Tensor
    labels: Tensor


def _make_background(height: int, width: int, generator: torch.Generator) -> Tensor:
    """Build a low contrast textured background in ``[0, 1]``."""
    # Start from low frequency noise so the terrain has large smooth patches,
    # then add a little high frequency grain.
    coarse = torch.rand((1, height // 8 + 1, width // 8 + 1), generator=generator)
    coarse = torch.nn.functional.interpolate(
        coarse[None], size=(height, width), mode="bilinear", align_corners=False
    )[0]
    grain = torch.rand((1, height, width), generator=generator) * 0.1
    background = 0.25 + 0.4 * coarse + grain
    background = background.clamp(0, 1)
    # Three channels with a slight per channel tint so it looks like RGB.
    tint = torch.tensor([1.0, 0.95, 0.85]).view(3, 1, 1)
    return (background * tint).clamp(0, 1)


def make_scene(
    height: int = 128,
    width: int = 128,
    num_objects: int = 8,
    min_size: int = 4,
    max_size: int = 12,
    seed: int | None = None,
) -> Scene:
    """Generate one synthetic aerial scene with ground truth boxes.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        num_objects: Number of objects to attempt to place. Overlapping
            placements are rejected, so the final count may be smaller.
        min_size: Minimum object side length in pixels.
        max_size: Maximum object side length in pixels.
        seed: Optional seed for reproducibility.

    Returns:
        A :class:`Scene`.
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    image = _make_background(height, width, generator)

    placed_boxes: list[list[float]] = []

    def overlaps(candidate: list[float]) -> bool:
        cx1, cy1, cx2, cy2 = candidate
        for bx1, by1, bx2, by2 in placed_boxes:
            ix1 = max(cx1, bx1)
            iy1 = max(cy1, by1)
            ix2 = min(cx2, bx2)
            iy2 = min(cy2, by2)
            if ix2 > ix1 and iy2 > iy1:
                return True
        return False

    attempts = 0
    max_attempts = num_objects * 30
    while len(placed_boxes) < num_objects and attempts < max_attempts:
        attempts += 1
        w = int(torch.randint(min_size, max_size + 1, (1,), generator=generator).item())
        h = int(torch.randint(min_size, max_size + 1, (1,), generator=generator).item())
        if w >= width or h >= height:
            continue
        x1 = int(torch.randint(0, width - w, (1,), generator=generator).item())
        y1 = int(torch.randint(0, height - h, (1,), generator=generator).item())
        x2, y2 = x1 + w, y1 + h
        candidate = [float(x1), float(y1), float(x2), float(y2)]
        # Leave a one pixel margin so neighbouring objects stay separable.
        margin = [x1 - 1, y1 - 1, x2 + 1, y2 + 1]
        if overlaps(margin):
            continue

        # Bright object with a small per object colour so the detector has a
        # real signal to latch onto rather than a single fixed value.
        base = 0.8 + 0.2 * torch.rand((3,), generator=generator)
        for c in range(3):
            image[c, y1:y2, x1:x2] = base[c]
        placed_boxes.append(candidate)

    if placed_boxes:
        boxes = torch.tensor(placed_boxes, dtype=torch.float32)
        labels = torch.ones((len(placed_boxes),), dtype=torch.long)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.long)

    return Scene(image=image, boxes=boxes, labels=labels)


def make_dataset(
    count: int,
    height: int = 128,
    width: int = 128,
    num_objects: int = 8,
    seed: int = 0,
    **kwargs,
) -> list[Scene]:
    """Generate a list of scenes with seeds derived from ``seed``."""
    return [
        make_scene(
            height=height,
            width=width,
            num_objects=num_objects,
            seed=seed + i,
            **kwargs,
        )
        for i in range(count)
    ]
