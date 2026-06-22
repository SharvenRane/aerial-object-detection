"""A small single stage anchor based detector.

The backbone is a shallow convolutional stack that downsamples the image by a
factor of ``stride``. On top of that feature map sit two heads: an objectness
classifier and a box regressor. Both predict one value set per anchor at every
spatial location. The design mirrors a single level RetinaNet or RPN, kept
small enough to train and run on CPU inside a unit test.

The box regression uses the standard center offset parameterisation so the
network only has to learn corrections relative to the anchor grid, which is
what makes small dense objects tractable.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .anchors import num_anchors_per_location


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.ReLU(inplace=True),
    )


class Backbone(nn.Module):
    """Shallow feature extractor with a fixed total stride of 4."""

    stride = 4

    def __init__(self, in_channels: int = 3, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_block(in_channels, width, stride=2),  # /2
            _conv_block(width, width),
            _conv_block(width, width * 2, stride=2),  # /4
            _conv_block(width * 2, width * 2),
        )
        self.out_channels = width * 2

    def forward(self, x: Tensor) -> Tensor:
        return self.stem(x)


class DetectionHead(nn.Module):
    """Predicts objectness logits and box deltas per anchor."""

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int = 1):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.shared = nn.Sequential(
            _conv_block(in_channels, in_channels),
            _conv_block(in_channels, in_channels),
        )
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.bbox_deltas = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, padding=1
        )

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.shared(feat)
        cls = self.cls_logits(feat)
        reg = self.bbox_deltas(feat)
        return cls, reg


class AerialDetector(nn.Module):
    """End to end single level anchor detector.

    Forward returns flattened per anchor predictions so they line up with the
    flat anchor grid from :func:`src.anchors.generate_anchors`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        sizes: tuple[int, ...] = (6, 10, 16),
        aspect_ratios: tuple[float, ...] = (1.0,),
        num_classes: int = 1,
    ):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_classes = num_classes
        self.backbone = Backbone(in_channels=in_channels, width=width)
        self.num_anchors = num_anchors_per_location(sizes, aspect_ratios)
        self.head = DetectionHead(
            self.backbone.out_channels, self.num_anchors, num_classes
        )

    @property
    def stride(self) -> int:
        return self.backbone.stride

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        """Run the detector.

        Args:
            images: Float tensor of shape ``(B, C, H, W)``.

        Returns:
            Dict with ``cls_logits`` of shape ``(B, L, num_classes)`` and
            ``bbox_deltas`` of shape ``(B, L, 4)`` where ``L`` is the number of
            anchors. The anchor ordering matches ``generate_anchors`` with the
            same image size, stride, sizes and aspect ratios.
        """
        if images.dim() != 4:
            raise ValueError(
                f"expected images of shape (B, C, H, W), got {tuple(images.shape)}"
            )
        feat = self.backbone(images)
        cls, reg = self.head(feat)

        b, _, fh, fw = cls.shape
        a = self.num_anchors

        # Reshape so the anchor axis matches generate_anchors: row major over
        # (h, w) locations, then over the anchor index at each location.
        cls = (
            cls.view(b, a, self.num_classes, fh, fw)
            .permute(0, 3, 4, 1, 2)
            .reshape(b, fh * fw * a, self.num_classes)
        )
        reg = (
            reg.view(b, a, 4, fh, fw)
            .permute(0, 3, 4, 1, 2)
            .reshape(b, fh * fw * a, 4)
        )
        return {"cls_logits": cls, "bbox_deltas": reg}
