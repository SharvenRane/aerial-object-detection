"""Aerial object detection on synthetic aerial and satellite scenes."""

from . import boxes, synthetic, anchors, model, postprocess, evaluate, detector

__all__ = [
    "boxes",
    "synthetic",
    "anchors",
    "model",
    "postprocess",
    "evaluate",
    "detector",
]
