"""Model backend contracts and evaluation utilities for Cinderella."""

from .contracts import BackendInfo, BackendKind, BackendStatus
from .vision import BoundingBox, DetectionResult, FaceFeatureFrame, NormalizedFeature

__all__ = [
    "BackendInfo",
    "BackendKind",
    "BackendStatus",
    "BoundingBox",
    "DetectionResult",
    "FaceFeatureFrame",
    "NormalizedFeature",
]
