from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .vision import BoundingBox, DetectionResult, intersection_over_union


@dataclass(frozen=True)
class DetectionEvaluation:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    mean_matched_iou: float


def evaluate_detections(
    predictions: Iterable[DetectionResult],
    ground_truth: Iterable[tuple[int, BoundingBox]],
    *,
    iou_threshold: float = 0.5,
) -> DetectionEvaluation:
    """Evaluate one frame with class-aware greedy IoU matching."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1].")
    remaining = list(ground_truth)
    true_positives = 0
    matched_ious: list[float] = []
    false_positives = 0
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        matches = [
            (index, intersection_over_union(prediction.box, target_box))
            for index, (class_id, target_box) in enumerate(remaining)
            if prediction.class_id == class_id
        ]
        match = max(matches, key=lambda item: item[1], default=None)
        if match is None or match[1] < iou_threshold:
            false_positives += 1
            continue
        true_positives += 1
        matched_ious.append(match[1])
        remaining.pop(match[0])
    false_negatives = len(remaining)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    return DetectionEvaluation(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=true_positives / precision_denominator if precision_denominator else 0.0,
        recall=true_positives / recall_denominator if recall_denominator else 0.0,
        mean_matched_iou=sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
    )
