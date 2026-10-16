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


@dataclass(frozen=True)
class DetectionBenchmarkSample:
    sample_id: str
    predictions: tuple[DetectionResult, ...]
    ground_truth: tuple[tuple[int, BoundingBox], ...]


@dataclass(frozen=True)
class DetectionBenchmarkReport:
    samples_evaluated: int
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
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not 0 <= iou_threshold <= 1
    ):
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


def evaluate_detection_benchmark(
    samples: Iterable[DetectionBenchmarkSample],
    *,
    iou_threshold: float = 0.5,
) -> DetectionBenchmarkReport:
    """Aggregate per-frame detector metrics over a fixed, caller-supplied dataset."""
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not 0.0 <= iou_threshold <= 1.0
    ):
        raise ValueError("iou_threshold must be in [0, 1].")
    sample_list = tuple(samples)
    if not sample_list:
        raise ValueError("At least one benchmark sample is required.")
    sample_ids = [sample.sample_id for sample in sample_list]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        raise TypeError("Benchmark sample IDs must be strings.")
    if any(not sample_id.strip() for sample_id in sample_ids):
        raise ValueError("Benchmark sample IDs must not be empty.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Benchmark sample IDs must be unique.")

    evaluations = [
        evaluate_detections(
            sample.predictions,
            sample.ground_truth,
            iou_threshold=iou_threshold,
        )
        for sample in sample_list
    ]
    true_positives = sum(item.true_positives for item in evaluations)
    false_positives = sum(item.false_positives for item in evaluations)
    false_negatives = sum(item.false_negatives for item in evaluations)
    matched_count = sum(item.true_positives for item in evaluations)
    predicted_count = true_positives + false_positives
    ground_truth_count = true_positives + false_negatives
    matched_iou_sum = sum(item.mean_matched_iou * item.true_positives for item in evaluations)
    return DetectionBenchmarkReport(
        samples_evaluated=len(sample_list),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=true_positives / predicted_count if predicted_count else 0.0,
        recall=true_positives / ground_truth_count if ground_truth_count else 0.0,
        mean_matched_iou=matched_iou_sum / matched_count if matched_count else 0.0,
    )
