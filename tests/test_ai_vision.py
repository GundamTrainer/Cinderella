from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.evaluation import evaluate_detections
from ai.vision import (
    BoundingBox,
    DetectionResult,
    DetectionStabilizer,
    DetectionState,
    FaceFeatureStabilizer,
)
from datasets.object_detection.schema import (
    DatasetValidationError,
    example_synthetic_metadata,
    validate_dataset_manifest,
)
from runtime.ai_pipeline import LocalVisionAnalyzer


def test_detection_stabilizer_only_promotes_repeated_confident_boxes() -> None:
    stabilizer = DetectionStabilizer(
        minimum_confidence=0.5,
        stable_hits=2,
        supported_class_ids={0},
    )
    box = BoundingBox(0.1, 0.1, 0.5, 0.6)
    detection = DetectionResult(0, "person", 0.8, box)

    first = stabilizer.update([detection], observed_at=1.0)
    second = stabilizer.update([detection], observed_at=1.4)

    assert first.stable_detections == ()
    assert len(second.stable_detections) == 1
    assert second.stable_detections[0].state == DetectionState.STABLE
    assert second.stable_detections[0].track_id == 1


def test_detection_stabilizer_keeps_low_confidence_as_unknown() -> None:
    stabilizer = DetectionStabilizer(
        minimum_confidence=0.5,
        supported_class_ids={0},
    )
    low_confidence = DetectionResult(
        0,
        "person",
        0.2,
        BoundingBox(0.1, 0.1, 0.5, 0.6),
    )

    frame = stabilizer.update([low_confidence], observed_at=1.0)

    assert frame.stable_detections == ()
    assert frame.unverified_detections[0].label == "unknown"
    assert frame.unverified_detections[0].state == DetectionState.LOW_CONFIDENCE


def test_detection_stabilizer_requires_nondecreasing_observation_times() -> None:
    stabilizer = DetectionStabilizer()
    stabilizer.update([], observed_at=1.0)

    with pytest.raises(ValueError, match="nondecreasing"):
        stabilizer.update([], observed_at=0.9)


def test_vision_analyzer_normalizes_boxes_and_only_exposes_stable_labels() -> None:
    class FakeCV2:
        COLOR_BGR2RGB = 1

        @staticmethod
        def cvtColor(frame: object, _conversion: int) -> object:
            del _conversion
            return frame

    class FakeImage:
        def __init__(self, image_format: object, data: object) -> None:
            del image_format, data

    box = SimpleNamespace(origin_x=10, origin_y=20, width=30, height=40)
    category = SimpleNamespace(index=0, category_name="person", score=0.9)
    detection = SimpleNamespace(categories=[category], bounding_box=box)
    analyzer = object.__new__(LocalVisionAnalyzer)
    analyzer._cv2 = FakeCV2()
    analyzer._mp = SimpleNamespace(
        Image=FakeImage,
        ImageFormat=SimpleNamespace(SRGB=1),
    )

    def detect_objects(_image: object, _timestamp: int) -> SimpleNamespace:
        del _image, _timestamp
        return SimpleNamespace(detections=[detection])

    def detect_pose(_image: object, _timestamp: int) -> SimpleNamespace:
        del _image, _timestamp
        return SimpleNamespace(pose_landmarks=[])

    def detect_faces(_image: object, _timestamp: int) -> SimpleNamespace:
        del _image, _timestamp
        return SimpleNamespace(face_landmarks=[], face_blendshapes=[])

    analyzer._object_detector = SimpleNamespace(
        detect_for_video=detect_objects
    )
    analyzer._pose_landmarker = SimpleNamespace(
        detect_for_video=detect_pose
    )
    analyzer._face_landmarker = SimpleNamespace(detect_for_video=detect_faces)
    analyzer._smoothed_blendshapes = {}
    analyzer._detection_stabilizer = DetectionStabilizer(
        minimum_confidence=0.5,
        stable_hits=2,
        supported_class_ids=set(range(80)),
    )
    analyzer._face_stabilizer = FaceFeatureStabilizer()
    frame = SimpleNamespace(shape=(100, 200, 3))

    first = analyzer.analyze(frame, timestamp_ms=1000)
    second = analyzer.analyze(frame, timestamp_ms=1400)

    assert first.objects == ()
    assert second.objects == ("person",)
    assert second.detections is not None
    assert second.detections.raw_detections[0].box == BoundingBox(0.05, 0.2, 0.2, 0.6)


def test_face_feature_stabilizer_reports_unknown_for_multiple_faces() -> None:
    stabilizer = FaceFeatureStabilizer(minimum_stable_duration=0.0)

    frame = stabilizer.update(
        {"mouth_openness": 0.5},
        face_count=2,
        observed_at=1.0,
    )

    assert frame.face_present is True
    assert frame.unknown is True
    assert frame.stable is False
    assert frame.features == {}


def test_face_feature_stabilizer_emits_normalized_features_after_stabilizing() -> None:
    stabilizer = FaceFeatureStabilizer(
        smoothing=0.5,
        minimum_stable_duration=0.2,
    )
    values = {"mouth_openness": 0.4}

    initial = stabilizer.update(values, face_count=1, observed_at=1.0)
    stabilizing = stabilizer.update(values, face_count=1, observed_at=1.1)
    stable = stabilizer.update(values, face_count=1, observed_at=1.3)

    assert initial.stable is False
    assert stabilizing.stable is False
    assert stable.stable is True
    assert stable.features["mouth_openness"].valid is True
    assert stable.features["mouth_openness"].value == pytest.approx(0.4)


def test_evaluation_matches_predictions_by_class_and_iou() -> None:
    box = BoundingBox(0.1, 0.1, 0.5, 0.5)
    result = evaluate_detections(
        [DetectionResult(1, "cup", 0.9, box)],
        [(1, box)],
    )

    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 0, 0)
    assert result.precision == result.recall == result.mean_matched_iou == 1.0


def test_dataset_manifest_accepts_synthetic_non_media_metadata() -> None:
    validate_dataset_manifest(example_synthetic_metadata())


def test_dataset_manifest_rejects_parent_path() -> None:
    manifest = example_synthetic_metadata()
    manifest["samples"][0]["image_ref"] = "../outside.png"

    with pytest.raises(DatasetValidationError, match="safe relative"):
        validate_dataset_manifest(manifest)
