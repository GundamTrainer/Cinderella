from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from ai.contracts import BackendKind
from ai.evaluation import (
    DetectionBenchmarkSample,
    evaluate_detection_benchmark,
    evaluate_detections,
)
from ai.external_backends import (
    ExternalMediaPipeBackend,
    ExternalOllamaLanguageBackend,
    ExternalSapiSpeechBackend,
    ExternalWhisperBackend,
)
from ai.scene import SceneObservation, WorldStateTracker, tentative_affect_hypotheses
from ai.vision import (
    BoundingBox,
    CoordinateSpace,
    DetectionFrame,
    DetectionResult,
    DetectionStabilizer,
    DetectionState,
    FaceFeatureStabilizer,
    convert_detector_results,
    normalize_bounding_box,
)
from datasets.object_detection.schema import (
    DatasetValidationError,
    example_synthetic_metadata,
    validate_dataset_manifest,
)
from runtime.ai_pipeline import LocalVisionAnalyzer
from runtime.audio_stream import AudioInputBuffer


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
    assert frame.unverified_detections[0].status == DetectionState.LOW_CONFIDENCE


def test_detection_stabilizer_requires_nondecreasing_observation_times() -> None:
    stabilizer = DetectionStabilizer()
    stabilizer.update([], observed_at=1.0)

    with pytest.raises(ValueError, match="nondecreasing"):
        stabilizer.update([], observed_at=0.9)


@pytest.mark.parametrize(
    "confidence",
    (float("nan"), float("inf"), -0.01, 1.01),
)
def test_detection_result_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        DetectionResult(
            0,
            "person",
            confidence,
            BoundingBox(0.1, 0.1, 0.5, 0.6),
        )


def test_bounding_box_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ValueError, match="finite values in"):
        BoundingBox(-0.01, 0.1, 0.5, 0.6)


@pytest.mark.parametrize(
    ("coordinate_space", "coordinates"),
    (
        (CoordinateSpace.PIXEL_XYWH, (10, 20, 30, 40)),
        (CoordinateSpace.PIXEL_XYXY, (10, 20, 40, 60)),
        (CoordinateSpace.NORMALIZED_XYWH, (0.05, 0.2, 0.15, 0.4)),
        (CoordinateSpace.NORMALIZED_XYXY, (0.05, 0.2, 0.2, 0.6)),
    ),
)
def test_box_converter_normalizes_all_supported_coordinate_spaces(
    coordinate_space: CoordinateSpace,
    coordinates: tuple[float, float, float, float],
) -> None:
    box = normalize_bounding_box(
        coordinates,
        coordinate_space=coordinate_space,
        image_width=200,
        image_height=100,
    )

    assert box.x_min == pytest.approx(0.05)
    assert box.y_min == pytest.approx(0.2)
    assert box.x_max == pytest.approx(0.2)
    assert box.y_max == pytest.approx(0.6)


def test_box_converter_clamps_only_within_pixel_tolerance() -> None:
    box = normalize_bounding_box(
        (-1, 10, 21, 30),
        coordinate_space=CoordinateSpace.PIXEL_XYWH,
        image_width=200,
        image_height=100,
        clamp_tolerance_pixels=2,
    )
    assert box == BoundingBox(0.0, 0.1, 0.1, 0.4)

    with pytest.raises(ValueError, match="frame bounds"):
        normalize_bounding_box(
            (-3, 10, 23, 30),
            coordinate_space=CoordinateSpace.PIXEL_XYWH,
            image_width=200,
            image_height=100,
            clamp_tolerance_pixels=2,
        )


@pytest.mark.parametrize(
    ("width", "height", "coordinates"),
    (
        (0, 100, (0, 0, 1, 1)),
        (100, -1, (0, 0, 1, 1)),
        (100, 100, (float("nan"), 0, 1, 1)),
        (100, 100, (0, 0, float("inf"), 1)),
        (100, 100, (0, 0, 0, 1)),
    ),
)
def test_box_converter_rejects_invalid_dimensions_or_geometry(
    width: int,
    height: int,
    coordinates: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError):
        normalize_bounding_box(
            coordinates,
            coordinate_space=CoordinateSpace.PIXEL_XYWH,
            image_width=width,
            image_height=height,
        )


def test_detector_conversion_keeps_missing_class_id_unverified() -> None:
    detection = SimpleNamespace(
        categories=[SimpleNamespace(index=None, category_name="person", score=0.9)],
        bounding_box=SimpleNamespace(origin_x=10, origin_y=20, width=30, height=40),
    )

    result = convert_detector_results(
        [detection],
        frame_id=7,
        timestamp=1.0,
        image_width=200,
        image_height=100,
    )

    assert result.detections == ()
    assert result.unverified[0].status == DetectionState.MISSING_CLASS
    assert result.unverified[0].label == "person"
    assert result.unverified[0].bounding_box == BoundingBox(0.05, 0.2, 0.2, 0.6)


@pytest.mark.parametrize(
    ("category", "expected_status"),
    (
        (SimpleNamespace(index=0, category_name="", score=0.9), DetectionState.UNKNOWN),
        (SimpleNamespace(index=-1, category_name="person", score=0.9), DetectionState.REJECTED),
        (SimpleNamespace(index=0, category_name="person", score=None), DetectionState.REJECTED),
    ),
)
def test_detector_conversion_does_not_promote_invalid_category_data(
    category: SimpleNamespace,
    expected_status: DetectionState,
) -> None:
    detection = SimpleNamespace(
        categories=[category],
        bounding_box=SimpleNamespace(origin_x=10, origin_y=20, width=30, height=40),
    )

    result = convert_detector_results(
        [detection],
        frame_id=1,
        timestamp=0.5,
        image_width=200,
        image_height=100,
    )

    assert result.detections == ()
    assert result.unverified[0].status == expected_status


def test_detector_conversion_selects_highest_valid_category_score() -> None:
    detection = SimpleNamespace(
        categories=[
            SimpleNamespace(index=2, category_name="cup", score=0.4),
            SimpleNamespace(index=1, category_name="book", score=0.9),
        ],
        bounding_box=SimpleNamespace(origin_x=10, origin_y=20, width=30, height=40),
    )

    result = convert_detector_results(
        [detection],
        frame_id=1,
        timestamp=0.5,
        image_width=200,
        image_height=100,
    )

    assert len(result.detections) == 1
    assert result.detections[0].class_id == 1
    assert result.detections[0].label == "book"
    assert result.detections[0].confidence == 0.9


def test_audio_input_buffer_is_bounded_and_marks_device_overflow() -> None:
    buffer = AudioInputBuffer(capacity=2)

    assert buffer.submit([1, 2], frame_count=2)
    assert buffer.submit([3, 4], frame_count=2, status="input overflow")
    assert not buffer.submit([5, 6], frame_count=2)

    first = buffer.get_nowait()
    second = buffer.get_nowait()
    assert (first.sequence_number, second.sequence_number) == (1, 2)
    assert not first.input_overflow
    assert second.input_overflow
    assert buffer.drain_diagnostics() == {
        "audio.frame_dropped": 2,
        "audio.input_overflow": 2,
        "audio.queue_pressure": 2,
    }


def test_audio_discard_advances_sequence_to_break_segments() -> None:
    buffer = AudioInputBuffer(capacity=2)
    assert buffer.submit([1, 2], frame_count=2)
    buffer.discard(frame_count=2, reason="tts_echo_guard")
    assert buffer.submit([5, 6], frame_count=2)

    assert buffer.get_nowait().sequence_number == 1
    assert buffer.get_nowait().sequence_number == 3
    assert buffer.drain_diagnostics() == {
        "audio.segment_rejected:tts_echo_guard": 2,
        "audio.queue_pressure": 2,
    }


def test_world_state_tracks_stable_motion_and_expires_missing_objects() -> None:
    tracker = WorldStateTracker(object_expiration_seconds=2.0)
    face_features = FaceFeatureStabilizer().update(
        None,
        face_count=0,
        observed_at=1.0,
    )

    def observation(timestamp: float, detection: DetectionResult | None) -> SceneObservation:
        detections = DetectionFrame(
            observed_at=timestamp,
            raw_detections=(detection,) if detection else (),
            stable_detections=(detection,) if detection else (),
        )
        return SceneObservation(
            frame_id=int(timestamp * 10),
            observed_at=timestamp,
            detections=detections,
            action_cues=(),
            face_features=face_features,
        )

    first = DetectionResult(
        1,
        "book",
        0.9,
        BoundingBox(0.1, 0.2, 0.3, 0.5),
        DetectionState.STABLE,
        track_id=4,
        last_seen=1.0,
    )
    second = DetectionResult(
        1,
        "book",
        0.9,
        BoundingBox(0.3, 0.2, 0.5, 0.5),
        DetectionState.STABLE,
        track_id=4,
        last_seen=2.0,
    )

    assert tracker.update(observation(1.0, first)).objects[0].motion == "unknown"
    assert tracker.update(observation(2.0, second)).objects[0].motion == "moving"
    assert tracker.update(observation(4.1, None)).objects == ()


def test_world_state_rejects_out_of_order_observations() -> None:
    tracker = WorldStateTracker()
    face_features = FaceFeatureStabilizer().update(None, face_count=0, observed_at=1.0)

    def observation(timestamp: float) -> SceneObservation:
        return SceneObservation(
            frame_id=1,
            observed_at=timestamp,
            detections=DetectionFrame(timestamp, (), ()),
            action_cues=(),
            face_features=face_features,
        )

    tracker.update(observation(1.0))
    with pytest.raises(ValueError, match="nondecreasing"):
        tracker.update(observation(0.5))


def test_tentative_affect_hypothesis_is_low_confidence_and_requires_confirmation() -> None:
    hypotheses = tentative_affect_hypotheses(
        ("smile-like facial movement",),
        observed_at=5.0,
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].hypothesis == "possible_positive_affect"
    assert hypotheses[0].confidence == 0.25
    assert hypotheses[0].uncertainty.startswith("high")
    assert hypotheses[0].requires_confirmation
    assert hypotheses[0].valid_until == 7.0
    assert tentative_affect_hypotheses((), observed_at=5.0) == ()


@pytest.mark.parametrize(
    ("category_name", "expected_objects"),
    (("person", ("person",)), ("", ()), ("unknown", ())),
)
def test_vision_analyzer_normalizes_boxes_and_only_exposes_stable_labels(
    category_name: str,
    expected_objects: tuple[str, ...],
) -> None:
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
    category = SimpleNamespace(index=0, category_name=category_name, score=0.9)
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

    analyzer._object_detector = SimpleNamespace(detect_for_video=detect_objects)
    analyzer._pose_landmarker = SimpleNamespace(detect_for_video=detect_pose)
    analyzer._face_landmarker = SimpleNamespace(detect_for_video=detect_faces)
    analyzer._smoothed_blendshapes = {}
    analyzer._detection_stabilizer = DetectionStabilizer(
        minimum_confidence=0.5,
        stable_hits=2,
        supported_class_ids=set(range(80)),
    )
    analyzer._face_stabilizer = FaceFeatureStabilizer()
    analyzer._box_clamp_tolerance_pixels = 2.0
    analyzer._frame_id = 0
    analyzer._diagnostic_totals = Counter()
    analyzer._diagnostic_log_milestones = {}
    analyzer._world_state_tracker = WorldStateTracker()
    frame = SimpleNamespace(shape=(100, 200, 3))

    first = analyzer.analyze(frame, timestamp_ms=1000)
    second = analyzer.analyze(frame, timestamp_ms=1400)

    assert first.objects == ()
    assert second.objects == expected_objects
    assert second.detections is not None
    assert second.observation is not None
    assert second.world_state is not None
    if expected_objects:
        assert second.world_state.objects[0].label == "person"
        assert "person at" in second.prompt_context()
    else:
        assert second.world_state.uncertain_labels == ("unknown",)
        assert "미확정 후보=unknown" in second.prompt_context()
    expected_box = BoundingBox(0.05, 0.2, 0.2, 0.6)
    if category_name in ("", "unknown"):
        assert second.detections.raw_detections == ()
        assert second.detections.unverified_detections[0].bounding_box == expected_box
    else:
        assert second.detections.raw_detections[0].box == expected_box
    if expected_objects:
        assert second.detections.stable_detections[0].state == DetectionState.STABLE
    else:
        assert second.detections.stable_detections == ()
        assert second.detections.unverified_detections[0].label == (
            None if category_name == "" else "unknown"
        )
        assert second.detections.unverified_detections[0].status == DetectionState.UNKNOWN


def test_vision_analyzer_closes_all_tasks_if_one_close_fails() -> None:
    closed: list[str] = []

    class FakeTask:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    analyzer = object.__new__(LocalVisionAnalyzer)
    analyzer._face_landmarker = FakeTask("face", fail=True)
    analyzer._pose_landmarker = FakeTask("pose")
    analyzer._object_detector = FakeTask("object")
    analyzer._detection_stabilizer = DetectionStabilizer()
    analyzer._face_stabilizer = FaceFeatureStabilizer()
    analyzer._world_state_tracker = WorldStateTracker()

    with pytest.raises(ExceptionGroup, match="MediaPipe tasks failed to close"):
        analyzer.close()

    assert closed == ["face", "pose", "object"]
    assert analyzer._face_landmarker is None
    assert analyzer._pose_landmarker is None
    assert analyzer._object_detector is None


def test_external_backend_metadata_identifies_provenance_and_pretrained_weights() -> None:
    backends = (
        ExternalWhisperBackend(),
        ExternalMediaPipeBackend(),
        ExternalOllamaLanguageBackend(),
        ExternalSapiSpeechBackend(),
    )

    assert all(backend.backend_info.kind == BackendKind.EXTERNAL for backend in backends)
    assert [backend.backend_info.uses_pretrained_weights for backend in backends] == [
        True,
        True,
        True,
        None,
    ]


def test_whisper_adapter_exposes_transcription_rejection_reason() -> None:
    backend = ExternalWhisperBackend()
    backend._delegate = SimpleNamespace(last_rejection_reason="speech_not_detected")

    assert backend.last_rejection_reason == "speech_not_detected"


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


def test_dataset_benchmark_aggregates_metrics_across_samples() -> None:
    exact_box = BoundingBox(0.1, 0.1, 0.5, 0.5)
    samples = (
        DetectionBenchmarkSample(
            "frame-1",
            (DetectionResult(1, "book", 0.9, exact_box),),
            ((1, exact_box),),
        ),
        DetectionBenchmarkSample("frame-2", (), ((2, exact_box),)),
        DetectionBenchmarkSample(
            "frame-3",
            (DetectionResult(3, "cup", 0.8, exact_box),),
            (),
        ),
    )

    report = evaluate_detection_benchmark(samples)

    assert report.samples_evaluated == 3
    assert (report.true_positives, report.false_positives, report.false_negatives) == (1, 1, 1)
    assert report.precision == pytest.approx(0.5)
    assert report.recall == pytest.approx(0.5)
    assert report.mean_matched_iou == pytest.approx(1.0)


def test_dataset_benchmark_rejects_duplicate_sample_ids() -> None:
    sample = DetectionBenchmarkSample("same-frame", (), ())

    with pytest.raises(ValueError, match="unique"):
        evaluate_detection_benchmark((sample, sample))


def test_dataset_manifest_accepts_synthetic_non_media_metadata() -> None:
    validate_dataset_manifest(example_synthetic_metadata())


def test_dataset_manifest_rejects_parent_path() -> None:
    manifest = example_synthetic_metadata()
    manifest["samples"][0]["image_ref"] = "../outside.png"

    with pytest.raises(DatasetValidationError, match="safe relative"):
        validate_dataset_manifest(manifest)
