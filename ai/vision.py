from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class DetectionState(StrEnum):
    CANDIDATE = "candidate"
    STABLE = "stable"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"
    UNSUPPORTED_CLASS = "unsupported_class"


@dataclass(frozen=True)
class BoundingBox:
    """Normalized xyxy coordinates in the inclusive 0.0-1.0 image range."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Bounding box coordinates must be finite values in [0, 1].")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Bounding box must have positive width and height.")

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass(frozen=True)
class DetectionResult:
    class_id: int
    label: str
    confidence: float
    box: BoundingBox
    state: DetectionState = DetectionState.CANDIDATE
    track_id: int | None = None

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("class_id must be zero or greater.")
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1].")


@dataclass(frozen=True)
class DetectionFrame:
    observed_at: float
    raw_detections: tuple[DetectionResult, ...]
    stable_detections: tuple[DetectionResult, ...]
    unverified_detections: tuple[DetectionResult, ...] = ()
    rejected_count: int = 0


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


class ClassAwareOverlapResolver:
    """Suppress same-class overlapping proposals, highest confidence first."""

    def __init__(self, iou_threshold: float = 0.65) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1].")
        self.iou_threshold = iou_threshold

    def resolve(self, detections: Iterable[DetectionResult]) -> tuple[DetectionResult, ...]:
        ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
        retained: list[DetectionResult] = []
        for candidate in ordered:
            if any(
                candidate.class_id == previous.class_id
                and intersection_over_union(candidate.box, previous.box) > self.iou_threshold
                for previous in retained
            ):
                continue
            retained.append(candidate)
        return tuple(retained)


@dataclass
class _DetectionTrack:
    track_id: int
    class_id: int
    label: str
    box: BoundingBox
    confidence: float
    hits: int
    missed_frames: int
    last_seen: float


class DetectionStabilizer:
    """Keep a separate multi-frame stable view without discarding raw detections."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.35,
        class_thresholds: dict[int, float] | None = None,
        minimum_area: float = 0.0005,
        maximum_area: float = 0.98,
        stable_hits: int = 2,
        missing_tolerance_seconds: float = 0.8,
        iou_threshold: float = 0.3,
        supported_class_ids: set[int] | None = None,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1].")
        if not 0 <= minimum_area < maximum_area <= 1:
            raise ValueError("area thresholds must satisfy 0 <= minimum < maximum <= 1.")
        if stable_hits < 1 or missing_tolerance_seconds < 0:
            raise ValueError("stable_hits must be positive and missing tolerance nonnegative.")
        if any(not 0 <= threshold <= 1 for threshold in (class_thresholds or {}).values()):
            raise ValueError("class confidence thresholds must be in [0, 1].")
        self.minimum_confidence = minimum_confidence
        self.class_thresholds = dict(class_thresholds or {})
        self.minimum_area = minimum_area
        self.maximum_area = maximum_area
        self.stable_hits = stable_hits
        self.missing_tolerance_seconds = missing_tolerance_seconds
        self.supported_class_ids = supported_class_ids
        self._overlap_resolver = ClassAwareOverlapResolver(iou_threshold)
        self._tracks: list[_DetectionTrack] = []
        self._next_track_id = 1
        self._last_observed_at: float | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._last_observed_at = None

    def update(
        self, raw_detections: Iterable[DetectionResult], *, observed_at: float
    ) -> DetectionFrame:
        if not isfinite(observed_at) or (
            self._last_observed_at is not None and observed_at < self._last_observed_at
        ):
            raise ValueError("observed_at must be finite and nondecreasing.")
        self._last_observed_at = observed_at
        raw = tuple(raw_detections)
        eligible: list[DetectionResult] = []
        rejected = 0
        unknown: list[DetectionResult] = []
        for detection in raw:
            threshold = self.class_thresholds.get(detection.class_id, self.minimum_confidence)
            if detection.state == DetectionState.UNSUPPORTED_CLASS or (
                self.supported_class_ids is not None
                and detection.class_id not in self.supported_class_ids
            ):
                unknown.append(
                    DetectionResult(
                        detection.class_id,
                        "unknown",
                        detection.confidence,
                        detection.box,
                        DetectionState.UNSUPPORTED_CLASS,
                    )
                )
            elif detection.confidence < threshold:
                unknown.append(
                    DetectionResult(
                        detection.class_id,
                        "unknown",
                        detection.confidence,
                        detection.box,
                        DetectionState.LOW_CONFIDENCE,
                    )
                )
            elif not self.minimum_area <= detection.box.area <= self.maximum_area:
                rejected += 1
            else:
                eligible.append(detection)
        resolved = self._overlap_resolver.resolve(eligible)

        for track in self._tracks:
            track.missed_frames += 1
        unmatched_tracks = set(range(len(self._tracks)))
        for detection in resolved:
            matches = [
                (index, intersection_over_union(detection.box, track.box))
                for index, track in enumerate(self._tracks)
                if index in unmatched_tracks and track.class_id == detection.class_id
            ]
            match = max(matches, key=lambda item: item[1], default=None)
            if match is not None and match[1] >= self._overlap_resolver.iou_threshold:
                index = match[0]
                track = self._tracks[index]
                unmatched_tracks.remove(index)
                track.box = detection.box
                track.confidence = detection.confidence
                track.hits += 1
                track.missed_frames = 0
                track.last_seen = observed_at
            else:
                track = _DetectionTrack(
                    self._next_track_id,
                    detection.class_id,
                    detection.label,
                    detection.box,
                    detection.confidence,
                    1,
                    0,
                    observed_at,
                )
                self._next_track_id += 1
                self._tracks.append(track)

        self._tracks = [
            track
            for track in self._tracks
            if observed_at - track.last_seen <= self.missing_tolerance_seconds
        ]
        stable = tuple(
            DetectionResult(
                track.class_id,
                track.label,
                track.confidence,
                track.box,
                DetectionState.STABLE if track.hits >= self.stable_hits else DetectionState.CANDIDATE,
                track.track_id,
            )
            for track in self._tracks
            if track.hits >= self.stable_hits
        )
        return DetectionFrame(observed_at, raw, stable, tuple(unknown), rejected)


@dataclass(frozen=True)
class NormalizedFeature:
    value: float
    confidence: float
    valid: bool
    observed_at: float
    smoothing_applied: bool
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.value, self.confidence, self.observed_at)):
            raise ValueError("Feature values, confidence, and observation time must be finite.")
        if not 0 <= self.minimum < self.maximum <= 1:
            raise ValueError("Feature normalization range must be inside [0, 1].")
        if not self.minimum <= self.value <= self.maximum:
            raise ValueError("Feature value is outside its normalization range.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Feature confidence must be in [0, 1].")


@dataclass(frozen=True)
class FaceFeatureFrame:
    face_present: bool
    face_count: int
    features: dict[str, NormalizedFeature]
    observed_at: float
    stable: bool
    unknown: bool = False


class FaceFeatureStabilizer:
    """EMA feature stabilizer with a stable-duration gate and missing-frame tolerance."""

    def __init__(
        self,
        *,
        smoothing: float = 0.45,
        minimum_stable_duration: float = 0.25,
        missing_tolerance_frames: int = 2,
        stability_threshold: float = 0.35,
    ) -> None:
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1].")
        if minimum_stable_duration < 0 or missing_tolerance_frames < 0:
            raise ValueError("duration and missing frame tolerance must be nonnegative.")
        if not 0 <= stability_threshold <= 1:
            raise ValueError("stability_threshold must be in [0, 1].")
        self.smoothing = smoothing
        self.minimum_stable_duration = minimum_stable_duration
        self.missing_tolerance_frames = missing_tolerance_frames
        self.stability_threshold = stability_threshold
        self._values: dict[str, float] = {}
        self._candidate_since: float | None = None
        self._missing_frames = 0
        self._previous: dict[str, float] = {}
        self._stability = 0.0

    def reset(self) -> None:
        self._values.clear()
        self._previous.clear()
        self._candidate_since = None
        self._missing_frames = 0
        self._stability = 0.0

    def update(
        self,
        features: dict[str, float] | None,
        *,
        face_count: int,
        observed_at: float,
    ) -> FaceFeatureFrame:
        if not isfinite(observed_at) or observed_at < 0:
            raise ValueError("observed_at must be a finite nonnegative timestamp.")
        if face_count < 0:
            raise ValueError("face_count must be nonnegative.")
        if face_count != 1 or features is None:
            self._missing_frames += 1
            if self._missing_frames > self.missing_tolerance_frames or face_count > 1:
                self.reset()
            return FaceFeatureFrame(
                face_present=face_count > 0,
                face_count=face_count,
                features={},
                observed_at=observed_at,
                stable=False,
                unknown=face_count > 1,
            )
        if any(not isfinite(value) or not 0 <= value <= 1 for value in features.values()):
            raise ValueError("Face feature measurements must be finite values in [0, 1].")
        self._missing_frames = 0
        if self._candidate_since is None:
            self._candidate_since = observed_at
        previous = self._values.copy()
        self._values = {
            name: (
                self.smoothing * value + (1 - self.smoothing) * previous.get(name, value)
            )
            for name, value in features.items()
        }
        differences = [
            abs(self._values[name] - previous[name])
            for name in self._values.keys() & previous.keys()
        ]
        stability = 1.0 - min(1.0, sum(differences) / len(differences)) if differences else 0.0
        self._stability = stability
        stable = (
            observed_at - self._candidate_since >= self.minimum_stable_duration
            and stability >= self.stability_threshold
        )
        values = {
            name: NormalizedFeature(
                value=value,
                confidence=self._stability if stable else 0.0,
                valid=stable,
                observed_at=observed_at,
                smoothing_applied=True,
            )
            for name, value in self._values.items()
        }
        self._previous = previous
        return FaceFeatureFrame(True, 1, values, observed_at, stable)
