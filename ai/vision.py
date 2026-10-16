from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any


class DetectionState(StrEnum):
    CANDIDATE = "candidate"
    STABLE = "stable"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"
    UNSUPPORTED_CLASS = "unsupported_class"
    MISSING_CLASS = "missing_class"
    INVALID_BOX = "invalid_box"
    UNSTABLE = "unstable"
    REJECTED = "rejected"


class CoordinateSpace(StrEnum):
    PIXEL_XYWH = "pixel_xywh"
    PIXEL_XYXY = "pixel_xyxy"
    NORMALIZED_XYWH = "normalized_xywh"
    NORMALIZED_XYXY = "normalized_xyxy"


@dataclass(frozen=True)
class BoundingBox:
    """Normalized xyxy coordinates in the inclusive 0.0-1.0 image range."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        try:
            numeric_values = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Bounding box coordinates must be finite values in [0, 1].") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ) or any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in numeric_values):
            raise ValueError("Bounding box coordinates must be finite values in [0, 1].")
        for field_name, value in zip(("x_min", "y_min", "x_max", "y_max"), numeric_values):
            object.__setattr__(self, field_name, value)
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
    last_seen: float | None = None
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED_XYXY
    frame_id: int = 0
    timestamp: float | None = None
    source_backend: str = "unknown"
    raw_class_id: int | float | str | None = None
    validation_errors: tuple[str, ...] = ()
    synthetic: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
        ):
            raise ValueError("class_id must be zero or greater.")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must not be empty.")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1].")
        if not isinstance(self.box, BoundingBox):
            raise TypeError("box must be a validated BoundingBox.")
        if not isinstance(self.state, DetectionState):
            raise TypeError("state must be a DetectionState.")
        if not isinstance(self.coordinate_space, CoordinateSpace):
            raise TypeError("coordinate_space must be a CoordinateSpace.")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer.")
        if self.timestamp is not None and (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not isfinite(self.timestamp)
        ):
            raise ValueError("timestamp must be finite.")
        if not isinstance(self.source_backend, str) or not self.source_backend.strip():
            raise ValueError("source_backend must not be empty.")
        if self.raw_class_id is not None and (
            isinstance(self.raw_class_id, bool)
            or not isinstance(self.raw_class_id, (int, float, str))
        ):
            raise TypeError("raw_class_id must be a string, number, or None.")
        if not isinstance(self.synthetic, bool):
            raise TypeError("synthetic must be a boolean.")
        if any(not isinstance(item, str) or not item.strip() for item in self.validation_errors):
            raise ValueError("validation_errors must contain non-empty strings.")
        if self.track_id is not None and (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id < 0
        ):
            raise ValueError("track_id must be a nonnegative integer.")
        if self.last_seen is not None and (
            isinstance(self.last_seen, bool)
            or not isinstance(self.last_seen, (int, float))
            or not isfinite(self.last_seen)
            or self.last_seen < 0
        ):
            raise ValueError("last_seen must be finite and nonnegative.")

    @property
    def normalized_class_id(self) -> int:
        return self.class_id

    @property
    def normalized_label(self) -> str:
        return self.label

    @property
    def tracking_id(self) -> int | None:
        return self.track_id

    @property
    def stable(self) -> bool:
        return self.state == DetectionState.STABLE

    @property
    def validation_status(self) -> DetectionState:
        return self.state


@dataclass(frozen=True)
class DetectionObservation:
    detection_id: str
    frame_id: int
    timestamp: float
    source_backend: str
    raw_class_id: float | str | None = None
    normalized_class_id: int | None = None
    label: str | None = None
    confidence: float | None = None
    bounding_box: BoundingBox | None = None
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED_XYXY
    status: DetectionState = DetectionState.UNKNOWN
    validation_errors: tuple[str, ...] = ()
    tracking_id: int | None = None
    synthetic: bool = False
    raw_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.detection_id, str) or not isinstance(self.source_backend, str):
            raise TypeError("detection_id and source_backend must be strings.")
        if not self.detection_id or not self.source_backend.strip():
            raise ValueError("detection_id and source_backend must not be empty.")
        if (
            isinstance(self.frame_id, bool)
            or not isinstance(self.frame_id, int)
            or self.frame_id < 0
        ):
            raise ValueError("frame_id must be a nonnegative integer.")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not isfinite(self.timestamp)
        ):
            raise ValueError("timestamp must be finite.")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1].")
        if self.bounding_box is not None and not isinstance(self.bounding_box, BoundingBox):
            raise TypeError("bounding_box must be a validated BoundingBox.")
        if not isinstance(self.coordinate_space, CoordinateSpace):
            raise TypeError("coordinate_space must be a CoordinateSpace.")
        if not isinstance(self.status, DetectionState):
            raise TypeError("status must be a DetectionState.")
        if not isinstance(self.synthetic, bool):
            raise TypeError("synthetic must be a boolean.")
        if any(not isinstance(item, str) or not item.strip() for item in self.validation_errors):
            raise ValueError("validation_errors must contain non-empty strings.")

    @property
    def validation_status(self) -> DetectionState:
        return self.status


@dataclass(frozen=True)
class DetectionFrame:
    observed_at: float
    raw_detections: tuple[DetectionResult, ...]
    stable_detections: tuple[DetectionResult, ...]
    unverified_detections: tuple[DetectionObservation, ...] = ()
    rejected_count: int = 0
    diagnostics: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class DetectionConversion:
    detections: tuple[DetectionResult, ...]
    unverified: tuple[DetectionObservation, ...]
    diagnostics: tuple[tuple[str, int], ...]


class BoundingBoxConversionError(ValueError):
    """Raised when a detector box cannot be safely normalized."""


def normalize_bounding_box(
    coordinates: tuple[Any, Any, Any, Any],
    *,
    coordinate_space: CoordinateSpace,
    image_width: int,
    image_height: int,
    clamp_tolerance_pixels: float = 2.0,
) -> BoundingBox:
    if (
        isinstance(image_width, bool)
        or not isinstance(image_width, int)
        or isinstance(image_height, bool)
        or not isinstance(image_height, int)
        or image_width <= 0
        or image_height <= 0
    ):
        raise BoundingBoxConversionError("image dimensions must be positive integers.")
    if (
        isinstance(clamp_tolerance_pixels, bool)
        or not isinstance(clamp_tolerance_pixels, (int, float))
        or not isfinite(clamp_tolerance_pixels)
        or clamp_tolerance_pixels < 0
    ):
        raise BoundingBoxConversionError("clamp tolerance must be finite and nonnegative.")
    if not isinstance(coordinate_space, CoordinateSpace):
        raise BoundingBoxConversionError("coordinate_space must be a CoordinateSpace value.")
    if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 4:
        raise BoundingBoxConversionError("box coordinates must contain exactly four finite numbers.")
    try:
        numeric_coordinates = tuple(float(value) for value in coordinates)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BoundingBoxConversionError("box coordinates must be finite numbers.") from exc
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in coordinates
    ) or any(not isfinite(value) for value in numeric_coordinates):
        raise BoundingBoxConversionError("box coordinates must be finite numbers.")

    first, second, third, fourth = numeric_coordinates
    if coordinate_space in {CoordinateSpace.PIXEL_XYWH, CoordinateSpace.NORMALIZED_XYWH}:
        if third <= 0 or fourth <= 0:
            raise BoundingBoxConversionError("box width and height must be positive.")
        x_min, y_min, x_max, y_max = first, second, first + third, second + fourth
    else:
        x_min, y_min, x_max, y_max = first, second, third, fourth
        if x_min >= x_max or y_min >= y_max:
            raise BoundingBoxConversionError("box corners must have positive width and height.")

    if coordinate_space in {CoordinateSpace.PIXEL_XYWH, CoordinateSpace.PIXEL_XYXY}:
        x_tolerance = clamp_tolerance_pixels
        y_tolerance = clamp_tolerance_pixels
        x_min /= image_width
        x_max /= image_width
        y_min /= image_height
        y_max /= image_height
        x_tolerance /= image_width
        y_tolerance /= image_height
    else:
        x_tolerance = clamp_tolerance_pixels / image_width
        y_tolerance = clamp_tolerance_pixels / image_height

    if (
        x_min < -x_tolerance
        or y_min < -y_tolerance
        or x_max > 1.0 + x_tolerance
        or y_max > 1.0 + y_tolerance
    ):
        raise BoundingBoxConversionError("box exceeds frame bounds beyond clamp tolerance.")
    try:
        return BoundingBox(
            min(1.0, max(0.0, x_min)),
            min(1.0, max(0.0, y_min)),
            min(1.0, max(0.0, x_max)),
            min(1.0, max(0.0, y_max)),
        )
    except ValueError as exc:
        raise BoundingBoxConversionError(str(exc)) from exc


def convert_detector_results(
    detections: Iterable[Any],
    *,
    frame_id: int,
    timestamp: float,
    image_width: int,
    image_height: int,
    source_backend: str = "mediapipe",
    minimum_confidence: float = 0.35,
    supported_class_ids: set[int] | None = None,
    clamp_tolerance_pixels: float = 2.0,
) -> DetectionConversion:
    if (
        isinstance(minimum_confidence, bool)
        or not isinstance(minimum_confidence, (int, float))
        or not isfinite(minimum_confidence)
        or not 0.0 <= minimum_confidence <= 1.0
    ):
        raise ValueError("minimum_confidence must be finite and in [0, 1].")
    if isinstance(clamp_tolerance_pixels, bool) or not isinstance(
        clamp_tolerance_pixels, (int, float)
    ):
        raise TypeError("clamp_tolerance_pixels must be a number.")
    if not isfinite(clamp_tolerance_pixels) or clamp_tolerance_pixels < 0:
        raise ValueError("clamp_tolerance_pixels must be finite and nonnegative.")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("frame_id must be a nonnegative integer.")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not isfinite(timestamp)
    ):
        raise ValueError("timestamp must be finite.")
    if not isinstance(source_backend, str) or not source_backend.strip():
        raise ValueError("source_backend must not be empty.")
    accepted: list[DetectionResult] = []
    unverified: list[DetectionObservation] = []
    diagnostics: Counter[str] = Counter()

    def reject(
        index: int,
        reason: str,
        *,
        raw_class_id: float | str | None = None,
        normalized_class_id: int | None = None,
        label: str | None = None,
        confidence: float | None = None,
        box: BoundingBox | None = None,
        status: DetectionState = DetectionState.REJECTED,
        coordinate_space: CoordinateSpace = CoordinateSpace.PIXEL_XYWH,
    ) -> None:
        diagnostics[reason] += 1
        unverified.append(
            DetectionObservation(
                detection_id=f"{frame_id}:{index}",
                frame_id=frame_id,
                timestamp=timestamp,
                source_backend=source_backend,
                raw_class_id=raw_class_id,
                normalized_class_id=normalized_class_id,
                label=label,
                confidence=confidence,
                bounding_box=box,
                coordinate_space=coordinate_space,
                status=status,
                validation_errors=(reason,),
                raw_label=label,
            )
        )

    for index, detection in enumerate(detections):
        categories = getattr(detection, "categories", None)
        if not isinstance(categories, (list, tuple)):
            reject(index, "invalid_categories", status=DetectionState.MISSING_CLASS)
            continue
        if not categories:
            reject(index, "missing_categories", status=DetectionState.MISSING_CLASS)
            continue

        scored_categories: list[tuple[float, Any]] = []
        for category in categories:
            score = getattr(category, "score", None)
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                diagnostics["invalid_confidence_candidate"] += 1
                continue
            scored_categories.append((float(score), category))
        if not scored_categories:
            reject(index, "missing_or_invalid_confidence")
            continue

        confidence, category = max(scored_categories, key=lambda item: item[0])
        raw_class_id = getattr(category, "index", None)
        raw_class_id_value = (
            raw_class_id
            if isinstance(raw_class_id, (int, float, str)) and not isinstance(raw_class_id, bool)
            else None
        )
        category_name = getattr(category, "category_name", None)
        label = category_name.strip() or None if isinstance(category_name, str) else None
        raw_box = getattr(detection, "bounding_box", None)
        box: BoundingBox | None = None
        box_error: str | None = None
        if raw_box is None:
            box_error = "missing_bounding_box"
        else:
            try:
                box = normalize_bounding_box(
                    (
                        getattr(raw_box, "origin_x", None),
                        getattr(raw_box, "origin_y", None),
                        getattr(raw_box, "width", None),
                        getattr(raw_box, "height", None),
                    ),
                    coordinate_space=CoordinateSpace.PIXEL_XYWH,
                    image_width=image_width,
                    image_height=image_height,
                    clamp_tolerance_pixels=clamp_tolerance_pixels,
                )
            except BoundingBoxConversionError as exc:
                box_error = str(exc)

        class_id: int | None
        if isinstance(raw_class_id, int) and not isinstance(raw_class_id, bool):
            class_id = raw_class_id
        else:
            class_id = None

        if box_error is not None:
            reject(
                index,
                box_error,
                raw_class_id=raw_class_id_value,
                normalized_class_id=class_id,
                label=label,
                confidence=confidence,
                status=DetectionState.INVALID_BOX,
                coordinate_space=CoordinateSpace.PIXEL_XYWH,
            )
            continue
        if class_id is None:
            reject(
                index,
                "missing_class_id" if raw_class_id is None else "invalid_class_id",
                raw_class_id=raw_class_id_value,
                label=label,
                confidence=confidence,
                box=box,
                status=DetectionState.MISSING_CLASS,
            )
            continue
        if class_id < 0:
            reject(
                index,
                "invalid_class_id",
                raw_class_id=raw_class_id_value,
                normalized_class_id=class_id,
                label=label,
                confidence=confidence,
                box=box,
                status=DetectionState.REJECTED,
            )
            continue
        if not label or label.casefold() == "unknown":
            reject(
                index,
                "missing_class_label",
                raw_class_id=raw_class_id_value,
                normalized_class_id=class_id,
                label=label,
                confidence=confidence,
                box=box,
                status=DetectionState.UNKNOWN,
            )
            continue

        if confidence < minimum_confidence:
            reject(
                index,
                "low_confidence",
                raw_class_id=raw_class_id_value,
                normalized_class_id=class_id,
                label=label,
                confidence=confidence,
                box=box,
                status=DetectionState.LOW_CONFIDENCE,
            )
        elif supported_class_ids is not None and class_id not in supported_class_ids:
            reject(
                index,
                "unsupported_class",
                raw_class_id=raw_class_id_value,
                normalized_class_id=class_id,
                label=label,
                confidence=confidence,
                box=box,
                status=DetectionState.UNSUPPORTED_CLASS,
            )
        else:
            accepted.append(
                DetectionResult(
                    class_id,
                    label,
                    confidence,
                    box,
                    coordinate_space=CoordinateSpace.PIXEL_XYWH,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    source_backend=source_backend,
                    raw_class_id=raw_class_id_value,
                )
            )

    return DetectionConversion(
        tuple(accepted), tuple(unverified), tuple(sorted(diagnostics.items()))
    )


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
    first_seen_at: float
    coordinate_space: CoordinateSpace
    frame_id: int
    source_backend: str
    raw_class_id: int | float | str | None
    synthetic: bool


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
        if (
            isinstance(stable_hits, bool)
            or not isinstance(stable_hits, int)
            or stable_hits < 1
            or not isfinite(missing_tolerance_seconds)
            or missing_tolerance_seconds < 0
        ):
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
        self,
        raw_detections: Iterable[DetectionResult],
        *,
        observed_at: float,
        frame_id: int = 0,
        source_backend: str = "unknown",
        unverified_detections: Iterable[DetectionObservation] = (),
        diagnostics: Iterable[tuple[str, int]] = (),
    ) -> DetectionFrame:
        if (
            isinstance(observed_at, bool)
            or not isinstance(observed_at, (int, float))
            or not isfinite(observed_at)
        ) or (self._last_observed_at is not None and observed_at < self._last_observed_at):
            raise ValueError("observed_at must be finite and nondecreasing.")
        self._last_observed_at = observed_at
        raw = tuple(raw_detections)
        eligible: list[DetectionResult] = []
        rejected = 0
        unknown = list(unverified_detections)
        diagnostic_counts = Counter(dict(diagnostics))
        for detection in raw:
            threshold = self.class_thresholds.get(detection.class_id, self.minimum_confidence)
            if detection.state == DetectionState.UNSUPPORTED_CLASS or (
                self.supported_class_ids is not None
                and detection.class_id not in self.supported_class_ids
            ):
                unknown.append(
                    self._observation(
                        detection,
                        observed_at,
                        frame_id,
                        source_backend,
                        DetectionState.UNSUPPORTED_CLASS,
                    )
                )
            elif detection.confidence < threshold:
                unknown.append(
                    self._observation(
                        detection,
                        observed_at,
                        frame_id,
                        source_backend,
                        DetectionState.LOW_CONFIDENCE,
                    )
                )
            elif not self.minimum_area <= detection.box.area <= self.maximum_area:
                rejected += 1
                diagnostic_counts["box_area_rejected"] += 1
                unknown.append(
                    self._observation(
                        detection,
                        observed_at,
                        frame_id,
                        source_backend,
                        DetectionState.REJECTED,
                    )
                )
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
                track.coordinate_space = detection.coordinate_space
                track.frame_id = frame_id
                track.source_backend = source_backend
                track.raw_class_id = detection.raw_class_id
                track.synthetic = detection.synthetic
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
                    observed_at,
                    detection.coordinate_space,
                    frame_id,
                    source_backend,
                    detection.raw_class_id,
                    detection.synthetic,
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
                DetectionState.STABLE
                if track.hits >= self.stable_hits
                else DetectionState.CANDIDATE,
                track.track_id,
                track.last_seen,
                coordinate_space=track.coordinate_space,
                frame_id=track.frame_id,
                timestamp=track.last_seen,
                source_backend=track.source_backend,
                raw_class_id=track.raw_class_id,
                synthetic=track.synthetic,
            )
            for track in self._tracks
            if track.hits >= self.stable_hits
        )
        for track in self._tracks:
            if track.hits < self.stable_hits:
                unknown.append(
                    DetectionObservation(
                        detection_id=f"{frame_id}:{track.track_id}",
                        frame_id=frame_id,
                        timestamp=observed_at,
                        source_backend=source_backend,
                        raw_class_id=track.class_id,
                        normalized_class_id=track.class_id,
                        label=track.label,
                        confidence=track.confidence,
                        bounding_box=track.box,
                        status=DetectionState.UNSTABLE,
                        tracking_id=track.track_id,
                        coordinate_space=track.coordinate_space,
                        raw_label=track.label,
                        synthetic=track.synthetic,
                    )
                )
        for reason, count in tuple(diagnostic_counts.items()):
            diagnostic_counts[reason] = count
        return DetectionFrame(
            observed_at,
            raw,
            stable,
            tuple(unknown),
            rejected,
            tuple(sorted(diagnostic_counts.items())),
        )

    @staticmethod
    def _observation(
        detection: DetectionResult,
        observed_at: float,
        frame_id: int,
        source_backend: str,
        status: DetectionState,
    ) -> DetectionObservation:
        return DetectionObservation(
            detection_id=f"{frame_id}:{detection.class_id}",
            frame_id=frame_id,
            timestamp=observed_at,
            source_backend=source_backend,
            raw_class_id=detection.class_id,
            normalized_class_id=detection.class_id,
            label="unknown",
            confidence=detection.confidence,
            bounding_box=detection.box,
            coordinate_space=detection.coordinate_space,
            status=status,
            tracking_id=detection.track_id,
            synthetic=detection.synthetic,
            raw_label=detection.label,
        )


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
            name: (self.smoothing * value + (1 - self.smoothing) * previous.get(name, value))
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
