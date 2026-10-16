from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from .vision import BoundingBox, DetectionFrame, DetectionState, FaceFeatureFrame


@dataclass(frozen=True)
class SceneObservation:
    frame_id: int
    observed_at: float
    detections: DetectionFrame
    action_cues: tuple[str, ...]
    face_features: FaceFeatureFrame


@dataclass(frozen=True)
class WorldObject:
    tracking_id: int
    label: str
    bounding_box: BoundingBox
    confidence: float
    region: str
    motion: str
    last_seen: float


@dataclass(frozen=True)
class WorldState:
    observed_at: float
    objects: tuple[WorldObject, ...]
    uncertain_labels: tuple[str, ...]


@dataclass(frozen=True)
class InterpretationHypothesis:
    hypothesis: str
    supporting_observations: tuple[str, ...]
    contradicting_observations: tuple[str, ...] = ()
    confidence: float = 0.0
    uncertainty: str = "unknown"
    valid_until: float | None = None
    requires_confirmation: bool = True


def tentative_affect_hypotheses(
    expression_cues: tuple[str, ...],
    *,
    observed_at: float,
    lifetime_seconds: float = 2.0,
) -> tuple[InterpretationHypothesis, ...]:
    """Create low-confidence, short-lived hypotheses from explicit visible cues only."""
    if not isfinite(observed_at) or not isfinite(lifetime_seconds) or lifetime_seconds <= 0:
        raise ValueError("hypothesis time and lifetime must be finite; lifetime must be positive.")

    cues = set(expression_cues)
    patterns = (
        ({"smile-like facial movement"}, "possible_positive_affect"),
        (
            {"eyebrow-raising movement", "eyes-widening movement"},
            "possible_surprise_like_response",
        ),
        (
            {"brow-lowering movement", "eyes-narrowing movement"},
            "possible_tension_like_response",
        ),
    )
    hypotheses = []
    for required_cues, label in patterns:
        if required_cues <= cues:
            hypotheses.append(
                InterpretationHypothesis(
                    hypothesis=label,
                    supporting_observations=tuple(sorted(required_cues)),
                    confidence=0.25,
                    uncertainty="high; heuristic and not calibrated",
                    valid_until=observed_at + lifetime_seconds,
                    requires_confirmation=True,
                )
            )
    return tuple(hypotheses)


class WorldStateTracker:
    """Build short-lived scene state from validated, temporally stable observations."""

    def __init__(
        self,
        *,
        object_expiration_seconds: float = 2.0,
        movement_threshold_per_second: float = 0.04,
    ) -> None:
        if not isfinite(object_expiration_seconds) or object_expiration_seconds <= 0:
            raise ValueError("object expiration must be positive.")
        if not isfinite(movement_threshold_per_second) or movement_threshold_per_second < 0:
            raise ValueError("movement threshold must be nonnegative.")
        self.object_expiration_seconds = object_expiration_seconds
        self.movement_threshold_per_second = movement_threshold_per_second
        self._objects: dict[int, WorldObject] = {}
        self._last_observed_at: float | None = None

    def reset(self) -> None:
        self._objects.clear()
        self._last_observed_at = None

    def update(self, observation: SceneObservation) -> WorldState:
        if (
            isinstance(observation.observed_at, bool)
            or not isinstance(observation.observed_at, (int, float))
            or not isfinite(observation.observed_at)
        ) or (
            self._last_observed_at is not None and observation.observed_at < self._last_observed_at
        ):
            raise ValueError("observation time must be finite and nondecreasing.")
        self._last_observed_at = observation.observed_at
        updated: dict[int, WorldObject] = {}
        for detection in observation.detections.stable_detections:
            if detection.state != DetectionState.STABLE or detection.track_id is None:
                continue
            previous = self._objects.get(detection.track_id)
            detection_time = (
                detection.last_seen if detection.last_seen is not None else observation.observed_at
            )
            motion = "unknown"
            if previous is not None and detection_time > previous.last_seen:
                elapsed = detection_time - previous.last_seen
                if elapsed > 0:
                    displacement = self._center_distance(previous.bounding_box, detection.box)
                    motion = (
                        "moving"
                        if displacement / elapsed >= self.movement_threshold_per_second
                        else "stationary"
                    )
            object_state = WorldObject(
                tracking_id=detection.track_id,
                label=detection.label,
                bounding_box=detection.box,
                confidence=detection.confidence,
                region=self._region(detection.box),
                motion=motion,
                last_seen=detection_time,
            )
            updated[detection.track_id] = object_state

        for track_id, previous in self._objects.items():
            if (
                track_id not in updated
                and observation.observed_at - previous.last_seen <= self.object_expiration_seconds
            ):
                updated[track_id] = previous
        self._objects = updated
        return WorldState(
            observed_at=observation.observed_at,
            objects=tuple(sorted(updated.values(), key=lambda item: item.tracking_id)),
            uncertain_labels=tuple(
                item.label or "unknown" for item in observation.detections.unverified_detections
            ),
        )

    @staticmethod
    def _center_distance(first: BoundingBox, second: BoundingBox) -> float:
        first_x = (first.x_min + first.x_max) / 2
        first_y = (first.y_min + first.y_max) / 2
        second_x = (second.x_min + second.x_max) / 2
        second_y = (second.y_min + second.y_max) / 2
        return hypot(second_x - first_x, second_y - first_y)

    @staticmethod
    def _region(box: BoundingBox) -> str:
        center_x = (box.x_min + box.x_max) / 2
        center_y = (box.y_min + box.y_max) / 2
        horizontal = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
        vertical = "upper" if center_y < 1 / 3 else "lower" if center_y > 2 / 3 else "middle"
        return f"{vertical}-{horizontal}"
