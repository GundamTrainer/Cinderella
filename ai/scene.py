from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import hypot, isfinite
from threading import RLock

from .observation import Observation
from .vision import (
    BoundingBox,
    DetectionFrame,
    DetectionResult,
    DetectionState,
    FaceFeatureFrame,
)


@dataclass(frozen=True)
class SceneObservation:
    frame_id: int
    observed_at: float
    detections: DetectionFrame
    action_cues: tuple[str, ...]
    face_features: FaceFeatureFrame
    observations: tuple[Observation, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer.")
        if (
            isinstance(self.observed_at, bool)
            or not isinstance(self.observed_at, (int, float))
            or not isfinite(self.observed_at)
        ):
            raise ValueError("observed_at must be finite.")
        if not isinstance(self.detections, DetectionFrame):
            raise TypeError("detections must be a DetectionFrame.")
        if not isinstance(self.face_features, FaceFeatureFrame):
            raise TypeError("face_features must be a FaceFeatureFrame.")
        if not isinstance(self.observations, tuple) or any(
            not isinstance(item, Observation) for item in self.observations
        ):
            raise TypeError("observations must be a tuple of Observation values.")


@dataclass(frozen=True)
class WorldObject:
    tracking_id: int
    label: str
    bounding_box: BoundingBox
    confidence: float
    region: str
    motion: str
    last_seen: float
    first_seen_at: float = 0.0
    observation_count: int = 1
    missing_frame_count: int = 0
    class_candidates: tuple[tuple[str, float], ...] = ()
    tracking_confidence: float = 0.0
    lifecycle_state: str = "active"
    expiration_time: float | None = None
    position_history: tuple[tuple[float, float], ...] = ()
    size_history: tuple[tuple[float, float], ...] = ()
    velocity_estimate: tuple[float, float] | None = None

    @property
    def stable_label(self) -> str:
        return self.label

    @property
    def stable_label_confidence(self) -> float:
        return self.confidence

    @property
    def visibility(self) -> str:
        return "visible" if self.lifecycle_state == "active" else "temporarily_missing"


@dataclass(frozen=True)
class WorldState:
    observed_at: float
    objects: tuple[WorldObject, ...]
    uncertain_labels: tuple[str, ...]
    active_beliefs: tuple[Belief, ...] = ()
    active_hypotheses: tuple[InterpretationHypothesis, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.observed_at, bool)
            or not isinstance(self.observed_at, (int, float))
            or not isfinite(self.observed_at)
        ):
            raise ValueError("world-state time must be finite.")
        if not isinstance(self.objects, tuple) or not isinstance(self.uncertain_labels, tuple):
            raise TypeError("world-state collections must be immutable tuples.")
        if not isinstance(self.active_beliefs, tuple) or not isinstance(
            self.active_hypotheses, tuple
        ):
            raise TypeError("world-state evidence collections must be immutable tuples.")
        if any(not isinstance(item, WorldObject) for item in self.objects):
            raise TypeError("world-state objects must be WorldObject values.")
        if any(not isinstance(label, str) or not label.strip() for label in self.uncertain_labels):
            raise ValueError("uncertain labels must be non-empty strings.")
        if any(not isinstance(item, Belief) for item in self.active_beliefs):
            raise TypeError("active_beliefs must contain Belief values.")
        if any(not isinstance(item, InterpretationHypothesis) for item in self.active_hypotheses):
            raise TypeError("active_hypotheses must contain InterpretationHypothesis values.")

    @property
    def timestamp(self) -> float:
        return self.observed_at

    @property
    def tracked_objects(self) -> tuple[WorldObject, ...]:
        return self.objects


class ConfirmationStatus(StrEnum):
    UNCONFIRMED = "unconfirmed"
    PROVISIONAL = "provisional"
    USER_CONFIRMED = "user_confirmed"
    EXTERNALLY_SUPPORTED = "externally_supported"
    CONTRADICTED = "contradicted"
    EXPIRED = "expired"


@dataclass(frozen=True)
class Belief:
    belief_id: str
    subject: str
    predicate: str
    value: str
    source_type: str
    source_references: tuple[str, ...]
    confidence: float
    created_at: float
    expires_at: float | None = None
    confirmation_status: ConfirmationStatus = ConfirmationStatus.UNCONFIRMED
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.belief_id,
                self.subject,
                self.predicate,
                self.value,
                self.source_type,
            )
        ):
            raise ValueError("belief identifiers and labels must not be empty.")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("belief confidence must be finite and in [0, 1].")
        if (
            isinstance(self.created_at, bool)
            or not isinstance(self.created_at, (int, float))
            or not isfinite(self.created_at)
        ):
            raise ValueError("belief created_at must be finite.")
        if self.expires_at is not None and (
            isinstance(self.expires_at, bool)
            or not isinstance(self.expires_at, (int, float))
            or not isfinite(self.expires_at)
            or self.expires_at < self.created_at
        ):
            raise ValueError("belief expires_at must be finite and no earlier than created_at.")
        if not isinstance(self.confirmation_status, ConfirmationStatus):
            raise TypeError("confirmation_status must be a ConfirmationStatus.")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (*self.source_references, *self.supporting_evidence, *self.contradicting_evidence)
        ):
            raise ValueError("belief evidence references must be non-empty strings.")


@dataclass(frozen=True)
class InterpretationHypothesis:
    hypothesis: str
    supporting_observations: tuple[str, ...]
    contradicting_observations: tuple[str, ...] = ()
    confidence: float = 0.0
    uncertainty: str = "unknown"
    valid_until: float | None = None
    requires_confirmation: bool = True
    hypothesis_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_types: tuple[str, ...] = ("unspecified",)
    sensitive: bool = False
    allowed_for_speech: bool = False
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hypothesis_id, str)
            or not self.hypothesis_id.strip()
            or not isinstance(self.hypothesis, str)
            or not self.hypothesis.strip()
        ):
            raise ValueError("hypothesis_id and hypothesis label must not be empty.")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("hypothesis confidence must be finite and between 0 and 1.")
        if self.sensitive and self.allowed_for_speech:
            raise ValueError("Sensitive hypotheses cannot be allowed for speech.")
        if (
            isinstance(self.created_at, bool)
            or not isinstance(self.created_at, (int, float))
            or not isfinite(self.created_at)
        ):
            raise ValueError("hypothesis created_at must be finite.")
        if self.valid_until is not None and (
            isinstance(self.valid_until, bool)
            or not isinstance(self.valid_until, (int, float))
            or not isfinite(self.valid_until)
            or self.valid_until < self.created_at
        ):
            raise ValueError("hypothesis valid_until must be finite and no earlier than created_at.")
        if not isinstance(self.allowed_for_speech, bool) or not isinstance(self.sensitive, bool):
            raise TypeError("hypothesis policy flags must be booleans.")

    @property
    def label(self) -> str:
        return self.hypothesis


def tentative_affect_hypotheses(
    expression_cues: tuple[str, ...],
    *,
    observed_at: float,
    lifetime_seconds: float = 2.0,
    corroborating_source_types: tuple[str, ...] = (),
) -> tuple[InterpretationHypothesis, ...]:
    """Create tentative cues only when independent non-facial evidence is present."""
    if not isfinite(observed_at) or not isfinite(lifetime_seconds) or lifetime_seconds <= 0:
        raise ValueError("hypothesis time and lifetime must be finite; lifetime must be positive.")
    allowed_sources = {"speech_prosody", "body_pose", "motion", "self_report", "context"}
    independent_sources = tuple(
        source for source in corroborating_source_types if source in allowed_sources
    )
    if not independent_sources:
        return ()

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
                    source_types=("visual_facial_movement", *independent_sources),
                    allowed_for_speech=False,
                    created_at=observed_at,
                )
            )
    return tuple(hypotheses)


class WorldStateTracker:
    """Build short-lived scene state from validated, temporally stable observations."""

    def __init__(
        self,
        *,
        object_expiration_seconds: float = 2.0,
        uncertainty_expiration_seconds: float = 2.0,
        movement_threshold_per_second: float = 0.04,
    ) -> None:
        if not isfinite(object_expiration_seconds) or object_expiration_seconds <= 0:
            raise ValueError("object expiration must be positive.")
        if not isfinite(uncertainty_expiration_seconds) or uncertainty_expiration_seconds <= 0:
            raise ValueError("uncertainty expiration must be positive.")
        if not isfinite(movement_threshold_per_second) or movement_threshold_per_second < 0:
            raise ValueError("movement threshold must be nonnegative.")
        self.object_expiration_seconds = object_expiration_seconds
        self.uncertainty_expiration_seconds = uncertainty_expiration_seconds
        self.movement_threshold_per_second = movement_threshold_per_second
        self._objects: dict[int, WorldObject] = {}
        self._uncertain_labels: dict[str, float] = {}
        self._beliefs: dict[str, Belief] = {}
        self._hypotheses: dict[str, InterpretationHypothesis] = {}
        self._last_observed_at: float | None = None
        self._lock = RLock()
        self._latest_state = WorldState(0.0, (), ())

    def reset(self) -> None:
        with self._lock:
            self._objects.clear()
            self._uncertain_labels.clear()
            self._beliefs.clear()
            self._hypotheses.clear()
            self._last_observed_at = None
            self._latest_state = WorldState(0.0, (), ())

    def add_belief(self, belief: Belief) -> None:
        if not isinstance(belief, Belief):
            raise TypeError("belief must be a Belief.")
        with self._lock:
            self._beliefs[belief.belief_id] = belief
            self._refresh_snapshot()

    def add_hypothesis(self, hypothesis: InterpretationHypothesis) -> None:
        if not isinstance(hypothesis, InterpretationHypothesis):
            raise TypeError("hypothesis must be an InterpretationHypothesis.")
        with self._lock:
            self._hypotheses[hypothesis.hypothesis_id] = hypothesis
            self._refresh_snapshot()

    def snapshot(self) -> WorldState:
        with self._lock:
            return self._latest_state

    def update(self, observation: SceneObservation) -> WorldState:
        with self._lock:
            return self._update_locked(observation)

    def _update_locked(self, observation: SceneObservation) -> WorldState:
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
            if detection_time > observation.observed_at:
                raise ValueError("detection last_seen cannot be later than its observation.")
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
                first_seen_at=(
                    previous.first_seen_at if previous is not None else detection_time
                ),
                observation_count=(
                    previous.observation_count + 1 if previous is not None else 1
                ),
                class_candidates=self._updated_class_candidates(previous, detection),
                tracking_confidence=detection.confidence,
                expiration_time=detection_time + self.object_expiration_seconds,
                position_history=(
                    (*previous.position_history, self._center(detection.box))[-8:]
                    if previous is not None
                    else (self._center(detection.box),)
                ),
                size_history=(
                    (*previous.size_history, self._size(detection.box))[-8:]
                    if previous is not None
                    else (self._size(detection.box),)
                ),
                velocity_estimate=(
                    self._velocity(previous, detection.box, detection_time)
                    if previous is not None
                    else None
                ),
            )
            updated[detection.track_id] = object_state

        for track_id, previous in self._objects.items():
            if (
                track_id not in updated
                and observation.observed_at - previous.last_seen <= self.object_expiration_seconds
            ):
                updated[track_id] = replace(
                    previous,
                    missing_frame_count=previous.missing_frame_count + 1,
                    lifecycle_state="temporarily_missing",
                )
        self._objects = updated
        for item in observation.detections.unverified_detections:
            label = item.label.strip() if item.label else "unknown"
            self._uncertain_labels[label] = observation.observed_at
        self._uncertain_labels = {
            label: last_seen
            for label, last_seen in self._uncertain_labels.items()
            if observation.observed_at - last_seen <= self.uncertainty_expiration_seconds
        }
        self._beliefs = {
            belief_id: belief
            for belief_id, belief in self._beliefs.items()
            if belief.expires_at is None or observation.observed_at <= belief.expires_at
        }
        self._hypotheses = {
            hypothesis_id: hypothesis
            for hypothesis_id, hypothesis in self._hypotheses.items()
            if hypothesis.valid_until is None or observation.observed_at <= hypothesis.valid_until
        }
        self._latest_state = WorldState(
            observed_at=observation.observed_at,
            objects=tuple(sorted(updated.values(), key=lambda item: item.tracking_id)),
            uncertain_labels=tuple(sorted(self._uncertain_labels)),
            active_beliefs=tuple(self._beliefs.values()),
            active_hypotheses=tuple(self._hypotheses.values()),
        )
        return self._latest_state

    def _refresh_snapshot(self) -> None:
        self._latest_state = WorldState(
            observed_at=self._latest_state.observed_at,
            objects=tuple(sorted(self._objects.values(), key=lambda item: item.tracking_id)),
            uncertain_labels=tuple(sorted(self._uncertain_labels)),
            active_beliefs=tuple(self._beliefs.values()),
            active_hypotheses=tuple(self._hypotheses.values()),
        )

    @staticmethod
    def _center_distance(first: BoundingBox, second: BoundingBox) -> float:
        first_x = (first.x_min + first.x_max) / 2
        first_y = (first.y_min + first.y_max) / 2
        second_x = (second.x_min + second.x_max) / 2
        second_y = (second.y_min + second.y_max) / 2
        return hypot(second_x - first_x, second_y - first_y)

    @staticmethod
    def _updated_class_candidates(
        previous: WorldObject | None,
        detection: DetectionResult,
    ) -> tuple[tuple[str, float], ...]:
        candidates = dict(previous.class_candidates) if previous is not None else {}
        prior_count = previous.observation_count if previous is not None else 0
        if detection.label in candidates and prior_count:
            candidates[detection.label] = (
                candidates[detection.label] * prior_count + detection.confidence
            ) / (prior_count + 1)
        else:
            candidates[detection.label] = detection.confidence
        return tuple(sorted(candidates.items(), key=lambda item: (-item[1], item[0])))

    @staticmethod
    def _center(box: BoundingBox) -> tuple[float, float]:
        return ((box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2)

    @staticmethod
    def _size(box: BoundingBox) -> tuple[float, float]:
        return (box.x_max - box.x_min, box.y_max - box.y_min)

    @classmethod
    def _velocity(
        cls,
        previous: WorldObject,
        current_box: BoundingBox,
        current_time: float,
    ) -> tuple[float, float] | None:
        elapsed = current_time - previous.last_seen
        if elapsed <= 0 or not previous.position_history:
            return None
        previous_x, previous_y = previous.position_history[-1]
        current_x, current_y = cls._center(current_box)
        return ((current_x - previous_x) / elapsed, (current_y - previous_y) / elapsed)

    @staticmethod
    def _region(box: BoundingBox) -> str:
        center_x = (box.x_min + box.x_max) / 2
        center_y = (box.y_min + box.y_max) / 2
        horizontal = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
        vertical = "upper" if center_y < 1 / 3 else "lower" if center_y > 2 / 3 else "middle"
        return f"{vertical}-{horizontal}"
