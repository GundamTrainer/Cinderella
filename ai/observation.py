from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ObservationType(StrEnum):
    PERSON_CANDIDATE = "person_candidate"
    OBJECT_CANDIDATE = "object_candidate"
    FACE_FEATURES = "face_features"
    BODY_POSE = "body_pose"
    HAND_POSE = "hand_pose"
    MOTION = "motion"
    SPEECH_SEGMENT = "speech_segment"
    TRANSCRIPT = "transcript"
    SOUND_EVENT = "sound_event"
    USER_TEXT = "user_text"
    SELF_REPORT = "self_report"
    SYSTEM_HEALTH = "system_health"


class ObservationValidity(StrEnum):
    VALID = "valid"
    LOW_CONFIDENCE = "low_confidence"
    INCOMPLETE = "incomplete"
    UNSTABLE = "unstable"
    INVALID = "invalid"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class UnknownObservationTypeError(ValueError):
    """Raised when an observation type is not in this schema version."""


def _validate_json(value: Any, path: str = "payload") -> JSONValue:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, list):
        return [_validate_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings.")
            result[key] = _validate_json(item, f"{path}.{key}")
        return result
    raise TypeError(f"{path} contains unsupported value of type {type(value).__name__}.")


@dataclass(frozen=True)
class Observation:
    observation_type: ObservationType
    source: str
    source_backend: str
    confidence: float
    validity: ObservationValidity
    payload: dict[str, JSONValue]
    observation_id: str = ""
    schema_version: str = "1.0"
    frame_id: int | None = None
    audio_sequence: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    monotonic_time: float = 0.0
    tracking_hint: str | None = None
    raw_reference: str | None = None
    synthetic: bool = False
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.observation_type, ObservationType):
            raise TypeError("observation_type must be an ObservationType.")
        if not isinstance(self.validity, ObservationValidity):
            raise TypeError("validity must be an ObservationValidity.")
        if self.observation_type in {
            ObservationType.PERSON_CANDIDATE,
            ObservationType.OBJECT_CANDIDATE,
            ObservationType.FACE_FEATURES,
            ObservationType.BODY_POSE,
            ObservationType.HAND_POSE,
            ObservationType.MOTION,
        } and self.frame_id is None:
            raise ValueError(f"{self.observation_type.value} requires frame_id.")
        if self.observation_type in {
            ObservationType.SPEECH_SEGMENT,
            ObservationType.TRANSCRIPT,
            ObservationType.SOUND_EVENT,
        } and self.audio_sequence is None:
            raise ValueError(f"{self.observation_type.value} requires audio_sequence.")
        for field_name in ("source", "source_backend", "schema_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
        if not self.observation_id:
            object.__setattr__(self, "observation_id", uuid.uuid4().hex)
        if not isinstance(self.observation_id, str) or not self.observation_id.strip():
            raise ValueError("observation_id must not be empty.")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must not be empty.")
        try:
            created_at = datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp.") from exc
        if created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone.")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1].")
        if (
            isinstance(self.monotonic_time, bool)
            or not isinstance(self.monotonic_time, (int, float))
            or not isfinite(self.monotonic_time)
            or self.monotonic_time < 0
        ):
            raise ValueError("monotonic_time must be finite and nonnegative.")
        for field_name in ("frame_id", "audio_sequence"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field_name} must be a nonnegative integer.")
        if not isinstance(self.synthetic, bool):
            raise TypeError("synthetic must be a boolean.")
        if self.raw_reference is not None and (
            not isinstance(self.raw_reference, str) or not self.raw_reference.strip()
        ):
            raise ValueError("raw_reference must be a non-empty reference, not sensor data.")
        for field_name in ("errors", "warnings"):
            if any(not isinstance(item, str) or not item.strip() for item in getattr(self, field_name)):
                raise ValueError(f"{field_name} must contain non-empty strings.")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a JSON object.")
        object.__setattr__(self, "payload", _validate_json(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "schema_version": self.schema_version,
            "observation_type": self.observation_type.value,
            "source": self.source,
            "source_backend": self.source_backend,
            "frame_id": self.frame_id,
            "audio_sequence": self.audio_sequence,
            "created_at": self.created_at,
            "monotonic_time": self.monotonic_time,
            "confidence": float(self.confidence),
            "validity": self.validity.value,
            "payload": self.payload,
            "tracking_hint": self.tracking_hint,
            "raw_reference": self.raw_reference,
            "synthetic": self.synthetic,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Observation:
        try:
            observation_type = ObservationType(values["observation_type"])
        except (KeyError, ValueError) as exc:
            raise UnknownObservationTypeError(
                f"Unknown observation type: {values.get('observation_type')!r}."
            ) from exc
        try:
            return cls(
                observation_type=observation_type,
                source=values["source"],
                source_backend=values["source_backend"],
                confidence=values["confidence"],
                validity=ObservationValidity(values["validity"]),
                payload=values["payload"],
                observation_id=values["observation_id"],
                schema_version=values["schema_version"],
                frame_id=values.get("frame_id"),
                audio_sequence=values.get("audio_sequence"),
                created_at=values["created_at"],
                monotonic_time=values["monotonic_time"],
                tracking_hint=values.get("tracking_hint"),
                raw_reference=values.get("raw_reference"),
                synthetic=values.get("synthetic", False),
                errors=tuple(values.get("errors", ())),
                warnings=tuple(values.get("warnings", ())),
            )
        except KeyError as exc:
            raise ValueError(f"Serialized observation is missing {exc.args[0]!r}.") from exc

    @classmethod
    def from_json(cls, encoded: str) -> Observation:
        value = json.loads(encoded)
        if not isinstance(value, dict):
            raise TypeError("Serialized observation must be a JSON object.")
        return cls.from_dict(value)
