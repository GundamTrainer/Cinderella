from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class UnknownEventTypeError(ValueError):
    """Raised when an event type is not part of the supported runtime schema."""


class EventType(StrEnum):
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPING = "system.stopping"
    AUDIO_SPEECH_STARTED = "audio.speech_started"
    AUDIO_SPEECH_ENDED = "audio.speech_ended"
    AUDIO_SOUND_DETECTED = "audio.sound_detected"
    SPEECH_PARTIAL_TRANSCRIPT = "speech.partial_transcript"
    SPEECH_FINAL_TRANSCRIPT = "speech.final_transcript"
    VISION_PERSON_DETECTED = "vision.person_detected"
    VISION_ACTION_DETECTED = "vision.action_detected"
    VISION_OBJECTS_DETECTED = "vision.objects_detected"
    VISION_EXPRESSION_CUES_DETECTED = "vision.expression_cues_detected"
    PLANNER_ACTION_PROPOSED = "planner.action_proposed"
    SAFETY_ACTION_APPROVED = "safety.action_approved"
    SAFETY_ACTION_REJECTED = "safety.action_rejected"
    HARDWARE_COMMAND_SIMULATED = "hardware.command_simulated"
    HARDWARE_EMERGENCY_STOP = "hardware.emergency_stop"


def _validate_payload(value: Any, *, path: str = "payload") -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a dictionary of JSON-serializable values.")

    result: dict[str, JSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{path} keys must be strings; got {type(key).__name__}.")
        result[key] = _validate_json_value(item, path=f"{path}.{key}")
    return result


def _validate_json_value(value: Any, *, path: str) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        validated: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings; got {type(key).__name__}.")
            validated[key] = _validate_json_value(item, path=f"{path}.{key}")
        return validated
    raise TypeError(
        f"{path} contains unsupported value of type {type(value).__name__}; "
        "only JSON-compatible values are allowed."
    )


@dataclass(frozen=True)
class Event:
    schema_version: str = "1.0"
    event_id: str | None = None
    event_type: EventType | str = EventType.SYSTEM_STARTED
    source: str = "runtime"
    created_at: str | None = None
    monotonic_time: float | None = None
    confidence: float = 1.0
    payload: dict[str, JSONValue] | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if self.event_id is None:
            object.__setattr__(self, "event_id", uuid.uuid4().hex)
        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.now(UTC).isoformat())
        if self.monotonic_time is None:
            object.__setattr__(self, "monotonic_time", time.monotonic())
        if self.payload is None:
            object.__setattr__(self, "payload", {})
        object.__setattr__(self, "payload", _validate_payload(self.payload, path="payload"))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0 inclusive.")
        if self.event_type is not None and not isinstance(self.event_type, EventType):
            try:
                object.__setattr__(self, "event_type", EventType(self.event_type))
            except ValueError as exc:
                raise UnknownEventTypeError(
                    f"Unknown event type: {self.event_type!r}. Supported types are {list(EventType)}."
                ) from exc
        if self.correlation_id is None:
            object.__setattr__(self, "correlation_id", self.event_id)

    @classmethod
    def create(
        cls,
        event_type: EventType | str,
        source: str,
        payload: dict[str, JSONValue] | None = None,
        *,
        confidence: float = 1.0,
        correlation_id: str | None = None,
        monotonic_time: float | None = None,
        event_id: str | None = None,
        created_at: str | None = None,
    ) -> Event:
        event_identifier = event_id or uuid.uuid4().hex
        resolved_correlation = correlation_id or event_identifier
        return cls(
            schema_version="1.0",
            event_id=event_identifier,
            event_type=event_type,
            source=source,
            created_at=created_at or datetime.now(UTC).isoformat(),
            monotonic_time=time.monotonic() if monotonic_time is None else monotonic_time,
            confidence=confidence,
            payload=payload or {},
            correlation_id=resolved_correlation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "source": self.source,
            "created_at": self.created_at,
            "monotonic_time": self.monotonic_time,
            "confidence": float(self.confidence),
            "payload": self.payload,
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Event:
        if "event_type" not in values:
            raise ValueError("Event payload is missing event_type.")
        event_type_value = values["event_type"]
        try:
            event_type = EventType(event_type_value)
        except ValueError as exc:
            raise UnknownEventTypeError(
                f"Unknown event type: {event_type_value!r}. Supported types are {list(EventType)}."
            ) from exc

        payload = values.get("payload", {})
        return cls(
            schema_version=values.get("schema_version", "1.0"),
            event_id=values["event_id"],
            event_type=event_type,
            source=values.get("source", "runtime"),
            created_at=values.get("created_at", datetime.now(UTC).isoformat()),
            monotonic_time=float(values.get("monotonic_time", time.monotonic())),
            confidence=float(values.get("confidence", 1.0)),
            payload=payload,
            correlation_id=values.get("correlation_id", values["event_id"]),
        )

    @classmethod
    def from_json(cls, json_text: str) -> Event:
        return cls.from_dict(json.loads(json_text))
