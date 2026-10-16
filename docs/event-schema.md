# Event schema

## Schema version

The current event schema version is `1.0`.

## Required fields

Every event instance includes the following fields:
- schema_version
- event_id
- event_type
- source
- created_at
- monotonic_time
- confidence
- payload
- correlation_id

## Event types

Supported event types:
- system.started
- system.stopping
- audio.speech_started
- audio.speech_ended
- audio.sound_detected
- speech.partial_transcript
- speech.final_transcript
- vision.person_detected
- vision.action_detected
- vision.objects_detected
- vision.expression_cues_detected
- planner.action_proposed
- safety.action_approved
- safety.action_rejected
- hardware.command_simulated
- hardware.emergency_stop

## Payload rules

Payload values are restricted to JSON-compatible primitives and nested dictionaries/lists. Arbitrary Python objects are rejected at validation time. This prevents uncontrolled memory growth and keeps events portable across serialization boundaries.

## Confidence rules

The `confidence` field must be a float in the inclusive range 0.0 to 1.0. Values outside that range raise a clear exception.

## ID semantics

- event_id identifies one specific event occurrence.
- correlation_id links multiple events that belong to the same causal flow or request.

These values are distinct so the runtime can correlate related events without losing the uniqueness of each individual event.

## Unknown event types

Unknown event types are rejected with a specific error. The runtime does not silently accept unrecognized event names because that would bypass the schema contract and make debugging unreliable.

The opt-in local AI runtime uses `speech.final_transcript`, `vision.objects_detected`,
`vision.action_detected`, and `vision.expression_cues_detected` for transient inference results.
These events are not persisted by the runtime.

For `vision.objects_detected`, `payload.objects` contains confirmed labels after temporal
stabilization. `payload.detections` contains those detections with class ID, label, confidence,
track ID, and normalized `bbox` coordinates (`x_min`, `y_min`, `x_max`, `y_max`). Raw and
unverified detector proposals remain available in the in-process `DetectionFrame`; they are not
reported as confirmed object labels.

## Serialization behavior

Events can be serialized to JSON and restored from JSON by round-tripping the dictionary representation. The dataclass structure and field semantics are preserved across serialization boundaries.
