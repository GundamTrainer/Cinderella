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

## Serialization behavior

Events can be serialized to JSON and restored from JSON by round-tripping the dictionary representation. The dataclass structure and field semantics are preserved across serialization boundaries.
