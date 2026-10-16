# Observation and temporal world state

## Implemented

`ai.observation.Observation` is the common typed envelope for a sensor or model result. It carries an observation kind, source/backend, frame or audio sequence reference, wall-clock and monotonic timestamps, bounded confidence, validity, a JSON-safe summary payload, optional tracking hint, synthetic marker, and diagnostic strings. Unknown kinds are rejected. Payloads reject raw bytes and values that are not JSON-safe; callers should store references and summaries, never raw frames or PCM.

The live MediaPipe object path currently creates observations for stable detections and unverified candidates. Stable labels are emitted only after supported class metadata, confidence/area checks, and repeated overlapping observations. A detection's `coordinate_space` records the source convention; its `BoundingBox` is always normalized `xyxy`. Unsupported, invalid, missing-class, low-confidence, and unstable data remain separate from confirmed detections.

`WorldStateTracker` keeps immutable snapshots of short-lived stable tracks and uncertain labels. Tracks record first/last seen times, counts, temporary-missing frames, position/size history, and an estimated image-space velocity. Uncertainty is retained briefly across empty frames, then expires. Beliefs and hypotheses may be explicitly added; they are not automatically created from every observation, and beliefs are never promoted to confirmed status by the tracker.

The simulator's motion gate requires a finite, non-future world-state timestamp within a one-second default freshness window. The clock can be injected for deterministic testing. This remains a simulation-only policy, not hardware control or a certified safety mechanism.

## Not implemented

- Generic audio, transcript, pose, hand, sound, and self-report observations are not yet wired into the live pipeline.
- Tracking is limited to the existing class-aware IoU stabilizer; there is no occlusion-aware assignment, velocity model for identity matching, or user-confirmed labels.
- Belief contradiction resolution, source reconciliation, persistence, expiry notifications, and automatic correction are not implemented.
- No calibrated multimodal affect model exists. Face movement alone creates no hypothesis; the optional heuristic requires independent evidence and is not connected to live speech.

See [architecture.md](architecture.md), [state-understanding.md](state-understanding.md), and [physical-ai.md](physical-ai.md) for the broader limits and simulation boundary.
