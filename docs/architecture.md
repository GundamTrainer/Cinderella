# Architecture overview

## Scope

This repository currently implements a minimal runtime foundation for a multimodal character agent. The design deliberately separates the runtime from actual model and hardware integrations so that future components can plug into a deterministic and testable base layer.

## Runtime layers

### Configuration

The configuration layer loads a typed runtime settings object from a small YAML subset file and validates the values before use. This avoids hidden third-party dependencies while keeping the settings explicit and testable.

### Event model

The event layer uses a dataclass-based Event model with a strict Enum for event types and a JSON-compatible payload validator. Unknown event types are rejected rather than silently accepted so that the runtime remains predictable.

### Event bus

The event bus is a bounded, thread-safe queue. It exposes publish, poll, drain, close, and size operations. Full queue state is treated as an explicit error, and close semantics are enforced to avoid accidental writes after shutdown. The live publisher uses the configured finite `event_publish_timeout`; events that still cannot be queued are explicitly logged as dropped.

### Components

Long-running components implement a shared lifecycle interface: initialize, start, stop, close, and health. State transitions are validation-based so invalid sequences raise a clear error.

### Simulation layer

Simulated audio, vision, and hardware components emit synthetic events only. They do not capture or store raw audio, images, or personal data.

### Local device preview

An opt-in command-line preview can display a camera feed and a microphone level meter using optional OpenCV and sounddevice dependencies. It is a local connectivity check only: it does not publish model-recognition events, run inference, or save recordings.

### Local AI inference and migration boundary

The opt-in AI mode calls faster-whisper (CPU/int8) through an external-backend adapter and MediaPipe Tasks through an adapter for EfficientDet Lite2 object detection, pose landmarks, and face blendshape features. Bounding boxes are normalized, confidence/area filtered, and tracked across frames; only repeated detections are exposed as stable object labels, while low-confidence and unsupported classes remain explicitly unverified. Face blendshape measurements are normalized and smoothed before visible movement cues are exposed. These stages improve consistency, not the underlying detector's COCO vocabulary or accuracy. Regular conversation uses a small local Ollama text model through its adapter; microphone speech and terminal text input create chat turns. Explicit visual questions additionally use Qwen2.5-VL with a temporary JPEG over loopback; images are not added to retained chat history. The installed Windows Korean SAPI voice speaks replies offline through an adapter. A scheduler can initiate short conversations after a quiet interval or stable visual cue; it does not infer emotion. Basic barge-in detection listens for sustained microphone audio above an estimated speaker-echo level. First-use model downloads are cached locally and are not repository assets. These adapters label existing third-party implementations as external; they are not self-developed models.

`ai/vision.py` contains reusable typed detection/face feature models and temporal stabilizers. `ai/observation.py` defines a JSON-safe, typed observation envelope that rejects unknown observation kinds and non-JSON sensor data. Live vision attaches normalized object summaries to that envelope; raw frames are never included. `datasets/object_detection/schema.py` validates annotation manifests without loading or storing media. `ai/evaluation.py` currently implements a small single-frame, class-aware IoU metric helper; it is not a full dataset benchmark.

### Perception observations and audio diagnostics

MediaPipe detections are converted into validated observations before entering the tracker. `Category.index` is optional, so a missing class ID, label, confidence, or box remains an unverified observation rather than being coerced into a confirmed label. Pixel `xywh` boxes are normalized against the actual frame size; coordinates may be clamped only within the configured pixel tolerance (default: 2 px), while malformed geometry and boxes farther outside the frame are rejected. Repeated detector issues are counted by reason and logged at increasing count milestones without logging media or feature data.

Confirmed object labels require valid class metadata, supported class IDs, confidence and area thresholds, and a repeated overlapping detection. A detection retains its source coordinate space separately from its normalized bounding box. The tracker retains only short-lived validated state; `WorldStateTracker` records coarse image regions, motion, bounded position history, first/last-seen times and temporary missing frames. Unverified labels persist briefly across empty frames (two seconds by default), then expire. World state can carry explicitly supplied beliefs and hypotheses, but observations do not automatically become confirmed beliefs.

Microphone callbacks hand copied sample chunks to a bounded queue configured by `audio_input_queue_capacity` (default: 24). Device overflow, dropped chunks, queue pressure, transcription acceptance/rejection, and TTS echo-guard discards are accumulated as counts and published as typed events. Sequence gaps and overflow markers prevent audio on opposite sides of a gap from being combined into one transcription window. Diagnostic event payloads contain counts and reason codes only, not samples or transcripts. The live transcription window remains fixed-length and is not flushed on shutdown.

The physical-AI simulator rejects movement proposals when the world-state timestamp is invalid, in the future, or older than its configured freshness window. This is a simulation policy only; it is not a hardware safety controller.

The former 54% estimate has been removed from the normal live overlay. A developer-only helper remains for compatibility with existing tests; `runtime.capabilities.capability_statuses()` is the preferred provenance/status view. Its static entries are not live health checks. Refer to [physical-ai.md](physical-ai.md) for the simulation-only action boundary and [evaluation.md](evaluation.md) for actual quality measurement.

### Orchestrator

The orchestrator is responsible for initializing components, starting them, consuming the event bus, and safely tearing down components in reverse order on shutdown or partial initialization failure.

### Integrations and companion state

`integrations/` provides provider contracts, default-deny metadata validation, four network modes, exact HTTPS host allowlists, redirect revalidation, response-size caps, bounded retries, and a bounded in-memory cache. Registration does not discover providers or make requests. The CLI and live conversation do not instantiate the registry.

MediaWiki and Open-Meteo adapters use injectable JSON transports, so unit tests do not use live network access. MediaWiki remains blocked until query-data retention is confirmed. Open-Meteo is blocked by the paid-service policy. Local status and caller-supplied approved records work without network.

`runtime/companion_state.py` defines identity, relationship, goal, memory, and response-plan models. `runtime/state_store.py` provides explicit local SQLite persistence; it writes nothing unless called and requires explicit consent before persisting memory. These models are not yet connected to the chat loop, automatically learned, or executed as autonomous actions.

External backend construction is centralized in `runtime/backend_factory.py`, but legacy model implementations and Ollama HTTP code remain in `runtime/ai_pipeline.py`. Backend isolation is partial, not complete; see [external-backends.md](external-backends.md).

## Planned functionality

The following areas are intentionally not implemented yet and are explicitly called out as future work:
- replacing each external model implementation with a trained internal backend and validating it against a versioned benchmark
- dataset acquisition/labeling tooling, training code, checkpoint lifecycle, and runtime backend selection configuration
- robust generalized action recognition and reliable emotional-state inference
- hardware adapters and embedded device drivers
- user-specific voice or character profiles
- external cloud services or internet-connected features
