# Architecture overview

## Scope

This repository currently implements a minimal runtime foundation for a multimodal character agent. The design deliberately separates the runtime from actual model and hardware integrations so that future components can plug into a deterministic and testable base layer.

## Runtime layers

### Configuration

The configuration layer loads a typed runtime settings object from a small YAML subset file and validates the values before use. This avoids hidden third-party dependencies while keeping the settings explicit and testable.

### Event model

The event layer uses a dataclass-based Event model with a strict Enum for event types and a JSON-compatible payload validator. Unknown event types are rejected rather than silently accepted so that the runtime remains predictable.

### Event bus

The event bus is a bounded, thread-safe queue. It exposes publish, poll, drain, close, and size operations. Full queue state is treated as an explicit error, and close semantics are enforced to avoid accidental writes after shutdown.

### Components

Long-running components implement a shared lifecycle interface: initialize, start, stop, close, and health. State transitions are validation-based so invalid sequences raise a clear error.

### Simulation layer

Simulated audio, vision, and hardware components emit synthetic events only. They do not capture or store raw audio, images, or personal data.

### Local device preview

An opt-in command-line preview can display a camera feed and a microphone level meter using optional OpenCV and sounddevice dependencies. It is a local connectivity check only: it does not publish model-recognition events, run inference, or save recordings.

### Local AI inference

The opt-in AI mode uses faster-whisper (CPU/int8) for Korean transcription and MediaPipe Tasks for EfficientDet Lite2 object detection, pose landmarks, and temporally smoothed face blendshape movement cues. Regular conversation uses a small local Ollama text model. Explicit visual questions additionally use Qwen2.5-VL with a temporary JPEG of the current frame over loopback; images are not added to retained chat history. The installed Windows Korean SAPI voice speaks replies offline. A scheduler can initiate short conversations after a quiet interval or stable visual cue; it does not claim to infer emotion. Basic barge-in detection listens for sustained microphone audio above an estimated speaker-echo level. First-use model downloads are cached locally and are not repository assets.

### Orchestrator

The orchestrator is responsible for initializing components, starting them, consuming the event bus, and safely tearing down components in reverse order on shutdown or partial initialization failure.

## Planned functionality

The following areas are intentionally not implemented yet and are explicitly called out as future work:
- robust generalized action recognition and reliable emotional-state inference
- hardware adapters and embedded device drivers
- user-specific voice or character profiles
- external cloud services or internet-connected features
