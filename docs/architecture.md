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

The opt-in AI mode uses faster-whisper (CPU/int8) for multilingual transcription and MediaPipe Tasks for object detection, pose landmarks, and face blendshape cues. Object, pose, expression, and transcript outputs are published as ephemeral events. A loopback-only Ollama client uses a small local chat model; it receives transcript text and abstract visual cues, never raw media. First-use model downloads are cached under the user's home cache directory and are not repository assets.

### Orchestrator

The orchestrator is responsible for initializing components, starting them, consuming the event bus, and safely tearing down components in reverse order on shutdown or partial initialization failure.

## Planned functionality

The following areas are intentionally not implemented yet and are explicitly called out as future work:
- speech synthesis
- generalized action recognition and reliable emotional-state inference
- hardware adapters and embedded device drivers
- user-specific voice or character profiles
- external cloud services or internet-connected features
