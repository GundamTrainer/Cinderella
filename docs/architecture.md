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

### Local AI inference and migration boundary

The opt-in AI mode calls faster-whisper (CPU/int8) through an external-backend adapter and MediaPipe Tasks through an adapter for EfficientDet Lite2 object detection, pose landmarks, and face blendshape features. Bounding boxes are normalized, confidence/area filtered, and tracked across frames; only repeated detections are exposed as stable object labels, while low-confidence and unsupported classes remain explicitly unverified. Face blendshape measurements are normalized and smoothed before visible movement cues are exposed. These stages improve consistency, not the underlying detector's COCO vocabulary or accuracy. Regular conversation uses a small local Ollama text model through its adapter; microphone speech and terminal text input create chat turns. Explicit visual questions additionally use Qwen2.5-VL with a temporary JPEG over loopback; images are not added to retained chat history. The installed Windows Korean SAPI voice speaks replies offline through an adapter. A scheduler can initiate short conversations after a quiet interval or stable visual cue; it does not infer emotion. Basic barge-in detection listens for sustained microphone audio above an estimated speaker-echo level. First-use model downloads are cached locally and are not repository assets. These adapters label existing third-party implementations as external; they are not self-developed models.

`ai/vision.py` contains reusable typed detection/face feature models and temporal stabilizers. `datasets/object_detection/schema.py` validates annotation manifests without loading or storing media. `ai/evaluation.py` currently implements a small single-frame, class-aware IoU metric helper; it is not a full dataset benchmark.

### Orchestrator

The orchestrator is responsible for initializing components, starting them, consuming the event bus, and safely tearing down components in reverse order on shutdown or partial initialization failure.

## Planned functionality

The following areas are intentionally not implemented yet and are explicitly called out as future work:
- replacing each external model implementation with a trained internal backend and validating it against a versioned benchmark
- dataset acquisition/labeling tooling, training code, checkpoint lifecycle, and runtime backend selection configuration
- robust generalized action recognition and reliable emotional-state inference
- hardware adapters and embedded device drivers
- user-specific voice or character profiles
- external cloud services or internet-connected features
