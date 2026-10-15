# Cinderella

Cinderella is a self-developed multimodal character AI project. The current implementation focuses on the runtime foundation needed before any actual model, microphone, camera, or hardware integrations are introduced.

## Current implementation status

Implemented:
- typed runtime configuration loader with validation
- event schema and JSON serialization
- thread-safe bounded event bus
- lifecycle state management for long-running components
- deterministic simulation components for audio, vision, and hardware
- runtime orchestrator with cleanup on partial initialization failure
- pytest coverage for the runtime foundation

Not implemented yet:
- actual AI model inference
- microphone input capture
- camera capture
- hardware device drivers
- voice cloning or character voice profiles
- internet connectivity or external API features

## Runtime foundation

The runtime is intentionally minimal and deterministic. It provides the infrastructure that future multimodal components can reuse without depending on external hardware or libraries.

## Quick start

```bash
python -m pytest
```

The project currently relies only on the Python standard library and pytest runtime.
