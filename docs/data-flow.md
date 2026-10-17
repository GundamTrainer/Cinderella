# Runtime data flow

## Current live path

```text
camera
  -> LocalVisionAnalyzer (MediaPipe Tasks)
  -> detector-result validation / pixel xywh normalization
  -> DetectionStabilizer
  -> SceneObservation / WorldState
  -> UI overlay and optional visual-question context

microphone
  -> sounddevice callback
  -> bounded AudioInputBuffer (sequence + aggregate diagnostics)
  -> fixed-window speech worker (faster-whisper)
  -> bounded transcript/chat queues
  -> Ollama text model
  -> text output and optional pyttsx3 / Windows SAPI

keyboard
  -> bounded chat queue
  -> Ollama text or explicit visual-question path
```

The optional visual-question path sends only a temporary, downscaled current-frame image to the configured Ollama loopback endpoint. Images are not added to retained conversation history. External information providers are not automatically selected or called by chat.

## Boundaries

- A detector output is a candidate, not a fact. Missing metadata, unsupported classes, invalid boxes, low confidence, and unstable tracks remain unverified.
- A normalized `BoundingBox` uses normalized `xyxy`. The `DetectionResult.coordinate_space` metadata records the input convention for compatibility and diagnostics; it does not change the box's normalized representation.
- Observation payloads contain structured summaries, never raw PCM or image data.
- Current WorldState and Belief/Hypothesis contracts are short-lived in-process models; they are not automatically persisted or attached to all live events.
- Safety proposals pass through a simulation policy. There is no real actuator execution path.

## Gaps for future phases

- `runtime/ai_pipeline.py` still performs provider-specific initialization and inference orchestration.
- Audio, transcript, system-health, and keyboard inputs are not all normalized through the common Observation model.
- Live backend failures are not independently degraded in every mode.
- Trace identifiers are not propagated end-to-end through model, cognition, and response operations.
- Memory, identity, relationship, attention, and response-plan models are not integrated into live chat.

See [current-state.md](current-state.md) and [target-architecture.md](target-architecture.md).
