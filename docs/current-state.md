# Current state audit

**Audit date:** 2026-10-17  
**Repository revision at audit:** `9e3884f`  
**Status:** audit snapshot; not a certification or safety assessment.

## Repository and execution

The project has a small Python core (`ai/`, `runtime/`, `integrations/`, `datasets/`, `tools/`) and four pytest modules. `main.py` exposes opt-in `--live`, `--preview`, `--devices`, and `--physical-ai-demo` modes. The runtime foundation and physical-AI demo can run without hardware; `--live` and `--preview` open local devices.

Live audio flows from a `sounddevice` callback into a bounded `AudioInputBuffer`, then to a worker that transcribes fixed windows with faster-whisper. Transcripts and keyboard input feed a chat worker backed by local Ollama; the optional Windows speech output uses pyttsx3/SAPI. Live video is analyzed through MediaPipe object, pose, and face-landmark Tasks. A temporary downscaled frame may be sent to the Ollama loopback endpoint for an explicit visual question. Provider classes under `integrations/` are not automatically called by chat.

## Ownership and external dependencies

Repository-owned code includes configuration validation, typed events and bounded event bus, lifecycle primitives, simulation components, camera/audio integration, detection normalization and stabilization, short-lived world state, privacy/network policy, provider contracts, state models, and explicit SQLite persistence.

The live inference stack depends on separately maintained faster-whisper/Whisper weights, MediaPipe model assets, Ollama-managed Qwen text and vision-language weights, and pyttsx3/Windows SAPI. Local execution does not make those implementations internally developed. This repository contains no self-trained speech, vision, language, or speech-synthesis model.

Backend records expose the installed library version where available and explicit not-reviewed
states for license, provenance, and approval. They still do not provide a complete release-grade
model card: checkpoint hashes, verified licenses, evaluation evidence, and rollback target are
incomplete or absent.

## Architecture and reliability findings

- `runtime/ai_pipeline.py` is approximately 1,575 lines. Its `run_ai_live` and `LocalVisionAnalyzer.analyze` functions combine device access, backend initialization, inference, worker orchestration, event publication, UI, and cleanup.
- `ai.external_backends` lazily imports implementations from `runtime.ai_pipeline`; the adapter boundary therefore points back into the runtime and does not isolate vendor API details.
- The live pipeline starts several bounded-queue workers as daemon threads. Shutdown joins them with a fixed timeout; blocking keyboard input or model inference can outlast cleanup.
- Startup currently initializes the live backends as a coupled path. Failure of a required backend may prevent the live loop instead of preserving text, audio, or preview capability independently.
- The generic orchestrator cleans up after initialization failure, but `start()` does not unwind components already started if a later start fails.
- Audio sequence gaps are used to avoid joining audio across missing chunks. Callback work includes copying audio and, during speech output, microphone-level calculation.
- Detection normalization handles missing/invalid categories and confidence, rejects unsupported classes, and distinguishes normalized boxes from source coordinate-space metadata. Detection stabilization and short-lived WorldState are tested, but do not establish detector accuracy.

## Privacy, security, and network

Raw audio and camera frames are not written to disk by default. Audio, transcript, and image bytes may still exist transiently in process memory; a downscaled current image is sent to the local Ollama loopback service for an explicit visual question. Model assets may be downloaded during first-time setup. Explicit state-store APIs require caller action and consent for persistent memories; they are not wired into live chat.

Provider HTTP requests use an injectable transport, allowlist and redirect checks, public-host validation, timeouts, content-type checks, and bounded response sizes. These providers are not automatically invoked by live chat. No shell/subprocess execution was found in application code. Current tests do not yet constitute a prompt-injection or comprehensive security suite.

Logs use aggregate audio/detection diagnostics and do not intentionally log raw media or full chat turns. Exception stack traces remain enabled for failures; review exception messages before adding new logging to avoid leaking local environment details.

## Evaluation and verification baseline

`ai/evaluation.py` provides a small caller-supplied object detection/IoU evaluation helper. The repository does not provide a versioned representative test dataset, measured benchmark thresholds, release gate, or model approval evidence.

At the audit revision:

- `python -m pytest`: 150 passed.
- `ruff check .`: passed.
- `git diff --check`: passed.
- The Python 3.14 project environment was used. No camera, microphone, Ollama, network provider, or SAPI was used for this audit.

## Scope decisions

Phase 0 audit artifacts and Phase 1 reliability contracts are complete. Audio diagnostics now
cover received/dropped callback frames, sequence gaps, and segment boundaries/outcomes without
including raw media. Backend records expose installed library versions when available and
explicit not-reviewed license, provenance, and approval states.

Post-Phase 1 verification: `python -m pytest` passed 153 tests and `python -m ruff check .`
passed using the repository's existing Python 3.15 virtual environment. This change does not
claim full backend isolation, enterprise certification, independent degradation for every live
backend, comprehensive telemetry, a production benchmark, or hardware safety. Phase 2 remains
separate work.
