# AI dependency and backend audit

## Current status

The live runtime uses external implementations isolated behind interfaces and adapters in `ai/contracts.py` and `ai/external_backends.py`. "External" means the implementation, model, or installed service originates outside this repository; local execution does not make it a self-developed model.

| Capability | Runtime implementation | Execution | Provenance | Pretrained weights |
| --- | --- | --- | --- | --- |
| Speech recognition | faster-whisper `small` (configurable model size) | Local CPU/int8 | faster-whisper / Systran checkpoint | Yes |
| Object and pose inference | MediaPipe Tasks EfficientDet Lite2 and Pose Lite | Local CPU | Google MediaPipe task assets | Yes |
| Face movement features | MediaPipe Face Landmarker blendshapes | Local CPU | Google MediaPipe task assets | Yes |
| Text and visual conversation | Ollama with Qwen2.5 3B / Qwen2.5-VL 3B | Local Ollama process; HTTP loopback only | Ollama model registry | Yes |
| Speech output | pyttsx3 with installed Korean Windows SAPI voice | Local Windows speech stack | OS-installed voice | Not applicable in current metadata |

First-run downloads are initiated for model assets as documented in the README. No hosted AI API is configured. The runtime rejects non-loopback Ollama URLs.

## Repository-developed components

`ai/contracts.py`, `ai/vision.py`, `ai/evaluation.py`, and `datasets/object_detection/schema.py` are repository code. They define backend contracts, detection/feature validation and stabilization, a basic evaluator, and annotation-manifest validation. They are not trained recognition models and do not replace the external model weights.

The adapters provide provenance metadata and delegate calls, but the concrete model
implementations and their direct third-party imports remain in `runtime/ai_pipeline.py`. The
implementation boundary is therefore partial, not a complete separation of integrations from
orchestration. Backend metadata exposes package version when installed and explicit
not-reviewed states for license, provenance, and approval. A missing version is represented as
unknown; this change does not claim license verification, checkpoint hashing, or model approval.

## Package groups

- Core runtime: Python standard library only.
- `dev`: pytest and Ruff.
- `hardware`: OpenCV and sounddevice.
- `ai`: faster-whisper, MediaPipe, and pyttsx3.
- Ollama is installed separately and is not a Python package dependency.

No new third-party runtime dependencies were added as part of the backend-isolation work.
