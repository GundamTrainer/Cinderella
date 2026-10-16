# Cinderella

Cinderella is a local multimodal runtime project with an opt-in camera/microphone AI mode. The live mode currently depends on separately maintained local/external backends; this repository does not contain self-trained speech, vision, or language models and does not establish human-like feelings or consciousness.

## Current implementation status

Implemented:
- typed runtime configuration loader with validation
- event schema and JSON serialization
- thread-safe bounded event bus
- lifecycle state management for long-running components
- deterministic simulation components for audio, vision, and hardware
- runtime orchestrator with cleanup on partial initialization failure
- optional laptop camera/microphone preview
- opt-in local Whisper transcription, MediaPipe object/pose/face-movement inference, local image-aware Ollama conversation, and offline Korean speech output, with external-backend provenance adapters; concrete model implementations are not yet fully separated from the live pipeline
- validated object observations that distinguish invalid, unsupported, low-confidence, and unstable results; normalized boxes are clamped only within a small frame-edge tolerance and stable labels require repeated detections. This improves result handling and consistency, not detector accuracy.
- JSON-safe typed observations plus short-lived world-state tracking with object regions, motion and position history; uncertain labels persist briefly across empty frames before expiring
- bounded microphone buffering with overflow, drop, queue-pressure, and segment diagnostics
- typed capability provenance/status records; the former 54% estimate is retained only as a developer helper and is no longer shown in the live overlay
- opt-in provider contracts, conservative policy validation, HTTPS allowlisting, redirect checks, response-size limits, bounded retries, and TTL caching
- offline-first provider scaffolding for MediaWiki, Open-Meteo, local status, approved static records, citations, and single-tool decision plans
- local SQLite persistence APIs for identity, relationship, and explicitly consented memories; not automatically connected to chat or initialized by the CLI
- typed identity-change review, evidence-bound relationship changes, approval-gated goals, and source-aware response plans
- face movement alone does not create a state hypothesis; the optional heuristic requires independent non-facial evidence and is not connected to the live dialogue path
- object-detection annotation validation and a basic class-aware IoU evaluation helper
- quiet-pause and stable visual-cue conversation prompts, with basic speech-interruption detection
- pytest coverage for the runtime foundation

Not implemented yet:
- self-developed or self-trained speech, vision, language, or speech-synthesis models; the live mode still relies on the external projects/models listed in the development guide
- a training pipeline, checkpoints, data collection, or full benchmark/evaluation suite
- robust general action recognition beyond simple pose cues
- emotion diagnosis (appearance and voice cues cannot establish a person's true emotional state)
- hardware device drivers
- voice cloning or character voice profiles
- automatic provider requests from chat or the CLI
- an operational Wikidata provider, face identity enrollment, automatic memory learning, autonomous goal execution, or integrated consent UI

New information providers remain offline and disabled by default. MediaWiki is blocked until query-data retention is verified. Open-Meteo is blocked by the required policy because its service offers paid/commercial and paid-data tiers; there is no override. The provider classes are not connected to the live conversation pipeline. See [docs/integrations.md](docs/integrations.md), [docs/provider-policy.md](docs/provider-policy.md), and [docs/offline-operation.md](docs/offline-operation.md).

## Runtime foundation

The simulated runtime is intentionally deterministic. The separate live preview and AI modes are opt-in and use local hardware only while running. World-state freshness is checked by the simulation safety gate; this is not a certified hardware safety controller. See [docs/model-migration-plan.md](docs/model-migration-plan.md) for the staged replacement plan, [docs/dependency-audit.md](docs/dependency-audit.md) for current backend provenance, and [docs/state-understanding.md](docs/state-understanding.md) for observation/belief boundaries.

To inspect the simulation-only physical-AI action/safety flow without opening hardware:

```powershell
.\.venv314\Scripts\python.exe main.py --physical-ai-demo
```

The demo emits JSON proposal/decision/simulated-command events; it does not actuate devices. Research boundaries and limitations are documented in [docs/physical-ai.md](docs/physical-ai.md).

The live-window 54% estimate was removed. It was a rough developer-status calculation, not an accuracy or human-similarity measure.

## Quick start

```bash
python -m pytest
```

## Local laptop preview and AI mode

Create a Python 3.14 environment, then install hardware and local AI dependencies (PowerShell):

```bash
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -e ".[dev,hardware,ai]"
```

List devices and run the lightweight preview:

```bash
.\.venv314\Scripts\python.exe main.py --devices
.\.venv314\Scripts\python.exe main.py --preview
```

Install Ollama and download the local vision-and-chat model:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
```

Start local speech transcription, image/object/pose/facial-movement cues, and conversation:

```bash
.\.venv314\Scripts\python.exe main.py --live
```

While live mode is running, type a message into the terminal and press Enter for temporary text chat. Replies are also spoken aloud; microphone conversation remains enabled. Type `/text-off` to stop terminal input only. Press Q/Esc in the camera window to stop live mode.

The first run downloads the Whisper `small` model and MediaPipe task assets into the current user's cache. Ollama downloads its model separately. These downloads require internet once; inference runs locally afterward. Use `--speech-model base` for a smaller/faster but less accurate speech model.

Live processing is opt-in. Raw audio and camera frames are not saved. A temporary JPEG thumbnail (maximum 448 pixels on its longest side) of the current frame is sent over loopback to the local vision model when a visual question is asked by voice or text; older images are not retained in chat history. Regular conversation uses a local Qwen2.5 3B text model with concise Korean conversational instructions, and proactive prompts use detected text cues. No hosted AI API is used, and the runtime rejects non-loopback Ollama addresses. Model weights are downloaded from configured public repositories on first use; after setup, inference runs locally. Visual answers can be slow on CPU-only systems. Transcripts and movement cues are sent only to the local Ollama service. Windows' installed Korean text-to-speech voice provides offline speech output. Proactive prompts are rate-limited to at least one minute apart and may be skipped if there is nothing useful to say. Headphones help avoid speaker echo and improve interruption detection. Facial output describes visible movement only, not emotion, intent, identity, or personality. Real-time object labels still come from the general-purpose COCO detector and may miss or mislabel objects. Stable boxes are shown only after repeated detection; low-confidence or unsupported results are not presented as confirmed object labels. Asking about the view by voice or text uses the separate local vision-language model.

Python 3.14 is recommended for the optional local inference stack. The original Python 3.15 environment may not have compatible wheels for all inference packages.
