# Cinderella

Cinderella is a self-developed multimodal character AI project. The current implementation includes a minimal runtime foundation and an opt-in local camera/microphone preview. AI models and recognition are not implemented.

## Current implementation status

Implemented:
- typed runtime configuration loader with validation
- event schema and JSON serialization
- thread-safe bounded event bus
- lifecycle state management for long-running components
- deterministic simulation components for audio, vision, and hardware
- runtime orchestrator with cleanup on partial initialization failure
- optional laptop camera/microphone preview
- opt-in local Whisper speech transcription, MediaPipe object/pose/face-cue inference, and Ollama chat
- pytest coverage for the runtime foundation

Not implemented yet:
- speech synthesis
- robust general action recognition beyond simple pose cues
- emotion diagnosis (appearance and voice cues cannot establish a person's true emotional state)
- hardware device drivers
- voice cloning or character voice profiles
- external AI APIs; network is used only for first-time model downloads

## Runtime foundation

The simulated runtime is intentionally deterministic. The separate live preview is opt-in and uses local hardware only while running.

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

Install Ollama and download a small Korean-capable model:

```bash
ollama pull qwen2.5:1.5b
```

Start local speech transcription, image/object/pose/facial-movement cues, and conversation:

```bash
.\.venv314\Scripts\python.exe main.py --live
```

The first run downloads the Whisper `small` model and MediaPipe task assets into the current user's cache. These downloads require internet once; inference runs locally afterward. Use `--speech-model base` for a smaller/faster but less accurate speech model. Press Q/Esc to stop.

Live processing is opt-in. Raw audio and camera frames are kept in memory only and are neither saved nor sent to Ollama; only recognized text and a short abstract perception summary are sent to the local Ollama service over loopback. Facial output is described as visible movement cues and is not a reliable diagnosis of emotion, intent, identity, or personality. Action recognition is currently limited to simple pose rules such as raised arms.

Python 3.14 is recommended for the optional local inference stack. The original Python 3.15 environment may not have compatible wheels for all inference packages.
