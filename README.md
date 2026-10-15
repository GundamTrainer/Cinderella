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
- opt-in local Whisper transcription, MediaPipe object/pose/face-movement inference, local image-aware Ollama conversation, and offline Korean speech output
- quiet-pause and stable visual-cue conversation prompts, with basic speech-interruption detection
- pytest coverage for the runtime foundation

Not implemented yet:
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

Install Ollama and download the local vision-and-chat model:

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5vl:3b
```

Start local speech transcription, image/object/pose/facial-movement cues, and conversation:

```bash
.\.venv314\Scripts\python.exe main.py --live
```

The first run downloads the Whisper `small` model and MediaPipe task assets into the current user's cache. Ollama downloads its model separately. These downloads require internet once; inference runs locally afterward. Use `--speech-model base` for a smaller/faster but less accurate speech model. Press Q/Esc to stop.

Live processing is opt-in. Raw audio and camera frames are not saved. A temporary JPEG thumbnail (maximum 448 pixels on its longest side) of the current frame is sent over loopback to the local vision model only when a visual question is recognized; older images are not retained in chat history. Regular conversation uses the smaller text model, and proactive prompts use detected text cues. Visual answers can be slow on CPU-only systems. Transcripts and movement cues are sent only to the local Ollama service. Windows' installed Korean text-to-speech voice provides offline speech output. Proactive prompts are rate-limited to at least one minute apart and may be skipped if there is nothing useful to say. Headphones help avoid speaker echo and improve interruption detection. Facial output describes visible movement only, not emotion, intent, identity, or personality. Object detection currently uses a general-purpose COCO detector, while action recognition remains limited to simple pose rules.

Python 3.14 is recommended for the optional local inference stack. The original Python 3.15 environment may not have compatible wheels for all inference packages.
