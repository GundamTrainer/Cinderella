# Development guide

## Environment

The runtime foundation uses the Python standard library and pytest. Real camera and microphone support is optional and uses OpenCV and sounddevice. Local inference additionally uses faster-whisper and MediaPipe Tasks. Python 3.14 is recommended; the existing Python 3.15 environment may not have wheels for all inference dependencies.

## Local camera and microphone preview

Create a supported Python environment and install the optional device and AI dependencies (PowerShell):

```bash
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -e ".[dev,hardware,ai]"
```

List available devices:

```bash
.\.venv314\Scripts\python.exe main.py --devices
```

Install Ollama separately, then download the local image-aware conversation model:

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5vl:3b
```

Start complete local inference:

```bash
.\.venv314\Scripts\python.exe main.py --live
```

The first run fetches the Whisper `small` weights and MediaPipe task assets to the user's cache. This is setup-time network access; subsequent inference is local. Use `--speech-model base` for lower memory use. Use `--camera-index N` and `--microphone-device N` to choose devices. Q or Esc stops capture. A basic preview without AI is available as `.\.venv314\Scripts\python.exe main.py --preview`.

AI chat communicates only with the local Ollama service over loopback. Regular conversation uses the text model. When the transcript appears to ask about the camera view, the vision model receives a temporary JPEG thumbnail (maximum 448 pixels on its longest side); that image is not retained in conversation history or written to disk. Visual answers can be slow on CPU-only systems. Speech is produced with the Korean voice installed in Windows; no audio is sent to a speech service. The conversation scheduler waits at least a minute after activity and uses jittered quiet intervals or stable visible cues; it does not infer mood. Basic interruption detection compares sustained mic input with initial speaker echo, so headphones are recommended.

## Model sources

- Speech recognition: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), multilingual `small` weights from [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small).
- Vision: MediaPipe Tasks assets fetched from Google's official `storage.googleapis.com/mediapipe-models` endpoint; EfficientDet Lite2 is used for object detection.
- Local text conversation: [Qwen2.5 1.5B in Ollama](https://ollama.com/library/qwen2.5).
- Local visual questions: [Qwen2.5-VL 3B in Ollama](https://ollama.com/library/qwen2.5vl). Ollama runs on loopback and must be installed separately.
- Offline speech: `pyttsx3` with an installed Windows Korean SAPI voice.

Review the respective project/model license terms before redistribution or commercial use.

## Minimal runtime workflow

1. Load runtime settings from `configs/default.yaml`.
2. Create a bounded `EventBus` with the configured capacity.
3. Declare runtime components and initialize them.
4. Start components and allow the orchestrator to process events.
5. On shutdown, stop components in reverse order and close them safely.

## Validation commands

```bash
python -m pytest
python -m ruff check .
```

## Forward-looking work

Robust generalized action recognition, reliable emotional-state inference, and hardware device drivers remain future work. Expression analysis intentionally reports visible movements rather than diagnosing emotional states.
