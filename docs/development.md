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

Install Ollama separately, then download the compact local chat model:

```bash
ollama pull qwen2.5:1.5b
```

Start complete local inference:

```bash
.\.venv314\Scripts\python.exe main.py --live
```

The first run fetches the Whisper `small` weights and MediaPipe task assets to the user's cache. This is setup-time network access; subsequent inference is local. Use `--speech-model base` for lower memory use. Use `--camera-index N` and `--microphone-device N` to choose devices. Q or Esc stops capture. A basic preview without AI is available as `.\.venv314\Scripts\python.exe main.py --preview`.

AI chat communicates only with the local Ollama service. It receives the transcribed utterance and text descriptions of current visual cues; raw frames and audio are not transmitted. Action cues currently use simple pose rules, and facial blendshapes represent movement rather than ground truth emotion.

## Model sources

- Speech recognition: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), multilingual `small` weights from [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small).
- Vision: MediaPipe Tasks assets fetched from Google's official `storage.googleapis.com/mediapipe-models` endpoint.
- Local conversation: [Qwen2.5 1.5B in Ollama](https://ollama.com/library/qwen2.5:1.5b). Ollama runs on loopback and must be installed separately.

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

Speech synthesis, stronger action recognition, and evaluated affect modeling remain future work.
