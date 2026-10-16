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
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
```

Start complete local inference:

```bash
.\.venv314\Scripts\python.exe main.py --live
```

During live mode, type into the terminal and press Enter to chat. Responses are spoken, and microphone conversation stays active. `/text-off` disables terminal input only. Use `--speech-model base` for lower memory use. Use `--camera-index N` and `--microphone-device N` to choose devices. Q or Esc stops capture. A basic preview without AI is available as `.\.venv314\Scripts\python.exe main.py --preview`.

The first run fetches the Whisper `small` weights and MediaPipe task assets to the user's cache. This is setup-time network access; subsequent inference is local.

The default settings are in `configs/default.yaml`. `box_clamp_tolerance_pixels` bounds how far detector boxes may exceed the camera frame before rejection; `audio_input_queue_capacity` limits callback-to-worker buffering. Invalid or unsupported detections stay unverified, and repeated object observations are required before a label is exposed as stable. This validation/stabilization layer does not improve the pretrained detector's underlying accuracy.

AI chat communicates only with the local Ollama service over loopback. Regular conversation uses the text model. When the recognized utterance or typed message asks about the camera view, the vision model receives a temporary JPEG thumbnail (maximum 448 pixels on its longest side); that image is not retained in conversation history or written to disk. Visual answers can be slow on CPU-only systems. Real-time object labels still use MediaPipe EfficientDet Lite2 and are limited to its COCO categories; use an explicit visual question to ask the local vision-language model about a scene. Speech is produced with the Korean voice installed in Windows; no audio is sent to a speech service. The conversation scheduler waits at least a minute after activity and uses jittered quiet intervals or stable visible cues; it does not infer mood. Basic interruption detection compares sustained mic input with initial speaker echo, so headphones are recommended.

## Model sources

Current live-mode integrations are external backends behind the contracts in `ai/contracts.py`; these wrappers do not imply that the models were developed in this repository. See [dependency-audit.md](dependency-audit.md) and [model-migration-plan.md](model-migration-plan.md) before changing a backend.

- Speech recognition: [faster-whisper](https://github.com/SYSTRAN/faster-whisper), multilingual `small` weights from [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small).
- Vision: MediaPipe Tasks assets fetched from Google's official `storage.googleapis.com/mediapipe-models` endpoint; EfficientDet Lite2 is used for object detection.
- Local text conversation: [Qwen2.5 3B in Ollama](https://ollama.com/library/qwen2.5).
- Local visual questions: [Qwen2.5-VL 3B in Ollama](https://ollama.com/library/qwen2.5vl). Ollama runs on loopback and must be installed separately.
- Offline speech: `pyttsx3` with an installed Windows Korean SAPI voice.

Review the respective project/model license terms before redistribution or commercial use.

## Minimal runtime workflow

1. Load runtime settings from `configs/default.yaml`.
2. Create a bounded `EventBus` with the configured capacity.
3. Declare runtime components and initialize them.
4. Start components and allow the orchestrator to process events.
5. On shutdown, stop components in reverse order and close them safely.

The optional live AI mode has its own device/model lifecycle in `runtime/ai_pipeline.py`; it is separate from the deterministic simulated orchestrator. Object annotation manifests use the schema in [dataset-format.md](dataset-format.md). The evaluation helper and its current limitations are documented in [evaluation.md](evaluation.md).

## Validation commands

```bash
python -m pytest
python -m ruff check .
```

## Forward-looking work

Robust generalized action recognition, reliable emotional-state inference, and hardware device drivers remain future work. Expression analysis intentionally reports visible movements rather than diagnosing emotional states.
