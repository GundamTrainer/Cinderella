# Development guide

## Environment

The runtime foundation uses the Python standard library and pytest. Real camera and microphone support is optional and uses OpenCV and sounddevice.

## Local camera and microphone preview

Install the optional hardware dependencies:

```bash
python -m pip install -e ".[hardware]"
```

List available devices:

```bash
python main.py --devices
```

Start a live camera window and microphone level meter:

```bash
python main.py --live
```

Use `--camera-index N` and `--microphone-device N` to select a listed device. The live preview is opt-in, stops with Q or Esc, and does not record or persist audio/video. Windows camera/microphone privacy permissions must be enabled for the Python application.

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

The live preview is a local input connectivity check only. It does not perform speech recognition, image recognition, or model inference. The repository intentionally distinguishes this implemented hardware preview from future multimodal work.
