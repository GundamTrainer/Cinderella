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
- optional live laptop camera and microphone preview (no recording)
- pytest coverage for the runtime foundation

Not implemented yet:
- actual AI model inference
- speech recognition or image recognition
- hardware device drivers
- voice cloning or character voice profiles
- internet connectivity or external API features

## Runtime foundation

The simulated runtime is intentionally deterministic. The separate live preview is opt-in and uses local hardware only while running.

## Quick start

```bash
python -m pytest
```

## Local laptop preview

Install optional camera and microphone support:

```bash
python -m pip install -e ".[hardware]"
```

List local devices and launch the non-recording preview:

```bash
python main.py --devices
python main.py --live
```

The live mode is opt-in, displays camera frames and microphone signal level locally, and does not save audio or video. It does not perform AI inference or speech/vision recognition. Device probing may briefly open camera indexes; grant camera and microphone permissions to Python in Windows privacy settings if access is denied.

The runtime foundation and tests otherwise use the Python standard library plus pytest.
