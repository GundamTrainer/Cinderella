# Development guide

## Environment

The project currently uses the Python standard library and pytest. No external YAML parser, microphone library, camera library, or hardware stack is required for the runtime foundation.

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

The repository intentionally distinguishes implemented runtime infrastructure from future multimodal work. Implementation of actual AI or hardware integration belongs in dedicated modules after the runtime contracts are stabilized.
