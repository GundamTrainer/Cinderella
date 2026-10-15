# Privacy and data handling

## Implemented policy

The current runtime foundation intentionally does not capture or store raw audio, images, facial data, voiceprints, or personal identifiers.

The simulation components emit synthetic metadata only. Example fields such as `synthetic: true` and deterministic metadata are used in tests and runtime scaffolding, but no raw media or personal data is created by the project.

## Planned privacy controls

Future multimodal components must implement explicit privacy boundaries before any real microphone, camera, or hardware input is connected. This includes:
- explicit opt-in for audio and vision capture
- access controls around stored recordings
- sanitization of generated metadata
- no raw media persistence by default

## Not implemented

This repository does not yet contain any real audio capture, camera capture, or inference pipeline. Any claim that actual multimodal or personal data processing is live would be inaccurate.
