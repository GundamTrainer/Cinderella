# Privacy and data handling

## Implemented policy

The deterministic simulation components do not access real hardware or create raw audio/video.

The simulation components emit synthetic metadata only. The separate, opt-in laptop preview does access real camera frames and microphone samples while active.

The laptop preview opens devices only when started with `python main.py --live`. Camera frames are displayed locally and microphone buffers are reduced to an in-memory signal level; neither raw media nor audio/video files are saved. Press Q or Esc to stop the preview. The device-list command may briefly open camera indexes while probing them.

## Planned privacy controls

Future multimodal processing components must implement explicit privacy boundaries before any recognition, inference, or persistence is added. This includes:
- explicit opt-in for audio and vision capture
- access controls around stored recordings
- sanitization of generated metadata
- no raw media persistence by default

## Not implemented

The preview does not provide speech recognition, image recognition, or an inference pipeline. The user must grant operating-system device permissions.
