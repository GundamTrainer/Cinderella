# Privacy and data handling

## Implemented policy

The deterministic simulation components do not access real hardware or create raw audio/video.

The simulation components emit synthetic metadata only. Separate, opt-in laptop modes access real camera frames and microphone samples while active.

`python main.py --preview` opens the devices for a local view and level meter. `python main.py --live` additionally runs local transcription, vision inference, chat, and offline speech output. Neither mode writes audio/video files. Raw microphone buffers and camera frames are processed in memory. Recognized text and abstract visual cues go to Ollama over the loopback interface on this computer. A temporary JPEG thumbnail (maximum 448 pixels on its longest side) of the current frame is included only when the recognized utterance is a visual question; the image is not retained in chat history or saved to disk. Voice output uses the Windows Korean SAPI voice locally. Press Q or Esc to stop capture. First use downloads model assets to the user's cache; later inference can run without internet.

## Planned privacy controls

Future multimodal processing components must implement explicit privacy boundaries before any additional recognition, inference, or persistence is added. This includes:
- explicit opt-in for audio and vision capture
- access controls around stored recordings
- sanitization of generated metadata
- no raw media persistence by default

## Not implemented

Facial movement cues and simple pose rules do not establish true emotion, intent, identity, or personality. The user must grant operating-system device permissions. The system does not perform continuous recording, face recognition/identification, or robust generalized action recognition. A local model may still produce incorrect visual descriptions; verify important observations yourself.
