# Privacy and data handling

## Implemented policy

The deterministic simulation components do not access real hardware or create raw audio/video.

The simulation components emit synthetic metadata only. Separate, opt-in laptop modes access real camera frames and microphone samples while active.

`python main.py --preview` opens the devices for a local view and level meter. `python main.py --live` additionally runs local transcription, vision inference, chat, and offline speech output. Neither mode writes audio/video files. Raw microphone buffers and camera frames are processed in memory. Recognized speech or typed chat text and abstract visual cues go to Ollama over an HTTP loopback address on this computer; remote Ollama endpoints are rejected. A temporary JPEG thumbnail (maximum 448 pixels on its longest side) of the current frame is included only when a voice or typed message asks about the camera view; the image is not retained in chat history or saved to disk. Voice output uses the Windows Korean SAPI voice locally. Model weights are downloaded from their configured public model repositories on first use; these are file downloads, not remote inference, and model requests are not sent to a hosted AI API. Press Q or Esc to stop capture. First use downloads model assets to the user's cache; later inference can run without internet.

The microphone callback uses a bounded queue and records only aggregate event counts and reason codes for overflow, dropped chunks, queue pressure, and rejected/accepted segments. Diagnostic logs and events do not contain raw samples, frames, transcripts, or face-feature vectors. Brief in-memory audio chunks and the current visual thumbnail are still used for live inference.

## Planned privacy controls

Future multimodal processing components must implement explicit privacy boundaries before any additional recognition, inference, or persistence is added. This includes:
- explicit opt-in for audio and vision capture
- access controls around stored recordings
- sanitization of generated metadata
- no raw media persistence by default

The detection evaluation/dataset interfaces use metadata and normalized boxes; the synthetic example contains no image file. Do not add camera-derived samples to a dataset without a separate consent, retention, and access-control design.

## Not implemented

Facial movement cues and simple pose rules do not establish true emotion, intent, identity, or personality. The user must grant operating-system device permissions. The system does not perform continuous recording, face recognition/identification, or robust generalized action recognition. A local model may still produce incorrect visual descriptions; verify important observations yourself.
