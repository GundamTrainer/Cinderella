# External backends

Current live integrations are third-party implementations/models:

| Function | Backend | Provenance |
|---|---|---|
| Speech recognition | faster-whisper / Whisper weights | External project and pretrained weights |
| Object, pose, face movement | MediaPipe Tasks assets | External project and pretrained assets |
| Text and visual conversation | Ollama with Qwen models | External local process and pretrained weights |
| Speech output | pyttsx3 / Windows SAPI | External speech library and installed OS voice |

`ai/contracts.py` labels these as external. Construction is centralized in `runtime/backend_factory.py`, but the adapter delegates still import legacy implementations from `runtime/ai_pipeline.py`; Ollama HTTP implementation and model asset downloads also remain there. This is partial isolation, not a completed backend extraction.

The provider network policy governs new information providers only. Ollama loopback is separately constrained. First-use model downloads remain a separate fixed-URL network path; do not assume `network.mode: offline` disables them.

No pretrained models are self-developed or self-trained in this repository.
