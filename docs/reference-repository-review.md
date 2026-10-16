# Reference repository review

Reviewed on 2026-10-16 using public repository pages and metadata. No repository was cloned, no third-party source was copied, and no model weights were downloaded as part of this review. A repository license does not automatically cover separately published datasets, checkpoints, or model weights.

## Decisions

| Repository | Assessment | Decision for Cinderella |
|---|---|---|
| [TimoBolkart/FLAME-Universe](https://github.com/TimoBolkart/FLAME-Universe) | Catalogue of 3D face-model resources, not an installable runtime. FLAME has separate access and licensing terms. | Do not add as a dependency. Revisit only for a specifically approved 3D-face feature. |
| [kudosscience/fer_and_ter_model](https://github.com/kudosscience/fer_and_ter_model) | Face, speech, and text emotion inference; heavyweight PyTorch/Transformers/audio dependencies. Its MIT claim was not matched by a root license file in the reviewed listing. | Defer; not needed for observable, non-diagnostic interaction. |
| [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) | Local gaze estimation with an MIT code license and ONNX path; checkpoints and face-detector provenance/availability need separate review. | Candidate only if gaze direction has an explicitly approved use. No weights or dependencies added. |
| [kimpmr/AI-Assistant-Reactive-UI](https://github.com/kimpmr/AI-Assistant-Reactive-UI) | Describes a deterministic Java UI/framework, not an LLM. The described source/build files and a confirmed license were not present in the reviewed root listing. | Reject as a runtime dependency. Build on Cinderella's existing UI path if a concrete UI requirement is selected. |
| [anthropics/claude-code](https://github.com/anthropics/claude-code) | Proprietary developer coding tool, not a local runtime model; repository describes service-associated data collection. | Reject for Cinderella runtime integration. |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | Broad YOLO vision toolkit; pretrained weights are downloaded separately. Code is AGPL-3.0, with Enterprise licensing also offered. | Defer pending an explicit license decision, weight review, and hardware benchmark. |
| [kijai/ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) | Work-in-progress ComfyUI custom node; model weights have separate provenance and licensing, and some postprocessing needs CUDA. | Defer; Cinderella does not use ComfyUI. |
| [serengil/deepface](https://github.com/serengil/deepface) | Face identity/verification and demographic/emotion attributes. MIT code does not establish rights for all fetched pretrained weights. | Reject for the baseline runtime; no identity or demographic analysis is needed. |
| [justinshenk/fer](https://github.com/justinshenk/fer) | Facial-expression/emotion classifier with bundled model files and additional ML dependencies. | Defer; visible expression is not a reliable determination of internal emotion. |
| [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) | Apache-2.0 video action-recognition toolbox, but requires a substantial PyTorch/OpenMMLab stack and separate checkpoints. | Defer until a concrete temporal-action requirement and supported hardware are specified. |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | MIT orchestration/integration framework; it does not itself improve the underlying model's dialogue capability. | Do not add for the current direct Ollama flow; revisit for a concrete provider/workflow need. |
| [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 on-device perception toolkit. | Already used by the current perception path; no duplicate integration needed. |
| [microsoft/autogen](https://github.com/microsoft/autogen) | Multi-agent framework whose README identifies maintenance mode and directs new users to its successor. It is not a local model. | Reject new adoption; unnecessary for the current single-model local pipeline. |

## Changes made from this review

- Removed broad Korean substring triggers such as `뭐야`, `뭐가`, and `누가` that incorrectly routed ordinary conversation to the vision model. Camera routing now uses more scene-specific phrases.
- General conversation no longer receives object, pose, face-movement, or affect-hypothesis context. For a visual question with no current image, only object/scene/pose observations are available as explicitly uncertain fallback context.
- A visual-model request retains recent text-only dialogue turns while attaching image bytes only to the current request.
- Failed model turns are not saved to conversation history. The live chat worker reports failures through its status callback instead of logging them silently from the user's perspective.
- Reworked the system prompt to avoid a fixed greeting/example template, adapt response length to the request, and avoid unnecessary follow-up questions without adding a character persona.

These changes improve routing, context relevance, and failure visibility; they do not make the underlying model equivalent to a hosted large-scale model or establish human-like understanding. External integration remains subject to a feature-specific need, code and weight licensing, dependency compatibility, provenance, privacy, and offline-operation review.
