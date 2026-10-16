# Model migration plan

## Goal

Replace external model implementations gradually with repository-developed models while keeping the current opt-in laptop workflow usable during the transition. The current adapters make provenance explicit; they do not claim that external models are self-developed.

## Current foundation

- Typed backend contracts describe speech, vision, language, and speech synthesis interfaces.
- External adapters identify current local implementations and model sources.
- Vision results have normalized boxes, confidence and area filtering, frame-to-frame tracking, and explicit unverified states.
- Face movement features are normalized and temporally stabilized; no psychological-state classifier is included.
- Dataset annotation metadata has validation rules, and a small class-aware IoU evaluator is available.

## Stages

1. **Measure before replacing.** Define a versioned evaluation set and task-specific metrics for speech, object detection, pose/action cues, language responses, and speech output. Record model version, configuration, device, and dataset split.
2. **Build governed datasets.** Collect only consented or appropriately licensed data. Split by capture session/person where relevant to prevent leakage. Keep raw media retention, access, deletion, and redaction policies separate from annotation metadata.
3. **Train one bounded capability.** Start with a narrow task and baseline. Record reproducible training configuration and checkpoint provenance. Compare against the external baseline on the same held-out data.
4. **Add an internal adapter.** Implement the backend contract and require explicit selection. Preserve the external adapter as a fallback until quality, latency, and privacy criteria pass.
5. **Roll out and audit.** Add regression tests and capability health reporting. Keep model provenance and evaluation results documented; remove external dependencies only after all active call paths have migrated.

## Not implemented

There is currently no training pipeline, model checkpoint registry, benchmark runner, automated dataset acquisition, or production backend-selection setting. The current IoU helper evaluates a single frame only. Do not treat the stabilizer as an accuracy upgrade to the underlying model.
