# Target architecture

## Intended layers

1. **Devices:** camera, microphone, speaker, keyboard, and future hardware adapters.
2. **Backends:** simulated, internal, or external adapters implementing repository-owned contracts.
3. **Observations:** versioned, typed, privacy-minimized sensor/model results.
4. **Temporal processing:** bounded audio segmentation, tracking, and observation stabilization.
5. **World model:** immutable snapshots of tracked entities, recent events, health, and unresolved uncertainty.
6. **Cognitive state:** explicit beliefs, hypotheses, consented memories, identity, relationships, and goals.
7. **Decision:** attention, tool selection, response planning, and action proposals.
8. **Policy:** privacy, network, memory, authorization, resource, and safety rules.
9. **Execution:** text/TTS, approved providers, simulated actions, and—only after a separate safety case—hardware.
10. **Evaluation and operations:** versioned schemas, model/data provenance, tests, benchmarks, traces, release gates, and rollback.

Dependencies should point toward stable repository-owned contracts. Core cognition should not import MediaPipe, faster-whisper, Ollama, or SAPI types. Device and model implementations should be replaceable adapters. A model response or external document is untrusted data and cannot directly mutate identity, authorize memory, execute a tool, or actuate hardware.

## Autonomy and privacy defaults

Default autonomy is `observe` or `suggest`; external requests, persistent memory, installation, deletion, message sending, payments, model deployment, and physical movement require explicit policy authorization. The current physical-action implementation remains simulation-only.

Raw audio/video, transcripts, secrets, embeddings, and personal memories are excluded from default telemetry. Every state or action should retain source, validity, confidence, and uncertainty without implying consciousness or human equivalence.

## Migration boundary

This is a target, not a claim about current implementation. Use the staged plan in [migration-plan.md](migration-plan.md); preserve existing CLI and settings while extracting one adapter at a time. Do not move all logic out of `runtime/ai_pipeline.py` in a single change.
