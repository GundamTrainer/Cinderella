# Migration plan

This plan records the audited migration boundary. It preserves the current CLI, configuration shape, and optional device/model features while reducing coupling incrementally.

| Phase | Scope | Exit criteria | Status |
|---|---|---|---|
| 0 — Audit and risk baseline | Inventory modules/data flows, identify coupling and failure/privacy risks, document threat model and target boundaries | Current-state, data-flow, target-architecture, risk, security, and migration records reviewed | Completed in this change |
| 1 — Input and adapter contracts | Preserve detector rejection semantics and source coordinate space; complete bounded audio sequence/drop diagnostics; strengthen backend provenance contracts and regression coverage | Targeted tests pass; full pytest and Ruff pass; CLI/API compatibility remains | Completed in this change |
| 2 — Temporal cognition | Consolidate Observation, tracking, WorldState, Belief/Hypothesis, and fixed synthetic evaluation formats | Deterministic tests cover schema, temporal expiration, uncertainty, and no implicit belief confirmation | Existing partial implementation; expand only after Phase 1 |
| 3 — Decision and fallback | Add attention, goals, response plans, template fallback, and per-backend degradation | Mocked failure tests prove unaffected modalities remain usable; outputs pass privacy/safety policy | Planned |
| 4 — Persistence governance | Integrate identity, relationship, and consent-based memory workflows | Authorization, edit/delete/expiry, correction, and persistence tests pass | Planned |
| 5 — Operations and evaluation | Add privacy-redacted telemetry, model/dataset metadata, benchmark harness, release gates, rollback | Versioned evidence and thresholds exist for each release capability | Planned |
| 6 — Tools and external sources | Add source records and policy-gated providers with prompt-injection isolation | Offline defaults, explicit authorization, mock adversarial tests, bounded requests | Planned |

Do not rewrite `runtime/ai_pipeline.py` wholesale. Extract one backend or responsibility behind a compatibility adapter per change. Preserve all pre-existing tests and CLI behavior; run tests without camera, microphone, network services, or model APIs. Hardware execution remains prohibited until a separate safety case and explicit enable mechanism are reviewed.
