# AI risk management

**Review date:** 2026-10-17  
**Framework:** risk work is organized under the NIST AI RMF functions GOVERN, MAP, MEASURE, and MANAGE. This project record is an implementation aid, not a claim of NIST conformance.

| Function | Current practice | Required record / next control |
|---|---|---|
| GOVERN | Offline-first, provenance distinctions, explicit memory consent APIs, simulated-only hardware | Assign feature owner; maintain approval state, policy version, risk acceptance, review date, and rollback owner |
| MAP | Live device, model, memory, provider, and action boundaries are documented | For each capability record data/model source, intended user/use, prohibited use, affected people, failure impact, and misuse cases |
| MEASURE | Unit tests cover core contracts and a small detection IoU helper | Establish consented versioned datasets, baselines, thresholds, uncertainty/abstention measures, and performance/resource measures |
| MANAGE | Detection validation, bounded queues, provider policy, and simulation gate provide local controls | Add release gates, incident workflow, model/data rollback, degraded-mode validation, and tracked remediation |

## Capability risk record

For each deployed or experimental capability, record:

- capability and accountable owner;
- data source, collection/retention/consent, and model/checkpoint provenance;
- intended user and intended use, plus prohibited use;
- known limitations and failure consequences;
- privacy, security, and autonomy impact;
- evaluation protocol, baseline, measured result, and confidence/uncertainty;
- approval state, approver, and last review date;
- rollback target and recovery procedure.

Missing evidence must be marked unknown or not evaluated, never replaced with an accuracy or human-similarity percentage. Face movement is not evidence of a person's true emotional, medical, or psychological state. The system must not claim consciousness or human equivalence.
