# Security threat model

**Review date:** 2026-10-17  
**Scope:** local multimodal prototype, optional loopback Ollama, and offline-by-default information providers. This is a development threat model, not a penetration-test report.

## Assets and trust boundaries

Assets include microphone/camera observations, transcripts, explicit persistent memories, local model/cache files, configuration, provider allowlists, model provenance, and simulated action decisions. Trust boundaries exist between devices and workers, model adapters and core code, untrusted model/provider output and policy decisions, the process and loopback Ollama, and the repository and downloaded dependencies/model assets.

## Threat register

| Threat | Current mitigation | Remaining exposure |
|---|---|---|
| Prompt injection / hidden context exposure | Provider output is not automatically sent as executable commands; no connected chat tool executor | Model prompts can still contain untrusted perception/context; no dedicated adversarial tests |
| Sensitive information disclosure | Offline-first providers; loopback-only Ollama; no default raw-media logging | Transcript/current image are transiently sent to local Ollama; memory/prompt policy not unified |
| Excessive agency / goal hijacking / tool misuse | No autonomous hardware driver or chat-connected provider execution; simulated safety gate | No centralized authorization engine or typed tool-plan gate |
| Supply-chain compromise | Optional dependencies and provenance docs; model files are external and cached | Dependency lock/hash verification, SBOM, signed model/checkpoint verification, and license review incomplete |
| Data poisoning / model poisoning / memory poisoning | No automatic training or automatic memory learning | No signed dataset pipeline or adversarial memory validation |
| Unbounded resource consumption | Bounded audio, transcript, chat, and event queues; bounded provider responses | Model inference and UI latency can be long; no unified resource budget or process-level memory cap |
| Misinformation / unsupported inference | Low-confidence and unknown detection states; face-only affect hypotheses disabled in live path | No response planner/fact-checking gate in live dialogue |
| Vector/embedding weakness | No production vector store or face-embedding persistence | Future memory/search must add access control, provenance, and poisoning tests |
| Improper output handling / unexpected code execution | No shell/subprocess executor found in application code; provider outputs are data records | No general output-sanitization boundary for future tools |
| Cascading failure | Exceptions surfaced; queues bounded; some backend cleanup present | Live initialization remains coupled; no complete degraded-mode/circuit-breaker behavior |
| Human-agent trust exploitation | Documentation disclaims consciousness and discourages dependency; no character imitation | Live model response safety is prompt-dependent and not independently policy-validated |

## Network and execution controls

Ollama is restricted to an unauthenticated loopback HTTP URL by the live client. Provider transport denies implicit proxies, validates allowlisted HTTPS destinations and redirects, rejects non-public resolved addresses, bounds response reads, and applies a timeout. Network mode defaults to offline. Application source contains no shell/subprocess invocation.

Model downloads are a separate first-run supply-chain path to public repositories; the inference path being local does not remove that exposure. No API credentials or externally hosted generation services are configured.

## Required future controls

- Pin dependencies and produce a reviewed software/model bill of materials.
- Verify checkpoint hashes, licenses, model cards, dataset consent, and rollback targets before deployment.
- Add mock-only adversarial tests for prompt injection, tool misuse, memory poisoning, output handling, and resource exhaustion.
- Centralize authorization so model text cannot authorize external requests, memory persistence, or device actions.
- Keep raw sensor content and personal text out of default traces; test redaction using synthetic sentinel secrets.
- Treat this register as a living risk record and record owner, severity, evidence, mitigation, and review date for each issue.
