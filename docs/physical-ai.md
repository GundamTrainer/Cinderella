# Physical AI: research and simulation design

## Research scope and evidence

This review uses public primary sources from OpenAI and Tesla. A company presentation is treated as evidence of what the company presented, not as an independent safety or performance evaluation.

### OpenAI research

- OpenAI-authored [Learning Dexterous In-Hand Manipulation](https://arxiv.org/abs/1808.00177) (first posted 2018-08-01; revised 2019-01-18) describes policies trained in simulation with randomized physical properties and appearance, a separate vision pose estimator, and a physical Shadow Dexterous Hand performing a narrow object-reorientation task. It is evidence for a specific sim-to-real research result, not general-purpose household robotics.
- OpenAI-authored [Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) (posted 2019-10-16) describes automatic domain randomization and memory-augmented policies for a specific Rubik's Cube task on a custom robot hand. Its reported result is bounded to its experiment and does not establish general robustness or safe operation around people.
- OpenAI's [Figure partnership announcement](https://openai.com/index/figure/) (2024-02-29) states a research partnership to develop AI models for humanoid robots. The announcement is not, by itself, a technical specification or independently evaluated performance result.

### Tesla materials

- Tesla's primary source [AI Day 2022](https://www.youtube.com/watch?v=ODSJsviD_SU) presented its Optimus development. A company event/demo does not establish production readiness, reliable general-purpose autonomy, or a safety case.
- Tesla's [AI page](https://www.tesla.com/AI) is an official company page, but its current technical claims could not be independently checked in this review. No detailed Optimus capability claims here rely on that page.

## Transferable design choices

The bounded research results motivate separations that are useful even without training a robot policy:

1. Keep observations, state estimates, action proposals, safety decisions, and actuator results as separate typed records/events.
2. Exercise policies against reproducible seeded scenes and recorded synthetic cases before evaluating on a held-out set.
3. Vary simulated scene and sensor parameters to probe failure cases. Randomization broadens test coverage; it does not prove real-world coverage.
4. Measure each narrow task using explicit success/failure criteria rather than a single “human-like” score.

The runtime implementation in `runtime/physical_ai.py` follows only the simulation-side action boundary: it emits a planner proposal, a separate safety decision, and—only for an approved motion/stop proposal—a `hardware.command_simulated` event. Motion is rejected when the scene contains stable or uncertain objects, when confidence is below the configured threshold, when a proposal is not synthetic, when the world-state timestamp is invalid, future-dated, or older than the one-second default freshness window, or when the simulated distance exceeds the small configured limit. The clock is injectable for deterministic tests. No camera, motor, robot, or actuator API is called. This simulation gate is a testable software policy, not a certified physical safety controller or emergency stop.

Run the deterministic clear-scene/uncertain-scene safety example without opening devices:

```powershell
.\.venv314\Scripts\python.exe main.py --physical-ai-demo
```

The demo serializes its proposal, approval/rejection, and simulated-command events as JSON. A move in a clear synthetic scene is approved only within the 0.1 m simulated limit; an uncertain scene is rejected. Neither result calls a physical device.

## Tentative affect hypotheses

Face movement alone does not generate a state hypothesis. The helper requires an explicitly supplied independent non-facial source and still marks any result as an uncalibrated, short-lived hypothesis that is not allowed for speech. The live pipeline does not supply such corroboration, so it currently creates no affect hypotheses. This is not emotion recognition or diagnosis.

## Quality targets and evidence still needed

The requested 5x/2.5x/250% multipliers cannot be claimed from feature-coverage percentages. A zero baseline cannot be multiplied into a meaningful target, and the repository does not contain a consented, labeled, representative vision/audio benchmark. `ai/evaluation.py` now aggregates object-detection precision, recall, and matched IoU across caller-supplied fixed samples, but no real benchmark run or measured improvement is included.

Before setting quality multipliers:

- define a fixed, consented, session-separated held-out set and label protocol;
- record model/checkpoint, preprocessing, thresholds, device, and code revision;
- report object-detection precision/recall/IoU by class and scene conditions;
- report speech WER and response/turn-taking latency on an approved speech set;
- report tentative-affect agreement only against an explicit, consented annotation policy, and retain an abstain/unknown option;
- report physical-AI simulated task completion, safety-gate rejection reasons, and collision/constraint violations across fixed and randomized seeds.

Do not use Internet-scraped personal media or automatically collect camera/microphone data for these benchmarks. Sim-to-real evidence would require a separate review and explicit hardware safety case; real actuators remain outside this implementation.
