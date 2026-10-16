# State understanding

Existing vision features report observable object, pose, and facial-movement cues. Face movement alone does not create an affect hypothesis. The optional heuristic requires an independently supplied non-facial source, remains uncalibrated and short-lived, and is not allowed to drive speech. The live pipeline supplies no such corroboration, so affect hypotheses are currently absent.

Face movement cannot establish emotion, mental health, identity, personality, or intent. Self-reported state must remain separate from visual/audio inference. `Observation`, `WorldState`, `Belief`, and `InterpretationHypothesis` are separate types; an observation does not automatically become a confirmed belief. The typed response plan can distinguish observations, hypotheses, external sources, and uncertainty, but it is not yet connected to live dialogue.

Face identity enrollment, multi-person identity matching, and a validated multimodal interpretation engine are not implemented.
