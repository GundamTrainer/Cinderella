# Dialogue policy

`decide_tool_use` prefers local answers, asks for clarification on unclear requests, uses valid cache before fresh retrieval, respects offline mode, and creates a plan for at most one explicitly selected provider. It does not fan out across APIs.

`ResponsePlan` models response type, text, confidence, tone parameters, evidence IDs, uncertainty, actions, privacy warnings, and silence rationale. This is a data contract only; Ollama still generates the live reply and the plan is not yet used to compose, speak, suppress, or correct that reply.

Live chat now sends camera observations only when the utterance specifically refers to the visible scene; general conversation no longer receives face-motion or object detections as unsolicited context. Visual questions retain recent text-only dialogue turns, but image bytes remain attached only to the current request. Unsuccessful replies do not enter the conversation history. Generic questions such as “너는 뭐야?” no longer select the vision model solely because they contain “뭐야”. Facial movement and tentative affect hypotheses are excluded from dialogue context and cannot independently trigger proactive conversation; the system does not infer a user's internal emotion from them.

The system prompt asks for context-sensitive response length and avoids fixed greeting templates. It still does not create a user-specific persona or guarantee human-like dialogue. Existing temporary terminal chat, microphone input, local TTS, and basic TTS echo guard are unchanged. Repeated-proactive-speech handling remains limited to the legacy scheduler.
