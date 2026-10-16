# Dialogue policy

`decide_tool_use` prefers local answers, asks for clarification on unclear requests, uses valid cache before fresh retrieval, respects offline mode, and creates a plan for at most one explicitly selected provider. It does not fan out across APIs.

`ResponsePlan` models response type, text, confidence, tone parameters, evidence IDs, uncertainty, actions, privacy warnings, and silence rationale. This is a data contract only; Ollama still generates the live reply and the plan is not yet used to compose, speak, suppress, or correct that reply.

Existing temporary terminal chat, microphone input, local TTS, and basic TTS echo guard are unchanged. Repeated-proactive-speech handling remains limited to the legacy scheduler.
