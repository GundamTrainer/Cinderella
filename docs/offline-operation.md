# Offline operation

New information providers default to `network.mode: offline` and `allow_external_requests: false`; provider code is not initialized by the CLI. Local system status, static approved records, event runtime, simulators, and companion models can operate without public network access.

The opt-in live inference path has separate existing requirements: first-use model assets may download, and local Ollama communicates over loopback. The configured provider offline mode does not disable those legacy model downloads. Once assets are present, the inference path is local; no hosted inference API is configured.

`--live` still requires optional device/model packages and local Ollama; it is not the offline core runtime. Missing providers leave local runtime intact because provider failures are not wired into it.
