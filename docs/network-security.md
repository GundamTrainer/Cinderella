# Network security

New public integrations use `NetworkPolicy` and an injected GET-only JSON transport:

- default mode `offline`; `allow_external_requests: false`
- exact host allowlist, HTTPS only, no credentials or nonstandard ports
- redirect destination is revalidated against the allowlist and must remain on the original provider host
- resolved non-global IPs are rejected
- configured request timeout, response byte cap, and retry limit
- transient failures only are retried; retry sleeps are bounded
- JSON content type and top-level object required
- bounded TTL cache; query text is not written to logs

Provider endpoints are fixed by provider implementations; there is no arbitrary URL tool. Requests do not carry audio or camera payloads. Personal data is rejected by policy. User-supplied location is required for weather and is never inferred from device location.

The local Ollama loopback path is a separate local-model backend. First-use downloads of model assets also remain a separate legacy path and are not governed by this provider transport; see [external-backends.md](external-backends.md). The offline provider setting does not prevent those existing model downloads.

Unit tests inject fake transport/responses and do not resolve public hosts or use the internet.

`python -m tools.provider_smoke` is the separate opt-in check. It requires `--allow-network` plus explicit confirmation flags and only works after provider policy metadata is approved. Currently both public adapters are blocked before any request.
