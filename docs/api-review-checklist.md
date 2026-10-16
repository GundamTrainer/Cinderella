# API review checklist

Before enabling a provider, verify and record:

- [ ] Official provider documentation, terms, and privacy policy URLs
- [ ] Free access, all paid tiers/quality upgrades, card requirements, and any automatic billing
- [ ] License and required attribution for every response type
- [ ] Current numeric or clearly described rate limits and retry/backoff behavior
- [ ] Whether query, IP, or other user information is transmitted and its retention
- [ ] No audio, camera frames, face data, or personal data are sent
- [ ] Fixed HTTPS endpoint and exact allowlist host; redirects are rechecked
- [ ] Timeout, response-size limit, content-type validation, and bounded retries
- [ ] Cache permissions and expiry; no automatic personal-memory writes
- [ ] Offline behavior, local fallback, source record, and conflict reporting
- [ ] Mock-only tests for success, errors, timeouts, redirects, and oversized output
- [ ] User-facing opt-in/confirmation and a way to disable/delete provider data
- [ ] Review date and an owner for future policy re-check

Any unknown or unmet requirement leaves the provider disabled. Do not create a hidden override.
