# Provider policy

Provider activation is default-deny. `ProviderPolicyValidator` rejects disabled providers, payment possibilities, paid quality tiers, card requirements, camera/audio transfer, missing terms/license/rate-limit/attribution/retention metadata, and unconfirmed data retention. There is no override flag.

The provider's `enabled` field must be explicitly true, and the separate network settings must permit requests. `offline` denies external calls. `approved_providers_only` permits only explicitly enabled and policy-valid providers. `ask_before_each_provider` also requires confirmation for each request. `development_test` is for test-marked mocked calls; it does not make public endpoints acceptable.

Provider records must document official HTTPS terms and privacy URLs, license, authentication, rate limits, data transfer, retention, caching, attribution, and an ISO review date no more than 365 days old. Re-review when terms change. A non-empty but unknown retention statement is not adequate; explicit unknown/unverified metadata is rejected.

Current provider outcomes:

- MediaWiki: inactive until query-data retention is verified.
- Open-Meteo: inactive because paid/commercial and paid-data tiers exist.
- Local and static providers: offline only.

These restrictions apply before provider execution; cache reads are local and do not create new network requests.
