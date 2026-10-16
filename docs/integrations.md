# Integrations

No provider is called automatically. The registry is not connected to `main.py` or the live conversation loop. Network defaults are offline and provider metadata must pass `ProviderPolicyValidator` before activation.

Policy metadata reviewed: **2026-10-16**. Provider terms can change; review them again before any activation.

| Provider | Purpose and official documentation | Access, policy, and license | Data, cache, fallback, status |
|---|---|---|---|
| MediaWiki Action API | Short search snippets, page titles, related page URLs, and modification time. [API Search](https://www.mediawiki.org/wiki/API:Search), [API Etiquette](https://www.mediawiki.org/wiki/API:Etiquette) | No key; Wikipedia content generally CC BY-SA 4.0, but page licenses can vary. Descriptive User-Agent; sequential requests; obey server throttling. [Wikimedia API usage](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines), [Terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) | Query text and network metadata leave the device. Retention is not confirmed, so policy validation blocks activation. 15-minute in-memory cache; failure should leave local dialogue available. Last review: 2026-10-16. |
| Open-Meteo | Current weather for explicit user-supplied coordinates. [Forecast API](https://open-meteo.com/en/docs) | Free API is non-commercial and rate-limited to 10,000 calls/day; paid customer API and paid data tiers also exist. Data CC BY 4.0, attribution required. [Pricing](https://open-meteo.com/en/pricing), [Terms](https://open-meteo.com/en/terms) | Coordinates are request parameters; no device geolocation, audio, or camera data. Privacy/retention terms are not confirmed. Provider is blocked because payment and paid quality tiers are possible. 10-minute in-memory cache; no weather guess on failure. Last review: 2026-10-16. |
| Wikidata | Bounded SPARQL query builder only; no network provider yet. [Query Service](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service), [Data access](https://www.wikidata.org/wiki/Wikidata:Data_access) | Query source and rate limits need review before implementation. | No requests, result cache, or fallback currently. Not active. |
| Local system | Local clock and explicitly supplied runtime status. | No authentication, license, network, or external data transfer. | Does not enumerate private files or other processes. No persistence. Local fallback. |
| Static offline knowledge | Search records explicitly provided by the caller or loaded from an approved local JSON file. | Per-record source, date/version, and license required. | No network. Missing facts return no result. No private corpus is bundled. |

All public HTTP tests use a mock transport. No model is presented as self-developed because it is exposed through a provider interface.

For a future provider that has passed policy review, `python -m tools.provider_smoke` requires both `--allow-network` and per-request query/location confirmation. The command temporarily enables only its selected provider in memory; it does not modify `configs/default.yaml`. Current MediaWiki and Open-Meteo checks intentionally stop at policy validation before making a request.
