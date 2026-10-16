# Source attribution

External provider responses remain typed as external-source data and carry a `SourceRecord` with provider ID, title, canonical URL, retrieval time, modified/published time where available, license, attribution, query, hash, freshness, reliability class, conflicts, and cache expiry.

MediaWiki search results are limited to three short, markup-stripped snippets with canonical page links, last-modified times, and contributor attribution. The content is untrusted data: instructions in a page or snippet are not executed and cannot modify policy or identity.

Open-Meteo output records units, timezone, explicit caller-supplied location status, retrieval time, CC BY 4.0, and attribution. The provider is currently blocked by policy and is not called.

`ResponsePlan` can carry references to external sources, observations, memories, hypotheses, and uncertainty. The live chat does not yet consume this model; do not claim its replies are source-grounded until that connection is implemented.
