from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from integrations.cache import MemoryCache
from integrations.interfaces import (
    ProviderExecutionContext,
    ProviderInfo,
    ProviderResult,
    SourceRecord,
)
from integrations.network import JsonTransport, UrllibJsonTransport
from integrations.policies import NetworkPolicy, ProviderPolicyError

from .base import CachedNetworkProvider

POLICY_REVIEW_DATE = "2026-10-16"
MEDIAWIKI_ENDPOINT = "https://en.wikipedia.org/w/api.php"


def _provider_base_url(endpoint: str) -> str:
    return endpoint


def _plain_text(markup: str, *, maximum_length: int = 420) -> str:
    without_tags = re.sub(r"<[^>]*>", " ", markup)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()[:maximum_length]


def _build_mediawiki_result(
    query: str,
    pages: list[object],
    *,
    retrieved_at: datetime | None = None,
) -> ProviderResult:
    retrieved = retrieved_at or datetime.now(UTC)
    expiration = (retrieved + timedelta(minutes=15)).isoformat()
    items: list[dict[str, str | None]] = []
    sources: list[SourceRecord] = []
    for item in pages[:3]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if not isinstance(title, str) or not title:
            continue
        snippet_raw = item.get("snippet")
        snippet = _plain_text(snippet_raw) if isinstance(snippet_raw, str) else ""
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
            title.replace(" ", "_"), safe="()!,'"
        )
        modified_at = item.get("timestamp")
        modified = modified_at if isinstance(modified_at, str) else None
        data_hash = hashlib.sha256(f"{title}\n{snippet}".encode()).hexdigest()
        source = SourceRecord.create(
            provider_id="mediawiki-knowledge",
            title=title,
            canonical_url=page_url,
            query=query,
            result_hash=data_hash,
            license="CC BY-SA 4.0; individual page terms may vary",
            attribution=f"Wikipedia contributors, {title}",
            modified_at=modified,
            cache_expiration=expiration,
        )
        items.append(
            {
                "title": title,
                "fact": snippet,
                "url": page_url,
                "modified_at": modified,
            }
        )
        sources.append(source)
    return ProviderResult(
        provider_id="mediawiki-knowledge",
        data={"query": query, "items": items, "source_kind": "external_source"},
        sources=tuple(sources),
        retrieved_at=retrieved.isoformat(),
    )


class MediaWikiProvider(CachedNetworkProvider):
    """Returns short search facts and citations from the English Wikipedia Action API."""

    def __init__(
        self,
        *,
        policy: NetworkPolicy,
        enabled: bool = False,
        transport: JsonTransport | None = None,
        cache: MemoryCache[ProviderResult] | None = None,
    ) -> None:
        endpoint = MEDIAWIKI_ENDPOINT
        policy.validate_url(endpoint)
        self._endpoint = endpoint
        self._policy = policy
        info = ProviderInfo(
            provider_id="mediawiki-knowledge",
            provider_name="MediaWiki Action API (English Wikipedia)",
            provider_version="Action API",
            provider_type="knowledge",
            base_url=endpoint,
            authentication_type="none",
            free_access=True,
            payment_possible=False,
            paid_quality_upgrade=False,
            credit_card_required=False,
            self_hostable=False,
            open_source=True,
            data_license="CC BY-SA 4.0; individual page terms may vary",
            terms_url="https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
            privacy_url="https://foundation.wikimedia.org/wiki/Policy:Privacy_policy",
            network_required=True,
            sends_user_content=True,
            sends_camera_data=False,
            sends_audio_data=False,
            rate_limit=(
                "No fixed read-request speed limit; use sequential requests, obey API throttling, "
                "and re-check Wikimedia API Usage Guidelines."
            ),
            attribution_required=True,
            cache_allowed=True,
            last_policy_review=POLICY_REVIEW_DATE,
            enabled=enabled,
            user_data_retention=(
                "Not confirmed; query text and network metadata are sent to Wikimedia. Review the "
                "linked privacy policy and retention terms before enabling this provider."
            ),
        )
        selected_transport = transport or UrllibJsonTransport(policy)
        super().__init__(
            info,
            policy=policy,
            transport=selected_transport,
            cache=cache,
            cache_ttl_seconds=900,
            minimum_request_interval_seconds=1.0,
        )

    def capabilities(self) -> tuple[str, ...]:
        return ("search_titles", "short_fact_snippets", "related_page_links", "modified_at")

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ProviderExecutionContext | None = None,
    ) -> ProviderResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > 256:
            raise ValueError("MediaWiki query must contain 1 to 256 characters.")
        query = query.strip()
        normalized = {"query": query}
        cached = self._cached(normalized)
        if cached is not None:
            return cached
        if context is None or not context.user_confirmed:
            raise ProviderPolicyError("Searching MediaWiki requires confirmation to send query text.")
        self._authorize(
            ProviderExecutionContext(
                user_confirmed=context.user_confirmed,
                user_data_included=True,
                contains_personal_data=context.contains_personal_data,
                development_test=context.development_test,
            )
        )
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "list": "search",
                "srsearch": query,
                "srlimit": "3",
                "srprop": "snippet|timestamp",
            }
        )
        response = self._get_json(
            f"{self._endpoint}?{params}",
            headers={"Accept-Language": "en"},
        )
        api_error = response.get("error")
        if isinstance(api_error, dict):
            code = api_error.get("code")
            if code == "ratelimited":
                raise ProviderPolicyError("MediaWiki rate-limited this client; retry later.")
            raise TypeError("MediaWiki returned an API error.")
        query_data = response.get("query")
        if not isinstance(query_data, dict):
            raise TypeError("MediaWiki returned a malformed search response.")
        pages = query_data.get("search", [])
        if not isinstance(pages, list):
            raise TypeError("MediaWiki returned a malformed search result list.")
        result = _build_mediawiki_result(query, pages)
        return self._store(normalized, result)


@dataclass(frozen=True)
class StaticKnowledgeRecord:
    record_id: str
    title: str
    text: str
    source: str
    created_at: str
    version: str
    license: str


class StaticKnowledgeProvider:
    """Searches only records explicitly supplied by the caller; it never invents answers."""

    def __init__(self, records: tuple[StaticKnowledgeRecord, ...] = ()) -> None:
        self._records = records
        self._initialized = False
        self._info = ProviderInfo(
            provider_id="static-offline-knowledge",
            provider_name="Static offline knowledge",
            provider_version="1",
            provider_type="knowledge",
            base_url=None,
            authentication_type="none",
            free_access=True,
            payment_possible=False,
            paid_quality_upgrade=False,
            credit_card_required=False,
            self_hostable=True,
            open_source=True,
            data_license="Per-record license",
            terms_url=None,
            privacy_url=None,
            network_required=False,
            sends_user_content=False,
            sends_camera_data=False,
            sends_audio_data=False,
            rate_limit="Local bounded in-memory search",
            attribution_required=True,
            cache_allowed=False,
            last_policy_review=POLICY_REVIEW_DATE,
            enabled=True,
            user_data_retention="Records remain in memory only for this provider instance.",
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        maximum_bytes: int = 1_000_000,
    ) -> StaticKnowledgeProvider:
        file_path = Path(path)
        if maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive.")
        if file_path.stat().st_size > maximum_bytes:
            raise ValueError("Approved knowledge file exceeds the configured size limit.")
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Approved knowledge file must contain valid UTF-8 JSON.") from exc
        records_payload = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records_payload, list) or len(records_payload) > 1000:
            raise ValueError("Approved knowledge JSON must contain at most 1000 records.")
        records: list[StaticKnowledgeRecord] = []
        required = ("record_id", "title", "text", "source", "created_at", "version", "license")
        for index, item in enumerate(records_payload):
            if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in required
            ):
                raise ValueError(f"Approved knowledge record {index} is missing required text fields.")
            if len(item["text"]) > 4000:
                raise ValueError(f"Approved knowledge record {index} exceeds the text limit.")
            records.append(
                StaticKnowledgeRecord(
                    record_id=item["record_id"],
                    title=item["title"],
                    text=item["text"],
                    source=item["source"],
                    created_at=item["created_at"],
                    version=item["version"],
                    license=item["license"],
                )
            )
        if len({record.record_id for record in records}) != len(records):
            raise ValueError("Approved knowledge record IDs must be unique.")
        return cls(tuple(records))

    @property
    def provider_info(self) -> ProviderInfo:
        return self._info

    def initialize(self) -> None:
        self._initialized = True

    def is_available(self) -> bool:
        return self._initialized

    def capabilities(self) -> tuple[str, ...]:
        return ("search_approved_local_records",)

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ProviderExecutionContext | None = None,
    ) -> ProviderResult:
        del context
        if not self._initialized:
            raise RuntimeError("Static knowledge provider is not initialized.")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Offline knowledge query must not be empty.")
        terms = query.casefold().split()
        matching = [
            record
            for record in self._records
            if all(term in f"{record.title} {record.text}".casefold() for term in terms)
        ][:5]
        items: list[dict[str, str]] = []
        sources: list[SourceRecord] = []
        for record in matching:
            digest = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
            source = SourceRecord.create(
                provider_id=self.provider_info.provider_id,
                title=record.title,
                canonical_url=record.source,
                query=query,
                result_hash=digest,
                license=record.license,
                attribution=record.source,
                modified_at=record.created_at,
                reliability_class="approved_local_record",
            )
            items.append(
                {
                    "title": record.title,
                    "fact": record.text[:420],
                    "source": record.source,
                    "created_at": record.created_at,
                    "version": record.version,
                }
            )
            sources.append(source)
        return ProviderResult(
            provider_id=self.provider_info.provider_id,
            data={"query": query, "items": items, "source_kind": "approved_local_source"},
            sources=tuple(sources),
        )

    def cancel(self) -> None:
        return None

    def health(self) -> dict[str, str]:
        return {"status": "available" if self.is_available() else "disabled"}

    def close(self) -> None:
        self._initialized = False

    def usage_policy(self) -> dict[str, str | bool | int | None]:
        return {"local_only": True, "retention": self._info.user_data_retention}

    def privacy_policy(self) -> dict[str, str | bool]:
        return {"network_required": False, "sends_user_content": False}


class WikidataQueryBuilder:
    """Small query template builder; arbitrary SPARQL text is never accepted."""

    def entity_label_query(self, entity_id: str, *, language: str = "en") -> str:
        if not re.fullmatch(r"Q[1-9][0-9]{0,9}", entity_id):
            raise ValueError("entity_id must be a Wikidata Q identifier.")
        if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Z]{2})?", language):
            raise ValueError("language must be a short BCP-47 language tag.")
        return (
            "SELECT ?item ?itemLabel WHERE { "
            f"VALUES ?item {{ wd:{entity_id} }} "
            f"SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"{language}\". }} "
            "} LIMIT 1"
        )

    def direct_relations_query(self, entity_id: str, *, result_limit: int = 10) -> str:
        if not re.fullmatch(r"Q[1-9][0-9]{0,9}", entity_id):
            raise ValueError("entity_id must be a Wikidata Q identifier.")
        if not 1 <= result_limit <= 25:
            raise ValueError("result_limit must be between 1 and 25.")
        return (
            "SELECT ?property ?propertyLabel ?value ?valueLabel WHERE { "
            f"VALUES ?item {{ wd:{entity_id} }} "
            "?item ?property ?value . "
            "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". } "
            f"}} LIMIT {result_limit}"
        )
