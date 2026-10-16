from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class ProviderInfo:
    provider_id: str
    provider_name: str
    provider_version: str
    provider_type: str
    base_url: str | None
    authentication_type: str
    free_access: bool
    payment_possible: bool
    paid_quality_upgrade: bool
    credit_card_required: bool
    self_hostable: bool
    open_source: bool
    data_license: str | None
    terms_url: str | None
    privacy_url: str | None
    network_required: bool
    sends_user_content: bool
    sends_camera_data: bool
    sends_audio_data: bool
    rate_limit: str | None
    attribution_required: bool
    cache_allowed: bool
    last_policy_review: str | None
    enabled: bool = False
    user_data_retention: str | None = None

    def __post_init__(self) -> None:
        if not self.provider_id.strip() or not self.provider_name.strip():
            raise ValueError("provider_id and provider_name must not be empty.")
        for name in (
            "free_access",
            "payment_possible",
            "paid_quality_upgrade",
            "credit_card_required",
            "self_hostable",
            "open_source",
            "network_required",
            "sends_user_content",
            "sends_camera_data",
            "sends_audio_data",
            "attribution_required",
            "cache_allowed",
            "enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        if self.rate_limit is not None and not self.rate_limit.strip():
            raise ValueError("rate_limit must be non-empty when specified.")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    provider_id: str
    title: str
    canonical_url: str
    retrieved_at: str
    published_or_modified_at: str | None
    license: str
    attribution: str
    query: str
    result_hash: str
    freshness: str
    reliability_class: str
    conflicting_sources: tuple[str, ...] = ()
    cache_expiration: str | None = None

    @classmethod
    def create(
        cls,
        *,
        provider_id: str,
        title: str,
        canonical_url: str,
        query: str,
        result_hash: str,
        license: str,
        attribution: str,
        modified_at: str | None = None,
        cache_expiration: str | None = None,
        reliability_class: str = "external_source_unverified",
    ) -> SourceRecord:
        import uuid

        return cls(
            source_id=uuid.uuid4().hex,
            provider_id=provider_id,
            title=title,
            canonical_url=canonical_url,
            retrieved_at=datetime.now(UTC).isoformat(),
            published_or_modified_at=modified_at,
            license=license,
            attribution=attribution,
            query=query,
            result_hash=result_hash,
            freshness="retrieved_now",
            reliability_class=reliability_class,
            cache_expiration=cache_expiration,
        )


@dataclass(frozen=True)
class ProviderResult:
    provider_id: str
    data: dict[str, JSONValue]
    sources: tuple[SourceRecord, ...] = ()
    retrieved_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    from_cache: bool = False

    def __post_init__(self) -> None:
        try:
            json.dumps(self.data, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Provider result data must contain only finite JSON values.") from exc


@dataclass(frozen=True)
class ProviderExecutionContext:
    user_confirmed: bool = False
    user_data_included: bool = False
    contains_personal_data: bool = False
    development_test: bool = False


class ProviderClarificationRequired(ValueError):
    """Raised when a provider needs additional explicit user input before a request."""


@runtime_checkable
class ToolProvider(Protocol):
    @property
    def provider_info(self) -> ProviderInfo: ...

    def initialize(self) -> None: ...

    def is_available(self) -> bool: ...

    def capabilities(self) -> tuple[str, ...]: ...

    def execute(
        self,
        arguments: dict[str, JSONValue],
        *,
        context: ProviderExecutionContext | None = None,
    ) -> ProviderResult: ...

    def cancel(self) -> None: ...

    def health(self) -> dict[str, str]: ...

    def close(self) -> None: ...

    def usage_policy(self) -> dict[str, str | bool | int | None]: ...

    def privacy_policy(self) -> dict[str, str | bool]: ...
