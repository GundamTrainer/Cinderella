from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from math import isfinite
from typing import ClassVar
from urllib.parse import urlsplit

from .interfaces import ProviderExecutionContext, ProviderInfo


class ProviderPolicyError(PermissionError):
    """Raised when provider metadata or a request violates local policy."""


class NetworkMode(StrEnum):
    OFFLINE = "offline"
    APPROVED_PROVIDERS_ONLY = "approved_providers_only"
    ASK_BEFORE_EACH_PROVIDER = "ask_before_each_provider"
    DEVELOPMENT_TEST = "development_test"


@dataclass(frozen=True)
class NetworkPolicy:
    mode: NetworkMode = NetworkMode.OFFLINE
    allow_external_requests: bool = False
    allowed_domains: tuple[str, ...] = ()
    require_user_confirmation_for_new_domain: bool = True
    request_timeout_seconds: float = 8.0
    retry_count: int = 1
    maximum_response_bytes: int = 1_000_000
    user_agent: str = "Cinderella/0.1.0"
    cache_enabled: bool = True
    redact_query_logs: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.allow_external_requests, bool):
            raise TypeError("allow_external_requests must be a boolean.")
        if not isinstance(self.require_user_confirmation_for_new_domain, bool):
            raise TypeError("require_user_confirmation_for_new_domain must be a boolean.")
        if not isinstance(self.cache_enabled, bool) or not isinstance(self.redact_query_logs, bool):
            raise TypeError("cache_enabled and redact_query_logs must be booleans.")
        try:
            mode = NetworkMode(self.mode)
        except ValueError as exc:
            raise ValueError(f"Unsupported network mode: {self.mode!r}.") from exc
        object.__setattr__(self, "mode", mode)
        if (
            isinstance(self.request_timeout_seconds, bool)
            or not isinstance(self.request_timeout_seconds, (int, float))
            or not isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be greater than zero.")
        if (
            isinstance(self.retry_count, bool)
            or not isinstance(self.retry_count, int)
            or not 0 <= self.retry_count <= 5
        ):
            raise ValueError("retry_count must be between 0 and 5.")
        if (
            isinstance(self.maximum_response_bytes, bool)
            or not isinstance(self.maximum_response_bytes, int)
            or self.maximum_response_bytes <= 0
        ):
            raise ValueError("maximum_response_bytes must be greater than zero.")
        if not self.user_agent.strip():
            raise ValueError("user_agent must not be empty.")
        if not isinstance(self.allowed_domains, (tuple, list)) or any(
            not isinstance(domain, str) for domain in self.allowed_domains
        ):
            raise TypeError("allowed_domains must be a tuple or list of hostnames.")
        normalized = tuple(domain.casefold().rstrip(".") for domain in self.allowed_domains)
        if any(
            not domain
            or not re.fullmatch(r"[a-z0-9.-]+", domain)
            or domain.startswith(".")
            or domain.endswith(".")
            or ".." in domain
            for domain in normalized
        ):
            raise ValueError("allowed_domains must contain hostnames only.")
        object.__setattr__(self, "allowed_domains", normalized)

    def validate_url(self, url: str) -> None:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold().rstrip(".")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ProviderPolicyError("Provider URL has an invalid port.") from exc
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or port not in (None, 443)
            or parsed.fragment
        ):
            raise ProviderPolicyError("Only credential-free HTTPS provider URLs are permitted.")
        if host not in self.allowed_domains:
            raise ProviderPolicyError(f"Provider host {host!r} is not on the network allowlist.")

    def authorize_provider(
        self,
        info: ProviderInfo,
        context: ProviderExecutionContext | None = None,
    ) -> None:
        context = context or ProviderExecutionContext()
        if not info.network_required:
            return
        if self.mode == NetworkMode.OFFLINE or not self.allow_external_requests:
            raise ProviderPolicyError("External requests are disabled by offline network policy.")
        if self.mode == NetworkMode.DEVELOPMENT_TEST and not context.development_test:
            raise ProviderPolicyError("Development-test mode only permits explicitly marked test calls.")
        if self.mode == NetworkMode.ASK_BEFORE_EACH_PROVIDER and not context.user_confirmed:
            raise ProviderPolicyError("This provider request requires per-request user confirmation.")
        if not info.enabled:
            raise ProviderPolicyError(f"Provider {info.provider_id!r} is disabled.")
        if context.contains_personal_data:
            raise ProviderPolicyError("Personal data must not be sent to external providers.")
        if (
            context.user_data_included
            and self.mode == NetworkMode.ASK_BEFORE_EACH_PROVIDER
            and not context.user_confirmed
        ):
            raise ProviderPolicyError("Sending user-provided query text requires user confirmation.")
        if info.base_url:
            self.validate_url(info.base_url)


class ProviderPolicyValidator:
    _required_metadata: ClassVar[dict[str, str]] = {
        "data_license": "data license",
        "terms_url": "terms URL",
        "privacy_url": "privacy URL",
        "last_policy_review": "policy review date",
        "user_data_retention": "user-data retention policy",
    }

    @classmethod
    def validate(cls, info: ProviderInfo) -> None:
        if not info.enabled:
            raise ProviderPolicyError(f"Provider {info.provider_id!r} is disabled by default.")
        failures: list[str] = []
        if not info.free_access:
            failures.append("free access is not confirmed")
        if info.payment_possible:
            failures.append("payment is possible")
        if info.paid_quality_upgrade:
            failures.append("paid quality upgrades are available")
        if info.credit_card_required:
            failures.append("a credit card is required")
        if info.sends_camera_data:
            failures.append("camera data would be sent")
        if info.sends_audio_data:
            failures.append("audio data would be sent")
        if not info.attribution_required:
            failures.append("attribution obligations are not recorded")
        if not info.rate_limit:
            failures.append("rate limit is not recorded")
        if info.network_required and not info.base_url:
            failures.append("provider endpoint is not recorded")
        for field_name, label in cls._required_metadata.items():
            value = getattr(info, field_name)
            if not isinstance(value, str) or not value.strip():
                failures.append(f"{label} is not recorded")
            elif field_name == "user_data_retention" and any(
                marker in value.casefold()
                for marker in ("unknown", "not confirmed", "must be reviewed", "unverified")
            ):
                failures.append("user-data retention policy is not confirmed")
            elif field_name == "last_policy_review":
                try:
                    reviewed = date.fromisoformat(value)
                except ValueError:
                    failures.append("policy review date is invalid")
                else:
                    if reviewed > date.today() or date.today() - reviewed > timedelta(days=365):
                        failures.append("policy review is stale")
            elif field_name in {"terms_url", "privacy_url"}:
                parsed = urlsplit(value)
                if parsed.scheme != "https" or not parsed.hostname:
                    failures.append(f"{label} must be an HTTPS URL")
        if failures:
            raise ProviderPolicyError(
                f"Provider {info.provider_id!r} failed policy validation: "
                + "; ".join(failures)
                + "."
            )
