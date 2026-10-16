from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit

from integrations.cache import MemoryCache, cache_key
from integrations.interfaces import (
    ProviderExecutionContext,
    ProviderInfo,
    ProviderResult,
)
from integrations.network import JsonTransport, ProviderRequestError
from integrations.policies import (
    NetworkMode,
    NetworkPolicy,
    ProviderPolicyError,
    ProviderPolicyValidator,
)


class CachedNetworkProvider:
    def __init__(
        self,
        info: ProviderInfo,
        *,
        policy: NetworkPolicy,
        transport: JsonTransport,
        cache: MemoryCache[ProviderResult] | None = None,
        cache_ttl_seconds: float = 300.0,
        minimum_request_interval_seconds: float = 0.0,
    ) -> None:
        self._info = info
        self._policy = policy
        self._transport = transport
        self._cache = cache if cache is not None else MemoryCache()
        self._cache_ttl_seconds = cache_ttl_seconds
        if minimum_request_interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds must not be negative.")
        self._minimum_request_interval_seconds = minimum_request_interval_seconds
        self._request_lock = threading.Lock()
        self._last_request_at: float | None = None
        self._initialized = False
        self._cancelled = False

    @property
    def provider_info(self) -> ProviderInfo:
        return self._info

    def initialize(self) -> None:
        if self._info.enabled:
            ProviderPolicyValidator.validate(self._info)
        self._initialized = True

    def is_available(self) -> bool:
        if not self._initialized or not self._info.enabled:
            return False
        return (
            self._policy.allow_external_requests
            and self._policy.mode
            in {NetworkMode.APPROVED_PROVIDERS_ONLY, NetworkMode.ASK_BEFORE_EACH_PROVIDER}
        )

    def cancel(self) -> None:
        self._cancelled = True

    def health(self) -> dict[str, str]:
        return {
            "status": "available" if self.is_available() else "disabled",
            "provider_id": self._info.provider_id,
        }

    def close(self) -> None:
        self._initialized = False
        self._cancelled = False

    def usage_policy(self) -> dict[str, str | bool | int | None]:
        return {
            "free_access": self._info.free_access,
            "payment_possible": self._info.payment_possible,
            "credit_card_required": self._info.credit_card_required,
            "rate_limit": self._info.rate_limit,
            "terms_url": self._info.terms_url,
            "data_license": self._info.data_license,
        }

    def privacy_policy(self) -> dict[str, str | bool]:
        return {
            "sends_user_content": self._info.sends_user_content,
            "sends_camera_data": self._info.sends_camera_data,
            "sends_audio_data": self._info.sends_audio_data,
            "user_data_retention": self._info.user_data_retention or "unknown",
        }

    def _cached(self, arguments: dict[str, object]) -> ProviderResult | None:
        if not self._policy.cache_enabled or not self._info.cache_allowed:
            return None
        cached = self._cache.get(cache_key(self._info.provider_id, arguments))
        return None if cached is None else ProviderResult(
            provider_id=cached.provider_id,
            data=cached.data,
            sources=cached.sources,
            retrieved_at=cached.retrieved_at,
            from_cache=True,
        )

    def _store(self, arguments: dict[str, object], result: ProviderResult) -> ProviderResult:
        if self._policy.cache_enabled and self._info.cache_allowed:
            self._cache.put(
                cache_key(self._info.provider_id, arguments),
                result,
                ttl_seconds=self._cache_ttl_seconds,
            )
        return result

    def _authorize(
        self,
        context: ProviderExecutionContext | None,
    ) -> ProviderExecutionContext:
        if not self._initialized:
            raise RuntimeError(f"Provider {self._info.provider_id!r} is not initialized.")
        if self._cancelled:
            raise RuntimeError(f"Provider {self._info.provider_id!r} was cancelled.")
        resolved = context or ProviderExecutionContext()
        ProviderPolicyValidator.validate(self._info)
        self._policy.authorize_provider(self._info, resolved)
        return resolved

    def _get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        last_error: ProviderRequestError | None = None
        with self._request_lock:
            for attempt in range(self._policy.retry_count + 1):
                if self._cancelled:
                    raise RuntimeError(f"Provider {self._info.provider_id!r} was cancelled.")
                if self._last_request_at is not None:
                    remaining = (
                        self._last_request_at
                        + self._minimum_request_interval_seconds
                        - time.monotonic()
                    )
                    if remaining > 0:
                        time.sleep(remaining)
                self._last_request_at = time.monotonic()
                try:
                    response = self._transport.get_json(url, headers=headers)
                    expected_host = urlsplit(url).hostname
                    result_host = urlsplit(response.final_url).hostname
                    if result_host != expected_host:
                        raise ProviderPolicyError(
                            "Provider response redirected to another allowlisted host."
                        )
                    return response.data
                except ProviderRequestError as exc:
                    last_error = exc
                    if attempt >= self._policy.retry_count or not exc.retryable:
                        break
                    time.sleep(min(0.1 * (2**attempt), 1.0))
        if last_error is None:
            raise RuntimeError("Provider request loop ended without a result.")
        raise last_error

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()
