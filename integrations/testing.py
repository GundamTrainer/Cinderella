from __future__ import annotations

from typing import Any

from .interfaces import ProviderExecutionContext, ProviderInfo, ProviderResult


class MockProvider:
    """Deterministic offline provider intended only for tests and development."""

    def __init__(
        self,
        *,
        provider_id: str = "mock",
        results: dict[str, ProviderResult] | None = None,
    ) -> None:
        self._results = dict(results or {})
        self._initialized = False
        self._info = ProviderInfo(
            provider_id=provider_id,
            provider_name="Mock provider",
            provider_version="1",
            provider_type="mock",
            base_url=None,
            authentication_type="none",
            free_access=True,
            payment_possible=False,
            paid_quality_upgrade=False,
            credit_card_required=False,
            self_hostable=True,
            open_source=True,
            data_license="Synthetic test data",
            terms_url="https://example.invalid/terms",
            privacy_url="https://example.invalid/privacy",
            network_required=False,
            sends_user_content=False,
            sends_camera_data=False,
            sends_audio_data=False,
            rate_limit="Local deterministic test fixture",
            attribution_required=True,
            cache_allowed=False,
            last_policy_review="2025-05-05",
            enabled=True,
            user_data_retention="No retention outside the test process.",
        )

    @property
    def provider_info(self) -> ProviderInfo:
        return self._info

    def initialize(self) -> None:
        self._initialized = True

    def is_available(self) -> bool:
        return self._initialized

    def capabilities(self) -> tuple[str, ...]:
        return ("deterministic_mock",)

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ProviderExecutionContext | None = None,
    ) -> ProviderResult:
        del context
        if not self._initialized:
            raise RuntimeError("Mock provider is not initialized.")
        key = str(arguments.get("query", ""))
        if key not in self._results:
            raise LookupError(f"No deterministic mock result for query {key!r}.")
        return self._results[key]

    def cancel(self) -> None:
        return None

    def health(self) -> dict[str, str]:
        return {"status": "available" if self.is_available() else "disabled"}

    def close(self) -> None:
        self._initialized = False

    def usage_policy(self) -> dict[str, str | bool | int | None]:
        return {"local_only": True, "synthetic": True}

    def privacy_policy(self) -> dict[str, str | bool]:
        return {"network_required": False, "sends_user_content": False}
