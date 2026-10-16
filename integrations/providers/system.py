from __future__ import annotations

from datetime import datetime
from time import monotonic
from typing import Any

from integrations.interfaces import ProviderExecutionContext, ProviderInfo, ProviderResult

REVIEW_DATE = "2026-10-16"


class LocalSystemProvider:
    """Reports only explicitly supplied runtime status and local clock information."""

    def __init__(
        self,
        *,
        started_at_monotonic: float | None = None,
        runtime_status: dict[str, str] | None = None,
    ) -> None:
        self._started = monotonic() if started_at_monotonic is None else started_at_monotonic
        self._runtime_status = dict(runtime_status or {})
        self._initialized = False
        self._info = ProviderInfo(
            provider_id="local-system",
            provider_name="Local system status",
            provider_version="1",
            provider_type="system",
            base_url=None,
            authentication_type="none",
            free_access=True,
            payment_possible=False,
            paid_quality_upgrade=False,
            credit_card_required=False,
            self_hostable=True,
            open_source=True,
            data_license="Local computation",
            terms_url="https://www.python.org/psf/license/",
            privacy_url="https://www.python.org/psf/privacy/",
            network_required=False,
            sends_user_content=False,
            sends_camera_data=False,
            sends_audio_data=False,
            rate_limit="Local-only",
            attribution_required=True,
            cache_allowed=False,
            last_policy_review=REVIEW_DATE,
            enabled=True,
            user_data_retention="No data is persisted by this provider.",
        )

    @property
    def provider_info(self) -> ProviderInfo:
        return self._info

    def initialize(self) -> None:
        self._initialized = True

    def is_available(self) -> bool:
        return self._initialized

    def capabilities(self) -> tuple[str, ...]:
        return ("local_time", "local_date", "uptime", "runtime_status")

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        context: ProviderExecutionContext | None = None,
    ) -> ProviderResult:
        del arguments, context
        if not self._initialized:
            raise RuntimeError("Local system provider is not initialized.")
        now = datetime.now().astimezone()
        return ProviderResult(
            provider_id=self.provider_info.provider_id,
            data={
                "local_datetime": now.isoformat(),
                "local_date": now.date().isoformat(),
                "uptime_seconds": max(0.0, monotonic() - self._started),
                "runtime_status": dict(self._runtime_status),
                "source_kind": "local_computation",
            },
        )

    def cancel(self) -> None:
        return None

    def health(self) -> dict[str, str]:
        return {"status": "available" if self.is_available() else "disabled"}

    def close(self) -> None:
        self._initialized = False

    def usage_policy(self) -> dict[str, str | bool | int | None]:
        return {"network_required": False, "retention": self._info.user_data_retention}

    def privacy_policy(self) -> dict[str, str | bool]:
        return {"network_required": False, "reads_private_files": False}
