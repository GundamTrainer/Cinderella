from __future__ import annotations

from collections.abc import Iterable

from integrations.interfaces import (
    JSONValue,
    ProviderExecutionContext,
    ProviderResult,
    ToolProvider,
)
from integrations.policies import NetworkMode, NetworkPolicy, ProviderPolicyError


class ProviderRegistry:
    """Explicit provider registry. Registration alone never authorizes network access."""

    def __init__(self, policy: NetworkPolicy) -> None:
        self._policy = policy
        self._providers: dict[str, ToolProvider] = {}
        self._initialized: list[ToolProvider] = []
        self._initialization_errors: dict[str, str] = {}

    def register(self, provider: ToolProvider) -> None:
        provider_id = provider.provider_info.provider_id
        if provider_id in self._providers:
            raise ValueError(f"Provider {provider_id!r} is already registered.")
        self._providers[provider_id] = provider

    def register_all(self, providers: Iterable[ToolProvider]) -> None:
        for provider in providers:
            self.register(provider)

    def initialize(self) -> None:
        for provider_id, provider in self._providers.items():
            info = provider.provider_info
            if info.network_required and not info.enabled:
                self._initialization_errors[provider_id] = "disabled by provider configuration"
                continue
            try:
                provider.initialize()
                self._initialized.append(provider)
            except Exception as exc:  # noqa: BLE001 - isolate provider failure and expose health
                self._initialization_errors[provider_id] = (
                    f"{type(exc).__name__}: {exc}"
                )
                try:
                    provider.close()
                except Exception as cleanup_error:  # noqa: BLE001 - retain cleanup detail
                    self._initialization_errors[provider_id] += (
                        f"; cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )

    @property
    def initialization_errors(self) -> dict[str, str]:
        return dict(self._initialization_errors)

    def health(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for provider_id, provider in self._providers.items():
            if provider_id in self._initialization_errors:
                result[provider_id] = {
                    "status": "failed" if provider.provider_info.enabled else "disabled",
                    "detail": self._initialization_errors[provider_id],
                }
            else:
                result[provider_id] = provider.health()
        return result

    def get(self, provider_id: str) -> ToolProvider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"Provider {provider_id!r} is not registered.") from exc

    def available_providers(self) -> tuple[str, ...]:
        return tuple(
            provider_id
            for provider_id, provider in self._providers.items()
            if provider.is_available()
        )

    def execute(
        self,
        provider_id: str,
        arguments: dict[str, JSONValue],
        *,
        context: ProviderExecutionContext | None = None,
    ) -> ProviderResult:
        provider = self.get(provider_id)
        if not provider.is_available():
            raise ProviderPolicyError(f"Provider {provider_id!r} is not available.")
        resolved_context = context or ProviderExecutionContext()
        info = provider.provider_info
        if (
            self._policy.mode == NetworkMode.DEVELOPMENT_TEST
            and info.network_required
            and not resolved_context.development_test
        ):
            raise ProviderPolicyError("Development-test mode requires an explicitly marked test call.")
        return provider.execute(arguments, context=resolved_context)

    def close(self) -> None:
        errors: list[Exception] = []
        for provider in reversed(self._initialized):
            try:
                provider.close()
            except Exception as exc:  # noqa: BLE001 - continue closing remaining providers
                errors.append(exc)
        self._initialized.clear()
        if errors:
            raise ExceptionGroup("One or more providers failed to close.", errors)
