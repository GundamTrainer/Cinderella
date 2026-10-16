from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .interfaces import ProviderExecutionContext, ProviderResult
from .network import ProviderRequestError
from .policies import NetworkMode


class ToolDecisionKind(StrEnum):
    ANSWER_LOCALLY = "answer_locally"
    CALL_TOOL = "call_tool"
    ASK_CLARIFICATION = "ask_clarification"
    WAIT_FOR_OBSERVATION = "wait_for_observation"
    DECLINE = "decline"
    REMAIN_SILENT = "remain_silent"


class PrivacyRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    ELEVATED = "elevated"
    PROHIBITED = "prohibited"


@dataclass(frozen=True)
class ToolCallPlan:
    tool_id: str
    purpose: str
    arguments: dict[str, Any]
    user_data_included: bool
    privacy_risk: PrivacyRisk
    expected_source: str
    cache_allowed: bool
    timeout_seconds: float
    fallback: str
    requires_confirmation: bool


@dataclass(frozen=True)
class ToolDecision:
    decision: ToolDecisionKind
    reason: str
    plan: ToolCallPlan | None = None


@dataclass(frozen=True)
class ToolExecutionOutcome:
    result: ProviderResult | None
    fallback_action: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.result is not None


@dataclass(frozen=True)
class ToolDecisionInput:
    user_request: str
    local_answer_available: bool
    request_is_clear: bool
    provider_id: str | None
    arguments: dict[str, Any]
    provider_available: bool
    network_mode: NetworkMode
    cache_hit: bool = False
    contains_personal_data: bool = False
    user_confirmed: bool = False
    timeout_seconds: float = 8.0


def decide_tool_use(request: ToolDecisionInput) -> ToolDecision:
    """Choose one path; never fan out to all available providers."""
    if not request.user_request.strip():
        return ToolDecision(ToolDecisionKind.REMAIN_SILENT, "empty_request")
    if request.local_answer_available:
        return ToolDecision(ToolDecisionKind.ANSWER_LOCALLY, "local_context_is_sufficient")
    if not request.request_is_clear:
        return ToolDecision(ToolDecisionKind.ASK_CLARIFICATION, "request_is_ambiguous")
    if request.contains_personal_data:
        return ToolDecision(ToolDecisionKind.DECLINE, "external_personal_data_transfer_prohibited")
    if request.cache_hit:
        return ToolDecision(ToolDecisionKind.ANSWER_LOCALLY, "valid_cached_result_available")
    if request.network_mode == NetworkMode.OFFLINE:
        return ToolDecision(ToolDecisionKind.ANSWER_LOCALLY, "offline_mode")
    if request.provider_id is None or not request.provider_available:
        return ToolDecision(ToolDecisionKind.ANSWER_LOCALLY, "no_approved_provider_available")
    needs_confirmation = (
        request.network_mode == NetworkMode.ASK_BEFORE_EACH_PROVIDER
        and not request.user_confirmed
    )
    if needs_confirmation:
        plan = ToolCallPlan(
            tool_id=request.provider_id,
            purpose="retrieve_requested_information",
            arguments=request.arguments,
            user_data_included=True,
            privacy_risk=PrivacyRisk.ELEVATED,
            expected_source="external_source",
            cache_allowed=True,
            timeout_seconds=request.timeout_seconds,
            fallback="answer_without_external_data",
            requires_confirmation=True,
        )
        return ToolDecision(ToolDecisionKind.ASK_CLARIFICATION, "provider_confirmation_required", plan)
    plan = ToolCallPlan(
        tool_id=request.provider_id,
        purpose="retrieve_requested_information",
        arguments=request.arguments,
        user_data_included=True,
        privacy_risk=PrivacyRisk.LOW,
        expected_source="external_source",
        cache_allowed=True,
        timeout_seconds=request.timeout_seconds,
        fallback="answer_without_external_data",
        requires_confirmation=False,
    )
    return ToolDecision(ToolDecisionKind.CALL_TOOL, "approved_provider_required_for_fresh_data", plan)


def execute_tool_plan(
    registry: Any,
    plan: ToolCallPlan,
    *,
    context: ProviderExecutionContext,
) -> ToolExecutionOutcome:
    """Return explicit provider failure metadata with the caller-selected local fallback."""
    try:
        result = registry.execute(plan.tool_id, plan.arguments, context=context)
    except (ProviderRequestError, TypeError, ValueError) as exc:
        return ToolExecutionOutcome(
            result=None,
            fallback_action=plan.fallback,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return ToolExecutionOutcome(result=result)
