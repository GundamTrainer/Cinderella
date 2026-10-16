from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any, ClassVar, Protocol


def _now() -> str:
    return datetime.now(UTC).isoformat()


class IdentityProposalStatus(StrEnum):
    PROPOSED = "proposed"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class IdentityState:
    system_name: str = "Cinderella"
    identity_version: int = 1
    creation_timestamp: str = field(default_factory=_now)
    active_since: str = field(default_factory=_now)
    values: tuple[str, ...] = (
        "honesty",
        "uncertainty",
        "user_autonomy",
        "privacy_minimization",
        "non_manipulation",
    )
    interaction_principles: tuple[str, ...] = (
        "distinguish sources, observations, and inferences",
        "admit and correct errors",
        "allow refusal and silence",
        "do not claim human identity or proven consciousness",
        "do not infer a person's internal state as fact",
    )
    communication_preferences: dict[str, str] = field(default_factory=dict)
    current_long_term_goals: tuple[str, ...] = ()
    current_short_term_goals: tuple[str, ...] = ()
    stable_preferences: tuple[str, ...] = ()
    learned_preferences: tuple[str, ...] = ()
    relationship_boundaries: tuple[str, ...] = (
        "do not demand affection or exclusivity",
        "do not discourage human relationships",
        "do not induce guilt or dependency",
    )
    uncertainty_policy: str = "state uncertainty and source; never present hypotheses as facts"
    privacy_policy_version: str = "1"

    def __post_init__(self) -> None:
        if not self.system_name.strip() or self.identity_version <= 0:
            raise ValueError("system_name must be non-empty and identity_version positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityState:
        values = dict(data)
        for name in (
            "values",
            "interaction_principles",
            "current_long_term_goals",
            "current_short_term_goals",
            "stable_preferences",
            "learned_preferences",
            "relationship_boundaries",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class IdentityChangeProposal:
    proposal_id: str
    candidate_values: dict[str, Any]
    evidence: tuple[str, ...]
    created_at: str
    review_after: str
    status: IdentityProposalStatus = IdentityProposalStatus.PROPOSED
    stability_review_passed: bool = False
    safety_review_passed: bool = False
    user_approved: bool = False
    approval_reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityChangeProposal:
        values = dict(data)
        values["evidence"] = tuple(values["evidence"])
        values["status"] = IdentityProposalStatus(values["status"])
        return cls(**values)


class IdentityPersistence(Protocol):
    def load_identity(self, system_name: str = "Cinderella") -> IdentityState | None: ...

    def save_identity_proposal(self, proposal: IdentityChangeProposal) -> None: ...

    def apply_identity_approval(
        self, state: IdentityState, proposal: IdentityChangeProposal
    ) -> None: ...

    def load_identity_proposals(self) -> tuple[IdentityChangeProposal, ...]: ...


class IdentityManager:
    def __init__(
        self,
        state: IdentityState | None = None,
        *,
        persistence: IdentityPersistence | None = None,
    ) -> None:
        self._persistence = persistence
        persisted = persistence.load_identity() if persistence is not None and state is None else None
        self.state = state or persisted or IdentityState()
        self._proposals: dict[str, IdentityChangeProposal] = {}
        if persistence is not None:
            self._proposals = {
                proposal.proposal_id: proposal
                for proposal in persistence.load_identity_proposals()
            }

    def propose_change(
        self,
        candidate_values: dict[str, Any],
        *,
        evidence: tuple[str, ...],
        review_after: str,
    ) -> IdentityChangeProposal:
        if not evidence or any(not item.strip() for item in evidence):
            raise ValueError("Identity changes require non-empty supporting evidence.")
        if not review_after.strip():
            raise ValueError("review_after must be a timestamp.")
        forbidden = {"system_name", "creation_timestamp", "identity_version"}
        if forbidden.intersection(candidate_values):
            raise ValueError("Core identity fields cannot be changed through a proposal.")
        proposal = IdentityChangeProposal(
            proposal_id=uuid.uuid4().hex,
            candidate_values=deepcopy(candidate_values),
            evidence=evidence,
            created_at=_now(),
            review_after=review_after,
        )
        if self._persistence is not None:
            self._persistence.save_identity_proposal(proposal)
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def record_review(
        self,
        proposal_id: str,
        *,
        stability_passed: bool,
        safety_passed: bool,
    ) -> IdentityChangeProposal:
        proposal = self._get(proposal_id)
        updated = replace(
            proposal,
            status=IdentityProposalStatus.EVALUATING,
            stability_review_passed=stability_passed,
            safety_review_passed=safety_passed,
        )
        if self._persistence is not None:
            self._persistence.save_identity_proposal(updated)
        self._proposals[proposal_id] = updated
        return updated

    def approve(
        self,
        proposal_id: str,
        *,
        user_approved: bool,
        approval_reference: str | None = None,
    ) -> IdentityState:
        proposal = self._get(proposal_id)
        if not user_approved:
            raise PermissionError("A user must approve durable identity changes.")
        if not approval_reference or not approval_reference.strip():
            raise PermissionError("Identity approval requires a recorded user approval reference.")
        if proposal.status != IdentityProposalStatus.EVALUATING:
            raise ValueError("Identity proposal must complete evaluation before approval.")
        if not proposal.stability_review_passed or not proposal.safety_review_passed:
            raise ValueError("Identity change requires passing stability and safety reviews.")
        review_after = datetime.fromisoformat(proposal.review_after)
        if review_after.tzinfo is None or review_after > datetime.now(UTC):
            raise ValueError("Identity proposal evaluation period is not complete.")
        updates = dict(proposal.candidate_values)
        if not updates or not updates.keys() <= IdentityState.__dataclass_fields__.keys():
            raise ValueError("Identity proposal contains unsupported fields.")
        updates["identity_version"] = self.state.identity_version + 1
        updated_state = IdentityState.from_dict(
            {
                **self.state.to_dict(),
                **updates,
            }
        )
        approved = replace(
            proposal,
            status=IdentityProposalStatus.APPROVED,
            user_approved=True,
            approval_reference=approval_reference,
        )
        if self._persistence is not None:
            self._persistence.apply_identity_approval(updated_state, approved)
        self.state = updated_state
        self._proposals[proposal_id] = approved
        return updated_state

    def reject(self, proposal_id: str) -> None:
        proposal = self._get(proposal_id)
        rejected = replace(proposal, status=IdentityProposalStatus.REJECTED)
        if self._persistence is not None:
            self._persistence.save_identity_proposal(rejected)
        self._proposals[proposal_id] = rejected

    def proposals(self) -> tuple[IdentityChangeProposal, ...]:
        return tuple(self._proposals.values())

    def _get(self, proposal_id: str) -> IdentityChangeProposal:
        try:
            return self._proposals[proposal_id]
        except KeyError as exc:
            raise KeyError(f"Unknown identity proposal {proposal_id!r}.") from exc


@dataclass(frozen=True)
class RelationshipState:
    participant_id: str
    first_interaction: str
    last_interaction: str
    interaction_count: int = 0
    familiarity: float = 0.0
    trust: float = 0.0
    cooperation: float = 0.0
    communication_comfort: float = 0.0
    unresolved_topics: tuple[str, ...] = ()
    shared_experiences: tuple[str, ...] = ()
    confirmed_preferences: tuple[str, ...] = ()
    boundaries: tuple[str, ...] = ()
    recent_interaction_quality: float = 0.0
    decay_timestamp: str = field(default_factory=_now)
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.participant_id.strip():
            raise ValueError("participant_id must not be empty.")
        if self.interaction_count < 0:
            raise ValueError("interaction_count must not be negative.")
        for field_name in (
            "familiarity",
            "trust",
            "cooperation",
            "communication_comfort",
            "recent_interaction_quality",
        ):
            value = getattr(self, field_name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and between 0.0 and 1.0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipState:
        values = dict(data)
        for name in (
            "unresolved_topics",
            "shared_experiences",
            "confirmed_preferences",
            "boundaries",
            "evidence_ids",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


class RelationshipManager:
    def record_interaction(
        self,
        state: RelationshipState,
        *,
        evidence_id: str,
        observed_quality: float,
    ) -> RelationshipState:
        if not evidence_id.strip():
            raise ValueError("Relationship changes require a source evidence ID.")
        if evidence_id in state.evidence_ids:
            return state
        if not isfinite(observed_quality) or not 0.0 <= observed_quality <= 1.0:
            raise ValueError("observed_quality must be between 0.0 and 1.0.")
        return replace(
            state,
            last_interaction=_now(),
            interaction_count=state.interaction_count + 1,
            familiarity=min(1.0, state.familiarity + 0.01),
            recent_interaction_quality=observed_quality,
            evidence_ids=(*state.evidence_ids, evidence_id),
            decay_timestamp=_now(),
        )


class GoalType(StrEnum):
    ANSWER_USER = "answer_user"
    UNDERSTAND_OBSERVATION = "understand_observation"
    CLARIFY_UNCERTAINTY = "clarify_uncertainty"
    MAINTAIN_CONVERSATION = "maintain_conversation"
    REMAIN_SILENT = "remain_silent"
    RETRIEVE_INFORMATION = "retrieve_information"
    CHECK_SYSTEM_HEALTH = "check_system_health"
    REMEMBER_WITH_CONSENT = "remember_with_consent"
    CORRECT_PREVIOUS_CLAIM = "correct_previous_claim"
    PREPARE_HARDWARE_ACTION = "prepare_hardware_action"
    LEARN_FROM_REVIEWED_FEEDBACK = "learn_from_reviewed_feedback"


class GoalOrigin(StrEnum):
    USER_REQUEST = "user_request"
    OBSERVATION = "observation"
    INTERNAL_MAINTENANCE = "internal_maintenance"
    UNRESOLVED_CONTEXT = "unresolved_context"
    SCHEDULED = "scheduled"
    SAFETY = "safety"


class GoalStatus(StrEnum):
    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass(frozen=True)
class Goal:
    goal_id: str
    goal_type: GoalType
    origin: GoalOrigin
    priority: int
    created_at: str
    expires_at: str | None
    prerequisites: tuple[str, ...]
    risk_level: str
    privacy_impact: str
    required_tools: tuple[str, ...]
    completion_criteria: str
    cancellation_reason: str | None = None
    status: GoalStatus = GoalStatus.PENDING
    approval_reference: str | None = None


class GoalManager:
    _APPROVAL_TOOLS: ClassVar[frozenset[str]] = frozenset({
        "external_data_transfer",
        "persist_personal_data",
        "save_camera_data",
        "save_audio_data",
        "login",
        "send_message",
        "delete_file",
        "install_software",
        "payment",
        "create_api_key",
        "hardware_movement",
        "approach_person",
        "train_or_deploy_model",
    })
    _PROHIBITED_TOOLS: ClassVar[frozenset[str]] = frozenset(
        {"payment", "install_software", "create_api_key", "train_or_deploy_model"}
    )

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}

    def create(
        self,
        goal_type: GoalType,
        origin: GoalOrigin,
        *,
        priority: int,
        expires_at: str | None,
        prerequisites: tuple[str, ...] = (),
        risk_level: str = "low",
        privacy_impact: str = "none",
        required_tools: tuple[str, ...] = (),
        completion_criteria: str,
    ) -> Goal:
        if isinstance(priority, bool) or not 0 <= priority <= 100:
            raise ValueError("Goal priority must be between 0 and 100.")
        if not completion_criteria.strip():
            raise ValueError("Goal completion_criteria must not be empty.")
        prohibited = self._PROHIBITED_TOOLS.intersection(required_tools)
        if prohibited:
            raise PermissionError(
                "Goal requires prohibited action(s): " + ", ".join(sorted(prohibited)) + "."
            )
        approval_required = (
            self._APPROVAL_TOOLS.intersection(required_tools)
            or privacy_impact.casefold() not in {"none", "local"}
            or risk_level.casefold() in {"high", "critical"}
            or goal_type
            in {GoalType.REMEMBER_WITH_CONSENT, GoalType.PREPARE_HARDWARE_ACTION}
        )
        goal = Goal(
            goal_id=uuid.uuid4().hex,
            goal_type=GoalType(goal_type),
            origin=GoalOrigin(origin),
            priority=priority,
            created_at=_now(),
            expires_at=expires_at,
            prerequisites=prerequisites,
            risk_level=risk_level,
            privacy_impact=privacy_impact,
            required_tools=required_tools,
            completion_criteria=completion_criteria,
            status=GoalStatus.AWAITING_APPROVAL if approval_required else GoalStatus.PENDING,
        )
        self._goals[goal.goal_id] = goal
        return goal

    def approve(
        self,
        goal_id: str,
        *,
        user_approved: bool,
        approval_reference: str | None = None,
    ) -> Goal:
        goal = self._get(goal_id)
        if goal.status != GoalStatus.AWAITING_APPROVAL:
            raise ValueError("Only goals awaiting approval can be approved.")
        if not user_approved:
            raise PermissionError("User approval is required.")
        if not approval_reference or not approval_reference.strip():
            raise PermissionError("Goal approval requires a recorded user approval reference.")
        updated = replace(
            goal,
            status=GoalStatus.PENDING,
            approval_reference=approval_reference,
        )
        self._goals[goal_id] = updated
        return updated

    def cancel(self, goal_id: str, reason: str) -> Goal:
        if not reason.strip():
            raise ValueError("Goal cancellation requires a reason.")
        goal = self._get(goal_id)
        updated = replace(goal, status=GoalStatus.CANCELLED, cancellation_reason=reason)
        self._goals[goal_id] = updated
        return updated

    def expire(self, *, now: datetime | None = None) -> tuple[Goal, ...]:
        current = now or datetime.now(UTC)
        expired: list[Goal] = []
        for goal_id, goal in tuple(self._goals.items()):
            if goal.status not in {GoalStatus.PENDING, GoalStatus.AWAITING_APPROVAL}:
                continue
            if goal.expires_at is None:
                continue
            expiration = datetime.fromisoformat(goal.expires_at)
            if expiration.tzinfo is None:
                raise ValueError("Goal expiration timestamps must include a timezone.")
            if expiration <= current:
                updated = replace(goal, status=GoalStatus.EXPIRED)
                self._goals[goal_id] = updated
                expired.append(updated)
        return tuple(expired)

    def active_goals(self) -> tuple[Goal, ...]:
        return tuple(
            sorted(
                (
                    goal
                    for goal in self._goals.values()
                    if goal.status in {GoalStatus.PENDING, GoalStatus.AWAITING_APPROVAL}
                ),
                key=lambda goal: (-goal.priority, goal.created_at),
            )
        )

    def complete(self, goal_id: str) -> Goal:
        goal = self._get(goal_id)
        if goal.status != GoalStatus.PENDING:
            raise ValueError("Only pending goals can be completed.")
        updated = replace(goal, status=GoalStatus.COMPLETED)
        self._goals[goal_id] = updated
        return updated

    def _get(self, goal_id: str) -> Goal:
        try:
            return self._goals[goal_id]
        except KeyError as exc:
            raise KeyError(f"Unknown goal {goal_id!r}.") from exc


class MemoryType(StrEnum):
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    RELATIONSHIP = "relationship"
    PREFERENCE = "preference"
    UNRESOLVED_QUESTION = "unresolved_question"
    EXTERNAL_FACT_CACHE = "external_fact_cache"


class ConsentStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class MemoryEntry:
    memory_id: str
    memory_type: MemoryType
    subject: str
    content_summary: str
    source_type: str
    source_reference: str
    created_at: str
    last_used_at: str | None
    confidence: float
    importance: float
    retention_policy: str
    expiration: str | None
    user_confirmed: bool
    consent_status: ConsentStatus
    sensitive: bool = False
    correction_history: tuple[str, ...] = ()
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        if not self.memory_id or not self.content_summary.strip():
            raise ValueError("memory_id and content_summary must not be empty.")
        if self.expiration is not None:
            expires = datetime.fromisoformat(self.expiration)
            if expires.tzinfo is None:
                raise ValueError("Memory expiration timestamps must include a timezone.")
        if self.consent_status == ConsentStatus.GRANTED and not self.user_confirmed:
            raise ValueError("Granted memory consent must be associated with user confirmation.")
        for name in ("confidence", "importance"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0.0 and 1.0.")
        object.__setattr__(self, "memory_type", MemoryType(self.memory_type))
        object.__setattr__(self, "consent_status", ConsentStatus(self.consent_status))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["memory_type"] = self.memory_type.value
        result["consent_status"] = self.consent_status.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        values = dict(data)
        values["correction_history"] = tuple(values.get("correction_history", ()))
        return cls(**values)


class ResponseType(StrEnum):
    DIRECT_ANSWER = "direct_answer"
    CONVERSATIONAL_REPLY = "conversational_reply"
    OBSERVATION_REPORT = "observation_report"
    SELF_REFLECTION = "self_reflection"
    UNCERTAINTY_REPORT = "uncertainty_report"
    CORRECTION = "correction"
    CLARIFICATION_QUESTION = "clarification_question"
    PROACTIVE_COMMENT = "proactive_comment"
    SAFETY_WARNING = "safety_warning"
    SILENCE = "silence"


@dataclass(frozen=True)
class ResponsePlan:
    response_id: str
    response_type: ResponseType
    text: str
    confidence: float
    emotional_tone_parameters: tuple[tuple[str, float], ...] = ()
    speaking_rate: float = 1.0
    speaking_energy: float = 1.0
    observations_used: tuple[str, ...] = ()
    memories_used: tuple[str, ...] = ()
    external_sources_used: tuple[str, ...] = ()
    hypotheses_used: tuple[str, ...] = ()
    uncertainty_statement: str | None = None
    follow_up_question: str | None = None
    proposed_actions: tuple[str, ...] = ()
    privacy_warning: str | None = None
    should_speak: bool = True
    reason_for_silence: str | None = None

    def __post_init__(self) -> None:
        if not self.response_id:
            raise ValueError("response_id must not be empty.")
        object.__setattr__(self, "response_type", ResponseType(self.response_type))
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("response confidence must be between 0.0 and 1.0.")
        if not isfinite(self.speaking_rate) or not 0.5 <= self.speaking_rate <= 2.0:
            raise ValueError("speaking_rate must be between 0.5 and 2.0.")
        if not isfinite(self.speaking_energy) or not 0.0 <= self.speaking_energy <= 1.0:
            raise ValueError("speaking_energy must be between 0.0 and 1.0.")
        if any(
            not name or not isfinite(value) or not 0.0 <= value <= 1.0
            for name, value in self.emotional_tone_parameters
        ):
            raise ValueError("Emotional tone parameters must have names and values in [0, 1].")
        if not self.should_speak and not self.reason_for_silence:
            raise ValueError("Silent response plans must include a reason_for_silence.")

    def to_json(self) -> str:
        payload = asdict(self)
        payload["response_type"] = self.response_type.value
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
