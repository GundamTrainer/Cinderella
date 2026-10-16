from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from runtime.companion_state import (
    ConsentStatus,
    GoalManager,
    GoalOrigin,
    GoalStatus,
    GoalType,
    IdentityManager,
    IdentityProposalStatus,
    IdentityState,
    MemoryEntry,
    MemoryType,
    RelationshipManager,
    RelationshipState,
    ResponsePlan,
    ResponseType,
)
from runtime.state_store import SQLiteStateStore


def _memory(
    *,
    sensitive: bool = False,
    memory_type: MemoryType = MemoryType.PREFERENCE,
    consent: ConsentStatus = ConsentStatus.GRANTED,
    confirmed: bool = True,
) -> MemoryEntry:
    return MemoryEntry(
        memory_id="memory-1",
        memory_type=memory_type,
        subject="language",
        content_summary="Prefers Korean replies",
        source_type="user_statement",
        source_reference="event-1",
        created_at="2025-01-01T00:00:00+00:00",
        last_used_at=None,
        confidence=0.9,
        importance=0.5,
        retention_policy="until deleted",
        expiration=None,
        user_confirmed=confirmed,
        consent_status=consent,
        sensitive=sensitive,
    )


def test_identity_state_json_round_trip_and_change_requires_reviews_and_approval() -> None:
    state = IdentityState()
    assert IdentityState.from_dict(state.to_dict()) == state

    manager = IdentityManager(state)
    proposal = manager.propose_change(
        {"communication_preferences": {"verbosity": "brief"}},
        evidence=("reviewed user preference",),
        review_after=(datetime.now(UTC) - timedelta(days=3)).isoformat(),
    )
    with pytest.raises(ValueError, match="evaluation"):
        manager.approve(
            proposal.proposal_id,
            user_approved=True,
            approval_reference="approval-event-1",
        )
    manager.record_review(proposal.proposal_id, stability_passed=True, safety_passed=True)
    with pytest.raises(PermissionError, match="user"):
        manager.approve(proposal.proposal_id, user_approved=False)

    with pytest.raises(PermissionError, match="approval reference"):
        manager.approve(proposal.proposal_id, user_approved=True)
    updated = manager.approve(
        proposal.proposal_id,
        user_approved=True,
        approval_reference="approval-event-1",
    )
    assert updated.identity_version == state.identity_version + 1
    assert updated.communication_preferences == {"verbosity": "brief"}
    assert manager._get(proposal.proposal_id).status == IdentityProposalStatus.APPROVED


def test_identity_proposal_rejects_api_mutation_of_core_identity() -> None:
    manager = IdentityManager()
    with pytest.raises(ValueError, match="Core identity"):
        manager.propose_change(
            {"system_name": "changed"},
            evidence=("external API response",),
            review_after=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
        )


def test_relationship_changes_need_unique_evidence_and_are_bounded() -> None:
    state = RelationshipState(
        participant_id="local-user",
        first_interaction="2025-01-01T00:00:00+00:00",
        last_interaction="2025-01-01T00:00:00+00:00",
    )
    manager = RelationshipManager()

    updated = manager.record_interaction(state, evidence_id="event-1", observed_quality=0.8)
    repeated = manager.record_interaction(updated, evidence_id="event-1", observed_quality=1.0)

    assert updated.interaction_count == 1
    assert updated.familiarity == pytest.approx(0.01)
    assert repeated == updated
    with pytest.raises(ValueError, match="evidence"):
        manager.record_interaction(state, evidence_id="", observed_quality=0.8)


def test_goal_manager_gates_risky_actions_and_supports_silence_expiry_and_cancel() -> None:
    manager = GoalManager()
    risky = manager.create(
        GoalType.RETRIEVE_INFORMATION,
        GoalOrigin.USER_REQUEST,
        priority=80,
        expires_at=None,
        privacy_impact="external",
        required_tools=("external_data_transfer",),
        completion_criteria="return a cited result",
    )
    assert risky.status == GoalStatus.AWAITING_APPROVAL
    with pytest.raises(PermissionError):
        manager.approve(risky.goal_id, user_approved=False)
    with pytest.raises(PermissionError, match="approval reference"):
        manager.approve(risky.goal_id, user_approved=True)
    assert (
        manager.approve(
            risky.goal_id,
            user_approved=True,
            approval_reference="approval-event-2",
        ).status
        == GoalStatus.PENDING
    )

    silence = manager.create(
        GoalType.REMAIN_SILENT,
        GoalOrigin.SAFETY,
        priority=50,
        expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        completion_criteria="do not interrupt",
    )
    assert silence.status == GoalStatus.PENDING
    assert manager.cancel(silence.goal_id, "user resumed conversation").status == GoalStatus.CANCELLED

    expiring = manager.create(
        GoalType.ANSWER_USER,
        GoalOrigin.USER_REQUEST,
        priority=50,
        expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        completion_criteria="answer once",
    )
    expired = manager.expire(now=datetime.now(UTC))
    assert expiring.goal_id in {goal.goal_id for goal in expired}


@pytest.mark.parametrize("tool", ("payment", "install_software"))
def test_goal_manager_rejects_payment_and_automatic_installation(tool: str) -> None:
    manager = GoalManager()
    with pytest.raises(PermissionError, match="prohibited"):
        manager.create(
            GoalType.RETRIEVE_INFORMATION,
            GoalOrigin.USER_REQUEST,
            priority=50,
            expires_at=None,
            required_tools=(tool,),
            completion_criteria="run a prohibited action",
        )


def test_memory_requires_explicit_consent_and_can_be_deleted_corrected_or_disabled(
    tmp_path,
) -> None:
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    store.initialize()

    with pytest.raises(PermissionError, match="explicit"):
        store.save_memory(_memory(confirmed=False, consent=ConsentStatus.NOT_REQUESTED))
    with pytest.raises(ValueError, match="External facts"):
        store.save_memory(_memory(memory_type=MemoryType.EXTERNAL_FACT_CACHE))

    store.save_memory(_memory(sensitive=True))
    found = store.find_memories("Korean replies")
    assert len(found) == 1
    assert store.list_memories() == found
    corrected = store.correct_memory("memory-1", "Prefers concise Korean replies")
    assert corrected.correction_history == ("Prefers Korean replies",)
    retained = store.set_memory_retention(
        "memory-1",
        retention_policy="30 days",
        expiration=(datetime.now(UTC) + timedelta(days=30)).isoformat(),
    )
    assert retained.retention_policy == "30 days"
    assert store.delete_memories_by_type(MemoryType.PREFERENCE) == 1

    store.save_memory(_memory())
    store.set_memory_enabled(False)
    assert store.find_memories("Korean") == ()
    with pytest.raises(PermissionError, match="disabled"):
        store.save_memory(_memory())
    assert store.delete_memory("memory-1")
    store.close()


def test_identity_and_relationship_states_persist_only_when_explicitly_saved(tmp_path) -> None:
    path = tmp_path / "companion.sqlite3"
    store = SQLiteStateStore(path)
    store.initialize()
    identity = IdentityState()
    relationship = RelationshipState(
        participant_id="participant-1",
        first_interaction="2025-01-01T00:00:00+00:00",
        last_interaction="2025-01-01T00:00:00+00:00",
    )
    store.save_identity(identity)
    store.save_relationship(relationship)
    store.close()

    reopened = SQLiteStateStore(path)
    reopened.initialize()
    assert reopened.load_identity() == identity
    assert reopened.load_relationship("participant-1") == relationship
    assert reopened.load_relationship("missing") is None
    reopened.close()


def test_identity_change_history_and_approval_persist_atomically(tmp_path) -> None:
    store = SQLiteStateStore(tmp_path / "identity.sqlite3")
    store.initialize()
    manager = IdentityManager(persistence=store)
    proposal = manager.propose_change(
        {"communication_preferences": {"verbosity": "concise"}},
        evidence=("reviewed user feedback",),
        review_after=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
    )
    manager.record_review(proposal.proposal_id, stability_passed=True, safety_passed=True)
    manager.approve(
        proposal.proposal_id,
        user_approved=True,
        approval_reference="approval-event-3",
    )
    store.close()

    reopened = SQLiteStateStore(tmp_path / "identity.sqlite3")
    reopened.initialize()
    restored = IdentityManager(persistence=reopened)
    assert restored.state.identity_version == 2
    assert restored.proposals()[0].status == IdentityProposalStatus.APPROVED
    assert restored.proposals()[0].evidence == ("reviewed user feedback",)
    reopened.close()


def test_response_plan_distinguishes_sources_uncertainty_and_silence() -> None:
    plan = ResponsePlan(
        response_id="response-1",
        response_type=ResponseType.UNCERTAINTY_REPORT,
        text="외부 자료와 현재 관찰이 다릅니다.",
        confidence=0.4,
        external_sources_used=("source-1",),
        uncertainty_statement="자료가 충돌합니다.",
    )
    assert '"external_sources_used": ["source-1"]' in plan.to_json()

    with pytest.raises(ValueError, match="reason_for_silence"):
        ResponsePlan(
            response_id="response-2",
            response_type=ResponseType.SILENCE,
            text="",
            confidence=1.0,
            should_speak=False,
        )
