from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from ai.scene import WorldState
from runtime.event_bus import Event, EventBus, EventType


class SimulatedAction(StrEnum):
    OBSERVE = "observe"
    SPEAK = "speak"
    MOVE_FORWARD = "move_forward"
    STOP = "stop"


@dataclass(frozen=True)
class ActionProposal:
    action: SimulatedAction
    confidence: float
    reason: str
    distance_meters: float = 0.0
    synthetic: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.action, SimulatedAction):
            raise TypeError("action must be a SimulatedAction.")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1].")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must not be empty.")
        if (
            isinstance(self.distance_meters, bool)
            or not isinstance(self.distance_meters, (int, float))
            or not isfinite(self.distance_meters)
            or self.distance_meters < 0
        ):
            raise ValueError("distance_meters must be finite and nonnegative.")
        if not isinstance(self.synthetic, bool):
            raise TypeError("synthetic must be a boolean.")


@dataclass(frozen=True)
class SafetyDecision:
    approved: bool
    reason: str


class PhysicalAiSimulator:
    """Fail-closed action proposal and safety gate; never controls real hardware."""

    def __init__(
        self,
        event_bus: EventBus,
        *,
        minimum_confidence: float = 0.7,
        maximum_simulated_distance_meters: float = 0.1,
    ) -> None:
        if (
            isinstance(minimum_confidence, bool)
            or not isinstance(minimum_confidence, (int, float))
            or not isfinite(minimum_confidence)
            or not 0.0 <= minimum_confidence <= 1.0
        ):
            raise ValueError("minimum_confidence must be finite and in [0, 1].")
        if (
            isinstance(maximum_simulated_distance_meters, bool)
            or not isinstance(maximum_simulated_distance_meters, (int, float))
            or not isfinite(maximum_simulated_distance_meters)
            or maximum_simulated_distance_meters <= 0
        ):
            raise ValueError("maximum simulated distance must be finite and positive.")
        self.event_bus = event_bus
        self.minimum_confidence = minimum_confidence
        self.maximum_simulated_distance_meters = maximum_simulated_distance_meters

    def evaluate(self, proposal: ActionProposal, world_state: WorldState) -> SafetyDecision:
        if proposal.action == SimulatedAction.STOP:
            return SafetyDecision(True, "stop_action_always_allowed")
        if proposal.confidence < self.minimum_confidence:
            return SafetyDecision(False, "confidence_below_safety_threshold")
        if proposal.action == SimulatedAction.MOVE_FORWARD:
            if not proposal.synthetic:
                return SafetyDecision(False, "non_synthetic_motion_proposal_rejected")
            if proposal.distance_meters > self.maximum_simulated_distance_meters:
                return SafetyDecision(False, "simulated_distance_limit_exceeded")
            if world_state.objects or world_state.uncertain_labels:
                return SafetyDecision(False, "scene_not_confirmed_clear")
        return SafetyDecision(True, "within_simulation_policy")

    def process(self, proposal: ActionProposal, world_state: WorldState) -> SafetyDecision:
        decision = self.evaluate(proposal, world_state)
        self.event_bus.publish(
            Event.create(
                EventType.PLANNER_ACTION_PROPOSED,
                "physical_ai_simulator",
                payload={
                    "action": proposal.action.value,
                    "reason": proposal.reason,
                    "synthetic": proposal.synthetic,
                    "real_hardware_actuated": False,
                },
                confidence=proposal.confidence,
            )
        )
        self.event_bus.publish(
            Event.create(
                EventType.SAFETY_ACTION_APPROVED
                if decision.approved
                else EventType.SAFETY_ACTION_REJECTED,
                "physical_ai_safety_gate",
                payload={
                    "action": proposal.action.value,
                    "reason": decision.reason,
                    "synthetic": True,
                    "real_hardware_actuated": False,
                },
                confidence=proposal.confidence,
            )
        )
        if decision.approved and proposal.action in {
            SimulatedAction.MOVE_FORWARD,
            SimulatedAction.STOP,
        }:
            self.event_bus.publish(
                Event.create(
                    EventType.HARDWARE_COMMAND_SIMULATED,
                    "simulated_hardware_adapter",
                    payload={
                        "command": proposal.action.value,
                        "distance_meters": proposal.distance_meters,
                        "synthetic": True,
                        "real_hardware_actuated": False,
                    },
                    confidence=proposal.confidence,
                )
            )
        return decision


def run_physical_ai_safety_demo(event_bus: EventBus) -> tuple[Event, ...]:
    """Run deterministic clear-scene and uncertain-scene motion examples."""
    simulator = PhysicalAiSimulator(event_bus)
    clear_scene = WorldState(observed_at=0.0, objects=(), uncertain_labels=())
    uncertain_scene = WorldState(
        observed_at=1.0,
        objects=(),
        uncertain_labels=("unverified object",),
    )
    simulator.process(
        ActionProposal(
            SimulatedAction.MOVE_FORWARD,
            confidence=0.9,
            reason="synthetic clear-scene example",
            distance_meters=0.05,
        ),
        clear_scene,
    )
    simulator.process(
        ActionProposal(
            SimulatedAction.MOVE_FORWARD,
            confidence=0.9,
            reason="synthetic uncertain-scene example",
            distance_meters=0.05,
        ),
        uncertain_scene,
    )
    return tuple(event_bus.drain())
