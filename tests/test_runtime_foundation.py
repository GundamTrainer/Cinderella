from __future__ import annotations

import pytest

from runtime.component import Component, ComponentState, InvalidStateTransitionError
from runtime.config import ConfigurationError, RuntimeSettings, load_runtime_settings
from runtime.event_bus import Event, EventBus, EventBusClosedError, EventBusFullError, EventType
from runtime.orchestrator import RuntimeOrchestrator
from runtime.simulators import SimulatedAudioSource, SimulatedHardwareAdapter, SimulatedVisionSource


class DummyComponent(Component):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _do_initialize(self) -> None:
        return None

    def _do_start(self) -> None:
        return None

    def _do_stop(self) -> None:
        return None

    def _do_close(self) -> None:
        return None


class FailingComponent(Component):
    def _do_initialize(self) -> None:
        raise RuntimeError("initialization failed")

    def _do_start(self) -> None:
        return None

    def _do_stop(self) -> None:
        return None

    def _do_close(self) -> None:
        return None


def test_valid_event_creation() -> None:
    event = Event.create(
        EventType.SYSTEM_STARTED,
        "runtime",
        payload={"synthetic": True, "status": "booting"},
        confidence=0.9,
    )

    assert event.event_type == EventType.SYSTEM_STARTED
    assert event.source == "runtime"
    assert event.payload["synthetic"] is True
    assert 0.0 <= event.confidence <= 1.0


def test_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Event.create(
            EventType.SYSTEM_STARTED,
            "runtime",
            payload={"synthetic": False},
            confidence=1.5,
        )


def test_json_round_trip_preserves_meaning() -> None:
    event = Event.create(
        EventType.VISION_PERSON_DETECTED,
        "simulated_vision",
        payload={"synthetic": True, "person_count": 1, "action": "wave"},
        confidence=0.88,
        correlation_id="corr-123",
    )

    restored = Event.from_json(event.to_json())

    assert restored == event
    assert restored.event_type == EventType.VISION_PERSON_DETECTED
    assert restored.payload == event.payload
    assert restored.correlation_id == "corr-123"


def test_queue_order_is_preserved() -> None:
    bus = EventBus(10)
    first = Event.create(EventType.SYSTEM_STARTED, "runtime", payload={"step": 1})
    second = Event.create(EventType.SYSTEM_STOPPING, "runtime", payload={"step": 2})

    bus.publish(first)
    bus.publish(second)

    assert bus.poll() == first
    assert bus.poll() == second


def test_queue_capacity_is_enforced() -> None:
    bus = EventBus(1)
    bus.publish(Event.create(EventType.SYSTEM_STARTED, "runtime"), block=True, timeout=0.01)

    with pytest.raises(EventBusFullError):
        bus.publish(Event.create(EventType.SYSTEM_STOPPING, "runtime"), block=True, timeout=0.01)


def test_close_rejects_future_publishes() -> None:
    bus = EventBus(2)
    bus.close()

    with pytest.raises(EventBusClosedError):
        bus.publish(Event.create(EventType.SYSTEM_STARTED, "runtime"))


def test_drain_removes_events() -> None:
    bus = EventBus(2)
    first = Event.create(EventType.SYSTEM_STARTED, "runtime", payload={"n": 1})
    second = Event.create(EventType.SYSTEM_STOPPING, "runtime", payload={"n": 2})
    bus.publish(first)
    bus.publish(second)

    drained = bus.drain(max_events=1)

    assert len(drained) == 1
    assert drained[0] == first
    assert bus.size() == 1


def test_normal_component_lifecycle_transitions() -> None:
    component = DummyComponent("dummy")

    component.initialize()
    assert component.state == ComponentState.INITIALIZED

    component.start()
    assert component.state == ComponentState.RUNNING

    component.stop()
    assert component.state == ComponentState.STOPPED

    component.close()
    assert component.state == ComponentState.CLOSED


def test_invalid_component_transitions_are_rejected() -> None:
    component = DummyComponent("dummy")

    with pytest.raises(InvalidStateTransitionError):
        component.start()


def test_simulated_components_are_deterministic_for_same_seed() -> None:
    audio_a = SimulatedAudioSource(event_bus=EventBus(16), seed=7)
    audio_b = SimulatedAudioSource(event_bus=EventBus(16), seed=7)
    vision_a = SimulatedVisionSource(event_bus=EventBus(16), seed=7)
    vision_b = SimulatedVisionSource(event_bus=EventBus(16), seed=7)
    hardware_a = SimulatedHardwareAdapter(event_bus=EventBus(16), seed=7)
    hardware_b = SimulatedHardwareAdapter(event_bus=EventBus(16), seed=7)

    assert audio_a.generate_events() == audio_b.generate_events()
    assert vision_a.generate_events() == vision_b.generate_events()
    assert hardware_a.generate_events() == hardware_b.generate_events()


def test_partial_initialization_failure_closes_initialized_components() -> None:
    bus = EventBus(8)
    good = DummyComponent("good")
    bad = FailingComponent("bad")
    orchestrator = RuntimeOrchestrator(RuntimeSettings.from_mapping({
        "runtime_queue_capacity": 8,
        "event_publish_timeout": 0.25,
        "shutdown_timeout": 1.0,
        "logging_level": "INFO",
        "simulation_seed": 42,
        "audio_simulator_enabled": True,
        "vision_simulator_enabled": True,
        "hardware_simulator_enabled": True,
    }), bus=bus, components=[good, bad])

    with pytest.raises(RuntimeError, match="initialization failed"):
        orchestrator.initialize()

    assert good.state == ComponentState.CLOSED


def test_shutdown_closes_all_components() -> None:
    bus = EventBus(16)
    first = DummyComponent("first")
    second = DummyComponent("second")
    settings = RuntimeSettings.from_mapping({
        "runtime_queue_capacity": 16,
        "event_publish_timeout": 0.25,
        "shutdown_timeout": 1.0,
        "logging_level": "INFO",
        "simulation_seed": 42,
        "audio_simulator_enabled": True,
        "vision_simulator_enabled": True,
        "hardware_simulator_enabled": True,
    })
    orchestrator = RuntimeOrchestrator(settings, bus=bus, components=[first, second])
    orchestrator.initialize()
    orchestrator.start()
    orchestrator.shutdown()

    assert first.state == ComponentState.CLOSED
    assert second.state == ComponentState.CLOSED
    assert bus.closed is True


def test_runtime_settings_load_valid_file() -> None:
    settings = load_runtime_settings("configs/default.yaml")

    assert settings.runtime_queue_capacity == 128
    assert settings.logging_level == "INFO"
    assert settings.audio_simulator_enabled is True


def test_runtime_settings_reject_invalid_values() -> None:
    with pytest.raises(ConfigurationError, match="runtime_queue_capacity"):
        RuntimeSettings.from_mapping({
            "runtime_queue_capacity": 0,
            "event_publish_timeout": 0.25,
            "shutdown_timeout": 1.0,
            "logging_level": "INFO",
            "simulation_seed": 42,
            "audio_simulator_enabled": True,
            "vision_simulator_enabled": True,
            "hardware_simulator_enabled": True,
        })
