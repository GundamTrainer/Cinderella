from __future__ import annotations

import random

from .component import Component
from .event_bus import Event, EventBus, EventType


class SimulatedAudioSource(Component):
    def __init__(self, event_bus: EventBus, *, seed: int = 42) -> None:
        super().__init__("simulated_audio_source")
        self.event_bus = event_bus
        self.seed = seed

    def _do_initialize(self) -> None:
        self._deterministic_events = self._build_events()

    def _do_start(self) -> None:
        for event in self._deterministic_events:
            self.event_bus.publish(event, block=True, timeout=0.25)

    def _do_stop(self) -> None:
        return None

    def _do_close(self) -> None:
        return None

    def _build_events(self) -> list[Event]:
        events: list[Event] = []
        rng = random.Random(self.seed)
        for index in range(3):
            confidence_value = round(rng.random(), 3)
            events.append(
                Event.create(
                    EventType.AUDIO_SPEECH_STARTED,
                    self.name,
                    payload={
                        "synthetic": True,
                        "seed": self.seed,
                        "sample_index": index,
                        "confidence_hint": confidence_value,
                    },
                    confidence=0.9,
                    correlation_id=f"audio.{self.seed}.{index}",
                    monotonic_time=float(index),
                    event_id=f"audio.{self.seed}.{index}.speech_started",
                    created_at=f"2024-01-01T00:00:{index:02d}.000000+00:00",
                )
            )
            events.append(
                Event.create(
                    EventType.AUDIO_SOUND_DETECTED,
                    self.name,
                    payload={
                        "synthetic": True,
                        "seed": self.seed,
                        "sample_index": index,
                        "signal_level": round(rng.random(), 3),
                    },
                    confidence=0.8,
                    correlation_id=f"audio.{self.seed}.{index}",
                    monotonic_time=float(index) + 0.5,
                    event_id=f"audio.{self.seed}.{index}.sound_detected",
                    created_at=f"2024-01-01T00:00:{index:02d}.500000+00:00",
                )
            )
        return events

    def generate_events(self) -> list[Event]:
        return self._build_events()


class SimulatedVisionSource(Component):
    def __init__(self, event_bus: EventBus, *, seed: int = 42) -> None:
        super().__init__("simulated_vision_source")
        self.event_bus = event_bus
        self.seed = seed

    def _do_initialize(self) -> None:
        self._deterministic_events = self._build_events()

    def _do_start(self) -> None:
        for event in self._deterministic_events:
            self.event_bus.publish(event, block=True, timeout=0.25)

    def _do_stop(self) -> None:
        return None

    def _do_close(self) -> None:
        return None

    def _build_events(self) -> list[Event]:
        events: list[Event] = []
        rng = random.Random(self.seed)
        for index in range(2):
            events.append(
                Event.create(
                    EventType.VISION_PERSON_DETECTED,
                    self.name,
                    payload={
                        "synthetic": True,
                        "seed": self.seed,
                        "frame_index": index,
                        "person_count": 1,
                        "confidence_hint": round(rng.random(), 3),
                    },
                    confidence=0.88,
                    correlation_id=f"vision.{self.seed}.{index}",
                    monotonic_time=float(index),
                    event_id=f"vision.{self.seed}.{index}.person_detected",
                    created_at=f"2024-01-02T00:00:{index:02d}.000000+00:00",
                )
            )
            events.append(
                Event.create(
                    EventType.VISION_ACTION_DETECTED,
                    self.name,
                    payload={
                        "synthetic": True,
                        "seed": self.seed,
                        "frame_index": index,
                        "action": "wave",
                        "confidence_hint": round(rng.random(), 3),
                    },
                    confidence=0.84,
                    correlation_id=f"vision.{self.seed}.{index}",
                    monotonic_time=float(index) + 0.25,
                    event_id=f"vision.{self.seed}.{index}.action_detected",
                    created_at=f"2024-01-02T00:00:{index:02d}.250000+00:00",
                )
            )
        return events

    def generate_events(self) -> list[Event]:
        return self._build_events()


class SimulatedHardwareAdapter(Component):
    def __init__(self, event_bus: EventBus, *, seed: int = 42) -> None:
        super().__init__("simulated_hardware_adapter")
        self.event_bus = event_bus
        self.seed = seed

    def _do_initialize(self) -> None:
        self._deterministic_events = self._build_events()

    def _do_start(self) -> None:
        for event in self._deterministic_events:
            self.event_bus.publish(event, block=True, timeout=0.25)

    def _do_stop(self) -> None:
        return None

    def _do_close(self) -> None:
        return None

    def _build_events(self) -> list[Event]:
        events: list[Event] = []
        rng = random.Random(self.seed)
        for index in range(2):
            events.append(
                Event.create(
                    EventType.HARDWARE_COMMAND_SIMULATED,
                    self.name,
                    payload={
                        "synthetic": True,
                        "seed": self.seed,
                        "command_index": index,
                        "command": "move_forward",
                        "power_level": round(rng.random(), 3),
                    },
                    confidence=0.92,
                    correlation_id=f"hardware.{self.seed}.{index}",
                    monotonic_time=float(index),
                    event_id=f"hardware.{self.seed}.{index}.command_simulated",
                    created_at=f"2024-01-03T00:00:{index:02d}.000000+00:00",
                )
            )
        return events

    def generate_events(self) -> list[Event]:
        return self._build_events()
