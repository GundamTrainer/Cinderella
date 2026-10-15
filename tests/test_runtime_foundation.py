from __future__ import annotations

from types import SimpleNamespace

import pytest

from runtime.ai_pipeline import (
    LocalSpeechRecognizer,
    OllamaChat,
    PerceptionSnapshot,
    ProactiveConversationScheduler,
    _asks_about_visual_context,
    _smooth_blendshapes,
    _visual_chat_frame,
    facial_expression_cues,
    raised_arm_cues,
)
from runtime.component import Component, ComponentState, InvalidStateTransitionError
from runtime.config import ConfigurationError, RuntimeSettings, load_runtime_settings
from runtime.event_bus import Event, EventBus, EventBusClosedError, EventBusFullError, EventType
from runtime.laptop_io import LaptopDeviceError, _audio_rms
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


def test_audio_rms_handles_empty_and_silent_buffers() -> None:
    class Samples:
        def __init__(self, values: list[float]) -> None:
            self.values = values
            self.size = len(values)

        def __pow__(self, power: int) -> Samples:
            return Samples([value**power for value in self.values])

        def mean(self) -> float:
            return sum(self.values) / len(self.values)

    assert _audio_rms(Samples([])) == 0.0
    assert _audio_rms(Samples([0.0, 0.0, 0.0, 0.0])) == 0.0
    assert _audio_rms(Samples([1.0, 1.0, 1.0, 1.0])) == 1.0


def test_live_preview_requires_optional_hardware_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    from runtime import laptop_io

    original_import_module = laptop_io.importlib.import_module

    def missing_cv2(name: str, *args: object, **kwargs: object) -> object:
        if name == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(laptop_io.importlib, "import_module", missing_cv2)

    with pytest.raises(LaptopDeviceError, match="hardware dependencies"):
        laptop_io._load_device_libraries()


def test_live_preview_stops_and_releases_devices_on_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    from runtime import laptop_io

    capture_state = {"released": False}
    stream_state = {"started": False, "stopped": False, "closed": False}

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, str]:
            return True, "frame"

        def release(self) -> None:
            capture_state["released"] = True

    class FakeStream:
        def start(self) -> None:
            stream_state["started"] = True

        def stop(self) -> None:
            stream_state["stopped"] = True

        def close(self) -> None:
            stream_state["closed"] = True

    fake_capture = FakeCapture()
    fake_stream = FakeStream()
    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda index: fake_capture,
        putText=lambda *args: None,
        imshow=lambda *args: None,
        waitKey=lambda delay: ord("q"),
        destroyAllWindows=lambda: None,
        FONT_HERSHEY_SIMPLEX=0,
    )
    fake_sounddevice = SimpleNamespace(
        query_devices=lambda *args: (
            [{"name": "fake mic", "max_input_channels": 1, "default_samplerate": 16000}]
            if not args
            else {"name": "fake mic", "max_input_channels": 1, "default_samplerate": 16000}
        ),
        default=SimpleNamespace(device=(0, 0)),
        InputStream=lambda **kwargs: fake_stream,
    )
    modules = {"cv2": fake_cv2, "sounddevice": fake_sounddevice}
    monkeypatch.setattr(laptop_io.importlib, "import_module", modules.__getitem__)

    laptop_io.run_live_preview(on_status=lambda message: None)

    assert capture_state["released"] is True
    assert stream_state == {"started": True, "stopped": True, "closed": True}


def test_pose_analysis_only_reports_observable_raised_arm_cues() -> None:
    landmarks = [SimpleNamespace(y=0.5, visibility=1.0) for _ in range(17)]
    landmarks[11].y = 0.4
    landmarks[12].y = 0.4
    landmarks[15].y = 0.2
    landmarks[16].y = 0.3

    assert raised_arm_cues(landmarks) == ("both arms raised",)

    landmarks[16].y = 0.6
    assert raised_arm_cues(landmarks) == ("left arm raised",)


def test_face_analysis_reports_movement_not_emotional_state() -> None:
    categories = [
        SimpleNamespace(category_name="mouthSmileLeft", score=0.8),
        SimpleNamespace(category_name="mouthSmileRight", score=0.7),
        SimpleNamespace(category_name="jawOpen", score=0.6),
    ]

    cues = facial_expression_cues(categories)

    assert cues == ("smile-like facial movement", "open-mouth movement")
    assert "happy" not in cues


def test_speech_recognizer_resamples_to_whisper_sample_rate() -> None:
    class FakeModel:
        received_length = 0

        def transcribe(self, audio: object, **kwargs: object) -> tuple[list[object], object]:
            self.received_length = len(audio)  # type: ignore[arg-type]
            return [SimpleNamespace(text="한국어 인식")], SimpleNamespace(language="ko")

    recognizer = object.__new__(LocalSpeechRecognizer)
    fake_model = FakeModel()
    recognizer._model = fake_model

    result = recognizer.transcribe([0.0] * 48_000, sample_rate=48_000)

    assert result == "한국어 인식"
    assert fake_model.received_length == 16_000


def test_local_chat_sends_only_transcript_and_textual_perception(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = OllamaChat()
    sent_payloads: list[dict[str, object]] = []

    def fake_post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        assert endpoint == "/api/chat"
        sent_payloads.append(payload)
        return {"message": {"content": "주변에 의자가 보입니다."}}

    monkeypatch.setattr(chat, "_post", fake_post)
    reply = chat.reply(
        "안녕하세요",
        PerceptionSnapshot(
            objects=("chair",),
            action_cues=("left arm raised",),
            expression_cues=("smile-like facial movement",),
        ),
    )

    serialized = str(sent_payloads[0])
    assert reply == "주변에 의자가 보입니다."
    assert "안녕하세요" in serialized
    assert "chair" in serialized
    assert "images" not in serialized
    assert "happy" not in serialized


def test_local_chat_uses_current_image_without_retaining_it(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = OllamaChat()
    sent_payloads: list[dict[str, object]] = []

    def fake_post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        sent_payloads.append(payload)
        return {"message": {"content": "책상 위에 컵이 보입니다."}}

    monkeypatch.setattr(chat, "_post", fake_post)
    chat.reply("지금 뭐가 보여?", PerceptionSnapshot(), image_jpeg=b"jpeg-frame")

    messages = sent_payloads[0]["messages"]
    assert isinstance(messages, list)
    assert "anBlZy1mcmFtZQ==" in str(messages[-1])
    assert sent_payloads[0]["model"] == chat.vision_model
    assert "images" not in str(chat._history)


def test_visual_context_is_only_requested_for_visual_questions() -> None:
    assert _asks_about_visual_context("지금 뭐가 보여?")
    assert _asks_about_visual_context("저 물건은 무슨 색이야?")
    assert not _asks_about_visual_context("오늘 날씨가 어때?")


def test_visual_chat_frame_is_resized_with_aspect_ratio_preserved() -> None:
    class FakeCv2:
        INTER_AREA = 1

        @staticmethod
        def resize(frame: object, size: tuple[int, int], interpolation: int) -> tuple[object, ...]:
            return frame, size, interpolation

    frame = SimpleNamespace(shape=(720, 1280, 3))

    assert _visual_chat_frame(FakeCv2(), frame) == (frame, (448, 252), FakeCv2.INTER_AREA)


def test_proactive_scheduler_waits_and_rate_limits_visual_cues() -> None:
    import random

    scheduler = ProactiveConversationScheduler(
        minimum_idle_seconds=60,
        maximum_idle_seconds=60,
        stable_cue_seconds=4,
        rng=random.Random(1),
        now=0,
    )
    scene = PerceptionSnapshot(objects=("cup",))

    assert scheduler.next_reason(scene, now=0) is None
    assert scheduler.next_reason(scene, now=59) is None
    assert scheduler.next_reason(scene, now=60) == "visual_change"

    scheduler.mark_prompted(scene, now=60)
    assert scheduler.next_reason(scene, now=119) is None
    assert scheduler.next_reason(scene, now=120) == "quiet_pause"


def test_blendshape_smoothing_reduces_single_frame_spikes() -> None:
    first = _smooth_blendshapes(
        {},
        [SimpleNamespace(category_name="mouthSmileLeft", score=1.0)],
    )
    second = _smooth_blendshapes(
        first,
        [SimpleNamespace(category_name="mouthSmileLeft", score=0.0)],
    )

    assert first["mouthSmileLeft"] == pytest.approx(0.45)
    assert second["mouthSmileLeft"] == pytest.approx(0.2475)
