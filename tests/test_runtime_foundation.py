from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.scene import WorldState
from runtime.ai_pipeline import (
    LocalChatError,
    LocalSpeechRecognizer,
    OllamaChat,
    PerceptionSnapshot,
    ProactiveConversationScheduler,
    _asks_about_visual_context,
    _ChatJob,
    _queue_text_chat_message,
    _smooth_blendshapes,
    _visual_chat_frame,
    facial_expression_cues,
    raised_arm_cues,
)
from runtime.capabilities import (
    CapabilityStatus,
    capability_overlay_lines,
    capability_statuses,
    implementation_coverage_percent,
)
from runtime.component import Component, ComponentState, InvalidStateTransitionError
from runtime.config import ConfigurationError, RuntimeSettings, load_runtime_settings
from runtime.event_bus import Event, EventBus, EventBusClosedError, EventBusFullError, EventType
from runtime.laptop_io import LaptopDeviceError, _audio_rms
from runtime.orchestrator import RuntimeOrchestrator
from runtime.physical_ai import (
    ActionProposal,
    PhysicalAiSimulator,
    SimulatedAction,
    run_physical_ai_safety_demo,
)
from runtime.simulators import SimulatedAudioSource, SimulatedHardwareAdapter, SimulatedVisionSource


def test_capability_overlay_reports_implementation_coverage_not_human_similarity() -> None:
    assert implementation_coverage_percent() == 54
    assert capability_overlay_lines() == (
        "Developer estimate only: 54% (not human similarity or accuracy)",
        "Reply 50% | Situation 50% | Affect cues 25% | Person 50%",
        "Object 50% | Dialogue 100%",
    )


def test_capability_statuses_are_provenance_not_an_accuracy_percentage() -> None:
    statuses = capability_statuses()
    by_name = {item.name: item for item in statuses}

    assert by_name["speech_recognition"].status == CapabilityStatus.DISABLED
    assert by_name["speech_recognition"].implementation_type == "external"
    assert by_name["simulated_physical_actions"].status == CapabilityStatus.SIMULATED
    assert by_name["external_information_tools"].status == CapabilityStatus.DISABLED


def test_physical_ai_simulation_emits_proposal_safety_and_simulated_command() -> None:
    bus = EventBus(8)
    simulator = PhysicalAiSimulator(bus)
    scene = WorldState(observed_at=1.0, objects=(), uncertain_labels=())
    proposal = ActionProposal(
        SimulatedAction.MOVE_FORWARD,
        confidence=0.9,
        reason="demonstration in empty synthetic scene",
        distance_meters=0.05,
    )

    decision = simulator.process(proposal, scene)
    events = bus.drain()

    assert decision.approved
    assert [event.event_type for event in events] == [
        EventType.PLANNER_ACTION_PROPOSED,
        EventType.SAFETY_ACTION_APPROVED,
        EventType.HARDWARE_COMMAND_SIMULATED,
    ]
    assert events[-1].payload["synthetic"] is True
    assert events[-1].payload["real_hardware_actuated"] is False


def test_physical_ai_safety_gate_rejects_motion_when_scene_is_uncertain() -> None:
    bus = EventBus(8)
    simulator = PhysicalAiSimulator(bus)
    scene = WorldState(
        observed_at=1.0,
        objects=(),
        uncertain_labels=("unknown object",),
    )
    proposal = ActionProposal(
        SimulatedAction.MOVE_FORWARD,
        confidence=0.99,
        reason="attempt motion",
        distance_meters=0.05,
    )

    decision = simulator.process(proposal, scene)
    events = bus.drain()

    assert not decision.approved
    assert decision.reason == "scene_not_confirmed_clear"
    assert [event.event_type for event in events] == [
        EventType.PLANNER_ACTION_PROPOSED,
        EventType.SAFETY_ACTION_REJECTED,
    ]


def test_physical_ai_rejects_out_of_bounds_and_low_confidence_motion() -> None:
    simulator = PhysicalAiSimulator(EventBus(8))
    scene = WorldState(observed_at=1.0, objects=(), uncertain_labels=())

    too_far = ActionProposal(
        SimulatedAction.MOVE_FORWARD,
        confidence=0.9,
        reason="oversized test motion",
        distance_meters=0.2,
    )
    low_confidence = ActionProposal(
        SimulatedAction.MOVE_FORWARD,
        confidence=0.4,
        reason="uncertain test motion",
        distance_meters=0.05,
    )

    assert simulator.evaluate(too_far, scene).reason == "simulated_distance_limit_exceeded"
    assert simulator.evaluate(low_confidence, scene).reason == "confidence_below_safety_threshold"


def test_physical_ai_demo_is_deterministic_and_never_actuates_hardware() -> None:
    first = run_physical_ai_safety_demo(EventBus(8))
    second = run_physical_ai_safety_demo(EventBus(8))

    assert [event.event_type for event in first] == [
        EventType.PLANNER_ACTION_PROPOSED,
        EventType.SAFETY_ACTION_APPROVED,
        EventType.HARDWARE_COMMAND_SIMULATED,
        EventType.PLANNER_ACTION_PROPOSED,
        EventType.SAFETY_ACTION_REJECTED,
    ]
    assert [(event.event_type, event.payload, event.confidence) for event in first] == [
        (event.event_type, event.payload, event.confidence) for event in second
    ]
    assert all(event.payload["synthetic"] is True for event in first)
    assert all(event.payload["real_hardware_actuated"] is False for event in first[1:])


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
    orchestrator = RuntimeOrchestrator(
        RuntimeSettings.from_mapping(
            {
                "runtime_queue_capacity": 8,
                "event_publish_timeout": 0.25,
                "shutdown_timeout": 1.0,
                "logging_level": "INFO",
                "simulation_seed": 42,
                "audio_simulator_enabled": True,
                "vision_simulator_enabled": True,
                "hardware_simulator_enabled": True,
            }
        ),
        bus=bus,
        components=[good, bad],
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        orchestrator.initialize()

    assert good.state == ComponentState.CLOSED


def test_shutdown_closes_all_components() -> None:
    bus = EventBus(16)
    first = DummyComponent("first")
    second = DummyComponent("second")
    settings = RuntimeSettings.from_mapping(
        {
            "runtime_queue_capacity": 16,
            "event_publish_timeout": 0.25,
            "shutdown_timeout": 1.0,
            "logging_level": "INFO",
            "simulation_seed": 42,
            "audio_simulator_enabled": True,
            "vision_simulator_enabled": True,
            "hardware_simulator_enabled": True,
        }
    )
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
    assert settings.audio_input_queue_capacity == 24
    assert settings.network.mode.value == "offline"
    assert settings.network.allow_external_requests is False
    assert settings.network.allowed_domains == ("en.wikipedia.org", "api.open-meteo.com")


def test_runtime_settings_reject_invalid_audio_queue_capacity() -> None:
    with pytest.raises(ConfigurationError, match="audio_input_queue_capacity"):
        RuntimeSettings.from_mapping(
            {
                "runtime_queue_capacity": 128,
                "event_publish_timeout": 0.25,
                "shutdown_timeout": 2.0,
                "logging_level": "INFO",
                "simulation_seed": 42,
                "audio_simulator_enabled": True,
                "vision_simulator_enabled": True,
                "hardware_simulator_enabled": True,
                "audio_input_queue_capacity": 0,
            }
        )


def test_runtime_settings_reject_invalid_values() -> None:
    with pytest.raises(ConfigurationError, match="runtime_queue_capacity"):
        RuntimeSettings.from_mapping(
            {
                "runtime_queue_capacity": 0,
                "event_publish_timeout": 0.25,
                "shutdown_timeout": 1.0,
                "logging_level": "INFO",
                "simulation_seed": 42,
                "audio_simulator_enabled": True,
                "vision_simulator_enabled": True,
                "hardware_simulator_enabled": True,
            }
        )


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


def test_live_preview_requires_optional_hardware_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_chat_context_excludes_face_and_emotion_inferences() -> None:
    chat = OllamaChat()
    perception = PerceptionSnapshot(
        face_detected=True,
        expression_cues=("smile-like facial movement",),
        tentative_interpretations=(
            SimpleNamespace(
                hypothesis="possible_positive_affect",
                confidence=0.25,
            ),
        )
    )

    context = perception.prompt_context()

    assert "시각 관찰 단서" in context
    assert "smile-like facial movement" not in context
    assert "possible_positive_affect" not in context
    assert "감정·의도·신원·건강을 단정하지 마" in chat._system


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


def test_local_chat_sends_only_transcript_for_nonvisual_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert "chair" not in serialized
    assert "left arm raised" not in serialized
    assert "smile-like facial movement" not in serialized
    assert "images" not in serialized
    assert "happy" not in serialized
    assert sent_payloads[0]["options"]["temperature"] == 0.72


@pytest.mark.parametrize(
    "base_url",
    (
        "https://127.0.0.1:11434",
        "http://ollama.example.com:11434",
        "http://192.168.1.15:11434",
        "http://user@localhost:11434",
    ),
)
def test_local_chat_rejects_non_loopback_endpoints(base_url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        OllamaChat(base_url=base_url)


def test_local_chat_uses_current_image_without_retaining_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert "지금 뭐가 보여?" in str(messages[-1])
    assert sent_payloads[0]["model"] == chat.vision_model
    assert "images" not in str(chat._history)
    assert "얼굴 움직임" not in str(messages)
    assert "감정 관련 가설" not in str(messages)


def test_visual_question_uses_only_non_face_observation_context_without_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = OllamaChat()
    sent_payloads: list[dict[str, object]] = []

    def fake_post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        sent_payloads.append(payload)
        return {"message": {"content": "감지된 물체 후보는 컵입니다."}}

    monkeypatch.setattr(chat, "_post", fake_post)
    chat.reply(
        "화면에 어떤 물건이 있어?",
        PerceptionSnapshot(
            objects=("cup",),
            expression_cues=("smile-like facial movement",),
            tentative_interpretations=(SimpleNamespace(hypothesis="possible_positive_affect"),),
        ),
    )

    serialized = str(sent_payloads[0]["messages"])
    assert "cup" in serialized
    assert "smile-like facial movement" not in serialized
    assert "possible_positive_affect" not in serialized


def test_visual_chat_keeps_dialogue_context_without_retaining_prior_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = OllamaChat()
    sent_payloads: list[dict[str, object]] = []

    def fake_post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        sent_payloads.append(payload)
        return {"message": {"content": f"응답 {len(sent_payloads)}"}}

    monkeypatch.setattr(chat, "_post", fake_post)
    chat.reply("오늘 기분이 별로야.", PerceptionSnapshot(), image_jpeg=b"old-frame")
    chat.reply("지금 뭐가 보여?", PerceptionSnapshot(objects=("chair",)), image_jpeg=b"new-frame")

    messages = sent_payloads[1]["messages"]
    assert isinstance(messages, list)
    assert any(
        message.get("role") == "user" and message.get("content") == "오늘 기분이 별로야."
        for message in messages
    )
    assert any(
        message.get("role") == "assistant" and message.get("content") == "응답 1"
        for message in messages
    )
    assert "b2xkLWZyYW1l" not in str(messages)
    assert "bmV3LWZyYW1l" in str(messages)
    assert "images" not in str(chat._history)


def test_failed_chat_turn_is_not_retained_in_conversation_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = OllamaChat()

    def fail_post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        del endpoint, payload
        raise LocalChatError("simulated local model failure")

    monkeypatch.setattr(chat, "_post", fail_post)
    with pytest.raises(LocalChatError, match="simulated"):
        chat.reply("이 질문은 실패해야 해.", PerceptionSnapshot())

    assert chat._history == []


def test_proactive_instruction_is_not_saved_as_a_user_chat_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = OllamaChat()
    sent_payloads: list[dict[str, object]] = []

    def fake_post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        sent_payloads.append(payload)
        return {"message": {"content": "잠깐 쉬어가는 중이군요."}}

    monkeypatch.setattr(chat, "_post", fake_post)
    chat.initiate(PerceptionSnapshot(), None, reason="quiet_pause")

    assert chat._history == [
        {"role": "assistant", "content": "잠깐 쉬어가는 중이군요."}
    ]
    chat.reply("맞아요.", PerceptionSnapshot())

    messages = sent_payloads[1]["messages"]
    assert isinstance(messages, list)
    assert any(
        message.get("role") == "assistant" and message.get("content") == "잠깐 쉬어가는 중이군요."
        for message in messages
    )
    assert "quiet pause in the conversation" not in str(messages)


def test_visual_context_is_only_requested_for_visual_questions() -> None:
    assert _asks_about_visual_context("지금 뭐가 보여?")
    assert _asks_about_visual_context("저 물건은 무슨 색이야?")
    assert _asks_about_visual_context("이게 뭔지 알려줘")
    assert not _asks_about_visual_context("오늘 날씨가 어때?")
    assert not _asks_about_visual_context("너는 뭐야?")
    assert not _asks_about_visual_context("누가 이겼어?")
    assert not _asks_about_visual_context("그 영화는 뭐야?")


def test_typed_chat_is_queued_without_disabling_microphone_path() -> None:
    import queue

    jobs: queue.Queue[_ChatJob | None] = queue.Queue(maxsize=1)
    scheduler = ProactiveConversationScheduler(now=0)

    assert _queue_text_chat_message("  안녕, 지금 뭐가 보여?  ", jobs, scheduler)
    assert not _queue_text_chat_message("두 번째 메시지", jobs, scheduler)
    job = jobs.get_nowait()
    assert job is not None
    assert job.transcript == "안녕, 지금 뭐가 보여?"
    assert job.source == "keyboard"
    assert _ChatJob(transcript="마이크 입력").source == "microphone"


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


def test_face_movement_alone_does_not_trigger_proactive_conversation() -> None:
    import random

    scheduler = ProactiveConversationScheduler(
        minimum_idle_seconds=60,
        maximum_idle_seconds=120,
        stable_cue_seconds=4,
        rng=random.Random(1),
        now=0,
    )
    face_movement = PerceptionSnapshot(expression_cues=("smile-like facial movement",))

    assert scheduler.next_reason(face_movement, now=59) is None
    assert scheduler.next_reason(face_movement, now=60) is None


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
