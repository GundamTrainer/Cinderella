from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class BackendKind(StrEnum):
    EXTERNAL = "external"
    INTERNAL = "internal"
    SIMULATED = "simulated"
    DISABLED = "disabled"


class BackendStatus(StrEnum):
    INITIALIZING = "initializing"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    FAILED = "failed"


@dataclass(frozen=True)
class BackendInfo:
    name: str
    kind: BackendKind
    model_version: str | None
    checkpoint_source: str | None
    uses_pretrained_weights: bool | None
    execution_device: str
    external_program: bool
    network_required: bool


@dataclass(frozen=True)
class BackendHealth:
    status: BackendStatus
    detail: str = ""


@runtime_checkable
class Backend(Protocol):
    @property
    def backend_info(self) -> BackendInfo: ...

    def initialize(self) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...

    def health(self) -> BackendHealth: ...


@runtime_checkable
class SpeechRecognizer(Backend, Protocol):
    def infer(self, audio_pcm: Any, sample_rate: int) -> str: ...


@runtime_checkable
class VoiceActivityDetector(Backend, Protocol):
    def infer(self, audio_pcm: Any, sample_rate: int) -> Any: ...


@runtime_checkable
class SpeakerIdentifier(Backend, Protocol):
    def infer(self, audio_pcm: Any, sample_rate: int) -> Any: ...


@runtime_checkable
class SoundEventRecognizer(Backend, Protocol):
    def infer(self, audio_pcm: Any, sample_rate: int) -> Any: ...


@runtime_checkable
class ObjectDetector(Backend, Protocol):
    def infer(self, frame: Any, observed_at: float) -> Any: ...


@runtime_checkable
class PoseEstimator(Backend, Protocol):
    def infer(self, frame: Any, observed_at: float) -> Any: ...


@runtime_checkable
class HandPoseEstimator(Backend, Protocol):
    def infer(self, frame: Any, observed_at: float) -> Any: ...


@runtime_checkable
class FaceFeatureExtractor(Backend, Protocol):
    def infer(self, frame: Any, observed_at: float) -> Any: ...


@runtime_checkable
class ActionRecognizer(Backend, Protocol):
    def infer(self, observations: Any, observed_at: float) -> Any: ...


@dataclass(frozen=True)
class LanguageRequest:
    user_text: str
    visual_events: tuple[Any, ...] = ()
    audio_events: tuple[Any, ...] = ()
    short_context: tuple[dict[str, str], ...] = ()
    retrieved_memories: tuple[str, ...] = ()
    feature_status: tuple[tuple[str, BackendStatus], ...] = ()


@dataclass(frozen=True)
class LanguageResponse:
    response_text: str
    confidence: float | None
    source_backend: str
    external_model: bool
    used_observations: tuple[str, ...] = ()
    proposed_actions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@runtime_checkable
class LanguageEngine(Backend, Protocol):
    def infer(self, request: LanguageRequest) -> LanguageResponse: ...


@runtime_checkable
class SpeechSynthesizer(Backend, Protocol):
    def synthesize(self, normalized_text: str) -> Any: ...

    def speak(self, normalized_text: str) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class VisionLanguageEngine(Backend, Protocol):
    def infer(self, request: LanguageRequest, frame: Any) -> LanguageResponse: ...
