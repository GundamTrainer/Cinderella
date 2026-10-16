from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ai.external_backends import (
    ExternalMediaPipeBackend,
    ExternalOllamaLanguageBackend,
    ExternalSapiSpeechBackend,
    ExternalWhisperBackend,
)


class VisionLiveBackend(Protocol):
    def initialize(self) -> None: ...

    def infer(self, frame: Any, observed_at: float) -> Any: ...

    def close(self) -> None: ...


class SpeechLiveBackend(Protocol):
    def initialize(self) -> None: ...

    def infer(self, audio_pcm: Any, sample_rate: int) -> str: ...

    def close(self) -> None: ...


class ConversationLiveBackend(Protocol):
    def initialize(self) -> None: ...

    def reply(self, text: str, perception: Any, image_jpeg: bytes | None = None) -> str: ...

    def initiate(self, perception: Any, *, reason: str) -> str: ...

    def close(self) -> None: ...


class VoiceLiveBackend(Protocol):
    @property
    def is_speaking(self) -> bool: ...

    def initialize(self) -> None: ...

    def observe_microphone_level(self, level: float) -> bool: ...

    def interrupt(self) -> None: ...

    def speak(self, text: str) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LiveBackends:
    vision: VisionLiveBackend
    speech: SpeechLiveBackend
    chat: ConversationLiveBackend
    voice: VoiceLiveBackend


def create_live_backends(
    *,
    speech_model: str,
    chat_model: str,
    vision_model: str,
    box_clamp_tolerance_pixels: float,
) -> LiveBackends:
    """Construct optional adapters in one place, outside the live pipeline."""
    return LiveBackends(
        vision=ExternalMediaPipeBackend(
            box_clamp_tolerance_pixels=box_clamp_tolerance_pixels
        ),
        speech=ExternalWhisperBackend(speech_model),
        chat=ExternalOllamaLanguageBackend(chat_model, vision_model),
        voice=ExternalSapiSpeechBackend(),
    )
