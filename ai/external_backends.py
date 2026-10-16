from __future__ import annotations

from typing import Any

from .contracts import BackendHealth, BackendInfo, BackendKind, BackendStatus


class ExternalBackendAdapter:
    """Lifecycle and provenance wrapper for a separately implemented legacy backend."""

    def __init__(
        self,
        name: str,
        *,
        model_version: str | None,
        checkpoint_source: str | None,
        execution_device: str,
        external_program: bool,
        network_required: bool,
    ) -> None:
        self._info = BackendInfo(
            name=name,
            kind=BackendKind.EXTERNAL,
            model_version=model_version,
            checkpoint_source=checkpoint_source,
            execution_device=execution_device,
            external_program=external_program,
            network_required=network_required,
        )
        self._status = BackendStatus.UNAVAILABLE
        self._detail = "Not initialized."

    @property
    def backend_info(self) -> BackendInfo:
        return self._info

    def health(self) -> BackendHealth:
        return BackendHealth(self._status, self._detail)

    def reset(self) -> None:
        return None

    def _mark_available(self) -> None:
        self._status = BackendStatus.AVAILABLE
        self._detail = ""

    def _mark_failed(self, exc: Exception) -> None:
        self._status = BackendStatus.FAILED
        self._detail = str(exc)


class ExternalWhisperBackend(ExternalBackendAdapter):
    def __init__(self, model_size: str = "small") -> None:
        super().__init__(
            "faster-whisper",
            model_version=model_size,
            checkpoint_source="Systran/faster-whisper",
            execution_device="cpu/int8",
            external_program=False,
            network_required=True,
        )
        self._model_size = model_size
        self._delegate: Any | None = None

    def initialize(self) -> None:
        from runtime.ai_pipeline import LocalSpeechRecognizer

        try:
            self._delegate = LocalSpeechRecognizer(self._model_size)
        except Exception as exc:
            self._mark_failed(exc)
            raise
        self._mark_available()

    def infer(self, audio_pcm: Any, sample_rate: int) -> str:
        if self._delegate is None:
            raise RuntimeError("External Whisper backend is not initialized.")
        return self._delegate.transcribe(audio_pcm, sample_rate)

    def close(self) -> None:
        self._delegate = None


class ExternalMediaPipeBackend(ExternalBackendAdapter):
    def __init__(self) -> None:
        super().__init__(
            "MediaPipe Tasks",
            model_version="EfficientDet Lite2 / Pose Lite / Face Landmarker",
            checkpoint_source="Google MediaPipe model assets",
            execution_device="cpu",
            external_program=False,
            network_required=True,
        )
        self._delegate: Any | None = None

    def initialize(self) -> None:
        from runtime.ai_pipeline import LocalVisionAnalyzer

        try:
            self._delegate = LocalVisionAnalyzer()
        except Exception as exc:
            self._mark_failed(exc)
            raise
        self._mark_available()

    def infer(self, frame: Any, observed_at: float) -> Any:
        if self._delegate is None:
            raise RuntimeError("External MediaPipe backend is not initialized.")
        return self._delegate.analyze(frame, round(observed_at * 1000))

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()
            self._delegate = None


class ExternalOllamaLanguageBackend(ExternalBackendAdapter):
    def __init__(self, model: str = "qwen2.5:3b", vision_model: str = "qwen2.5vl:3b") -> None:
        super().__init__(
            "Ollama local process",
            model_version=f"{model}; {vision_model}",
            checkpoint_source="Ollama model registry",
            execution_device="local Ollama host",
            external_program=True,
            network_required=True,
        )
        self._model = model
        self._vision_model = vision_model
        self._delegate: Any | None = None

    def initialize(self) -> None:
        from runtime.ai_pipeline import OllamaChat

        self._delegate = OllamaChat(self._model, self._vision_model)
        try:
            self._delegate.ensure_ready()
        except Exception as exc:
            self._mark_failed(exc)
            raise
        self._mark_available()

    def reply(self, text: str, perception: Any, image_jpeg: bytes | None = None) -> str:
        if self._delegate is None:
            raise RuntimeError("External Ollama backend is not initialized.")
        return self._delegate.reply(text, perception, image_jpeg)

    def initiate(self, perception: Any, *, reason: str) -> str:
        if self._delegate is None:
            raise RuntimeError("External Ollama backend is not initialized.")
        return self._delegate.initiate(perception, None, reason=reason)

    def close(self) -> None:
        self._delegate = None


class ExternalSapiSpeechBackend(ExternalBackendAdapter):
    def __init__(self) -> None:
        super().__init__(
            "Windows SAPI via pyttsx3",
            model_version=None,
            checkpoint_source="Installed Windows Korean SAPI voice",
            execution_device="local Windows audio output",
            external_program=False,
            network_required=False,
        )
        self._delegate: Any | None = None

    def initialize(self) -> None:
        from runtime.ai_pipeline import LocalVoiceOutput

        try:
            self._delegate = LocalVoiceOutput()
        except Exception as exc:
            self._mark_failed(exc)
            raise
        self._mark_available()

    def speak(self, text: str) -> None:
        if self._delegate is None:
            raise RuntimeError("External SAPI backend is not initialized.")
        self._delegate.speak(text)

    @property
    def is_speaking(self) -> bool:
        return bool(self._delegate and self._delegate.is_speaking)

    def observe_microphone_level(self, level: float) -> bool:
        if self._delegate is None:
            raise RuntimeError("External SAPI backend is not initialized.")
        return self._delegate.observe_microphone_level(level)

    def interrupt(self) -> None:
        if self._delegate is not None:
            self._delegate.interrupt()

    def stop(self) -> None:
        self.interrupt()

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()
            self._delegate = None
