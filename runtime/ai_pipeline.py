from __future__ import annotations

import base64
import json
import logging
import queue
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .event_bus import Event, EventBus, EventBusFullError, EventType

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:1.5b"
VISION_CHAT_MODEL = "qwen2.5vl:3b"
WHISPER_MODEL = "small"
SPEECH_WINDOW_SECONDS = 5
VISION_INTERVAL_SECONDS = 0.4
VISUAL_CHAT_MAX_SIDE = 448
PROACTIVE_MIN_IDLE_SECONDS = 60
PROACTIVE_MAX_IDLE_SECONDS = 140
MODEL_ASSETS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "efficientdet_lite2.tflite": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite2/int8/1/efficientdet_lite2.tflite"
    ),
}


class LocalInferenceError(RuntimeError):
    """Raised when a local recognition or inference component cannot run."""


class LocalChatError(LocalInferenceError):
    """Raised when the local Ollama chat service is unavailable or fails."""


@dataclass(frozen=True)
class PerceptionSnapshot:
    objects: tuple[str, ...] = ()
    action_cues: tuple[str, ...] = ()
    expression_cues: tuple[str, ...] = ()
    confidence: float = 0.0
    face_detected: bool = False

    def prompt_context(self) -> str:
        return (
            "카메라 영상 관찰 단서(불확실할 수 있으며 감정·의도의 확정 판정이 아님): "
            f"물체={', '.join(self.objects) if self.objects else '없음/미검출'}, "
            f"자세·행동 단서={', '.join(self.action_cues) if self.action_cues else '없음/미검출'}, "
            f"얼굴 감지={'예' if self.face_detected else '아니오'}, "
            f"얼굴 움직임 단서={', '.join(self.expression_cues) if self.expression_cues else '없음/미검출'}."
        )


def model_cache_directory() -> Path:
    cache = Path.home() / ".cache" / "cinderella" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _asks_about_visual_context(transcript: str) -> bool:
    markers = (
        "보여",
        "보이는",
        "보입",
        "뭐가",
        "뭐야",
        "물체",
        "사물",
        "장면",
        "색깔",
        "색이",
        "사진",
        "이미지",
        "카메라",
        "앞에",
        "옆에",
        "이거",
        "이건",
        "저거",
        "저건",
        "표정",
        "누가",
        "what do you see",
        "what is visible",
        "describe this image",
        "picture",
        "camera",
        "object",
        "color",
    )
    normalized = transcript.casefold()
    return any(marker in normalized for marker in markers)


def _ensure_model_asset(filename: str) -> Path:
    target = model_cache_directory() / filename
    if target.is_file() and target.stat().st_size > 0:
        return target

    temporary = target.with_suffix(target.suffix + ".download")
    try:
        request = urllib.request.Request(
            MODEL_ASSETS[filename], headers={"User-Agent": "Cinderella-local-runtime/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size == 0:
            raise LocalInferenceError(f"Downloaded model asset is empty: {filename}")
        temporary.replace(target)
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise LocalInferenceError(
            f"Could not download model asset {filename}. Check your internet connection and retry: {exc}"
        ) from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def raised_arm_cues(landmarks: list[Any]) -> tuple[str, ...]:
    """Describe only clear wrist-above-shoulder pose cues from MediaPipe landmarks."""
    if len(landmarks) <= 16:
        return ()
    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
    left_wrist, right_wrist = landmarks[15], landmarks[16]
    left_raised = (
        getattr(left_shoulder, "visibility", 1.0) >= 0.5
        and getattr(left_wrist, "visibility", 1.0) >= 0.5
        and left_wrist.y < left_shoulder.y - 0.05
    )
    right_raised = (
        getattr(right_shoulder, "visibility", 1.0) >= 0.5
        and getattr(right_wrist, "visibility", 1.0) >= 0.5
        and right_wrist.y < right_shoulder.y - 0.05
    )
    if left_raised and right_raised:
        return ("both arms raised",)
    if left_raised:
        return ("left arm raised",)
    if right_raised:
        return ("right arm raised",)
    return ("person posture detected",)


def facial_expression_cues(categories: list[Any]) -> tuple[str, ...]:
    """Map smoothed face blendshapes to visible movements, not psychological states."""
    scores = {category.category_name: float(category.score) for category in categories}
    cues: list[str] = []
    smile = (scores.get("mouthSmileLeft", 0.0) + scores.get("mouthSmileRight", 0.0)) / 2
    brow_down = (scores.get("browDownLeft", 0.0) + scores.get("browDownRight", 0.0)) / 2
    brow_up = (scores.get("browOuterUpLeft", 0.0) + scores.get("browOuterUpRight", 0.0)) / 2
    eyes_wide = (scores.get("eyeWideLeft", 0.0) + scores.get("eyeWideRight", 0.0)) / 2
    eyes_squint = (scores.get("eyeSquintLeft", 0.0) + scores.get("eyeSquintRight", 0.0)) / 2
    if smile >= 0.5:
        cues.append("smile-like facial movement")
    if brow_down >= 0.5:
        cues.append("brow-lowering movement")
    if brow_up >= 0.5:
        cues.append("eyebrow-raising movement")
    if eyes_wide >= 0.55:
        cues.append("eyes-widening movement")
    if eyes_squint >= 0.55:
        cues.append("eyes-narrowing movement")
    if scores.get("jawOpen", 0.0) >= 0.6:
        cues.append("open-mouth movement")
    return tuple(cues)


class LocalVisionAnalyzer:
    """CPU-friendly local object, pose, and facial-movement analysis."""

    def __init__(self) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise LocalInferenceError(
                'Vision runtime is missing. Install it with: python -m pip install -e ".[ai,hardware]"'
            ) from exc

        self._cv2 = cv2
        self._mp = mp
        self._vision = vision
        self._object_detector = None
        self._pose_landmarker = None
        self._face_landmarker = None
        self._smoothed_blendshapes: dict[str, float] = {}
        try:
            self._object_detector = vision.ObjectDetector.create_from_options(
                vision.ObjectDetectorOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(_ensure_model_asset("efficientdet_lite2.tflite"))
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    max_results=10,
                    score_threshold=0.25,
                )
            )
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(_ensure_model_asset("pose_landmarker_lite.task"))
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                )
            )
            self._face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(_ensure_model_asset("face_landmarker.task"))
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=2,
                    output_face_blendshapes=True,
                    min_face_detection_confidence=0.5,
                )
            )
        except Exception as exc:
            self.close()
            raise LocalInferenceError(f"Could not initialize MediaPipe vision models: {exc}") from exc

    def analyze(self, frame: Any, timestamp_ms: int) -> PerceptionSnapshot:
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        object_result = self._object_detector.detect_for_video(image, timestamp_ms)
        pose_result = self._pose_landmarker.detect_for_video(image, timestamp_ms)
        face_result = self._face_landmarker.detect_for_video(image, timestamp_ms)

        objects: list[str] = []
        confidences: list[float] = []
        for detection in object_result.detections:
            if detection.categories:
                category = detection.categories[0]
                objects.append(category.category_name)
                confidences.append(float(category.score))

        action_cues: tuple[str, ...] = ()
        if pose_result.pose_landmarks:
            action_cues = raised_arm_cues(pose_result.pose_landmarks[0])

        expression_cues: tuple[str, ...] = ()
        if face_result.face_blendshapes:
            self._smoothed_blendshapes = _smooth_blendshapes(
                self._smoothed_blendshapes,
                face_result.face_blendshapes[0],
            )
            expression_cues = facial_expression_cues(
                [
                    SimpleNamespace(category_name=name, score=score)
                    for name, score in self._smoothed_blendshapes.items()
                ]
            )
        else:
            self._smoothed_blendshapes = {}

        return PerceptionSnapshot(
            objects=tuple(objects),
            action_cues=action_cues,
            expression_cues=expression_cues,
            confidence=max(confidences, default=0.0),
            face_detected=bool(face_result.face_landmarks),
        )

    def close(self) -> None:
        for task in (self._face_landmarker, self._pose_landmarker, self._object_detector):
            if task is not None:
                task.close()


def _smooth_blendshapes(
    previous: dict[str, float],
    categories: list[Any],
    *,
    smoothing: float = 0.45,
) -> dict[str, float]:
    current = {category.category_name: float(category.score) for category in categories}
    names = previous.keys() | current.keys()
    return {
        name: smoothing * current.get(name, 0.0) + (1.0 - smoothing) * previous.get(name, 0.0)
        for name in names
    }


class LocalSpeechRecognizer:
    """Multilingual local Whisper transcription on CPU."""

    def __init__(self, model_size: str = WHISPER_MODEL) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise LocalInferenceError(
                'Speech runtime is missing. Install it with: python -m pip install -e ".[ai]"'
            ) from exc
        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            download_root=str(model_cache_directory() / "whisper"),
        )

    def transcribe(self, samples: Any, sample_rate: int) -> str:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")
        if len(samples) == 0:
            return ""
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        target_rate = 16_000
        if sample_rate != target_rate:
            output_size = round(len(audio) * target_rate / sample_rate)
            if output_size == 0:
                return ""
            source_positions = np.arange(len(audio), dtype=np.float64)
            target_positions = np.arange(output_size, dtype=np.float64) * sample_rate / target_rate
            audio = np.interp(target_positions, source_positions, audio).astype(np.float32)
        segments, _ = self._model.transcribe(
            audio,
            language="ko",
            vad_filter=True,
            condition_on_previous_text=False,
            beam_size=3,
        )
        return "".join(segment.text for segment in segments).strip()


class OllamaChat:
    """Chat with a local Ollama model over loopback HTTP only."""

    def __init__(
        self,
        model: str = CHAT_MODEL,
        vision_model: str = VISION_CHAT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.vision_model = vision_model
        self.base_url = base_url.rstrip("/")
        self._history: list[dict[str, Any]] = []
        self._system = (
            "You are a locally running conversational AI. Speak naturally and warmly in Korean, "
            "like an attentive conversation partner, but never pretend to be human or claim to "
            "have real feelings. You may describe concrete visible objects, posture, and facial "
            "movements as uncertain observations. Never infer true emotion, intention, identity, "
            "health, or personality from a face or voice. If unsure, ask instead of guessing. "
            "The current user image, when present, is primary evidence: inspect it directly and "
            "do not treat an automated 'none detected' cue as proof that something is absent. "
            "Keep spoken replies concise and avoid repetitive questions."
        )

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise LocalChatError(
                "Could not reach local Ollama. Install/start Ollama and run "
                f"`ollama pull {self.model}`. No external AI service is used."
            ) from exc

    def ensure_ready(self) -> None:
        request = urllib.request.Request(f"{self.base_url}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise LocalChatError(
                "Local Ollama is not responding. Install Ollama, start it, then run "
                f"`ollama pull {self.model}`."
            ) from exc
        installed = {item.get("name") for item in data.get("models", [])}
        for model in dict.fromkeys((self.model, self.vision_model)):
            if model not in installed:
                raise LocalChatError(
                    f"Local model is missing. Download it with `ollama pull {model}`."
                )

    def reply(
        self,
        transcript: str,
        perception: PerceptionSnapshot,
        image_jpeg: bytes | None = None,
    ) -> str:
        context = perception.prompt_context()
        use_vision_model = image_jpeg is not None and _asks_about_visual_context(transcript)
        context += "\n자동 검출 결과는 참고용이며, 미검출이 물체의 부재를 뜻하지는 않습니다."
        if use_vision_model:
            context = (
                f"{context}\nThe user asked in Korean: {transcript!r}. Inspect the attached "
                "image directly. In one concise Korean sentence, describe the visible objects, "
                "colors, and layout to answer the user. If the question is unclear, describe the "
                "image rather than asking what the user means."
            )
        images = (
            [base64.b64encode(image_jpeg).decode("ascii")]
            if image_jpeg is not None and use_vision_model
            else []
        )
        user_message: dict[str, Any] = {
            "role": "user",
            "content": (
                context
                if use_vision_model
                else f"{context}\n\n사용자 발화: {transcript}"
            ),
        }
        request_message = dict(user_message)
        if images:
            request_message["images"] = images
            request_message["content"] = (
                "Inspect the attached image directly. Describe its visible colors and layout "
                "in Korean. The detector may have missed objects."
            )
        self._history.append(user_message)
        self._history = self._history[-8:]
        history_messages = (
            [{"role": "system", "content": self._system}, request_message]
            if use_vision_model
            else [
                {"role": "system", "content": self._system},
                *self._history[:-1],
                request_message,
            ]
        )
        result = self._post(
            "/api/chat",
            {
                "model": self.vision_model if use_vision_model else self.model,
                "messages": history_messages,
                "stream": False,
                "options": {
                    "temperature": 0.1 if use_vision_model else 0.4,
                    "num_ctx": 4096,
                    "num_predict": 100 if use_vision_model else 256,
                },
            },
        )
        answer = result.get("message", {}).get("content", "").strip()
        if not answer:
            raise LocalChatError("Local Ollama returned an empty response.")
        self._history.append({"role": "assistant", "content": answer})
        self._history = self._history[-8:]
        return answer

    def initiate(
        self,
        perception: PerceptionSnapshot,
        image_jpeg: bytes | None,
        *,
        reason: str,
    ) -> str:
        images = [base64.b64encode(image_jpeg).decode("ascii")] if image_jpeg else []
        if reason == "visual_change":
            instruction = (
                "You are initiating a conversation because a visible scene or movement cue "
                "changed after a quiet interval. Make one brief, natural Korean observation or "
                "ask a gentle question about something clearly visible. Do not label the user's "
                "emotion or say you are bored. If there is nothing useful to say, reply exactly "
                "SKIP."
            )
        else:
            instruction = (
                "There has been a quiet pause in the conversation. Optionally initiate one brief, "
                "natural Korean check-in or open question. Do not claim to be bored, lonely, or "
                "human, and do not infer the user's mood. If no useful prompt comes to mind, "
                "reply exactly SKIP."
            )
        message: dict[str, Any] = {
            "role": "user",
            "content": f"{instruction}\n\n{perception.prompt_context()}",
        }
        request_message = dict(message)
        if images:
            request_message["images"] = images
        result = self._post(
            "/api/chat",
            {
                "model": self.vision_model if images else self.model,
                "messages": [
                    {"role": "system", "content": self._system},
                    *self._history,
                    request_message,
                ],
                "stream": False,
                "options": {"temperature": 0.75, "num_ctx": 4096},
            },
        )
        answer = result.get("message", {}).get("content", "").strip()
        if not answer:
            raise LocalChatError("Local Ollama returned an empty proactive response.")
        if answer.casefold() == "skip":
            return ""
        self._history.extend([message, {"role": "assistant", "content": answer}])
        self._history = self._history[-8:]
        return answer


class _LiveState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.perception = PerceptionSnapshot()
        self.transcript = ""
        self.reply = ""
        self.image_jpeg: bytes | None = None

    def update_perception(self, value: PerceptionSnapshot) -> None:
        with self.lock:
            self.perception = value

    def read(self) -> tuple[PerceptionSnapshot, str, str]:
        with self.lock:
            return self.perception, self.transcript, self.reply

    def update_transcript(self, value: str) -> PerceptionSnapshot:
        with self.lock:
            self.transcript = value
            return self.perception

    def update_reply(self, value: str) -> None:
        with self.lock:
            self.reply = value

    def update_image(self, value: bytes) -> None:
        with self.lock:
            self.image_jpeg = value

    def read_image(self) -> bytes | None:
        with self.lock:
            return self.image_jpeg


class ProactiveConversationScheduler:
    """Rate-limit unsolicited conversation and require stable visual cues."""

    def __init__(
        self,
        *,
        minimum_idle_seconds: float = PROACTIVE_MIN_IDLE_SECONDS,
        maximum_idle_seconds: float = PROACTIVE_MAX_IDLE_SECONDS,
        stable_cue_seconds: float = 4.0,
        rng: Any | None = None,
        now: float | None = None,
    ) -> None:
        if minimum_idle_seconds <= 0 or maximum_idle_seconds < minimum_idle_seconds:
            raise ValueError("Proactive idle interval must be positive and ordered.")
        self.minimum_idle_seconds = minimum_idle_seconds
        self.maximum_idle_seconds = maximum_idle_seconds
        self.stable_cue_seconds = stable_cue_seconds
        self._rng = rng or random.SystemRandom()
        self._last_activity = time.monotonic() if now is None else now
        self._next_idle_prompt = self._new_deadline(self._last_activity)
        self._candidate_signature: tuple[str, ...] = ()
        self._candidate_since = self._last_activity
        self._last_prompt_signature: tuple[str, ...] = ()
        self._lock = threading.Lock()

    def _new_deadline(self, now: float) -> float:
        return now + self._rng.uniform(self.minimum_idle_seconds, self.maximum_idle_seconds)

    @staticmethod
    def _signature(perception: PerceptionSnapshot) -> tuple[str, ...]:
        return tuple(
            sorted(
                (*perception.objects, *perception.action_cues, *perception.expression_cues)
            )
        )

    def mark_activity(self, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        with self._lock:
            self._last_activity = current
            self._next_idle_prompt = self._new_deadline(current)

    def next_reason(
        self, perception: PerceptionSnapshot, *, now: float | None = None
    ) -> str | None:
        current = time.monotonic() if now is None else now
        signature = self._signature(perception)
        with self._lock:
            if signature != self._candidate_signature:
                self._candidate_signature = signature
                self._candidate_since = current
            idle_for = current - self._last_activity
            if idle_for < self.minimum_idle_seconds:
                return None
            if (
                signature
                and signature != self._last_prompt_signature
                and current - self._candidate_since >= self.stable_cue_seconds
            ):
                return "visual_change"
            if current >= self._next_idle_prompt:
                return "quiet_pause"
            return None

    def mark_prompted(self, perception: PerceptionSnapshot, *, now: float | None = None) -> None:
        with self._lock:
            self._last_prompt_signature = self._signature(perception)
        self.mark_activity(now=now)


class LocalVoiceOutput:
    """Offline Windows speech output using an installed Korean SAPI voice."""

    def __init__(self) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise LocalInferenceError(
                'Offline speech output is missing. Install it with: python -m pip install -e ".[ai]"'
            ) from exc
        self._pyttsx3 = pyttsx3
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._engine: Any | None = None
        self._error: Exception | None = None
        self._speaking_since = 0.0
        self._playback_level = 0.0
        self._barge_in_frames = 0
        self._worker = threading.Thread(target=self._run, name="offline-korean-voice", daemon=True)
        self._worker.start()
        if not self._ready.wait(timeout=15):
            raise LocalInferenceError("Timed out initializing the offline Korean speech voice.")
        if self._error is not None:
            raise LocalInferenceError(f"Could not initialize offline Korean speech: {self._error}") from self._error

    def _run(self) -> None:
        try:
            engine = self._pyttsx3.init()
            korean_voices = [
                voice
                for voice in engine.getProperty("voices")
                if "ko" in f"{voice.id} {voice.name} {voice.languages}".casefold()
                or "korean" in f"{voice.id} {voice.name} {voice.languages}".casefold()
            ]
            if not korean_voices:
                raise LocalInferenceError(
                    "No Korean Windows speech voice is installed. Add a Korean voice in "
                    "Windows Settings > Time & language > Speech."
                )
            engine.setProperty("voice", korean_voices[0].id)
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            self._engine = engine
        except (AttributeError, ImportError, LocalInferenceError, OSError, RuntimeError) as exc:
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            text = self._queue.get()
            if text is None:
                return
            with self._lock:
                self._speaking_since = time.monotonic()
                self._playback_level = 0.0
                self._barge_in_frames = 0
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                logger.exception("Offline Korean speech output failed.")
            finally:
                with self._lock:
                    self._speaking_since = 0.0

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking_since > 0

    def observe_microphone_level(self, level: float) -> bool:
        """Detect sustained speech substantially louder than the initial speaker echo."""
        with self._lock:
            if self._speaking_since == 0:
                self._barge_in_frames = 0
                return False
            elapsed = time.monotonic() - self._speaking_since
            if elapsed < 0.35:
                self._playback_level = max(self._playback_level, level)
                return False
            threshold = min(0.2, max(0.045, self._playback_level * 2.0))
            self._barge_in_frames = self._barge_in_frames + 1 if level > threshold else 0
            return self._barge_in_frames >= 7

    def speak(self, text: str) -> None:
        if text:
            try:
                self._queue.put(text, timeout=0.5)
            except queue.Full as exc:
                raise LocalInferenceError("Offline speech output queue is full.") from exc

    def interrupt(self) -> None:
        with self._lock:
            engine = self._engine
        if engine is not None and self.is_speaking:
            engine.stop()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self.interrupt()
        try:
            self._queue.put(None, timeout=1)
        except queue.Full as exc:
            raise LocalInferenceError("Could not stop the offline speech output worker.") from exc
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            raise LocalInferenceError("Offline speech output did not stop within five seconds.")


@dataclass(frozen=True)
class _ChatJob:
    transcript: str | None = None
    proactive_reason: str | None = None


def _publish_event(event_bus: EventBus, event: Event) -> None:
    try:
        event_bus.publish(event, block=False)
    except EventBusFullError:
        logger.warning("Event bus is full; dropping local inference event %s.", event.event_type)


def run_ai_live(
    *,
    camera_index: int = 0,
    microphone_device: int | None = None,
    event_bus: EventBus,
    chat_model: str = CHAT_MODEL,
    vision_model: str = VISION_CHAT_MODEL,
    speech_model: str = WHISPER_MODEL,
    on_status: Callable[[str], None] = print,
    on_event: Callable[[Event], None] | None = None,
) -> None:
    """Run local-only camera, microphone transcription, perception, and chat."""
    try:
        import cv2
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise LocalInferenceError(
            'Live device support is missing. Install it with: python -m pip install -e ".[hardware,ai]"'
        ) from exc

    if camera_index < 0:
        raise ValueError("camera_index must be zero or greater.")
    if microphone_device is not None and microphone_device < 0:
        raise ValueError("microphone_device must be zero or greater.")

    on_status("Loading local speech and vision models. First launch may download several model files.")
    try:
        vision = LocalVisionAnalyzer()
        speech = LocalSpeechRecognizer(speech_model)
    except Exception as exc:
        if "vision" in locals():
            vision.close()
        if isinstance(exc, LocalInferenceError):
            raise
        raise LocalInferenceError(f"Could not initialize local inference models: {exc}") from exc
    chat = OllamaChat(chat_model, vision_model)
    try:
        chat.ensure_ready()
    except Exception:
        vision.close()
        raise

    try:
        capture = cv2.VideoCapture(camera_index)
    except Exception as exc:
        vision.close()
        raise LocalInferenceError(f"Could not initialize camera index {camera_index}: {exc}") from exc
    if not capture.isOpened():
        capture.release()
        vision.close()
        raise LocalInferenceError(
            f"Could not open camera index {camera_index}. Check camera permissions or use --camera-index."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        devices = sd.query_devices()
        input_devices = [
            index for index, device in enumerate(devices) if device["max_input_channels"] > 0
        ]
        if not input_devices:
            raise LocalInferenceError("No microphone input device is available.")
        if microphone_device is None:
            default_input = sd.default.device[0]
            selected_device = (
                default_input if default_input in input_devices else input_devices[0]
            )
        else:
            selected_device = microphone_device
            if selected_device not in input_devices:
                raise LocalInferenceError(f"Microphone input index {selected_device} is not available.")
        sample_rate = int(sd.query_devices(selected_device, "input")["default_samplerate"])
    except LocalInferenceError:
        capture.release()
        vision.close()
        raise
    except Exception as exc:
        capture.release()
        vision.close()
        raise LocalInferenceError(
            f"Could not initialize microphone device {microphone_device}: {exc}. "
            "Check Windows microphone privacy permissions and the selected input."
        ) from exc

    try:
        voice = LocalVoiceOutput()
    except Exception:
        capture.release()
        vision.close()
        raise

    audio_chunks: queue.Queue[Any] = queue.Queue(maxsize=24)
    transcript_jobs: queue.Queue[_ChatJob | None] = queue.Queue(maxsize=4)
    stopping = threading.Event()
    barge_in = threading.Event()
    state = _LiveState()
    scheduler = ProactiveConversationScheduler()
    latest_vision_event_at = 0.0

    def audio_callback(samples: Any, frames: int, timing: Any, status: Any) -> None:
        del frames, timing
        if status:
            logger.warning("Microphone stream status: %s", status)
        if voice.is_speaking:
            level = float(np.sqrt(np.mean(np.square(samples))))
            if voice.observe_microphone_level(level):
                barge_in.set()
            return
        try:
            audio_chunks.put_nowait(samples.copy())
        except queue.Full:
            logger.warning("Audio inference queue is full; dropping a microphone buffer.")

    def transcribe_worker() -> None:
        buffered: list[Any] = []
        sample_count = 0
        target_count = sample_rate * SPEECH_WINDOW_SECONDS
        while not stopping.is_set():
            try:
                chunk = audio_chunks.get(timeout=0.25)
            except queue.Empty:
                continue
            if chunk is None:
                break
            buffered.append(chunk)
            sample_count += len(chunk)
            if sample_count < target_count:
                continue
            try:
                transcript = speech.transcribe(np.concatenate(buffered), sample_rate)
                if transcript:
                    state.update_transcript(transcript)
                    scheduler.mark_activity()
                    _publish_event(
                        event_bus,
                        Event.create(
                            EventType.SPEECH_FINAL_TRANSCRIPT,
                            "laptop_microphone",
                            payload={"text": transcript, "local_inference": True},
                        ),
                    )
                    try:
                        transcript_jobs.put(_ChatJob(transcript=transcript), timeout=0.5)
                    except queue.Full:
                        logger.warning("Conversation queue is full; dropping a recognized utterance.")
            except Exception:
                logger.exception("Local speech transcription failed.")
            buffered = []
            sample_count = 0

    def chat_worker() -> None:
        while not stopping.is_set():
            try:
                job = transcript_jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            if job is None:
                break
            perception, _, _ = state.read()
            image_jpeg = state.read_image()
            try:
                if job.transcript is not None:
                    answer = chat.reply(job.transcript, perception, image_jpeg)
                else:
                    answer = chat.initiate(
                        perception,
                        None,
                        reason=job.proactive_reason or "quiet_pause",
                    )
                state.update_reply(answer)
                if answer:
                    voice.speak(answer)
                    on_status("Cinderella replied aloud in Korean.")
                scheduler.mark_prompted(perception)
            except Exception:
                logger.exception("Local conversational inference failed.")

    workers = [
        threading.Thread(target=transcribe_worker, name="speech-transcriber", daemon=True),
        threading.Thread(target=chat_worker, name="local-chat", daemon=True),
    ]
    stream_started = False

    try:
        try:
            stream = sd.InputStream(
                device=selected_device,
                channels=1,
                samplerate=sample_rate,
                dtype="float32",
                callback=audio_callback,
            )
            stream.start()
            stream_started = True
        except Exception as exc:
            raise LocalInferenceError(
                f"Could not start microphone device {selected_device}: {exc}. "
                "Check Windows microphone permissions and close other apps using the microphone."
            ) from exc
        for worker in workers:
            worker.start()
        on_status(
            f"Local AI active (camera {camera_index}, microphone {selected_device}, "
            f"Whisper {speech_model}, Ollama text {chat_model}, vision {vision_model}). "
            "Korean offline voice enabled; "
            "no media is saved or sent outside this computer. Speak naturally to interrupt; "
            "press Q or Esc to stop. Headphones reduce speaker echo."
        )
        last_timestamp_ms = 0
        last_vision_at = 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                raise LocalInferenceError("Camera stopped returning frames.")
            now = time.monotonic()
            if now - last_vision_at >= VISION_INTERVAL_SECONDS:
                timestamp_ms = max(last_timestamp_ms + 1, int(now * 1000))
                last_timestamp_ms = timestamp_ms
                snapshot = vision.analyze(frame, timestamp_ms)
                state.update_perception(snapshot)
                encoded_ok, encoded_frame = cv2.imencode(
                    ".jpg",
                    _visual_chat_frame(cv2, frame),
                    [cv2.IMWRITE_JPEG_QUALITY, 72],
                )
                if not encoded_ok:
                    raise LocalInferenceError("Could not encode a camera frame for local visual chat.")
                state.update_image(encoded_frame.tobytes())
                if now - latest_vision_event_at >= 1.0:
                    if snapshot.objects:
                        _publish_event(
                            event_bus,
                            Event.create(
                                EventType.VISION_OBJECTS_DETECTED,
                                "laptop_camera",
                                payload={
                                    "objects": list(snapshot.objects),
                                    "local_inference": True,
                                },
                                confidence=snapshot.confidence,
                            ),
                        )
                    if snapshot.action_cues and snapshot.action_cues != ("person posture detected",):
                        _publish_event(
                            event_bus,
                            Event.create(
                                EventType.VISION_ACTION_DETECTED,
                                "laptop_camera",
                                payload={
                                    "action_cues": list(snapshot.action_cues),
                                    "local_inference": True,
                                },
                                confidence=0.6,
                            ),
                        )
                    if snapshot.expression_cues:
                        _publish_event(
                            event_bus,
                            Event.create(
                                EventType.VISION_EXPRESSION_CUES_DETECTED,
                                "laptop_camera",
                                payload={
                                    "expression_cues": list(snapshot.expression_cues),
                                    "is_emotion_diagnosis": False,
                                    "local_inference": True,
                                },
                                confidence=0.5,
                            ),
                        )
                    latest_vision_event_at = now
                last_vision_at = now

            if barge_in.is_set():
                voice.interrupt()
                scheduler.mark_activity()
                barge_in.clear()
                on_status("Speech interruption detected; listening to you.")

            snapshot, _, _ = state.read()
            proactive_reason = scheduler.next_reason(snapshot, now=now)
            if proactive_reason is not None:
                try:
                    transcript_jobs.put_nowait(
                        _ChatJob(proactive_reason=proactive_reason)
                    )
                except queue.Full:
                    logger.warning("Conversation queue is full; postponing proactive speech.")
                else:
                    scheduler.mark_prompted(snapshot, now=now)

            snapshot, _, _ = state.read()
            lines = [
                f"Objects: {', '.join(snapshot.objects) or 'none detected'}",
                f"Pose cues: {', '.join(snapshot.action_cues) or 'none detected'}",
                f"Expression cues (not emotion): {', '.join(snapshot.expression_cues) or 'none detected'}",
            ]
            for index, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (12, 25 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 40),
                    1,
                )
            cv2.imshow("Cinderella - Local AI (Q/Esc to quit)", frame)
            for event in event_bus.drain():
                if on_event is not None:
                    on_event(event)
                else:
                    logger.debug("Consumed local inference event: %s", event.event_type)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        stopping.set()
        try:
            if "stream" in locals():
                try:
                    if stream_started:
                        stream.stop()
                finally:
                    stream.close()
        finally:
            try:
                for work_queue in (audio_chunks, transcript_jobs):
                    try:
                        work_queue.put_nowait(None)
                    except queue.Full:
                        logger.warning("Worker queue is full while stopping; worker will exit after timeout.")
                for worker in workers:
                    if worker.is_alive():
                        worker.join(timeout=5)
                        if worker.is_alive():
                            logger.warning(
                                "Worker %s is still finishing inference during shutdown.", worker.name
                            )
            finally:
                try:
                    capture.release()
                finally:
                    try:
                        voice.close()
                    finally:
                        try:
                            cv2.destroyAllWindows()
                        finally:
                            vision.close()


def _visual_chat_frame(cv2: Any, frame: Any) -> Any:
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side <= VISUAL_CHAT_MAX_SIDE:
        return frame
    scale = VISUAL_CHAT_MAX_SIDE / longest_side
    resized = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized
