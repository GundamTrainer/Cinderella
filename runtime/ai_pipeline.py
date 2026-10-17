from __future__ import annotations

import base64
import ipaddress
import json
import logging
import queue
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ai.observation import Observation, ObservationType, ObservationValidity
from ai.scene import (
    InterpretationHypothesis,
    SceneObservation,
    WorldState,
    WorldStateTracker,
    tentative_affect_hypotheses,
)
from ai.vision import (
    DetectionFrame,
    DetectionStabilizer,
    DetectionState,
    FaceFeatureFrame,
    FaceFeatureStabilizer,
    convert_detector_results,
)
from runtime.audio_stream import AudioInputBuffer
from runtime.backend_factory import create_live_backends

from .event_bus import Event, EventBus, EventBusFullError, EventType

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:3b"
VISION_CHAT_MODEL = "qwen2.5vl:3b"
WHISPER_MODEL = "small"
SPEECH_WINDOW_SECONDS = 5
VISION_INTERVAL_SECONDS = 0.4
VISUAL_CHAT_MAX_SIDE = 448
PROACTIVE_MIN_IDLE_SECONDS = 60
PROACTIVE_MAX_IDLE_SECONDS = 140
_AUDIO_DIAGNOSTIC_EVENT_TYPES = {
    "audio.frame_received": EventType.AUDIO_FRAME_RECEIVED,
    "audio.frame_dropped": EventType.AUDIO_FRAME_DROPPED,
    "audio.sequence_gap": EventType.AUDIO_SEQUENCE_GAP,
    "audio.input_overflow": EventType.AUDIO_INPUT_OVERFLOW,
    "audio.queue_pressure": EventType.AUDIO_QUEUE_PRESSURE,
    "audio.segment_started": EventType.AUDIO_SEGMENT_STARTED,
    "audio.segment_accepted": EventType.AUDIO_SEGMENT_ACCEPTED,
    "audio.segment_rejected": EventType.AUDIO_SEGMENT_REJECTED,
    "audio.segment_ended": EventType.AUDIO_SEGMENT_ENDED,
}
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
    detections: DetectionFrame | None = None
    face_features: FaceFeatureFrame | None = None
    observation: SceneObservation | None = None
    world_state: WorldState | None = None
    tentative_interpretations: tuple[InterpretationHypothesis, ...] = ()

    def prompt_context(self) -> str:
        scene = (
            "; ".join(
                f"{item.label} at {item.region}, {item.motion}" for item in self.world_state.objects
            )
            if self.world_state and self.world_state.objects
            else "no stable objects"
        )
        uncertain = (
            ", ".join(self.world_state.uncertain_labels)
            if self.world_state and self.world_state.uncertain_labels
            else "none"
        )
        return (
            "사용자가 물어본 경우에만 참고할 수 있는 시각 관찰 단서(오류 가능): "
            f"물체={', '.join(self.objects) if self.objects else '없음/미검출'}, "
            f"시간적 장면 상태={scene}, 미확정 후보={uncertain}, "
            f"관찰 가능한 자세 단서={', '.join(self.action_cues) if self.action_cues else '없음/미검출'}."
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
        "보이니",
        "보이냐",
        "뭘 보고",
        "뭘 들고",
        "뭐가 있어",
        "뭐가 보",
        "무엇이 보",
        "무엇이 있",
        "이게 뭐",
        "이건 뭐",
        "이거 뭐",
        "저게 뭐",
        "저건 뭐",
        "저거 뭐",
        "뭔지",
        "무엇인지",
        "물체",
        "사물",
        "장면",
        "색깔",
        "색이",
        "사진",
        "이미지",
        "카메라",
        "화면",
        "카메라 앞에",
        "화면 앞에",
        "앞에 뭐",
        "앞에 있는",
        "앞에 놓인",
        "옆에 뭐",
        "옆에 있는",
        "표정",
        "사람 있어",
        "사람이 있어",
        "몇 명",
        "무슨 물건",
        "어떤 물건",
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
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
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

    def __init__(self, *, box_clamp_tolerance_pixels: float = 2.0) -> None:
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
        self._detection_stabilizer = DetectionStabilizer(
            minimum_confidence=0.35,
            class_thresholds={0: 0.45},
            supported_class_ids=set(range(80)),
        )
        self._face_stabilizer = FaceFeatureStabilizer(
            smoothing=0.45,
            minimum_stable_duration=0.25,
            missing_tolerance_frames=2,
        )
        self._box_clamp_tolerance_pixels = box_clamp_tolerance_pixels
        self._frame_id = 0
        self._diagnostic_totals: Counter[str] = Counter()
        self._diagnostic_log_milestones: dict[str, int] = {}
        self._world_state_tracker = WorldStateTracker()
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
            try:
                self.close()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve both failures
                raise ExceptionGroup(
                    "Could not initialize MediaPipe models and cleanup also failed.",
                    [exc, cleanup_error],
                ) from exc
            raise LocalInferenceError(
                f"Could not initialize MediaPipe vision models: {exc}"
            ) from exc

    def analyze(self, frame: Any, timestamp_ms: int) -> PerceptionSnapshot:
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        object_result = self._object_detector.detect_for_video(image, timestamp_ms)
        pose_result = self._pose_landmarker.detect_for_video(image, timestamp_ms)
        face_result = self._face_landmarker.detect_for_video(image, timestamp_ms)

        height, width = frame.shape[:2]
        observed_at = timestamp_ms / 1000.0
        self._frame_id += 1
        conversion = convert_detector_results(
            object_result.detections,
            frame_id=self._frame_id,
            timestamp=observed_at,
            image_width=width,
            image_height=height,
            source_backend="mediapipe",
            minimum_confidence=self._detection_stabilizer.minimum_confidence,
            supported_class_ids=self._detection_stabilizer.supported_class_ids,
            clamp_tolerance_pixels=self._box_clamp_tolerance_pixels,
        )
        self._diagnostic_totals.update(dict(conversion.diagnostics))
        report_reasons: list[str] = []
        for reason, _ in conversion.diagnostics:
            total = self._diagnostic_totals[reason]
            milestone = self._diagnostic_log_milestones.get(reason, 1)
            if total >= milestone:
                report_reasons.append(reason)
                while milestone <= total:
                    milestone *= 10
                self._diagnostic_log_milestones[reason] = milestone
        if report_reasons:
            logger.warning(
                "Object detector diagnostics reached reporting milestones.",
                extra={
                    "frame_id": self._frame_id,
                    "diagnostic_reasons": report_reasons,
                    "cumulative_diagnostics": {
                        reason: self._diagnostic_totals[reason] for reason in report_reasons
                    },
                },
            )
        detection_frame = self._detection_stabilizer.update(
            conversion.detections,
            observed_at=observed_at,
            frame_id=self._frame_id,
            source_backend="mediapipe",
            unverified_detections=conversion.unverified,
            diagnostics=conversion.diagnostics,
        )
        objects = tuple(
            detection.label
            for detection in detection_frame.stable_detections
            if detection.state == DetectionState.STABLE
        )
        confidences = [
            detection.confidence
            for detection in detection_frame.stable_detections
            if detection.state == DetectionState.STABLE
        ]

        action_cues: tuple[str, ...] = ()
        if pose_result.pose_landmarks:
            action_cues = raised_arm_cues(pose_result.pose_landmarks[0])

        expression_cues: tuple[str, ...] = ()
        face_count = len(face_result.face_landmarks)
        face_measurements: dict[str, float] | None = None
        if face_count == 1 and face_result.face_blendshapes:
            self._smoothed_blendshapes = _smooth_blendshapes(
                self._smoothed_blendshapes,
                face_result.face_blendshapes[0],
            )
            scores = self._smoothed_blendshapes
            face_measurements = {
                "eye_openness_left": 1.0 - scores.get("eyeBlinkLeft", 0.0),
                "eye_openness_right": 1.0 - scores.get("eyeBlinkRight", 0.0),
                "mouth_openness": scores.get("jawOpen", 0.0),
                "mouth_corner_displacement": (
                    scores.get("mouthSmileLeft", 0.0) + scores.get("mouthSmileRight", 0.0)
                )
                / 2,
                "brow_displacement": (
                    scores.get("browOuterUpLeft", 0.0)
                    + scores.get("browOuterUpRight", 0.0)
                    + scores.get("browDownLeft", 0.0)
                    + scores.get("browDownRight", 0.0)
                )
                / 4,
            }
        elif face_count > 1:
            self._smoothed_blendshapes = {}
        face_features = self._face_stabilizer.update(
            face_measurements,
            face_count=face_count,
            observed_at=observed_at,
        )
        if face_features.stable:
            expression_cues = facial_expression_cues(
                [
                    SimpleNamespace(category_name=name, score=score)
                    for name, score in self._smoothed_blendshapes.items()
                ]
            )

        scene_observations = [
            Observation(
                observation_type=ObservationType.OBJECT_CANDIDATE,
                source="camera",
                source_backend=detection.source_backend,
                frame_id=self._frame_id,
                monotonic_time=observed_at,
                confidence=detection.confidence,
                validity=ObservationValidity.VALID,
                payload={
                    "class_id": detection.class_id,
                    "label": detection.label,
                    "bounding_box": {
                        "x_min": detection.box.x_min,
                        "y_min": detection.box.y_min,
                        "x_max": detection.box.x_max,
                        "y_max": detection.box.y_max,
                    },
                },
                tracking_hint=(
                    str(detection.track_id) if detection.track_id is not None else None
                ),
                synthetic=detection.synthetic,
            )
            for detection in detection_frame.stable_detections
        ]
        validity_by_status = {
            DetectionState.LOW_CONFIDENCE: ObservationValidity.LOW_CONFIDENCE,
            DetectionState.UNSTABLE: ObservationValidity.UNSTABLE,
            DetectionState.INVALID_BOX: ObservationValidity.INVALID,
            DetectionState.REJECTED: ObservationValidity.REJECTED,
        }
        scene_observations.extend(
            Observation(
                observation_type=ObservationType.OBJECT_CANDIDATE,
                source="camera",
                source_backend=item.source_backend,
                frame_id=item.frame_id,
                monotonic_time=item.timestamp,
                confidence=item.confidence if item.confidence is not None else 0.0,
                validity=validity_by_status.get(item.status, ObservationValidity.UNKNOWN),
                payload={
                    "class_id": item.normalized_class_id,
                    "label": item.label or "unknown",
                    "bounding_box": (
                        {
                            "x_min": item.bounding_box.x_min,
                            "y_min": item.bounding_box.y_min,
                            "x_max": item.bounding_box.x_max,
                            "y_max": item.bounding_box.y_max,
                        }
                        if item.bounding_box is not None
                        else None
                    ),
                },
                tracking_hint=str(item.tracking_id) if item.tracking_id is not None else None,
                errors=item.validation_errors,
                synthetic=item.synthetic,
            )
            for item in detection_frame.unverified_detections
        )
        observation = SceneObservation(
            frame_id=self._frame_id,
            observed_at=observed_at,
            detections=detection_frame,
            action_cues=action_cues,
            face_features=face_features,
            observations=tuple(scene_observations),
        )
        world_state = self._world_state_tracker.update(observation)
        tentative_interpretations = tentative_affect_hypotheses(
            expression_cues,
            observed_at=observed_at,
        )
        return PerceptionSnapshot(
            objects=tuple(objects),
            action_cues=action_cues,
            expression_cues=expression_cues,
            confidence=max(confidences, default=0.0),
            face_detected=face_count == 1,
            detections=detection_frame,
            face_features=face_features,
            observation=observation,
            world_state=world_state,
            tentative_interpretations=tentative_interpretations,
        )

    def close(self) -> None:
        errors: list[Exception] = []
        for name in ("_face_landmarker", "_pose_landmarker", "_object_detector"):
            task = getattr(self, name)
            if task is not None:
                try:
                    task.close()
                except Exception as exc:  # noqa: BLE001 - continue closing remaining tasks
                    errors.append(exc)
                finally:
                    setattr(self, name, None)
        self._detection_stabilizer.reset()
        self._face_stabilizer.reset()
        self._world_state_tracker.reset()
        if errors:
            raise ExceptionGroup("One or more MediaPipe tasks failed to close.", errors)


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
        self.last_rejection_reason: str | None = None

    def transcribe(self, samples: Any, sample_rate: int) -> str:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero.")
        if len(samples) == 0:
            self.last_rejection_reason = "speech_not_detected"
            return ""
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        target_rate = 16_000
        if sample_rate != target_rate:
            output_size = round(len(audio) * target_rate / sample_rate)
            if output_size == 0:
                self.last_rejection_reason = "speech_not_detected"
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
        transcript = "".join(segment.text for segment in segments).strip()
        self.last_rejection_reason = None if transcript else "speech_not_detected"
        return transcript


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
        parsed_url = urllib.parse.urlsplit(base_url)
        hostname = parsed_url.hostname
        is_loopback_ip = False
        if hostname is not None:
            try:
                is_loopback_ip = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback_ip = hostname.casefold() == "localhost"
        if (
            parsed_url.scheme != "http"
            or not is_loopback_ip
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("Ollama must use an unauthenticated HTTP loopback URL.")
        self.base_url = base_url.rstrip("/")
        self._history: list[dict[str, Any]] = []
        self._system = (
            "너는 사용자의 말을 잘 듣고 대화를 주고받는 AI야. "
            "사용자가 쓴 언어와 대화의 맥락에 맞춰 자연스럽고 직접적으로 답해. "
            "정해진 인사말이나 답변 틀을 반복하지 말고, 직전 말의 구체적인 내용에 반응해. "
            "간단한 말에는 간결하게 답하되, 복잡하거나 감정적인 주제는 필요한 만큼 충분히 다뤄. "
            "대화를 이어갈 질문은 실제로 도움이 될 때만 하나 덧붙이고, 사용자가 원치 않으면 "
            "질문을 계속하지 마. 사용자의 말을 그대로 길게 되풀이하거나 매번 요약하지 마. "
            "정해지지 않은 캐릭터 성격을 임의로 만들거나 사람인 척하지 마. "
            "얼굴이나 목소리만으로 감정·의도·신원·건강을 단정하지 마. "
            "감정 관련 단서는 검증되지 않은 낮은 확신의 가설일 뿐이야. "
            "감정을 사실처럼 말하지 말고, 꼭 필요하면 사용자가 직접 확인할 수 있도록 조심스럽게 질문해. "
            "카메라 내용은 사용자가 화면에 대해 물었을 때만 언급해. "
            "카메라 질문에는 첨부된 이미지를 직접 확인하고, 자동 검출이 놓쳤을 수 있어. "
            "얼굴로 사람의 신원을 알아내거나 추측하지 마. 확실하지 않은 물체는 지어내지 말고 "
            "불확실하다고 말해."
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
        visual_question = _asks_about_visual_context(transcript)
        use_vision_model = image_jpeg is not None and visual_question
        if visual_question and image_jpeg is None:
            content = f"{perception.prompt_context()}\n사용자 발화: {transcript}"
        else:
            content = transcript
        images = (
            [base64.b64encode(image_jpeg).decode("ascii")]
            if image_jpeg is not None and use_vision_model
            else []
        )
        user_message: dict[str, Any] = {"role": "user", "content": content}
        request_message = dict(user_message)
        if images:
            request_message["images"] = images
            request_message["content"] = (
                f"사용자의 질문: {transcript}\n"
                "첨부된 현재 이미지를 직접 살펴 질문에 한국어로 답해. "
                "자동 검출이 놓친 물체가 있을 수 있으니 화면에서 확인되는 것만 설명하고, "
                "얼굴로 사람의 신원을 추측하지 마. 확실하지 않은 부분은 불확실하다고 말해."
            )
        pending_history = [*self._history, user_message][-8:]
        history_messages = [
            {"role": "system", "content": self._system},
            *pending_history[:-1],
            request_message,
        ]
        result = self._post(
            "/api/chat",
            {
                "model": self.vision_model if use_vision_model else self.model,
                "messages": history_messages,
                "stream": False,
                "options": {
                    "temperature": 0.1 if use_vision_model else 0.72,
                    "num_ctx": 4096,
                    "num_predict": 100 if use_vision_model else 256,
                    "top_p": 0.9,
                    "repeat_penalty": 1.12,
                },
            },
        )
        answer = result.get("message", {}).get("content", "").strip()
        if not answer:
            raise LocalChatError("Local Ollama returned an empty response.")
        self._history = [*pending_history, {"role": "assistant", "content": answer}][-8:]
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
        self._history.append({"role": "assistant", "content": answer})
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
        return tuple(sorted((*perception.objects, *perception.action_cues)))

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
            raise LocalInferenceError(
                f"Could not initialize offline Korean speech: {self._error}"
            ) from self._error

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
    source: str = "microphone"


def _queue_text_chat_message(
    text: str,
    jobs: queue.Queue[_ChatJob | None],
    scheduler: ProactiveConversationScheduler,
) -> bool:
    message = text.strip()
    if not message:
        return False
    try:
        jobs.put_nowait(_ChatJob(transcript=message, source="keyboard"))
    except queue.Full:
        logger.warning("Conversation queue is full; could not enqueue typed chat.")
        return False
    scheduler.mark_activity()
    return True


def _publish_event(event_bus: EventBus, event: Event, timeout: float = 0.25) -> None:
    try:
        event_bus.publish(event, block=True, timeout=timeout)
    except EventBusFullError:
        logger.warning("Event bus is full; dropping local inference event %s.", event.event_type)


def _audio_diagnostic_event(diagnostic: str, count: int) -> Event:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Audio diagnostic count must be a positive integer.")
    event_name, separator, reason = diagnostic.partition(":")
    event_type = _AUDIO_DIAGNOSTIC_EVENT_TYPES.get(event_name)
    if event_type is None:
        raise ValueError(f"Unsupported audio diagnostic: {diagnostic}")
    return Event.create(
        event_type,
        "laptop_microphone",
        payload={
            "count": count,
            **({"reason": reason} if separator else {}),
            "raw_audio_logged": False,
        },
    )


def run_ai_live(
    *,
    camera_index: int = 0,
    microphone_device: int | None = None,
    event_bus: EventBus,
    chat_model: str = CHAT_MODEL,
    vision_model: str = VISION_CHAT_MODEL,
    speech_model: str = WHISPER_MODEL,
    box_clamp_tolerance_pixels: float = 2.0,
    audio_input_queue_capacity: int = 24,
    event_publish_timeout: float = 0.25,
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
    if (
        isinstance(event_publish_timeout, bool)
        or not isinstance(event_publish_timeout, (int, float))
        or not isfinite(event_publish_timeout)
        or event_publish_timeout <= 0
    ):
        raise ValueError("event_publish_timeout must be finite and greater than zero.")
    if (
        isinstance(audio_input_queue_capacity, bool)
        or not isinstance(audio_input_queue_capacity, int)
        or audio_input_queue_capacity <= 0
    ):
        raise ValueError("audio_input_queue_capacity must be positive.")

    on_status(
        "Loading local speech and vision models. First launch may download several model files."
    )
    backends = create_live_backends(
        speech_model=speech_model,
        chat_model=chat_model,
        vision_model=vision_model,
        box_clamp_tolerance_pixels=box_clamp_tolerance_pixels,
    )
    vision = backends.vision
    speech = backends.speech
    chat = backends.chat
    voice = backends.voice

    def close_backends() -> None:
        errors: list[Exception] = []
        for backend in (voice, chat, speech, vision):
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - continue closing remaining backends
                errors.append(exc)
        if errors:
            raise ExceptionGroup("One or more AI backends failed to close.", errors)

    def cleanup_after_error(error: Exception) -> None:
        try:
            close_backends()
        except Exception as cleanup_error:  # noqa: BLE001 - preserve both failures
            raise ExceptionGroup(
                "AI initialization failed and backend cleanup also failed.",
                [error, cleanup_error],
            ) from error

    try:
        vision.initialize()
        speech.initialize()
        chat.initialize()
    except Exception as exc:
        cleanup_after_error(exc)
        raise

    try:
        capture = cv2.VideoCapture(camera_index)
    except Exception as exc:
        cleanup_after_error(exc)
        raise LocalInferenceError(
            f"Could not initialize camera index {camera_index}: {exc}"
        ) from exc
    if not capture.isOpened():
        capture.release()
        error = LocalInferenceError(
            f"Could not open camera index {camera_index}. Check camera permissions or use --camera-index."
        )
        cleanup_after_error(error)
        raise error
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
            selected_device = default_input if default_input in input_devices else input_devices[0]
        else:
            selected_device = microphone_device
            if selected_device not in input_devices:
                raise LocalInferenceError(
                    f"Microphone input index {selected_device} is not available."
                )
        sample_rate = int(sd.query_devices(selected_device, "input")["default_samplerate"])
        if sample_rate <= 0:
            raise LocalInferenceError("Microphone sample rate must be greater than zero.")
    except LocalInferenceError as exc:
        capture.release()
        cleanup_after_error(exc)
        raise
    except Exception as exc:
        capture.release()
        cleanup_after_error(exc)
        raise LocalInferenceError(
            f"Could not initialize microphone device {microphone_device}: {exc}. "
            "Check Windows microphone privacy permissions and the selected input."
        ) from exc

    try:
        voice.initialize()
    except Exception as exc:
        capture.release()
        cleanup_after_error(exc)
        raise

    audio_chunks = AudioInputBuffer(capacity=audio_input_queue_capacity)
    transcript_jobs: queue.Queue[_ChatJob | None] = queue.Queue(maxsize=4)
    stopping = threading.Event()
    barge_in = threading.Event()
    state = _LiveState()
    scheduler = ProactiveConversationScheduler()
    latest_vision_event_at = 0.0

    def publish_audio_diagnostics() -> None:
        for diagnostic, count in audio_chunks.drain_diagnostics().items():
            _publish_event(
                event_bus,
                _audio_diagnostic_event(diagnostic, count),
                timeout=event_publish_timeout,
            )

    def audio_callback(samples: Any, frames: int, timing: Any, status: Any) -> None:
        del frames, timing
        if voice.is_speaking:
            level = float(np.sqrt(np.mean(np.square(samples))))
            if voice.observe_microphone_level(level):
                barge_in.set()
            audio_chunks.discard(
                frame_count=len(samples),
                status=status,
                reason="tts_echo_guard",
            )
            return
        audio_chunks.submit(samples, frame_count=len(samples), status=status)

    def transcribe_worker() -> None:
        buffered: list[Any] = []
        sample_count = 0
        target_count = sample_rate * SPEECH_WINDOW_SECONDS
        last_sequence_number: int | None = None
        segment_started = False
        while not stopping.is_set():
            try:
                chunk = audio_chunks.get(timeout=0.25)
            except queue.Empty:
                continue
            sequence_gap = (
                chunk.sequence_number - last_sequence_number - 1
                if last_sequence_number is not None
                else 0
            )
            if sequence_gap > 0:
                audio_chunks.record_sequence_gap(sequence_gap)
            if chunk.input_overflow or sequence_gap != 0 or (
                last_sequence_number is not None
                and chunk.sequence_number != last_sequence_number + 1
            ):
                if buffered:
                    audio_chunks.record_segment(accepted=False, reason="queue_gap")
                if segment_started:
                    audio_chunks.record_segment_ended()
                buffered = []
                sample_count = 0
                segment_started = False
            last_sequence_number = chunk.sequence_number
            if not segment_started:
                audio_chunks.record_segment_started()
                segment_started = True
            buffered.append(chunk.samples)
            sample_count += chunk.frame_count
            if sample_count < target_count:
                continue
            try:
                transcript = speech.infer(np.concatenate(buffered), sample_rate)
                if transcript:
                    audio_chunks.record_segment(accepted=True)
                    state.update_transcript(transcript)
                    scheduler.mark_activity()
                    _publish_event(
                        event_bus,
                        Event.create(
                            EventType.SPEECH_FINAL_TRANSCRIPT,
                            "laptop_microphone",
                            payload={"text": transcript, "local_inference": True},
                        ),
                        timeout=event_publish_timeout,
                    )
                    try:
                        transcript_jobs.put(_ChatJob(transcript=transcript), timeout=0.5)
                    except queue.Full:
                        logger.warning(
                            "Conversation queue is full; dropping a recognized utterance."
                        )
                else:
                    audio_chunks.record_segment(
                        accepted=False,
                        reason=getattr(speech, "last_rejection_reason", None) or "unknown",
                    )
            except Exception:
                logger.exception("Local speech transcription failed.")
                audio_chunks.record_segment(accepted=False, reason="inference_error")
            buffered = []
            sample_count = 0
            audio_chunks.record_segment_ended()
            segment_started = False
        if segment_started:
            audio_chunks.record_segment(accepted=False, reason="shutdown")
            audio_chunks.record_segment_ended()

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
                        reason=job.proactive_reason or "quiet_pause",
                    )
                state.update_reply(answer)
                if answer:
                    voice.speak(answer)
                    speaker = "typed chat" if job.source == "keyboard" else "microphone"
                    print(f"\nYou ({speaker}): {job.transcript or '(proactive prompt)'}")
                    print(f"Cinderella: {answer}\n")
                    on_status("Cinderella replied aloud in Korean.")
                scheduler.mark_prompted(perception)
            except Exception as exc:
                logger.exception("Local conversational inference failed.")
                on_status(f"Local reply failed: {exc}")

    def text_input_worker() -> None:
        on_status(
            "Text chat is ready: type a message and press Enter. "
            "Type /text-off to stop keyboard chat; microphone chat remains active."
        )
        while not stopping.is_set():
            try:
                message = input("You (text): ").strip()
            except EOFError:
                on_status("Console input ended; microphone chat remains active.")
                return
            except OSError:
                logger.exception("Could not read typed chat input.")
                return
            if message.casefold() == "/text-off":
                on_status("Keyboard chat stopped; microphone chat remains active.")
                return
            _queue_text_chat_message(message, transcript_jobs, scheduler)

    workers = [
        threading.Thread(target=transcribe_worker, name="speech-transcriber", daemon=True),
        threading.Thread(target=chat_worker, name="local-chat", daemon=True),
        threading.Thread(target=text_input_worker, name="typed-chat-input", daemon=True),
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
            "Korean offline voice and terminal text chat enabled; "
            "no media is saved or sent outside this computer. Speak naturally to interrupt; "
            "type in the terminal to chat; press Q or Esc to stop. Headphones reduce speaker echo."
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
                snapshot = vision.infer(frame, now)
                state.update_perception(snapshot)
                encoded_ok, encoded_frame = cv2.imencode(
                    ".jpg",
                    _visual_chat_frame(cv2, frame),
                    [cv2.IMWRITE_JPEG_QUALITY, 72],
                )
                if not encoded_ok:
                    raise LocalInferenceError(
                        "Could not encode a camera frame for local visual chat."
                    )
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
                                    "detections": [
                                        {
                                            "class_id": detection.class_id,
                                            "label": detection.label,
                                            "confidence": detection.confidence,
                                            "track_id": detection.track_id,
                                            "bbox": {
                                                "x_min": detection.box.x_min,
                                                "y_min": detection.box.y_min,
                                                "x_max": detection.box.x_max,
                                                "y_max": detection.box.y_max,
                                            },
                                        }
                                        for detection in (
                                            snapshot.detections.stable_detections
                                            if snapshot.detections
                                            else ()
                                        )
                                        if detection.state == DetectionState.STABLE
                                    ],
                                    "local_inference": True,
                                },
                                confidence=snapshot.confidence,
                            ),
                            timeout=event_publish_timeout,
                        )
                    if snapshot.action_cues and snapshot.action_cues != (
                        "person posture detected",
                    ):
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
                            timeout=event_publish_timeout,
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
                            timeout=event_publish_timeout,
                        )
                    latest_vision_event_at = now
                last_vision_at = now

            publish_audio_diagnostics()

            if barge_in.is_set():
                voice.interrupt()
                scheduler.mark_activity()
                barge_in.clear()
                on_status("Speech interruption detected; listening to you.")

            snapshot, _, _ = state.read()
            proactive_reason = scheduler.next_reason(snapshot, now=now)
            if proactive_reason is not None:
                try:
                    transcript_jobs.put_nowait(_ChatJob(proactive_reason=proactive_reason))
                except queue.Full:
                    logger.warning("Conversation queue is full; postponing proactive speech.")
                else:
                    scheduler.mark_prompted(snapshot, now=now)

            snapshot, _, _ = state.read()
            if snapshot.detections is not None:
                height, width = frame.shape[:2]
                for detection in snapshot.detections.stable_detections:
                    if detection.state != DetectionState.STABLE:
                        continue
                    left = round(detection.box.x_min * width)
                    top = round(detection.box.y_min * height)
                    right = round(detection.box.x_max * width)
                    bottom = round(detection.box.y_max * height)
                    cv2.rectangle(frame, (left, top), (right, bottom), (40, 220, 40), 2)
                    cv2.putText(
                        frame,
                        f"{detection.label} {detection.confidence:.2f}",
                        (left, max(18, top - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (40, 220, 40),
                        1,
                    )
            lines = [
                f"Objects: {', '.join(snapshot.objects) or 'none detected'}",
                f"Pose cues: {', '.join(snapshot.action_cues) or 'none detected'}",
                f"Expression cues (not emotion): {', '.join(snapshot.expression_cues) or 'none detected'}",
                (
                    "Tentative affect: "
                    + (
                        ", ".join(item.hypothesis for item in snapshot.tentative_interpretations)
                        if snapshot.tentative_interpretations
                        else "none"
                    )
                    + " (unverified)"
                ),
            ]
            for index, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (12, 25 + index * 24),
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
                for work_queue in (transcript_jobs,):
                    try:
                        work_queue.put_nowait(None)
                    except queue.Full:
                        logger.warning(
                            "Worker queue is full while stopping; worker will exit after timeout."
                        )
                for worker in workers:
                    if worker.is_alive():
                        worker.join(timeout=5)
                        if worker.is_alive():
                            logger.warning(
                                "Worker %s is still finishing inference during shutdown.",
                                worker.name,
                            )
            finally:
                try:
                    publish_audio_diagnostics()
                finally:
                    try:
                        capture.release()
                    finally:
                        try:
                            cv2.destroyAllWindows()
                        finally:
                            close_backends()


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
