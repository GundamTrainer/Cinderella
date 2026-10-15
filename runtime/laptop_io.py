from __future__ import annotations

import importlib
import logging
import math
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class LaptopDeviceError(RuntimeError):
    """Raised when an optional device library or requested device is unavailable."""


def _load_device_libraries() -> tuple[Any, Any]:
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise LaptopDeviceError(
            "Camera support is not installed. Install the optional hardware dependencies "
            'with: python -m pip install -e ".[hardware]"'
        ) from exc
    try:
        sounddevice = importlib.import_module("sounddevice")
    except ImportError as exc:
        raise LaptopDeviceError(
            "Microphone support is not installed. Install the optional hardware dependencies "
            'with: python -m pip install -e ".[hardware]"'
        ) from exc
    return cv2, sounddevice


def _audio_rms(samples: Any) -> float:
    """Return the normalized RMS level for a sounddevice callback buffer."""
    if getattr(samples, "size", 0) == 0:
        return 0.0
    return math.sqrt(float((samples**2).mean()))


def list_audio_inputs() -> list[dict[str, Any]]:
    """Return available microphone device metadata without starting a stream."""
    try:
        sounddevice = importlib.import_module("sounddevice")
    except ImportError as exc:
        raise LaptopDeviceError(
            "Microphone support is not installed. Install the optional hardware dependencies "
            'with: python -m pip install -e ".[hardware]"'
        ) from exc
    try:
        devices = sounddevice.query_devices()
    except Exception as exc:
        raise LaptopDeviceError(f"Could not query microphone devices: {exc}") from exc
    return [
        {"index": index, "name": device["name"], "channels": device["max_input_channels"]}
        for index, device in enumerate(devices)
        if device["max_input_channels"] > 0
    ]


def list_cameras(*, max_index: int = 5) -> list[int]:
    """Probe common camera indexes. Opening a device may briefly activate its indicator."""
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise LaptopDeviceError(
            "Camera support is not installed. Install the optional hardware dependencies "
            'with: python -m pip install -e ".[hardware]"'
        ) from exc
    available: list[int] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                available.append(index)
        finally:
            capture.release()
    return available


def run_live_preview(
    *,
    camera_index: int = 0,
    microphone_device: int | None = None,
    on_status: Callable[[str], None] = print,
) -> None:
    """Show live camera frames and microphone level without saving audio or video."""
    if camera_index < 0:
        raise ValueError("camera_index must be zero or greater.")
    if microphone_device is not None and microphone_device < 0:
        raise ValueError("microphone_device must be zero or greater.")
    cv2, sounddevice = _load_device_libraries()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise LaptopDeviceError(
            f"Could not open camera index {camera_index}. Check Windows camera privacy "
            "permissions, close other camera apps, or try --camera-index with another index."
        )

    level_lock = threading.Lock()
    latest_level = 0.0

    def audio_callback(
        samples: Any,
        frames: int,
        timing: Any,
        status: Any,
    ) -> None:
        nonlocal latest_level
        del frames, timing
        if status:
            logger.warning("Microphone stream status: %s", status)
        level = _audio_rms(samples)
        with level_lock:
            latest_level = level

    try:
        try:
            devices = sounddevice.query_devices()
            input_devices = [
                (index, device)
                for index, device in enumerate(devices)
                if device["max_input_channels"] > 0
            ]
            if not input_devices:
                raise LaptopDeviceError(
                    "No microphone input device was found. Connect or enable a microphone "
                    "in Windows Sound settings, then try again."
                )
            if microphone_device is None:
                default_input = sounddevice.default.device[0]
                selected_device = (
                    default_input
                    if default_input is not None
                    and default_input >= 0
                    and any(index == default_input for index, _ in input_devices)
                    else input_devices[0][0]
                )
            else:
                selected_device = microphone_device
                if not any(index == selected_device for index, _ in input_devices):
                    raise LaptopDeviceError(
                        f"Microphone device index {selected_device} is unavailable or has no "
                        "input channels. Run `python main.py --devices` to list available inputs."
                    )
            device_info = sounddevice.query_devices(selected_device, "input")
            sample_rate = int(device_info["default_samplerate"])
            stream = sounddevice.InputStream(
                device=selected_device,
                channels=1,
                samplerate=sample_rate,
                callback=audio_callback,
            )
        except LaptopDeviceError:
            raise
        except Exception as exc:
            raise LaptopDeviceError(
                f"Could not initialize microphone device: {exc}. Check Windows microphone "
                "privacy permissions and the selected input device."
            ) from exc

        on_status(
            f"Live preview started (camera {camera_index}, microphone {selected_device}). "
            "No audio or video is recorded. Press Q or Esc to stop."
        )
        stream.start()
        while True:
            ok, frame = capture.read()
            if not ok:
                raise LaptopDeviceError("Camera stopped returning frames.")
            with level_lock:
                level = latest_level
            cv2.putText(
                frame,
                f"Mic level: {level:.3f} | no recording",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 220, 40),
                2,
            )
            cv2.imshow("Cinderella - Live Camera and Microphone", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        try:
            if "stream" in locals():
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            capture.release()
            cv2.destroyAllWindows()
