from __future__ import annotations

import argparse
import logging

from runtime.ai_pipeline import LocalInferenceError
from runtime.config import load_runtime_settings
from runtime.event_bus import EventBus
from runtime.laptop_io import LaptopDeviceError, list_audio_inputs, list_cameras, run_live_preview

PROJECT_NAME = "Cinderella"
PROJECT_VERSION = "0.1.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cinderella local runtime")
    parser.add_argument(
        "--devices",
        action="store_true",
        help="List microphone inputs and probe common camera indexes.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run local speech transcription, vision recognition, and AI conversation.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open a camera and microphone level preview without model inference.",
    )
    parser.add_argument("--camera-index", type=int, help="Camera index (default from config).")
    parser.add_argument(
        "--microphone-device",
        type=int,
        help="Microphone input index (default from config or Windows default input).",
    )
    parser.add_argument("--chat-model", default="qwen2.5:1.5b", help="Local Ollama model name.")
    parser.add_argument(
        "--speech-model",
        default="small",
        choices=("base", "small", "medium"),
        help="Local multilingual Whisper model size (first use downloads model files).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"{PROJECT_NAME} v{PROJECT_VERSION}")
    if not args.live and not args.preview and not args.devices:
        print("Development environment initialized.")
        print("Use --devices to check hardware, --preview for device checks, or --live for local AI.")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.devices:
            cameras = list_cameras()
            microphones = list_audio_inputs()
            print(f"Camera indexes: {cameras if cameras else 'none detected'}")
            if microphones:
                print("Microphone inputs:")
                for device in microphones:
                    print(f"  [{device['index']}] {device['name']} ({device['channels']} channels)")
            else:
                print("Microphone inputs: none detected")
            if not args.live and not args.preview:
                return

        settings = load_runtime_settings()
        camera_index = settings.camera_index if args.camera_index is None else args.camera_index
        microphone_device = (
            settings.microphone_device if args.microphone_device is None else args.microphone_device
        )
        if args.live:
            from runtime.ai_pipeline import run_ai_live

            run_ai_live(
                camera_index=camera_index,
                microphone_device=microphone_device,
                event_bus=EventBus(settings.runtime_queue_capacity),
                chat_model=args.chat_model,
                speech_model=args.speech_model,
            )
        else:
            run_live_preview(camera_index=camera_index, microphone_device=microphone_device)
    except LaptopDeviceError as exc:
        raise SystemExit(f"Device startup failed: {exc}") from exc
    except LocalInferenceError as exc:
        raise SystemExit(f"Local AI startup failed: {exc}") from exc


if __name__ == "__main__":
    main()