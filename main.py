from __future__ import annotations

import argparse
import logging

from runtime.config import load_runtime_settings
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
        help="Open a live camera and microphone preview without recording.",
    )
    parser.add_argument("--camera-index", type=int, help="Camera index (default from config).")
    parser.add_argument(
        "--microphone-device",
        type=int,
        help="Microphone input index (default from config or Windows default input).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"{PROJECT_NAME} v{PROJECT_VERSION}")
    if not args.live and not args.devices:
        print("Development environment initialized.")
        print("Use --devices to check local hardware or --live to start a non-recording preview.")
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
            if not args.live:
                return

        settings = load_runtime_settings()
        run_live_preview(
            camera_index=settings.camera_index if args.camera_index is None else args.camera_index,
            microphone_device=(
                settings.microphone_device
                if args.microphone_device is None
                else args.microphone_device
            ),
        )
    except LaptopDeviceError as exc:
        raise SystemExit(f"Device startup failed: {exc}") from exc


if __name__ == "__main__":
    main()