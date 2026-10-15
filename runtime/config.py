from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a configuration value is invalid."""


LOGGING_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True)
class RuntimeSettings:
    runtime_queue_capacity: int
    event_publish_timeout: float
    shutdown_timeout: float
    logging_level: str
    simulation_seed: int
    audio_simulator_enabled: bool
    vision_simulator_enabled: bool
    hardware_simulator_enabled: bool

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> RuntimeSettings:
        try:
            runtime_queue_capacity = int(values["runtime_queue_capacity"])
            event_publish_timeout = float(values["event_publish_timeout"])
            shutdown_timeout = float(values["shutdown_timeout"])
            logging_level = str(values["logging_level"]).upper()
            simulation_seed = int(values["simulation_seed"])
            audio_simulator_enabled = bool(values["audio_simulator_enabled"])
            vision_simulator_enabled = bool(values["vision_simulator_enabled"])
            hardware_simulator_enabled = bool(values["hardware_simulator_enabled"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ConfigurationError(f"Missing required config key: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid config value type: {exc}") from exc

        if runtime_queue_capacity <= 0:
            raise ConfigurationError("runtime_queue_capacity must be greater than zero.")
        if event_publish_timeout <= 0:
            raise ConfigurationError("event_publish_timeout must be greater than zero.")
        if shutdown_timeout <= 0:
            raise ConfigurationError("shutdown_timeout must be greater than zero.")
        if logging_level not in LOGGING_LEVELS:
            raise ConfigurationError(
                "logging_level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )

        return cls(
            runtime_queue_capacity=runtime_queue_capacity,
            event_publish_timeout=event_publish_timeout,
            shutdown_timeout=shutdown_timeout,
            logging_level=logging_level,
            simulation_seed=simulation_seed,
            audio_simulator_enabled=audio_simulator_enabled,
            vision_simulator_enabled=vision_simulator_enabled,
            hardware_simulator_enabled=hardware_simulator_enabled,
        )


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""
    if value.startswith("#"):
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        return value[1:-1]
    if "." in value:
        try:
            return float(value)
        except ValueError:
            pass
    try:
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if ":" not in raw_line:
            raise ConfigurationError(f"Invalid YAML syntax at line {line_number}: {raw_line!r}")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        if not key:
            raise ConfigurationError(f"Missing config key at line {line_number}: {raw_line!r}")
        mapping[key] = _parse_scalar(raw_value)
    return mapping


def load_runtime_settings(path: str | Path | None = None) -> RuntimeSettings:
    """Load a minimal, typed runtime configuration from a YAML subset file.

    The project intentionally avoids adding a third-party YAML parser dependency.
    The supported format is a flat mapping of key: value entries, which is enough for
    the minimal runtime configuration used by the project.
    """

    config_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    if not config_path.exists():
        raise ConfigurationError(f"Runtime configuration file not found: {config_path}")
    raw_values = _load_simple_yaml(config_path)
    return RuntimeSettings.from_mapping(raw_values)
