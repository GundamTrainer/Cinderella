from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from integrations.policies import NetworkMode, NetworkPolicy


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
    camera_index: int = 0
    microphone_device: int | None = None
    box_clamp_tolerance_pixels: float = 2.0
    audio_input_queue_capacity: int = 24
    network: NetworkPolicy = field(default_factory=NetworkPolicy)

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
            camera_index = int(values.get("camera_index", 0))
            microphone_value = values.get("microphone_device")
            microphone_device = None if microphone_value is None else int(microphone_value)
            box_clamp_tolerance_pixels = float(
                values.get("box_clamp_tolerance_pixels", 2.0)
            )
            audio_input_queue_capacity = int(values.get("audio_input_queue_capacity", 24))
            network_values = values.get("network", {})
            if not isinstance(network_values, dict):
                raise ConfigurationError("network must be a mapping.")
            network = _network_policy_from_mapping(network_values)
        except ConfigurationError:
            raise
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
        if camera_index < 0:
            raise ConfigurationError("camera_index must be zero or greater.")
        if microphone_device is not None and microphone_device < 0:
            raise ConfigurationError("microphone_device must be zero or greater.")
        if not 0.0 <= box_clamp_tolerance_pixels <= 32.0:
            raise ConfigurationError(
                "box_clamp_tolerance_pixels must be between 0 and 32."
            )
        if audio_input_queue_capacity <= 0:
            raise ConfigurationError("audio_input_queue_capacity must be positive.")

        return cls(
            runtime_queue_capacity=runtime_queue_capacity,
            event_publish_timeout=event_publish_timeout,
            shutdown_timeout=shutdown_timeout,
            logging_level=logging_level,
            simulation_seed=simulation_seed,
            audio_simulator_enabled=audio_simulator_enabled,
            vision_simulator_enabled=vision_simulator_enabled,
            hardware_simulator_enabled=hardware_simulator_enabled,
            camera_index=camera_index,
            microphone_device=microphone_device,
            box_clamp_tolerance_pixels=box_clamp_tolerance_pixels,
            audio_input_queue_capacity=audio_input_queue_capacity,
            network=network,
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
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        try:
            import json

            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid inline list value: {value!r}") from exc
        if not isinstance(parsed, list):
            raise ConfigurationError(f"Expected a list value, got {value!r}.")
        return parsed
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
    current_section: dict[str, Any] | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indentation = len(raw_line) - len(raw_line.lstrip())
        if indentation == 0 and raw_line.rstrip().endswith(":"):
            section_name = raw_line.strip()[:-1].strip()
            if not section_name or section_name in mapping:
                raise ConfigurationError(
                    f"Invalid or duplicate config section at line {line_number}."
                )
            current_section = {}
            mapping[section_name] = current_section
            continue
        if ":" not in raw_line:
            raise ConfigurationError(f"Invalid YAML syntax at line {line_number}: {raw_line!r}")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        if not key:
            raise ConfigurationError(f"Missing config key at line {line_number}: {raw_line!r}")
        if indentation:
            if indentation != 2 or current_section is None:
                raise ConfigurationError(f"Invalid indentation at line {line_number}.")
            if key in current_section:
                raise ConfigurationError(f"Duplicate config key at line {line_number}: {key!r}.")
            current_section[key] = _parse_scalar(raw_value)
        else:
            if key in mapping:
                raise ConfigurationError(f"Duplicate config key at line {line_number}: {key!r}.")
            mapping[key] = _parse_scalar(raw_value)
            current_section = None
    return mapping


def _network_policy_from_mapping(values: dict[str, Any]) -> NetworkPolicy:
    defaults = NetworkPolicy()
    permitted = {
        "mode",
        "allow_external_requests",
        "allowed_domains",
        "require_user_confirmation_for_new_domain",
        "request_timeout_seconds",
        "retry_count",
        "maximum_response_bytes",
        "user_agent",
        "cache_enabled",
        "redact_query_logs",
    }
    unknown = set(values) - permitted
    if unknown:
        raise ConfigurationError(f"Unknown network config key(s): {', '.join(sorted(unknown))}.")
    try:
        mode = NetworkMode(values.get("mode", defaults.mode.value))
        allow_external_requests = values.get(
            "allow_external_requests", defaults.allow_external_requests
        )
        allowed_domains = values.get("allowed_domains", list(defaults.allowed_domains))
        require_confirmation = values.get(
            "require_user_confirmation_for_new_domain",
            defaults.require_user_confirmation_for_new_domain,
        )
        timeout = values.get("request_timeout_seconds", defaults.request_timeout_seconds)
        retries = values.get("retry_count", defaults.retry_count)
        maximum_bytes = values.get("maximum_response_bytes", defaults.maximum_response_bytes)
        user_agent = values.get("user_agent", defaults.user_agent)
        cache_enabled = values.get("cache_enabled", defaults.cache_enabled)
        redact_query_logs = values.get("redact_query_logs", defaults.redact_query_logs)
        for key, value in (
            ("allow_external_requests", allow_external_requests),
            ("require_user_confirmation_for_new_domain", require_confirmation),
            ("cache_enabled", cache_enabled),
            ("redact_query_logs", redact_query_logs),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"network.{key} must be a boolean.")
        if (
            not isinstance(allowed_domains, list)
            or any(not isinstance(domain, str) for domain in allowed_domains)
        ):
            raise TypeError("network.allowed_domains must be a list of hostnames.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("network.request_timeout_seconds must be numeric.")
        if isinstance(retries, bool) or not isinstance(retries, int):
            raise TypeError("network.retry_count must be an integer.")
        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int):
            raise TypeError("network.maximum_response_bytes must be an integer.")
        if not isinstance(user_agent, str):
            raise TypeError("network.user_agent must be a string.")
        return NetworkPolicy(
            mode=mode,
            allow_external_requests=allow_external_requests,
            allowed_domains=tuple(allowed_domains),
            require_user_confirmation_for_new_domain=require_confirmation,
            request_timeout_seconds=float(timeout),
            retry_count=retries,
            maximum_response_bytes=maximum_bytes,
            user_agent=user_agent,
            cache_enabled=cache_enabled,
            redact_query_logs=redact_query_logs,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid network configuration: {exc}") from exc


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
