from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CapabilityStatus(StrEnum):
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    SIMULATED = "simulated"
    EXTERNAL_ACTIVE = "external_active"
    EXTERNAL_DEGRADED = "external_degraded"
    INTERNAL_SCAFFOLD = "internal_scaffold"
    INTERNAL_TRAINING = "internal_training"
    INTERNAL_AVAILABLE = "internal_available"
    EVALUATED = "evaluated"
    FAILED = "failed"


@dataclass(frozen=True)
class CapabilityRecord:
    name: str
    status: CapabilityStatus
    implementation_type: str
    backend: str | None
    model: str | None
    network_required: bool
    privacy_impact: str
    last_health_check: str | None
    evaluation_status: str
    known_limitations: tuple[str, ...]


def capability_statuses() -> tuple[CapabilityRecord, ...]:
    """Coarse implementation provenance, not a measure of perception quality."""
    return (
        CapabilityRecord(
            "speech_recognition",
            CapabilityStatus.DISABLED,
            "external",
            "faster-whisper",
            "Whisper small/base/medium pretrained weights",
            True,
            "Processes transient microphone speech locally",
            None,
            "not_benchmarked",
            ("Requires optional dependencies and first-use model download.",),
        ),
        CapabilityRecord(
            "vision",
            CapabilityStatus.DISABLED,
            "external",
            "MediaPipe Tasks",
            "Pretrained object, pose, and face movement models",
            True,
            "Processes transient camera frames locally",
            None,
            "not_benchmarked",
            ("Object labels depend on pretrained supported classes.",),
        ),
        CapabilityRecord(
            "conversation",
            CapabilityStatus.DISABLED,
            "external_local_backend",
            "Ollama",
            "Qwen text and vision-language models",
            True,
            "Text and requested current-frame thumbnail go to loopback Ollama",
            None,
            "not_benchmarked",
            ("Backend implementation is not developed in this repository.",),
        ),
        CapabilityRecord(
            "speech_output",
            CapabilityStatus.DISABLED,
            "external_local_backend",
            "pyttsx3 / Windows SAPI",
            None,
            False,
            "Synthesizes speech locally",
            None,
            "not_benchmarked",
            ("Available Korean voice depends on the Windows installation.",),
        ),
        CapabilityRecord(
            "simulated_physical_actions",
            CapabilityStatus.SIMULATED,
            "internal_simulation",
            "PhysicalAiSimulator",
            None,
            False,
            "Synthetic scene and commands only",
            None,
            "not_applicable",
            ("Does not control real hardware.",),
        ),
        CapabilityRecord(
            "external_information_tools",
            CapabilityStatus.DISABLED,
            "opt_in_provider_scaffold",
            "MediaWiki / Open-Meteo",
            None,
            True,
            "No external requests unless explicitly enabled and policy-approved",
            None,
            "not_evaluated",
            ("Open-Meteo provider is blocked by the paid-service policy.",),
        ),
    )


@dataclass(frozen=True)
class CapabilityCoverage:
    name: str
    implementation: float


CAPABILITY_COVERAGE = (
    CapabilityCoverage("Reply", 0.5),
    CapabilityCoverage("Situation", 0.5),
    CapabilityCoverage("Affect cues", 0.25),
    CapabilityCoverage("Person", 0.5),
    CapabilityCoverage("Object", 0.5),
    CapabilityCoverage("Dialogue", 1.0),
)


def implementation_coverage_percent() -> int:
    """Return equal-weight implementation coverage, not human similarity or accuracy."""
    if not CAPABILITY_COVERAGE:
        return 0
    return round(
        100 * sum(item.implementation for item in CAPABILITY_COVERAGE) / len(CAPABILITY_COVERAGE)
    )


def capability_overlay_lines() -> tuple[str, str, str]:
    values = {item.name: round(item.implementation * 100) for item in CAPABILITY_COVERAGE}
    return (
        (
            f"Developer estimate only: {implementation_coverage_percent()}% "
            "(not human similarity or accuracy)"
        ),
        (
            f"Reply {values['Reply']}% | Situation {values['Situation']}% | "
            f"Affect cues {values['Affect cues']}% | Person {values['Person']}%"
        ),
        f"Object {values['Object']}% | Dialogue {values['Dialogue']}%",
    )
