from __future__ import annotations

from dataclasses import dataclass


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
            f"Feature implementation coverage: {implementation_coverage_percent()}% "
            "(not human similarity or accuracy)"
        ),
        (
            f"Reply {values['Reply']}% | Situation {values['Situation']}% | "
            f"Affect cues {values['Affect cues']}% | Person {values['Person']}%"
        ),
        f"Object {values['Object']}% | Dialogue {values['Dialogue']}%",
    )
