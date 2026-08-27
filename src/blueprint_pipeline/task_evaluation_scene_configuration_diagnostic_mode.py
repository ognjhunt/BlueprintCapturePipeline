"""Fail-closed bootstrap modes for scene-configuration diagnostics."""

from __future__ import annotations

from typing import Final


FRESH_DIAGNOSTIC_BOOTSTRAP_MODE: Final = "fresh"
CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE: Final = "checkpoint_resume"
DIAGNOSTIC_BOOTSTRAP_MODES: Final = frozenset(
    {
        FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
        CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE,
    }
)


def validate_diagnostic_bootstrap_mode(value: object) -> str:
    """Return one explicit mode; never infer it from missing evidence."""

    if not isinstance(value, str) or value not in DIAGNOSTIC_BOOTSTRAP_MODES:
        raise ValueError("scene_configuration_diagnostic_bootstrap_mode_invalid")
    return value


__all__ = [
    "CHECKPOINT_RESUME_DIAGNOSTIC_BOOTSTRAP_MODE",
    "DIAGNOSTIC_BOOTSTRAP_MODES",
    "FRESH_DIAGNOSTIC_BOOTSTRAP_MODE",
    "validate_diagnostic_bootstrap_mode",
]
