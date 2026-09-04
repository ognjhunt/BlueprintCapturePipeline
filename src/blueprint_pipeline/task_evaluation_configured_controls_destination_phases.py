"""Closed phase shapes for legacy and rigid-destination controls autostart."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LEGACY_INTENT_SCHEMA_VERSION = "task_evaluation_configured_controls_autostart_intent.v2"
DESTINATION_INTENT_SCHEMA_VERSION = "task_evaluation_configured_controls_autostart_intent.v3"
INTENT_SCHEMA_VERSIONS = {
    LEGACY_INTENT_SCHEMA_VERSION,
    DESTINATION_INTENT_SCHEMA_VERSION,
}
LEGACY_PHASE_PATHS = {
    "construction": {
        "release_window_template_path",
        "lineage_path",
        "authorization_path",
        "launch_authority_path",
    },
    "controls": {
        "release_window_template_path",
        "authorization_path",
        "launch_authority_path",
    },
}
DESTINATION_PHASE_PATHS = {
    "destination": {
        "release_window_template_path",
        "lineage_path",
        "authorization_path",
        "launch_authority_path",
    },
    "construction": {
        "release_window_template_path",
        "authorization_path",
        "launch_authority_path",
    },
    "controls": {
        "release_window_template_path",
        "authorization_path",
        "launch_authority_path",
    },
}


def phase_paths(phases: Any) -> Mapping[str, set[str]] | None:
    if not isinstance(phases, Mapping):
        return None
    if set(phases) == set(DESTINATION_PHASE_PATHS):
        return DESTINATION_PHASE_PATHS
    if set(phases) == set(LEGACY_PHASE_PATHS):
        return LEGACY_PHASE_PATHS
    return None


def schema_for_phases(phases: Any) -> str:
    return (
        DESTINATION_INTENT_SCHEMA_VERSION
        if phase_paths(phases) is DESTINATION_PHASE_PATHS
        else LEGACY_INTENT_SCHEMA_VERSION
    )


__all__ = [
    "DESTINATION_INTENT_SCHEMA_VERSION",
    "DESTINATION_PHASE_PATHS",
    "INTENT_SCHEMA_VERSIONS",
    "LEGACY_INTENT_SCHEMA_VERSION",
    "LEGACY_PHASE_PATHS",
    "phase_paths",
    "schema_for_phases",
]
