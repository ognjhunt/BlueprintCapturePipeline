"""Pure deferred-input declarations shared by intake validation and materialization."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TRAJECTORY_MODE = "derive_from_configured_revision"


OVERVIEW_MODE = "configured_task_thumbnail"


SCENE_BUNDLE_MODE = "configured_scene_bundle"


DEFERRED_KEY = "deferred"


DEFERRABLE_MODES = {
    "native_trajectory_plan_path": TRAJECTORY_MODE,
    "overview_image_paths": OVERVIEW_MODE,
}


class ConfiguredControlsDeferredInputError(RuntimeError):
    """A deferred controls input could not be derived from exact published bytes."""


def deferred_declarations(paths: Any) -> dict[str, str]:
    """Return ``{input_name: mode}`` for every deferred input the intent declares."""

    if not isinstance(paths, Mapping):
        return {}
    declared: dict[str, str] = {}
    for name, value in paths.items():
        if not isinstance(value, Mapping):
            continue
        mode = DEFERRABLE_MODES.get(str(name))
        if (
            mode is None
            or set(value) != {DEFERRED_KEY}
            or value.get(DEFERRED_KEY) != mode
        ):
            raise ConfiguredControlsDeferredInputError(
                f"configured_controls_deferred_declaration_invalid:{name}"
            )
        declared[str(name)] = mode
    return declared


def concrete_paths(paths: Mapping[str, Any]) -> dict[str, Any]:
    """Return the intent paths without their deferred declarations."""

    declared = deferred_declarations(paths)
    return {name: value for name, value in paths.items() if name not in declared}
