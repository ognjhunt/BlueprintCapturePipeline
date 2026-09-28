"""Order scene object cleanup after remote-writer terminal proof."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def cleanup_scene_staging(
    *,
    adapter: Mapping[str, Any],
    staging_dir: Path,
    cleanup: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    if (
        adapter.get("continuing_spend_from_this_run") is True
        and adapter.get("retained_owned") is not True
    ):
        return {
            "status": "deferred_until_provider_absent",
            "all_objects_absent": False,
            "continuing_spend_from_this_run": True,
            "raw_secret_values_recorded": False,
        }
    return cleanup(staging_dir)


__all__ = ["cleanup_scene_staging"]
