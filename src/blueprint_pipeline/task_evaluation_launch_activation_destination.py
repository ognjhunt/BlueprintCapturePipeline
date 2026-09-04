"""Rigid-destination additions to launch activation context assembly."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def destination_probe_reference(
    *,
    run_mode: str,
    compilation: Mapping[str, Any],
    compiled_root: Path,
    error_factory: Callable[[str], Exception],
) -> dict[str, Path]:
    if run_mode != "destination_qualification":
        return {}
    probe_path = Path(
        str(compilation.get("destination_native_probe_request_path") or "")
    ).resolve()
    try:
        probe_path.relative_to(compiled_root)
    except ValueError as exc:
        raise error_factory("launch_activation_destination_probe_request_invalid") from exc
    if (
        probe_path.is_symlink()
        or not probe_path.is_file()
        or _sha256(probe_path)
        != compilation.get("destination_native_probe_request_digest")
    ):
        raise error_factory("launch_activation_destination_probe_request_invalid")
    return {"execution_adapter.destination_probe_request": probe_path}


def destination_native_operations(
    *, run_mode: str, materialized: Mapping[str, Path]
) -> dict[str, str]:
    if run_mode != "destination_qualification":
        return {}
    return {
        "destination_probe_request": str(
            materialized["execution_adapter.destination_probe_request"]
        ),
        "configured_scene_support_plane": str(
            materialized["scene.configured_revision.registration.support_plane"]
        ),
        "destination_static_qualification": str(
            materialized["task.destination.static_qualification"]
        ),
        "destination_native_import_qualification": str(
            materialized["task.destination.native_import_qualification"]
        ),
        "destination_geometry": str(materialized["task.destination.geometry"]),
    }


def lineage_operations(
    lineage: Mapping[str, Any], materialized: Mapping[str, Path]
) -> dict[str, str]:
    if lineage["kind"] == "initial_project":
        names = ("project_spend_reconciliation", "initial_provider_zero")
        return {name: str(materialized[f"lineage.{name}"]) for name in names}
    required = (
        "prior_authority",
        "prior_result",
        "prior_launch_receipt",
        "prior_webapp_sync",
        "prior_provider_zero",
        "prior_spend_reconciliation",
    )
    optional = (
        "construction_result",
        "destination_qualification_result",
        "zero_action_result",
        "controls_qualification_manifest",
    )
    result = {name: str(materialized[f"lineage.{name}"]) for name in required}
    result.update({
        name: str(materialized[f"lineage.{name}"])
        for name in optional
        if name in lineage
    })
    return result


__all__ = [
    "destination_native_operations",
    "destination_probe_reference",
    "lineage_operations",
]
