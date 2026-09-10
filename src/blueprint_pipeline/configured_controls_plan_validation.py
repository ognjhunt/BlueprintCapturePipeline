"""Validate sealed controls progression plans before any queue mutation.

Preserves the worker's schema, exact artifact inventory, modes and future-output
checks; service progression and pause/omission decisions remain in the worker.
"""
from __future__ import annotations

from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest


def load_configured_controls_plan(
    path: Path, *, load_json: Callable[..., dict[str, Any]], sha256: Callable[[Path], str],
    error_factory: Callable[[str], Exception], plan_schema: str, destination_plan_schema: str,
    commit_pattern: re.Pattern[str],
) -> dict[str, Any]:
    value = load_json(path, blocker="configured_controls_worker_plan_invalid")
    schema = value.get("schema_version")
    expected_phases = (
        {"destination", "construction", "controls"}
        if schema == destination_plan_schema
        else {"construction", "controls"}
    )
    first_phase = (
        "destination" if "destination" in expected_phases else "construction"
    )
    if (
        schema not in {plan_schema, destination_plan_schema}
        or value.get("enabled") is not True
        or value.get("plan_digest") != canonical_digest(value, digest_field="plan_digest")
        or not str(value.get("source_launch_id") or "").strip()
        or not str(value.get("source_launch_receipt_digest") or "").startswith(
            "sha256:"
        )
        or commit_pattern.fullmatch(
            str(value.get("source_configuration_commit") or "")
        )
        is None
        or commit_pattern.fullmatch(str(value.get("expected_production_commit") or ""))
        is None
        or not str(value.get("submitted_by") or "").strip()
        or set(value.get("phases") or {}) != expected_phases
        or any(
            set(value["phases"].get(phase) or {})
            != {
                "release_window_template_path",
                "authorization_path",
                "launch_authority_path",
                *({"lineage_path"} if phase == first_phase else set()),
            }
            for phase in expected_phases
        )
        or not Path(str(value.get("profile_dir") or "")).is_absolute()
        or Path(str(value.get("profile_dir") or "")).is_symlink()
        or not Path(str(value.get("profile_dir") or "")).is_dir()
    ):
        raise error_factory(
            "configured_controls_worker_plan_invalid"
        )
    inventory = value.get("artifact_inventory")
    declared_paths: set[str] = set()

    def collect_paths(row: Any, key: str = "") -> None:
        if isinstance(row, Mapping):
            for child_key, child in row.items():
                if child_key.endswith("_path") and isinstance(child, str):
                    declared_paths.add(child)
                elif child_key == "lineage_artifact_paths" and isinstance(
                    child, Mapping
                ):
                    declared_paths.update(str(item) for item in child.values())
                elif child_key not in {"artifact_inventory", "future_outputs"}:
                    collect_paths(child, child_key)

    collect_paths(value)
    if not isinstance(inventory, Mapping) or not inventory:
        raise error_factory(
            "configured_controls_worker_plan_inventory_invalid"
        )
    inventory_paths: set[str] = set()
    for row in inventory.values():
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "digest",
            "size_bytes",
            "mode",
        }:
            raise error_factory(
                "configured_controls_worker_plan_inventory_invalid"
            )
        artifact = Path(str(row.get("path") or ""))
        try:
            metadata = artifact.stat()
        except OSError as exc:
            raise error_factory(
                "configured_controls_worker_plan_inventory_invalid"
            ) from exc
        if (
            not artifact.is_absolute()
            or artifact.is_symlink()
            or not artifact.is_file()
            or sha256(artifact) != row.get("digest")
            or metadata.st_size != row.get("size_bytes")
            or f"{stat.S_IMODE(metadata.st_mode):04o}" != row.get("mode")
        ):
            raise error_factory(
                "configured_controls_worker_plan_inventory_invalid"
            )
        inventory_paths.add(str(artifact))
    future = value.get("future_outputs")
    if not isinstance(future, Mapping) or set(future) != expected_phases:
        raise error_factory(
            "configured_controls_worker_plan_future_outputs_invalid"
        )
    for phase in expected_phases:
        row = future.get(phase)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"expected_activation_id"}
            or not str(row.get("expected_activation_id") or "")
        ):
            raise error_factory(
                "configured_controls_worker_plan_future_outputs_invalid"
            )
    if inventory_paths != declared_paths:
        raise error_factory(
            "configured_controls_worker_plan_inventory_invalid"
        )
    return value

