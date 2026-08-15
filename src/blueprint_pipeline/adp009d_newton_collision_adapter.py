"""Newton-only adapter for the sealed SAGE task-collision derivative."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package layout
    from .decision_evidence_contracts import canonical_digest


NEWTON_TASK_COLLISION_ADAPTER_FILENAME = (
    "sage_task_collision_newton_controls_adapter.usda"
)
SAGE_TASK_COLLISION_SHAPE_LABELS = (
    "SM_floorplan",
    "Z6TL2HRVAIIBIPTUKE888888",
    "ZBRQEFBVAI3DWPTUKY888888",
    "ZE6ZHARVAII2IPTUL4888888",
    "ZEMALJZVAJTQWPTUK4888888",
    "ZEO7DVBVAI7DEPTUKU888888",
    "ZEOP4DRVAIJFSPTUKE888888",
    "ZHQYBPJVAI3AUPTULE888888",
    "ZHQYGJJVAJYEYPTUK4888888",
    "ZV67OQJVAJSVCPTULY888888",
    "ZXXPXAZVAJ3T6PTULI888888",
    "_IMCHJBVAV7AMPTUKI888888",
    "_K7DXDRVAZU7IPTULI888888_004",
    "_LTFTHJVAZ3VMPTUJU888888",
    "_PROTIZVAJTMCPTULU888888",
)
NEWTON_INCOMPATIBLE_CONCAVE_COLLISION_LABELS = ("SM_floorplan",)
NEWTON_SAGE_COLLISION_SHAPE_LABELS = tuple(
    label
    for label in SAGE_TASK_COLLISION_SHAPE_LABELS
    if label not in NEWTON_INCOMPATIBLE_CONCAVE_COLLISION_LABELS
)
NEWTON_SAGE_COLLISION_FILTER_SHAPE_EXPRS = tuple(
    f"*{label}" for label in NEWTON_SAGE_COLLISION_SHAPE_LABELS
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _adapter_text(source_path: Path) -> str:
    source = source_path.resolve().as_posix()
    return f'''#usda 1.0
(
    defaultPrim = "Root"
    subLayers = [
        @{source}@
    ]
)

over "Root"
{{
    over "SM_floorplan" (
        active = false
    )
    {{
    }}
}}
'''


def materialize_newton_sage_collision_adapter(
    source_path: Path, *, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Write and digest-bind the Newton controls-only collision overlay."""

    if not source_path.is_file():
        raise RuntimeError("adp009d_newton_sage_collision_source_missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / NEWTON_TASK_COLLISION_ADAPTER_FILENAME
    if adapter_path.exists():
        raise RuntimeError("adp009d_newton_sage_collision_adapter_exists")
    adapter_path.write_text(_adapter_text(source_path), encoding="utf-8")
    receipt: dict[str, Any] = {
        "schema_version": "adp009d_newton_sage_collision_adapter.v1",
        "status": "ready",
        "physics_backend": "newton",
        "source_derivative_sha256": _sha256(source_path),
        "adapter_sha256": _sha256(adapter_path),
        "disabled_source_prim_paths": ["/Root/SM_floorplan"],
        "retained_shape_labels": list(NEWTON_SAGE_COLLISION_SHAPE_LABELS),
        "source_derivative_mutated": False,
        "reason": "mujoco_convex_hull_would_fill_concave_room_volume",
        "comparison_eligible": False,
        "claim_ceiling": "newton_controls_only",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return adapter_path, receipt


def newton_sage_collision_runtime_profile(
    task_collision_manifest: Mapping[str, Any],
) -> dict[str, int]:
    """Derive the exact live profile after the one typed Newton exclusion."""

    rows = task_collision_manifest.get("source_prim_rows")
    if not isinstance(rows, list):
        raise RuntimeError("adp009d_newton_sage_collision_rows_missing")
    excluded = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("source_prim") == "/Root/SM_floorplan"
    ]
    if len(excluded) != 1:
        raise RuntimeError("adp009d_newton_sage_floorplan_row_invalid")
    row = excluded[0]
    try:
        point_count = int(task_collision_manifest["derived_point_count"]) - int(
            row["derived_point_count"]
        )
        face_count = int(task_collision_manifest["derived_face_count"]) - int(
            row["derived_face_count"]
        )
        mesh_count = int(task_collision_manifest["active_source_prim_count"]) - 1
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("adp009d_newton_sage_floorplan_row_invalid") from exc
    profile = {
        "active_mesh_count": mesh_count,
        "active_point_count": point_count,
        "active_face_count": face_count,
        "rigid_body_count": 0,
        "triangle_mesh_count": mesh_count,
    }
    if (
        mesh_count != len(NEWTON_SAGE_COLLISION_SHAPE_LABELS)
        or point_count <= 0
        or face_count <= 0
    ):
        raise RuntimeError("adp009d_newton_sage_collision_profile_invalid")
    return profile


def validated_sage_task_collision_profile(
    task_collision_manifest: Mapping[str, Any],
) -> dict[str, int]:
    """Validate and return the sealed SAGE collision derivative profile."""

    profile = {
        "active_mesh_count": int(task_collision_manifest["active_source_prim_count"]),
        "active_point_count": int(task_collision_manifest["derived_point_count"]),
        "active_face_count": int(task_collision_manifest["derived_face_count"]),
        "rigid_body_count": 0,
        "triangle_mesh_count": int(task_collision_manifest["active_source_prim_count"]),
    }
    shape_labels = tuple(
        str(row.get("source_prim", "")).rsplit("/", 1)[-1]
        for row in task_collision_manifest.get("source_prim_rows", [])
    )
    if (
        shape_labels != SAGE_TASK_COLLISION_SHAPE_LABELS
        or task_collision_manifest.get("candidate_source_prim_count") != 16
        or task_collision_manifest.get("source_face_count") != 47_359
        or task_collision_manifest.get("roi_min_m") != [2.4681748, -4.3100837, -0.1]
        or task_collision_manifest.get("roi_max_m") != [4.4681748, -1.9100837, 1.8]
        or task_collision_manifest.get("maximum_edge_limit_m") != 0.5
        or float(task_collision_manifest.get("observed_maximum_edge_m", math.inf))
        > 0.500001
        or float(task_collision_manifest.get("relative_surface_area_error", math.inf))
        > 1.0e-6
    ):
        raise RuntimeError("sage_task_collision_profile_invalid")
    return profile


def validate_sage_task_collision_binding(
    task_collision_manifest: Mapping[str, Any],
    task_collision_path: Path,
    *,
    expected_source_sha256: str,
    derivative_filename: str,
) -> None:
    """Reopen the sealed source-to-derivative binding before composition."""

    if (
        task_collision_manifest.get("status") != "ready"
        or task_collision_manifest.get("sealed_source_sha256")
        != expected_source_sha256
        or task_collision_manifest.get("sealed_source_mutated") is not False
        or task_collision_manifest.get("derivative_filename") != derivative_filename
        or not task_collision_path.is_file()
        or _sha256(task_collision_path)
        != task_collision_manifest.get("derivative_sha256")
        or task_collision_manifest.get("claim_ceiling")
        != "preregistered_franka_task_envelope_only"
    ):
        raise RuntimeError("sage_task_collision_derivative_binding_invalid")


def prepare_newton_sage_collision_adapter(
    source_path: Path,
    *,
    task_collision_manifest: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any], dict[str, int]]:
    """Materialize and reopen the exact adapter before runtime composition."""

    adapter_path, receipt = materialize_newton_sage_collision_adapter(
        source_path, output_dir=output_dir
    )
    if (
        receipt.get("status") != "ready"
        or receipt.get("source_derivative_sha256")
        != task_collision_manifest.get("derivative_sha256")
        or receipt.get("disabled_source_prim_paths") != ["/Root/SM_floorplan"]
        or receipt.get("comparison_eligible") is not False
    ):
        raise RuntimeError("adp009d_newton_sage_collision_adapter_invalid")
    return (
        adapter_path,
        receipt,
        newton_sage_collision_runtime_profile(task_collision_manifest),
    )


__all__ = [
    "NEWTON_SAGE_COLLISION_FILTER_SHAPE_EXPRS",
    "NEWTON_SAGE_COLLISION_SHAPE_LABELS",
    "NEWTON_TASK_COLLISION_ADAPTER_FILENAME",
    "SAGE_TASK_COLLISION_SHAPE_LABELS",
    "materialize_newton_sage_collision_adapter",
    "newton_sage_collision_runtime_profile",
    "prepare_newton_sage_collision_adapter",
    "validate_sage_task_collision_binding",
    "validated_sage_task_collision_profile",
]
