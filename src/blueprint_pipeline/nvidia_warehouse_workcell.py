"""Pinned NVIDIA Warehouse workcell materialization and native-canary freeze.

This module intentionally stops before Isaac execution.  It materializes the
previously selected sorting workcell and its local USD composition closure,
records unresolved non-dataset references, and writes the exact camera/runtime
checks a native Isaac canary must satisfy before a policy/WAM loop may run.
"""

from __future__ import annotations

import json
import posixpath
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .common import write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "nvidia_warehouse_workcell_materialization.v1"
CANARY_SPEC_SCHEMA_VERSION = "nvidia_warehouse_native_camera_canary_spec.v1"
DATASET_ID = "nvidia/PhysicalAI-SimReady-Warehouse-01"
DATASET_REVISION = "c7fe115cb79c7ddbd0532630d7768b5736b0ecc4"
DATASET_RESOLVE_BASE = (
    "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/"
    f"{DATASET_REVISION}/"
)
ROOT_USD = "physical_ai_simready_warehouse_01.usd"
SORTING_USD = "SubLayers/sorting_area_physics.usd"
WORKCELL_USD = "Props/assembly/SM_CratePacking_Table_A1/SM_CratePacking_Table_A1_physics.usd"
TABLE_USD = (
    "Props/general/SM_HeavyDutyPackingTable_C02_01/"
    "SM_HeavyDutyPackingTable_C02_01_physics.usd"
)
SPRAYCAN_USD = (
    "Props/general/HandManipulation/paint_container_spraycan_a/"
    "sm_paint_container_spraycan_a01_simready_01.usd"
)
PROVENANCE_FILES = (ROOT_USD, SORTING_USD)
RUNTIME_SEEDS = (WORKCELL_USD, SPRAYCAN_USD)
PINNED_SHA256 = {
    ROOT_USD: "4e01e55f055689ff0cd669f33c6c305f73add8903547eaf3ef070d515cb2ec8a",
    SORTING_USD: "331d05475875ab18613e8c7112a92f0bede3b6d3bafd082633eea37b0483c398",
    WORKCELL_USD: "2173941e43b672206983ced927ea2f56f82247e1fc4a43c0a3727205f5c0c9aa",
    TABLE_USD: "7c7348b37d7be04eafabcaf20f1f5a05c7a9c0633bc1ba7048ff4fae2ba005c5",
    SPRAYCAN_USD: "fe34476927334e5cb6cba9b90cfa3b442e46580af8150bda1ae867827b3c40a2",
}
MAX_MATERIALIZED_BYTES = 512 * 1024 * 1024


def _safe_relative(value: str) -> str:
    normalized = posixpath.normpath(str(value).replace("\\", "/"))
    path = PurePosixPath(normalized)
    if normalized in {"", "."} or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"nvidia_warehouse_dependency_path_unsafe:{value}")
    return path.as_posix()


def _default_fetch(relative_path: str, destination: Path) -> None:
    url = DATASET_RESOLVE_BASE + urllib.parse.quote(relative_path, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": "BlueprintDiagnostic/1"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _usd_dependencies(path: Path) -> Sequence[str]:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise ValueError(f"nvidia_warehouse_usd_layer_open_failed:{path}")
    return tuple(str(value) for value in layer.GetCompositionAssetDependencies())


def _resolve_dependency(owner: str, reference: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(reference)
    if parsed.scheme:
        return "external", reference
    relative = posixpath.normpath(posixpath.join(posixpath.dirname(owner), reference))
    return "dataset", _safe_relative(relative)


def materialize_pinned_workcell(
    *,
    output_root: str | Path,
    fetcher: Callable[[str, Path], None] = _default_fetch,
    dependency_reader: Callable[[Path], Sequence[str]] = _usd_dependencies,
    pinned_sha256: Mapping[str, str] = PINNED_SHA256,
    max_materialized_bytes: int = MAX_MATERIALIZED_BYTES,
) -> dict[str, Any]:
    """Download and hash the selected workcell's dataset-local USD closure."""

    root = Path(output_root).expanduser().resolve()
    if max_materialized_bytes <= 0:
        raise ValueError("nvidia_warehouse_materialization_byte_cap_invalid")
    root.mkdir(parents=True, exist_ok=True)
    fetched: set[str] = set()
    external_dependencies: set[str] = set()
    total_bytes = 0

    def ensure(relative: str) -> Path:
        nonlocal total_bytes
        safe = _safe_relative(relative)
        destination = (root / safe).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("nvidia_warehouse_dependency_escaped_output_root")
        if not destination.is_file():
            fetcher(safe, destination)
        if not destination.is_file() or destination.is_symlink() or destination.stat().st_size <= 0:
            raise ValueError(f"nvidia_warehouse_dependency_fetch_invalid:{safe}")
        if safe not in fetched:
            total_bytes += destination.stat().st_size
            if total_bytes > max_materialized_bytes:
                raise ValueError("nvidia_warehouse_materialization_byte_cap_exceeded")
            fetched.add(safe)
        expected = pinned_sha256.get(safe)
        if expected is not None and file_sha256(destination) != expected:
            raise ValueError(f"nvidia_warehouse_pinned_sha256_mismatch:{safe}")
        return destination

    for relative in PROVENANCE_FILES:
        ensure(relative)

    queue = deque(RUNTIME_SEEDS)
    traversed: set[str] = set()
    while queue:
        relative = _safe_relative(queue.popleft())
        path = ensure(relative)
        if relative in traversed:
            continue
        traversed.add(relative)
        for reference in dependency_reader(path):
            kind, resolved = _resolve_dependency(relative, reference)
            if kind == "external":
                external_dependencies.add(resolved)
            elif resolved not in traversed:
                queue.append(resolved)

    missing_pinned = sorted(path for path in pinned_sha256 if path not in fetched)
    if missing_pinned:
        raise ValueError(f"nvidia_warehouse_required_pinned_asset_missing:{missing_pinned[0]}")
    files = [
        {
            "relative_path": relative,
            "path": str(root / relative),
            "sha256": file_sha256(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in sorted(fetched)
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "output_root": str(root),
        "runtime_scope": "selected_sorting_area_crate_packing_workcell_only",
        "whole_warehouse_materialized": False,
        "provenance_files": list(PROVENANCE_FILES),
        "runtime_seeds": list(RUNTIME_SEEDS),
        "file_count": len(files),
        "size_bytes": sum(int(row["size_bytes"]) for row in files),
        "files": files,
        "external_dependencies": sorted(external_dependencies),
        "dataset_local_dependency_closure_complete": True,
        "optional_external_material_dependencies_resolved": not external_dependencies,
        "rankings_or_policy_outcomes_accessed": False,
        "claim_boundary": {
            "native_isaac_scene_loaded": False,
            "camera_accuracy_proven": False,
            "physics_execution_proven": False,
            "policy_wam_loop_proven": False,
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    write_json(root / "materialization_manifest.json", payload)
    return payload


def build_native_camera_canary_spec(
    *, materialization_manifest_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Freeze the native scene/camera checks before inspecting policy rankings."""

    manifest_path = Path(materialization_manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("status") != "completed":
        raise ValueError("nvidia_warehouse_materialization_manifest_not_completed")
    declared = manifest.get("manifest_sha256")
    material = dict(manifest)
    material.pop("manifest_sha256", None)
    if declared != canonical_sha256(material):
        raise ValueError("nvidia_warehouse_materialization_manifest_sha256_invalid")
    if manifest.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("nvidia_warehouse_materialization_revision_invalid")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("nvidia_warehouse_native_canary_spec_overwrite_forbidden")
    spec: dict[str, Any] = {
        "schema_version": CANARY_SPEC_SCHEMA_VERSION,
        "status": "prospective_native_isaac_camera_canary_not_run",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "materialization_manifest_path": str(manifest_path),
        "materialization_manifest_file_sha256": file_sha256(manifest_path),
        "materialization_manifest_sha256": declared,
        "scene": {
            "workcell_usd": WORKCELL_USD,
            "spraycan_usd": SPRAYCAN_USD,
            "task_instruction": "Pick up the spray can and place it inside the marked tray.",
            "franka_asset": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            "franka_dof_count": 9,
            "spraycan_rigid_body_must_be_added_at_runtime": True,
            "source_colliders_must_be_preserved": True,
            "placements": {
                "workcell_translation_m": [0.0, 0.0, 0.0],
                "franka_base_translation_m": [-0.50, 0.0, 0.995],
                "spraycan_translation_m": [0.0, 0.075, 1.085],
                "tray_center_translation_m": [-0.05, 0.32, 1.025],
                "source": "frozen_mujoco_task_offsets_registered_to_workcell_tabletop",
                "must_pass_scripted_feasibility_before_policy_or_wam": True,
            },
        },
        "cameras": {
            "external": {
                "resolution": [640, 480],
                "position_m": [0.75, -0.85, 2.095],
                "look_at_m": [-0.07, 0.16, 1.165],
                "vertical_fov_deg": 52.0,
                "fixed_in_world": True,
            },
            "wrist": {
                "resolution": [640, 480],
                "parent_link_suffix": "/panda_hand",
                "mount_translation_m": [0.0, 0.10, 0.03],
                "mount_forward_parent": [0.0, 0.0, 1.0],
                "mount_up_parent": [0.0, 1.0, 0.0],
                "vertical_fov_deg": 82.0,
                "inherits_parent_transform": True,
                "per_frame_task_reaim_forbidden": True,
            },
        },
        "required_checks": [
            "isaac_sim_6_exact_image_digest_recorded",
            "workcell_stage_loaded_without_missing_dataset_local_dependencies",
            "franka_articulation_has_exactly_9_dofs",
            "spraycan_has_collision_and_runtime_rigid_body",
            "external_rgb_nonblank_and_franka_spraycan_tray_visible",
            "wrist_rgb_nonblank_and_task_object_visible_at_initial_pose",
            "wrist_camera_world_pose_changes_under_command",
            "wrist_camera_to_panda_hand_transform_remains_constant",
            "external_and_wrist_timestamps_match_physics_steps",
            "at_least_two_policy_calls_separated_by_one_wam_generated_observation",
        ],
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "paid_gpu_execution_admitted": False,
        "claim_boundary": {
            "ranking_accuracy": False,
            "physical_success": False,
            "captured_site_transfer_validation": False,
            "phase_b_confirmation": False,
        },
    }
    spec["spec_sha256"] = canonical_sha256(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, spec)
    return spec


__all__ = [
    "CANARY_SPEC_SCHEMA_VERSION",
    "DATASET_REVISION",
    "PINNED_SHA256",
    "SCHEMA_VERSION",
    "build_native_camera_canary_spec",
    "materialize_pinned_workcell",
]
