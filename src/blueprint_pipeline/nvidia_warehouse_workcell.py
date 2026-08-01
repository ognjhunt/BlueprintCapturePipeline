"""Pinned NVIDIA Warehouse workcell materialization and native-canary freeze.

This module intentionally stops before Isaac execution.  It materializes the
previously selected sorting workcell and its local USD composition closure,
records unresolved non-dataset references, and writes the exact camera/runtime
checks a native Isaac canary must satisfy before a policy/WAM loop may run.
"""

from __future__ import annotations

import json
import posixpath
import re
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .common import write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "nvidia_warehouse_workcell_materialization.v3"
CANARY_SPEC_SCHEMA_VERSION = "nvidia_warehouse_native_camera_canary_spec.v1"
DATASET_ID = "nvidia/PhysicalAI-SimReady-Warehouse-01"
DATASET_REVISION = "c7fe115cb79c7ddbd0532630d7768b5736b0ecc4"
DATASET_RESOLVE_BASE = (
    "https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/resolve/"
    f"{DATASET_REVISION}/"
)
DATASET_TREE_BASE = (
    "https://huggingface.co/api/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01/tree/"
    f"{DATASET_REVISION}/"
)
ROOT_USD = "physical_ai_simready_warehouse_01.usd"
SORTING_USD = "SubLayers/sorting_area_physics.usd"
WORKCELL_USD = "Props/assembly/SM_CratePacking_Table_A1/SM_CratePacking_Table_A1_physics.usd"
TABLE_USD = (
    "Props/general/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01_physics.usd"
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
MAX_MATERIALIZED_BYTES = 3 * 1024 * 1024 * 1024
_MDL_TEXTURE_REFERENCE = re.compile(
    r'"([^"\n]+\.(?:bmp|exr|hdr|jpeg|jpg|png|tga|tif|tiff))"', re.IGNORECASE
)
DEPENDENCY_CONTRACT = {
    "usd_composition_dependencies_included": True,
    "usd_authored_asset_fields_included": True,
    "dataset_local_mdl_texture_literals_included": True,
    "udim_patterns_expanded_against_pinned_revision": True,
    "same_asset_directory_external_mirrors_materialized": True,
}


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
        # DATASET_RESOLVE_BASE is a fixed HTTPS origin and relative_path was made safe above.
        with (
            urllib.request.urlopen(  # nosec B310
                request, timeout=120
            ) as response,
            temporary.open("wb") as out,
        ):
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


def _usd_authored_asset_dependencies(path: Path) -> Sequence[str]:
    """Return asset-valued USD fields such as local textures and MDL modules."""

    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise ValueError(f"nvidia_warehouse_usd_layer_open_failed:{path}")
    dependencies: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Sdf.AssetPath):
            if value.path:
                dependencies.add(str(value.path))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                collect(key)
                collect(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                collect(item)

    def visit(spec_path: Any) -> None:
        spec = layer.GetObjectAtPath(spec_path)
        if spec is None:
            return
        for key in spec.ListInfoKeys():
            try:
                collect(spec.GetInfo(key))
            except TypeError:
                # Some pxr builds expose metadata values without a Python
                # converter. Those values cannot contain inspectable AssetPath
                # objects in this process, so continue over the remaining fields.
                continue

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    composition = {str(value) for value in layer.GetCompositionAssetDependencies()}
    return tuple(sorted(dependencies - composition))


def _mdl_dependencies(path: Path) -> Sequence[str]:
    """Return texture literals authored in a dataset-local MDL module."""

    text = path.read_text(encoding="utf-8")
    return tuple(sorted(set(_MDL_TEXTURE_REFERENCE.findall(text))))


def _default_expand_dependency(relative_path: str) -> Sequence[str]:
    """Expand a dataset-local UDIM pattern against the pinned repository tree."""

    safe = _safe_relative(relative_path)
    if "<UDIM>" not in safe:
        return (safe,)
    parent = posixpath.dirname(safe)
    filename = posixpath.basename(safe)
    pattern = re.compile("^" + re.escape(filename).replace(re.escape("<UDIM>"), r"\d{4}") + "$")
    url = DATASET_TREE_BASE + urllib.parse.quote(parent, safe="/")
    request = urllib.request.Request(
        url + "?recursive=false&expand=false&limit=1000",
        headers={"User-Agent": "BlueprintDiagnostic/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
        values = json.load(response)
    if not isinstance(values, list) or len(values) >= 1000:
        raise ValueError("nvidia_warehouse_dependency_listing_invalid_or_truncated")
    matches = sorted(
        _safe_relative(str(value.get("path") or ""))
        for value in values
        if isinstance(value, Mapping)
        and value.get("type") == "file"
        and pattern.fullmatch(posixpath.basename(str(value.get("path") or "")))
    )
    if not matches:
        raise ValueError(f"nvidia_warehouse_udim_dependency_unresolved:{safe}")
    return tuple(matches)


def _resolve_dependency(owner: str, reference: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(reference)
    if parsed.scheme:
        return "external", reference
    relative = posixpath.normpath(posixpath.join(posixpath.dirname(owner), reference))
    return "dataset", _safe_relative(relative)


def _resolve_authored_asset(owner: str, reference: str) -> tuple[str, str]:
    """Distinguish explicit dataset-relative assets from runtime-provided modules."""

    parsed = urllib.parse.urlparse(reference)
    if parsed.scheme or not reference.startswith(("./", "../")):
        return "external", reference
    relative = posixpath.normpath(posixpath.join(posixpath.dirname(owner), reference))
    return "dataset", _safe_relative(relative)


def _dataset_mirror_for_external_asset(owner: str, reference: str) -> str | None:
    """Map an external URI to a same-asset-directory mirror in this dataset."""

    parsed = urllib.parse.urlparse(reference)
    if not parsed.scheme:
        return None
    owner_directory = posixpath.dirname(owner)
    asset_directory_name = posixpath.basename(owner_directory)
    decoded_path = urllib.parse.unquote(parsed.path)
    marker = f"/{asset_directory_name}/"
    if marker not in decoded_path:
        return None
    suffix = decoded_path.rsplit(marker, 1)[1]
    if not suffix:
        return None
    return _safe_relative(posixpath.join(owner_directory, suffix))


def materialize_pinned_workcell(
    *,
    output_root: str | Path,
    fetcher: Callable[[str, Path], None] = _default_fetch,
    dependency_reader: Callable[[Path], Sequence[str]] = _usd_dependencies,
    asset_dependency_reader: Callable[[Path], Sequence[str]] = _usd_authored_asset_dependencies,
    mdl_dependency_reader: Callable[[Path], Sequence[str]] = _mdl_dependencies,
    dependency_expander: Callable[[str], Sequence[str]] = _default_expand_dependency,
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

    asset_queue: deque[tuple[str, str]] = deque()
    for relative in sorted(traversed):
        if relative.lower().endswith((".usd", ".usda", ".usdc")):
            for reference in asset_dependency_reader(root / relative):
                asset_queue.append((relative, str(reference)))
    asset_dependency_sources: set[str] = set()
    runtime_asset_relocations: list[dict[str, str]] = []
    while asset_queue:
        owner, reference = asset_queue.popleft()
        kind, resolved = _resolve_authored_asset(owner, reference)
        if kind == "external":
            mirror = _dataset_mirror_for_external_asset(owner, resolved)
            if mirror is not None:
                ensure(mirror)
                replacement = posixpath.relpath(mirror, posixpath.dirname(owner))
                if not replacement.startswith(("./", "../")):
                    replacement = "./" + replacement
                runtime_asset_relocations.append(
                    {
                        "owner_relative_path": owner,
                        "source_asset_uri": resolved,
                        "replacement_relative_path": mirror,
                        "replacement_authored_path": replacement,
                    }
                )
                continue
            external_dependencies.add(resolved)
            continue
        expanded = tuple(dependency_expander(resolved))
        if not expanded:
            raise ValueError(f"nvidia_warehouse_dependency_expansion_empty:{resolved}")
        for candidate in expanded:
            safe = _safe_relative(candidate)
            path = ensure(safe)
            if safe.lower().endswith(".mdl") and safe not in asset_dependency_sources:
                asset_dependency_sources.add(safe)
                for nested in mdl_dependency_reader(path):
                    asset_queue.append((safe, str(nested)))

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
        "dependency_contract": dict(DEPENDENCY_CONTRACT),
        "runtime_asset_relocations": sorted(
            runtime_asset_relocations,
            key=lambda row: (row["owner_relative_path"], row["source_asset_uri"]),
        ),
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
    *,
    materialization_manifest_path: str | Path,
    output_path: str | Path,
    include_ctrl_world_external_2: bool = False,
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
    if manifest.get("dependency_contract") != DEPENDENCY_CONTRACT:
        raise ValueError("nvidia_warehouse_materialization_dependency_contract_invalid")
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
                # The previous [0.0, 0.075, 1.085] placement was wholly
                # embedded in the workcell's existing clay bottle. This
                # collision-free pocket is selected from pinned scene AABBs,
                # before policy/WAM execution or access to any ranking.
                "spraycan_translation_m": [0.10, -0.05, 1.005],
                "tray_center_translation_m": [-0.05, 0.32, 1.025],
                "initial_target_allowed_enclosing_prim_paths": [
                    "/World/WarehouseWorkcell/SM_Crate_A07_Yellow_04"
                ],
                "source": (
                    "deterministic_pinned_scene_aabb_clearance_inside_declared_open_crate"
                ),
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
                "rigid_mount_orientation": {
                    "mode": "one_time_initial_task_framing_rigid_parent_local_mount",
                    # The wrist validity contract requires the manipulated
                    # object at the initial pose. Aiming at a target/tray
                    # midpoint can center neither and needlessly weakens that
                    # observation without changing the task.
                    "target_entity_ids": ["spraycan"],
                    "world_up": [0.0, 0.0, 1.0],
                    "calibrated_before_initial_observation": True,
                    "calibrated_after_initial_joint_hold": True,
                    "per_frame_task_reaim": False,
                },
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
            "spraycan_initial_placement_has_no_undeclared_workcell_aabb_intersection",
            "external_rgb_nonblank_and_franka_spraycan_tray_visible",
            "wrist_rgb_nonblank_and_task_object_visible_at_initial_pose",
            "wrist_camera_world_pose_changes_under_command",
            "wrist_camera_to_panda_hand_transform_remains_constant",
            "franka_joint_states_are_rendered_kinematically_without_physics_advance",
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
    if include_ctrl_world_external_2:
        spec["cameras"]["external_2"] = {
            "resolution": [640, 480],
            "position_m": [-0.85, -0.65, 1.95],
            "look_at_m": [-0.05, 0.18, 1.14],
            "vertical_fov_deg": 52.0,
            "fixed_in_world": True,
        }
        spec["required_views"] = ["external", "external_2", "wrist"]
        spec["required_checks"].remove("external_and_wrist_timestamps_match_physics_steps")
        spec["required_checks"].extend(
            [
                "external_2_rgb_nonblank_and_franka_spraycan_tray_visible",
                "external_and_external_2_frames_are_distinct",
                "external_external_2_and_wrist_timestamps_match_physics_steps",
            ]
        )
    spec["spec_sha256"] = canonical_sha256(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, spec)
    return spec


__all__ = [
    "CANARY_SPEC_SCHEMA_VERSION",
    "DATASET_REVISION",
    "DEPENDENCY_CONTRACT",
    "PINNED_SHA256",
    "SCHEMA_VERSION",
    "build_native_camera_canary_spec",
    "materialize_pinned_workcell",
]
