"""Share exact supplemental-destination task semantics across preparation and execution."""
from __future__ import annotations

from collections.abc import Mapping
import json
import hashlib
import re
from pathlib import Path
from typing import Any

DESTINATION_FIELDS = ("asset", "rights_admission", "static_qualification", "native_import_qualification",
                      "geometry", "placement_qualification")
BUNDLE_CONTRACT = "scene.configured_revision.configured_scene_bundle"


def destination_references(revision: Mapping[str, Any]) -> dict[str, Any]:
    destination = revision["task_template"].get("destination")
    if destination is None:
        return {}
    return {"task.destination." + name: destination[name] for name in DESTINATION_FIELDS if name in destination}


def bind_destination_task_spec(task_spec: Mapping[str, Any], destination: Mapping[str, Any], *,
                               source_support: Mapping[str, Any], support_plane_digest: str) -> dict[str, Any]:
    """Bind the same fitted region, release pose and initial support as the native compiler."""
    from .task_evaluation_rigid_destination_geometry import bind_destination_trajectory

    spec = dict(task_spec)
    spec["initial_source_support"] = {"scene_prim_paths": [source_support["sage_prim_path"]],
        "support_plane_digest": support_plane_digest,
        "contact_permission": "initial_pickup_until_first_separation_or_lift"}
    spec = bind_destination_trajectory(spec, destination)
    spec.update(destination_relation=destination["relation"], destination_support_asset_id=destination["asset_id"],
        destination_placement_support_prim_paths=destination["placement_support_scene_prim_paths"])
    for field in ("destination_position_bounds_destination_frame_m", "subject_collision_bounds_scoring_frame_m",
                  "destination_interior_bounds_body_frame_m", "destination_pose_world",
                  "destination_reset_translation_tolerance_m", "destination_reset_rotation_tolerance_rad"):
        spec[field] = destination[field]
    return spec


def destination_binding_from_geometry(destination: Mapping[str, Any], geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Candidate binding for a construction probe before a full-scene placement receipt exists."""
    from .task_evaluation_rigid_destination_geometry import destination_trajectory_geometry

    limits = destination["native_probe"]["qualification_limits"]
    runtime_id = re.sub(r"[^A-Za-z0-9_]", "_", str(destination["identity"]["id"]))
    if not runtime_id or not runtime_id.replace("_", "a").isalnum():
        raise ValueError("episode_compiler_runtime_asset_id_invalid:destination")
    pose = destination["pose_world"]
    return {**destination_trajectory_geometry(destination, geometry),
        "asset_id": runtime_id, "relation": destination["relation"],
        "placement_support_scene_prim_paths": destination["native_probe"]["placement_support_scene_prim_paths"],
        "destination_position_bounds_destination_frame_m": geometry["destination_position_bounds_destination_frame_m"],
        "destination_interior_bounds_body_frame_m": geometry["destination_interior_bounds_body_frame_m"],
        "destination_pose_world": [*pose["position_world_m"], *pose["orientation_xyzw"]],
        "destination_reset_translation_tolerance_m": limits["reset_translation_tolerance_m"],
        "destination_reset_rotation_tolerance_rad": limits["reset_rotation_tolerance_rad"]}


def stage_revision_destination(*, revision: Mapping[str, Any], documents: Mapping[str, Path],
                               task_spec: Mapping[str, Any], collision_path: Path, output_root: Path,
                               require_placement: bool,
                               placement_qualification_path: Path | None = None,
                               expected_execution_binding: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Reuse native qualification validation over exact published files.

    A robot-free check can use the declared native placement-probe limits;
    policy execution still requires the independently recorded full-scene
    placement qualification.
    """
    from . import task_evaluation_configured_controls_deferred_inputs as inputs
    from .task_evaluation_native_arena_episode_compiler import _stage_destination_asset

    destination = revision["task_template"].get("destination")
    if destination is None:
        return None
    destination = dict(destination)
    documents = dict(documents)
    if placement_qualification_path is not None:
        path = Path(placement_qualification_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("team_scene_destination_placement_qualification_invalid")
        payload = path.read_bytes()
        destination["placement_qualification"] = {"uri": "local:destination-placement-qualification",
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        documents["task.destination.placement_qualification"] = path
    if require_placement and "placement_qualification" not in destination:
        raise ValueError("team_scene_destination_placement_qualification_required")
    qualified = require_placement
    if qualified and (not isinstance(expected_execution_binding, Mapping)
            or not re.fullmatch(r"[0-9a-f]{40}", str(expected_execution_binding.get("execution_commit") or ""))
            or not isinstance(expected_execution_binding.get("runtime_identity"), Mapping)
            or set(expected_execution_binding["runtime_identity"]) != {"id", "version"}
            or any(not isinstance(expected_execution_binding["runtime_identity"].get(key), str)
                or not expected_execution_binding["runtime_identity"][key].strip() for key in ("id", "version"))):
        raise ValueError("team_scene_destination_execution_binding_required")
    references = inputs._materialized_references(revision=revision, documents=documents)
    declared = {"task.destination." + key: destination[key] for key in DESTINATION_FIELDS if key in destination}
    for name, reference in declared.items():
        local = Path(documents[name])
        if not inputs._retained_matches(local, reference=reference):
            raise ValueError("rigid_destination_published_reference_changed:" + name)
        references[name] = {**dict(reference), "contract_path": name, "materialized_path": str(local),
                            "full_byte_service_account_readback_passed": True}
    expected = expected_execution_binding or {}
    request = {"task": {"strategy": "pick_and_place", "destination": destination,
        "subject": {"identity": revision["replacement"]["identity"]},
        "configured_scene_revision_digest": revision["revision_digest"]},
        "expected_production_commit": expected.get("execution_commit"),
        "runtime": {"identity": expected.get("runtime_identity")}}
    return _stage_destination_asset(request=request, materialized_references=references, output_root=output_root,
        configured_collision_path=collision_path, task_spec=task_spec, qualification_only=not qualified)


def probe_destination_metadata(destination: Mapping[str, Any], documents: Mapping[str, Path]) -> dict[str, Any]:
    """The qualified asset and source body identities the robot-free simulator imports."""
    static = json.loads(Path(documents["task.destination.static_qualification"]).read_text())
    return {"asset_path": str(documents["task.destination.asset"]), "asset_digest": destination["asset_digest"],
        "pose_world": destination["destination_pose_world"],
        "intended_support_prim_paths": destination["intended_support_prim_paths"],
        "collision_prim_paths": static["observed_structure"].get("collision_prim_paths", [])}
