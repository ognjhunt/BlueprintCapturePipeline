"""Compile a current beta task into the pinned G1 body without ADP authoring.

This is CPU preparation from retained scene geometry. Stance and camera setup
are disclosed simulation assumptions, not capture measurements or task success.
No model, network, simulator, allocation or human-review action occurs here.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import SOURCE_FIELDS

SCHEMA = "native_g1_beta_scene_preparation.v1"
FILENAME = SCHEMA + ".json"
RECIPE = "native_g1_beta_retained_scene.v1"
# The hand the episode carries the object with (a simulation assumption); its stance is chosen for it.
TASK_HAND = "right"
HEAD_CAMERA = {
    "role": "head", "policy_input": True, "review_only": False, "scoring_input": False,
    "pose_frame": "robot_body", "parent_prim_path": "{ENV_REGEX_NS}/Robot/head_link",
    "optical_convention": "opencv",
    "frame_from_camera_matrix": [0.0, -math.sqrt(0.5), math.sqrt(0.5), 0.08,
        -1.0, 0.0, 0.0, 0.0, 0.0, -math.sqrt(0.5), -math.sqrt(0.5), 0.04, 0.0, 0.0, 0.0, 1.0],
    "intrinsics": {"width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 319.5, "cy": 239.5},
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(path: Path) -> dict:
    from .beta_policy_screening_launch import _record as record
    return record(path)


def _bound(record: Mapping[str, Any]) -> Path:
    from .beta_policy_screening_launch import _bound_path
    path = _bound_path(record)
    if path != path.resolve():
        raise ValueError("g1_beta_preparation_symlink_refused")
    return path


def _copy(source: Path, destination: Path, *, sha256: str, size_bytes: int) -> None:
    def matches(path):
        return (not path.is_symlink() and path.is_file() and path.stat().st_size == size_bytes
                and "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == sha256)
    if not matches(source):
        raise ValueError("g1_beta_preparation_source_changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if matches(destination):
        return
    if destination.is_symlink() or destination.exists() and not destination.is_file():
        raise ValueError("g1_beta_preparation_source_changed")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="." + destination.name + ".copy-", delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(source, temporary)
        if not matches(temporary):
            raise ValueError("g1_beta_preparation_source_changed")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if matches(destination):
            return
        if destination.exists():
            # An older interrupted copier could leave an invalid final file.
            # Preserve those bytes; never replace a verified retained file.
            _preserve_incomplete(destination)
        os.link(temporary, destination)  # Exclusive publish, never overwrite.
        _sync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preserve_incomplete(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("g1_beta_preparation_symlink_refused")
    retained = Path(tempfile.mkdtemp(prefix="." + path.name + ".incomplete-", dir=path.parent))
    os.rename(path, retained / path.name)
    _sync_directory(retained)
    _sync_directory(path.parent)


@contextmanager
def _namespace_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("g1_beta_preparation_lock_invalid")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("g1_beta_preparation_in_progress") from exc
        yield
    finally:
        os.close(descriptor)  # Process exit releases this lock, including SIGKILL.


def _request(*, base_root: Path, base_receipt: Mapping[str, Any], robot_asset: Path,
             evidence_root: Path, source: Mapping[str, Any], site_id: str,
             task_success_contract: Mapping[str, Any]) -> tuple[dict, dict]:
    from .adp_task_scoring import validate_rigid_task_success_contract
    from .native_g1_arena_robot_configuration import build_pinned_g1_arena_robot_configuration
    from .native_task_robot_contact_topology import _asset_rigid_body_paths
    from .native_g1_scene_reach_probe import STANCE_RULE, g1_stance_static_reach
    from .scene_placement.robot_profile import UNITREE_G1_PROFILE
    from .team_embodiment_stance import GRID_STEP_M, SEARCH_RADIUS_M, _scene_collision, ground_stance

    original = _read(base_root / "native_task_arena_packet_request.v1.json")
    plan = _read(base_root / "native_task_arena_scene_plan.v1.json")
    if (original.get("scene_id") != site_id or original.get("task_id") != source["task_id"]
            or original.get("task_spec", {}).get("task_kind") != "rigid_pick_place"
            or plan.get("plan_digest") != base_receipt["arena_scene_plan_digest"]):
        raise ValueError("g1_beta_preparation_task_binding_mismatch")
    task = deepcopy(original["task_spec"])
    contract = validate_rigid_task_success_contract(task_success_contract, expected_task_id=source["task_id"])
    if any(document.get("task_success_contract") not in (None, contract) for document in (task, plan["task_spec"])):
        raise ValueError("g1_beta_preparation_task_contract_changed")
    task.update(task_success_contract=contract, task_success_contract_digest=contract["contract_digest"])
    selected = ((original.get("configured_task_template_adapter") or {}).get("source_documents") or {}).get("documents", {}).get("source_object")
    if selected:
        bounds = {"minimum": selected["aabb_min_xyz_m"], "maximum": selected["aabb_max_xyz_m"]}
        bounds_basis = "retained_source_object_selection_world_aabb"
    else:
        local = task.get("subject_collision_bounds_scoring_frame_m") or {}
        start = task["start_pose_world"]
        if start[3:] != [0.0, 0.0, 0.0, 1.0]:
            raise ValueError("g1_beta_preparation_subject_bounds_unavailable")
        bounds = {key: [float(start[i]) + float(local[key][i]) for i in range(3)] for key in ("minimum", "maximum")}
        bounds_basis = "retained_unrotated_task_scoring_bounds"
    collision, collision_digest, offset = _scene_collision(plan, base_root)
    # The closest clear stance from which the G1's own nominal arm span covers the pick and the place;
    # with none, the closest, flagged. Advisory: reach refuses nothing here.
    stances = ground_stance(collision_path=collision, collision_sha256=collision_digest, offset=offset,
        subject_center=list(task["start_pose_world"][:3]), subject_minimum=bounds["minimum"], subject_maximum=bounds["maximum"],
        footprint_half_extent_m=list(UNITREE_G1_PROFILE.footprint_half_extent_xyz[:2]), root_height_m=UNITREE_G1_PROFILE.pelvis_height_m,
        retain_large_collision_bounds=True, ranked=(2 * math.floor(SEARCH_RADIUS_M / GRID_STEP_M + 1e-9) + 1) ** 2)
    try:
        chosen, stance_reach = g1_stance_static_reach(stances, start=task["start_pose_world"][:3],
                                                      target=task["target_position_world_m"], task_hand=TASK_HAND)
    except (KeyError, TypeError, ValueError):
        chosen, stance_reach = 0, {"rule": STANCE_RULE, "status": "unassessed",
                                   "reason": "g1_stance_reach_inputs_invalid"}
    stance = stances[chosen]
    pose = {"position_world_m": stance["position_world_m"], "orientation_xyzw": stance["orientation_xyzw"]}
    robot = build_pinned_g1_arena_robot_configuration(asset=robot_asset, evidence_root=evidence_root,
        base_pose_world=pose, task_hand=TASK_HAND)
    if HEAD_CAMERA["parent_prim_path"] not in _asset_rigid_body_paths(robot_asset):
        raise ValueError("g1_beta_preparation_head_body_missing")
    overview = [row for row in original.get("cameras", []) if row.get("role") == "overview"]
    if len(overview) != 1 or overview[0].get("pose_frame") != "world" or overview[0].get("policy_input"):
        raise ValueError("g1_beta_preparation_overview_missing")
    duration = min(float(task["maximum_episode_seconds"]),
                   int(task["maximum_action_steps"]) / float(task["control_frequency_hz"]))
    if not math.isfinite(duration) or duration < 0.02:
        raise ValueError("g1_beta_preparation_episode_budget_invalid")
    task.update(control_frequency_hz=50.0, maximum_action_steps=math.floor(duration * 50),
                maximum_episode_seconds=duration)
    by_role = {row["semantic_role"]: row for row in base_receipt["source_bindings"]}
    assets = []
    for row in original["assets"]:
        binding = by_role[row["semantic_role"]]
        destination = evidence_root / "scene" / row["filename"]
        _copy(base_root / binding["staged_relative_path"], destination,
              sha256=binding["staged_sha256"], size_bytes=binding["staged_size_bytes"])
        assets.append({**deepcopy(row), "source": {"root": "evidence", "relative_path": "scene/" + row["filename"],
            "size_bytes": binding["staged_size_bytes"], "sha256": binding["staged_sha256"]}})
    assumptions = {"recipe_id": RECIPE, "current_task_source": dict(source), "site_id": site_id,
        "source_packet_receipt_digest": base_receipt["receipt_digest"],
        "source_scene_plan_digest": base_receipt["arena_scene_plan_digest"],
        "robot_usd_sha256": robot["usd_sha256"], "stance": stance, "stance_reach": stance_reach,
        "subject_bounds_world_m": bounds, "subject_bounds_basis": bounds_basis,
        "task_success_contract_digest": contract["contract_digest"],
        "stance_method": "existing_scene_collision_floor_and_clear_footprint_search",
        "large_collision_bounds_retained": True,
        "stance_is_geometric_estimate": True, "measured_robot_calibration_claimed": False,
        "footprint_half_extent_m": list(UNITREE_G1_PROFILE.footprint_half_extent_xyz[:2]),
        "root_height_above_inferred_floor_m": UNITREE_G1_PROFILE.pelvis_height_m,
        "head_camera": deepcopy(HEAD_CAMERA), "head_camera_calibration_measured": False,
        "head_camera_basis": "existing_g1_development_mount_simulation_assumption",
        "overview_basis": "unchanged_current_task_review_camera",
        "task_hand": TASK_HAND, "task_hand_is_simulation_assumption": True,
        "control_frequency_hz": 50.0, "physics_frequency_hz": 200.0,
        "source_control_frequency_hz": original["task_spec"]["control_frequency_hz"],
        "source_maximum_action_steps": original["task_spec"]["maximum_action_steps"],
        "source_maximum_episode_seconds": original["task_spec"]["maximum_episode_seconds"],
        "maximum_action_steps": task["maximum_action_steps"], "maximum_episode_seconds": duration,
        "navigation_goal_added": False, "source_geometry_changed": False,
        "policy_queried": False, "simulator_executed": False, "physical_outcome_claimed": False,
        "new_human_review_claimed": False, "current_consent_revalidated": False}
    assumptions["assumptions_digest"] = canonical_digest(assumptions, digest_field="assumptions_digest")
    # Retain all task/source provenance, replacing only robot-dependent setup.
    request = deepcopy(original)
    request.pop("policy_canary_camera_start_configuration", None)
    request.update(assets=assets, robot_configuration=robot, robot_base_pose_world=pose,
        robot_joint_reset_positions_rad=robot["joint_reset_positions_rad"], task_spec=task,
        cameras=[deepcopy(HEAD_CAMERA), deepcopy(overview[0])], physics_frequency_hz=200.0,
        g1_scene_derivation={"schema_version": "native_g1_beta_scene_packet_derivation.v1",
            "claim_ceiling": "development_only", "source_packet_receipt_digest": base_receipt["receipt_digest"],
            "source_scene_plan_digest": base_receipt["arena_scene_plan_digest"],
            "assumptions": assumptions})
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request, assumptions


def prepare_native_g1_beta_packet(*, base_packet_record: Mapping[str, Any], source: Mapping[str, Any],
                                  site_id: str, output_root: Path, robot_asset_record: Mapping[str, Any],
                                  task_success_contract: Mapping[str, Any]) -> dict:
    """Prepare under `<output_root>/native-g1/<base-digest>/<source-digest>/`."""
    from .native_g1_arena_robot_configuration import G1_USD_SHA256, G1_USD_SIZE_BYTES
    from .native_task_arena_bundle import verify_native_task_arena_packet
    from .native_task_arena_packet import materialize_native_task_arena_packet
    from .deployment_embodiment_preparation_contract import write_once

    if set(source) != set(SOURCE_FIELDS):
        raise ValueError("g1_beta_preparation_source_invalid")
    base_root, receipt, _ = verify_native_task_arena_packet(_bound(base_packet_record).parent)
    robot_source = _bound(robot_asset_record)
    if robot_asset_record["sha256"] != "sha256:" + G1_USD_SHA256 or robot_asset_record["size_bytes"] != G1_USD_SIZE_BYTES:
        raise ValueError("g1_beta_robot_asset_mismatch")
    root = Path(output_root).absolute()
    parent = root / "native-g1" / receipt["receipt_digest"][7:]
    source_key = canonical_digest(source)[7:]
    destination = parent / source_key
    evidence = parent / ("." + source_key + ".inputs")
    for path in (destination, evidence):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("g1_beta_preparation_symlink_refused")
    with _namespace_lock(parent / ("." + source_key + ".lock")):
        if destination.exists() and not (destination / "native_task_arena_packet_receipt.v1.json").exists():
            if not destination.is_dir():
                raise ValueError("g1_beta_preparation_existing_packet_conflict")
            # A killed packet compiler has no complete receipt. Keep its
            # partial bytes outside the final namespace and retry the same
            # final path, so portable and embedded source paths stay stable.
            _preserve_incomplete(destination)
        robot_asset = evidence / "robot" / robot_source.name
        _copy(robot_source, robot_asset, sha256=robot_asset_record["sha256"], size_bytes=robot_asset_record["size_bytes"])
        request, assumptions = _request(base_root=base_root, base_receipt=receipt, robot_asset=robot_asset,
            evidence_root=evidence, source=source, site_id=site_id, task_success_contract=task_success_contract)
        if destination.exists():
            _, prior, _ = verify_native_task_arena_packet(destination)
            if _read(destination / "native_task_arena_packet_request.v1.json") != request:
                raise ValueError("g1_beta_preparation_existing_packet_conflict")
        else:
            prior = materialize_native_task_arena_packet(request=request, evidence_root=evidence, output_dir=destination)
            verify_native_task_arena_packet(destination)
        packet = _record(destination / "native_task_arena_packet_receipt.v1.json")
        result = {"schema_version": SCHEMA, "status": "prepared_not_executed", "recipe_id": RECIPE,
            "source": dict(source), "site_id": site_id, "base_packet": dict(base_packet_record),
            "scene_packet": packet, "assumptions": assumptions, "scene_plan_digest": prior["arena_scene_plan_digest"],
            "candidate_reasons": {key: ["g1_beta_current_navigation_goal_authority_missing"] for key in (
                "humanoidarena_dp_g1_dex3_sonic_vision_navi", "humanoidarena_pi05_g1_dex3_sonic_vision_navi")},
            "candidate_policy_queried": False, "simulator_executed": False, "provider_allocations_performed": 0,
            "new_human_review_claimed": False, "current_consent_revalidated": False,
            "claim_ceiling": "development_only"}
        result["preparation_digest"] = canonical_digest(result, digest_field="preparation_digest")
        write_once(destination / FILENAME, result)
        return {**result, "preparation_receipt": _record(destination / FILENAME)}
