"""Compile a current beta task into the pinned G1 body without ADP authoring.

This is CPU preparation from retained scene geometry. Stance and camera setup
are disclosed simulation assumptions, not capture measurements or task success.
The whole body at the chosen stance is measured against the room on CPU
(``native_g1_spawn_clearance``), a clearing stance replaces one that overlaps,
and the report is kept beside the packet for planning to read.
A navigation base packet (``goal_region_navigation``, the G1 at the compiled start) keeps that start as
its stance (``_navigation_request``), and the body is also measured at each start offset its cells use
(``native_g1_spawn_clearance.START_OFFSETS_FILENAME``); the rigid request is unchanged (``_request``).
A scene floor that is a development-only estimate, as the packet's task carries it
(``native_g1_spawn_clearance.scene_floor_estimate``), is a disclosed assumption of the receipt
(``scene_floor_estimate``), with its band spread; planning flags the row for it, never refuses it.
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
from .usd_binary_read_cache import read_scoped

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


def g1_stance_search(*, collision_path: Path, collision_sha256: str, offset: list[float], start: Any,
                     target: Any, bounds: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """The G1's clear floor stances around the task, closest first, and the one its arm span picks.

    Its own footprint and pelvis height on the scene's floor (``ground_stance``, keeping long collision
    fixtures). The chosen one is the closest from which its nominal arm span covers the pick and the
    place; with none, the closest, flagged. Advisory: reach refuses nothing here. With no clear stance
    the search raises by name (``TeamEmbodimentError``). The website robot fit precheck searches the
    same way before any spend (``website_robot_fit_precheck``).
    """
    from .native_g1_scene_reach_probe import STANCE_RULE, g1_stance_static_reach
    from .scene_placement.robot_profile import UNITREE_G1_PROFILE
    from .team_embodiment_stance import GRID_STEP_M, SEARCH_RADIUS_M, ground_stance

    stances = ground_stance(collision_path=collision_path, collision_sha256=collision_sha256, offset=offset,
        subject_center=list(start[:3]), subject_minimum=bounds["minimum"], subject_maximum=bounds["maximum"],
        footprint_half_extent_m=list(UNITREE_G1_PROFILE.footprint_half_extent_xyz[:2]), root_height_m=UNITREE_G1_PROFILE.pelvis_height_m,
        retain_large_collision_bounds=True, ranked=(2 * math.floor(SEARCH_RADIUS_M / GRID_STEP_M + 1e-9) + 1) ** 2)
    try:
        chosen, stance_reach = g1_stance_static_reach(stances, start=start[:3], target=target, task_hand=TASK_HAND)
    except (KeyError, TypeError, ValueError):
        chosen, stance_reach = 0, {"rule": STANCE_RULE, "status": "unassessed",
                                   "reason": "g1_stance_reach_inputs_invalid"}
    return stances, chosen, stance_reach


def g1_spawn_stance(stances: list[dict[str, Any]], chosen: int, stance_reach: dict[str, Any], *,
                    robot_asset: Path, joint_reset_positions: Mapping[str, float], collision_path: Path,
                    offset: list[float], start: Any, target: Any) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """The searched stance whose whole body starts clear of the room (``native_g1_spawn_clearance``).

    The reach-chosen one unless its body overlaps the scene collision; then the first other stance that
    clears, those in nominal arm span first, each in the search's order; with none, the chosen one and
    its overlap. Returns the stance index, its reach record and the measurement.
    """
    from .native_g1_scene_reach_probe import g1_stance_static_reach
    from .native_g1_spawn_clearance import choose_spawn_stance

    reach: dict[int, dict[str, Any] | None] = {}

    def reach_at(index: int) -> dict[str, Any] | None:
        if index not in reach:
            try:
                _, record = g1_stance_static_reach([stances[index]], start=start, target=target, task_hand=TASK_HAND)
                reach[index] = {**record, "clear_stances_measured": len(stances), "chosen_rank": index}
            except (KeyError, TypeError, ValueError):
                reach[index] = None
        return reach[index]

    order = sorted((index for index in range(len(stances)) if index != chosen),
                   key=lambda index: not (reach_at(index) or {}).get("within_nominal_arm_span"))
    used, clearance = choose_spawn_stance(robot_usd=robot_asset, joint_reset_positions=joint_reset_positions,
        stances=stances, chosen=chosen, fallback_order=order, background_usd=collision_path,
        background_pose_world={"position_world_m": offset, "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]})
    return used, (stance_reach if used == chosen else reach_at(used) or stance_reach), clearance


def _request(*, base_root: Path, base_receipt: Mapping[str, Any], robot_asset: Path,
             evidence_root: Path, source: Mapping[str, Any], site_id: str,
             task_success_contract: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    from .adp_task_scoring import validate_rigid_task_success_contract
    from .native_g1_arena_robot_configuration import build_pinned_g1_arena_robot_configuration
    from .native_task_robot_contact_topology import _asset_rigid_body_paths
    from .scene_placement.robot_profile import UNITREE_G1_PROFILE
    from .team_embodiment_stance import _scene_collision

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
    stances, chosen, stance_reach = g1_stance_search(
        collision_path=collision, collision_sha256=collision_digest, offset=offset,
        start=task["start_pose_world"][:3], target=task["target_position_world_m"], bounds=bounds)

    def pose_of(stance):
        return {"position_world_m": stance["position_world_m"], "orientation_xyzw": stance["orientation_xyzw"]}

    robot = build_pinned_g1_arena_robot_configuration(asset=robot_asset, evidence_root=evidence_root,
        base_pose_world=pose_of(stances[chosen]), task_hand=TASK_HAND)
    # The footprint search clears the floor only; the whole body at the reset pose is measured here.
    used, stance_reach, clearance = g1_spawn_stance(stances, chosen, stance_reach, robot_asset=robot_asset,
        joint_reset_positions=robot["joint_reset_positions_rad"], collision_path=collision, offset=offset,
        start=task["start_pose_world"][:3], target=task["target_position_world_m"])
    stance = stances[used]
    pose = pose_of(stance)
    if used != chosen:
        robot = build_pinned_g1_arena_robot_configuration(asset=robot_asset, evidence_root=evidence_root,
            base_pose_world=pose, task_hand=TASK_HAND)
    from .native_g1_spawn_clearance import binding
    clearance.update(binding(robot_usd_sha256=robot["usd_sha256"], scene_collision_sha256=collision_digest,
        base_pose_world=pose, joint_reset_positions=robot["joint_reset_positions_rad"]))
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
    return request, assumptions, clearance


#: Where a navigation stance comes from: the base packet's own robot pose, the compiled start.
NAVIGATION_STANCE_METHOD = "compiled_start_pose"


def _navigation_fallbacks(*, collision: Path, collision_sha256: str, offset: list[float],
                          start: list[float]) -> tuple[list[dict[str, Any]], list[str]]:
    """The G1's clear floor stances around the compiled start, closest first, or none and the search's reasons.

    ``ground_stance`` around the start point itself: a navigation task has no subject, so the point stands in
    for one, with any standoff from it allowed, and its stances are ranked by their distance to the start.
    Each faces the start point, as ``ground_stance`` places a stance.
    """
    from .native_g1_spawn_clearance import MAXIMUM_FALLBACK_STANCES
    from .scene_placement.robot_profile import UNITREE_G1_PROFILE
    from .team_embodiment import TeamEmbodimentError
    from .team_embodiment_stance import SEARCH_RADIUS_M, ground_stance

    point = [float(value) for value in start[:3]]
    try:
        stances = ground_stance(collision_path=collision, collision_sha256=collision_sha256, offset=offset,
            subject_center=point, subject_minimum=point, subject_maximum=point,
            footprint_half_extent_m=list(UNITREE_G1_PROFILE.footprint_half_extent_xyz[:2]),
            root_height_m=UNITREE_G1_PROFILE.pelvis_height_m, retain_large_collision_bounds=True,
            standoff_range_m=(0.0, SEARCH_RADIUS_M), ranked=MAXIMUM_FALLBACK_STANCES)
    except TeamEmbodimentError as exc:
        return [], list(exc.blockers)
    return list(stances), []


def _navigation_request(*, base_root: Path, base_receipt: Mapping[str, Any], robot_asset: Path,
                        evidence_root: Path, source: Mapping[str, Any], site_id: str,
                        task_success_contract: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    """The G1 request for a navigation base packet: the G1 at the compiled start, the task kept as compiled.

    The stance is the base packet's own robot pose, the compiled start (``website_navigation_scene_inputs``),
    never an object-centred ``ground_stance``. Its whole body is measured there; only if it overlaps the room
    is a fallback measured: the clear floor stances around the start, closest first (``_navigation_fallbacks``),
    through ``choose_spawn_stance``. No subject bounds exist and no goal is added. The spawn check at each
    start offset is measured from the packet once it exists (``native_g1_spawn_clearance.packet_start_offsets``).
    """
    from .native_g1_arena_robot_configuration import build_pinned_g1_arena_robot_configuration
    from .native_g1_navigation_goal import NAVIGATION_TASK_KIND, validate_navigation_task_success_contract
    from .native_g1_spawn_clearance import binding, choose_spawn_stance
    from .native_task_robot_contact_topology import _asset_rigid_body_paths
    from .scene_placement.robot_profile import UNITREE_G1_PROFILE
    from .team_embodiment_stance import _scene_collision

    original = _read(base_root / "native_task_arena_packet_request.v1.json")
    plan = _read(base_root / "native_task_arena_scene_plan.v1.json")
    if (original.get("scene_id") != site_id or original.get("task_id") != source["task_id"]
            or original.get("task_spec", {}).get("task_kind") != NAVIGATION_TASK_KIND
            or plan.get("task_kind") != NAVIGATION_TASK_KIND
            or plan.get("plan_digest") != base_receipt["arena_scene_plan_digest"]):
        raise ValueError("g1_beta_preparation_task_binding_mismatch")
    task = deepcopy(original["task_spec"])
    try:
        contract = validate_navigation_task_success_contract(task_success_contract, task_spec=task)
    except ValueError as exc:
        raise ValueError("g1_beta_preparation_task_contract_changed") from exc
    if any(document.get("task_success_contract") not in (None, contract) for document in (task, plan["task_spec"])):
        raise ValueError("g1_beta_preparation_task_contract_changed")
    task.update(task_success_contract=contract, task_success_contract_digest=contract["contract_digest"])
    start, compiled = plan["robot"]["base_pose_world"], [float(value) for value in task["start_pose_world"]]
    if ([float(value) for value in start["position_world_m"][:2]] != compiled[:2]
            or [float(value) for value in start["orientation_xyzw"]] != compiled[3:]):
        raise ValueError("g1_beta_preparation_compiled_start_mismatch")
    compiled_start = {"source": NAVIGATION_STANCE_METHOD,
                      "position_world_m": [float(value) for value in start["position_world_m"]],
                      "orientation_xyzw": compiled[3:], "start_pose_world": compiled}
    collision, collision_digest, offset = _scene_collision(plan, base_root)

    def pose_of(stance):
        return {"position_world_m": stance["position_world_m"], "orientation_xyzw": stance["orientation_xyzw"]}

    robot = build_pinned_g1_arena_robot_configuration(asset=robot_asset, evidence_root=evidence_root,
        base_pose_world=pose_of(compiled_start), task_hand=TASK_HAND)
    measure = {"robot_usd": robot_asset, "joint_reset_positions": robot["joint_reset_positions_rad"],
               "background_usd": collision,
               "background_pose_world": {"position_world_m": offset, "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}}
    stances, refusals = [compiled_start], []
    used, clearance = choose_spawn_stance(stances=stances, chosen=0, fallback_order=[], **measure)
    if clearance["status"] == "overlap":
        # Only a body inside the room at the start is replaced: the clear floor stances around it, closest first.
        fallbacks, refusals = _navigation_fallbacks(collision=collision, collision_sha256=collision_digest,
                                                    offset=offset, start=compiled)
        stances = [compiled_start, *fallbacks]
        used, clearance = choose_spawn_stance(stances=stances, chosen=0,
                                              fallback_order=list(range(1, len(stances))), **measure)
    clearance["stance_selection"] = {**clearance["stance_selection"], "source": NAVIGATION_STANCE_METHOD,
                                     "fallback_search_refusals": refusals}
    stance = stances[used]
    pose = pose_of(stance)
    if used != 0:
        robot = build_pinned_g1_arena_robot_configuration(asset=robot_asset, evidence_root=evidence_root,
            base_pose_world=pose, task_hand=TASK_HAND)
    clearance.update(binding(robot_usd_sha256=robot["usd_sha256"], scene_collision_sha256=collision_digest,
        base_pose_world=pose, joint_reset_positions=robot["joint_reset_positions_rad"]))
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
        binding_row = by_role[row["semantic_role"]]
        destination = evidence_root / "scene" / row["filename"]
        _copy(base_root / binding_row["staged_relative_path"], destination,
              sha256=binding_row["staged_sha256"], size_bytes=binding_row["staged_size_bytes"])
        assets.append({**deepcopy(row), "source": {"root": "evidence", "relative_path": "scene/" + row["filename"],
            "size_bytes": binding_row["staged_size_bytes"], "sha256": binding_row["staged_sha256"]}})
    assumptions = {"recipe_id": RECIPE, "current_task_source": dict(source), "site_id": site_id,
        "source_packet_receipt_digest": base_receipt["receipt_digest"],
        "source_scene_plan_digest": base_receipt["arena_scene_plan_digest"],
        "source_task_spec_digest": original["task_spec"]["task_spec_digest"],
        "robot_usd_sha256": robot["usd_sha256"], "stance": stance,
        "subject_bounds_world_m": None, "subject_bounds_basis": None,
        "task_success_contract_digest": contract["contract_digest"],
        "stance_method": NAVIGATION_STANCE_METHOD, "large_collision_bounds_retained": True,
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
    # Retain all task/source provenance (the navigation derivation included), replacing only robot-dependent setup.
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
    return request, assumptions, clearance


def _base_task_kind(base_root: Path) -> Any:
    return (_read(base_root / "native_task_arena_packet_request.v1.json").get("task_spec") or {}).get("task_kind")


@read_scoped
def prepare_native_g1_beta_packet(*, base_packet_record: Mapping[str, Any], source: Mapping[str, Any],
                                  site_id: str, output_root: Path, robot_asset_record: Mapping[str, Any],
                                  task_success_contract: Mapping[str, Any]) -> dict:
    """Prepare under `<output_root>/native-g1/<base-digest>/<source-digest>/`.

    The stance search and the packet read the base packet's scene collider through one read scope: a USDA
    text room is read into a binary copy each reader opens (``usd_binary_read_cache``), not parsed per open.
    """
    from . import native_g1_spawn_clearance as spawn
    from .native_g1_arena_robot_configuration import G1_USD_SHA256, G1_USD_SIZE_BYTES
    from .native_task_arena_bundle import verify_native_task_arena_packet
    from .native_task_arena_packet import materialize_native_task_arena_packet
    from .deployment_embodiment_preparation_contract import write_once
    from .native_g1_navigation_goal import NAVIGATION_TASK_KIND

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
        # A navigation base packet (the G1 at the compiled start) has its own request; every other kind the rigid one.
        navigation = _base_task_kind(base_root) == NAVIGATION_TASK_KIND
        request, assumptions, clearance = (_navigation_request if navigation else _request)(
            base_root=base_root, base_receipt=receipt, robot_asset=robot_asset, evidence_root=evidence,
            source=source, site_id=site_id, task_success_contract=task_success_contract)
        if destination.exists():
            _, prior, _ = verify_native_task_arena_packet(destination)
            if _read(destination / "native_task_arena_packet_request.v1.json") != request:
                raise ValueError("g1_beta_preparation_existing_packet_conflict")
        else:
            prior = materialize_native_task_arena_packet(request=request, evidence_root=evidence, output_dir=destination)
            verify_native_task_arena_packet(destination)
        packet = _record(destination / "native_task_arena_packet_receipt.v1.json")
        # Beside the packet, not in its sealed request: the door's registration pins that request's assumptions.
        # A read error is not kept: the site's next registration pass measures the packet itself.
        spawn_path = destination / spawn.FILENAME
        if not spawn_path.exists() and not clearance["retryable"]:
            write_once(spawn_path, spawn.seal({**clearance, "scene_plan_digest": prior["arena_scene_plan_digest"]}))
        spawn_report = _read(spawn_path) if spawn_path.exists() else clearance
        result = {"schema_version": SCHEMA, "status": "prepared_not_executed", "recipe_id": RECIPE,
            "source": dict(source), "site_id": site_id, "base_packet": dict(base_packet_record),
            "scene_packet": packet, "assumptions": assumptions, "scene_plan_digest": prior["arena_scene_plan_digest"],
            "candidate_reasons": {key: ["g1_beta_current_navigation_goal_authority_missing"] for key in (
                "humanoidarena_dp_g1_dex3_sonic_vision_navi", "humanoidarena_pi05_g1_dex3_sonic_vision_navi")},
            "candidate_policy_queried": False, "simulator_executed": False, "provider_allocations_performed": 0,
            "new_human_review_claimed": False, "current_consent_revalidated": False,
            "claim_ceiling": "development_only",
            "spawn_clearance": {"status": spawn_report["status"], "max_depth_m": spawn_report["max_depth_m"],
                "fallback_used": spawn_report["stance_selection"]["fallback_used"],
                "report": _record(spawn_path) if spawn_path.exists() else None,
                "disclosed_assumption": spawn.DISCLOSED_ASSUMPTION}}
        if navigation:
            # The spawn check at each start offset, from the packet's own files; a read error is not kept, as above.
            offsets = spawn.ensure_packet_start_offsets(destination)
            offsets_path = destination / spawn.START_OFFSETS_FILENAME
            result["candidate_reasons"] = {
                **{key: ["g1_beta_manipulation_policy_task_mismatch"] for key in (
                    "humanoidarena_dp_g1_dex3_sonic", "humanoidarena_pi05_g1_dex3_sonic")},
                **result["candidate_reasons"]}
            result["start_offsets"] = {
                "report": _record(offsets_path) if offsets_path.exists() else None,
                "offsets": {row["offset_id"]: {"status": row["status"], "max_depth_m": row["max_depth_m"],
                                               "unsupported_feet": row["unsupported_feet"]}
                            for row in offsets["offsets"]}}
        floor = spawn.scene_floor_estimate(request["task_spec"])
        if floor is not None:
            # A flat floor standing in for an uneven scanned one: disclosed, flagged at planning, never a gate.
            # Absent, the receipt is byte for byte the one before.
            result["scene_floor_estimate"] = {**floor, "setup_quality_flag": spawn.FLOOR_ESTIMATED_REASON,
                                              "disclosed_assumption": spawn.floor_estimate_assumption(floor)}
        result["preparation_digest"] = canonical_digest(result, digest_field="preparation_digest")
        write_once(destination / FILENAME, result)
        return {**result, "preparation_receipt": _record(destination / FILENAME)}
