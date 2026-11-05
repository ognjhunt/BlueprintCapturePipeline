"""A prepared scene's own half of a team evaluation, compiled on the CPU.

Beta journey step: L5 team policy evaluation. A team brings its robot and its
policy; the scene brings its geometry, its task and the success contract the
simulator judges by. Nothing else is needed before a team can run, so a
published configured scene revision (robot-neutral by construction) is
compiled here straight into the scene half a team run reads:

- the scene's packet: its collision, its native appearance and the task object
  from the revision's configured bundle, and the task its adapter derives from
  the revision's own documents: rigid pick-and-place and push through
  ``task_evaluation_rigid_relocation_native_adapter``, and opening or closing a
  jointed part (door, drawer, lid, rack) through
  ``task_evaluation_articulated_open_close_native_adapter``, as the website
  episode compiler does, with the same runtime joint overlay;
- a robot station: the packet format carries one robot, so Blueprint's DROID
  Franka stands in it (cameras and reset from the operator's DROID profile)
  on a mount level with the task's support (on the floor for a floor-standing
  fixture), at the closest clear floor spot facing the task
  (``team_embodiment_stance``). For a jointed part the station also stays out
  of the part's path to success and out of any fall gravity carries it
  through, preferring a spot the part never reaches. A team on the DROID
  interface runs this Franka; a team's own fixed arm is bound to the packet's
  station unless its run takes its own (below);
- a station per registered team body, when the scene is compiled with their
  manifests: each chosen from that body's own manifest (base, footprint,
  declared reach) on the same floor, task and moving part, never with the
  Franka's model or numbers; what a body declares no model for stays
  unassessed (``_embodiment_stations``, recorded as ``embodiment_stations``).
  A body given its robot model (fixed, or free from where it starts) takes
  the closest such stance from which its own arm reaches every phase of the
  task in that model, else keeps the earlier one (``_own_arm_stations``);
- the right exterior camera, for DROID policies that read both exteriors,
  where a pose beside the station sees the task clear of the captured scene
  (``droid_exterior_cameras``); the packet is then compiled once more with it,
  and without it the record says why;
- the sealed success contract: the scene owner's when the task requires one,
  else the strategy's compatibility default (as every canary setup seals it);
- ``team_scene_inputs.v1.json``, which the team registry reads
  (``team_candidate_quick10_registry``) in place of a frozen internal pair.

No robot placement inference, no GPU and no prior run. The packet is never
construction evidence: its receipt says no native application happened, the
first shard of every team run checks the scene before any team episode (the
robot-free probe), and void episodes past the cap refuse the run. A passive
destination uses the native compiler's exact fitted geometry and qualified
asset. Policy packets require its independent full-scene placement receipt
and the expected execution binding in the operator's sealed DROID profile;
missing placement evidence remains a precise preparation task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .usd_binary_read_cache import binary_copy, read_scoped

SCHEMA = "blueprint.team_scene_inputs.v1"
CLAIM_CEILING = "diagnostic_policy_execution"
RECORD_FILENAME = "team_scene_inputs.v1.json"
PACKET_DIRECTORY = "native-task-packet"
OWN_ARM_RULE = "closest_clear_stance_own_arm_reaches_every_phase"
# The same scene compiled with the right exterior camera, when one is placed.
THREE_VIEW_PACKET_DIRECTORY = "native-task-packet-three-view"
# The room's collider with the task's support planarized to its success plane (``rigid_support_pad``), when derived.
SUPPORT_PAD_DIRECTORY = "support-pad"
# A pedestal for an arm's base: its half extent on the floor, metres.
STATION_FOOTPRINT_HALF_EXTENT_M = (0.15, 0.15)
EXECUTION_AUTHORITY = {"maximum_provider_allocations": 1, "retry_cap": 0,
                       "single_warm_provider_session_required": True,
                       "caller_surviving_watchdog_required": True,
                       "billing_teardown_provider_zero_required": True}
# Controls are never run before a team evaluation. A contract that requires them per cell runs
# without them under the typed diagnostic omission while the platform owner's control-trial pause
# stands, and is otherwise refused by the registry (``team_candidate_strict_controls_runtime_unavailable``).
PENDING_CONTROLS = {"mode": "nonblocking_diagnostic_pending", "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False}
RIGID_STRATEGIES = ("pick_and_place", "planar_push")
ARTICULATED_STRATEGY = "articulated_open_close"


class TeamScenePacketError(ValueError):
    """The scene cannot carry a team evaluation; the message names why."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TeamScenePacketError(code) from exc
    if not isinstance(value, dict):
        raise TeamScenePacketError(code)
    return value


def _droid_profile(path: Path, digest: str) -> dict[str, Any]:
    if Path(path).is_symlink() or not Path(path).is_file() or _sha256(Path(path)) != digest:
        raise TeamScenePacketError("team_scene_droid_profile_binding_invalid")
    profile = _read(path, "team_scene_droid_profile_invalid")
    roles = {str(row.get("role")): row for row in profile.get("cameras") or [] if isinstance(row, Mapping)}
    if (profile.get("schema_version") != "native_task_arena_packet_request.v1"
            or profile.get("request_digest") != canonical_digest(profile, digest_field="request_digest")
            or set(roles) != {"external", "wrist", "overview"} or roles["wrist"].get("pose_frame") != "robot_body"
            or not isinstance(roles["external"].get("intrinsics"), Mapping)
            or not isinstance(profile.get("robot_joint_reset_positions_rad"), Mapping)):
        raise TeamScenePacketError("team_scene_droid_profile_invalid")
    return profile


def _runtime_source(path: Path) -> dict[str, Any]:
    receipt = _read(path, "team_scene_runtime_source_invalid")
    if (receipt.get("schema_version") != "native_task_runtime_source_packet.v1"
            or not str(receipt.get("packet_sha256") or "").startswith("sha256:")
            or int(receipt.get("packet_size_bytes") or 0) <= 0 or receipt.get("redistribution_permitted") is not True):
        raise TeamScenePacketError("team_scene_runtime_source_invalid")
    return _record(path)


#: How the Franka's station is chosen among the clear stances (``_carrying_station``).
STATION_RULE = "first_in_reach_stance_the_scripted_carry_clears_else_nearest_graspable_point"


def _carrying_station(stances: list[dict[str, Any]], *, task_spec: Mapping[str, Any], collision: Path,
                      profile: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The Franka's station among the clear ``stances`` (ranked, best first), and the record of the choice.

    The hard rule first: from each clear stance, whether any point of the object lies within the arm's
    reach bound at its start and wherever the task accepts it (``rigid_reach_shortfall``,
    ``object_reach_inputs``). A learned policy chooses its own grasp, so that is all a station needs.
    The scripted carry (``robot_arm_path_clearance.first_carrying_stance``: the task's own grasp, then
    the intake's tilted and side-on ones, ``carry_grasps``) is tried from the first ``CARRY_STATIONS``
    stances in reach, in rank order, as evidence: the first it clears from is the station. Without one,
    the stance in reach whose object is nearest (the farther of its pick and place) stands, and the
    record flags ``SCRIPTED_CARRY_FLAG`` with the attempt that got furthest; nothing is refused for it.
    Only with no clear stance in reach does the closest stance stand ``out_of_reach``
    (``reach_shortfall``), and the bundle's gate refuses the packet by name before any GPU run.
    """
    from .robot_arm_path_clearance import (
        CARRY_STATIONS,
        NOMINAL_SEED_RAD,
        REST_SEED_RAD,
        SCRIPTED_CARRY_FLAG,
        RobotArmPathClearanceError,
        carry_grasps,
        first_carrying_stance,
        object_reach_inputs,
        rigid_reach_shortfall,
    )

    try:
        affordance = task_spec["interaction_affordance"]
        reset = profile["robot_joint_reset_positions_rad"]
        task = {"start_pose_world": task_spec["start_pose_world"],
                "target_position_world": task_spec["target_position_world_m"],
                "contact_point_m": affordance["contact_point_scoring_frame_m"],
                "approach_unit": affordance["approach_unit_scoring_frame"],
                "gripper_orientation_xyzw": affordance["gripper_orientation_scoring_frame_xyzw"],
                "pregrasp_clearance_m": float(affordance["pregrasp_clearance_m"])}
        bounds = task_spec.get("subject_collision_bounds_scoring_frame_m")
        reach_inputs = object_reach_inputs(task_spec)
    except (KeyError, TypeError, ValueError) as exc:
        raise TeamScenePacketError("team_scene_carry_inputs_invalid") from exc
    reaches: list[dict[str, Any] | None] = []
    for stance in stances:
        try:
            reaches.append(rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=stance, **reach_inputs))
        except (KeyError, TypeError, ValueError, RobotArmPathClearanceError):
            reaches.append(None)
    in_reach = [i for i, reach in enumerate(reaches) if reach is not None and reach["status"] == "within_bound"]
    tried = [stances[i] for i in in_reach[:CARRY_STATIONS]] if in_reach else list(stances[:CARRY_STATIONS])
    try:
        cleared, furthest, attempts = first_carrying_stance(tried, collision_path=collision, robot_id="franka_panda",
            seeds=[[float(reset[f"panda_joint{i}"]) for i in range(1, 8)], list(NOMINAL_SEED_RAD),
                   list(REST_SEED_RAD["franka_panda"])], grasps=carry_grasps(task, bounds), **task)
    except (KeyError, TypeError, ValueError) as exc:
        raise TeamScenePacketError("team_scene_carry_inputs_invalid") from exc

    def nearest(index: int) -> float:
        rows = (reaches[index] or {}).get("phases") or []
        return max((float(row.get("object_nearest_distance_m", row["distance_m"])) for row in rows), default=math.inf)

    if cleared is not None:
        station, basis = cleared, "scripted_carry_cleared"
    elif in_reach:
        station, basis = stances[min(in_reach, key=lambda i: (nearest(i), i))], "nearest_graspable_point_in_reach"
    else:
        station, basis = stances[0], "no_clear_stance_in_reach"
    rank = next(i for i, row in enumerate(stances) if row is station)
    record: dict[str, Any] = {
        "rule": STATION_RULE, "cleared": cleared is not None, "chosen_basis": basis, "chosen_rank": rank,
        "stances_measured": len(stances), "stances_in_reach": len(in_reach),
        "flags": [SCRIPTED_CARRY_FLAG] if cleared is None and in_reach else [],
        "grasps_tried": len({report.get("grasp") for _, report in attempts}),
        "results": [{"stance_digest": row["stance_digest"], "status": report["status"],
                     "blockers": report["blockers"], **({"grasp": report["grasp"]} if "grasp" in report else {})}
                    for row, report in attempts]}
    if reaches[rank] is not None:
        reach = reaches[rank]
        record["object_reach"] = {key: reach[key] for key in ("rule", "status", "reach_bound_m", "beyond")} | {
            "phases": [{key: row[key] for key in ("phase", "distance_m", "object_nearest_distance_m") if key in row}
                       for row in reach["phases"]]}
    if cleared is None:
        best = next(row for row, report in attempts if report is furthest)
        record["best_attempt"] = {"stance_digest": best["stance_digest"], "grasp": furthest.get("grasp"),
                                  "blockers": list(furthest["blockers"]),
                                  "furthest_sample": furthest.get("furthest_sample"),
                                  "path_samples": furthest.get("path_samples")}
        record["reach_shortfall"] = reaches[rank] if reaches[rank] is not None else {
            "status": "unassessed", "reason": "reach_shortfall_inputs_invalid"}
    return station, record


def _embodiment_stations(manifests: Sequence[Mapping[str, Any]], *, place: Mapping[str, Any],
                         points: Sequence[Sequence[float]], grasp: Mapping[str, Any] | None,
                         ranked: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, dict[str, Any]]:
    """Each registered body's own station on this scene, from its own manifest, keyed by its digest.

    ``place`` is the Franka station's search (``ground_stance``): the same floor, task and moving part. A fixed
    base stands on the station's pedestal level with the task's support and takes the closest clear stance
    from which its declared ``reach_m`` covers every task point (a rigid task's search ranks by the farther
    of them already); with none, the closest stands, out of reach.
    A free base stands on its own footprint at its own root height and declares no reach. No Franka model or
    number enters: a moving part is kept off the mount or footprint, and the body itself, its reach without
    one, and a grip force no manifest declares stay unassessed. Each declared tool is screened against the
    ``grasp`` (``website_task_preparation.screen_physics``, or the handle screen for a jointed part). A fixed
    or free base's clear stances, closest first, are kept in ``ranked`` for ``_own_arm_stations``.
    """
    from .team_embodiment import ROBOT_ID, TeamEmbodimentError, validate_embodiment_manifest
    from .team_embodiment_stance import GRID_STEP_M, PEDESTAL_SEARCH, SEARCH_RADIUS_M, ground_stance
    from .website_task_preparation import embodiment_grippers, screen_articulated_physics, screen_physics

    every =(2 * math.floor(SEARCH_RADIUS_M / GRID_STEP_M + 1e-9) + 1) ** 2  # every clear stance searched
    rows: dict[str, dict[str, Any]] = {}
    for raw in manifests:
        manifest = validate_embodiment_manifest(raw)
        base, motion = manifest["base"], place.get("task_motion")
        unassessed = ["embodiment_body_task_motion_clearance_unmodeled"] if motion else []
        row: dict[str, Any] = {"embodiment_id": manifest["embodiment_id"],
                               "embodiment_digest": manifest["embodiment_digest"], "base_mode": base["mode"],
                               "status": "unassessed", "station": None, "blockers": [], "reach": None}
        search = {**place, **({"task_motion": {**motion, "robot_id": ROBOT_ID}} if motion else {})}
        try:
            if base["mode"] == "fixed":
                stances = ground_stance(**search, ranked=every)
                if ranked is not None:
                    ranked[manifest["embodiment_digest"]] = stances
                reach = float(manifest["reach_m"])
                farthest = [max(math.dist(stance["position_world_m"], point) for point in points)
                            for stance in stances]
                index = next((i for i, distance in enumerate(farthest) if distance <= reach), 0)
                within = farthest[index] <= reach
                row.update(status="chosen" if within else "out_of_reach", station=stances[index],
                           blockers=[] if within else ["team_embodiment_station_out_of_reach"],
                           mount="station_pedestal_level_with_task_support",
                           reach={"rule": "closest_clear_stance_within_declared_reach", "reach_m": reach,
                                  "task_points_world_m": [list(point) for point in points],
                                  "farthest_task_point_m": farthest[index], "clear_stances_measured": len(stances)})
            else:
                # A free base stands on its own footprint with the stand validator's clearance and standoff,
                # closest to the task first: the pedestal's search and ranking are a fixed mount's.
                own = {key: value for key, value in search.items()
                       if key not in {*PEDESTAL_SEARCH, "task_points_world_m"}}
                stances = ground_stance(**{**own, "root_world_z_m": None,
                                           "root_height_m": float(base["root_height_m"]),
                                           "footprint_half_extent_m": list(base["footprint_half_extent_m"])},
                                        ranked=every)
                if ranked is not None:
                    ranked[manifest["embodiment_digest"]] = stances
                row.update(status="chosen", station=stances[0], mount="own_footprint_on_floor_at_root_height")
                unassessed.append("embodiment_free_base_reach_not_modeled")
        except TeamEmbodimentError as exc:
            row["blockers"] = list(exc.blockers)
        except (KeyError, TypeError, ValueError):
            row["blockers"] = ["team_scene_embodiment_station_inputs_invalid"]
        if grasp is not None:
            tools = embodiment_grippers(manifest)
            row["grippers"] = [
                screen_physics(grasp["dimensions_m"], gripper=tool) if grasp["joint_type"] is None
                else screen_articulated_physics(grasp["dimensions_m"], joint_type=grasp["joint_type"], gripper=tool)
                for tool in tools]
            unassessed += [] if tools else ["embodiment_gripper_not_declared"]
        rows[manifest["embodiment_digest"]] = {**row, "unassessed": unassessed}
    return rows


def _own_arm_stations(stations: dict[str, dict[str, Any]], ranked: Mapping[str, Sequence[Mapping[str, Any]]], *,
                      plan: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]],
                      models: Mapping[str, str | Path]) -> None:
    """Choose each body's station with its own arm, where its robot model is given.

    The closest clear stance from which one of the body's own arms reaches every authored phase of the
    task in its own model (``team_arm_reach.first_reaching_stance``) replaces the declared-reach choice;
    a stance its declared ``reach_m`` missed is then chosen too, since the model measured it. A free base
    is measured the same way from each floor stance it could start at (its own footprint and root
    height), so it starts where its arm already reaches. When no stance reaches, or the body or task
    cannot be modelled, the earlier choice stands. The arm's verdict is recorded either way
    (``reach.own_arm``) and refuses nothing: reach is not a privacy, rights or money control
    (``AGENTS.md``).
    """
    from .team_arm_reach import first_reaching_stance
    from .team_embodiment import ROBOT_ID

    for digest, row in stations.items():
        stances, model, manifest = ranked.get(digest), models.get(digest), manifests.get(digest)
        free = row.get("base_mode") == "free"
        if not stances or model is None or not manifest or not (free or row.get("reach")):
            continue
        robot = {"robot_id": ROBOT_ID, "embodiment_id": row["embodiment_id"], "embodiment_digest": digest,
                 "base_mode": "free_on_floor" if free else "fixed_at_station",
                 "usd_sha256": manifest["model"]["sha256"],
                 "base_pose_world": {"position_world_m": list(row["station"]["position_world_m"]),
                                     "orientation_xyzw": list(row["station"]["orientation_xyzw"])}}
        found = None
        try:
            found = first_reaching_stance({**plan, "robot": robot}, stances, model_path=model, manifest=manifest,
                                          free_base=free)
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            arm = {"rule": OWN_ARM_RULE, "status": "unassessed", "reason": "own_arm_reach_inputs_invalid",
                   "detail": type(exc).__name__}
        else:
            report = found["report"]
            arm = {"rule": OWN_ARM_RULE, "status": report["status"], "chosen_rank": found["index"],
                   "stances_solved": found["stances_solved"],
                   "reaching_arms": list(report.get("reaching_arms") or []),
                   "advisories": list(report["advisories"]), "claim_ceiling": report["claim_ceiling"],
                   "report_digest": report["report_digest"],
                   **{key: report[key] for key in ("reason", "detail", "beyond_reach") if key in report}}
        reached = found is not None and found["index"] is not None
        if free:
            row["reach"] = {"rule": OWN_ARM_RULE if reached else "closest_clear_floor_stance", "own_arm": arm}
            if arm["status"] != "unassessed":
                row["unassessed"] = [code for code in row["unassessed"]
                                     if code != "embodiment_free_base_reach_not_modeled"]
        else:
            row["reach"]["own_arm"] = arm
        if not reached:
            continue
        station = stances[found["index"]]
        row.update(status="chosen", station=station, blockers=[])
        if not free:
            points = row["reach"]["task_points_world_m"]
            farthest = max(math.dist(station["position_world_m"], point) for point in points)
            row["reach"].update(rule=OWN_ARM_RULE, farthest_task_point_m=farthest,
                                declared_reach_covers=farthest <= row["reach"]["reach_m"])


def _cameras(profile: Mapping[str, Any], *, base: list[float], start: list[float],
             target: list[float], whole_region: bool = False) -> list[dict[str, Any]]:
    from .task_evaluation_diagnostic_native_arena_compiler import _world_camera

    roles = {str(row["role"]): dict(row) for row in profile["cameras"]}
    intrinsics = roles["external"]["intrinsics"]
    direction = [target[0] - start[0], target[1] - start[1]]
    norm = math.hypot(*direction) or 1.0
    lateral = [-direction[1] / norm, direction[0] / norm, 0.0]
    midpoint = [(start[i] + target[i]) / 2.0 for i in range(3)]
    overview = _world_camera("overview", position=[midpoint[0] + 0.9 * lateral[0], midpoint[1] + 0.9 * lateral[1],
                                                   max(base[2], start[2]) + 1.45], target=midpoint,
                             intrinsics=intrinsics)
    overview["policy_input"] = False
    return [_world_camera("external", position=[base[0], base[1], base[2] + 1.35],
                          target=midpoint if whole_region else start,
                          intrinsics=intrinsics), json.loads(json.dumps(roles["wrist"])), overview]


def _exterior_right(packet: Path, *, base: Mapping[str, Any],
                    aim: Sequence[float]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Prefer a clear third view; keep a real authored camera with any measured limitation."""
    from .droid_exterior_cameras import (
        ARENA_EXTERIOR_INTRINSICS, arena_mount_world, camera_row, estimated_arena_mount, place_exterior_right,
    )
    from .scene_camera_visibility import SceneCameraVisibilityError, packet_view_evaluator
    from .scene_physics_preflight import ScenePhysicsPreflightError

    try:
        camera, receipt = place_exterior_right(base_pose_world=base, aim_world=aim, intrinsics=ARENA_EXTERIOR_INTRINSICS,
                                              evaluate=packet_view_evaluator(packet))
        if camera is None:
            # An occluded RGB stream is still a valid policy input. Retain the
            # actual sensor and the measured limitation; task execution records
            # whether the policy succeeds with those observations.
            camera = camera_row(arena_mount_world(base), ARENA_EXTERIOR_INTRINSICS)
            receipt = {**receipt, "status": "limited", "placement": "arena_mount", "camera_available": True,
                       "visibility_measured": True, "clear_view_found": False, "simulation_estimate": True}
            receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        return camera, receipt
    except (SceneCameraVisibilityError, ScenePhysicsPreflightError, ImportError, KeyError, TypeError,
            ValueError) as exc:
        # Unknown visibility is a recorded simulation assumption. Keep Arena's
        # real authored third camera so the budgeted candidate can be tested.
        return estimated_arena_mount(base_pose_world=base, code=str(exc)[:160] or type(exc).__name__)


def _articulated_task(*, revision: Mapping[str, Any], references: Mapping[str, Any],
                      replacement: Path, root: Path,
                      owner_success_confirmation_scope: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The revision's open/close task as the website episode compiler builds it, plus its part's motion.

    An owner's confirmation applies before construction, so every binding derived from the task holds it.
    """

    import numpy as np

    from . import task_evaluation_native_arena_episode_compiler as compiler
    from .native_articulated_construction_inputs import materialize_articulated_construction_inputs
    from .scene_physics_preflight import _pose_matrix, asset_envelope, load_asset_model, task_motion_surface
    from .task_evaluation_articulated_open_close_native_adapter import adapt_articulated_open_close_task_template
    from .task_evaluation_configured_controls_deferred_inputs import articulated_placement_plan_from_adapter
    from .task_evaluation_diagnostic_native_arena_compiler import _released_task_motion
    from .task_evaluation_rigid_relocation_native_adapter import (
        STATIC_QUALIFICATION_CONTRACT_PATH,
        SUPPORT_PLANE_CONTRACT_PATH,
    )
    from .task_evaluation_robot_placement_trajectory import placement_trajectory_from_native_plan

    adapter = adapt_articulated_open_close_task_template(configured_revision=revision,
                                                         materialized_references=references)
    definition = adapter["native_task_definition"]
    task_spec = compiler._runtime_subject_task_spec(definition["task_spec"])
    task_spec["success_criteria"] = adapter["native_success_criteria"].get("criteria")
    task_spec, confirmation = _owner_confirmed(task_spec, revision=revision, scope=owner_success_confirmation_scope)
    sources = adapter["source_documents"]["documents"]
    digest, size = compiler._sha256_and_size(replacement)
    trajectory = placement_trajectory_from_native_plan(articulated_placement_plan_from_adapter(
        adapted=adapter, asset=replacement, asset_reference={"digest": digest, "size_bytes": size},
        qualified_static_digest=revision["replacement"]["static_qualification"]["digest"],
        configured_revision_digest=revision["revision_digest"]))
    contact = next(phase for phase in trajectory["phases"] if phase["phase_id"] == "handle_contact")
    runtime, adaptation = compiler._derive_runtime_joint_overlay(
        task_spec=task_spec, task_definition=definition, replacement_path=replacement, output_root=root)
    construction = materialize_articulated_construction_inputs(
        task_spec=task_spec, definition=definition, asset=runtime,
        asset_digest=compiler._sha256_and_size(runtime)[0],
        link_prim_paths=sources[STATIC_QUALIFICATION_CONTRACT_PATH]["link_prim_paths"])
    pose = definition["task_object_pose_world"]
    motion = _released_task_motion(construction["task_spec"], definition=definition, asset=replacement, pose=pose)

    def sweep(task_range: list[float] | None) -> Any:
        return task_motion_surface(asset_usd=runtime, asset_pose_world=pose, joint_id=motion["joint_id"],
                                   task_range=task_range, reset_joint_positions=motion["reset_joint_positions"])

    whole = sweep(None)
    joint_positions = dict(definition.get("task_object_reset_joint_positions") or {})
    if not joint_positions:
        raise TeamScenePacketError("team_scene_articulated_reset_joints_missing")
    # The placed fixture's own envelope at reset, in the world: what the station must stand clear of.
    low, high = asset_envelope(load_asset_model(runtime), joint_positions)
    corners = np.array([[x, y, z, 1.0] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    world = (corners @ _pose_matrix(pose).T)[:, :3]
    row = {**compiler.evidence_asset_row(runtime, root=root, semantic_role="task_object", pose_world=pose),
           "asset_id": task_spec["subject_asset_id"], "source_asset_id": task_spec["source_subject_identity"],
           "object_type": "ARTICULATION", "reset_state": {"root_pose_world": pose, "joint_positions": joint_positions},
           **({"articulation_adaptation": adaptation} if adaptation is not None else {})}
    return {"adapter": adapter, "definition": definition, "task_spec": construction["task_spec"],
            "owner_success_confirmation": confirmation,
            "task_state_binding": construction["task_state_binding"],
            "kinematic_path": construction["articulated_kinematic_path_receipt"], "row": row,
            "start": [float(v) for v in contact["position_world_m"]],
            "target": [float(v) for v in trajectory["phases"][-1]["position_world_m"]],
            "support_top_z_m": float(sources[SUPPORT_PLANE_CONTRACT_PATH]["top_z_m"]),
            "subject_minimum": world.min(axis=0).tolist(), "subject_maximum": world.max(axis=0).tolist(),
            # Blueprint's DROID Franka stands in the station (module docstring).
            "task_motion": {"robot_id": "franka_panda", "path_points": sweep(motion["task_range"]),
                            "fall_points": np.zeros((0, 3)) if motion["holds_released_position"] else whole,
                            "limit_points": whole}}


def _with_scoring_bounds(spec: Mapping[str, Any], bounds: Mapping[str, Any] | None) -> dict[str, Any]:
    """``spec`` with the object's qualified collision bounds in its scoring frame, as a surface target binds them.

    A task without a surface target carries none; they are the qualified body bounds less the scoring
    frame's offset from the root (``asset_root_from_scoring_frame``), as the goal reading takes them.
    """

    value = dict(spec)
    if value.get("subject_collision_bounds_scoring_frame_m") is None and isinstance(bounds, Mapping):
        transform = (value.get("interaction_affordance") or {}).get("asset_root_from_scoring_frame") or {}
        offset = [float(item) for item in transform.get("position_m") or []]
        if len(offset) == 3 and [float(item) for item in transform.get("orientation_xyzw") or []] == [0, 0, 0, 1]:
            value["subject_collision_bounds_scoring_frame_m"] = {
                key: [float(bounds[key][axis]) - offset[axis] for axis in range(3)] for key in ("minimum", "maximum")}
    return value


def _seated_rigid_adapter(*, revision: Mapping[str, Any], references: Mapping[str, Any], collision: Path,
                          staged_destination: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The rigid task adapted with the replacement seated on the support the physics runs on, the reading, the pad.

    The registered support top is the highest face of the whole support, not
    where the object rests (capture-447ae4f4: the tray spawned 14.7 mm above
    the start the scorer reads within 2 mm). The collider the episode collides
    with is read under the replacement's qualified collision footprint at the
    start, which seats it there, and at the goal, which is recorded against the
    sealed success window (``rigid_support_seating``). Where the task starts and
    ends on one support, that support is first made the flat one its success
    contract describes (``rigid_support_pad``; capture-447ae4f4's tray tilted
    0.044 rad on the scan and settled under the window): the seating then reads
    the planarized triangles, and the third value is the modelling record with,
    under ``plan``, what ``_stage_support_pad`` writes (never for a task ending on
    a ``staged_destination`` asset, or on another surface). A collider that cannot be
    read keeps the registered top, by name; nothing here changes the revision,
    its success document or the owner's confirmation of it.
    """

    from .rigid_support_pad import (
        MAXIMUM_EDGE_REACH_M,
        RigidSupportPadError,
        pad_region,
        plan_support_pad,
    )
    from .rigid_support_seating import SEATING_AGREEMENT_M, collider_triangles, triangle_support_surface
    from .task_evaluation_rigid_relocation_native_adapter import (
        adapt_rigid_relocation_task_template,
    )

    adapter = adapt_rigid_relocation_task_template(configured_revision=revision, materialized_references=references)
    spec = adapter["native_task_definition"]["task_spec"]
    documents = adapter["source_documents"]["documents"]
    observed = documents["static_qualification"].get("observed_structure") or {}
    bounds = observed.get("collision_bounds_body_frame_m")
    other_surface = staged_destination or documents["definition"].get("destination_support_plane") is not None
    if not isinstance(bounds, Mapping):
        return adapter, {"status": "registered_support_top", "reason": "qualified_collision_bounds_missing"}, {
            "status": "unchecked", "reason": "rigid_support_qualified_collision_bounds_missing"}
    try:
        center = [float(value) for value in observed["center_of_mass_m"]]
        corners = [(float(x) - center[0], float(y) - center[1])
                   for x in (bounds["minimum"][0], bounds["maximum"][0])
                   for y in (bounds["minimum"][1], bounds["maximum"][1])]
        reach = max(math.hypot(x, y) for x, y in corners)
        points: list[Any] = [spec["start_pose_world"][:2], spec["target_position_world_m"][:2]]
        if not other_surface:
            try:
                # The pad's whole region, and the edges its selection may grow by.
                points, reach = pad_region(_with_scoring_bounds(spec, bounds),
                                           support_bounds=documents["support_plane"]).tolist(), (
                    MAXIMUM_EDGE_REACH_M * math.sqrt(2.0))
            except RigidSupportPadError:
                pass
        triangles = collider_triangles(collision, points_xy=points, reach_m=reach,
                                       height_z=float(documents["support_plane"]["top_z_m"]))
    except Exception as exc:  # noqa: BLE001 - an unread collider keeps the registered top; the compile goes on
        reason = f"{type(exc).__name__}:{str(exc)[:160]}"
        return adapter, {"status": "registered_support_top", "reason": reason}, {
            "status": "unchecked", "reason": "rigid_support_collider_unread", "detail": reason}
    pad: dict[str, Any] | None = None
    if other_surface:
        support: dict[str, Any] = {"status": "not_applicable", "reason": (
            "rigid_support_destination_is_a_staged_asset" if staged_destination
            else "rigid_support_destination_on_another_surface")}
    else:
        try:
            pad = plan_support_pad(_with_scoring_bounds(spec, bounds), support_plane=documents["support_plane"],
                                   triangles=triangles)
            support = dict(pad["record"])
        except (KeyError, TypeError, ValueError) as exc:  # RigidSupportPadError names its own reason
            support = {"status": "not_applicable", "reason": str(exc)[:200]}
    planar = pad is not None and pad["record"]["plane"]["consistent"] is True
    physics = pad["planarized_triangles"] if planar else triangles
    seated = adapt_rigid_relocation_task_template(configured_revision=revision, materialized_references=references,
                                                  support_surface=triangle_support_surface(physics))
    definition = seated["native_task_definition"]
    alignment = definition["task_spec"]["interaction_affordance"]["support_alignment"]
    reading: dict[str, Any] = {"status": "measured_on_scene_collider", "start": alignment.get("start_support"),
                               "goal": alignment.get("goal_support"),
                               **({"support": "planarized_to_success_plane"} if planar else {})}
    if (alignment.get("goal_support") or {}).get("within_success_window") is False:
        # The sealed success document's height window misses where a correct placement rests:
        # only a corrected success document, confirmed by the owner, can change it.
        reading["flag"] = "goal_rest_outside_sealed_success_window"
    rest = (alignment.get("goal_support") or {}).get("rest_scoring_z_m")
    moved = max(abs(float(definition["task_object_pose_world"]["position_world_m"][2])
                    - float(adapter["native_task_definition"]["task_object_pose_world"]["position_world_m"][2])),
                abs(float(rest) - float(spec["target_position_world_m"][2])) if rest is not None else 0.0)
    if pad is not None:
        support["plan"] = {"selection": pad["selection"], "band": pad["record"]["scan_band_z_m"],
                           "plane_z_m": pad["record"]["plane_z_m"], "planarize": planar,
                           "triangles": physics, "scan_triangles": triangles, "collision_bounds_body_frame_m": bounds}
        if not planar:
            support.update(status="held", reasons=[pad["record"]["plane"]["reason"]],
                           requires=pad["record"]["plane"].get("requires"))
    if moved <= SEATING_AGREEMENT_M:
        # The collider agrees with the registered support at both ends: the task stays exactly the
        # revision's (rigid_task_for_revision), and the reading is kept here only.
        return adapter, {**reading, "applied": False}, support
    return seated, {**reading, "applied": True}, support


def _stage_support_pad(support: dict[str, Any], *, task_spec: Mapping[str, Any], sealed_collision: Path,
                       root: Path) -> Path | None:
    """Check where the task's object rests on the support the physics runs on; stage that support when derived.

    ``support`` is ``_seated_rigid_adapter``'s record, completed here in place:
    its ``rest_check`` (``rigid_support_pad.support_rest_check``), its status
    (``consistent``, or ``held`` with typed reasons before any probe spends), and,
    when the plan moved any scanned point, the derived collider it writes and
    reads back under ``SUPPORT_PAD_DIRECTORY``. Returns that collider, or None.
    """

    from .rigid_support_pad import (
        hold_reason,
        read_back,
        support_rest_check,
        write_planarized_collider,
    )

    plan = support.pop("plan", None)
    if plan is None:
        return None
    spec = _with_scoring_bounds(task_spec, plan["collision_bounds_body_frame_m"])
    check = support_rest_check(plan["triangles"], spec)
    derived = None
    if plan["planarize"] and support.get("read_points_moved", 0) > 0:
        derived = root / SUPPORT_PAD_DIRECTORY / "collision.usdc"
        try:
            written = write_planarized_collider(sealed_collision, derived, selection=plan["selection"],
                                                band=plan["band"], plane_z=plan["plane_z_m"])
            # The physics' own bytes, read back where the object rests.
            readback = read_back(derived, spec, record=support)
        except Exception as exc:  # noqa: BLE001 - a support that cannot be written is named, never guessed at
            raise TeamScenePacketError(f"team_scene_support_pad_unwritten:{type(exc).__name__}:{str(exc)[:120]}") from exc
        if readback["status"] != check["status"] or readback["reasons"] != check["reasons"]:
            raise TeamScenePacketError("team_scene_support_pad_readback_disagrees")
        digest, size = _sha256(derived), derived.stat().st_size
        support["derived_collider"] = {
            "relative_path": derived.relative_to(root).as_posix(), "sha256": digest, "size_bytes": size,
            "derived_from_sha256": _sha256(sealed_collision), "edit": "point_heights_only", **written,
            "rest_check_read_back": True}
    elif plan["planarize"]:
        # Flat at the success plane already: a sealed collider the website preparation planarized
        # (``website_support_pad``) or a flat room. Nothing is derived; the physics runs on the sealed bytes.
        support["derived_collider"] = None
        support["collider_already_planar"] = True
    if plan["planarize"]:
        # What the scan as captured would have done with the object seated on its highest point (the
        # seating before the pad): the reason the pad exists, kept beside the check that replaces it.
        scan = support_rest_check(plan["scan_triangles"], spec, seated_on_highest_contact=True)
        support["scan_rest_check"] = {key: scan.get(key) for key in ("status", "reasons", "start", "goal")}
    support["rest_check"] = check
    if support.get("status") != "held":
        support["status"] = check["status"]
        if check["status"] == "held":
            support["reasons"] = list(check["reasons"])
    elif check["status"] == "held":
        support["reasons"] = [*support.get("reasons", []), *check["reasons"]]
    support["hold_reason"] = hold_reason(support)
    return derived


def _owner_confirmed(task_spec: Mapping[str, Any], *, revision: Mapping[str, Any],
                     scope: Mapping[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    """The task with a ``proposal_only`` owner authority the owner confirmed, and that confirmation's digest.

    Only the owner's sealed record for this exact intent, revision, success document, thresholds and
    proposal confirms it (``task_evaluation_owner_success_confirmation``); without one the task is
    unchanged and the owner contract refuses as before. No threshold changes either way.
    """

    if scope is None:
        return dict(task_spec), None
    from .task_evaluation_owner_success_confirmation import confirmed_owner_authority

    authority = confirmed_owner_authority(task_spec=task_spec, revision=revision, intent=scope["intent"],
                                          directory=scope["directory"])
    if authority is None:
        return dict(task_spec), None
    return {**task_spec, "configured_owner_authority": authority}, authority["confirmation_digest"]


def _robot_access_proposal(movable_furniture: Mapping[str, Any], *, stances: Sequence[Mapping[str, Any]],
                           carry: Mapping[str, Any], named: Sequence[Mapping[str, Any]],
                           refused: Sequence[Mapping[str, Any]], subject_center: Sequence[float],
                           task_spec: Mapping[str, Any], support_top: float, revision: Mapping[str, Any],
                           collision_sha256: str, task_id: str, support_label: str,
                           scope: Mapping[str, Any] | None, offer: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What the half records when no clear stance has the object in reach: the owner's proposal, a measurement
    due, or why neither.

    With the scene's movable furniture measured (the sidecar, ``robot_access_measurement``), the Franka's
    search named it as movable obstacles and this reads its counterfactual: the proposal to move the
    fewest items (``robot_access_furniture.station_proposal``). Without it, and only when furniture
    plausibly blocks the nearer stances and the scene's review names some, the measurement is due
    (``measurement_due``, with its binding and most): Blueprint's own preparation spend, which the worker
    runs; the site is never asked about it. Nothing here spends or changes the scene. ``offer`` is that
    measurement offer as the host already read it for ``scope`` (a remote compile, which cannot read the
    intent's preparation: ``team_scene_packet_remote``); without it the offer is read here.
    """

    from .robot_access_furniture import MEASUREMENT_DUE, MODEL, PROPOSAL_SCHEMA, station_proposal
    from .robot_access_measurement import measurement_offer, plausibly_blocked

    why = {"reason": "no_ranked_stance_carries", "carry_cleared": bool(carry.get("cleared")),
           "stances_tried": len(carry.get("results") or []),
           "reach_shortfall": (carry.get("reach_shortfall") or {}).get("status")}
    base = {"schema_version": PROPOSAL_SCHEMA, "model": MODEL, "chosen": False, "task_id": task_id}
    if movable_furniture.get("status") != "measured":
        trigger = plausibly_blocked(stances)
        if not trigger["plausible"]:
            return {**base, "status": "not_proposed", "reason": "nearer_stances_not_blocked_off_the_support",
                    "why": why, "trigger": trigger}
        offer = (dict(offer) if offer is not None
                 else measurement_offer(scope, revision_digest=str(revision["revision_digest"])))
        if offer["status"] != "available":
            return {**base, "status": "not_proposed", "reason": "measurement_" + str(offer["status"])[:80],
                    "why": why, "trigger": trigger}
        binding = offer["binding"]
        return {**base, "status": MEASUREMENT_DUE, "why": why, "trigger": trigger,
                "items": [dict(row) for row in binding["items"]], "binding": binding,
                "binding_digest": offer["binding_digest"], "maximum_cost_usd": binding["maximum_cost_usd"],
                "spend": "website_preparation_spend_within_the_jobs_approved_caps"}
    try:
        from .robot_arm_path_clearance import object_reach_inputs

        reach_inputs = object_reach_inputs(task_spec)
        task = {"start_pose_world": reach_inputs["start_pose_world"],
                "target_position_world_m": reach_inputs["target_position_world"],
                "contact_point_m": reach_inputs["contact_point_m"],
                "subject_bounds_m": reach_inputs["subject_bounds_m"],
                "destination_bounds_world_m": reach_inputs["destination_bounds_world_m"],
                "place_yaw_free_rad": reach_inputs["place_yaw_free_rad"]}
        proposal = station_proposal(stances=stances, carry=carry, named=named, refused=refused,
                                    subject_center=subject_center, task=task, root_z=float(support_top),
                                    revision_digest=str(revision["revision_digest"]),
                                    collision_sha256=collision_sha256, support_label=support_label)
    except (KeyError, TypeError, ValueError) as exc:
        return {**base, "status": "not_proposed", "reason": "robot_access_proposal_inputs_invalid",
                "detail": type(exc).__name__}
    return {**proposal, "task_id": task_id, "sidecar_digest": movable_furniture.get("sidecar_digest"),
            "preparation_digest": movable_furniture.get("preparation_digest")}


def _robot_access_applied(access: Mapping[str, Any], *, scope: Mapping[str, Any], physics_collision: Path, root: Path,
                          place: Mapping[str, Any], floor_z: float, station: Mapping[str, Any],
                          task_spec: Mapping[str, Any], profile: Mapping[str, Any], support: Mapping[str, Any] | None,
                          adapter: Mapping[str, Any], start: Sequence[float], target: Sequence[float],
                          support_top: float) -> dict[str, Any] | None:
    """The half rebuilt without the items the site owner approved moving, checked on the CPU; None without it.

    Only the owner's sealed approval of exactly this proposal's binding counts
    (``robot_access_furniture.find_confirmation``). A derived copy of the collider the physics runs on
    is written without the approved items' faces and read back; the flat-support rest check
    (``rigid_support_pad.support_rest_check``) must read the same there as on the collider before, or
    nothing is applied and the record says why. Then the stance is searched again on the derived
    collider and the carry tried again from it (the CPU reach and arm-path check the bundle's gate
    repeats before any GPU run). ``station`` is ``None`` in the result when nothing is applied.
    """

    from .rigid_support_pad import support_rest_check
    from .rigid_support_seating import collider_triangles
    from .robot_access_furniture import (
        APPEARANCE,
        ASSUMPTION,
        DIRECTORY,
        MODEL,
        faces_left_inside,
        find_confirmation,
        write_collider_without,
    )
    from .robot_arm_path_clearance import RobotArmPathClearanceError, object_reach_inputs, rigid_reach_shortfall
    from .team_embodiment import TeamEmbodimentError
    from .team_embodiment_stance import EVERY_CLEAR_STANCE, ground_stance

    try:
        confirmation = find_confirmation(scope["directory"], scope["intent"],
                                         binding_digest=str(access["binding_digest"]))
    except (KeyError, TypeError, ValueError, OSError):
        confirmation = None
    if confirmation is None:
        return None
    items = list(access["items"])
    derived = root / DIRECTORY / "collision.usdc"
    try:
        written = write_collider_without(physics_collision, derived, items=items, floor_z=floor_z)
        left = faces_left_inside(derived, items=items, floor_z=floor_z)
    except Exception as exc:  # noqa: BLE001 - a collider that cannot be derived is named, never guessed at
        raise TeamScenePacketError(f"team_scene_robot_access_unwritten:{type(exc).__name__}:{str(exc)[:120]}") from exc
    if left:
        raise TeamScenePacketError("team_scene_robot_access_readback_disagrees")
    record: dict[str, Any] = {
        **{key: access[key] for key in ("schema_version", "rule", "why", "items", "support_label", "removal_binding",
                                        "removal_binding_digest", "binding", "binding_digest", "enabled_stance",
                                        "task_id", "sidecar_digest", "preparation_digest") if key in access},
        "model": MODEL, "assumption": ASSUMPTION, "appearance": dict(APPEARANCE),
        "confirmation_digest": confirmation["confirmation_digest"], "proposal_digest": confirmation["proposal_digest"],
        "approved_by": confirmation["approved_by"], "issuer_channel": confirmation["issuer_channel"],
        "station_before": {"stance_digest": station["stance_digest"],
                           "position_world_m": list(station["position_world_m"])},
        "derived_collider": {"relative_path": derived.relative_to(root).as_posix(), "sha256": _sha256(derived),
                             "size_bytes": derived.stat().st_size, "derived_from_sha256": _sha256(physics_collision),
                             **written, "read_back_faces_left_inside": left}}
    # The object must rest where the flat-support check found it, on the collider the physics will run on.
    rest_before = (support or {}).get("rest_check") if isinstance(support, Mapping) else None
    try:
        observed = adapter["source_documents"]["documents"]["static_qualification"].get("observed_structure") or {}
        spec = _with_scoring_bounds(task_spec, observed.get("collision_bounds_body_frame_m"))
        bounds = spec.get("subject_collision_bounds_scoring_frame_m") or {}
        reach_m = max(math.hypot(float(bounds["maximum"][0]) - float(bounds["minimum"][0]),
                                 float(bounds["maximum"][1]) - float(bounds["minimum"][1])) / 2.0, 0.05)
        rest_after = support_rest_check(collider_triangles(derived, points_xy=[list(start[:2]), list(target[:2])],
                                                           reach_m=reach_m, height_z=float(support_top)), spec)
    except Exception as exc:  # noqa: BLE001 - an unread support is named; nothing is applied on a guess
        rest_after = {"status": "unchecked", "reason": f"robot_access_rest_check_unread:{type(exc).__name__}"}
    record["cpu_checks"] = {"rest_check": {key: rest_after.get(key) for key in ("status", "reasons", "reason")
                                           if key in rest_after},
                            "rest_check_before": None if not isinstance(rest_before, Mapping) else {
                                key: rest_before.get(key) for key in ("status", "reasons") if key in rest_before}}
    before_status = rest_before.get("status") if isinstance(rest_before, Mapping) else None
    if rest_after.get("status") == "held" or (before_status is not None and rest_after.get("status") != before_status):
        record.update(status="held", hold_reason=(
            "robot_access_removal_changed_the_support_rest"
            if before_status is not None and rest_after.get("status") != before_status
            else "robot_access_removal_support_rest_held"))
        return {"record": record, "collider": derived, "station": None, "place": place, "search": None}
    moved_place = {**place, "collision_path": derived, "collision_sha256": record["derived_collider"]["sha256"]}
    try:
        stances = ground_stance(**moved_place, ranked=EVERY_CLEAR_STANCE)
    except TeamEmbodimentError as exc:
        record.update(status="held", hold_reason="robot_access_removal_no_clear_stance",
                      blockers=list(exc.blockers))
        return {"record": record, "collider": derived, "station": None, "place": place, "search": None}
    moved_station, search = _carrying_station(stances, task_spec=task_spec, collision=derived, profile=profile)
    try:
        reach = rigid_reach_shortfall(robot_id="franka_panda", base_pose_world=moved_station,
                                      **object_reach_inputs(task_spec))
        record["cpu_checks"]["reach"] = {key: reach[key] for key in ("status", "reach_bound_m", "beyond")}
    except (KeyError, TypeError, ValueError, RobotArmPathClearanceError):
        record["cpu_checks"]["reach"] = {"status": "unassessed", "reason": "reach_inputs_invalid"}
    record["cpu_checks"]["carry_cleared"] = bool(search["cleared"])
    record["cpu_checks"]["stances_in_reach"] = int(search.get("stances_in_reach") or 0)
    # Only the hard reach rule holds: a scripted carry miss with the object in reach is a flag (``_carrying_station``).
    record.update(status="applied", station_after={"stance_digest": moved_station["stance_digest"],
                                                   "position_world_m": list(moved_station["position_world_m"])},
                  **({} if search["cleared"] or search.get("stances_in_reach")
                     else {"hold_reason": "robot_access_removal_carry_still_blocked"}))
    return {"record": record, "collider": derived, "station": moved_station, "place": moved_place, "search": search}


@read_scoped
def materialize_team_scene(*, revision: Mapping[str, Any], documents: Mapping[str, Path],
                           configured_scene_bundle_path: str | Path, droid_profile_path: str | Path,
                           droid_profile_digest: str, runtime_source_receipt_path: str | Path,
                           output_root: str | Path,
                           destination_placement_qualification_path: str | Path | None = None,
                           native_appearance_materializer: Callable[..., Mapping[str, Any]] | None = None,
                           embodiment_manifests: Sequence[Mapping[str, Any]] = (),
                           embodiment_models: Mapping[str, str | Path] | None = None,
                           owner_success_confirmation_scope: Mapping[str, Any] | None = None,
                           movable_furniture: Mapping[str, Any] | None = None,
                           robot_access_scope: Mapping[str, Any] | None = None,
                           robot_access_offer: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compile the revision's scene half under ``output_root`` and seal its team scene inputs.

    Each of ``embodiment_manifests`` (registered team bodies) also gets its own station on the same scene
    (``_embodiment_stations``); without any, the record is exactly the Franka-only one. A body whose
    robot model is in ``embodiment_models`` (by embodiment digest) has that station chosen with its own
    arm (``_own_arm_stations``). ``owner_success_confirmation_scope`` (``intent`` and its ``directory``,
    bound to this scene by the caller) is where a ``proposal_only`` owner authority may find the owner's
    sealed confirmation of these exact thresholds (``_owner_confirmed``).
    ``movable_furniture`` is what the scene knows of its own movable furniture: its measurement's
    sidecar, or that none was made (``robot_access_measurement.furniture_for_scope``). When none was made
    and furniture plausibly blocks the carry, the record says the measurement is due (``measurement_due``),
    which the worker runs as preparation spend. When no ranked stance carries a rigid
    pick and place, the record proposes moving the fewest of those items that opens a stance within
    the arm's reach (``robot_access_removal``), and nothing else changes. Only the site owner's sealed
    approval of exactly that proposal, found under ``robot_access_scope`` (the request's own intent),
    rebuilds the half without them (``_robot_access_applied``). ``robot_access_offer`` is the measurement
    offer for ``robot_access_scope`` when the host already read it (``team_scene_packet_remote``).
    """

    from . import task_evaluation_configured_controls_deferred_inputs as deferred
    from . import task_evaluation_native_arena_episode_compiler as compiler
    from .adp_articulated_task_success_contract import (
        seal_task_success_contract,
        validate_task_success_contract,
    )
    from .adp_task_scoring import TaskNeutralScoringError
    from .native_task_arena_packet import materialize_native_task_arena_packet
    from .task_evaluation_rigid_owner_contract import _derive_configured_owner_success_contract
    from .task_evaluation_rigid_relocation_native_adapter import (
        DEFINITION_CONTRACT_PATH,
        SOURCE_OBJECT_CONTRACT_PATH,
    )
    from .team_embodiment import TeamEmbodimentError
    from .team_embodiment_stance import ground_stance

    root = Path(output_root).resolve()
    if root.exists():
        raise TeamScenePacketError("team_scene_output_exists")
    destination_declaration = revision["task_template"].get("destination")
    if (destination_declaration is not None and destination_placement_qualification_path is None
            and "placement_qualification" not in destination_declaration):
        raise TeamScenePacketError("team_scene_destination_placement_qualification_required")
    strategy = _read(documents[DEFINITION_CONTRACT_PATH], "team_scene_task_invalid").get("strategy")
    articulated = strategy == ARTICULATED_STRATEGY
    destination_support = None
    if not articulated and strategy not in RIGID_STRATEGIES:
        raise TeamScenePacketError("team_scene_task_kind_unsupported")
    bundle = Path(configured_scene_bundle_path)
    declared = revision.get("configured_scene_bundle") or {}
    if (bundle.is_symlink() or not bundle.is_file() or bundle.stat().st_size != declared.get("size_bytes")
            or _sha256(bundle) != declared.get("digest")):
        raise TeamScenePacketError("team_scene_configured_bundle_binding_invalid")
    profile = _droid_profile(Path(droid_profile_path), droid_profile_digest)
    runtime_source = _runtime_source(Path(runtime_source_receipt_path))
    root.mkdir(parents=True, mode=0o750)

    references = deferred._materialized_references(revision=revision, documents=documents)
    scene_id, task_id = str(revision["scene_identity"]["id"]), str(revision["task_template"]["identity"]["id"])
    assets = compiler._extract_configured_assets(bundle, output_root=root)
    # Every reader of the room's collider below (the rigid task's seating, the station searches and the
    # packet steps) opens this one binary copy of a USDA text room (``usd_binary_read_cache``); receipts and
    # stances name the published file by its digest.
    collision = binary_copy(assets["collision"], root / ".collider-read-cache")
    physics_collision = assets["collision"]
    owner, motion, destination_asset, seating, support = None, None, None, None, None
    if articulated:
        try:
            task = _articulated_task(revision=revision, references=references, replacement=assets["replacement"],
                                     root=root, owner_success_confirmation_scope=owner_success_confirmation_scope)
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise TeamScenePacketError("team_scene_articulated_task_invalid:" + str(exc)) from exc
        adapter, definition, task_spec = task["adapter"], task["definition"], task["task_spec"]
        confirmation = task["owner_success_confirmation"]
        task_row, state_binding, motion = task["row"], task["task_state_binding"], task["task_motion"]
        start, target, support_top = task["start"], task["target"], task["support_top_z_m"]
        subject_bounds = object_bounds = (task["subject_minimum"], task["subject_maximum"])
    else:
        adapter, seating, support = _seated_rigid_adapter(
            revision=revision, references=references, collision=collision,
            staged_destination=destination_declaration is not None)
        definition = adapter["native_task_definition"]
        task_spec = compiler._runtime_subject_task_spec(definition["task_spec"])
        task_spec["success_criteria"] = adapter["native_success_criteria"].get("criteria")
        if destination_declaration is not None:
            from .rigid_destination_task_binding import bind_destination_task_spec, stage_revision_destination
            from .task_evaluation_rigid_relocation_native_adapter import SUPPORT_PLANE_CONTRACT_PATH
            try:
                destination_asset = stage_revision_destination(revision=revision, documents=documents,
                    task_spec=task_spec, collision_path=assets["collision"], output_root=root, require_placement=True,
                    placement_qualification_path=(Path(destination_placement_qualification_path)
                        if destination_placement_qualification_path is not None else None),
                    expected_execution_binding=profile.get("destination_execution_binding"))
                assert destination_asset is not None
                task_spec = bind_destination_task_spec(task_spec, destination_asset,
                    source_support=adapter["source_documents"]["documents"]["support_plane"],
                    support_plane_digest=_sha256(Path(documents[SUPPORT_PLANE_CONTRACT_PATH])))
            except (ValueError, KeyError, TypeError) as exc:
                raise TeamScenePacketError("team_scene_destination_qualification_invalid:" + str(exc)) from exc
        task_spec, confirmation = _owner_confirmed(task_spec, revision=revision,
                                                   scope=owner_success_confirmation_scope)
        try:
            owner = _derive_configured_owner_success_contract(task_spec, site_id=scene_id, task_id=task_id,
                                                             team_namespace=revision.get("team_namespace"))
        except TaskNeutralScoringError as exc:
            raise TeamScenePacketError("team_scene_owner_contract_invalid:" + ",".join(exc.errors)) from exc
        if owner is not None:
            task_spec.update(task_success_contract=owner, task_success_contract_digest=owner["contract_digest"])
        # The support the physics runs on: planarized to the success plane when the task is on one
        # support (a derived collider every reader below and the packet use), and checked either way.
        derived = _stage_support_pad(support, task_spec=task_spec, sealed_collision=assets["collision"], root=root)
        if derived is not None:
            physics_collision = collision = derived
        pose = definition["task_object_pose_world"]
        task_row = {**compiler.evidence_asset_row(assets["replacement"], root=root, semantic_role="task_object",
                                                  pose_world=pose),
                    "asset_id": task_spec["subject_asset_id"], "source_asset_id": task_spec["source_subject_identity"],
                    "object_type": "RIGID", "reset_state": {"root_pose_world": pose, "joint_positions": {}}}
        state_binding = definition.get("task_state_binding")
        start = [float(v) for v in task_spec["start_pose_world"][:3]]
        target = [float(v) for v in task_spec["target_position_world_m"]]
        support_top = float(adapter["source_documents"]["documents"]["support_plane"]["top_z_m"])
        subject = _read(documents[SOURCE_OBJECT_CONTRACT_PATH], "team_scene_subject_bounds_invalid")
        subject_bounds = object_bounds = (subject.get("aabb_min_xyz_m"), subject.get("aabb_max_xyz_m"))
        destination_support = adapter["source_documents"]["documents"]["definition"].get("destination_support_plane")
        if destination_support is not None:
            # The object ends on another surface (floor to desk): the mount is level with the
            # lower surface and the station faces the whole region from the start to the target.
            try:
                support_top = min(support_top, float(destination_support["top_z_m"]))
                half = [(float(b) - float(a)) / 2.0 for a, b in zip(*subject_bounds, strict=True)]
                subject_bounds = ([min(float(subject_bounds[0][i]), target[i] - half[i]) for i in range(3)],
                                  [max(float(subject_bounds[1][i]), target[i] + half[i]) for i in range(3)])
            except (KeyError, TypeError, ValueError) as exc:
                raise TeamScenePacketError("team_scene_destination_support_invalid") from exc
            stance_center = [(a + b) / 2.0 for a, b in zip(*subject_bounds, strict=True)]

    appearance = dict((native_appearance_materializer or compiler._materialize_native_particlefield_appearance)(
        source_path=assets["appearance"], output_root=root / "native-appearance"))
    rows = [compiler.evidence_asset_row(Path(appearance["path"]), root=root, semantic_role="scene_appearance"),
            compiler.evidence_asset_row(physics_collision, root=root, semantic_role="scene_collision"), task_row]
    if destination_asset is not None:
        rows.append({**compiler.evidence_asset_row(destination_asset["path"], root=root, semantic_role="task_support"),
            "asset_id": destination_asset["asset_id"], "source_asset_id": destination_asset["source_asset_id"],
            "object_type": "RIGID", "pose_world": destination_asset["pose_world"],
            "reset_state": {"root_pose_world": destination_asset["pose_world"], "joint_positions": {}}})

    # The station: a mount level with the task's support, on the closest clear floor facing the task,
    # out of a moving part's way. A rigid pick and place is measured from every clear stance: it takes the
    # first in the object's reach that the scripted carry clears from, else the one in reach whose object is
    # nearest, the miss flagged (``_carrying_station``), as the bundle's gate will measure it. The pedestal keeps the arm's own
    # margin from the room and the object (``PEDESTAL_SEARCH``), and a rigid task's stances rank by the
    # farther of its start and target, a footprint apart, so the carry is tried from several spots.
    whole_region = not articulated and destination_support is not None
    carry = not articulated and task_spec.get("manipulation_strategy") == "pick_and_place"
    from .scene_physics_preflight import collision_kept_loaded
    from .team_embodiment_stance import EVERY_CLEAR_STANCE, PEDESTAL_SEARCH

    # Every search below reads the collider's one binary copy, and the packet steps after them. The hold
    # keeps one parse should no copy be made.
    # The scene's own movable furniture, named to the Franka's search only for a carry (it is only reported).
    named, refused = [], []
    if carry and movable_furniture is not None and movable_furniture.get("status") == "measured":
        from .robot_access_furniture import eligible_obstacles, search_obstacles

        documents_in = adapter["source_documents"]["documents"]
        named, refused = eligible_obstacles(movable_furniture.get("record"), supports=[
            row for row in (documents_in.get("support_plane"), destination_support) if isinstance(row, Mapping)])
    access: dict[str, Any] | None = None
    with collision_kept_loaded(collision):
        try:
            place = dict(
                collision_path=collision, collision_sha256=_sha256(physics_collision), offset=[0.0, 0.0, 0.0],
                subject_center=stance_center if whole_region else start,
                subject_minimum=[float(v) for v in subject_bounds[0]],
                subject_maximum=[float(v) for v in subject_bounds[1]],
                footprint_half_extent_m=list(STATION_FOOTPRINT_HALF_EXTENT_M), root_world_z_m=support_top,
                task_motion=motion, **PEDESTAL_SEARCH,
                **({} if articulated else {"task_points_world_m": [start, target]}))
            station = ground_stance(**place, ranked=EVERY_CLEAR_STANCE if carry else 0,
                                    **({"movable_obstacles": search_obstacles(named)} if named else {}))
        except TeamEmbodimentError as exc:
            # A ValueError itself: caught first, so a named stance refusal keeps its own reason.
            raise TeamScenePacketError(",".join(exc.blockers) or "team_scene_station_inputs_invalid") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise TeamScenePacketError("team_scene_station_inputs_invalid") from exc
        ranked_stances = station if carry else None
        station, station_search = (_carrying_station(station, task_spec=task_spec, collision=collision,
                                                     profile=profile) if carry else (station, None))
        # Moving furniture is proposed only when no clear stance has the object in reach (the hard rule);
        # a scripted carry miss with the object in reach is a flag, and the half goes on.
        if (carry and station_search is not None and not station_search["cleared"]
                and not station_search.get("stances_in_reach") and movable_furniture is not None):
            access = _robot_access_proposal(
                movable_furniture, stances=ranked_stances, carry=station_search, named=named, refused=refused,
                subject_center=place["subject_center"], task_spec=task_spec, support_top=support_top,
                revision=revision, collision_sha256=place["collision_sha256"], task_id=task_id,
                support_label=str(adapter["source_documents"]["documents"]["support_plane"].get("publisher_label")
                                  or ""), scope=robot_access_scope, offer=robot_access_offer)
            if access["status"] == "proposed" and robot_access_scope is not None:
                applied = _robot_access_applied(
                    access, scope=robot_access_scope, physics_collision=physics_collision, root=root, place=place,
                    floor_z=float(station["floor_z_m"]), station=station, task_spec=task_spec, profile=profile,
                    support=support, adapter=adapter, start=start, target=target, support_top=support_top)
                if applied is not None:
                    access, derived_access = applied["record"], applied["collider"]
                    if applied["station"] is not None:
                        physics_collision = collision = derived_access
                        place, station, station_search = applied["place"], applied["station"], applied["search"]
                        rows[1] = compiler.evidence_asset_row(physics_collision, root=root,
                                                              semantic_role="scene_collision")
        stations: dict[str, dict[str, Any]] = {}
        ranked: dict[str, list[dict[str, Any]]] = {}
        if embodiment_manifests:
            # The object a tool holds: a rigid carry's own extent, or an articulated part's handle (its joint's kind).
            joints = (task_spec.get("articulation_graph") or {}).get("joints") or []
            grasp = None if not (carry or articulated) else {
                "dimensions_m": [float(b) - float(a) for a, b in zip(*object_bounds, strict=True)],
                "joint_type": next((str(row.get("joint_type")) for row in joints if row.get("role") == "target"),
                                   "unknown") if articulated else None}
            stations = _embodiment_stations(embodiment_manifests, place=place, points=[start, target], grasp=grasp,
                                            ranked=ranked)
    base = {"position_world_m": station["position_world_m"], "orientation_xyzw": station["orientation_xyzw"]}
    request: dict[str, Any] = {
        "schema_version": "native_task_arena_packet_request.v1", "scene_id": scene_id, "task_id": task_id,
        "task_spec": task_spec, "task_joint_bindings": definition.get("task_joint_bindings") or [],
        "task_state_binding": state_binding,
        **({"articulated_kinematic_path_receipt": task["kinematic_path"]} if articulated else {}),
        "assets": rows, "robot_base_pose_world": base,
        "robot_joint_reset_positions_rad": dict(profile["robot_joint_reset_positions_rad"]),
        "cameras": _cameras(profile, base=station["position_world_m"], start=start, target=target,
                            whole_region=whole_region),
        "scenario": adapter["native_episode_execution"].get("scenario"),
        "physics_frequency_hz": adapter["native_episode_execution"].get("physics_frequency_hz"),
        "configured_task_template_adapter": adapter,
        "appearance_variant": compiler.packet_appearance_variant(appearance), "request_digest": ""}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    packet = root / PACKET_DIRECTORY
    materialize_native_task_arena_packet(request=request, evidence_root=root, output_dir=packet,
                                         link_sources_within=root)
    exterior_right, placement = _exterior_right(packet, base=base, aim=start)
    camera_layout: dict[str, Any] = {"exterior_right": placement}
    if exterior_right is not None:
        packet_field = "measured_on_packet" if placement["status"] in {"placed", "limited"} else "estimated_from_packet"
        camera_layout[packet_field] = _record(packet / "native_task_arena_packet_receipt.v1.json")
        request = {**request, "cameras": [*request["cameras"], exterior_right], "request_digest": ""}
        request["request_digest"] = canonical_digest(request, digest_field="request_digest")
        packet = root / THREE_VIEW_PACKET_DIRECTORY
        materialize_native_task_arena_packet(request=request, evidence_root=root, output_dir=packet,
                                             link_sources_within=root)
    plan = _read(packet / "native_task_arena_scene_plan.v1.json", "team_scene_packet_invalid")
    receipt = packet / "native_task_arena_packet_receipt.v1.json"
    task_kind = "articulated_open_close" if articulated else "rigid_pick_place"
    if stations and embodiment_models:
        from .team_embodiment import validate_embodiment_manifest

        bodies = {row["embodiment_digest"]: row for row in map(validate_embodiment_manifest, embodiment_manifests)}
        _own_arm_stations(stations, ranked, plan={**plan, "task_kind": task_kind}, manifests=bodies,
                          models=embodiment_models)
    if articulated and (plan["task_spec"].get("configured_success_criteria") or {}).get(
            "owner_success_contract_required") is True:
        # The owner's frozen criteria, translated without changing scoring (as the operator canary seals them).
        from .task_evaluation_scene_control_omission import confirmed_articulated_contract

        try:
            owner = confirmed_articulated_contract(task_spec=plan["task_spec"], site_id=str(plan["scene_id"]),
                                                   task_id=str(plan["task_id"]))
        except ValueError as exc:
            raise TeamScenePacketError("team_scene_owner_contract_invalid:" + str(exc)) from exc
    try:
        contract = owner or seal_task_success_contract(
            task_kind=task_kind, task_spec=plan["task_spec"], site_id=str(plan["scene_id"]),
            task_id=str(plan["task_id"]), author_source="compatibility_default",
            author_id="blueprint:manipulation_strategy_defaults.v1", confirmation_status="confirmed")
        contract = validate_task_success_contract(contract, task_kind=task_kind)
    except TaskNeutralScoringError as exc:
        raise TeamScenePacketError("team_scene_success_contract_invalid:" + ",".join(exc.errors)) from exc
    packet_record = _record(receipt)
    value = {
        "schema_version": SCHEMA, "claim_ceiling": CLAIM_CEILING,
        "scene_revision_digest": revision["revision_digest"],
        "configuration_digest": canonical_digest({"scene_revision_digest": revision["revision_digest"],
                                                  "packet": packet_record["sha256"],
                                                  "station": station["stance_digest"]}),
        "plan_digest": plan["plan_digest"], "task_success_contract": contract,
        "task_success_contract_digest": contract["contract_digest"],
        "base_native_packet": packet_record, "runtime_source": runtime_source, "construction_result": packet_record,
        "execution_authority": dict(EXECUTION_AUTHORITY), "control_diagnostic": dict(PENDING_CONTROLS),
        "robot_station": station, "camera_layout": camera_layout, "droid_profile_digest": droid_profile_digest,
        **({"robot_station_search": station_search} if station_search is not None else {}),
        **({"task_support_seating": seating} if seating is not None else {}),
        **({"task_support_pad": support} if support is not None else {}),
        **({"embodiment_stations": stations} if stations else {}),
        **({"robot_access_removal": access} if access is not None else {}),
        "inputs_digest": ""}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    path = root / RECORD_FILENAME
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return {"path": str(path), "inputs_digest": value["inputs_digest"], "scene_id": scene_id, "task_id": task_id,
            "robot_station": station,
            **({"owner_success_confirmation": confirmation} if confirmation is not None else {}),
            **({"robot_access_removal": {key: access[key] for key in ("status", "reason", "confirmation_digest",
                                                                      "binding_digest") if key in access}}
               if access is not None else {})}


def read_team_scene_inputs(path: str | Path) -> dict[str, Any]:
    """The scene half a team run stages from (``team_candidate_quick10_staging.SCENE_INPUT_FIELDS``), or raise."""

    from .team_candidate_quick10_staging import SCENE_INPUT_FIELDS

    value = _read(Path(path), "team_scene_inputs_invalid")
    if (value.get("schema_version") != SCHEMA or value.get("claim_ceiling") != CLAIM_CEILING
            or value.get("inputs_digest") != canonical_digest(value, digest_field="inputs_digest")
            or value.get("construction_result") != value.get("base_native_packet")):
        raise TeamScenePacketError("team_scene_inputs_invalid")
    for field in ("base_native_packet", "runtime_source"):
        record = value.get(field) or {}
        if not Path(str(record.get("path") or "")).is_file() or _record(Path(record["path"])) != record:
            raise TeamScenePacketError(f"team_scene_inputs_{field}_changed")
    return {**{key: value[key] for key in SCENE_INPUT_FIELDS if key in value},
            "control_diagnostic": dict(value["control_diagnostic"])}


def recorded_embodiment_stations(path: str | Path | None, *, plan_digest: Any) -> dict[str, dict[str, Any]]:
    """The registered bodies' own stations a scene half was compiled with, keyed by embodiment digest.

    Empty when the half has none, cannot be read, or was compiled for another scene plan.
    """

    if not path:
        return {}
    try:
        read_team_scene_inputs(path)
        value = _read(Path(path), "team_scene_inputs_invalid")
    except (OSError, ValueError, KeyError):
        return {}
    stations = value.get("embodiment_stations")
    return dict(stations) if value.get("plan_digest") == plan_digest and isinstance(stations, Mapping) else {}


def recorded_station_search(path: str | Path | None, *, plan_digest: Any) -> dict[str, Any]:
    """The packet station and the carry search that chose it, as the scene half recorded them.

    ``{"robot_station", "robot_station_search"}``; empty when the half has no search (an articulated
    task, an older half), cannot be read, or was compiled for another scene plan.
    """

    if not path:
        return {}
    try:
        read_team_scene_inputs(path)
        value = _read(Path(path), "team_scene_inputs_invalid")
    except (OSError, ValueError, KeyError):
        return {}
    if value.get("plan_digest") != plan_digest or not isinstance(value.get("robot_station_search"), Mapping):
        return {}
    return {"robot_station": dict(value["robot_station"]), "robot_station_search": dict(value["robot_station_search"])}


SCENE_DIRECTORY = "scene"


def fetch_team_scene_sources(*, run_root: str | Path, output_root: str | Path,
                             fetcher: Callable[[Mapping[str, Any]], bytes] | None = None
                             ) -> tuple[dict[str, Any], dict[str, Path]]:
    """A completed source launch's published revision, and its documents fetched under ``output_root``."""

    from .robot_free_readiness_retention import _revision
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .task_evaluation_scene_rigid_robot_free_readiness import fetch_revision_documents

    result, _receipt, _zero = _validate_source(Path(run_root))
    revision = _revision(result["configured_scene_revision_reference"],
                         revision_digest=result["configured_scene_revision_digest"], fetcher=fetcher)
    documents = fetch_revision_documents(revision=revision, output_root=Path(output_root), fetcher=fetcher)
    return revision, documents


def with_movable_furniture(packet_inputs: Mapping[str, Any], *, revision: Mapping[str, Any]) -> dict[str, Any]:
    """``packet_inputs`` with what the request's intent knows of its scene's movable furniture, read on the host.

    A compile given the request's own intent as ``robot_access_scope`` also reads what that intent's current
    preparation knows of its movable furniture (``robot_access_measurement.furniture_for_scope``: the
    owner-approved sidecar, if any), unless the caller already passed ``movable_furniture``.
    """

    packet_inputs = dict(packet_inputs)
    if packet_inputs.get("robot_access_scope") is not None and "movable_furniture" not in packet_inputs:
        from .robot_access_measurement import furniture_for_scope

        known = furniture_for_scope(packet_inputs["robot_access_scope"], revision_digest=str(revision["revision_digest"]))
        # The offer to measure is made only when a carry fails, so it is read then (``_robot_access_proposal``).
        packet_inputs["movable_furniture"] = {key: value for key, value in known.items() if key != "offer"}
    return packet_inputs


def record_team_scene(compiled: Mapping[str, Any], *, run_root: str | Path, capture_root: str | Path,
                      scenario_id: str, scene_revision_digest: str) -> dict[str, Any]:
    """Record a compiled scene half where the team registry reads it, and queue its robot-free probe.

    The launch's run root is also where its robot-free probe receipt is retained
    (``robot_free_readiness_retention``), so it is the scene's probe root.
    """

    from .team_candidate_quick10_registry import write_derived_profile

    descriptor = _read(Path(capture_root) / "capture_descriptor.json", "team_scene_capture_descriptor_invalid")
    written = write_derived_profile({
        "capture_root": str(Path(capture_root).resolve()), "scene_id": str(descriptor.get("scene_id") or ""),
        "task_id": compiled["task_id"], "scenario_id": scenario_id,
        "robot_free_probe_root": str(Path(run_root).resolve()),
        "scene_revision_digest": scene_revision_digest, "team_scene_inputs_path": compiled["path"]})
    probe = {"status": "profile_not_written", "queued": False}
    if written.get("status") == "written":
        from .rigid_robot_free_probe_runner import enqueue_rigid_robot_free_probe

        probe = enqueue_rigid_robot_free_probe(capture_root)
    return {**compiled, "registry": written, "robot_free_probe": probe}


def prepare_team_scene(*, run_root: str | Path, capture_root: str | Path, scenario_id: str,
                       output_root: str | Path, fetcher: Callable[[Mapping[str, Any]], bytes] | None = None,
                       **packet_inputs: Any) -> dict[str, Any]:
    """Compile a completed source launch's published scene and record it where the team registry reads it.

    Three steps, each its own function so a remote CPU worker can run the middle one
    (``team_scene_packet_remote``): fetch the revision and its documents under ``output_root`` (and what the
    intent knows of its movable furniture, ``with_movable_furniture``), compile ``output_root/scene``, record
    the derived row.
    """

    root = Path(output_root)
    revision, documents = fetch_team_scene_sources(run_root=run_root, output_root=root, fetcher=fetcher)
    compiled = materialize_team_scene(revision=revision, documents=documents, output_root=root / SCENE_DIRECTORY,
                                      **with_movable_furniture(packet_inputs, revision=revision))
    return record_team_scene(compiled, run_root=run_root, capture_root=capture_root, scenario_id=scenario_id,
                             scene_revision_digest=revision["revision_digest"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("run-root", "capture-root", "scenario-id", "output-root", "configured-scene-bundle",
                 "droid-profile", "droid-profile-digest", "runtime-source-receipt"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--destination-placement-qualification")
    args = parser.parse_args(argv)
    result = prepare_team_scene(
        run_root=args.run_root, capture_root=args.capture_root, scenario_id=args.scenario_id,
        output_root=args.output_root, configured_scene_bundle_path=args.configured_scene_bundle,
        droid_profile_path=args.droid_profile, droid_profile_digest=args.droid_profile_digest,
        runtime_source_receipt_path=args.runtime_source_receipt,
        destination_placement_qualification_path=args.destination_placement_qualification)
    if result["registry"].get("status") == "written":
        # The job lists the evaluation once its offer is republished with the scene half.
        from .agent_execution_offer import publish_agent_execution_offer

        result["offer"] = publish_agent_execution_offer(Path(args.capture_root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if (result.get("offer") or {}).get("status") == "published" else 1


__all__ = [
    "CLAIM_CEILING",
    "RECORD_FILENAME",
    "SCENE_DIRECTORY",
    "SCHEMA",
    "STATION_FOOTPRINT_HALF_EXTENT_M",
    "TeamScenePacketError",
    "fetch_team_scene_sources",
    "materialize_team_scene",
    "prepare_team_scene",
    "read_team_scene_inputs",
    "record_team_scene",
    "recorded_embodiment_stations",
    "recorded_station_search",
    "with_movable_furniture",
]


if __name__ == "__main__":
    raise SystemExit(main())
