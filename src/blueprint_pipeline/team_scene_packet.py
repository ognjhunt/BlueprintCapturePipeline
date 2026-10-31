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

SCHEMA = "blueprint.team_scene_inputs.v1"
CLAIM_CEILING = "diagnostic_policy_execution"
RECORD_FILENAME = "team_scene_inputs.v1.json"
PACKET_DIRECTORY = "native-task-packet"
OWN_ARM_RULE = "closest_clear_stance_own_arm_reaches_every_phase"
# The same scene compiled with the right exterior camera, when one is placed.
THREE_VIEW_PACKET_DIRECTORY = "native-task-packet-three-view"
# A pedestal for an arm's base: its half extent on the floor, metres.
STATION_FOOTPRINT_HALF_EXTENT_M = (0.15, 0.15)
EXECUTION_AUTHORITY = {"maximum_provider_allocations": 1, "retry_cap": 0,
                       "single_warm_provider_session_required": True,
                       "caller_surviving_watchdog_required": True,
                       "billing_teardown_provider_zero_required": True}
# Controls are never run before a team evaluation; a contract that requires them per cell is refused
# by the registry (``team_candidate_strict_controls_runtime_unavailable``).
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


def _carrying_station(stances: list[dict[str, Any]], *, task_spec: Mapping[str, Any], collision: Path,
                      profile: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The first ranked stance the Franka can carry the object from (``robot_arm_path_clearance``).

    With none, the closest stance stands and the bundle's gate refuses the packet by name before any GPU run.
    """
    from .robot_arm_path_clearance import NOMINAL_SEED_RAD, REST_SEED_RAD, first_carrying_stance

    try:
        affordance = task_spec["interaction_affordance"]
        reset = profile["robot_joint_reset_positions_rad"]
        cleared, _, attempts = first_carrying_stance(stances, collision_path=collision, robot_id="franka_panda",
            seeds=[[float(reset[f"panda_joint{i}"]) for i in range(1, 8)], list(NOMINAL_SEED_RAD),
                   list(REST_SEED_RAD["franka_panda"])],
            start_pose_world=task_spec["start_pose_world"], target_position_world=task_spec["target_position_world_m"],
            contact_point_m=affordance["contact_point_scoring_frame_m"],
            approach_unit=affordance["approach_unit_scoring_frame"],
            gripper_orientation_xyzw=affordance["gripper_orientation_scoring_frame_xyzw"],
            pregrasp_clearance_m=float(affordance["pregrasp_clearance_m"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise TeamScenePacketError("team_scene_carry_inputs_invalid") from exc
    station = cleared or stances[0]
    return station, {"rule": "first_ranked_stance_that_carries", "cleared": cleared is not None,
                     "chosen_rank": next(i for i, row in enumerate(stances) if row is station),
                     "results": [{"stance_digest": row["stance_digest"], "status": report["status"],
                                  "blockers": report["blockers"]} for row, report in attempts]}


def _embodiment_stations(manifests: Sequence[Mapping[str, Any]], *, place: Mapping[str, Any],
                         points: Sequence[Sequence[float]], grasp: Mapping[str, Any] | None,
                         ranked: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, dict[str, Any]]:
    """Each registered body's own station on this scene, from its own manifest, keyed by its digest.

    ``place`` is the Franka station's search (``ground_stance``): the same floor, task and moving part. A fixed
    base stands on the station's pedestal level with the task's support and takes the closest clear stance
    from which its declared ``reach_m`` covers every task point; with none, the closest stands, out of reach.
    A free base stands on its own footprint at its own root height and declares no reach. No Franka model or
    number enters: a moving part is kept off the mount or footprint, and the body itself, its reach without
    one, and a grip force no manifest declares stay unassessed. Each declared tool is screened against the
    ``grasp`` (``website_task_preparation.screen_physics``, or the handle screen for a jointed part). A fixed
    or free base's clear stances, closest first, are kept in ``ranked`` for ``_own_arm_stations``.
    """
    from .team_embodiment import ROBOT_ID, TeamEmbodimentError, validate_embodiment_manifest
    from .team_embodiment_stance import GRID_STEP_M, SEARCH_RADIUS_M, ground_stance
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
                stances = ground_stance(**{**search, "root_world_z_m": None,
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
                   **({"reason": report["reason"]} if "reason" in report else {})}
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
                      replacement: Path, root: Path) -> dict[str, Any]:
    """The revision's open/close task as the website episode compiler builds it, plus its part's motion."""

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


def materialize_team_scene(*, revision: Mapping[str, Any], documents: Mapping[str, Path],
                           configured_scene_bundle_path: str | Path, droid_profile_path: str | Path,
                           droid_profile_digest: str, runtime_source_receipt_path: str | Path,
                           output_root: str | Path,
                           destination_placement_qualification_path: str | Path | None = None,
                           native_appearance_materializer: Callable[..., Mapping[str, Any]] | None = None,
                           embodiment_manifests: Sequence[Mapping[str, Any]] = (),
                           embodiment_models: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
    """Compile the revision's scene half under ``output_root`` and seal its team scene inputs.

    Each of ``embodiment_manifests`` (registered team bodies) also gets its own station on the same scene
    (``_embodiment_stations``); without any, the record is exactly the Franka-only one. A body whose
    robot model is in ``embodiment_models`` (by embodiment digest) has that station chosen with its own
    arm (``_own_arm_stations``).
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
        adapt_rigid_relocation_task_template,
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
    owner, motion, destination_asset = None, None, None
    if articulated:
        try:
            task = _articulated_task(revision=revision, references=references, replacement=assets["replacement"],
                                     root=root)
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise TeamScenePacketError("team_scene_articulated_task_invalid:" + str(exc)) from exc
        adapter, definition, task_spec = task["adapter"], task["definition"], task["task_spec"]
        task_row, state_binding, motion = task["row"], task["task_state_binding"], task["task_motion"]
        start, target, support_top = task["start"], task["target"], task["support_top_z_m"]
        subject_bounds = object_bounds = (task["subject_minimum"], task["subject_maximum"])
    else:
        adapter = adapt_rigid_relocation_task_template(configured_revision=revision,
                                                       materialized_references=references)
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
        try:
            owner = _derive_configured_owner_success_contract(task_spec, site_id=scene_id, task_id=task_id,
                                                             team_namespace=revision.get("team_namespace"))
        except TaskNeutralScoringError as exc:
            raise TeamScenePacketError("team_scene_owner_contract_invalid:" + ",".join(exc.errors)) from exc
        if owner is not None:
            task_spec.update(task_success_contract=owner, task_success_contract_digest=owner["contract_digest"])
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
            compiler.evidence_asset_row(assets["collision"], root=root, semantic_role="scene_collision"), task_row]
    if destination_asset is not None:
        rows.append({**compiler.evidence_asset_row(destination_asset["path"], root=root, semantic_role="task_support"),
            "asset_id": destination_asset["asset_id"], "source_asset_id": destination_asset["source_asset_id"],
            "object_type": "RIGID", "pose_world": destination_asset["pose_world"],
            "reset_state": {"root_pose_world": destination_asset["pose_world"], "joint_positions": {}}})

    # The station: a mount level with the task's support, on the closest clear floor facing the task,
    # out of a moving part's way. A rigid pick and place takes the first such stance the arm can carry
    # the object from, measured as the bundle's gate will measure it.
    whole_region = not articulated and destination_support is not None
    carry = not articulated and task_spec.get("manipulation_strategy") == "pick_and_place"
    from .robot_arm_path_clearance import CARRY_STATIONS
    try:
        place = dict(
            collision_path=assets["collision"], collision_sha256=_sha256(assets["collision"]), offset=[0.0, 0.0, 0.0],
            subject_center=stance_center if whole_region else start,
            subject_minimum=[float(v) for v in subject_bounds[0]],
            subject_maximum=[float(v) for v in subject_bounds[1]],
            footprint_half_extent_m=list(STATION_FOOTPRINT_HALF_EXTENT_M), root_world_z_m=support_top,
            task_motion=motion)
        station = ground_stance(**place, ranked=CARRY_STATIONS if carry else 0)
    except TeamEmbodimentError as exc:
        # A ValueError itself: caught first, so a named stance refusal keeps its own reason.
        raise TeamScenePacketError(",".join(exc.blockers) or "team_scene_station_inputs_invalid") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise TeamScenePacketError("team_scene_station_inputs_invalid") from exc
    station, station_search = (_carrying_station(station, task_spec=task_spec, collision=assets["collision"],
                                                 profile=profile) if carry else (station, None))
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
        **({"embodiment_stations": stations} if stations else {}),
        "inputs_digest": ""}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    path = root / RECORD_FILENAME
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return {"path": str(path), "inputs_digest": value["inputs_digest"], "scene_id": scene_id, "task_id": task_id,
            "robot_station": station}


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


def prepare_team_scene(*, run_root: str | Path, capture_root: str | Path, scenario_id: str,
                       output_root: str | Path, fetcher: Callable[[Mapping[str, Any]], bytes] | None = None,
                       **packet_inputs: Any) -> dict[str, Any]:
    """Compile a completed source launch's published scene and record it where the team registry reads it.

    The launch's run root is also where its robot-free probe receipt is retained
    (``robot_free_readiness_retention``), so it is the scene's probe root.
    """

    from .robot_free_readiness_retention import _revision
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .task_evaluation_scene_rigid_robot_free_readiness import fetch_revision_documents
    from .team_candidate_quick10_registry import write_derived_profile

    result, _receipt, _zero = _validate_source(Path(run_root))
    revision = _revision(result["configured_scene_revision_reference"],
                         revision_digest=result["configured_scene_revision_digest"], fetcher=fetcher)
    root = Path(output_root)
    documents = fetch_revision_documents(revision=revision, output_root=root, fetcher=fetcher)
    compiled = materialize_team_scene(revision=revision, documents=documents, output_root=root / "scene",
                                      **packet_inputs)
    descriptor = _read(Path(capture_root) / "capture_descriptor.json", "team_scene_capture_descriptor_invalid")
    written = write_derived_profile({
        "capture_root": str(Path(capture_root).resolve()), "scene_id": str(descriptor.get("scene_id") or ""),
        "task_id": compiled["task_id"], "scenario_id": scenario_id,
        "robot_free_probe_root": str(Path(run_root).resolve()),
        "scene_revision_digest": revision["revision_digest"], "team_scene_inputs_path": compiled["path"]})
    probe = {"status": "profile_not_written", "queued": False}
    if written.get("status") == "written":
        from .rigid_robot_free_probe_runner import enqueue_rigid_robot_free_probe

        probe = enqueue_rigid_robot_free_probe(capture_root)
    return {**compiled, "registry": written, "robot_free_probe": probe}


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
    "SCHEMA",
    "STATION_FOOTPRINT_HALF_EXTENT_M",
    "TeamScenePacketError",
    "materialize_team_scene",
    "prepare_team_scene",
    "read_team_scene_inputs",
    "recorded_embodiment_stations",
]


if __name__ == "__main__":
    raise SystemExit(main())
