"""Before any construction, whether an enabled embodiment can carry a rigid task at all (ADP-009D).

The station is the one the team packet takes (``team_scene_packet``): a mount
level with the lower of the task's supports, on the closest clear floor, facing
the start or, when the object ends on another surface, the whole region from
the start to the target. From one of the closest clear stances the arm must
carry the object with one held grasp, clear of the reconstructed scene with its
hand's body measured (``robot_arm_path_clearance.rigid_pick_place_clearance``).
Grasps are tried best first: top-down, then tilted and side-on from four
directions. Every embodiment an enabled screening candidate runs on is checked
(``config/beta_policy_candidates.v1.json``); one the gate has no arm model of
is unassessed, never refused. A refusal names the furthest phase any stance
reached, before the owner is asked to pay for the scene build. A pass means
one such carry exists, not that a policy finds it.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest

SCHEMA = "website_rigid_task_capability_verdict.v1"
#: The arm model each enabled candidate's robot profile runs on; a profile not named here is unassessed.
ROBOT_PROFILE_ARMS = {"droid_franka_panda_robotiq_2f85_v1": "franka_panda"}
DEFAULT_PREGRASP_CLEARANCE_M = 0.10
#: After top-down, the hand tilts this far from vertical, approaching from each of these world directions.
GRASP_TILTS_DEG = (45.0, 90.0)
GRASP_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)


def enabled_robot_profiles() -> list[str]:
    """The robot profiles of the sealed catalog's enabled screening candidates."""
    from .beta_policy_candidate_catalog import load_default_catalog

    return sorted({(row.get("interface") or {}).get("robot_profile_id")
                   for row in load_default_catalog()["candidates"]
                   if (row.get("eligibility") or {}).get("enabled")
                   and (row.get("interface") or {}).get("robot_profile_id")})


def _grasps(lower: np.ndarray, upper: np.ndarray) -> list[dict[str, Any]]:
    """Top-down closing across the narrower side, then tilted and side-on, each closing horizontally."""
    from .native_franka_action_math import grasp_orientation_contact_xyzw

    jaw = [1.0, 0.0, 0.0] if upper[0] - lower[0] <= upper[1] - lower[1] else [0.0, 1.0, 0.0]
    rows = [{"name": "top_down", "approach_unit": [0.0, 0.0, 1.0],
             "gripper_orientation_xyzw": grasp_orientation_contact_xyzw(approach_axis=[0.0, 0.0, -1.0], jaw_axis=jaw)}]
    for tilt in GRASP_TILTS_DEG:
        for yaw in GRASP_YAWS_DEG:
            t, y = math.radians(tilt), math.radians(yaw)
            outward = [math.sin(t) * math.cos(y), math.sin(t) * math.sin(y), math.cos(t)]
            rows.append({"name": f"tilt{tilt:g}_yaw{yaw:g}", "approach_unit": outward,
                         "gripper_orientation_xyzw": grasp_orientation_contact_xyzw(
                             approach_axis=[-v for v in outward], jaw_axis=[-math.sin(y), math.cos(y), 0.0])})
    return rows


def rigid_task_verdict(*, preparation: Mapping[str, Any], collision_path: Path,
                       robot_profiles: Sequence[str] | None = None) -> dict[str, Any]:
    """The station and carry verdict for a prepared rigid pick-and-place, from CPU geometry only.

    ``robot_profiles`` defaults to the enabled candidates' (``enabled_robot_profiles``).
    """
    import hashlib

    from .robot_arm_path_clearance import CARRY_STATIONS, NOMINAL_SEED_RAD, REST_SEED_RAD, first_carrying_stance
    from .team_embodiment import TeamEmbodimentError
    from .team_embodiment_stance import ground_stance
    from .team_scene_packet import STATION_FOOTPRINT_HALF_EXTENT_M

    task = preparation["intake_request"]["task"]
    lower = np.asarray(preparation["subject"]["aabb_min_xyz"], dtype=float)
    upper = np.asarray(preparation["subject"]["aabb_max_xyz"], dtype=float)
    destination = preparation["destination"]
    start, half = (lower + upper) / 2.0, (upper - lower) / 2.0
    target = np.asarray(destination["position_world_m"], dtype=float) + [0.0, 0.0, half[2]]
    supports = [task["support"]] + ([destination["support"]] if "support" in destination else [])
    region_lower, region_upper = (np.minimum(lower, target - half), np.maximum(upper, target + half)) \
        if "support" in destination else (lower, upper)
    collision = Path(collision_path)
    digest = "sha256:" + hashlib.sha256(collision.read_bytes()).hexdigest()
    profiles = sorted(enabled_robot_profiles() if robot_profiles is None else robot_profiles)
    arms = sorted({ROBOT_PROFILE_ARMS[p] for p in profiles if p in ROBOT_PROFILE_ARMS})
    unassessed = [p for p in profiles if p not in ROBOT_PROFILE_ARMS]
    value: dict[str, Any] = {"schema_version": SCHEMA, "preparation_digest": preparation["digest"],
                             "robot_id": arms[0] if arms else None, "robot_profiles": profiles,
                             "unassessed_robot_profiles": unassessed, "collision_sha256": digest,
                             "grasp_basis": "one_held_grasp_top_down_first_then_tilted_and_side_on",
                             "mount_rule": "level_with_lower_support", "claim_ceiling": "development_only",
                             "physical_world_truth_claimed": False}

    def settle(status: str, blockers: list[str], **fields: Any) -> dict[str, Any]:
        # An embodiment the gate cannot measure may still carry the task: unassessed, never refused.
        if status == "abstained" and unassessed:
            status, blockers = "unassessed", blockers + ["robot_profile_unmodeled:" + p for p in unassessed]
        value.update(status=status, blockers=blockers, **fields)
        value["verdict_digest"] = canonical_digest(value, digest_field="verdict_digest")
        return value

    if not arms:
        return settle("abstained", [] if profiles else ["robot_profile_none_enabled"], station=None, reach=None,
                      embodiments=[])
    try:
        stations = ground_stance(collision_path=collision, collision_sha256=digest, offset=[0.0, 0.0, 0.0],
            subject_center=((region_lower + region_upper) / 2.0).tolist(), subject_minimum=region_lower.tolist(),
            subject_maximum=region_upper.tolist(), footprint_half_extent_m=list(STATION_FOOTPRINT_HALF_EXTENT_M),
            root_world_z_m=min(float(row["aabb_max_xyz"][2]) for row in supports), ranked=CARRY_STATIONS)
    except TeamEmbodimentError as exc:
        return settle("abstained", ["robot_station_" + item for item in exc.blockers], station=None, reach=None,
                      embodiments=[])
    carry = {}
    if destination.get("relation") == "inside":
        carry["carry_height_world_m"] = (float(destination["interior_bounds_world_m"]["maximum"][2])
                                         + float(half[2]) + 0.08)
    results = []
    for arm in arms:
        cleared, reach, attempts = first_carrying_stance(stations, collision_path=collision, robot_id=arm,
            seeds=[list(NOMINAL_SEED_RAD), list(REST_SEED_RAD[arm])], grasps=_grasps(lower, upper),
            start_pose_world=[*start.tolist(), 0.0, 0.0, 0.0, 1.0], target_position_world=target.tolist(),
            contact_point_m=[0.0, 0.0, 0.0],
            pregrasp_clearance_m=float((task.get("success") or {}).get("pregrasp_clearance_m")
                                       or DEFAULT_PREGRASP_CLEARANCE_M), **carry)
        station = cleared or next(s for s, r in attempts if r is reach)
        results.append((arm, station, reach, attempts))
        if reach["status"] == "clear":
            break
    arm, station, reach, attempts = next((row for row in results if row[2]["status"] == "clear"), results[0])
    embodiments = [{"robot_id": a, "status": r["status"], "blockers": list(r["blockers"]),
                    "grasp": r.get("grasp")} for a, _s, r, _t in results]
    blockers = list(reach["blockers"]) if len(arms) == 1 else [
        f"{a}:{blocker}" for a, _s, r, _t in results for blocker in r["blockers"]]
    value["robot_id"] = arm
    return settle("clear" if reach["status"] == "clear" else "abstained", blockers, station=station, reach=reach,
                  embodiments=embodiments, stations_tried=len({id(s) for s, _r in attempts}),
                  grasps_tried=len({r.get("grasp") for _s, r in attempts}),
                  station_results=[{"position_world_m": s["position_world_m"], "yaw_rad": s["yaw_rad"],
                                    "grasp": r.get("grasp"), "status": r["status"], "blockers": r["blockers"]}
                                   for s, r in attempts])
