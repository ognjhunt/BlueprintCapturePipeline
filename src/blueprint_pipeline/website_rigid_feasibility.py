"""Before any construction, whether an enabled embodiment can carry a rigid task at all (ADP-009D).

The station is the one the team packet takes (``team_scene_packet``): a mount
level with the lower of the task's supports, on the closest clear floor, facing
the start or, when the object ends on another surface, the whole region from
the start to the target. The hard reach rule decides: from some clear stance,
some point of the object must lie within the arm's chain reach bound at its
start and at its place (``robot_arm_path_clearance.rigid_reach_shortfall``); a
learned policy chooses its own grasp, so that is all a refusal may rest on.
With no clear stance in reach the verdict abstains with
``robot_reach_insufficient:<phase>:<distance>m_vs_<bound>m`` before the owner is
asked to pay for the scene build. The scripted carry is evidence: from the
closest clear stances in reach the arm carries the object with one held grasp,
clear of the reconstructed scene with its hand's body measured
(``robot_arm_path_clearance.rigid_pick_place_clearance``), grasps tried best
first: top-down, then tilted and side-on from four directions. A carry found is
``clear``; none found, with the object in reach, is ``flagged``
(``scripted_carry_path_not_found``, with the attempt that got furthest) and the
job goes on. Every embodiment an enabled screening candidate runs on is checked
(``config/beta_policy_candidates.v1.json``); one the gate has no arm model of
is unassessed, never refused. A pass means the arm can reach the object, not
that a policy finds a grasp. A target region that evaluation's surface-target
admission refused in preparation travels with the verdict as a risk; it never
changes the status.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .usd_binary_read_cache import read_scoped

SCHEMA = "website_rigid_task_capability_verdict.v1"
DEFAULT_PREGRASP_CLEARANCE_M = 0.10


def enabled_robot_profiles() -> list[str]:
    """The robot profiles of the sealed catalog's enabled screening candidates."""
    from .beta_policy_candidate_catalog import load_default_catalog

    return sorted({(row.get("interface") or {}).get("robot_profile_id")
                   for row in load_default_catalog()["candidates"]
                   if (row.get("eligibility") or {}).get("enabled")
                   and (row.get("interface") or {}).get("robot_profile_id")})


def _grasps(lower: np.ndarray, upper: np.ndarray) -> list[dict[str, Any]]:
    """Top-down closing across the narrower side, then tilted and side-on, each closing horizontally."""
    from .robot_arm_path_clearance import intake_grasps

    return intake_grasps(lower, upper)


def carry_station_search(preparation: Mapping[str, Any]) -> dict[str, Any]:
    """The ``ground_stance`` arguments the verdict's stations come from, less the collision binding.

    A mount level with the lower of the task's supports, facing the subject or, when the object ends
    on another surface, the whole region from its start to its target; the ``CARRY_STATIONS`` clear
    stances nearest the farther of the start and the target, a footprint apart, with the pedestal's
    own clearance (``team_embodiment_stance.PEDESTAL_SEARCH``), as the team packet searches. The robot
    fit table (``website_robot_fit``) searches the same way.
    """
    from .robot_arm_path_clearance import CARRY_STATIONS
    from .team_embodiment_stance import PEDESTAL_SEARCH
    from .team_scene_packet import STATION_FOOTPRINT_HALF_EXTENT_M

    task, destination = preparation["intake_request"]["task"], preparation["destination"]
    lower = np.asarray(preparation["subject"]["aabb_min_xyz"], dtype=float)
    upper = np.asarray(preparation["subject"]["aabb_max_xyz"], dtype=float)
    half = (upper - lower) / 2.0
    target = np.asarray(destination["position_world_m"], dtype=float) + [0.0, 0.0, half[2]]
    supports = [task["support"]] + ([destination["support"]] if "support" in destination else [])
    region_lower, region_upper = (np.minimum(lower, target - half), np.maximum(upper, target + half)) \
        if "support" in destination else (lower, upper)
    return {"offset": [0.0, 0.0, 0.0], "subject_center": ((region_lower + region_upper) / 2.0).tolist(),
            "subject_minimum": region_lower.tolist(), "subject_maximum": region_upper.tolist(),
            "footprint_half_extent_m": list(STATION_FOOTPRINT_HALF_EXTENT_M),
            "root_world_z_m": min(float(row["aabb_max_xyz"][2]) for row in supports), "ranked": CARRY_STATIONS,
            "task_points_world_m": [((lower + upper) / 2.0).tolist(), target.tolist()], **PEDESTAL_SEARCH}


@read_scoped
def rigid_task_verdict(*, preparation: Mapping[str, Any], collision_path: Path,
                       robot_profiles: Sequence[str] | None = None) -> dict[str, Any]:
    """The station and carry verdict for a prepared rigid pick-and-place, from CPU geometry only.

    ``robot_profiles`` defaults to the enabled candidates' (``enabled_robot_profiles``).
    """
    import hashlib

    from .robot_arm_path_clearance import (
        CARRY_STATIONS,
        NOMINAL_SEED_RAD,
        REST_SEED_RAD,
        ROBOT_PROFILE_ARMS,
        SCRIPTED_CARRY_FLAG,
        RobotArmPathClearanceError,
        first_carrying_stance,
        reach_shortfall_code,
        rigid_reach_shortfall,
    )
    from .team_embodiment import TeamEmbodimentError
    from .team_embodiment_stance import EVERY_CLEAR_STANCE, ground_stance

    task = preparation["intake_request"]["task"]
    lower = np.asarray(preparation["subject"]["aabb_min_xyz"], dtype=float)
    upper = np.asarray(preparation["subject"]["aabb_max_xyz"], dtype=float)
    destination = preparation["destination"]
    start, half = (lower + upper) / 2.0, (upper - lower) / 2.0
    target = np.asarray(destination["position_world_m"], dtype=float) + [0.0, 0.0, half[2]]
    collision = Path(collision_path)
    with collision.open("rb") as stream:  # in blocks: a website collider is hundreds of MB of USDA text
        digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    profiles = sorted(enabled_robot_profiles() if robot_profiles is None else robot_profiles)
    arms = sorted({ROBOT_PROFILE_ARMS[p] for p in profiles if p in ROBOT_PROFILE_ARMS})
    unassessed = [p for p in profiles if p not in ROBOT_PROFILE_ARMS]
    # Evaluation's surface-target admission already refused this region in
    # preparation: a carry that clears still does not make the task "clear".
    admission = preparation.get("surface_target_admission")
    risks = ([{"code": admission["code"], "basis": admission["basis"]}]
             if isinstance(admission, Mapping) and admission.get("status") == "refused" else [])
    value: dict[str, Any] = {"schema_version": SCHEMA, "preparation_digest": preparation["digest"],
                             "robot_id": arms[0] if arms else None, "robot_profiles": profiles,
                             "unassessed_robot_profiles": unassessed, "collision_sha256": digest,
                             "grasp_basis": "one_held_grasp_top_down_first_then_tilted_and_side_on",
                             "reach_rule": "nearest_point_of_object_within_reach_from_a_clear_stance",
                             "mount_rule": "level_with_lower_support", "claim_ceiling": "development_only",
                             "physical_world_truth_claimed": False, **({"risks": risks} if risks else {})}

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
        stations = ground_stance(collision_path=collision, collision_sha256=digest,
                                 **{**carry_station_search(preparation), "ranked": EVERY_CLEAR_STANCE})
    except TeamEmbodimentError as exc:
        return settle("abstained", ["robot_station_" + item for item in exc.blockers], station=None, reach=None,
                      embodiments=[])
    carry = {}
    if destination.get("relation") == "inside":
        carry["carry_height_world_m"] = (float(destination["interior_bounds_world_m"]["maximum"][2])
                                         + float(half[2]) + 0.08)
    # The place is wherever the task will accept the object: within the intake's planar target error of the
    # target, turned any way (no success contract is sealed yet; the team half reads the sealed one).
    tolerance = float((task.get("success") or {}).get("maximum_final_planar_target_error_m") or 0.0)
    region = {"minimum": (target - [tolerance, tolerance, 0.0]).tolist(),
              "maximum": (target + [tolerance, tolerance, 0.0]).tolist()}
    reach_task = {"start_pose_world": [*start.tolist(), 0.0, 0.0, 0.0, 1.0], "target_position_world": target.tolist(),
                  "contact_point_m": [0.0, 0.0, 0.0], "subject_bounds_m": {"minimum": (-half).tolist(),
                                                                          "maximum": half.tolist()},
                  "destination_bounds_world_m": region, "place_yaw_free_rad": math.pi}
    results = []
    for arm in arms:
        reaches = []
        for stance in stations:
            try:
                reaches.append(rigid_reach_shortfall(robot_id=arm, base_pose_world=stance, **reach_task))
            except (KeyError, TypeError, ValueError, RobotArmPathClearanceError):
                reaches.append(None)
        in_reach = [i for i, row in enumerate(reaches) if row is not None and row["status"] == "within_bound"]
        tried = [stations[i] for i in in_reach[:CARRY_STATIONS]] if in_reach else stations[:CARRY_STATIONS]
        cleared, reach, attempts = first_carrying_stance(tried, collision_path=collision, robot_id=arm,
            seeds=[list(NOMINAL_SEED_RAD), list(REST_SEED_RAD[arm])], grasps=_grasps(lower, upper),
            start_pose_world=reach_task["start_pose_world"], target_position_world=target.tolist(),
            contact_point_m=[0.0, 0.0, 0.0],
            pregrasp_clearance_m=float((task.get("success") or {}).get("pregrasp_clearance_m")
                                       or DEFAULT_PREGRASP_CLEARANCE_M), **carry)
        if cleared is not None:
            station, status = cleared, "clear"
        elif in_reach:
            def nearest(index: int) -> float:
                return max(float(row["object_nearest_distance_m"]) for row in reaches[index]["phases"])
            station, status = stations[min(in_reach, key=lambda i: (nearest(i), i))], "flagged"
        elif all(row is not None and row["status"] == "out_of_reach" for row in reaches):
            station, status = stations[0], "abstained"
        else:
            # A stance whose reach could not be measured proves nothing: flagged, never refused.
            station, status = stations[0], "flagged"
        object_reach = reaches[next(i for i, row in enumerate(stations) if row is station)]
        results.append((arm, station, reach, attempts, status, object_reach, len(in_reach)))
        if status == "clear":
            break
    order = {"clear": 0, "flagged": 1, "abstained": 2}
    arm, station, reach, attempts, status, object_reach, in_reach_count = min(
        results, key=lambda row: order[row[4]])
    embodiments = [{"robot_id": a, "status": r["status"], "blockers": list(r["blockers"]), "grasp": r.get("grasp"),
                    "object_reach": None if o is None else o["status"], "stances_in_reach": n,
                    **({"flags": [SCRIPTED_CARRY_FLAG]} if v == "flagged" else {})}
                   for a, _s, r, _t, v, o, n in results]

    # Only the hard reach rule refuses: the object beyond reach from every clear stance, named at the closest.
    blockers = [] if status != "abstained" else [reach_shortfall_code(object_reach)] if len(arms) == 1 else [
        f"{row[0]}:{reach_shortfall_code(row[5])}" for row in results]
    value["robot_id"] = arm
    return settle(status, blockers, station=station, reach=reach, embodiments=embodiments,
                  object_reach=object_reach, stances_in_reach=in_reach_count,
                  **({"flags": [SCRIPTED_CARRY_FLAG],
                      "scripted_carry_blockers": list(reach["blockers"])} if status == "flagged" else {}),
                  stations_tried=len({id(s) for s, _r in attempts}),
                  grasps_tried=len({r.get("grasp") for _s, r in attempts}),
                  station_results=[{"position_world_m": s["position_world_m"], "yaw_rad": s["yaw_rad"],
                                    "grasp": r.get("grasp"), "status": r["status"], "blockers": r["blockers"]}
                                   for s, r in attempts])
