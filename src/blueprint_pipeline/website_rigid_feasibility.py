"""Before any construction, whether the admitted arm can carry a rigid task at all (ADP-009D).

The station is the one the team packet takes (``team_scene_packet``): a mount
level with the lower of the task's supports, on the closest clear floor, facing
the start or, when the object ends on another surface, the whole region from
the start to the target. From one of the closest clear stances the arm must
carry the object with the task's top-down grasp, clear of the reconstructed
scene (``robot_arm_path_clearance.rigid_pick_place_clearance``). A refusal
names the furthest phase any stance reached, before the owner is asked to pay
for the scene build. A pass means one such carry exists, not that a policy
finds it.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest

SCHEMA = "website_rigid_task_capability_verdict.v1"
#: Every executable candidate runs on the DROID Franka (``config/beta_policy_candidates.v1.json``).
ROBOT_ID = "franka_panda"
DEFAULT_PREGRASP_CLEARANCE_M = 0.10


def rigid_task_verdict(*, preparation: Mapping[str, Any], collision_path: Path) -> dict[str, Any]:
    """The station and carry verdict for a prepared rigid pick-and-place, from CPU geometry only."""
    import hashlib

    from .native_franka_action_math import grasp_orientation_contact_xyzw
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
    value: dict[str, Any] = {"schema_version": SCHEMA, "preparation_digest": preparation["digest"],
                             "robot_id": ROBOT_ID, "collision_sha256": digest,
                             "grasp_basis": "task_top_down_grasp_held_through_the_carry",
                             "mount_rule": "level_with_lower_support", "claim_ceiling": "development_only",
                             "physical_world_truth_claimed": False}
    try:
        stations = ground_stance(collision_path=collision, collision_sha256=digest, offset=[0.0, 0.0, 0.0],
            subject_center=((region_lower + region_upper) / 2.0).tolist(), subject_minimum=region_lower.tolist(),
            subject_maximum=region_upper.tolist(), footprint_half_extent_m=list(STATION_FOOTPRINT_HALF_EXTENT_M),
            root_world_z_m=min(float(row["aabb_max_xyz"][2]) for row in supports), ranked=CARRY_STATIONS)
    except TeamEmbodimentError as exc:
        value.update(status="abstained", blockers=["robot_station_" + item for item in exc.errors], station=None, reach=None)
        value["verdict_digest"] = canonical_digest(value, digest_field="verdict_digest")
        return value
    jaw = [1.0, 0.0, 0.0] if upper[0] - lower[0] <= upper[1] - lower[1] else [0.0, 1.0, 0.0]
    carry = {}
    if destination.get("relation") == "inside":
        carry["carry_height_world_m"] = (float(destination["interior_bounds_world_m"]["maximum"][2])
                                         + float(half[2]) + 0.08)
    cleared, reach, attempts = first_carrying_stance(stations, collision_path=collision, robot_id=ROBOT_ID,
        seeds=[list(NOMINAL_SEED_RAD), list(REST_SEED_RAD[ROBOT_ID])],
        start_pose_world=[*start.tolist(), 0.0, 0.0, 0.0, 1.0], target_position_world=target.tolist(),
        contact_point_m=[0.0, 0.0, 0.0], approach_unit=[0.0, 0.0, 1.0],
        gripper_orientation_xyzw=grasp_orientation_contact_xyzw(approach_axis=[0.0, 0.0, -1.0], jaw_axis=jaw),
        pregrasp_clearance_m=float((task.get("success") or {}).get("pregrasp_clearance_m")
                                   or DEFAULT_PREGRASP_CLEARANCE_M), **carry)
    station = cleared or next(s for s, r in attempts if r is reach)
    value.update(status="clear" if reach["status"] == "clear" else "abstained", blockers=list(reach["blockers"]),
                 station=station, reach=reach, stations_tried=len(attempts),
                 station_results=[{"position_world_m": s["position_world_m"], "yaw_rad": s["yaw_rad"],
                                   "status": r["status"], "blockers": r["blockers"]} for s, r in attempts])
    value["verdict_digest"] = canonical_digest(value, digest_field="verdict_digest")
    return value
