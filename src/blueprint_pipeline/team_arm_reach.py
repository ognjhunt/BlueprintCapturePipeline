"""Whether a team's own arm reaches the task's authored poses from its station: a CPU bounded estimate.

The pre-GPU arm-path gate solves Blueprint's DROID Franka along its rigid pick-and-place path with the
published Franka model (``robot_arm_path_clearance``). A team's robot replaces that Franka at the station
(``team_embodiment.apply_team_embodiment``), so the gate measures the team's body instead, from the robot
model the run loads (``team_body_kinematics``):

* the task's authored phases (``materialize_native_task_construction_phase_plan``: pre-grasp, grasp,
  lift, carry, place, retreat) are the world grasp poses the scripted baseline drives;
* each arm's TCP pose for a phase is that grasp composed with the body's audited grasp binding
  (``scripted_robot_runtime_dispatch.build_policy_spec(...)["grasp_bindings"]``); a body without one is
  solved for position only and says so;
* each arm is solved through the phases in order, each from the previous solution, from the base it has
  at the station (``world_from_root @ chain.root_from_base``).

``passed`` means some arm reaches every phase in the model. ``blocked`` is reserved for a proof: every
arm has a phase beyond its chain's reach bound, so no motion of this body from this station can do the
task. A phase that no start reached, or that was reached only outside the joint limits, is a search
result, not a proof, and is reported as ``flagged`` with an advisory: the run goes on. A body or task
this cannot model is ``unassessed`` with a typed reason. Scene clearance is not measured here.

Nothing here runs a simulator, queries a policy or claims that a policy or the physical robot does it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CLAIM_CEILING = "cpu_bounded_estimate"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def team_arm_reach(plan: Mapping[str, Any], *, model_path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The arm-path report for a team robot applied at the station of ``plan``."""
    from .decision_evidence_contracts import canonical_digest
    from .robot_arm_path_clearance import SCHEMA_VERSION
    from .team_arm_kinematics import pose_matrix

    robot = plan.get("robot") or {}
    head = {"schema_version": SCHEMA_VERSION, "task_kind": plan.get("task_kind"), "robot_id": robot.get("robot_id"),
            "embodiment_id": robot.get("embodiment_id"), "embodiment_digest": robot.get("embodiment_digest"),
            "base_mode": robot.get("base_mode"), "arena_scene_plan_digest": plan.get("plan_digest"),
            "claim_ceiling": CLAIM_CEILING, "simulator_executed": False, "scene_clearance_measured": False}

    def finish(status: str, *, reason: str | None = None, blockers: list[str] | None = None,
               advisories: list[str] | None = None, **extra: Any) -> dict[str, Any]:
        result = {**head, "status": status, **({"reason": reason} if reason else {}), "blockers": blockers or [],
                  "advisories": advisories if advisories is not None else (
                      ["robot_arm_path_unassessed:" + str(robot.get("embodiment_id"))] if status == "unassessed" else []),
                  **extra, "report_digest": ""}
        result["report_digest"] = canonical_digest(result, digest_field="report_digest")
        return result

    if robot.get("base_mode") != "fixed_at_station":
        return finish("unassessed", reason="robot_base_not_fixed_at_station")
    if plan.get("task_kind") != "rigid_pick_place":
        return finish("unassessed", reason="task_kind_unmodeled")
    model = Path(model_path)
    if not model.is_file() or _sha256(model) != robot.get("usd_sha256"):
        return finish("unassessed", reason="team_model_unbound")
    try:
        from .native_task_construction_plan import materialize_native_task_construction_phase_plan

        authored = materialize_native_task_construction_phase_plan(plan)
        if authored.get("manipulation_strategy") != "pick_and_place":
            return finish("unassessed", reason="manipulation_strategy_unmodeled")
        phases = [(str(row["phase_id"]), pose_matrix(row["position_world_m"], row["orientation_world_xyzw"]))
                  for row in authored["phases"] if row.get("phase_id") != "recovery"]
        world_from_root = pose_matrix(robot["base_pose_world"]["position_world_m"],
                                      robot["base_pose_world"]["orientation_xyzw"])
    except (KeyError, TypeError, ValueError) as exc:
        return finish("unassessed", reason="task_phases_unavailable", detail=type(exc).__name__)
    if not phases:
        return finish("unassessed", reason="task_phases_unavailable")

    from .team_body_kinematics import IK_ORIENTATION_WEIGHT_M, body_arm_chains, solve

    found = body_arm_chains(model, manifest)
    if not found["chains"]:
        return finish("unassessed", reason="robot_arm_unmodeled", unassessed_arms=dict(found["unassessed"]))
    advisories = [f"robot_arm_unmodeled:{arm}:{code}" for arm, code in sorted(found["unassessed"].items())]
    try:
        from .scripted_robot_runtime_dispatch import build_policy_spec

        bindings = build_policy_spec(manifest)["grasp_bindings"]
    except (KeyError, TypeError, ValueError):
        bindings = None
        advisories.append("robot_grasp_orientation_unbound:position_only")

    arms: dict[str, Any] = {}
    for name, chain in sorted(found["chains"].items()):
        base_from_world = np.linalg.inv(world_from_root @ chain.root_from_base)
        grasp = (bindings or {}).get(name)
        grasp_from_tcp = (np.linalg.inv(pose_matrix(grasp["tcp_from_grasp"]["position_m"],
                                                    grasp["tcp_from_grasp"]["orientation_xyzw"]))
                          if grasp else np.eye(4))
        weight = IK_ORIENTATION_WEIGHT_M if grasp else 0.0
        rows, previous = [], None
        for phase_id, world_from_grasp in phases:
            result = solve(chain, base_from_world @ world_from_grasp @ grasp_from_tcp,
                           seeds=None if previous is None else [previous], orientation_weight_m=weight)
            rows.append({"phase_id": phase_id, "reached": bool(result.ok), "failure": result.failure,
                         "detail": result.detail, "position_error_m": (float(result.position_error_m)
                                                                       if np.isfinite(result.position_error_m) else None)})
            if not result.ok:
                break
            previous = result.joints
        arms[name] = {"orientation_bound": bool(grasp), "phases": rows,
                      "reached_every_phase": len(rows) == len(phases) and rows[-1]["reached"]}

    reached = sorted(name for name, row in arms.items() if row["reached_every_phase"])
    if reached:
        return finish("passed", advisories=advisories, arms=arms, reaching_arms=reached,
                      phase_ids=[phase_id for phase_id, _ in phases])
    misses = {name: row["phases"][-1] for name, row in arms.items()}
    if all(miss["detail"] == "beyond_reach_bound" for miss in misses.values()) and not found["unassessed"]:
        # Furthest phase first: the one the best arm got to.
        name, miss = max(misses.items(), key=lambda item: len(arms[item[0]]["phases"]))
        return finish("blocked", blockers=["robot_arm_task_beyond_reach:" + miss["phase_id"]], advisories=advisories,
                      arms=arms, phase_ids=[phase_id for phase_id, _ in phases])
    advisories += [f"robot_arm_task_unreached:{name}:{miss['phase_id']}:{miss['failure']}:{miss['detail']}"
                   for name, miss in sorted(misses.items())]
    return finish("flagged", advisories=advisories, arms=arms, phase_ids=[phase_id for phase_id, _ in phases])


__all__ = ["CLAIM_CEILING", "team_arm_reach"]
