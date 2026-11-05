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

``passed`` means some arm reaches every phase in the model. ``out_of_reach`` is reserved for a proof:
every arm has a phase beyond its chain's reach bound, so no motion of this body from this station can do
the task in the model. A phase that no start reached, or that was reached only outside the joint limits,
is a search result, not a proof, and is reported as ``flagged``. Neither refuses anything: reach is not a
privacy, rights or money control (``AGENTS.md``), so the verdict is reported with the run's evidence and
the run goes on. A body or task this cannot model is ``unassessed`` with a typed reason. Scene clearance
is not measured here.

Nothing here runs a simulator, queries a policy or claims that a policy or the physical robot does it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CLAIM_CEILING = "cpu_bounded_estimate"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _head(plan: Mapping[str, Any]) -> dict[str, Any]:
    from .robot_arm_path_clearance import SCHEMA_VERSION

    robot = plan.get("robot") or {}
    return {"schema_version": SCHEMA_VERSION, "task_kind": plan.get("task_kind"), "robot_id": robot.get("robot_id"),
            "embodiment_id": robot.get("embodiment_id"), "embodiment_digest": robot.get("embodiment_digest"),
            "base_mode": robot.get("base_mode"), "arena_scene_plan_digest": plan.get("plan_digest"),
            "claim_ceiling": CLAIM_CEILING, "simulator_executed": False, "scene_clearance_measured": False}


def _finish(plan: Mapping[str, Any], status: str, *, reason: str | None = None, advisories: list[str] | None = None,
            **extra: Any) -> dict[str, Any]:
    from .decision_evidence_contracts import canonical_digest

    robot = plan.get("robot") or {}
    result = {**_head(plan), "status": status, **({"reason": reason} if reason else {}), "blockers": [],
              "advisories": advisories if advisories is not None else (
                  ["robot_arm_path_unassessed:" + str(robot.get("embodiment_id"))] if status == "unassessed" else []),
              **extra, "report_digest": ""}
    result["report_digest"] = canonical_digest(result, digest_field="report_digest")
    return result


#: The authored phase plan refuses a packet whose task start pose and spawned task-object root disagree by
#: more than the reset tolerance (a support alignment raised the root after the start was fixed).
RESET_MISMATCH = "native_rigid_construction_scoring_frame_reset_mismatch"


def _detail(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    return ",".join(errors) if errors else type(exc).__name__


def _start_on_spawned_root(plan: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    """The plan with the task's start pose where its object spawns: the subject root and its scoring frame."""
    from .team_arm_kinematics import pose_matrix, quaternion_xyzw

    (subject,) = [row for row in plan.get("objects") or [] if isinstance(row, Mapping) and row.get("task_subject") is True]
    root = (subject.get("reset_state") or {}).get("root_pose_world") or subject["pose_world"]
    task = plan["task_spec"]
    scoring = task["interaction_affordance"]["asset_root_from_scoring_frame"]
    start = (pose_matrix(root["position_world_m"], root["orientation_xyzw"])
             @ pose_matrix(scoring["position_m"], scoring["orientation_xyzw"]))
    position = [float(v) for v in start[:3, 3]]
    offset = float(np.linalg.norm(np.asarray(position) - np.asarray(task["start_pose_world"][:3], dtype=float)))
    return {**plan, "task_spec": {**task, "start_pose_world": [*position, *quaternion_xyzw(start[:3, :3])]}}, offset


def _authored_phases(plan: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The task's authored phases, and advisories on how they were derived.

    When the phase plan refuses only because the start pose and the spawned object disagree
    (``RESET_MISMATCH``), the phases are derived with the start where the object spawns, which is where an
    arm has to grasp it, and the advisory names the offset. Reach is measured either way; the run's own
    phase plan is unchanged.
    """
    from .native_task_construction_plan import materialize_native_task_construction_phase_plan

    try:
        return materialize_native_task_construction_phase_plan(plan), []
    except ValueError as exc:
        if tuple(getattr(exc, "errors", ()) or ()) != (RESET_MISMATCH,):
            raise
    reseated, offset = _start_on_spawned_root(plan)
    return (materialize_native_task_construction_phase_plan(reseated),
            [f"task_start_reseated_on_spawned_root:{offset:.4f}m"])


def _prepare(plan: Mapping[str, Any], model_path: str | Path, manifest: Mapping[str, Any], *,
             free_base: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The body's arms, grasp bindings and the task's phases, read once; or the unassessed report.

    ``free_base`` also measures a free base (``free_on_floor``) from the stance it starts at.
    """
    from .team_arm_kinematics import pose_matrix

    robot = plan.get("robot") or {}
    if robot.get("base_mode") != "fixed_at_station" and not (free_base and robot.get("base_mode") == "free_on_floor"):
        return _finish(plan, "unassessed", reason="robot_base_not_fixed_at_station"), None
    if plan.get("task_kind") != "rigid_pick_place":
        return _finish(plan, "unassessed", reason="task_kind_unmodeled"), None
    model = Path(model_path)
    if not model.is_file() or _sha256(model) != robot.get("usd_sha256"):
        return _finish(plan, "unassessed", reason="team_model_unbound"), None
    try:
        authored, advisories = _authored_phases(plan)
        if authored.get("manipulation_strategy") != "pick_and_place":
            return _finish(plan, "unassessed", reason="manipulation_strategy_unmodeled"), None
        phases = [(str(row["phase_id"]), pose_matrix(row["position_world_m"], row["orientation_world_xyzw"]))
                  for row in authored["phases"] if row.get("phase_id") != "recovery"]
    except (KeyError, TypeError, ValueError) as exc:
        return _finish(plan, "unassessed", reason="task_phases_unavailable", detail=_detail(exc)), None
    if not phases:
        return _finish(plan, "unassessed", reason="task_phases_unavailable"), None

    from .team_body_kinematics import body_arm_chains

    found = body_arm_chains(model, manifest)
    if not found["chains"]:
        return _finish(plan, "unassessed", reason="robot_arm_unmodeled", unassessed_arms=dict(found["unassessed"])), None
    advisories += [f"robot_arm_unmodeled:{arm}:{code}" for arm, code in sorted(found["unassessed"].items())]
    try:
        from .scripted_robot_runtime_dispatch import build_policy_spec

        bindings = build_policy_spec(manifest)["grasp_bindings"]
    except (KeyError, TypeError, ValueError):
        bindings = None
        advisories.append("robot_grasp_orientation_unbound:position_only")
    return None, {"phases": phases, "found": found, "bindings": bindings, "advisories": advisories}


def _evaluate(plan: Mapping[str, Any], prepared: Mapping[str, Any], base_pose_world: Mapping[str, Any], *,
              classify: bool = True) -> dict[str, Any]:
    """The verdict with the body's base at ``base_pose_world``."""
    from .team_arm_kinematics import pose_matrix
    from .team_body_kinematics import IK_ORIENTATION_WEIGHT_M, solve

    phases, found, bindings = prepared["phases"], prepared["found"], prepared["bindings"]
    advisories = list(prepared["advisories"])
    fixed = (plan.get("robot") or {}).get("base_mode") == "fixed_at_station"
    if not fixed:
        # The base can move during the episode: the verdict is from the stance it starts at, never a proof.
        advisories.append("robot_free_base_measured_from_start_stance")
    world_from_root = pose_matrix(base_pose_world["position_world_m"], base_pose_world["orientation_xyzw"])
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
            target = base_from_world @ world_from_grasp @ grasp_from_tcp
            result = solve(chain, target, classify=classify,
                           seeds=None if previous is None else [previous], orientation_weight_m=weight)
            rows.append({"phase_id": phase_id, "reached": bool(result.ok), "failure": result.failure,
                         "detail": result.detail, "position_error_m": (float(result.position_error_m)
                                                                       if np.isfinite(result.position_error_m) else None)})
            if result.detail == "beyond_reach_bound":
                # The proof's numbers: the TCP target's distance from the chain's first joint, and its bound.
                rows[-1].update(distance_m=round(float(np.linalg.norm(target[:3, 3] - chain.before[0][:3, 3])), 4),
                                reach_bound_m=round(float(chain.reach_bound_m), 4))
            if not result.ok:
                break
            previous = result.joints
        arms[name] = {"orientation_bound": bool(grasp), "phases": rows,
                      "reached_every_phase": len(rows) == len(phases) and rows[-1]["reached"]}
    phase_ids = [phase_id for phase_id, _ in phases]
    reached = sorted(name for name, row in arms.items() if row["reached_every_phase"])
    if reached:
        return _finish(plan, "passed", advisories=advisories, arms=arms, reaching_arms=reached, phase_ids=phase_ids)
    misses = {name: row["phases"][-1] for name, row in arms.items()}
    if fixed and all(miss["detail"] == "beyond_reach_bound" for miss in misses.values()) and not found["unassessed"]:
        # Furthest phase first: the one the best arm got to.
        name, miss = max(misses.items(), key=lambda item: len(arms[item[0]]["phases"]))
        return _finish(plan, "out_of_reach", advisories=[*advisories, "robot_arm_task_beyond_reach:" + miss["phase_id"]],
                       arms=arms, phase_ids=phase_ids,
                       beyond_reach={"arm": name, "phase_id": miss["phase_id"],
                                     "distance_m": miss.get("distance_m"), "reach_bound_m": miss.get("reach_bound_m")})
    advisories += [f"robot_arm_task_unreached:{name}:{miss['phase_id']}:{miss['failure']}:{miss['detail']}"
                   for name, miss in sorted(misses.items())]
    return _finish(plan, "flagged", advisories=advisories, arms=arms, phase_ids=phase_ids)


def team_arm_reach(plan: Mapping[str, Any], *, model_path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The arm-path report for a team robot applied at the station of ``plan``."""
    refused, prepared = _prepare(plan, model_path, manifest)
    if refused is not None:
        return refused
    try:
        base = (plan.get("robot") or {})["base_pose_world"]
        return _evaluate(plan, prepared, base)
    except (KeyError, TypeError, ValueError) as exc:
        return _finish(plan, "unassessed", reason="task_phases_unavailable", detail=type(exc).__name__)


def first_reaching_stance(plan: Mapping[str, Any], stances: Sequence[Mapping[str, Any]], *, model_path: str | Path,
                          manifest: Mapping[str, Any], limit: int = 24, free_base: bool = False) -> dict[str, Any]:
    """The first of ``stances`` (closest first) from which one of the body's arms reaches every phase.

    ``plan`` carries the team robot row (``base_mode`` and the bound ``usd_sha256``); its station is
    replaced by each stance in turn. At most ``limit`` stances are solved, each without the slower
    miss classification. ``free_base`` also measures a free base from each floor stance it could start
    at; a miss there is never ``out_of_reach``, since the base can move. Returns ``{"index", "stances_solved", "report"}``: ``index`` is ``None`` when
    none of them reached, and ``report`` is the chosen stance's verdict, or the first stance's.
    """
    refused, prepared = _prepare(plan, model_path, manifest, free_base=free_base)
    if refused is not None:
        return {"index": None, "stances_solved": 0, "report": refused}
    first = None
    for index, stance in enumerate(stances[:max(0, int(limit))]):
        report = _evaluate(plan, prepared, stance, classify=False)
        first = first or report
        if report["status"] == "passed":
            return {"index": index, "stances_solved": index + 1, "report": report}
    return {"index": None, "stances_solved": min(len(stances), max(0, int(limit))),
            "report": first or _finish(plan, "unassessed", reason="no_clear_stance")}


__all__ = ["CLAIM_CEILING", "first_reaching_stance", "team_arm_reach"]
