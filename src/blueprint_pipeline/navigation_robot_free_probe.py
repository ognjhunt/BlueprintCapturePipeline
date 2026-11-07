"""A navigation scene's robot-free probe: its scorer, run on synthetic root traces (CPU only, pure).

Beta journey step: L5 team policy evaluation. Owner decision 2026-10-12: no robot is scored on a
scene until, under the scorer its episodes are judged by, doing nothing does not succeed and doing
the task does (``robot_free_probe_gate``). A rigid scene proves that by placing its object without
a robot in Isaac (``rigid_robot_free_readiness_score``). A ``goal_region_navigation`` scene has no
task object: what succeeds or fails is the robot's own root. So this probe hands the real scorer
(``native_g1_navigation_goal.score_g1_navigation_episode``) synthetic root-position traces built
from the sealed plan's own task spec (its ``start_pose_world`` and ``g1_navigation_goal``):

- ``zero_action_not_success``: the root never leaves the compiled start;
- ``direct_arrival_held_success``: a straight walk to the goal centre, held for the settle window;
- ``arrival_without_hold_not_success``: the same arrival, left one sample before the settle window;
- ``root_height_drift_not_success``: the held arrival, with the root height beyond
  ``max_root_height_drift_m`` part way along the route.

Each check also requires the scorer to fail for the reason the trace was built to exercise (no
hold, or the height), so a scorer that fails everything does not pass.

What it proves is only that this scorer, with this plan's goal, tells those traces apart. No
simulator, robot, policy, scene geometry or obstacle is involved: a trace walks straight through
whatever stands between the start and the goal. The receipt says so in its proof boundary, and its
claim ceiling is ``development_only``. It binds the plan digest (the scene revision a navigation
scene's inputs record), the task spec the run is scored with, its success contract and goal, and
the scorer's version and module bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import native_g1_navigation_goal as scorer
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

# Pinned to ``robot_free_probe_gate.NAVIGATION_RECEIPT_SCHEMA`` / ``REQUIRED_CHECKS`` by a test.
SCHEMA = "task_evaluation_configured_scene_navigation_scorer_probe.v1"
TASK_KIND = scorer.NAVIGATION_TASK_KIND
REQUIRED_CHECKS = ("zero_action_not_success", "direct_arrival_held_success", "arrival_without_hold_not_success",
                   "root_height_drift_not_success")
CLAIM_CEILING = "development_only"
SCORER_FUNCTION = "blueprint_pipeline.native_g1_navigation_goal.score_g1_navigation_episode"
#: Route samples from the start to a point just outside the goal; the next sample is the goal centre.
ROUTE_SAMPLES = 20
#: How far outside the acceptance radius the route stops before its last step onto the goal centre.
OUTSIDE_MARGIN_M = 0.1
#: How far beyond ``max_root_height_drift_m`` the drift trace lowers the root.
DRIFT_MARGIN_M = 0.05
TRACE_RULE = {
    "root_xy_at_start": "task_spec.start_pose_world[0:2]",
    "root_height": "task_spec.start_pose_world[2]_held_constant_the_scorer_reads_height_relative_to_the_first_sample",
    "route": "straight_line_to_a_point_outside_the_acceptance_radius_then_one_step_onto_the_goal_centre",
    "route_samples": ROUTE_SAMPLES, "outside_margin_m": OUTSIDE_MARGIN_M,
    "height_drift_beyond_limit_m": DRIFT_MARGIN_M,
    "height_drift_samples": "the_middle_third_of_the_route",
    "leave": "back_to_the_point_outside_the_acceptance_radius",
}
PROOF_BOUNDARY = {
    "robot_embodiment_used": False, "simulator_used": False, "policy_executed": False,
    "synthetic_root_traces_only": True, "scene_geometry_read": False, "obstacle_contact_scored": False,
    "proves": "the_scorer_separates_a_held_goal_arrival_from_no_motion_a_short_hold_and_a_height_drift"
              "_on_synthetic_root_traces_built_from_this_plans_task_spec",
    "not_proved": ["the_route_is_walkable_or_clear_of_obstacles", "the_robot_can_reach_the_goal",
                   "any_simulator_outcome", "any_physical_outcome"],
    "simulator_outcome_claimed": False, "physical_outcome_claimed": False,
}


class NavigationRobotFreeProbeError(ValueError):
    """The plan cannot be probed, or a receipt is not this plan's; the message names why."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise NavigationRobotFreeProbeError(code)


def _scorer_identity() -> dict[str, Any]:
    source = Path(scorer.__file__).read_bytes()
    return {"function": SCORER_FUNCTION, "score_schema_version": scorer.SCORE_SCHEMA_VERSION,
            "module_sha256": "sha256:" + hashlib.sha256(source).hexdigest()}


def _plan_parts(plan: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The sealed plan's task spec, goal, contract and what a receipt binds of them."""
    try:
        _require(isinstance(plan, Mapping)
                 and plan.get("plan_digest") == canonical_digest(dict(plan), digest_field="plan_digest")
                 and plan.get("task_kind") == TASK_KIND, "navigation_robot_free_probe_plan_invalid")
        task = json.loads(json.dumps(plan["task_spec"], allow_nan=False))
        goal = scorer.validate_g1_navigation_goal(task)
        contract = scorer.validate_navigation_task_success_contract(task["task_success_contract"], task_spec=task)
        start = [float(value) for value in task["start_pose_world"]]
        _require(task.get("task_kind") == TASK_KIND and len(start) == 7
                 and all(math.isfinite(value) for value in start)
                 and task.get("task_success_contract_digest") == contract["contract_digest"],
                 "navigation_robot_free_probe_plan_invalid")
    except NavigationRobotFreeProbeError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise NavigationRobotFreeProbeError("navigation_robot_free_probe_plan_invalid") from exc
    binding = {"task_kind": TASK_KIND, "configured_scene_revision_digest": plan["plan_digest"],
               "plan_digest": plan["plan_digest"], "task_spec_digest": canonical_digest(task),
               "navigation_task_spec_digest": task.get("task_spec_digest"),
               "task_success_contract_digest": contract["contract_digest"],
               "goal_digest": cross_runtime_canonical_digest(goal)}
    return task, goal, contract, binding


def _rows(points: list[tuple[float, float, float]]) -> list[dict[str, Any]]:
    return [{"step_index": index, "root_position_world_m": list(point)} for index, point in enumerate(points)]


def _traces(task: Mapping[str, Any], goal: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    start_x, start_y, height = (float(value) for value in task["start_pose_world"][:3])
    centre_x, centre_y = goal["center_world_m"][:2]
    radius, settle = goal["acceptance_radius_m"], goal["settle_window_samples"]
    distance = math.hypot(start_x - centre_x, start_y - centre_y)
    # The route stops on the start-to-goal line just outside the goal (at the start when it is nearer),
    # so every route sample is outside and the goal is entered only on the step onto its centre.
    reach = min(distance, radius + OUTSIDE_MARGIN_M)
    scale = reach / distance if distance > 0.0 else 1.0
    edge = (centre_x + (start_x - centre_x) * scale, centre_y + (start_y - centre_y) * scale, height)
    last = ROUTE_SAMPLES - 1
    route = [(start_x + (edge[0] - start_x) * index / last, start_y + (edge[1] - start_y) * index / last, height)
             for index in range(ROUTE_SAMPLES)]
    at_goal = (centre_x, centre_y, height)
    held = route + [at_goal] * settle
    lowered = height - (goal["max_root_height_drift_m"] + DRIFT_MARGIN_M)
    drifted = [(x, y, lowered) if ROUTE_SAMPLES // 3 <= index < 2 * ROUTE_SAMPLES // 3 else (x, y, z)
               for index, (x, y, z) in enumerate(held)]
    return {
        "zero_action": _rows([(start_x, start_y, height)] * len(held)),
        "direct_arrival_held": _rows(held),
        "arrival_without_hold": _rows(route + [at_goal] * (settle - 1) + [edge] * settle),
        "root_height_drift": _rows(drifted),
    }


def synthetic_traces(plan: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The four root traces the probe scores for ``plan``, keyed by case."""
    task, goal, _contract, _binding = _plan_parts(plan)
    return _traces(task, goal)


#: Each case: must the scorer call it a success, and which report fields must it show.
_EXPECTED = {
    "zero_action": {"outcome": "failure", "terminal_goal_hold": False, "root_height_stable_throughout": True},
    "direct_arrival_held": {"outcome": "success"},
    "arrival_without_hold": {"outcome": "failure", "terminal_goal_hold": False, "first_settled_step": None,
                             "root_height_stable_throughout": True},
    "root_height_drift": {"outcome": "failure", "terminal_goal_hold": True, "root_height_stable_throughout": False},
}
_CHECK_OF_CASE = {"zero_action": "zero_action_not_success", "direct_arrival_held": "direct_arrival_held_success",
                  "arrival_without_hold": "arrival_without_hold_not_success",
                  "root_height_drift": "root_height_drift_not_success"}


def _case(task: Mapping[str, Any], name: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    expected = _EXPECTED[name]
    result: dict[str, Any] = {"must_succeed": expected["outcome"] == "success", "sample_count": len(samples),
                              "trace_digest": canonical_digest({"samples": samples}), "expected": dict(expected)}
    try:
        report = scorer.score_g1_navigation_episode(task_spec=task, samples=samples)
    except ValueError as exc:
        # The scorer refused the trace (for one, a start already inside the goal): no verdict, so no pass.
        return {**result, "status": "unscorable", "error": str(exc)[:200], "holds": False}
    observed = {key: report[key] for key in ("outcome", "terminal_goal_hold", "first_settled_step",
                                             "root_height_stable_throughout")}
    return {**result, "status": report["status"], **observed,
            "maximum_root_height_drift_observed_m": report["maximum_root_height_drift_observed_m"],
            "terminal_distance_m": report["terminal_distance_m"], "score_digest": report["score_digest"],
            "holds": report["status"] == "scored" and all(observed[key] == value for key, value in expected.items())}


def score_navigation_robot_free_probe(*, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sealed receipt; ``passed`` only when every required check holds.

    Refuses by name (``navigation_robot_free_probe_plan_invalid``) a plan that is not a sealed
    ``goal_region_navigation`` plan whose task spec carries its own success contract.
    """
    task, goal, _contract, binding = _plan_parts(plan)
    cases = {name: _case(task, name, samples) for name, samples in _traces(task, goal).items()}
    checks = {_CHECK_OF_CASE[name]: case["holds"] for name, case in cases.items()}
    passed = all(checks[name] is True for name in REQUIRED_CHECKS)
    receipt = {
        "schema_version": SCHEMA, **binding, "scorer": _scorer_identity(), "synthetic_trace_rule": dict(TRACE_RULE),
        "cases": cases, "checks": checks, "required_checks": list(REQUIRED_CHECKS),
        "status": "passed" if passed else "failed", "learned_policy_evaluation_admitted": passed,
        "claim_ceiling": CLAIM_CEILING, "proof_boundary": json.loads(json.dumps(PROOF_BOUNDARY)),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def validate_navigation_robot_free_probe_receipt(receipt: Any, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    """The receipt, when it is sealed, well formed and this plan's; passed or failed alike.

    Whether it admits a scored run is the gate's decision (``robot_free_probe_gate``). Raises
    ``navigation_robot_free_probe_receipt_invalid`` for a tampered receipt, another plan's, or one
    whose checks are not exactly the required ones or disagree with its status. The probe costs
    nothing, so it is run again here: a receipt is valid only while it is exactly what today's scorer
    (its module bytes included) says for this plan, never on the scorer identity it records.
    """
    _task, _goal, _contract, binding = _plan_parts(plan)
    try:
        value = json.loads(json.dumps(receipt, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NavigationRobotFreeProbeError("navigation_robot_free_probe_receipt_invalid") from exc
    checks = value.get("checks") if isinstance(value, dict) else None
    _require(isinstance(value, dict) and value.get("schema_version") == SCHEMA
             and value.get("receipt_digest") == canonical_digest(value, digest_field="receipt_digest")
             and all(value.get(key) == expected for key, expected in binding.items())
             and value.get("claim_ceiling") == CLAIM_CEILING and value.get("proof_boundary") == PROOF_BOUNDARY
             and value.get("required_checks") == list(REQUIRED_CHECKS)
             and isinstance(checks, dict) and set(checks) == set(REQUIRED_CHECKS)
             and all(isinstance(flag, bool) for flag in checks.values())
             and value.get("status") == ("passed" if all(checks.values()) else "failed")
             and value.get("learned_policy_evaluation_admitted") is (value.get("status") == "passed"),
             "navigation_robot_free_probe_receipt_invalid")
    _require(value == score_navigation_robot_free_probe(plan=plan), "navigation_robot_free_probe_receipt_invalid")
    return value


__all__ = [
    "CLAIM_CEILING",
    "NavigationRobotFreeProbeError",
    "PROOF_BOUNDARY",
    "REQUIRED_CHECKS",
    "SCHEMA",
    "TASK_KIND",
    "score_navigation_robot_free_probe",
    "synthetic_traces",
    "validate_navigation_robot_free_probe_receipt",
]
