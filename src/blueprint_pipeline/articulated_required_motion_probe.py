"""Plan and check every required mechanism motion; no task or policy grading.

The executor receives simulator callbacks. It never sets a joint state itself.
Coordinates are radians for revolute joints and metres for prismatic joints.
This file is also copied into the standalone native diagnostic bundle.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping


SCHEMA = "articulated_required_motion_probe_plan.v1"
RESULT_SCHEMA = "articulated_required_motion_probe_result.v1"
DIAGNOSTIC_DRIVE = {"drive_stiffness": 2000.0, "drive_damping": 150.0, "drive_max_force": 5000.0}


def _digest(value, field):
    return "sha256:" + hashlib.sha256(json.dumps(
        {key: item for key, item in value.items() if key != field},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError("required_motion_probe_" + reason)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def build_required_motion_probe_plan(*, readback, graph, requirements=(), source_frames=()):
    """Use observed chronology when present, retaining candidate order otherwise.

    Open the earlier mechanisms before testing later ones, hold those earlier
    positions, then reset in reverse order. This is a candidate physics probe,
    not a claim that the source video establishes mechanical dependencies.
    """
    motions = list(readback["motions"])
    _require(bool(motions), "motions_missing")
    joints = {row["joint_id"]: row for row in graph["joints"]}
    timestamps = {row["frame_id"]: row["timestamp_seconds"] for row in source_frames
                  if isinstance(row, Mapping) and row.get("frame_id") and _number(row.get("timestamp_seconds"))}
    evidence = {row["motion_id"]: row.get("source_frame_ids", []) for row in requirements}
    observed_order = all(evidence.get(row["motion_id"]) and
                         all(frame in timestamps for frame in evidence[row["motion_id"]]) for row in motions)
    if observed_order:
        motions.sort(key=lambda row: min(timestamps[frame] for frame in evidence[row["motion_id"]]))
    rows = []
    for motion in motions:
        joint = joints[motion["joint_id"]]
        low, high = map(float, joint["limits"])
        reset = float(joint["reset_position"])
        _require(joint["joint_type"] == motion["joint_type"] and
                 [low, high] == list(motion["limits"]) and low <= reset <= high, "joint_binding_changed")
        # Exercise the available travel away from reset, including negative or
        # nonzero-origin ranges. Tolerances scale with each mechanism's travel.
        endpoint = max((low, high), key=lambda bound: abs(bound - reset))
        travel = high - low
        rows.append({"motion_id": motion["motion_id"], "joint_id": joint["joint_id"],
                     "joint_type": joint["joint_type"], "limits": [low, high],
                     "reset_position": reset, "endpoint_position": endpoint,
                     "position_tolerance": max(float(joint["reset_tolerance"]), travel * 0.03),
                     "maximum_abs_velocity": travel * 0.1})
    current = {row["joint_id"]: row["reset_position"] for row in rows}
    phases = [{"phase_id": "reset_initial", "positions": dict(current)}]
    for row in rows:
        for fraction, suffix in ((0.5, "midpoint"), (1.0, "endpoint")):
            current[row["joint_id"]] = row["reset_position"] + fraction * (row["endpoint_position"] - row["reset_position"])
            phases.append({"phase_id": row["motion_id"] + "_" + suffix, "positions": dict(current)})
    for row in reversed(rows):
        current[row["joint_id"]] = row["reset_position"]
        phases.append({"phase_id": row["motion_id"] + "_reset", "positions": dict(current)})
    plan = {"schema_version": SCHEMA, "requirements_digest": readback["requirements_digest"],
            "motion_order_basis": "source_frame_chronology" if observed_order else "candidate_authoring_requirement_order",
            "dependency_order_is_measured": False, "motions": rows, "phases": phases,
            "task_success_proven": False, "physics_authority_granted": False}
    plan["plan_digest"] = _digest(plan, "plan_digest")
    return validate_required_motion_probe_plan(plan)


def validate_required_motion_probe_plan(plan):
    _require(isinstance(plan, Mapping) and plan.get("schema_version") == SCHEMA and
             plan.get("task_success_proven") is False and plan.get("physics_authority_granted") is False and
             plan.get("dependency_order_is_measured") is False and
             plan.get("plan_digest") == _digest(plan, "plan_digest"), "plan_invalid")
    _require(isinstance(plan.get("requirements_digest"), str) and
             re.fullmatch(r"sha256:[0-9a-f]{64}", plan["requirements_digest"]), "requirements_binding_invalid")
    motions, phases = plan.get("motions"), plan.get("phases")
    _require(isinstance(motions, list) and bool(motions) and isinstance(phases, list) and bool(phases), "plan_invalid")
    ids, joint_ids = set(), set()
    for row in motions:
        _require(isinstance(row, Mapping) and isinstance(row.get("motion_id"), str) and
                 re.fullmatch(r"[a-z][a-z0-9_]*", row["motion_id"]) and row["motion_id"] not in ids and
                 isinstance(row.get("joint_id"), str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", row["joint_id"]) and
                 row["joint_id"] not in joint_ids and row.get("joint_type") in {"revolute", "prismatic"}, "motion_invalid")
        bounds = row.get("limits")
        _require(isinstance(bounds, list) and len(bounds) == 2 and all(_number(v) for v in bounds) and
                 bounds[0] < bounds[1] and all(_number(row.get(key)) for key in
                 ("reset_position", "endpoint_position", "position_tolerance", "maximum_abs_velocity")), "motion_invalid")
        travel = bounds[1] - bounds[0]
        _require(bounds[0] <= row["reset_position"] <= bounds[1] and bounds[0] <= row["endpoint_position"] <= bounds[1] and
                 abs(row["endpoint_position"] - row["reset_position"]) >= travel / 2 and
                 0 < row["position_tolerance"] <= travel * 0.1 and 0 < row["maximum_abs_velocity"] <= travel,
                 "motion_invalid")
        ids.add(row["motion_id"])
        joint_ids.add(row["joint_id"])
    _require(len(phases) == 1 + 3 * len(motions), "phase_coverage_invalid")
    phase_ids = set()
    for phase in phases:
        _require(isinstance(phase, Mapping) and isinstance(phase.get("phase_id"), str) and
                 re.fullmatch(r"[a-z][a-z0-9_]*", phase["phase_id"]) and phase["phase_id"] not in phase_ids and
                 isinstance(phase.get("positions"), Mapping) and set(phase["positions"]) == joint_ids, "phase_invalid")
        phase_ids.add(phase["phase_id"])
        for row in motions:
            value = phase["positions"][row["joint_id"]]
            _require(_number(value) and row["limits"][0] <= value <= row["limits"][1], "phase_invalid")
    reset = {row["joint_id"]: row["reset_position"] for row in motions}
    _require(phases[0]["positions"] == reset and phases[-1]["positions"] == reset, "reset_missing")
    for row in motions:
        _require(any(phase["positions"][row["joint_id"]] == row["endpoint_position"] for phase in phases), "motion_coverage_missing")
    return dict(plan)


def execute_required_motion_probe(plan, *, command, read_state, capture=None):
    """Command/ramp/settle callbacks must use real simulation, then read it back."""
    plan = validate_required_motion_probe_plan(plan)
    rows, errors = [], []
    for phase in plan["phases"]:
        try:
            command(phase["positions"])
            state = read_state()
            _require(isinstance(state, Mapping) and set(state) == set(phase["positions"]), "readback_joint_set_invalid")
            checks = {}
            for motion in plan["motions"]:
                name = motion["joint_id"]
                pair = state[name]
                _require(isinstance(pair, (list, tuple)) and len(pair) == 2 and all(_number(v) for v in pair), "readback_nonfinite")
                checks[name] = {"position": float(pair[0]), "velocity": float(pair[1]),
                    "absolute_error": abs(pair[0] - phase["positions"][name]),
                    "position_passed": abs(pair[0] - phase["positions"][name]) <= motion["position_tolerance"],
                    "settled": abs(pair[1]) <= motion["maximum_abs_velocity"]}
            row = {"phase_id": phase["phase_id"], "commanded_positions": dict(phase["positions"]), "joint_readback": checks}
            rows.append(row)
            if capture is not None:
                capture(phase["phase_id"], row)
            if any(not check["position_passed"] or not check["settled"] for check in checks.values()):
                errors.append("required_motion_probe_phase_failed:" + phase["phase_id"])
                break
        except Exception as exc:
            errors.append(type(exc).__name__ + ":" + str(exc)[:300])
            break
    result = {"schema_version": RESULT_SCHEMA, "status": "passed" if not errors else "failed",
              "plan_digest": plan["plan_digest"], "requirements_digest": plan["requirements_digest"],
              "motion_ids": [row["motion_id"] for row in plan["motions"]], "phase_rows": rows,
              "blockers": errors, "robot_embodiment_used": False, "learned_policy_queried": False,
              "task_success_proven": False, "physics_authority_granted": False}
    result["result_digest"] = _digest(result, "result_digest")
    return result


def required_motion_plan_from_static_receipt(receipt):
    """Read a sealed plan, or derive a candidate probe for an older receipt."""
    readback = receipt.get("source_constraints", {}).get("required_motion_readback")
    if not readback or not readback.get("motions"):
        return None
    plan = receipt.get("required_motion_probe_plan") or build_required_motion_probe_plan(
        readback=readback, graph=receipt["articulation_graph"])
    plan = validate_required_motion_probe_plan(plan)
    joints = {row["joint_id"]: row for row in receipt["articulation_graph"]["joints"]}
    _require(all(row["joint_id"] in joints and
                 row["reset_position"] == joints[row["joint_id"]]["reset_position"] and
                 row["joint_type"] == joints[row["joint_id"]]["joint_type"] and
                 row["limits"] == joints[row["joint_id"]]["limits"] for row in plan["motions"]), "static_binding_changed")
    _require(plan["requirements_digest"] == readback["requirements_digest"] and
             {(row["motion_id"], row["joint_id"], row["joint_type"], tuple(row["limits"])) for row in plan["motions"]} ==
             {(row["motion_id"], row["joint_id"], row["joint_type"], tuple(row["limits"])) for row in readback["motions"]},
             "static_binding_changed")
    return plan


def validate_required_motion_probe_result(result, plan):
    """Independently check measured coordinates; a producer's pass flag is insufficient."""
    plan = validate_required_motion_probe_plan(plan)
    _require(isinstance(result, Mapping) and result.get("schema_version") == RESULT_SCHEMA and
             result.get("status") == "passed" and result.get("blockers") == [] and
             result.get("plan_digest") == plan["plan_digest"] and
             result.get("requirements_digest") == plan["requirements_digest"] and
             result.get("result_digest") == _digest(result, "result_digest") and
             result.get("robot_embodiment_used") is False and result.get("learned_policy_queried") is False and
             result.get("task_success_proven") is False and result.get("physics_authority_granted") is False,
             "result_invalid")
    rows = result.get("phase_rows")
    _require(isinstance(rows, list) and len(rows) == len(plan["phases"]) and
             result.get("motion_ids") == [row["motion_id"] for row in plan["motions"]], "result_coverage_invalid")
    for row, phase in zip(rows, plan["phases"], strict=True):
        _require(row.get("phase_id") == phase["phase_id"] and row.get("commanded_positions") == phase["positions"] and
                 isinstance(row.get("joint_readback"), Mapping) and set(row["joint_readback"]) == set(phase["positions"]),
                 "result_coverage_invalid")
        for motion in plan["motions"]:
            joint = motion["joint_id"]
            observed = row["joint_readback"][joint]
            _require(_number(observed.get("position")) and _number(observed.get("velocity")) and
                     abs(observed["position"] - phase["positions"][joint]) <= motion["position_tolerance"] and
                     abs(observed["velocity"]) <= motion["maximum_abs_velocity"], "result_motion_failed")
    return dict(result)
