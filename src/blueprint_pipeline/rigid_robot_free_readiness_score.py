"""Score a rigid scene's robot-free probe with the exact success contract teams are judged by (pure).

Beta journey step: L5 team policy evaluation. The scoring core of
``task_evaluation_scene_rigid_robot_free_readiness`` (owner decision
2026-10-12), split out so the GPU host can score a probe it just recorded
before any team episode runs, without the control plane's revision
documents. See that module for the checks and what they prove.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .adp_rigid_task_scoring import score_rigid_task_episode
from .adp_task_scoring import TaskNeutralScoringError
from .decision_evidence_contracts import canonical_digest
from .rigid_robot_free_probe_plan import (
    PHYSICS_HZ,
    ROBOT_SIDE_CRITERIA,
    RigidRobotFreeProbeUnsupported,
    contract_cases,
    physics_plan,
    probe_context,
    scorer_samples,
)
from .rigid_robot_free_probe_recorder import TRACE_SCHEMA

SCHEMA = "task_evaluation_configured_scene_rigid_robot_free_readiness.v1"
TASK_KIND = "rigid_pick_place"
REQUIRED_CHECKS = ("zero_action_not_success", "direct_placement_success", "contract_untouched_not_success",
                   "contract_goal_success", "contract_short_hold_not_success", "contract_drift_not_success",
                   "contract_outside_not_success")


def _score(spec: Mapping[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        report = score_rigid_task_episode(task_spec=spec, samples=samples)
    except TaskNeutralScoringError as exc:
        return {"status": "unscorable", "task_succeeded": None, "errors": list(exc.errors)[:20],
                "sample_count": len(samples)}
    return {"status": report["status"], "task_succeeded": report["task_succeeded"], "outcome": report["outcome"],
            "failed_criteria": report["failed_criteria"], "sample_count": len(samples)}


def _holds(result: Mapping[str, Any], *, succeed: bool) -> bool:
    if succeed:
        return result["task_succeeded"] is True
    return result["status"] == "scored" and result["task_succeeded"] is False


def _contract_checks(spec: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    cases, checks = {}, {}
    for name, (succeed, rows) in contract_cases(context).items():
        cases[name] = {**_score(spec, scorer_samples(context, rows)), "must_succeed": succeed}
        cases[name]["holds"] = _holds(cases[name], succeed=succeed)
    for name, result in cases.items():
        if not name.startswith("outside_"):
            checks[f"contract_{name}_{'success' if result['must_succeed'] else 'not_success'}"] = result["holds"]
    checks["contract_outside_not_success"] = all(r["holds"] for n, r in cases.items() if n.startswith("outside_"))
    not_applicable = {}
    if "tipped" not in cases:
        not_applicable["contract_tipped_not_success"] = "orientation_not_required"
    if "dropped" not in cases:
        not_applicable["contract_dropped_not_success"] = "no_drop_not_required"
    return cases, checks, not_applicable


class _Unavailable(Exception):
    pass


def _physics(spec: Mapping[str, Any], context: Mapping[str, Any], trace: Any, *, binding: Mapping[str, Any]
             ) -> dict[str, Any]:
    if not isinstance(trace, Mapping):
        raise _Unavailable("physics_trace_missing")
    if trace.get("schema_version") != TRACE_SCHEMA:
        raise _Unavailable("physics_trace_schema_invalid")
    if trace.get("trace_digest") != canonical_digest(dict(trace), digest_field="trace_digest"):
        raise _Unavailable("physics_trace_digest_invalid")
    if trace.get("status") != "recorded":
        raise _Unavailable("physics_trace_not_recorded:" + str(trace.get("error"))[:200])
    if trace.get("plan") != physics_plan(context):
        raise _Unavailable("physics_trace_plan_mismatch")
    recorded = trace.get("binding") if isinstance(trace.get("binding"), Mapping) else {}
    for key, expected in binding.items():
        if expected is not None and recorded.get(key) != expected:
            raise _Unavailable("physics_trace_binding_mismatch:" + key)
    hz = context["control_hz"]
    if trace.get("physics_hz") != PHYSICS_HZ or not hz or PHYSICS_HZ % hz:
        raise _Unavailable("physics_trace_rate_mismatch")
    decimation = round(PHYSICS_HZ / hz)
    rows: Sequence[Mapping[str, Any]] = trace.get("samples") or []
    untouched = [row for row in rows if row.get("phase") == "zero_action"]
    driven = [row for row in rows if row.get("phase") in {"carry", "release"}]
    proof = trace.get("proof_boundary") if isinstance(trace.get("proof_boundary"), Mapping) else {}
    if not untouched or not driven or proof.get("robot_embodiment_used") is not False:
        raise _Unavailable("physics_trace_incomplete")
    # One reading per control step, as a robot team's episode is sampled; the
    # placement episode starts from the untouched scene's final reading.
    zero = _score(spec, scorer_samples(context, untouched[decimation - 1::decimation]))
    placed = _score(spec, scorer_samples(context, untouched[-1:] + driven[decimation - 1::decimation]))
    return {"zero_action": zero, "direct_placement": placed, "physics_hz": PHYSICS_HZ, "control_frequency_hz": hz,
            "plan_digest": trace["plan"]["plan_digest"], "scene_context": recorded.get("scene_context"),
            "scene_collisions_modeled": recorded.get("scene_collisions_modeled"), "drive": proof.get("drive")}


def score_rigid_robot_free_readiness(*, task_spec: Mapping[str, Any], physics_probe: Mapping[str, Any] | None,
                                     configured_scene_revision_digest: str | None = None,
                                     replacement_asset_digest: str | None = None,
                                     unsupported_reason: str | None = None) -> dict[str, Any]:
    """Return a sealed receipt; only every contract and physics check passing admits evaluation."""

    spec = dict(task_spec)
    binding = {"configured_scene_revision_digest": configured_scene_revision_digest,
               "task_spec_digest": canonical_digest(spec), "replacement_asset_digest": replacement_asset_digest}
    base: dict[str, Any] = {
        "schema_version": SCHEMA, "task_kind": TASK_KIND, **binding,
        "physics_probe_digest": physics_probe.get("trace_digest") if isinstance(physics_probe, Mapping) else None}
    boundary: dict[str, Any] = {
        "robot_embodiment_used": False, "virtual_carrier": "kinematic_state_setting_then_release",
        "robot_side_criteria_not_exercised": list(ROBOT_SIDE_CRITERIA),
        "contract_checks_simulated": False, "simulator_evidence_is_physical_truth": False}
    try:
        if unsupported_reason is not None:
            raise RigidRobotFreeProbeUnsupported(unsupported_reason)
        context = probe_context(spec)
    except RigidRobotFreeProbeUnsupported as exc:
        receipt = {**base, "status": "probe_unavailable", "probe_error": exc.reason, "proof_boundary": boundary,
                   "learned_policy_evaluation_admitted": False, "receipt_digest": ""}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        return receipt
    cases, checks, not_applicable = _contract_checks(spec, context)
    # The contract the scorer actually applied: the authored one, or the
    # compatibility default derived from this exact specification.
    base["task_success_contract_digest"] = context["criteria_contract_digest"]
    try:
        physics = _physics(spec, context, physics_probe, binding=binding)
    except _Unavailable as exc:
        physics, error = None, str(exc)
    else:
        error = None
        checks["zero_action_not_success"] = _holds(physics["zero_action"], succeed=False)
        checks["direct_placement_success"] = _holds(physics["direct_placement"], succeed=True)
        boundary.update(scene_context=physics["scene_context"],
                        scene_collisions_modeled=physics["scene_collisions_modeled"])
    passed = physics is not None and all(checks.values()) and all(name in checks for name in REQUIRED_CHECKS)
    receipt = {**base, "status": "passed" if passed else "failed" if physics is not None else "probe_unavailable",
               **({"probe_error": error} if error else {}),
               "contract_cases": cases, "physics": physics, "checks": checks, "required_checks": list(REQUIRED_CHECKS),
               "not_applicable_checks": not_applicable, "proof_boundary": boundary,
               "learned_policy_evaluation_admitted": passed, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = ["REQUIRED_CHECKS", "SCHEMA", "TASK_KIND", "score_rigid_robot_free_readiness"]
