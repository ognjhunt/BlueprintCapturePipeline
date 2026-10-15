"""Rigid robot-free readiness: score the probe with the exact configured success contract.

Beta journey step: L5 team policy evaluation. Owner decision 2026-10-12:
robots never run in construction, and a configured scene becomes ready for
robot teams' evaluation only after robot-free checks of its success contract.
``task_evaluation_scene_robot_free_readiness`` does that for articulated
open/close; this is the rigid pick/place counterpart, scored through
``score_rigid_task_episode`` and the exact task specification teams are judged
by, one reading per control step.

Two kinds of evidence are sealed together, and only both admit evaluation:

- contract checks (no simulator): robot-free states set directly and held
  still. The untouched start, the object just outside every horizontal edge
  of the success region, tipped (when orientation matters), dropped (when the
  contract forbids drops), released for one sample less than the hold window,
  and drifting through the hold window must all fail; the authored goal must
  pass. These show the contract can tell success from failure at all.
- physics checks (a recorded simulator trace from
  ``rigid_robot_free_probe_recorder``): the scene left untouched must not
  succeed, and the object carried to the goal and let go must succeed once
  physics alone holds it there through the hold window.

Without a recorded trace bound to this revision, task specification and
replacement asset, the receipt is ``probe_unavailable`` and admits nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
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


def fetch_revision_documents(*, revision: Mapping[str, Any], output_root: str | Path,
                             fetcher: Callable[[Mapping[str, Any]], bytes] | None = None) -> dict[str, Path]:
    """The revision's documents, each verified against its published digest before use."""

    from . import task_evaluation_configured_controls_deferred_inputs as inputs

    root = Path(output_root) / "documents"
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    documents: dict[str, Path] = {}
    for contract_path, declared in inputs._document_references(revision).items():
        reference = inputs._reference(declared, blocker=f"robot_free_readiness_reference_invalid:{contract_path}")
        path = root / inputs._document_name(contract_path)
        if not inputs._retained_matches(path, reference=reference):
            payload = inputs._fetched(reference, fetcher=fetcher or inputs.default_reference_fetcher)
            inputs._write_immutable_bytes(path, payload, conflict="robot_free_readiness_document_conflict")
        documents[contract_path] = path
    return documents


def rigid_task_for_revision(*, revision: Mapping[str, Any], documents: Mapping[str, Path]) -> dict[str, Any]:
    """Adapt the revision's rigid task exactly as a robot team's evaluation would."""

    from . import task_evaluation_configured_controls_deferred_inputs as inputs
    from .task_evaluation_rigid_relocation_native_adapter import (
        NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
        adapt_rigid_relocation_task_template,
    )

    references = inputs._materialized_references(revision=revision, documents=documents)
    definition = adapt_rigid_relocation_task_template(
        configured_revision=revision, materialized_references=references)["native_task_definition"]
    native_import = json.loads(documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH].read_text())
    return {"task_spec": definition["task_spec"], "task_object_pose_world": definition["task_object_pose_world"],
            "replacement_asset_digest": native_import.get("asset_digest"),
            # The evaluation binds a destination body on top of this spec; scoring the
            # point task instead would probe a contract no team is judged by.
            "unsupported_reason": ("destination_body_relation"
                                   if revision["task_template"].get("destination") is not None else None)}


def rigid_robot_free_readiness_for_revision(*, revision: Mapping[str, Any], output_root: str | Path,
                                            physics_probe: Mapping[str, Any] | None,
                                            fetcher: Callable[[Mapping[str, Any]], bytes] | None = None
                                            ) -> dict[str, Any]:
    """Fetch the revision's exact documents, adapt its rigid task, and score."""

    documents = fetch_revision_documents(revision=revision, output_root=output_root, fetcher=fetcher)
    task = rigid_task_for_revision(revision=revision, documents=documents)
    return score_rigid_robot_free_readiness(
        task_spec=task["task_spec"], physics_probe=physics_probe,
        configured_scene_revision_digest=revision.get("revision_digest"),
        replacement_asset_digest=task["replacement_asset_digest"], unsupported_reason=task["unsupported_reason"])


__all__ = [
    "REQUIRED_CHECKS",
    "SCHEMA",
    "TASK_KIND",
    "fetch_revision_documents",
    "rigid_robot_free_readiness_for_revision",
    "rigid_task_for_revision",
    "score_rigid_robot_free_readiness",
]
