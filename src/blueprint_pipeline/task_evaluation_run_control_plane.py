"""Durable plan, authorization, local execution, and aggregation for one run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import (
    DecisionEvidenceRequest,
    EvidenceMethodProfile,
    EvidencePlan,
    MaintainedSiteTaskTestbed,
    QualificationRecord,
    canonical_digest,
    canonical_json,
)
from .core.security_controls import strict_identifier
from .decision_evidence_execution import build_decision_envelope, execute_evidence_plan
from .decision_evidence_router import route_decision_evidence
from .local_evidence_adapters import authorized_local_evidence_adapter_registry
from .task_evaluation_run_state import TaskEvaluationRunStateStore
from .task_evaluation_method_catalog import validate_task_evaluation_method_catalog
from .task_evaluation_run_webapp_sync import sync_task_evaluation_run_to_webapp


class TaskEvaluationRunControlPlaneError(ValueError):
    pass


def _run_root(root: str | Path, run_id: str) -> Path:
    return Path(root).expanduser().resolve() / "runs" / str(run_id)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise TaskEvaluationRunControlPlaneError(
                f"immutable_run_artifact_conflict:{path.name}"
            )


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TaskEvaluationRunControlPlaneError(f"run_artifact_missing:{path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TaskEvaluationRunControlPlaneError(f"run_artifact_invalid:{path.name}")
    return value


def _binding(
    request: Mapping[str, Any], testbed: Mapping[str, Any]
) -> dict[str, Any]:
    sources = testbed.get("source_capture_bundles")
    source_rows = sources if isinstance(sources, list) else []
    return {
        "capture_digest": canonical_digest({"source_capture_bundles": source_rows}),
        "testbed_digest": testbed["testbed_digest"],
        "request_digest": request["request_digest"],
    }


def prepare_task_evaluation_run(
    *,
    state_root: str | Path,
    run_id: str,
    capture_session_id: str,
    intake_id: str,
    request_value: Mapping[str, Any],
    testbed_value: Mapping[str, Any],
    method_values: Sequence[Mapping[str, Any]],
    qualification_values: Sequence[Mapping[str, Any]],
    idempotency_key: str,
    method_catalog_value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        run_id = strict_identifier(run_id, field="run_id", max_length=192)
        capture_session_id = strict_identifier(
            capture_session_id, field="capture_session_id", max_length=192
        )
        intake_id = strict_identifier(intake_id, field="intake_id", max_length=192)
    except ValueError as exc:
        raise TaskEvaluationRunControlPlaneError(str(exc)) from exc
    request = DecisionEvidenceRequest.from_mapping(request_value).to_mapping()
    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed_value).to_mapping()
    matching_sources = [
        source
        for source in testbed["source_capture_bundles"]
        if isinstance(source, Mapping) and source.get("bundle_id") == intake_id
    ]
    if len(matching_sources) != 1:
        raise TaskEvaluationRunControlPlaneError("run_intake_testbed_binding_mismatch")
    catalog = (
        validate_task_evaluation_method_catalog(method_catalog_value)
        if method_catalog_value is not None
        else None
    )
    effective_methods = catalog["method_profiles"] if catalog else method_values
    effective_qualifications = catalog["qualifications"] if catalog else qualification_values
    methods = [EvidenceMethodProfile.from_mapping(row).to_mapping() for row in effective_methods]
    qualifications = [
        QualificationRecord.from_mapping(row).to_mapping()
        for row in effective_qualifications
    ]
    plan = route_decision_evidence(request, testbed, methods, qualifications).to_mapping()
    root = _run_root(state_root, run_id)
    artifacts = root / "artifacts"
    artifact_values: list[tuple[str, Mapping[str, Any]]] = [
        ("run_context.json", {
            "schema_version": "task_evaluation_run_context.v1",
            "run_id": run_id,
            "capture_session_id": capture_session_id,
            "intake_id": intake_id,
        }),
        ("request.json", request),
        ("testbed.json", testbed),
        ("evidence_plan.json", plan),
        ("method_profiles.json", {"method_profiles": methods}),
        ("qualifications.json", {"qualifications": qualifications}),
    ]
    if catalog is not None:
        artifact_values.append(("method_catalog.json", catalog))
    for name, value in artifact_values:
        _write_immutable(artifacts / name, value)
    store = TaskEvaluationRunStateStore(state_root)
    binding = _binding(request, testbed)
    actor = {"role": "pipeline", "identity": "pipeline:decision-evidence-router"}
    store.transition(
        run_id=run_id,
        from_state=None,
        to_state="testbed_ready",
        idempotency_key=f"{idempotency_key}-testbed-ready",
        actor=actor,
        binding=binding,
        artifacts={"testbed_digest": testbed["testbed_digest"]},
    )
    store.transition(
        run_id=run_id,
        from_state="testbed_ready",
        to_state="planning",
        idempotency_key=f"{idempotency_key}-planning",
        actor=actor,
        binding=binding,
        artifacts={"request_digest": request["request_digest"]},
    )
    state = store.transition(
        run_id=run_id,
        from_state="planning",
        to_state="authorization_required",
        idempotency_key=f"{idempotency_key}-authorization-required",
        actor=actor,
        binding=binding,
        artifacts={"plan_digest": plan["plan_digest"]},
    )
    method_by_digest = {row["method_profile_digest"]: row for row in methods}
    planned_digests = sorted({
        str(step.get("method_profile_digest") or "")
        for claim_plan in plan.get("claim_plans", [])
        if isinstance(claim_plan, Mapping)
        for step in [
            *claim_plan.get("selected_methods", []),
            *claim_plan.get("escalation_methods", []),
        ]
        if isinstance(step, Mapping) and step.get("method_profile_digest")
    })
    return {
        "schema_version": "task_evaluation_run_preparation.v1",
        "run_id": run_id,
        "capture_session_id": capture_session_id,
        "intake_id": intake_id,
        "state": state["to_state"],
        "request": request,
        "evidence_plan": plan,
        "method_catalog": (
            {
                "catalog_id": catalog["catalog_id"],
                "version": catalog["version"],
                "catalog_digest": catalog["catalog_digest"],
                "pipeline_owned": True,
            }
            if catalog is not None
            else {"pipeline_owned": False, "source": "caller_supplied_v1_compatibility"}
        ),
        "authorization_candidates": [
            {
                "adapter_reference": method_by_digest[digest]["adapter_reference"],
                "method_id": method_by_digest[digest]["method_id"],
                "method_version": method_by_digest[digest]["version"],
                "method_profile_digest": digest,
                "method_family": method_by_digest[digest]["method_family"],
                "expected_cost_usd": method_by_digest[digest]["expected_cost_usd"],
                "proof_tier": method_by_digest[digest]["proof_tier"],
                "execution_authorized": False,
            }
            for digest in planned_digests
        ],
        "execution_started": False,
        "proof_boundary": state["proof_boundary"],
    }


def authorize_task_evaluation_run(
    *,
    state_root: str | Path,
    run_id: str,
    plan_digest: str,
    authorized_adapter_references: Sequence[str],
    actor: Mapping[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    root = _run_root(state_root, run_id)
    plan = EvidencePlan.from_mapping(_read(root / "artifacts" / "evidence_plan.json")).to_mapping()
    state = TaskEvaluationRunStateStore(state_root).inspect(run_id)
    if state["state"] != "authorization_required":
        raise TaskEvaluationRunControlPlaneError("run_not_awaiting_authorization")
    if plan["plan_digest"] != plan_digest:
        raise TaskEvaluationRunControlPlaneError("authorization_plan_digest_mismatch")
    registry = authorized_local_evidence_adapter_registry(authorized_adapter_references)
    actor_value = dict(actor)
    if any(
        str(key).lower() in {"authorization", "credential", "credentials", "password", "secret", "token"}
        or str(key).lower().endswith(("_token", "_secret", "_password"))
        for key in actor_value
    ):
        raise TaskEvaluationRunControlPlaneError("authorization_actor_secret_forbidden")
    authorization = {
        "schema_version": "task_evaluation_run_execution_authorization.v1",
        "run_id": run_id,
        "plan_digest": plan_digest,
        "authorized_adapter_references": registry.manifest(),
        "actor": actor_value,
        "idempotency_key": idempotency_key,
        "live_provider_execution": False,
        "paid_compute_authorized": False,
        "physical_robot_run_authorized": False,
        "proof_boundary": {
            "authorization_is_method_qualification": False,
            "simulation_is_physical_success": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }
    authorization["authorization_digest"] = canonical_digest(
        authorization, digest_field="authorization_digest"
    )
    _write_immutable(root / "artifacts" / "execution_authorization.json", authorization)
    return authorization


def execute_and_aggregate_task_evaluation_run(
    *, state_root: str | Path, run_id: str
) -> dict[str, Any]:
    root = _run_root(state_root, run_id)
    artifacts = root / "artifacts"
    request = DecisionEvidenceRequest.from_mapping(_read(artifacts / "request.json")).to_mapping()
    testbed = MaintainedSiteTaskTestbed.from_mapping(_read(artifacts / "testbed.json")).to_mapping()
    plan = EvidencePlan.from_mapping(_read(artifacts / "evidence_plan.json")).to_mapping()
    methods = [
        EvidenceMethodProfile.from_mapping(row).to_mapping()
        for row in _read(artifacts / "method_profiles.json").get("method_profiles", [])
    ]
    qualifications = [
        QualificationRecord.from_mapping(row).to_mapping()
        for row in _read(artifacts / "qualifications.json").get("qualifications", [])
    ]
    authorization = _read(artifacts / "execution_authorization.json")
    if authorization.get("plan_digest") != plan["plan_digest"]:
        raise TaskEvaluationRunControlPlaneError("execution_authorization_stale")
    registry = authorized_local_evidence_adapter_registry(
        authorization.get("authorized_adapter_references", [])
    )
    store = TaskEvaluationRunStateStore(state_root)
    current = store.inspect(run_id)["state"]
    terminal = {"decided", "partially_decided", "abstained"}
    if current in terminal:
        context = _read(artifacts / "run_context.json")
        envelope = _read(artifacts / "decision_envelope.json")
        webapp_sync = sync_task_evaluation_run_to_webapp(
            capture_session_id=str(context.get("capture_session_id") or ""),
            intake_id=str(context.get("intake_id") or ""),
            run_id=run_id,
            state=current,
            evidence_plan=plan,
            decision_envelope=envelope,
        )
        return {
            "schema_version": "task_evaluation_run_execution_result.v1",
            "run_id": run_id,
            "state": current,
            "already_exists": True,
            "decision_envelope": envelope,
            "webapp_sync": webapp_sync,
        }
    binding = _binding(request, testbed)
    actor = {"role": "pipeline", "identity": "pipeline:evidence-executor"}
    if current == "authorization_required":
        store.transition(
            run_id=run_id,
            from_state=current,
            to_state="executing",
            idempotency_key=f"execute-{plan['plan_digest'][7:23]}",
            actor=actor,
            binding=binding,
            artifacts={"authorization_digest": authorization["authorization_digest"]},
        )
        current = "executing"
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        methods,
        qualifications,
        registry=registry,
    )
    result_values = [result.to_mapping() for result in execution.results]
    _write_immutable(artifacts / "evidence_execution.json", dict(execution.execution_manifest))
    for result in result_values:
        _write_immutable(artifacts / "evidence_results" / f"{result['result_digest'][7:]}.json", result)
    if current == "executing":
        store.transition(
            run_id=run_id,
            from_state="executing",
            to_state="aggregating",
            idempotency_key=f"aggregate-{plan['plan_digest'][7:23]}",
            actor=actor,
            binding=binding,
            artifacts={"execution_status": execution.execution_manifest["status"]},
        )
    envelope = build_decision_envelope(request, testbed, plan, result_values).to_mapping()
    _write_immutable(artifacts / "decision_envelope.json", envelope)
    terminal_state = {
        "decision": "decided",
        "partial_decision": "partially_decided",
        "abstention": "abstained",
    }[envelope["overall_outcome"]]
    state = store.transition(
        run_id=run_id,
        from_state="aggregating",
        to_state=terminal_state,
        idempotency_key=f"decision-{envelope['decision_envelope_digest'][7:23]}",
        actor={"role": "pipeline", "identity": "pipeline:decision-aggregator"},
        binding=binding,
        artifacts={"decision_envelope_digest": envelope["decision_envelope_digest"]},
    )
    context = _read(artifacts / "run_context.json")
    webapp_sync = sync_task_evaluation_run_to_webapp(
        capture_session_id=str(context.get("capture_session_id") or ""),
        intake_id=str(context.get("intake_id") or ""),
        run_id=run_id,
        state=state["to_state"],
        evidence_plan=plan,
        decision_envelope=envelope,
    )
    return {
        "schema_version": "task_evaluation_run_execution_result.v1",
        "run_id": run_id,
        "state": state["to_state"],
        "already_exists": False,
        "execution_manifest": execution.execution_manifest,
        "evidence_results": result_values,
        "decision_envelope": envelope,
        "webapp_sync": webapp_sync,
    }


__all__ = [
    "TaskEvaluationRunControlPlaneError",
    "authorize_task_evaluation_run",
    "execute_and_aggregate_task_evaluation_run",
    "prepare_task_evaluation_run",
]
