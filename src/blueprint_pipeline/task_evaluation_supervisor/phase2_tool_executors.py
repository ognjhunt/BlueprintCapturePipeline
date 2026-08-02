"""Phase-2 artifact executors for the Task Evaluation Supervisor tool registry."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from ..common import write_json
from ..decision_evidence_contracts import canonical_digest
from ..decision_evidence_router import route_decision_evidence
from ..evaluation_run_contract import validate_evaluation_run_spec
from .phase2_artifacts import (
    authorization_request,
    clarification_request,
    scenario_proposal_set,
    targeted_recapture_request,
    write_phase2_artifact,
)


def _safe_artifact_name(value: Any) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-.")
    return rendered[:192] or "leaf-run"


def _materialize_leaf_runs(
    *,
    context: Any,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request = context.decision_request
    testbed = context.testbed
    request_digest = request.get("request_digest") if isinstance(request, Mapping) else None
    testbed_digest = testbed.get("testbed_digest") if isinstance(testbed, Mapping) else None
    if (
        not isinstance(request, Mapping)
        or not isinstance(testbed, Mapping)
        or not request_digest
        or not testbed_digest
        or arguments.get("request_digest") != request_digest
        or arguments.get("testbed_digest") != testbed_digest
    ):
        raise ValueError("registered_tool_bound_artifact_mismatch:materialize_compiled_leaf_runs")
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:materialize_compiled_leaf_runs")
    generated_root = Path(root_value) / "generated"
    output_root = generated_root / "compiled_leaf_runs"
    plan = route_decision_evidence(
        request,
        testbed,
        context.method_profiles,
        context.qualifications,
    ).to_mapping()
    if isinstance(context.evidence_plan, Mapping):
        supplied_digest = context.evidence_plan.get("plan_digest")
        if supplied_digest != plan.get("plan_digest"):
            raise ValueError("deterministic_evidence_plan_drift")
    plan_path = generated_root / "evidence_plan.json"
    write_json(plan_path, plan)
    plan_reference = {
        "artifact_path": str(plan_path.relative_to(Path(root_value))),
        "artifact_digest": plan["plan_digest"],
        "artifact_type": "evidence_plan.v1",
        "plan_id": plan["plan_id"],
    }
    rows = plan.get("compiled_evaluation_run_specs")
    if not isinstance(rows, list):
        raise ValueError("compiled_leaf_run_specs_not_list")
    references: list[dict[str, Any]] = [plan_reference]
    seen_run_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("compiled_leaf_run_spec_not_mapping")
        spec = dict(row)
        validation = validate_evaluation_run_spec(spec)
        if validation.get("status") != "passed":
            raise ValueError("compiled_leaf_run_spec_invalid")
        run_id = str(spec.get("run_id") or "")
        if not run_id or run_id in seen_run_ids:
            raise ValueError("compiled_leaf_run_id_missing_or_duplicate")
        seen_run_ids.add(run_id)
        artifact_path = output_root / f"{_safe_artifact_name(run_id)}.json"
        write_json(artifact_path, spec)
        references.append(
            {
                "artifact_path": str(artifact_path.relative_to(Path(root_value))),
                "artifact_digest": canonical_digest(spec),
                "artifact_type": "evaluation_run_spec.v1",
                "run_id": run_id,
            }
        )
    return (
        {
            "contract_present": True,
            "digest_matches": True,
            "plan_digest": plan["plan_digest"],
            "compiled_leaf_run_count": len(references) - 1,
            "compiled_leaf_run_references": references[1:],
            "provider_execution_started": False,
            "proof_state_changed": False,
        },
        references,
    )


def _materialize_targeted_recapture_request(
    *,
    context: Any,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:propose_targeted_recapture")
    testbed_digest = (
        context.testbed.get("testbed_digest") if isinstance(context.testbed, Mapping) else None
    )
    capture_digest = (
        context.capture_build.get("capture_build_digest")
        if isinstance(context.capture_build, Mapping)
        else None
    )
    source_digest = arguments.get("source_digest")
    if source_digest not in {testbed_digest, capture_digest}:
        raise ValueError("registered_tool_bound_artifact_mismatch:propose_targeted_recapture")
    missing = arguments.get("missing_evidence")
    if not isinstance(missing, list) or not missing:
        raise ValueError("targeted_recapture_missing_evidence_required")
    normalized_missing = sorted(
        {str(item).strip() for item in missing if isinstance(item, str) and str(item).strip()}
    )
    if not normalized_missing:
        raise ValueError("targeted_recapture_missing_evidence_required")
    if len(normalized_missing) > 50 or any(len(item) > 500 for item in normalized_missing):
        raise ValueError("targeted_recapture_scope_out_of_range")
    if arguments.get("full_site_recapture_requested") is True:
        raise ValueError("full_site_recapture_requires_separate_operator_authorization")
    request = targeted_recapture_request(
        run_id=context.run_id,
        source_digest=str(source_digest),
        source_type="site_task_testbed" if source_digest == testbed_digest else "capture_build",
        missing_evidence=normalized_missing,
    )
    artifact_path = (
        Path(root_value)
        / "generated"
        / "targeted_recapture_requests"
        / f"{_safe_artifact_name(context.run_id)}.json"
    )
    write_json(artifact_path, request)
    reference = {
        "artifact_path": str(artifact_path.relative_to(Path(root_value))),
        "artifact_digest": request["targeted_recapture_request_digest"],
        "artifact_type": "targeted_recapture_request.v1",
        "request_id": request["request_id"],
    }
    return (
        {
            "contract_present": True,
            "digest_matches": True,
            "request_id": request["request_id"],
            "targeted_recapture_request_digest": request["targeted_recapture_request_digest"],
            "capture_started": False,
            "proof_state_changed": False,
        },
        [reference],
    )


def _source_digest(context: Any, value: Any) -> str:
    candidates = {
        (context.capture_build or {}).get("capture_build_digest"),
        (context.decision_request or {}).get("request_digest"),
        (context.testbed or {}).get("testbed_digest"),
    }
    if value not in candidates:
        raise ValueError("registered_tool_source_digest_mismatch")
    return str(value)


def _materialize_clarification_request(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:materialize_clarification_request"
        )
    artifact = clarification_request(
        run_id=context.run_id,
        source_digest=_source_digest(context, arguments.get("source_digest")),
        questions=arguments.get("questions") or [],
        blocking_fields=arguments.get("blocking_fields") or [],
    )
    path = write_phase2_artifact(
        root_value,
        "generated/clarification_requests/request.json",
        artifact,
    )
    reference = {
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": artifact["clarification_request_digest"],
        "artifact_type": "task_evaluation_clarification_request.v1",
        "request_id": artifact["request_id"],
    }
    return {
        "contract_present": True,
        "digest_matches": True,
        "request_id": artifact["request_id"],
        "awaiting_customer_response": True,
        "proof_state_changed": False,
    }, [reference]


def _materialize_authorization_request(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:materialize_authorization_request"
        )
    authority = context.authority_envelope or {}
    artifact = authorization_request(
        run_id=context.run_id,
        tool_id=str(arguments.get("tool_id") or ""),
        reason=str(arguments.get("reason") or ""),
        requested_max_cost_usd=float(arguments.get("requested_max_cost_usd") or 0.0),
        requested_ttl_seconds=int(arguments.get("requested_ttl_seconds") or 0),
        immutable_input_digests=authority.get("immutable_input_digests") or [],
        requested_retry_count=int(arguments.get("requested_retry_count") or 0),
        requested_provider_ids=arguments.get("requested_provider_ids") or [],
        requested_action_ids=arguments.get("requested_action_ids") or [],
    )
    path = write_phase2_artifact(
        root_value,
        "generated/authorization_requests/request.json",
        artifact,
    )
    reference = {
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": artifact["authorization_request_digest"],
        "artifact_type": "task_evaluation_authorization_request.v1",
        "request_id": artifact["request_id"],
    }
    return {
        "contract_present": True,
        "digest_matches": True,
        "request_id": artifact["request_id"],
        "authorization_granted": False,
        "proof_state_changed": False,
    }, [reference]


def _materialize_scenario_proposal_set(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:propose_adversarial_scenarios")
    request_digest = _source_digest(context, arguments.get("request_digest"))
    scenarios = arguments.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("scenario_proposals_must_be_list")
    artifact = scenario_proposal_set(
        run_id=context.run_id,
        request_digest=request_digest,
        scenarios=[row for row in scenarios if isinstance(row, Mapping)],
        candidate_results_observed=arguments.get("candidate_results_observed") is True,
    )
    path = write_phase2_artifact(
        root_value,
        "generated/scenario_proposals/proposal_set.json",
        artifact,
    )
    reference = {
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": artifact["scenario_proposal_set_digest"],
        "artifact_type": "task_evaluation_scenario_proposal_set.v1",
        "proposal_set_id": artifact["proposal_set_id"],
    }
    return {
        "contract_present": True,
        "digest_matches": True,
        "scenario_count": len(artifact["scenarios"]),
        "frozen": False,
        "hidden_labels_included": False,
        "proof_state_changed": False,
    }, [reference]


def _execute_preauthorized_recovery(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    controller = getattr(context, "recovery_controller", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:execute_preauthorized_recovery")
    if controller is None:
        raise ValueError("preauthorized_recovery_controller_missing")
    result = controller.execute(arguments)
    path = write_phase2_artifact(
        root_value,
        (f"generated/recovery_attempts/{_safe_artifact_name(result['attempt_id'])}.json"),
        result,
    )
    reference = {
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": result["recovery_result_digest"],
        "artifact_type": "task_evaluation_recovery_result.v1",
        "attempt_id": result["attempt_id"],
    }
    return result, [reference]
