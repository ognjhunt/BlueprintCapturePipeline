"""Bridge from a selected measurement route to development execution.

Connects a ``route_selected`` measurement routing decision to the fail-closed
adapter execution boundary so a planned stage can actually run locally and
attach receipt-bound development evidence to the Evidence Plan flow.

Boundaries:

- a routing decision never authorizes execution; the caller's explicit
  ``execute`` flag plus the execution boundary's own gates do;
- when the routed stage's method differs from the executing worker's
  candidate (for example a fixture-labeled routed engine demonstrated by the
  real MuJoCo development worker), the record says so:
  ``binding_kind="development_demonstration_only"``;
- outputs are development evidence only — no qualification, no catalog
  mutation, no physical-success effect.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import EvidencePlan
from .measurement_adapter_execution import (
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
    validate_measurement_adapter_execution_bundle,
)


ROUTED_EXECUTION_SCHEMA_VERSION = "routed_development_execution.v1"
EVIDENCE_PLAN_ATTACHMENT_SCHEMA_VERSION = (
    "evidence_plan_routed_development_attachment.v1"
)
ROUTED_CROSS_ENGINE_REPORT_SCHEMA_VERSION = (
    "routed_cross_engine_development_report.v1"
)


class RoutedExecutionError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def execute_routed_development_stage(
    measurement_decision: Mapping[str, Any],
    *,
    stage_index: int,
    descriptor: Mapping[str, Any],
    benchmark_spec: Mapping[str, Any],
    case_manifest: Mapping[str, Any],
    worker_argv: Sequence[str],
    execution_id: str,
    implementation_id: str,
    implementation_version: str,
    implementation_digest: str,
    backend_id: str,
    precision: str,
    seed: int,
    solver_settings: Mapping[str, Any],
    execute: bool = False,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    decision = dict(measurement_decision)
    if decision.get("status") != "route_selected":
        raise RoutedExecutionError("routed_execution_requires_selected_route")
    stages = list(dict(decision.get("selected_route") or {}).get("stages") or [])
    if not 0 <= stage_index < len(stages):
        raise RoutedExecutionError("routed_execution_stage_index_invalid")
    stage = dict(stages[stage_index])
    routed_method_id = _string(stage.get("method_id"))
    if not routed_method_id:
        raise RoutedExecutionError("routed_execution_stage_method_missing")

    request = build_measurement_adapter_execution_request(
        descriptor,
        benchmark_spec,
        case_manifest,
        execution_id=execution_id,
        implementation_id=implementation_id,
        implementation_version=implementation_version,
        implementation_digest=implementation_digest,
        backend_id=backend_id,
        precision=precision,
        seed=seed,
        solver_settings=solver_settings,
        timeout_seconds=timeout_seconds,
    )
    bundle = run_measurement_adapter_execution(
        request, command_argv=list(worker_argv), execute=execute
    )
    receipt = bundle["receipt"]
    executing_candidate_id = _string(receipt.get("candidate_id"))
    record = {
        "schema_version": ROUTED_EXECUTION_SCHEMA_VERSION,
        "routing_id": decision.get("routing_id"),
        "routing_decision_digest": decision.get("routing_decision_digest"),
        "deterministic_policy_signature": decision.get("deterministic_policy_signature"),
        "stage_index": stage_index,
        "routed_method_id": routed_method_id,
        "routed_qualification_id": stage.get("qualification_id"),
        "routed_qualification_digest": stage.get("qualification_digest"),
        "executing_candidate_id": executing_candidate_id,
        "binding_kind": (
            "routed_method_development_execution"
            if executing_candidate_id == routed_method_id
            else "development_demonstration_only"
        ),
        "execution_bundle_digest": bundle.get("execution_bundle_digest"),
        "execution_receipt_digest": receipt.get("execution_receipt_digest"),
        "execution_status": receipt.get("status"),
        "evidence_class": receipt.get("evidence_class"),
        "prediction_digest": (
            dict(bundle.get("prediction") or {}).get("prediction_digest")
        ),
        "development_evidence_only": True,
        "route_authorized_execution": False,
        "qualification_created": False,
        "catalog_mutated": False,
        "physical_success_established": False,
    }
    record["routed_development_execution_digest"] = _digest(
        record, "routed_development_execution_digest"
    )
    return {"record": record, "bundle": bundle}


def attach_routed_development_evidence(
    evidence_plan_value: Mapping[str, Any],
    *,
    claim_id: str,
    routed_outcome: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an immutable, plan-bound attachment for one routed execution.

    Evidence plans are planning artifacts, so execution does not rewrite the
    plan or its digest.  This attachment is the result-side link: it binds the
    claim's exact measurement-routing decision to the execution receipt and
    prediction while retaining a development-only claim ceiling.
    """

    plan = EvidencePlan.from_mapping(evidence_plan_value).to_mapping()
    normalized_claim_id = _string(claim_id)
    claim_plans = [
        dict(row)
        for row in plan.get("claim_plans") or []
        if isinstance(row, Mapping) and _string(row.get("claim_id")) == normalized_claim_id
    ]
    if len(claim_plans) != 1:
        raise RoutedExecutionError("routed_attachment_claim_plan_missing_or_ambiguous")
    routing_decision = dict(
        claim_plans[0].get("measurement_routing_decision") or {}
    )
    if routing_decision.get("status") != "route_selected":
        raise RoutedExecutionError("routed_attachment_requires_selected_route")

    record = dict(routed_outcome.get("record") or {})
    try:
        bundle = validate_measurement_adapter_execution_bundle(
            routed_outcome.get("bundle") or {}
        )
    except (TypeError, ValueError) as exc:
        raise RoutedExecutionError("routed_attachment_execution_bundle_invalid") from exc
    receipt = dict(bundle["receipt"])
    prediction = dict(bundle.get("prediction") or {})
    expected_decision_digest = routing_decision.get("routing_decision_digest")
    errors: list[str] = []
    if record.get("schema_version") != ROUTED_EXECUTION_SCHEMA_VERSION:
        errors.append("routed_attachment_execution_record_schema_invalid")
    if not expected_decision_digest or record.get(
        "routing_decision_digest"
    ) != expected_decision_digest:
        errors.append("routed_attachment_routing_decision_mismatch")
    if record.get("execution_receipt_digest") != receipt.get(
        "execution_receipt_digest"
    ):
        errors.append("routed_attachment_receipt_digest_mismatch")
    if not prediction or record.get("prediction_digest") != prediction.get(
        "prediction_digest"
    ):
        errors.append("routed_attachment_prediction_digest_mismatch")
    if receipt.get("status") != "completed" or receipt.get(
        "evidence_class"
    ) != "development_execution":
        errors.append("routed_attachment_completed_development_execution_required")
    for key in (
        "development_evidence_only",
        "route_authorized_execution",
        "qualification_created",
        "physical_success_established",
    ):
        expected = key == "development_evidence_only"
        if record.get(key) is not expected:
            errors.append(f"routed_attachment_{key}_invalid")
    if errors:
        raise RoutedExecutionError(*errors)

    request = dict(bundle["request"])
    case = dict(request["case_manifest"])
    operating_point = dict(case.get("operating_point") or {})
    comparison_shape = operating_point.get("comparison_case_shape")
    comparison_shape = (
        dict(comparison_shape) if isinstance(comparison_shape, Mapping) else None
    )
    attachment = {
        "schema_version": EVIDENCE_PLAN_ATTACHMENT_SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "plan_digest": plan["plan_digest"],
        "claim_id": normalized_claim_id,
        "routing_decision_digest": expected_decision_digest,
        "routed_development_execution": record,
        "execution_receipt_digest": receipt["execution_receipt_digest"],
        "prediction": prediction,
        "case_binding": {
            "benchmark_id": case["benchmark_id"],
            "case_id": case["case_id"],
            "case_manifest_digest": case["case_manifest_digest"],
            "task_class": case["task_class"],
            "material_regime": case["material_regime"],
            "comparison_case_shape": comparison_shape,
        },
        "development_evidence_only": True,
        "qualification_created": False,
        "physical_success_established": False,
        "plan_mutated": False,
    }
    attachment["evidence_plan_attachment_digest"] = _digest(
        attachment, "evidence_plan_attachment_digest"
    )
    return attachment


def build_routed_cross_engine_development_report(
    attachments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare receipt-bound engine outputs for one routed logical case.

    The shared ``comparison_case_shape`` is carried inside each engine's case
    manifest.  Engine-specific manifests and solver settings remain distinct;
    this report only compares numeric metrics present in both predictions and
    never treats agreement as qualification or physical accuracy.
    """

    rows = [dict(value) for value in attachments]
    if len(rows) < 2:
        raise RoutedExecutionError("routed_cross_engine_requires_two_attachments")
    plan_digests = {_string(row.get("plan_digest")) for row in rows}
    claim_ids = {_string(row.get("claim_id")) for row in rows}
    routing_digests = {_string(row.get("routing_decision_digest")) for row in rows}
    if len(plan_digests) != 1 or "" in plan_digests:
        raise RoutedExecutionError("routed_cross_engine_plan_mismatch")
    if len(claim_ids) != 1 or "" in claim_ids:
        raise RoutedExecutionError("routed_cross_engine_claim_mismatch")
    if len(routing_digests) != 1 or "" in routing_digests:
        raise RoutedExecutionError("routed_cross_engine_routing_mismatch")

    shapes: list[dict[str, Any]] = []
    engine_rows: list[dict[str, Any]] = []
    for row in rows:
        expected_attachment_digest = _digest(row, "evidence_plan_attachment_digest")
        if row.get("schema_version") != EVIDENCE_PLAN_ATTACHMENT_SCHEMA_VERSION or row.get(
            "evidence_plan_attachment_digest"
        ) != expected_attachment_digest:
            raise RoutedExecutionError("routed_cross_engine_attachment_invalid")
        if row.get("development_evidence_only") is not True or any(
            row.get(key) is not False
            for key in ("qualification_created", "physical_success_established", "plan_mutated")
        ):
            raise RoutedExecutionError("routed_cross_engine_authority_boundary_invalid")
        case_binding = dict(row.get("case_binding") or {})
        shape = case_binding.get("comparison_case_shape")
        if not isinstance(shape, Mapping) or not shape:
            raise RoutedExecutionError("routed_cross_engine_case_shape_missing")
        shapes.append(dict(shape))
        execution = dict(row.get("routed_development_execution") or {})
        prediction = dict(row.get("prediction") or {})
        metrics = dict(prediction.get("observed_metrics") or {})
        engine_rows.append(
            {
                "executing_candidate_id": execution.get("executing_candidate_id"),
                "case_id": case_binding.get("case_id"),
                "case_manifest_digest": case_binding.get("case_manifest_digest"),
                "execution_receipt_digest": row.get("execution_receipt_digest"),
                "prediction_digest": prediction.get("prediction_digest"),
                "observed_metrics": metrics,
            }
        )
    encoded_shapes = {
        json.dumps(shape, sort_keys=True, separators=(",", ":")) for shape in shapes
    }
    if len(encoded_shapes) != 1:
        raise RoutedExecutionError("routed_cross_engine_case_shape_mismatch")
    candidate_ids = {
        _string(row.get("executing_candidate_id")) for row in engine_rows
    }
    if len(candidate_ids) != len(engine_rows) or "" in candidate_ids:
        raise RoutedExecutionError("routed_cross_engine_candidate_identity_invalid")

    metric_names = set.intersection(
        *[set(dict(row["observed_metrics"])) for row in engine_rows]
    )
    numeric_deltas: dict[str, float] = {}
    for metric_name in sorted(metric_names):
        values = [dict(row["observed_metrics"]).get(metric_name) for row in engine_rows]
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            numeric = [float(value) for value in values]
            numeric_deltas[metric_name] = max(numeric) - min(numeric)
    report = {
        "schema_version": ROUTED_CROSS_ENGINE_REPORT_SCHEMA_VERSION,
        "plan_digest": next(iter(plan_digests)),
        "claim_id": next(iter(claim_ids)),
        "routing_decision_digest": next(iter(routing_digests)),
        "comparison_case_shape": shapes[0],
        "engine_rows": sorted(
            engine_rows, key=lambda row: _string(row["executing_candidate_id"])
        ),
        "numeric_metric_ranges": numeric_deltas,
        "development_evidence_only": True,
        "engine_agreement_is_qualification": False,
        "physical_accuracy_established": False,
        "qualification_created": False,
    }
    report["routed_cross_engine_report_digest"] = _digest(
        report, "routed_cross_engine_report_digest"
    )
    return report


__all__ = [
    "EVIDENCE_PLAN_ATTACHMENT_SCHEMA_VERSION",
    "ROUTED_CROSS_ENGINE_REPORT_SCHEMA_VERSION",
    "ROUTED_EXECUTION_SCHEMA_VERSION",
    "RoutedExecutionError",
    "attach_routed_development_evidence",
    "build_routed_cross_engine_development_report",
    "execute_routed_development_stage",
]
