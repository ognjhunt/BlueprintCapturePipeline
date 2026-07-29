"""Append-only physical outcome join and conservative learning update."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import (
    DecisionEnvelope,
    EvidenceMethodProfile,
    MaintainedSiteTaskTestbed,
    PhysicalOutcomeJoin,
    QualificationRecord,
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sample_ids(value: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for container_key in ("prediction", "observed_outcome"):
        container = value.get(container_key)
        if not isinstance(container, Mapping):
            continue
        sample_id = _string(container.get("sample_id"))
        if sample_id:
            ids.add(sample_id)
        raw_ids = container.get("sample_ids")
        if isinstance(raw_ids, list):
            ids.update(_string(item) for item in raw_ids if _string(item))
    return ids


def _next_version(version: str) -> str:
    try:
        return str(int(version) + 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("testbed_version_not_incrementable") from exc


@dataclass(frozen=True)
class PhysicalOutcomeLearningUpdate:
    physical_outcome: PhysicalOutcomeJoin
    new_testbed: MaintainedSiteTaskTestbed
    calibration_record: QualificationRecord
    history_immutable: bool = True


def join_physical_outcome(
    *,
    testbed_value: Mapping[str, Any],
    decision_value: Mapping[str, Any],
    outcome_value: Mapping[str, Any],
    method_profile_value: Mapping[str, Any],
    existing_outcome_values: Sequence[Mapping[str, Any]] = (),
) -> PhysicalOutcomeLearningUpdate:
    """Join later reality without rewriting the historical plan or decision.

    Sample identifiers may belong to calibration or held-out, never both.  The
    update emits a new immutable testbed version and a method-bound calibration
    record; cross-site/task/embodiment transfer remains disabled.
    """

    testbed = MaintainedSiteTaskTestbed.from_mapping(testbed_value)
    decision = DecisionEnvelope.from_mapping(decision_value)
    outcome = PhysicalOutcomeJoin.from_mapping(outcome_value)
    method = EvidenceMethodProfile.from_mapping(method_profile_value)
    testbed_mapping = testbed.to_mapping()
    decision_mapping = decision.to_mapping()
    outcome_mapping = outcome.to_mapping()
    method_mapping = method.to_mapping()

    if outcome_mapping["testbed_id"] != testbed_mapping["testbed_id"]:
        raise ValueError("physical_outcome_testbed_id_mismatch")
    if outcome_mapping["testbed_version"] != testbed_mapping["version"]:
        raise ValueError("physical_outcome_testbed_version_mismatch")
    if outcome_mapping["testbed_digest"] != testbed.digest:
        raise ValueError("physical_outcome_testbed_digest_mismatch")
    if outcome_mapping["prediction_digest"] not in set(
        decision_mapping.get("input_run_result_testbed_digests") or []
    ):
        raise ValueError("physical_outcome_prediction_not_bound_to_decision")

    incoming_partition = _string(outcome_mapping.get("partition"))
    incoming_samples = _sample_ids(outcome_mapping)
    for raw_existing in existing_outcome_values:
        existing = PhysicalOutcomeJoin.from_mapping(raw_existing).to_mapping()
        overlap = incoming_samples & _sample_ids(existing)
        if overlap and _string(existing.get("partition")) != incoming_partition:
            raise ValueError(
                "calibration_heldout_leakage:" + ",".join(sorted(overlap))
            )
        if existing["physical_outcome_digest"] == outcome.digest:
            raise ValueError("physical_outcome_already_joined")

    new_testbed = copy.deepcopy(testbed_mapping)
    new_testbed.pop("testbed_digest", None)
    new_testbed["version"] = _next_version(_string(testbed_mapping.get("version")))
    new_testbed["predecessor_testbed_digest"] = testbed.digest
    supersedes = list(new_testbed.get("supersedes") or [])
    if testbed.digest not in supersedes:
        supersedes.append(testbed.digest)
    new_testbed["supersedes"] = sorted(supersedes)
    history = list(new_testbed.get("physical_outcome_history_refs") or [])
    history.append(
        {
            "outcome_id": outcome_mapping.get("outcome_id"),
            "physical_outcome_digest": outcome.digest,
            "partition": incoming_partition,
        }
    )
    new_testbed["physical_outcome_history_refs"] = sorted(
        history, key=lambda row: _string(row.get("physical_outcome_digest"))
    )
    new_testbed["lifecycle_state"] = "active"
    new_testbed["cross_domain_calibration_transfer_enabled"] = False
    new_testbed_artifact = MaintainedSiteTaskTestbed.from_mapping(new_testbed)

    prediction = dict(outcome_mapping.get("prediction") or {})
    observed = dict(outcome_mapping.get("observed_outcome") or {})
    metrics = dict(observed.get("qualification_metrics") or {})
    status = "qualified" if metrics.get("qualification_gate_passed") is True else "debug_only"
    qualification_value = {
        "schema_version": "evidence_method_qualification.v1",
        "qualification_id": f"physical-join-{outcome_mapping['outcome_id']}",
        "method_id": method_mapping.get("method_id"),
        "method_version": method_mapping.get("version"),
        "method_profile_digest": method.digest,
        "implementation_digest": method_mapping.get("implementation_digest"),
        "claim_type": _string(prediction.get("claim_type")) or "physical_task_success",
        "task_family": _string(prediction.get("task_family")) or outcome_mapping.get("task_id"),
        "site_domain_conditions": outcome_mapping.get("condition"),
        "embodiment": outcome_mapping.get("robot_embodiment"),
        "sensors": outcome_mapping.get("sensors"),
        "controller_action_representation": outcome_mapping.get("controller"),
        "evaluator": outcome_mapping.get("evaluator"),
        "evaluator_digest": outcome_mapping.get("evaluator_digest"),
        "predictions": [
            {
                "prediction_digest": outcome_mapping.get("prediction_digest"),
                **prediction,
            }
        ],
        "accepted_real_outcomes": [
            {
                "physical_outcome_digest": outcome.digest,
                **observed,
            }
        ],
        "calibration_partition": incoming_partition,
        "confidence_intervals": dict(
            metrics.get("confidence_intervals")
            if isinstance(metrics.get("confidence_intervals"), Mapping)
            else {"status": "not_estimated", "reason": "single_outcome"}
        ),
        "coverage": _number(metrics.get("coverage"), 0.0),
        "abstention_rate": _number(metrics.get("abstention_rate"), 1.0),
        "false_safe_rate": _number(metrics.get("false_safe_rate"), 1.0),
        "false_reject_rate": _number(metrics.get("false_reject_rate"), 1.0),
        "provenance": {
            **dict(outcome_mapping.get("provenance") or {}),
            "physical_outcome_digest": outcome.digest,
            "source_testbed_digest": testbed.digest,
            "new_testbed_digest": new_testbed_artifact.digest,
            "cross_domain_transfer_enabled": False,
        },
        "owner_evidence": list(outcome_mapping.get("owner_evidence") or []),
        "status": status,
        "self_grading": False,
        "subject_provider_id": _string(
            dict(outcome_mapping.get("runtime_provider") or {}).get("provider_id")
        ),
        "evaluator_provider_id": _string(
            dict(outcome_mapping.get("evaluator") or {}).get("provider_id")
        ),
    }
    calibration = QualificationRecord.from_mapping(qualification_value)
    return PhysicalOutcomeLearningUpdate(
        physical_outcome=outcome,
        new_testbed=new_testbed_artifact,
        calibration_record=calibration,
    )


__all__ = ["PhysicalOutcomeLearningUpdate", "join_physical_outcome"]
