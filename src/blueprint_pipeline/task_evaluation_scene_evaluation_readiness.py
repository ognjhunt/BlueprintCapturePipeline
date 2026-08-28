"""Promote a configured scene only after canonical native controls pass."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_controls_admission import (
    validate_native_task_controls_admission,
)
from .task_evaluation_configured_scene_revision import (
    TaskEvaluationConfiguredSceneRevisionError,
    validate_configured_scene_revision,
)


RECEIPT_SCHEMA_VERSION = (
    "task_evaluation_configured_scene_evaluation_readiness.v1"
)
CONFIGURATION_COMPLETE_OFFERING_STATUSES = frozenset(
    {"launch_ready", "configured_controls_pending"}
)


class TaskEvaluationSceneEvaluationReadinessError(RuntimeError):
    """A configured scene could not be admitted for learned evaluation."""


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationSceneEvaluationReadinessError(
            "configured_scene_evaluation_readiness_input_invalid"
        ) from exc


def promote_configured_scene_evaluation_readiness(
    *,
    configured_scene_revision: Mapping[str, Any],
    configured_scene_offering: Mapping[str, Any],
    adapter_result: Mapping[str, Any],
    scene_plan: Mapping[str, Any],
    construction_result: Mapping[str, Any],
    control_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a digest-bound promotion receipt and evaluation-ready offering."""

    try:
        revision = validate_configured_scene_revision(
            _copy(configured_scene_revision)
        )
    except TaskEvaluationConfiguredSceneRevisionError as exc:
        raise TaskEvaluationSceneEvaluationReadinessError(
            "configured_scene_evaluation_readiness_revision_invalid"
        ) from exc
    offering = _copy(configured_scene_offering)
    adapter = _copy(adapter_result)
    construction = _copy(construction_result)
    controls = _copy(control_result)
    binding = offering.get("evaluation_preparation_binding")
    admission_declaration = offering.get("evaluation_admission")
    if (
        offering.get("schema_version")
        != "task_evaluation_configured_scene_offering.v1"
        or offering.get("status") != "configured_controls_pending"
        or offering.get("configuration_run_id")
        != revision.get("configuration_run_id")
        or offering.get("team_namespace") != revision.get("team_namespace")
        or not isinstance(binding, Mapping)
        or binding.get("configured_scene_revision_digest")
        != revision.get("revision_digest")
        or binding.get("configured_scene_bundle")
        != revision.get("configured_scene_bundle")
        or not isinstance(admission_declaration, Mapping)
        or admission_declaration.get("zero_action_required") is not True
        or admission_declaration.get("scripted_positive_required") is not True
        or admission_declaration.get("learned_policy_evaluation_admitted")
        is not False
        or offering.get("offering_digest")
        != canonical_digest(offering, digest_field="offering_digest")
    ):
        raise TaskEvaluationSceneEvaluationReadinessError(
            "configured_scene_evaluation_readiness_offering_invalid"
        )

    adapter_digest = adapter.get("result_digest")
    packet_receipt_digest = adapter.get("packet_receipt_digest")
    if (
        adapter.get("schema_version")
        != "task_evaluation_native_arena_adapter_result.v1"
        or adapter.get("status") != "native_arena_adapter_materialized"
        or adapter.get("configured_scene_revision_digest")
        != revision.get("revision_digest")
        or adapter.get("provider_mutation_performed") is not False
        or adapter.get("paid_execution_requested") is not False
        or not isinstance(packet_receipt_digest, str)
        or len(packet_receipt_digest) != 71
        or not packet_receipt_digest.startswith("sha256:")
        or adapter_digest
        != canonical_digest(adapter, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneEvaluationReadinessError(
            "configured_scene_evaluation_readiness_adapter_invalid"
        )

    try:
        controls_admission = validate_native_task_controls_admission(
            scene_plan=scene_plan,
            construction_result=construction,
            control_result=controls,
        )
    except ValueError as exc:
        raise TaskEvaluationSceneEvaluationReadinessError(
            f"configured_scene_evaluation_readiness_controls_invalid:{exc}"
        ) from exc
    if (
        construction.get("packet_receipt_digest") != packet_receipt_digest
        or controls.get("packet_receipt_digest") != packet_receipt_digest
        or controls_admission.get("packet_receipt_digest")
        != packet_receipt_digest
    ):
        raise TaskEvaluationSceneEvaluationReadinessError(
            "configured_scene_evaluation_readiness_packet_binding_mismatch"
        )

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "evaluation_ready",
        "configuration_run_id": revision["configuration_run_id"],
        "configured_scene_revision_digest": revision["revision_digest"],
        "base_configured_scene_offering_digest": offering["offering_digest"],
        "adapter_result_digest": adapter_digest,
        "packet_receipt_digest": packet_receipt_digest,
        "task_id": controls_admission["task_id"],
        "canonical_cell_id": controls_admission["cell_id"],
        "scene_plan_digest": controls_admission["scene_plan_digest"],
        "construction_result_digest": controls_admission[
            "construction_result_digest"
        ],
        "control_result_digest": controls_admission["control_result_digest"],
        "control_pair_digest": controls_admission["control_pair_digest"],
        "controls": controls_admission["controls"],
        "learned_policy_evaluation_admitted": True,
        "variation_matrix_started": False,
        "candidate_policy_queried": False,
        "simulator_evidence_is_physical_truth": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    ready_offering = _copy(offering)
    ready_offering["status"] = "evaluation_ready"
    ready_offering["base_configured_scene_offering_digest"] = offering[
        "offering_digest"
    ]
    ready_offering["evaluation_admission"] = {
        "zero_action_required": True,
        "scripted_positive_required": True,
        "learned_policy_evaluation_admitted": True,
    }
    ready_offering["evaluation_readiness"] = {
        key: receipt[key]
        for key in (
            "receipt_digest",
            "configured_scene_revision_digest",
            "packet_receipt_digest",
            "scene_plan_digest",
            "construction_result_digest",
            "control_result_digest",
            "control_pair_digest",
            "canonical_cell_id",
        )
    }
    ready_offering["offering_digest"] = ""
    ready_offering["offering_digest"] = canonical_digest(
        ready_offering, digest_field="offering_digest"
    )
    return {
        "schema_version": (
            "task_evaluation_configured_scene_evaluation_promotion.v1"
        ),
        "status": "evaluation_ready",
        "readiness_receipt": receipt,
        "configured_scene_offering": ready_offering,
    }


__all__ = [
    "CONFIGURATION_COMPLETE_OFFERING_STATUSES",
    "RECEIPT_SCHEMA_VERSION",
    "TaskEvaluationSceneEvaluationReadinessError",
    "promote_configured_scene_evaluation_readiness",
]
