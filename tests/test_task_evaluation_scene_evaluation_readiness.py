from __future__ import annotations

import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_evaluation_readiness import (
    TaskEvaluationSceneEvaluationReadinessError,
    promote_configured_scene_evaluation_readiness,
)
from tests.test_native_task_arena_bundle import (
    _articulated_packet,
    _qualified_construction,
    _qualified_controls,
)
from tests.test_task_evaluation_configured_scene_revision import revision


def _inputs(tmp_path):
    configured_revision = revision()
    offering = {
        "schema_version": "task_evaluation_configured_scene_offering.v1",
        "status": "configured_controls_pending",
        "configuration_run_id": configured_revision["configuration_run_id"],
        "team_namespace": configured_revision["team_namespace"],
        "catalog_visibility": "team_only",
        "scene_identity": configured_revision["scene_identity"],
        "task": {"identity": configured_revision["task_template"]["identity"]},
        "presentation": configured_revision["presentation"],
        "evaluation_preparation_binding": {
            "configured_scene_revision_digest": configured_revision[
                "revision_digest"
            ],
            "configured_scene_bundle": configured_revision[
                "configured_scene_bundle"
            ],
        },
        "evaluation_admission": {
            "zero_action_required": True,
            "scripted_positive_required": True,
            "learned_policy_evaluation_admitted": False,
        },
        "proof_boundary": {
            "configuration_is_policy_evaluation": False,
        },
        "offering_digest": "",
    }
    offering["offering_digest"] = canonical_digest(
        offering, digest_field="offering_digest"
    )

    _packet, scene_plan = _articulated_packet(tmp_path)
    construction_path = _qualified_construction(tmp_path, scene_plan)
    control_path = _qualified_controls(tmp_path, scene_plan, construction_path)
    construction = json.loads(construction_path.read_text())
    controls = json.loads(control_path.read_text())
    adapter = {
        "schema_version": "task_evaluation_native_arena_adapter_result.v1",
        "status": "native_arena_adapter_materialized",
        "configured_scene_revision_digest": configured_revision[
            "revision_digest"
        ],
        "packet_receipt_digest": construction["packet_receipt_digest"],
        "provider_mutation_performed": False,
        "catalog_mutation_performed": False,
        "paid_execution_requested": False,
        "result_digest": "",
    }
    adapter["result_digest"] = canonical_digest(
        adapter, digest_field="result_digest"
    )
    return {
        "configured_scene_revision": configured_revision,
        "configured_scene_offering": offering,
        "adapter_result": adapter,
        "scene_plan": scene_plan,
        "construction_result": construction,
        "control_result": controls,
    }


def test_promotes_only_exact_qualified_control_pair(tmp_path) -> None:
    inputs = _inputs(tmp_path)

    result = promote_configured_scene_evaluation_readiness(**inputs)

    receipt = result["readiness_receipt"]
    ready = result["configured_scene_offering"]
    assert result["status"] == "evaluation_ready"
    assert receipt["status"] == "evaluation_ready"
    assert receipt["candidate_policy_queried"] is False
    assert receipt["variation_matrix_started"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert [row["control_id"] for row in receipt["controls"]] == [
        "zero_action_negative",
        "deterministic_scripted_positive",
    ]
    assert all(row["control_passed"] for row in receipt["controls"])
    assert ready["status"] == "evaluation_ready"
    assert ready["base_configured_scene_offering_digest"] == inputs[
        "configured_scene_offering"
    ]["offering_digest"]
    assert ready["evaluation_admission"][
        "learned_policy_evaluation_admitted"
    ] is True
    assert ready["offering_digest"] == canonical_digest(
        ready, digest_field="offering_digest"
    )


def test_refuses_canonical_pair_when_scripted_positive_failed(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    controls = copy.deepcopy(inputs["control_result"])
    pair = controls["control_pair"]
    pair["controls"][1]["control_passed"] = False
    pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
    controls["result_digest"] = canonical_digest(
        controls, digest_field="result_digest"
    )
    inputs["control_result"] = controls

    with pytest.raises(
        TaskEvaluationSceneEvaluationReadinessError,
        match="configured_scene_evaluation_readiness_controls_invalid",
    ):
        promote_configured_scene_evaluation_readiness(**inputs)


def test_refuses_controls_from_another_packet(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    controls = copy.deepcopy(inputs["control_result"])
    controls["packet_receipt_digest"] = "sha256:" + "9" * 64
    controls["result_digest"] = canonical_digest(
        controls, digest_field="result_digest"
    )
    inputs["control_result"] = controls

    with pytest.raises(
        TaskEvaluationSceneEvaluationReadinessError,
        match="configured_scene_evaluation_readiness_packet_binding_mismatch",
    ):
        promote_configured_scene_evaluation_readiness(**inputs)


def test_refuses_launch_ready_label_without_controls_pending_state(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    offering = copy.deepcopy(inputs["configured_scene_offering"])
    offering["status"] = "launch_ready"
    offering["offering_digest"] = canonical_digest(
        offering, digest_field="offering_digest"
    )
    inputs["configured_scene_offering"] = offering

    with pytest.raises(
        TaskEvaluationSceneEvaluationReadinessError,
        match="configured_scene_evaluation_readiness_offering_invalid",
    ):
        promote_configured_scene_evaluation_readiness(**inputs)
