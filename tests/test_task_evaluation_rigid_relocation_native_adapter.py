from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    TaskEvaluationRigidRelocationNativeAdapterError,
    adapt_rigid_relocation_task_template,
)
from tests.test_task_evaluation_configured_scene_revision import revision
from tests.test_task_evaluation_launch_preparation_contract import request


DEFINITION = "scene.configured_revision.task_template.definition"
SUCCESS = "scene.configured_revision.task_template.success_criteria"
EXECUTION = "scene.configured_revision.task_template.execution"


def _documents(task_identity: dict, object_identity: dict) -> dict[str, dict]:
    success = {
        "authority": "deterministic_simulator_state",
        "forbidden_collision_allowed": False,
        "joint_limit_violation_allowed": False,
        "maximum_final_planar_target_error_m": 0.05,
        "minimum_planar_displacement_m": 0.1,
        "object_must_remain_on_registered_support": True,
    }
    return {
        DEFINITION: {
            "schema_version": "task_evaluation_rigid_relocation_template.v1",
            "status": "preregistered_candidate_pending_configured_scene_revision",
            "task_identity": task_identity,
            "object_identity": object_identity,
            "strategy": "planar_push",
            "start_center_xyz_m": [2.9742285, -6.7605156, 0.818319],
            "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319],
            "control_frequency_hz": 20,
            "maximum_episode_seconds": 12.0,
            "maximum_step_count": 240,
            "resolved_seed": 839873104,
            "controls_order": ["zero_action", "deterministic_scripted"],
            "failure_metrics": [
                "insufficient_displacement",
                "final_target_error_exceeded",
                "object_left_support",
                "forbidden_collision",
                "joint_limit_violation",
                "timeout",
            ],
            "preregistration_rule": (
                "Any scientific task change creates a new immutable template."
            ),
            "success": success,
        },
        SUCCESS: {
            "schema_version": (
                "task_evaluation_rigid_relocation_success_criteria.v1"
            ),
            "status": "preregistered_before_any_episode",
            **success,
            "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319],
        },
        EXECUTION: {
            "schema_version": (
                "task_evaluation_rigid_relocation_execution_spec.v1"
            ),
            "status": "preregistered_before_any_episode",
            "strategy": "planar_push",
            "start_center_xyz_m": [2.9742285, -6.7605156, 0.818319],
            "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319],
            "control_frequency_hz": 20,
            "maximum_episode_seconds": 12.0,
            "maximum_step_count": 240,
            "resolved_seed": 839873104,
            "action_bounds_m_per_step": {"minimum": -0.02, "maximum": 0.02},
            "collision_exclusions": [
                "robot_self_collision_pairs_declared_by_robot_configuration"
            ],
            "termination": [
                "success",
                "object_left_support",
                "forbidden_collision",
                "joint_limit_violation",
                "timeout",
            ],
        },
    }


def _case(tmp_path: Path) -> tuple[dict, dict, dict, dict[str, dict]]:
    configured = revision()
    launch = request()
    launch["team_namespace"] = configured["team_namespace"]
    launch["expected_production_commit"] = configured["source_commit"]
    launch["scene"]["identity"] = configured["scene_identity"]
    launch["task"]["identity"] = configured["task_template"]["identity"]
    launch["task"]["subject"]["identity"] = configured["replacement"]["identity"]
    docs = _documents(
        configured["task_template"]["identity"],
        configured["replacement"]["identity"],
    )
    references: dict[str, dict] = {}
    for contract_path, document in docs.items():
        payload = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
        path = tmp_path / f"{len(references)}.json"
        path.write_bytes(payload)
        reference = {
            "uri": f"s3://blueprint-production-inputs/{path.name}",
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        references[contract_path] = {
            "contract_path": contract_path,
            **reference,
            "materialized_path": str(path),
            "full_byte_service_account_readback_passed": True,
        }
        configured["task_template"][
            {
                DEFINITION: "definition",
                SUCCESS: "success_criteria",
                EXECUTION: "execution",
            }[contract_path]
        ] = reference
    configured["revision_digest"] = canonical_digest(
        configured, digest_field="revision_digest"
    )
    launch["task"]["configured_scene_revision_digest"] = configured[
        "revision_digest"
    ]
    return launch, configured, references, docs


def _rewrite(
    *,
    tmp_path: Path,
    configured: dict,
    references: dict[str, dict],
    contract_path: str,
    document: dict,
) -> None:
    payload = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    path = tmp_path / f"mutated-{len(list(tmp_path.iterdir()))}.json"
    path.write_bytes(payload)
    reference = {
        "uri": f"s3://blueprint-production-inputs/{path.name}",
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    references[contract_path] = {
        "contract_path": contract_path,
        **reference,
        "materialized_path": str(path),
        "full_byte_service_account_readback_passed": True,
    }
    configured["task_template"][
        {
            DEFINITION: "definition",
            SUCCESS: "success_criteria",
            EXECUTION: "execution",
        }[contract_path]
    ] = reference
    configured["revision_digest"] = canonical_digest(
        configured, digest_field="revision_digest"
    )


def test_scene839873_task_truth_is_preserved_in_native_packet_inputs(
    tmp_path: Path,
) -> None:
    launch, configured, references, docs = _case(tmp_path)
    result = adapt_rigid_relocation_task_template(
        request=launch,
        configured_revision=configured,
        materialized_references=references,
    )

    task_spec = result["native_task_definition"]["task_spec"]
    execution = result["native_episode_execution"]
    assert result["native_task_kind"] == "rigid_pick_place"
    assert result["manipulation_strategy"] == "planar_push"
    assert task_spec["manipulation_strategy"] == "planar_push"
    assert task_spec["start_pose_world"][:3] == docs[DEFINITION][
        "start_center_xyz_m"
    ]
    assert task_spec["target_position_world_m"] == docs[DEFINITION][
        "target_center_xyz_m"
    ]
    assert task_spec["configured_success_criteria"] == {
        key: value
        for key, value in docs[SUCCESS].items()
        if key not in {"schema_version", "status"}
    }
    assert execution["physics_frequency_hz"] == 120
    assert execution["control_frequency_hz"] == 20.0
    assert execution["control_decimation"] == 6
    assert execution["maximum_step_count"] == 240
    assert execution["maximum_episode_seconds"] == 12.0
    assert execution["scenario"]["seed"] == 839873104
    assert execution["scenario"]["cell_id"] == (
        "configured_scene_canonical.seed_839873104"
    )
    assert result["source_documents"]["documents"]["definition"] == docs[
        DEFINITION
    ]
    assert {
        row["digest"] for row in result["source_documents"]["bindings"]
    } == {references[path]["digest"] for path in (DEFINITION, SUCCESS, EXECUTION)}
    assert result["adapter_digest"] == canonical_digest(
        result, digest_field="adapter_digest"
    )


@pytest.mark.parametrize(
    ("contract_path", "mutation", "blocker"),
    [
        (
            EXECUTION,
            lambda value: value.update(strategy="pick_and_place"),
            "identity_or_strategy_mismatch",
        ),
        (
            EXECUTION,
            lambda value: value.update(target_center_xyz_m=[3.2, -6.7, 0.818319]),
            "task_pose_mismatch",
        ),
        (
            SUCCESS,
            lambda value: value.update(minimum_planar_displacement_m=0.11),
            "success_bounds_mismatch",
        ),
        (
            EXECUTION,
            lambda value: value.update(maximum_step_count=239),
            "execution_timing_mismatch",
        ),
    ],
)
def test_refuses_cross_document_drift(
    tmp_path: Path, contract_path: str, mutation, blocker: str
) -> None:
    launch, configured, references, docs = _case(tmp_path)
    changed = copy.deepcopy(docs[contract_path])
    mutation(changed)
    _rewrite(
        tmp_path=tmp_path,
        configured=configured,
        references=references,
        contract_path=contract_path,
        document=changed,
    )
    launch["task"]["configured_scene_revision_digest"] = configured[
        "revision_digest"
    ]

    with pytest.raises(
        TaskEvaluationRigidRelocationNativeAdapterError,
        match=f"rigid_relocation_native_adapter_{blocker}",
    ):
        adapt_rigid_relocation_task_template(
            request=launch,
            configured_revision=configured,
            materialized_references=references,
        )


def test_refuses_reference_that_is_not_the_revision_bound_object(
    tmp_path: Path,
) -> None:
    launch, configured, references, _docs = _case(tmp_path)
    references[DEFINITION]["uri"] = "s3://blueprint-production-inputs/wrong.json"
    with pytest.raises(
        TaskEvaluationRigidRelocationNativeAdapterError,
        match="rigid_relocation_native_adapter_source_invalid",
    ):
        adapt_rigid_relocation_task_template(
            request=launch,
            configured_revision=configured,
            materialized_references=references,
        )
