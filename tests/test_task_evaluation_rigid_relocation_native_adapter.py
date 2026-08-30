from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    DEFINITION_CONTRACT_PATH,
    DIAGNOSTIC_SCHEMA_VERSION,
    DIAGNOSTIC_SOURCE_ALIASES,
    TaskEvaluationRigidRelocationNativeAdapterError,
    _source_document,
    adapt_rigid_relocation_task_template,
)
from blueprint_pipeline.native_task_construction_plan import (
    materialize_rigid_construction_phase_plan,
)
from blueprint_pipeline.native_franka_action_math import (
    grasp_orientation_contact_xyzw,
)
from tests.test_task_evaluation_configured_scene_revision import revision
from tests.test_task_evaluation_launch_preparation_contract import request


DEFINITION = "scene.configured_revision.task_template.definition"
SUCCESS = "scene.configured_revision.task_template.success_criteria"
EXECUTION = "scene.configured_revision.task_template.execution"
SUPPORT = "scene.configured_revision.registration.support_plane"
SOURCE_OBJECT = "scene.configured_revision.replacement.source_object"
STATIC = "scene.configured_revision.replacement.static_qualification"
NATIVE_IMPORT = (
    "scene.configured_revision.replacement.native_import_qualification"
)


def _documents(task_identity: dict, object_identity: dict) -> dict[str, dict]:
    success = {
        "authority": "deterministic_simulator_state",
        "forbidden_collision_allowed": False,
        "joint_limit_violation_allowed": False,
        "maximum_final_planar_target_error_m": 0.05,
        "minimum_planar_displacement_m": 0.1,
        "object_must_remain_on_registered_support": True,
    }
    static = {
        "schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
        "status": "authored_structure_statically_qualified",
        "replacement_identity": object_identity,
        "observed_structure": {
            "center_of_mass_m": [0.0, 0.0, 0.063819],
            "rigid_body_paths": ["/Asset"],
        },
        "result_digest": "",
    }
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    native = {
        "schema_version": "task_evaluation_replacement_native_import_result.v1",
        "status": "qualified",
        "replacement_identity": object_identity,
        "native_simulator_import_qualified": True,
        "blockers": [],
        "result_digest": "",
    }
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
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
        SUPPORT: {
            "schema_version": "task_evaluation_support_plane_input.v1",
            "status": "frozen_candidate_pending_production_validation",
            "scene_id": "839873",
            "sage_prim_path": "/Root/Support",
            "bounds_min_xyz_m": [2.5, -9.5, 0.0],
            "bounds_max_xyz_m": [4.5, -1.4, 0.7545],
            "top_z_m": 0.7545,
        },
        SOURCE_OBJECT: {
            "schema_version": "task_evaluation_source_object_selection.v1",
            "status": "frozen_before_scene_configuration_run",
            "scene_id": "839873",
            "center_xyz_m": [2.9742285, -6.7605156, 0.818319],
            "aabb_min_xyz_m": [2.9103536, -6.8264092, 0.7545],
            "aabb_max_xyz_m": [3.0381034, -6.6946220, 0.882138],
        },
        STATIC: static,
        NATIVE_IMPORT: native,
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
        target = {
            DEFINITION: ("task_template", "definition"),
            SUCCESS: ("task_template", "success_criteria"),
            EXECUTION: ("task_template", "execution"),
            SUPPORT: ("registration", "support_plane"),
            SOURCE_OBJECT: ("replacement", "source_object"),
            STATIC: ("replacement", "static_qualification"),
            NATIVE_IMPORT: ("replacement", "native_import_qualification"),
        }[contract_path]
        configured[target[0]][target[1]] = reference
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
    target = {
        DEFINITION: ("task_template", "definition"),
        SUCCESS: ("task_template", "success_criteria"),
        EXECUTION: ("task_template", "execution"),
        SUPPORT: ("registration", "support_plane"),
        SOURCE_OBJECT: ("replacement", "source_object"),
        STATIC: ("replacement", "static_qualification"),
        NATIVE_IMPORT: ("replacement", "native_import_qualification"),
    }[contract_path]
    configured[target[0]][target[1]] = reference
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
    affordance = task_spec["interaction_affordance"]
    assert affordance["gripper_orientation_scoring_frame_xyzw"] == pytest.approx(
        grasp_orientation_contact_xyzw(
            approach_axis=[1.0, 0.0, 0.0],
            jaw_axis=[0.0, 1.0, 0.0],
        )
    )
    assert affordance["gripper_orientation_scoring_frame_xyzw"] != [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result["source_documents"]["documents"]["definition"] == docs[
        DEFINITION
    ]
    assert {
        row["digest"] for row in result["source_documents"]["bindings"]
    } == {references[path]["digest"] for path in docs}
    assert result["adapter_digest"] == canonical_digest(
        result, digest_field="adapter_digest"
    )
    phase_plan = materialize_rigid_construction_phase_plan(
        {
            "task_kind": "rigid_pick_place",
            "cadence": {
                "maximum_action_steps": task_spec["maximum_action_steps"]
            },
            "task_spec": task_spec,
            "objects": [
                {
                    "task_subject": True,
                    "asset_id": configured["replacement"]["identity"]["id"],
                    "object_type": "RIGID",
                    "pose_world": result["native_task_definition"][
                        "task_object_pose_world"
                    ],
                    "reset_state": {
                        "root_pose_world": result["native_task_definition"][
                            "task_object_pose_world"
                        ],
                        "joint_positions": {},
                    },
                }
            ],
        }
    )
    assert phase_plan["manipulation_strategy"] == "planar_push"
    assert phase_plan["phases"][0]["phase_id"] == "precontact"
    assert phase_plan["phases"][1]["phase_id"] == "push_contact"


def test_rigid_root_cannot_begin_below_registered_support(tmp_path: Path) -> None:
    launch, configured, references, docs = _case(tmp_path)
    source = copy.deepcopy(docs[SOURCE_OBJECT])
    source["aabb_min_xyz_m"][2] = 0.7539999997558593
    _rewrite(
        tmp_path=tmp_path,
        configured=configured,
        references=references,
        contract_path=SOURCE_OBJECT,
        document=source,
    )
    static = copy.deepcopy(docs[STATIC])
    static["observed_structure"]["center_of_mass_m"][2] = 0.06400000303983688
    static["result_digest"] = canonical_digest(
        static, digest_field="result_digest"
    )
    _rewrite(
        tmp_path=tmp_path,
        configured=configured,
        references=references,
        contract_path=STATIC,
        document=static,
    )
    launch["task"]["configured_scene_revision_digest"] = configured[
        "revision_digest"
    ]

    result = adapt_rigid_relocation_task_template(
        request=launch,
        configured_revision=configured,
        materialized_references=references,
    )

    pose = result["native_task_definition"]["task_object_pose_world"]
    expected_root_z = 0.7545 - (
        0.06400000303983688 + 0.7539999997558593 - 0.818319
    )
    assert pose["position_world_m"][2] == pytest.approx(expected_root_z)
    alignment = result["native_task_definition"]["task_spec"][
        "interaction_affordance"
    ]["support_alignment"]
    assert alignment["support_aligned_root_z_m"] == pytest.approx(
        expected_root_z
    )
    assert alignment["initial_support_penetration_permitted"] is False
    assert result["adapter_digest"] == canonical_digest(
        result, digest_field="adapter_digest"
    )


def test_diagnostic_authority_reuses_exact_documents_without_revision_claim(
    tmp_path: Path,
) -> None:
    _, _, references, _ = _case(tmp_path)
    rows = []
    for contract_path, alias in DIAGNOSTIC_SOURCE_ALIASES.items():
        reference = references[contract_path]
        rows.append(
            {
                "contract_path": alias,
                "uri": reference["uri"],
                "digest": reference["digest"],
                "size_bytes": reference["size_bytes"],
                "path": reference["materialized_path"],
                "full_byte_readback_passed": True,
            }
        )
    provider = {
        "schema_version": (
            "task_evaluation_scene_configuration_diagnostic_provider_result.v1"
        ),
        "status": "completed_diagnostic_only_not_qualification_eligible",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "executed_inside_one_parent_provider_run": False,
        "configured_revision_publication_permitted": False,
        "result_digest": "",
    }
    provider["result_digest"] = canonical_digest(
        provider, digest_field="result_digest"
    )
    payload = (json.dumps(provider, sort_keys=True) + "\n").encode()
    provider_path = tmp_path / "provider-result.json"
    provider_path.write_bytes(payload)
    rows.append(
        {
            "contract_path": (
                "diagnostic_output."
                "task_evaluation_scene_configuration_provider_result.v1.json"
            ),
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "path": str(provider_path),
            "full_byte_readback_passed": True,
        }
    )
    authority = {
        "schema_version": (
            "task_evaluation_configured_scene_diagnostic_controls_input.v1"
        ),
        "status": "materialized",
        "claim_ceiling": (
            "development_only_downstream_construction_and_controls_diagnostic"
        ),
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "evaluation_ready_promotion_permitted": False,
        "materialized_inputs": rows,
        "receipt_digest": "",
    }
    authority["receipt_digest"] = canonical_digest(
        authority, digest_field="receipt_digest"
    )

    result = adapt_rigid_relocation_task_template(
        materialized_references={},
        diagnostic_controls_input=authority,
    )

    assert result["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert result["qualification_eligible"] is False
    assert "configured_scene_revision_digest" not in result
    assert result["diagnostic_controls_input_receipt_digest"] == authority[
        "receipt_digest"
    ]
    assert result["native_task_definition"]["task_spec"][
        "manipulation_strategy"
    ] == "planar_push"


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


def test_source_document_refuses_symlink_even_when_target_bytes_match(
    tmp_path: Path,
) -> None:
    document = _documents(
        {"id": "task", "version": "v1"},
        {"id": "object", "version": "v1"},
    )[DEFINITION_CONTRACT_PATH]
    payload = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    target = tmp_path / "definition.json"
    target.write_bytes(payload)
    link = tmp_path / "definition-link.json"
    link.symlink_to(target)
    expected = {
        "uri": "s3://blueprint-production-inputs/definition.json",
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    references = {
        DEFINITION_CONTRACT_PATH: {
            "contract_path": DEFINITION_CONTRACT_PATH,
            **expected,
            "materialized_path": str(link),
            "full_byte_service_account_readback_passed": True,
        }
    }

    with pytest.raises(
        TaskEvaluationRigidRelocationNativeAdapterError,
        match="rigid_relocation_native_adapter_source_invalid",
    ):
        _source_document(
            references,
            contract_path=DEFINITION_CONTRACT_PATH,
            expected_reference=expected,
        )
