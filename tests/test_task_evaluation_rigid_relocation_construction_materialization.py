from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_rigid_relocation_construction_materialization import (
    TaskEvaluationRigidRelocationConstructionMaterializationError,
    materialize_rigid_relocation_construction_authority,
)
from tests.test_task_evaluation_configured_scene_revision import revision


STATIC = "scene.configured_revision.replacement.static_qualification"
NATIVE = "scene.configured_revision.replacement.native_import_qualification"
SUPPORT = "scene.configured_revision.registration.support_plane"
MOUNT = "scene.configured_revision.registration.robot_mount_interface"
WORKSPACE = "scene.configured_revision.registration.workspace_clearance"
DEFINITION = "scene.configured_revision.task_template.definition"
SUCCESS = "scene.configured_revision.task_template.success_criteria"
EXECUTION = "scene.configured_revision.task_template.execution"


def _write(tmp_path: Path, name: str, value: dict) -> tuple[dict, dict]:
    payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    path = tmp_path / name
    path.write_bytes(payload)
    reference = {
        "uri": f"s3://blueprint-production-inputs/scene-839873/{name}",
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return reference, {
        "materialized_path": str(path),
        "full_byte_service_account_readback_passed": True,
        **reference,
    }


def _case(tmp_path: Path) -> tuple[dict, dict[str, dict]]:
    configured = revision()
    identity = configured["replacement"]["identity"]
    asset_digest = configured["replacement"]["asset"]["digest"]
    definition = {
        "schema_version": "task_evaluation_rigid_relocation_template.v1",
        "status": "preregistered_candidate_pending_configured_scene_revision",
        "task_identity": configured["task_template"]["identity"],
        "object_identity": identity,
        "strategy": "planar_push",
        "start_center_xyz_m": [2.9742285, -6.7605156, 0.818319],
        "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319],
    }
    success = {
        "schema_version": "task_evaluation_rigid_relocation_success_criteria.v1",
        "status": "preregistered_before_any_episode",
        "minimum_planar_displacement_m": 0.1,
        "maximum_final_planar_target_error_m": 0.05,
        "target_center_xyz_m": definition["target_center_xyz_m"],
    }
    execution = {
        "schema_version": "task_evaluation_rigid_relocation_execution_spec.v1",
        "status": "preregistered_before_any_episode",
        "strategy": "planar_push",
        "start_center_xyz_m": definition["start_center_xyz_m"],
        "target_center_xyz_m": definition["target_center_xyz_m"],
        "control_frequency_hz": 20,
        "action_bounds_m_per_step": {"minimum": -0.02, "maximum": 0.02},
    }
    static = {
        "schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
        "status": "authored_structure_statically_qualified",
        "replacement_identity": identity,
        "replacement_usd": {"sha256": asset_digest, "size_bytes": 12345},
        "observed_structure": {
            "rigid_body_paths": ["/Asset"],
            "collision_prim_paths": ["/Asset/Geometry/Collision"],
            "collision_bounds_asset_root_m": {
                "minimum": [-0.06, -0.04, -0.06],
                "maximum": [0.06, 0.04, 0.06],
            },
            "collision_dimensions_m": [0.12, 0.08, 0.12],
        },
        "authored_structure_statically_qualified": True,
        "structural_findings": [],
        "result_digest": "",
    }
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    static_reference, static_row = _write(tmp_path, "static.json", static)
    stable_state = {
        "position_m": [0.0, 0.0, 0.062],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    repeats = [
        {
            "final_state": stable_state,
            "final_state_digest": canonical_digest(stable_state),
        }
        for _ in range(3)
    ]
    native = {
        "schema_version": "task_evaluation_replacement_native_import_result.v1",
        "status": "qualified",
        "replacement_identity": identity,
        "asset_digest": asset_digest,
        "static_qualification_digest": static_reference["digest"],
        "native_isaac_executed": True,
        "native_simulator_import_qualified": True,
        "support_contact_observed": True,
        "deterministic_reset_state_digest_repeat_count": 3,
        "deterministic_reset_state_digest": canonical_digest(stable_state),
        "maximum_observed_settle_translation_m": 0.004,
        "maximum_observed_settle_rotation_rad": 0.02,
        "qualification_limits": {
            "gravity_settle_seconds": 3.0,
            "maximum_settle_translation_m": 0.01,
            "maximum_settle_rotation_rad": 0.08,
            "state_digest_repeat_count": 3,
        },
        "repeats": repeats,
        "blockers": [],
        "result_digest": "",
    }
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    support = {
        "schema_version": "task_evaluation_support_plane_input.v1",
        "status": "frozen_candidate_pending_production_validation",
        "scene_id": "839873",
        "sage_prim_path": "/Root/ConferenceTable",
        "top_z_m": 0.758319,
        "bounds_min_xyz_m": [2.56029, -9.56589, 0.0015],
        "bounds_max_xyz_m": [4.57771, -1.48498, 0.758319],
        "required_validation": [
            "planarity",
            "finite_bounds",
            "support_contact",
            "target_region_inside_bounds",
        ],
    }
    mount = {
        "schema_version": "task_evaluation_robot_mount_interface_plan.v1",
        "status": "publish_during_scene_configuration_run",
        "scene_id": "839873",
        "minimum_non_target_clearance_m": 0.03,
        "workspace_clearance_envelope_required": True,
        "configuration_run_must_not_claim_any_robot_qualified": True,
    }
    workspace = {
        "schema_version": "registered_sage_franka_placement_packet.v1",
        "status": "blocked",
        "request": {"candidate_may_self_authorize": False},
        "native_contact_reachability_qualified": False,
        "policy_execution_authorized": False,
        "blockers": ["franka_native_reset_contact_reachability_missing"],
    }
    docs = {
        NATIVE: ("native.json", native),
        SUPPORT: ("support.json", support),
        MOUNT: ("mount.json", mount),
        WORKSPACE: ("workspace.json", workspace),
        DEFINITION: ("definition.json", definition),
        SUCCESS: ("success.json", success),
        EXECUTION: ("execution.json", execution),
    }
    references = {STATIC: {"contract_path": STATIC, **static_row}}
    configured["replacement"]["static_qualification"] = static_reference
    revision_slots = {
        NATIVE: ("replacement", "native_import_qualification"),
        SUPPORT: ("registration", "support_plane"),
        MOUNT: ("registration", "robot_mount_interface"),
        WORKSPACE: ("registration", "workspace_clearance"),
        DEFINITION: ("task_template", "definition"),
        SUCCESS: ("task_template", "success_criteria"),
        EXECUTION: ("task_template", "execution"),
    }
    for contract_path, (name, document) in docs.items():
        reference, row = _write(tmp_path, name, document)
        references[contract_path] = {"contract_path": contract_path, **row}
        outer, field = revision_slots[contract_path]
        configured[outer][field] = reference
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    return configured, references


def test_scene839873_materializes_exact_geometry_derived_planar_push(
    tmp_path: Path,
) -> None:
    configured, references = _case(tmp_path)
    result = materialize_rigid_relocation_construction_authority(
        configured_revision=configured,
        materialized_references=references,
    )

    fields = result["task_spec_fields"]
    affordance = fields["interaction_affordance"]
    assert result["status"] == "materialized_pending_native_construction"
    assert result["configured_scene_revision_digest"] == configured["revision_digest"]
    assert affordance["contact_point_scoring_frame_m"] == [-0.06, 0.0, 0.0]
    assert affordance["approach_unit_scoring_frame"] == [-1.0, 0.0, 0.0]
    assert affordance["allowed_contact_prim_paths"] == ["/Asset"]
    assert affordance["intended_support_prim_paths"] == ["/Root/ConferenceTable"]
    assert affordance["affordance_digest"] == canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    assert fields["support_height_interval_m"] == [0.808319, 0.828319]
    assert fields["settle_window_samples"] == 60
    assert fields["task_contact_minimum_force_n"] == 0.5
    assert fields["collision_failure_minimum_force_n"] == 1.0
    assert result["task_object_pose_world"] == {
        "position_world_m": [2.9742285, -6.7605156, 0.818319],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    assert result["scenario_parameters"]["object_height_m"] == 0.12
    assert result["scenario_parameters"]["object_radius_m"] == pytest.approx(
        math.hypot(0.12, 0.08) / 2.0
    )
    assert len(result["source_bindings"]) == 8
    assert result["claim_boundary"]["native_construction_qualified"] is False
    assert result["claim_boundary"]["learned_policy_outcomes_consulted"] is False
    assert result["materialization_digest"] == canonical_digest(
        result, digest_field="materialization_digest"
    )


def test_refuses_pre_fix_static_receipt_without_qualified_collision_bounds(
    tmp_path: Path,
) -> None:
    configured, references = _case(tmp_path)
    static = json.loads(Path(references[STATIC]["materialized_path"]).read_text())
    static["observed_structure"].pop("collision_bounds_asset_root_m")
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    reference, row = _write(tmp_path, "static-without-bounds.json", static)
    configured["replacement"]["static_qualification"] = reference
    references[STATIC] = {"contract_path": STATIC, **row}
    native = json.loads(Path(references[NATIVE]["materialized_path"]).read_text())
    native["static_qualification_digest"] = reference["digest"]
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    native_reference, native_row = _write(tmp_path, "native-rebound.json", native)
    configured["replacement"]["native_import_qualification"] = native_reference
    references[NATIVE] = {"contract_path": NATIVE, **native_row}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")

    with pytest.raises(
        TaskEvaluationRigidRelocationConstructionMaterializationError,
        match="rigid_construction_static_geometry_missing",
    ):
        materialize_rigid_relocation_construction_authority(
            configured_revision=configured,
            materialized_references=references,
        )

def test_refuses_unbound_workspace_bytes(tmp_path: Path) -> None:
    configured, references = _case(tmp_path)
    references[WORKSPACE]["digest"] = "sha256:" + "f" * 64
    with pytest.raises(
        TaskEvaluationRigidRelocationConstructionMaterializationError,
        match=f"rigid_construction_source_invalid:{WORKSPACE}",
    ):
        materialize_rigid_relocation_construction_authority(
            configured_revision=configured,
            materialized_references=references,
        )


def test_refuses_task_outside_conservatively_inset_support(tmp_path: Path) -> None:
    configured, references = _case(tmp_path)
    definition = json.loads(Path(references[DEFINITION]["materialized_path"]).read_text())
    definition["target_center_xyz_m"][0] = 4.55
    reference, row = _write(tmp_path, "definition-outside.json", definition)
    configured["task_template"]["definition"] = reference
    references[DEFINITION] = {"contract_path": DEFINITION, **row}
    for contract_path in (SUCCESS, EXECUTION):
        document = json.loads(Path(references[contract_path]["materialized_path"]).read_text())
        document["target_center_xyz_m"][0] = 4.55
        ref, rebound = _write(
            tmp_path, f"{contract_path.rsplit('.', 1)[-1]}-outside.json", document
        )
        configured["task_template"][
            "success_criteria" if contract_path == SUCCESS else "execution"
        ] = ref
        references[contract_path] = {"contract_path": contract_path, **rebound}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")

    with pytest.raises(
        TaskEvaluationRigidRelocationConstructionMaterializationError,
        match="rigid_construction_task_outside_registered_support",
    ):
        materialize_rigid_relocation_construction_authority(
            configured_revision=configured,
            materialized_references=references,
        )


def test_refuses_native_receipt_without_preregistered_stability_limits(
    tmp_path: Path,
) -> None:
    configured, references = _case(tmp_path)
    native = json.loads(Path(references[NATIVE]["materialized_path"]).read_text())
    native.pop("qualification_limits")
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    reference, row = _write(tmp_path, "native-without-limits.json", native)
    configured["replacement"]["native_import_qualification"] = reference
    references[NATIVE] = {"contract_path": NATIVE, **row}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")

    with pytest.raises(
        TaskEvaluationRigidRelocationConstructionMaterializationError,
        match="rigid_construction_native_stability_authority_missing",
    ):
        materialize_rigid_relocation_construction_authority(
            configured_revision=configured,
            materialized_references=references,
        )
