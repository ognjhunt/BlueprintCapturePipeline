"""Capability-gated tool registry for the Task Evaluation Supervisor.

Phase 0/1 descriptors expose proposal and inspection surfaces only.  No tool
in this registry owns proof transitions, paid allocation, or physical actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from ..common import write_json
from ..arkit_reconstruction_dataset import (
    build_arkit_reconstruction_dataset_export_request,
)
from ..decision_evidence_contracts import canonical_digest
from ..decision_evidence_router import route_decision_evidence
from ..evaluation_run_contract import validate_evaluation_run_spec
from ..external_reconstruction_import import (
    build_external_reconstruction_import_receipt,
    build_external_reconstruction_import_request,
)
from ..reconstruction_geometry_contracts import (
    build_collider_candidate_manifest,
    build_collider_qualification_report,
    build_isaac_asset_verification_result,
    build_metric_geometry_manifest,
    build_nurec_openusd_packaging_request,
    build_nurec_openusd_packaging_result,
)
from ..reconstruction_heldout_evaluation import (
    build_heldout_appearance_evaluation_request,
    build_visual_heldout_evaluation_report,
)
from ..reconstruction_failure_diagnosis import (
    build_reconstruction_failure_diagnosis_request,
    diagnose_reconstruction_failure,
)
from ..reconstruction_capability import normalize_reconstruction_result
from ..reconstruction_worker_contracts import (
    build_pose_estimation_request,
    build_pose_estimation_result,
    build_training_request,
    build_training_result,
)
from .contracts import (
    TOOL_OBSERVATION_SCHEMA_VERSION,
    ActionProposal,
    AuthorityEnvelope,
    AutonomyMode,
    ToolDescriptor,
    ToolObservation,
)
from .capture_reconstruction_routing import (
    build_capture_reconstruction_route,
    validate_capture_reconstruction_route,
)
from .phase2_artifacts import (
    authorization_request,
    clarification_request,
    scenario_proposal_set,
    targeted_recapture_request,
    write_phase2_artifact,
)


TOOL_REGISTRY_SCHEMA_VERSION = "task_evaluation_supervisor_tool_registry.v1"


def _output_schema(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "required": list(properties),
        "properties": {key: dict(schema) for key, schema in properties.items()},
        "additionalProperties": additional_properties,
    }


_TOOL_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "inspect_site_task_testbed": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "evidence_inventory_count": {"type": "integer"},
            "governance": {"type": "object"},
        }
    ),
    "plan_capture_reconstruction_route": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "route": {"type": "object"},
            "route_digest": {"type": "string"},
            "capture_authority_profile": {},
            "execution_authorized_by_route": {"const": False},
            "proof_state_changed": {"const": False},
        }
    ),
    "compile_frozen_frame_dataset": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "dataset_manifest_digest": {"type": "string"},
            "split_digest": {"type": "string"},
            "hidden_heldout_isolated": {"const": True},
            "candidate_can_change_split": {"const": False},
            "proof_state_changed": {"const": False},
        }
    ),
    "compile_arkit_metric_scaffold": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "reconstruction_result_digest": {"type": "string"},
            "metric_scaffold_digest": {"type": "string"},
            "arkit_export_digest": {"type": "string"},
            "decoded_pts_verified": {"const": True},
            "raw_arkit_poses_modified": {"const": False},
            "metric_scale_independently_validated": {"const": False},
            "claim_ceiling": {"const": "sensor_declared_metric_scaffold"},
            "proof_state_changed": {"const": False},
        }
    ),
    "export_arkit_reconstruction_dataset": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "arkit_export_digest": {"type": "string"},
            "camera_calibration_digest": {"type": "string"},
            "camera_observation_digest": {"type": "string"},
            "pose_refinement_request_digest": {"type": "string"},
            "hidden_heldout_pixels_included": {"const": False},
            "raw_arkit_poses_modified": {"const": False},
            "claim_ceiling": {"const": "calibrated_camera_trajectory"},
            "proof_state_changed": {"const": False},
        }
    ),
    "normalize_native_360_capture": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "normalization_digest": {"type": "string"},
            "rig_declaration_digest": {"type": "string"},
            "dual_fisheye_binding_digest": {"type": "string"},
            "status": {"enum": ["normalized", "blocked"]},
            "claim_ceiling": {
                "enum": ["calibrated_camera_rig", "decoded_native_container"]
            },
            "agent_altered_calibration": {"const": False},
            "proof_state_changed": {"const": False},
        }
    ),
    "compile_equirectangular_virtual_rig": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "compilation_digest": {"type": "string"},
            "virtual_rig_digest": {"type": "string"},
            "virtual_observation_count": {"type": "integer"},
            "shared_optical_center_required": {"const": True},
            "virtual_views_are_captured_evidence": {"const": False},
            "proof_state_changed": {"const": False},
        }
    ),
    "run_pose_estimation": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "status": {"enum": ["succeeded", "failed", "timed_out", "interrupted"]},
            "failure_code": {},
            "pose_estimation_result_digest": {"type": "string"},
            "registered_observation_count": {"type": "integer"},
            "rejected_observation_count": {"type": "integer"},
            "heldout_labels_included": {"const": False},
            "candidate_self_graded": {"const": False},
            "claim_ceiling": {"const": "calibrated_camera_trajectory"},
            "proof_state_changed": {"const": False},
        }
    ),
    "train_gaussian_reconstruction": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "status": {"enum": ["succeeded", "failed", "timed_out", "interrupted"]},
            "failure_code": {},
            "reconstruction_training_result_digest": {"type": "string"},
            "checkpoint_count": {"type": "integer"},
            "heldout_labels_included": {"const": False},
            "candidate_self_graded": {"const": False},
            "claim_ceiling": {"const": "appearance_reconstruction"},
            "proof_state_changed": {"const": False},
        }
    ),
    "evaluate_heldout_appearance": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "artifact_digest": {"type": "string"},
            "claim_ceiling": {"const": "appearance_reconstruction"},
            "decision": {
                "enum": ["passed_appearance_only", "rejected_appearance_quality"]
            },
            "proof_state_changed": {"const": False},
        }
    ),
    "compile_metric_geometry": _output_schema(
        {
            "contract_present": {"const": True}, "digest_matches": {"const": True},
            "artifact_digest": {"type": "string"},
            "claim_ceiling": {"const": "metric_reference_geometry"},
            "decision": {}, "proof_state_changed": {"const": False},
        }
    ),
    "compile_collision_candidate": _output_schema(
        {
            "contract_present": {"const": True}, "digest_matches": {"const": True},
            "artifact_digest": {"type": "string"},
            "claim_ceiling": {"const": "collision_geometry_candidate"},
            "decision": {}, "proof_state_changed": {"const": False},
        }
    ),
    "qualify_collision_candidate": _output_schema(
        {
            "contract_present": {"const": True}, "digest_matches": {"const": True},
            "artifact_digest": {"type": "string"},
            "claim_ceiling": {"const": "bounded_navigation_simulation"},
            "decision": {"enum": ["accepted_bounded_navigation", "rejected"]},
            "proof_state_changed": {"const": False},
        }
    ),
    "package_nurec_openusd": _output_schema(
        {
            "contract_present": {"const": True}, "digest_matches": {"const": True},
            "artifact_digest": {"type": "string"},
            "claim_ceiling": {"const": "openusd_package"},
            "decision": {}, "proof_state_changed": {"const": False},
        }
    ),
    "verify_isaac_asset": _output_schema(
        {
            "contract_present": {"const": True}, "digest_matches": {"const": True},
            "artifact_digest": {"type": "string"},
            "claim_ceiling": {"const": "isaac_load_render_compatibility"},
            "decision": {"const": "verified_compatibility_only"},
            "proof_state_changed": {"const": False},
        }
    ),
    "import_external_reconstruction": _output_schema(
        {
            "contract_present": {"const": True}, "digest_matches": {"const": True},
            "artifact_digest": {"type": "string"},
            "claim_ceiling": {"const": "external_reconstruction_import"},
            "decision": {"const": "imported_derived_support_only"},
            "proof_state_changed": {"const": False},
        }
    ),
    "diagnose_reconstruction_failure": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "diagnosis_digest": {"type": "string"},
            "diagnosed_failure_code": {"type": "string"},
            "identical_attempt_count": {"type": "integer"},
            "unchanged_deterministic_retry_allowed": {"type": "boolean"},
            "terminal_for_current_configuration": {"type": "boolean"},
            "legal_next_actions": {"type": "array"},
            "failed_evidence_preserved": {"const": True},
            "proof_state_changed": {"const": False},
        }
    ),
    "invoke_authorized_reconstruction_provider": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "status": {"enum": ["succeeded_unqualified", "failed", "interrupted"]},
            "provider_execution_receipt_digest": {"type": "string"},
            "claim_ceiling": {"const": "external_reconstruction_import"},
            "proof_state_changed": {"const": False},
        }
    ),
    "validate_proposed_claim_graph": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "claim_ids": {"type": "array"},
        }
    ),
    "materialize_clarification_request": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "request_id": {"type": "string"},
            "awaiting_customer_response": {"const": True},
            "proof_state_changed": {"const": False},
        }
    ),
    "compile_deterministic_evidence_plan": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "step_count": {"type": "integer"},
            "compiled_by_agent": {"const": False},
        }
    ),
    "materialize_compiled_leaf_runs": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "plan_digest": {"type": "string"},
            "compiled_leaf_run_count": {"type": "integer"},
            "compiled_leaf_run_references": {"type": "array"},
            "provider_execution_started": {"const": False},
            "proof_state_changed": {"const": False},
        }
    ),
    "inspect_normalized_evidence_results": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "status": {},
            "failure_type": {},
            "execution_requested": {"type": "boolean"},
        }
    ),
    "propose_targeted_recapture": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "request_id": {"type": "string"},
            "targeted_recapture_request_digest": {"type": "string"},
            "capture_started": {"const": False},
            "proof_state_changed": {"const": False},
        }
    ),
    "propose_adversarial_scenarios": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "scenario_count": {"type": "integer"},
            "frozen": {"const": False},
            "hidden_labels_included": {"const": False},
            "proof_state_changed": {"const": False},
        }
    ),
    "materialize_authorization_request": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "request_id": {"type": "string"},
            "authorization_granted": {"const": False},
            "proof_state_changed": {"const": False},
        }
    ),
    "execute_preauthorized_recovery": _output_schema(
        {
            "schema_version": {"const": "task_evaluation_recovery_result.v1"},
            "run_id": {"type": "string"},
            "attempt_id": {"type": "string"},
            "status": {"enum": ["completed", "failed", "timed_out", "failed_teardown"]},
            "typed_result": {"type": "object"},
            "shared_paid_resource_admission_validated": {"const": True},
            "proof_effect": {"const": "none"},
            "scientific_validity_inferred": {"const": False},
            "recovery_result_digest": {"type": "string"},
        },
        additional_properties=True,
    ),
    "explain_deterministic_decision": _output_schema(
        {
            "contract_present": {"const": True},
            "digest_matches": {"const": True},
            "overall_outcome": {},
            "claim_ceiling": {},
            "verdict_changed_by_tool": {"const": False},
        }
    ),
}


def _descriptor(
    tool_id: str,
    category: str,
    *,
    expected_artifacts: Sequence[str],
    input_properties: Mapping[str, Mapping[str, Any]],
    required_inputs: Sequence[str],
    mutability: str = "read_only",
    allowed_modes: Sequence[str] = (
        "shadow",
        "advise",
        "execute_non_spend",
        "execute_preauthorized",
    ),
    minimum_mode: str = "shadow",
    max_cost_usd: float = 0.0,
    max_retries: int = 0,
    timeout_seconds: float = 30.0,
    idempotency: str = "deterministic_for_bound_inputs",
) -> ToolDescriptor:
    output_schema = _TOOL_OUTPUT_SCHEMAS.get(tool_id)
    if output_schema is None:
        raise ValueError(f"registered_tool_output_schema_missing:{tool_id}")
    safety_level = {
        "read_only": "proof_safe_read_only",
        "reversible_mutation": "proof_safe_reversible_non_spend",
        "external_side_effect": "preauthorized_external_side_effect",
    }[mutability]
    rollback_reason = {
        "read_only": "read_only",
        "reversible_mutation": "delete_supervisor_scoped_generated_artifacts",
        "external_side_effect": "mandatory_provider_teardown_and_provider_zero_proof",
    }[mutability]
    return ToolDescriptor.from_mapping(
        {
            "schema_version": "task_evaluation_supervisor_tool.v1",
            "tool_id": tool_id,
            "version": "1",
            "category": category,
            "mutability": mutability,
            "idempotency": idempotency,
            "input_schema": {
                "type": "object",
                "required": list(required_inputs),
                "properties": {key: dict(schema) for key, schema in input_properties.items()},
                "additionalProperties": False,
            },
            "output_schema": output_schema,
            "expected_artifacts": list(expected_artifacts),
            "max_cost_usd": max_cost_usd,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "safety_level": safety_level,
            "required_authority": {"minimum_mode": minimum_mode},
            "allowed_modes": list(allowed_modes),
            "proof_effect": "none",
            "evidence_requirements": [],
            "rollback": {
                "required": mutability != "read_only",
                "reason": rollback_reason,
            },
        }
    )


def default_tool_descriptors() -> tuple[ToolDescriptor, ...]:
    """Return the stable Phase 0/1 tool surface.

    These are descriptors, not generic execution handles.  Later phases may
    bind implementations only after their mutation and authority contracts are
    independently tested.
    """

    return (
        _descriptor(
            "inspect_site_task_testbed",
            "capture_testbed_inspection",
            expected_artifacts=["capture_testbed_inspection.v1"],
            input_properties={"testbed_digest": {"type": "string"}},
            required_inputs=["testbed_digest"],
        ),
        _descriptor(
            "plan_capture_reconstruction_route",
            "capture_reconstruction_routing",
            expected_artifacts=["task_evaluation_capture_reconstruction_route.v1"],
            input_properties={
                "capture_build_digest": {"type": "string"},
                "requested_claim_types": {"type": "array"},
            },
            required_inputs=["capture_build_digest", "requested_claim_types"],
        ),
        _descriptor(
            "compile_frozen_frame_dataset",
            "capture_reconstruction_dataset_compilation",
            expected_artifacts=["reconstruction_dataset_manifest.v1"],
            input_properties={
                "capture_build_digest": {"type": "string"},
                "capture_reconstruction_route_digest": {"type": "string"},
            },
            required_inputs=[
                "capture_build_digest",
                "capture_reconstruction_route_digest",
            ],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=300.0,
        ),
        _descriptor(
            "compile_arkit_metric_scaffold",
            "arkit_metric_scaffold_compilation",
            expected_artifacts=["reconstruction_result.v1", "arkit_metric_scaffold.v1"],
            input_properties={
                "capture_build_digest": {"type": "string"},
                "capture_reconstruction_route_digest": {"type": "string"},
            },
            required_inputs=[
                "capture_build_digest",
                "capture_reconstruction_route_digest",
            ],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=1_800.0,
            idempotency="content_addressed_raw_contract_3_2_compilation",
        ),
        _descriptor(
            "export_arkit_reconstruction_dataset",
            "arkit_reconstruction_dataset_export",
            expected_artifacts=["arkit_reconstruction_dataset_export.v1"],
            input_properties={"arkit_export_request_digest": {"type": "string"}},
            required_inputs=["arkit_export_request_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=600.0,
            idempotency="content_addressed_candidate_only_export",
        ),
        _descriptor(
            "normalize_native_360_capture",
            "native_360_normalization",
            expected_artifacts=["native_360_capture_normalization.v1"],
            input_properties={
                "capture_build_digest": {"type": "string"},
                "capture_reconstruction_route_digest": {"type": "string"},
            },
            required_inputs=[
                "capture_build_digest",
                "capture_reconstruction_route_digest",
            ],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=300.0,
        ),
        _descriptor(
            "compile_equirectangular_virtual_rig",
            "equirectangular_normalization",
            expected_artifacts=["equirectangular_virtual_rig_compilation.v1"],
            input_properties={
                "capture_build_digest": {"type": "string"},
                "capture_reconstruction_route_digest": {"type": "string"},
            },
            required_inputs=[
                "capture_build_digest",
                "capture_reconstruction_route_digest",
            ],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=600.0,
        ),
        _descriptor(
            "run_pose_estimation",
            "capture_reconstruction_pose_estimation",
            expected_artifacts=["pose_estimation_result.v1"],
            input_properties={"pose_estimation_request_digest": {"type": "string"}},
            required_inputs=["pose_estimation_request_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=3_600.0,
        ),
        _descriptor(
            "train_gaussian_reconstruction",
            "capture_reconstruction_training",
            expected_artifacts=["reconstruction_training_result.v1"],
            input_properties={
                "reconstruction_training_request_digest": {"type": "string"}
            },
            required_inputs=["reconstruction_training_request_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=14_400.0,
        ),
        _descriptor(
            "evaluate_heldout_appearance",
            "capture_reconstruction_independent_appearance_qa",
            expected_artifacts=["visual_heldout_evaluation_report.v1"],
            input_properties={
                "heldout_appearance_evaluation_request_digest": {"type": "string"}
            },
            required_inputs=["heldout_appearance_evaluation_request_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=1_800.0,
            idempotency="frozen_hidden_split_independent_evaluator",
        ),
        _descriptor(
            "compile_metric_geometry", "capture_reconstruction_metric_geometry",
            expected_artifacts=["metric_geometry_manifest.v1"],
            input_properties={"source_artifact_digest": {"type": "string"}},
            required_inputs=["source_artifact_digest"], mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend", timeout_seconds=3_600.0,
        ),
        _descriptor(
            "compile_collision_candidate", "capture_reconstruction_collision",
            expected_artifacts=["mesh_collider_candidate_manifest.v1"],
            input_properties={"metric_geometry_manifest_digest": {"type": "string"}},
            required_inputs=["metric_geometry_manifest_digest"], mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend", timeout_seconds=3_600.0,
        ),
        _descriptor(
            "qualify_collision_candidate", "capture_reconstruction_collision_qa",
            expected_artifacts=["collider_qualification_report.v1"],
            input_properties={"collider_candidate_manifest_digest": {"type": "string"}},
            required_inputs=["collider_candidate_manifest_digest"], mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend", timeout_seconds=1_800.0,
        ),
        _descriptor(
            "package_nurec_openusd", "capture_reconstruction_packaging",
            expected_artifacts=["nurec_openusd_packaging_result.v1"],
            input_properties={"packaging_request_digest": {"type": "string"}},
            required_inputs=["packaging_request_digest"], mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend", timeout_seconds=3_600.0,
            idempotency="content_addressed_receipt_replay",
        ),
        _descriptor(
            "verify_isaac_asset", "capture_reconstruction_isaac_verification",
            expected_artifacts=["isaac_asset_verification_result.v1"],
            input_properties={"isaac_verification_request_digest": {"type": "string"}},
            required_inputs=["isaac_verification_request_digest"], mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend", timeout_seconds=7_200.0,
        ),
        _descriptor(
            "import_external_reconstruction", "capture_reconstruction_external_import",
            expected_artifacts=["external_reconstruction_import_receipt.v1"],
            input_properties={"external_import_request_digest": {"type": "string"}},
            required_inputs=["external_import_request_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend"],
            minimum_mode="execute_non_spend",
            timeout_seconds=1_800.0,
            idempotency="content_addressed_receipt_replay",
        ),
        _descriptor(
            "invoke_authorized_reconstruction_provider",
            "capture_reconstruction_remote_provider",
            expected_artifacts=[
                "reconstruction_provider_execution_receipt.v1",
                "reconstruction_provider_deletion_receipt.v1",
            ],
            input_properties={"provider_execution_request_digest": {"type": "string"}},
            required_inputs=["provider_execution_request_digest"],
            mutability="external_side_effect",
            allowed_modes=["execute_preauthorized"],
            minimum_mode="execute_preauthorized",
            max_cost_usd=100.0,
            max_retries=3,
            timeout_seconds=86_400.0,
            idempotency="provider_job_identity_bound_no_unchanged_blocker_retry",
        ),
        _descriptor(
            "diagnose_reconstruction_failure",
            "capture_reconstruction_failure_diagnosis",
            expected_artifacts=["reconstruction_failure_diagnosis.v1"],
            input_properties={
                "reconstruction_failure_diagnosis_request_digest": {"type": "string"}
            },
            required_inputs=["reconstruction_failure_diagnosis_request_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=30.0,
            idempotency="deterministic_failure_fingerprint_no_unchanged_retry",
        ),
        _descriptor(
            "validate_proposed_claim_graph",
            "claim_contract_validation",
            expected_artifacts=["proposed_claim_graph.v1"],
            input_properties={"request_digest": {"type": "string"}},
            required_inputs=["request_digest"],
        ),
        _descriptor(
            "materialize_clarification_request",
            "claim_contract_validation",
            expected_artifacts=["task_evaluation_clarification_request.v1"],
            input_properties={
                "source_digest": {"type": "string"},
                "questions": {"type": "array"},
                "blocking_fields": {"type": "array"},
            },
            required_inputs=["source_digest", "questions", "blocking_fields"],
            mutability="reversible_mutation",
        ),
        _descriptor(
            "compile_deterministic_evidence_plan",
            "evidence_method_routing",
            expected_artifacts=["evidence_plan.v1"],
            input_properties={"plan_digest": {"type": "string"}},
            required_inputs=["plan_digest"],
        ),
        _descriptor(
            "materialize_compiled_leaf_runs",
            "local_leaf_run_compilation",
            expected_artifacts=["evidence_plan.v1", "evaluation_run_spec.v1"],
            input_properties={
                "request_digest": {"type": "string"},
                "testbed_digest": {"type": "string"},
            },
            required_inputs=["request_digest", "testbed_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
        ),
        _descriptor(
            "inspect_normalized_evidence_results",
            "runtime_failure_diagnosis",
            expected_artifacts=["typed_failure_diagnosis.v1"],
            input_properties={
                "result_digest": {"type": "string"},
                "execution_requested": {"type": "boolean"},
            },
            required_inputs=["result_digest", "execution_requested"],
        ),
        _descriptor(
            "propose_targeted_recapture",
            "capture_testbed_inspection",
            expected_artifacts=["targeted_recapture_request.v1"],
            input_properties={
                "source_digest": {"type": "string"},
                "missing_evidence": {"type": "array"},
                "full_site_recapture_requested": {"type": "boolean"},
            },
            required_inputs=[
                "source_digest",
                "missing_evidence",
                "full_site_recapture_requested",
            ],
            mutability="reversible_mutation",
        ),
        _descriptor(
            "propose_adversarial_scenarios",
            "scenario_generation",
            expected_artifacts=["task_evaluation_scenario_proposal_set.v1"],
            input_properties={
                "request_digest": {"type": "string"},
                "scenarios": {"type": "array"},
                "candidate_results_observed": {"type": "boolean"},
            },
            required_inputs=["request_digest", "scenarios", "candidate_results_observed"],
            mutability="reversible_mutation",
        ),
        _descriptor(
            "materialize_authorization_request",
            "runtime_failure_diagnosis",
            expected_artifacts=["task_evaluation_authorization_request.v1"],
            input_properties={
                "tool_id": {"type": "string"},
                "reason": {"type": "string"},
                "requested_max_cost_usd": {"type": "number"},
                "requested_ttl_seconds": {"type": "integer"},
                "requested_retry_count": {"type": "integer"},
                "requested_provider_ids": {"type": "array"},
                "requested_action_ids": {"type": "array"},
            },
            required_inputs=[
                "tool_id",
                "reason",
                "requested_max_cost_usd",
                "requested_ttl_seconds",
                "requested_retry_count",
                "requested_provider_ids",
                "requested_action_ids",
            ],
            mutability="reversible_mutation",
        ),
        _descriptor(
            "execute_preauthorized_recovery",
            "runtime_failure_recovery",
            expected_artifacts=["task_evaluation_recovery_result.v1"],
            input_properties={
                "action_id": {"type": "string"},
                "provider_id": {"type": "string"},
                "immutable_commit_sha": {"type": "string"},
                "input_digests": {"type": "array"},
                "projected_cost_usd": {"type": "number"},
                "failure_type": {"type": "string"},
            },
            required_inputs=[
                "action_id",
                "provider_id",
                "immutable_commit_sha",
                "input_digests",
                "projected_cost_usd",
                "failure_type",
            ],
            mutability="external_side_effect",
            allowed_modes=["execute_preauthorized"],
            minimum_mode="execute_preauthorized",
            max_cost_usd=100.0,
            max_retries=3,
            timeout_seconds=3_600.0,
            idempotency="receipt_bound_attempt;provider_action_not_assumed_idempotent",
        ),
        _descriptor(
            "explain_deterministic_decision",
            "post_run_diagnosis",
            expected_artifacts=["post_run_diagnosis.v1"],
            input_properties={"decision_envelope_digest": {"type": "string"}},
            required_inputs=["decision_envelope_digest"],
        ),
    )


@dataclass(frozen=True)
class ToolRegistry:
    _tools: Mapping[str, ToolDescriptor]

    @classmethod
    def from_descriptors(cls, values: Sequence[ToolDescriptor]) -> "ToolRegistry":
        tools: dict[str, ToolDescriptor] = {}
        for descriptor in values:
            mapping = descriptor.to_mapping()
            tool_id = str(mapping["tool_id"])
            if tool_id in tools:
                raise ValueError(f"duplicate_supervisor_tool:{tool_id}")
            tools[tool_id] = descriptor
        return cls(tools)

    @classmethod
    def default(cls) -> "ToolRegistry":
        return cls.from_descriptors(default_tool_descriptors())

    def resolve(self, tool_id: str) -> ToolDescriptor | None:
        return self._tools.get(str(tool_id or "").strip())

    def allowed_tool_ids_for_capability(self, capability: str) -> tuple[str, ...]:
        return tuple(
            tool_id
            for tool_id in _CAPABILITY_TOOL_IDS.get(str(capability or ""), ())
            if tool_id in self._tools
        )

    def manifest(self) -> dict[str, Any]:
        descriptors = [self._tools[tool_id].to_mapping() for tool_id in sorted(self._tools)]
        value = {
            "schema_version": TOOL_REGISTRY_SCHEMA_VERSION,
            "tools": descriptors,
            "unrestricted_shell_available": False,
            "unrestricted_filesystem_available": False,
            "unrestricted_network_available": False,
            "unrestricted_provider_access_available": False,
            "proof_mutation_tools_registered": False,
            "paid_tools_registered": any(
                float(row.get("max_cost_usd") or 0) > 0 for row in descriptors
            ),
            "physical_action_tools_registered": False,
        }
        value["tool_registry_digest"] = canonical_digest(value, digest_field="tool_registry_digest")
        return value

    @property
    def digest(self) -> str:
        return str(self.manifest()["tool_registry_digest"])

    def disposition(
        self,
        proposal_value: Mapping[str, Any],
        authority_value: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...]]:
        """Deterministically classify a proposal without executing it."""

        proposal = ActionProposal.from_mapping(proposal_value).to_mapping()
        authority = AuthorityEnvelope.from_mapping(authority_value).to_mapping()
        mode = AutonomyMode(str(authority["mode"]))
        blockers: list[str] = []
        tool_id = str(proposal.get("tool_id") or "")
        tool = self.resolve(tool_id) if tool_id else None
        if tool_id and tool is None:
            blockers.append("unregistered_tool")
        tool_value = tool.to_mapping() if tool is not None else {}
        if (
            tool is not None
            and mode is not AutonomyMode.ADVISE
            and mode.value not in set(tool_value.get("allowed_modes") or [])
        ):
            blockers.append("tool_not_allowed_in_mode")
        if tool_id and tool_id not in set(authority.get("allowed_tool_ids") or []):
            blockers.append("tool_not_in_authority_envelope")
        proposal_cost = float(proposal.get("estimated_cost_usd") or 0)
        if tool is not None and proposal_cost > float(tool_value.get("max_cost_usd") or 0):
            blockers.append("proposal_exceeds_tool_cost_limit")
        if mode is not AutonomyMode.ADVISE and proposal_cost > float(
            authority.get("max_cost_usd") or 0
        ):
            blockers.append("proposal_exceeds_cost_authority")
        if str(proposal.get("requested_proof_effect") or "") != "none":
            blockers.append("proof_mutation_requested")
        if tool is not None:
            blockers.extend(
                self._input_schema_errors(
                    proposal.get("parameters"), tool_value.get("input_schema")
                )
            )
        if blockers:
            return "refused", tuple(sorted(set(blockers)))
        if mode is AutonomyMode.DISABLED:
            return "refused", ("supervisor_disabled",)
        if mode is AutonomyMode.SHADOW:
            return "shadow_only", ()
        if mode is AutonomyMode.ADVISE:
            return "requires_operator_approval", ()
        return "eligible", ()

    @staticmethod
    def _schema_errors(value: Any, schema_value: Any, *, prefix: str) -> list[str]:
        if not isinstance(value, Mapping):
            return [f"{prefix}_not_mapping"]
        schema = dict(schema_value) if isinstance(schema_value, Mapping) else {}
        properties = dict(schema.get("properties") or {})
        required = {str(item) for item in schema.get("required") or []}
        errors = [f"{prefix}_missing:{key}" for key in sorted(required - set(value))]
        if schema.get("additionalProperties") is False:
            errors.extend(f"{prefix}_unknown:{key}" for key in sorted(set(value) - set(properties)))
        type_checks = {
            "string": lambda item: isinstance(item, str),
            "boolean": lambda item: isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            ),
            "array": lambda item: isinstance(item, list),
            "object": lambda item: isinstance(item, Mapping),
        }
        for key in sorted(set(value) & set(properties)):
            property_schema = dict(properties[key])
            expected = str(property_schema.get("type") or "")
            check = type_checks.get(expected)
            if check is not None and not check(value[key]):
                errors.append(f"{prefix}_type:{key}:{expected}")
            if "const" in property_schema and value[key] != property_schema["const"]:
                errors.append(f"{prefix}_const:{key}")
            if "enum" in property_schema and value[key] not in property_schema["enum"]:
                errors.append(f"{prefix}_enum:{key}")
        return errors

    @classmethod
    def _input_schema_errors(cls, value: Any, schema_value: Any) -> list[str]:
        return cls._schema_errors(value, schema_value, prefix="tool_input")


def validate_tool_observation_binding(
    observation_value: Mapping[str, Any],
    *,
    run_id: str,
    capability: str,
    registry: ToolRegistry,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a tool result against the exact registered execution scope."""

    observation = ToolObservation.from_mapping(observation_value).to_mapping()
    validated_authority = AuthorityEnvelope.from_mapping(authority).to_mapping()
    if observation["run_id"] != run_id:
        raise ValueError("tool_observation_run_mismatch")
    if observation["capability"] != capability:
        raise ValueError("tool_observation_capability_mismatch")
    if observation["authority_digest"] != validated_authority["authority_digest"]:
        raise ValueError("tool_observation_authority_mismatch")
    tool_id = str(observation["tool_id"])
    descriptor = registry.resolve(tool_id)
    if descriptor is None:
        raise ValueError("tool_observation_unregistered_tool")
    descriptor_value = descriptor.to_mapping()
    if tool_id not in registry.allowed_tool_ids_for_capability(capability):
        raise ValueError("tool_observation_capability_tool_mismatch")
    if tool_id not in set(validated_authority.get("allowed_tool_ids") or []):
        raise ValueError("tool_observation_tool_not_authorized")
    if observation["tool_version"] != descriptor_value["version"]:
        raise ValueError("tool_observation_version_mismatch")
    if observation["mutability"] != descriptor_value["mutability"]:
        raise ValueError("tool_observation_mutability_mismatch")
    expected_runtime = (
        "blueprint_preauthorized_recovery_controller"
        if observation["mutability"] == "external_side_effect"
        else "blueprint_local_deterministic_non_spend"
    )
    if observation["runtime_identity"] != expected_runtime:
        raise ValueError("tool_observation_runtime_identity_mismatch")
    if observation["output_digest"] != canonical_digest(observation["typed_result"]):
        raise ValueError("tool_observation_output_digest_mismatch")
    if observation["status"] != "refused":
        output_errors = registry._schema_errors(
            observation["typed_result"],
            descriptor_value["output_schema"],
            prefix="tool_output",
        )
        if output_errors:
            raise ValueError("tool_observation_output_schema_invalid:" + ",".join(output_errors))
        if tool_id == "plan_capture_reconstruction_route":
            route = validate_capture_reconstruction_route(observation["typed_result"]["route"])
            if (
                observation["typed_result"].get("route_digest")
                != route["capture_reconstruction_route_digest"]
                or observation["typed_result"].get("capture_authority_profile")
                != route["capture_authority_profile"]
            ):
                raise ValueError("tool_observation_capture_reconstruction_route_binding_mismatch")
    produced_artifact_types = {
        str(row.get("artifact_type") or "") for row in observation["produced_artifact_references"]
    }
    if not produced_artifact_types.issubset(set(descriptor_value["expected_artifacts"])):
        raise ValueError("tool_observation_artifact_type_not_registered")
    cost = float(observation["cost_usd"])
    duration = float(observation["duration_seconds"])
    retries = int(observation["retries"])
    if cost > float(descriptor_value["max_cost_usd"]):
        raise ValueError("tool_observation_tool_cost_exceeded")
    if cost > float(validated_authority["max_cost_usd"]):
        raise ValueError("tool_observation_authority_cost_exceeded")
    if duration > float(descriptor_value["timeout_seconds"]):
        raise ValueError("tool_observation_tool_duration_exceeded")
    if duration > float(validated_authority["max_duration_seconds"]):
        raise ValueError("tool_observation_authority_duration_exceeded")
    if retries > int(descriptor_value["max_retries"]):
        raise ValueError("tool_observation_tool_retries_exceeded")
    if retries > int(validated_authority["max_retries"]):
        raise ValueError("tool_observation_authority_retries_exceeded")
    if observation["mutability"] != "external_side_effect" and cost != 0:
        raise ValueError("tool_observation_non_spend_cost_nonzero")
    if observation["mutability"] == "external_side_effect" and (
        validated_authority["mode"] != AutonomyMode.EXECUTE_PREAUTHORIZED.value
    ):
        raise ValueError("tool_observation_external_side_effect_wrong_mode")
    return observation


@dataclass(frozen=True)
class RegisteredToolBinding:
    """One SDK-callable binding to a deterministic, read-only Blueprint tool."""

    tool_id: str
    description: str
    input_schema: Mapping[str, Any]
    timeout_seconds: float
    invoke: Callable[[Mapping[str, Any]], Mapping[str, Any]]


_CAPABILITY_TOOL_IDS: dict[str, tuple[str, ...]] = {
    "claim_task_interpreter": (
        "validate_proposed_claim_graph",
        "materialize_clarification_request",
    ),
    "capture_testbed_supervisor": (
        "inspect_site_task_testbed",
        "plan_capture_reconstruction_route",
        "compile_frozen_frame_dataset",
        "compile_arkit_metric_scaffold",
        "export_arkit_reconstruction_dataset",
        "normalize_native_360_capture",
        "compile_equirectangular_virtual_rig",
        "run_pose_estimation",
        "train_gaussian_reconstruction",
        "evaluate_heldout_appearance",
        "compile_metric_geometry",
        "compile_collision_candidate",
        "qualify_collision_candidate",
        "package_nurec_openusd",
        "verify_isaac_asset",
        "import_external_reconstruction",
        "propose_targeted_recapture",
    ),
    "evaluation_method_router": (
        "compile_deterministic_evidence_plan",
        "materialize_compiled_leaf_runs",
    ),
    "runtime_failure_recovery": (
        "inspect_normalized_evidence_results",
        "diagnose_reconstruction_failure",
        "materialize_authorization_request",
        "execute_preauthorized_recovery",
    ),
    "scenario_adversarial_proposer": ("propose_adversarial_scenarios",),
    "post_run_diagnostician": ("explain_deterministic_decision",),
}


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


def _compile_frozen_frame_dataset(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    compiler = getattr(context, "reconstruction_dataset_compiler", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:compile_frozen_frame_dataset")
    if not callable(compiler):
        raise ValueError("reconstruction_dataset_compiler_not_injected")
    capture_build = context.capture_build
    expected_capture_digest = arguments.get("capture_build_digest")
    actual_capture_digest = (
        capture_build.get("capture_build_digest") if isinstance(capture_build, Mapping) else None
    )
    claim_types = sorted(
        {
            str(row.get("claim_type") or "").strip()
            for row in (
                context.decision_request.get("claims", [])
                if isinstance(context.decision_request, Mapping)
                else []
            )
            if isinstance(row, Mapping) and str(row.get("claim_type") or "").strip()
        }
    )
    route = (
        build_capture_reconstruction_route(capture_build, requested_claim_types=claim_types)
        if isinstance(capture_build, Mapping)
        and expected_capture_digest
        and expected_capture_digest == actual_capture_digest
        else None
    )
    expected_route_digest = arguments.get("capture_reconstruction_route_digest")
    if (
        route is None
        or route.get("status") != "route_proposed"
        or expected_route_digest != route.get("capture_reconstruction_route_digest")
    ):
        raise ValueError("registered_tool_reconstruction_route_binding_mismatch")
    output_root = Path(root_value) / "generated" / "reconstruction_frame_dataset"
    compiled = compiler(
        request={
            "capture_build_digest": actual_capture_digest,
            "capture_reconstruction_route_digest": expected_route_digest,
            "capture_authority_profile": route["capture_authority_profile"],
            "requested_claim_types": claim_types,
        },
        output_root=output_root,
    )
    if not isinstance(compiled, Mapping):
        raise ValueError("reconstruction_dataset_compiler_result_not_object")
    dataset = dict(compiled)
    dataset_digest = dataset.get("dataset_manifest_digest")
    split_digest = dataset.get("train_heldout_split_digest")
    parent = dataset.get("parent_artifact_or_event")
    if (
        dataset.get("schema_version") != "reconstruction_dataset_manifest.v1"
        or not isinstance(parent, Mapping)
        or parent.get("capture_build_digest") != actual_capture_digest
        or dataset.get("capture_authority_profile") != route["capture_authority_profile"]
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(dataset_digest or "")) is None
        or dataset_digest != canonical_digest(dataset, digest_field="dataset_manifest_digest")
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(split_digest or "")) is None
        or dataset.get("candidate_dataset_contains_hidden_heldout_pixels") is not False
        or dataset.get("candidate_can_modify_split") is not False
        or dataset.get("proof_effect") != "decoded_observation_availability_only"
    ):
        raise ValueError("reconstruction_dataset_compiler_result_contract_invalid")
    path = write_phase2_artifact(
        root_value,
        "generated/reconstruction_frame_dataset/reconstruction_dataset_manifest.json",
        dataset,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "dataset_manifest_digest": dataset_digest,
        "split_digest": split_digest,
        "hidden_heldout_isolated": True,
        "candidate_can_change_split": False,
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": dataset_digest,
            "artifact_type": "reconstruction_dataset_manifest.v1",
        }
    ]


def _capture_reconstruction_route_for_tool(
    *, context: Any, arguments: Mapping[str, Any], required_profile: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    capture_build = context.capture_build
    expected_capture_digest = arguments.get("capture_build_digest")
    actual_capture_digest = (
        capture_build.get("capture_build_digest")
        if isinstance(capture_build, Mapping)
        else None
    )
    claim_types = sorted(
        {
            str(row.get("claim_type") or "").strip()
            for row in (
                context.decision_request.get("claims", [])
                if isinstance(context.decision_request, Mapping)
                else []
            )
            if isinstance(row, Mapping) and str(row.get("claim_type") or "").strip()
        }
    )
    route = (
        build_capture_reconstruction_route(
            capture_build, requested_claim_types=claim_types
        )
        if isinstance(capture_build, Mapping)
        and expected_capture_digest
        and expected_capture_digest == actual_capture_digest
        else None
    )
    if (
        route is None
        or route.get("status") != "route_proposed"
        or route.get("capture_authority_profile") != required_profile
        or arguments.get("capture_reconstruction_route_digest")
        != route.get("capture_reconstruction_route_digest")
    ):
        raise ValueError("registered_tool_reconstruction_route_binding_mismatch")
    return capture_build, route


def _compile_arkit_metric_scaffold(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    compiler = getattr(context, "arkit_metric_scaffold_compiler", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:compile_arkit_metric_scaffold"
        )
    if not callable(compiler):
        raise ValueError("arkit_metric_scaffold_compiler_not_injected")
    _, route = _capture_reconstruction_route_for_tool(
        context=context,
        arguments=arguments,
        required_profile="iphone_arkit_lidar",
    )
    output_root = Path(root_value) / "generated" / "arkit_metric_scaffold"
    emitted = compiler(
        request={
            "capture_build_digest": route["capture_build_digest"],
            "capture_reconstruction_route_digest": route[
                "capture_reconstruction_route_digest"
            ],
            "capture_authority_profile": "iphone_arkit_lidar",
            "requested_claim_types": route["requested_claim_types"],
        },
        output_root=output_root,
    )
    if not isinstance(emitted, Mapping):
        raise ValueError("arkit_metric_scaffold_compiler_result_not_object")
    try:
        result = normalize_reconstruction_result(emitted)
    except ValueError as exc:
        raise ValueError("arkit_metric_scaffold_compiler_result_contract_invalid") from exc
    assets = result.get("asset_references")
    metrics = result.get("validation_metrics")
    claim_ceiling = result.get("claim_ceiling")
    metric_reference = assets.get("metric_scaffold") if isinstance(assets, Mapping) else None
    export_reference = (
        assets.get("arkit_reconstruction_dataset_export")
        if isinstance(assets, Mapping)
        else None
    )
    if (
        result.get("method_id") != "local_arkit_metric_scaffold"
        or not isinstance(metric_reference, Mapping)
        or not isinstance(export_reference, Mapping)
        or not isinstance(metrics, Mapping)
        or metrics.get("decoded_pts_verified") is not True
        or metrics.get("pose_refinement_executed") is not False
        or metrics.get("independent_metric_scale_validation_passed") is not False
        or not isinstance(claim_ceiling, Mapping)
        or claim_ceiling.get("calibrated_camera_poses") is not True
        or claim_ceiling.get("sensor_declared_metric_scale") is not True
        or claim_ceiling.get("metric_scale") is not False
        or claim_ceiling.get("metric_reference_layer") is not False
        or claim_ceiling.get("collision_geometry") is not False
        or claim_ceiling.get("physical_task_success") is not False
    ):
        raise ValueError("arkit_metric_scaffold_compiler_result_contract_invalid")
    result_digest = result["reconstruction_result_digest"]
    path = write_phase2_artifact(
        root_value,
        "generated/arkit_metric_scaffold/reconstruction_result.json",
        result,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "reconstruction_result_digest": result_digest,
        "metric_scaffold_digest": metric_reference["digest"],
        "arkit_export_digest": export_reference["digest"],
        "decoded_pts_verified": True,
        "raw_arkit_poses_modified": False,
        "metric_scale_independently_validated": False,
        "claim_ceiling": "sensor_declared_metric_scaffold",
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": result_digest,
            "artifact_type": "reconstruction_result.v1",
        }
    ]


def _export_arkit_reconstruction_dataset(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    source_value = getattr(context, "arkit_reconstruction_dataset_request", None)
    exporter = getattr(context, "arkit_reconstruction_dataset_exporter", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:export_arkit_reconstruction_dataset"
        )
    if not isinstance(source_value, Mapping) or not callable(exporter):
        raise ValueError("arkit_reconstruction_dataset_exporter_not_injected")
    try:
        request = build_arkit_reconstruction_dataset_export_request(source_value)
    except ValueError as exc:
        raise ValueError("arkit_export_request_contract_invalid") from exc
    expected_digest = arguments.get("arkit_export_request_digest")
    if expected_digest != request["arkit_export_request_digest"]:
        raise ValueError("registered_tool_source_digest_mismatch:export_arkit_reconstruction_dataset")
    output_root = Path(root_value) / "generated" / "arkit_reconstruction_dataset_export"
    emitted = exporter(source_artifact=request, output_root=output_root)
    if not isinstance(emitted, Mapping):
        raise ValueError("arkit_reconstruction_dataset_export_result_not_object")
    result = dict(emitted)
    result_digest = result.get("arkit_reconstruction_dataset_export_digest")
    if (
        result.get("schema_version") != "arkit_reconstruction_dataset_export.v1"
        or result_digest
        != canonical_digest(
            result, digest_field="arkit_reconstruction_dataset_export_digest"
        )
        or result.get("source_capture_digest") != request["source_capture_digest"]
        or result.get("reconstruction_dataset_digest")
        != request["dataset_manifest"].get("dataset_manifest_digest")
        or result.get("frozen_split_digest")
        != request["split_manifest"].get("split_digest")
        or result.get("metric_scaffold_digest") != request["metric_scaffold_digest"]
        or result.get("hidden_heldout_pixels_included") is not False
        or result.get("raw_arkit_poses_modified") is not False
        or result.get("metric_scale_validation_status") != "not_executed"
        or result.get("proof_effect") != "calibrated_reconstruction_request_only"
        or result.get("claim_ceiling") != "calibrated_camera_trajectory"
    ):
        raise ValueError("arkit_reconstruction_dataset_export_result_contract_invalid")
    path = write_phase2_artifact(
        root_value,
        "generated/arkit_reconstruction_dataset_export/arkit_reconstruction_dataset_export.json",
        result,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "arkit_export_digest": result_digest,
        "camera_calibration_digest": result["camera_calibration_digest"],
        "camera_observation_digest": result["camera_observation_digest"],
        "pose_refinement_request_digest": result["pose_refinement_request_digest"],
        "hidden_heldout_pixels_included": False,
        "raw_arkit_poses_modified": False,
        "claim_ceiling": "calibrated_camera_trajectory",
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": result_digest,
            "artifact_type": "arkit_reconstruction_dataset_export.v1",
        }
    ]


def _normalize_native_360_capture(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    normalizer = getattr(context, "native_360_normalizer", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:normalize_native_360_capture"
        )
    if not callable(normalizer):
        raise ValueError("native_360_normalizer_not_injected")
    capture_build = context.capture_build
    expected_capture_digest = arguments.get("capture_build_digest")
    actual_capture_digest = (
        capture_build.get("capture_build_digest")
        if isinstance(capture_build, Mapping)
        else None
    )
    claim_types = sorted(
        {
            str(row.get("claim_type") or "").strip()
            for row in (
                context.decision_request.get("claims", [])
                if isinstance(context.decision_request, Mapping)
                else []
            )
            if isinstance(row, Mapping) and str(row.get("claim_type") or "").strip()
        }
    )
    route = (
        build_capture_reconstruction_route(
            capture_build, requested_claim_types=claim_types
        )
        if isinstance(capture_build, Mapping)
        and expected_capture_digest
        and expected_capture_digest == actual_capture_digest
        else None
    )
    expected_route_digest = arguments.get("capture_reconstruction_route_digest")
    if (
        route is None
        or route.get("status") != "route_proposed"
        or route.get("capture_authority_profile") != "camera_360_native"
        or expected_route_digest != route.get("capture_reconstruction_route_digest")
    ):
        raise ValueError("registered_tool_native_360_route_binding_mismatch")
    output_root = Path(root_value) / "generated" / "native_360_normalization"
    normalized = normalizer(
        request={
            "capture_build_digest": actual_capture_digest,
            "capture_reconstruction_route_digest": expected_route_digest,
            "capture_authority_profile": route["capture_authority_profile"],
            "requested_claim_types": claim_types,
        },
        output_root=output_root,
    )
    if not isinstance(normalized, Mapping):
        raise ValueError("native_360_normalizer_result_not_object")
    result = dict(normalized)
    result_digest = result.get("native_360_normalization_digest")
    parent = result.get("parent_artifact_or_event")
    status = result.get("status")
    blockers = result.get("blockers")
    status_consistent = bool(
        (status == "normalized" and blockers == [])
        or (status == "blocked" and isinstance(blockers, list) and blockers)
    )
    proof_consistent = bool(
        (
            status == "normalized"
            and result.get("proof_effect") == "calibrated_native_360_rig_only"
            and result.get("claim_ceiling") == "calibrated_camera_rig"
        )
        or (
            status == "blocked"
            and result.get("proof_effect") == "none"
            and result.get("claim_ceiling") == "decoded_native_container"
        )
    )
    if (
        result.get("schema_version") != "native_360_capture_normalization.v1"
        or not isinstance(parent, Mapping)
        or parent.get("capture_build_digest") != actual_capture_digest
        or parent.get("capture_reconstruction_route_digest") != expected_route_digest
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(result_digest or "")) is None
        or result_digest
        != canonical_digest(result, digest_field="native_360_normalization_digest")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(result.get("rig_declaration_digest") or "")
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(result.get("dual_fisheye_binding_digest") or ""),
        )
        is None
        or not status_consistent
        or not proof_consistent
        or result.get("raw_native_bytes_remain_authoritative") is not True
        or result.get("original_native_bytes_modified") is not False
        or result.get("agent_altered_calibration") is not False
        or result.get("metric_scale_status") != "not_established"
        or result.get("appearance_reconstruction_proven") is not False
        or result.get("metric_geometry_proven") is not False
        or result.get("collision_geometry_proven") is not False
        or result.get("isaac_compatibility_proven") is not False
    ):
        raise ValueError("native_360_normalizer_result_contract_invalid")
    path = write_phase2_artifact(
        root_value,
        "generated/native_360_normalization/native_360_capture_normalization.json",
        result,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "normalization_digest": result_digest,
        "rig_declaration_digest": result["rig_declaration_digest"],
        "dual_fisheye_binding_digest": result["dual_fisheye_binding_digest"],
        "status": status,
        "claim_ceiling": result["claim_ceiling"],
        "agent_altered_calibration": False,
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": result_digest,
            "artifact_type": "native_360_capture_normalization.v1",
        }
    ]


def _compile_equirectangular_virtual_rig(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    compiler = getattr(context, "equirectangular_virtual_rig_compiler", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:compile_equirectangular_virtual_rig"
        )
    if not callable(compiler):
        raise ValueError("equirectangular_virtual_rig_compiler_not_injected")
    capture_build = context.capture_build
    expected_capture_digest = arguments.get("capture_build_digest")
    actual_capture_digest = (
        capture_build.get("capture_build_digest")
        if isinstance(capture_build, Mapping)
        else None
    )
    claim_types = sorted(
        {
            str(row.get("claim_type") or "").strip()
            for row in (
                context.decision_request.get("claims", [])
                if isinstance(context.decision_request, Mapping)
                else []
            )
            if isinstance(row, Mapping) and str(row.get("claim_type") or "").strip()
        }
    )
    route = (
        build_capture_reconstruction_route(
            capture_build, requested_claim_types=claim_types
        )
        if isinstance(capture_build, Mapping)
        and expected_capture_digest
        and expected_capture_digest == actual_capture_digest
        else None
    )
    expected_route_digest = arguments.get("capture_reconstruction_route_digest")
    if (
        route is None
        or route.get("status") != "route_proposed"
        or route.get("capture_authority_profile")
        not in {"camera_360_equirectangular", "camera_360_native"}
        or expected_route_digest != route.get("capture_reconstruction_route_digest")
    ):
        raise ValueError("registered_tool_equirectangular_route_binding_mismatch")
    output_root = Path(root_value) / "generated" / "equirectangular_virtual_rig"
    compiled = compiler(
        request={
            "capture_build_digest": actual_capture_digest,
            "capture_reconstruction_route_digest": expected_route_digest,
            "capture_authority_profile": route["capture_authority_profile"],
            "requested_claim_types": claim_types,
            "access_scope": "candidate_training_and_validation_only",
        },
        output_root=output_root,
    )
    if not isinstance(compiled, Mapping):
        raise ValueError("equirectangular_virtual_rig_compiler_result_not_object")
    result = dict(compiled)
    result_digest = result.get("equirectangular_compilation_digest")
    rig_digest = result.get("output_digests", {}).get("virtual_rig_digest")
    parent = result.get("parent_artifact_or_event")
    view_count = result.get("virtual_observation_count")
    if (
        result.get("schema_version")
        != "equirectangular_virtual_rig_compilation.v1"
        or not isinstance(parent, Mapping)
        or parent.get("capture_build_digest") != actual_capture_digest
        or parent.get("capture_reconstruction_route_digest") != expected_route_digest
        or result.get("access_scope")
        != "candidate_training_and_validation_only"
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(result_digest or "")) is None
        or result_digest
        != canonical_digest(result, digest_field="equirectangular_compilation_digest")
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(rig_digest or "")) is None
        or isinstance(view_count, bool)
        or not isinstance(view_count, int)
        or view_count < 12
        or view_count % 12 != 0
        or result.get("proof_effect")
        != "deterministic_shared_center_projection_only"
        or result.get("claim_ceiling")
        != "equirectangular_virtual_camera_rig"
        or result.get("source_panorama_pixels_remain_authoritative") is not True
        or result.get("virtual_views_are_captured_evidence") is not False
        or result.get("virtual_views_are_independent_physical_cameras") is not False
        or result.get("camera_trajectory_proven") is not False
        or result.get("metric_scale_proven") is not False
        or result.get("appearance_reconstruction_proven") is not False
        or result.get("collision_geometry_proven") is not False
        or result.get("isaac_compatibility_proven") is not False
    ):
        raise ValueError("equirectangular_virtual_rig_compiler_result_contract_invalid")
    path = write_phase2_artifact(
        root_value,
        "generated/equirectangular_virtual_rig/equirectangular_virtual_rig_compilation.json",
        result,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "compilation_digest": result_digest,
        "virtual_rig_digest": rig_digest,
        "virtual_observation_count": view_count,
        "shared_optical_center_required": True,
        "virtual_views_are_captured_evidence": False,
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": result_digest,
            "artifact_type": "equirectangular_virtual_rig_compilation.v1",
        }
    ]


def _run_pose_estimation(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    runtime = getattr(context, "pose_estimator", None)
    request_value = getattr(context, "pose_estimation_request", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:run_pose_estimation")
    if not callable(runtime):
        raise ValueError("pose_estimator_not_injected")
    if not isinstance(request_value, Mapping):
        raise ValueError("pose_estimation_request_not_injected")
    try:
        request = build_pose_estimation_request(request_value)
    except ValueError as exc:
        raise ValueError("pose_estimation_request_contract_invalid") from exc
    expected_digest = arguments.get("pose_estimation_request_digest")
    if expected_digest != request["pose_estimation_request_digest"]:
        raise ValueError("registered_tool_pose_request_binding_mismatch")
    output_root = Path(root_value) / "generated" / "pose_estimation"
    emitted = runtime(request=request, output_root=output_root)
    if not isinstance(emitted, Mapping):
        raise ValueError("pose_estimator_result_not_object")
    try:
        result = build_pose_estimation_result(emitted)
    except ValueError as exc:
        raise ValueError("pose_estimator_result_contract_invalid") from exc
    if (
        result.get("pose_estimation_request_digest") != expected_digest
        or result.get("source_capture_digest") != request.get("source_capture_digest")
        or result.get("train_heldout_split_digest")
        != request.get("train_heldout_split_digest")
        or result.get("container_image_digest") != request.get("container_image_digest")
        or result.get("source_commit_sha") != request.get("source_commit_sha")
        or result.get("camera_calibration_binding")
        != request.get("camera_calibration_binding")
    ):
        raise ValueError("pose_estimator_result_lineage_mismatch")
    result_digest = result["pose_estimation_result_digest"]
    path = write_phase2_artifact(
        root_value,
        "generated/pose_estimation/pose_estimation_result.json",
        result,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "status": result["status"],
        "failure_code": result.get("failure_code"),
        "pose_estimation_result_digest": result_digest,
        "registered_observation_count": len(result["registered_observation_ids"]),
        "rejected_observation_count": len(result["rejected_observation_ids"]),
        "heldout_labels_included": False,
        "candidate_self_graded": False,
        "claim_ceiling": "calibrated_camera_trajectory",
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": result_digest,
            "artifact_type": "pose_estimation_result.v1",
        }
    ]


def _train_gaussian_reconstruction(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    runtime = getattr(context, "gaussian_reconstruction_trainer", None)
    request_value = getattr(context, "reconstruction_training_request", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:train_gaussian_reconstruction"
        )
    if not callable(runtime):
        raise ValueError("gaussian_reconstruction_trainer_not_injected")
    if not isinstance(request_value, Mapping):
        raise ValueError("reconstruction_training_request_not_injected")
    try:
        request = build_training_request(request_value)
    except ValueError as exc:
        raise ValueError("reconstruction_training_request_contract_invalid") from exc
    expected_digest = arguments.get("reconstruction_training_request_digest")
    if expected_digest != request["reconstruction_training_request_digest"]:
        raise ValueError("registered_tool_training_request_binding_mismatch")
    output_root = Path(root_value) / "generated" / "gaussian_reconstruction"
    emitted = runtime(request=request, output_root=output_root)
    if not isinstance(emitted, Mapping):
        raise ValueError("gaussian_reconstruction_trainer_result_not_object")
    try:
        result = build_training_result(emitted)
    except ValueError as exc:
        raise ValueError("gaussian_reconstruction_trainer_result_contract_invalid") from exc
    if (
        result.get("reconstruction_training_request_digest") != expected_digest
        or result.get("source_capture_digest") != request.get("source_capture_digest")
        or result.get("train_heldout_split_digest")
        != request.get("train_heldout_split_digest")
        or result.get("container_image_digest") != request.get("container_image_digest")
        or result.get("source_commit_sha") != request.get("source_commit_sha")
        or result.get("camera_calibration_binding")
        != request.get("camera_calibration_binding")
    ):
        raise ValueError("gaussian_reconstruction_trainer_result_lineage_mismatch")
    result_digest = result["reconstruction_training_result_digest"]
    path = write_phase2_artifact(
        root_value,
        "generated/gaussian_reconstruction/reconstruction_training_result.json",
        result,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "status": result["status"],
        "failure_code": result.get("failure_code"),
        "reconstruction_training_result_digest": result_digest,
        "checkpoint_count": len(result["checkpoint_references"]),
        "heldout_labels_included": False,
        "candidate_self_graded": False,
        "claim_ceiling": "appearance_reconstruction",
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": result_digest,
            "artifact_type": "reconstruction_training_result.v1",
        }
    ]


_GEOMETRY_TOOL_CONFIG = {
    "evaluate_heldout_appearance": (
        "heldout_appearance_evaluation_request", "heldout_appearance_evaluator",
        "heldout_appearance_evaluation_request_digest",
        build_visual_heldout_evaluation_report,
        "visual_heldout_evaluation_report_digest",
        "visual_heldout_evaluation_report.v1", "appearance_reconstruction", "status",
    ),
    "compile_metric_geometry": (
        "metric_geometry_source", "metric_geometry_compiler", "source_artifact_digest",
        build_metric_geometry_manifest, "metric_geometry_manifest_digest",
        "metric_geometry_manifest.v1", "metric_reference_geometry", None,
    ),
    "compile_collision_candidate": (
        "metric_geometry_manifest", "collision_candidate_compiler",
        "metric_geometry_manifest_digest", build_collider_candidate_manifest,
        "collider_candidate_manifest_digest", "mesh_collider_candidate_manifest.v1",
        "collision_geometry_candidate", None,
    ),
    "qualify_collision_candidate": (
        "collider_candidate_manifest", "collision_candidate_qualifier",
        "collider_candidate_manifest_digest", build_collider_qualification_report,
        "collider_qualification_digest", "collider_qualification_report.v1",
        "bounded_navigation_simulation", "decision",
    ),
    "package_nurec_openusd": (
        "nurec_packaging_request", "nurec_openusd_packager", "packaging_request_digest",
        build_nurec_openusd_packaging_result, "packaging_result_digest",
        "nurec_openusd_packaging_result.v1", "openusd_package", None,
    ),
    "verify_isaac_asset": (
        "isaac_verification_request", "isaac_asset_verifier",
        "isaac_verification_request_digest", build_isaac_asset_verification_result,
        "isaac_verification_result_digest", "isaac_asset_verification_result.v1",
        "isaac_load_render_compatibility", "status",
    ),
    "import_external_reconstruction": (
        "external_reconstruction_import_request", "external_reconstruction_importer",
        "external_import_request_digest", build_external_reconstruction_import_receipt,
        "external_import_receipt_digest", "external_reconstruction_import_receipt.v1",
        "external_reconstruction_import", "status",
    ),
}


def _execute_geometry_contract_tool(
    *, context: Any, tool_id: str, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(f"registered_tool_execution_scope_missing:{tool_id}")
    source_attr, runtime_attr, digest_field, builder, result_digest_field, artifact_type, ceiling, decision_field = _GEOMETRY_TOOL_CONFIG[tool_id]
    source = getattr(context, source_attr, None)
    runtime = getattr(context, runtime_attr, None)
    if not isinstance(source, Mapping) or not callable(runtime):
        raise ValueError(f"registered_tool_runtime_not_injected:{tool_id}")
    if tool_id == "package_nurec_openusd":
        try:
            source = build_nurec_openusd_packaging_request(source)
        except ValueError as exc:
            raise ValueError(f"registered_tool_request_contract_invalid:{tool_id}") from exc
    if tool_id == "evaluate_heldout_appearance":
        try:
            source = build_heldout_appearance_evaluation_request(source)
        except ValueError as exc:
            raise ValueError(f"registered_tool_request_contract_invalid:{tool_id}") from exc
    if tool_id == "import_external_reconstruction":
        try:
            source = build_external_reconstruction_import_request(source)
        except ValueError as exc:
            raise ValueError(f"registered_tool_request_contract_invalid:{tool_id}") from exc
    expected = arguments.get(digest_field)
    actual = source.get(digest_field)
    if not isinstance(actual, str) or expected != actual:
        raise ValueError(f"registered_tool_source_digest_mismatch:{tool_id}")
    if digest_field.endswith("request_digest") and actual != canonical_digest(
        source, digest_field=digest_field
    ):
        raise ValueError(f"registered_tool_request_contract_invalid:{tool_id}")
    output_root = Path(root_value) / "generated" / tool_id
    emitted = runtime(source_artifact=dict(source), output_root=output_root)
    if not isinstance(emitted, Mapping):
        raise ValueError(f"registered_tool_result_not_object:{tool_id}")
    try:
        result = builder(emitted)
    except ValueError as exc:
        raise ValueError(f"registered_tool_result_contract_invalid:{tool_id}") from exc
    if tool_id == "compile_collision_candidate" and result.get(digest_field) != actual:
        raise ValueError(f"registered_tool_result_lineage_mismatch:{tool_id}")
    if tool_id == "qualify_collision_candidate" and result.get(digest_field) != actual:
        raise ValueError(f"registered_tool_result_lineage_mismatch:{tool_id}")
    if tool_id == "verify_isaac_asset" and result.get("packaging_result_digest") != source.get(
        "packaging_result_digest"
    ):
        raise ValueError(f"registered_tool_result_lineage_mismatch:{tool_id}")
    if tool_id == "package_nurec_openusd" and any(
        result.get(field) != source.get(field)
        for field in (
            "metric_geometry_manifest_digest",
            "collider_candidate_manifest_digest",
            "collider_qualification_digest",
            "collider_qualification_decision",
        )
    ):
        raise ValueError(f"registered_tool_result_lineage_mismatch:{tool_id}")
    if tool_id == "import_external_reconstruction" and (
        result.get("external_import_request_digest") != actual
        or result.get("source_capture_digest") != source.get("source_capture_digest")
    ):
        raise ValueError(f"registered_tool_result_lineage_mismatch:{tool_id}")
    if tool_id == "evaluate_heldout_appearance" and any(
        result.get(field) != source.get(source_field)
        for field, source_field in (
            ("source_capture_digest", "source_capture_digest"),
            ("reconstruction_dataset_digest", "reconstruction_dataset_digest"),
            ("frozen_split_digest", "frozen_split_digest"),
            (
                "candidate_reconstruction_result_digest",
                "candidate_reconstruction_result_digest",
            ),
            (
                "evaluation_request_digest",
                "heldout_appearance_evaluation_request_digest",
            ),
        )
    ):
        raise ValueError(f"registered_tool_result_lineage_mismatch:{tool_id}")
    result_digest = result[result_digest_field]
    path = write_phase2_artifact(
        root_value, f"generated/{tool_id}/{artifact_type}.json", result
    )
    decision = result.get(decision_field) if decision_field else None
    return {
        "contract_present": True,
        "digest_matches": True,
        "artifact_digest": result_digest,
        "claim_ceiling": ceiling,
        "decision": decision,
        "proof_state_changed": False,
    }, [{
        "artifact_path": str(path.relative_to(Path(root_value))),
        "artifact_digest": result_digest,
        "artifact_type": artifact_type,
    }]


def _diagnose_reconstruction_failure(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    source_value = getattr(context, "reconstruction_failure_diagnosis_request", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(
            "registered_tool_execution_scope_missing:diagnose_reconstruction_failure"
        )
    if not isinstance(source_value, Mapping):
        raise ValueError("reconstruction_failure_diagnosis_request_not_injected")
    try:
        request = build_reconstruction_failure_diagnosis_request(source_value)
    except ValueError as exc:
        raise ValueError("reconstruction_failure_diagnosis_request_contract_invalid") from exc
    expected_digest = arguments.get("reconstruction_failure_diagnosis_request_digest")
    if expected_digest != request["reconstruction_failure_diagnosis_request_digest"]:
        raise ValueError(
            "registered_tool_source_digest_mismatch:diagnose_reconstruction_failure"
        )
    diagnosis = diagnose_reconstruction_failure(request)
    path = write_phase2_artifact(
        root_value,
        "generated/reconstruction_failure_diagnosis/reconstruction_failure_diagnosis.json",
        diagnosis,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "diagnosis_digest": diagnosis["reconstruction_failure_diagnosis_digest"],
        "diagnosed_failure_code": diagnosis["diagnosed_failure_code"],
        "identical_attempt_count": diagnosis["identical_attempt_count"],
        "unchanged_deterministic_retry_allowed": diagnosis[
            "unchanged_deterministic_retry_allowed"
        ],
        "terminal_for_current_configuration": diagnosis[
            "terminal_for_current_configuration"
        ],
        "legal_next_actions": diagnosis["legal_next_actions"],
        "failed_evidence_preserved": True,
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": diagnosis["reconstruction_failure_diagnosis_digest"],
            "artifact_type": "reconstruction_failure_diagnosis.v1",
        }
    ]


def _bound_artifact(
    context: Any,
    *,
    tool_id: str,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    produced_artifact_references: list[dict[str, Any]] = []
    if tool_id == "validate_proposed_claim_graph":
        value = context.decision_request
        digest_key = "request_digest"
        expected = arguments.get(digest_key)
        actual = value.get(digest_key) if isinstance(value, Mapping) else None
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "claim_ids": [
                str(row.get("claim_id"))
                for row in (value or {}).get("claims", [])
                if isinstance(row, Mapping) and row.get("claim_id")
            ]
            if isinstance(value, Mapping)
            else [],
        }
    elif tool_id == "materialize_clarification_request":
        return _materialize_clarification_request(context=context, arguments=arguments)
    elif tool_id == "inspect_site_task_testbed":
        value = context.testbed
        expected = arguments.get("testbed_digest")
        actual = value.get("testbed_digest") if isinstance(value, Mapping) else None
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "evidence_inventory_count": len((value or {}).get("evidence_inventory", []))
            if isinstance(value, Mapping)
            else 0,
            "governance": dict((value or {}).get("governance") or {})
            if isinstance(value, Mapping)
            else {},
        }
    elif tool_id == "plan_capture_reconstruction_route":
        value = context.capture_build
        expected = arguments.get("capture_build_digest")
        actual = value.get("capture_build_digest") if isinstance(value, Mapping) else None
        supplied_claim_types = arguments.get("requested_claim_types")
        expected_claim_types = sorted(
            {
                str(row.get("claim_type") or "").strip()
                for row in (
                    context.decision_request.get("claims", [])
                    if isinstance(context.decision_request, Mapping)
                    else []
                )
                if isinstance(row, Mapping) and str(row.get("claim_type") or "").strip()
            }
        )
        if (
            not isinstance(supplied_claim_types, list)
            or not all(isinstance(item, str) and item.strip() for item in supplied_claim_types)
            or sorted(set(supplied_claim_types)) != expected_claim_types
        ):
            raise ValueError(
                "registered_tool_claim_scope_mismatch:plan_capture_reconstruction_route"
            )
        route = (
            build_capture_reconstruction_route(
                value,
                requested_claim_types=expected_claim_types,
            )
            if isinstance(value, Mapping) and actual and expected == actual
            else None
        )
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "route": route or {},
            "route_digest": (route or {}).get("capture_reconstruction_route_digest"),
            "capture_authority_profile": (route or {}).get("capture_authority_profile"),
            "execution_authorized_by_route": False,
            "proof_state_changed": False,
        }
    elif tool_id == "compile_frozen_frame_dataset":
        return _compile_frozen_frame_dataset(context=context, arguments=arguments)
    elif tool_id == "compile_arkit_metric_scaffold":
        return _compile_arkit_metric_scaffold(context=context, arguments=arguments)
    elif tool_id == "export_arkit_reconstruction_dataset":
        return _export_arkit_reconstruction_dataset(
            context=context, arguments=arguments
        )
    elif tool_id == "normalize_native_360_capture":
        return _normalize_native_360_capture(context=context, arguments=arguments)
    elif tool_id == "compile_equirectangular_virtual_rig":
        return _compile_equirectangular_virtual_rig(
            context=context, arguments=arguments
        )
    elif tool_id == "run_pose_estimation":
        return _run_pose_estimation(context=context, arguments=arguments)
    elif tool_id == "train_gaussian_reconstruction":
        return _train_gaussian_reconstruction(context=context, arguments=arguments)
    elif tool_id in _GEOMETRY_TOOL_CONFIG:
        return _execute_geometry_contract_tool(
            context=context, tool_id=tool_id, arguments=arguments
        )
    elif tool_id == "diagnose_reconstruction_failure":
        return _diagnose_reconstruction_failure(
            context=context, arguments=arguments
        )
    elif tool_id == "compile_deterministic_evidence_plan":
        value = context.evidence_plan
        expected = arguments.get("plan_digest")
        actual = value.get("plan_digest") if isinstance(value, Mapping) else None
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "step_count": len((value or {}).get("steps", [])) if isinstance(value, Mapping) else 0,
            "compiled_by_agent": False,
        }
    elif tool_id == "materialize_compiled_leaf_runs":
        return _materialize_leaf_runs(context=context, arguments=arguments)
    elif tool_id == "propose_targeted_recapture":
        return _materialize_targeted_recapture_request(
            context=context,
            arguments=arguments,
        )
    elif tool_id == "materialize_authorization_request":
        return _materialize_authorization_request(context=context, arguments=arguments)
    elif tool_id == "execute_preauthorized_recovery":
        return _execute_preauthorized_recovery(context=context, arguments=arguments)
    elif tool_id == "propose_adversarial_scenarios":
        return _materialize_scenario_proposal_set(context=context, arguments=arguments)
    elif tool_id == "inspect_normalized_evidence_results":
        expected = arguments.get("result_digest")
        selected = next(
            (
                row
                for row in context.evidence_results
                if isinstance(row, Mapping) and row.get("result_digest") == expected
            ),
            None,
        )
        typed_result = {
            "contract_present": selected is not None,
            "digest_matches": selected is not None,
            "status": selected.get("status") if selected is not None else None,
            "failure_type": selected.get("failure_type") if selected is not None else None,
            "execution_requested": arguments.get("execution_requested") is True,
        }
    elif tool_id == "explain_deterministic_decision":
        value = context.decision_envelope
        expected = arguments.get("decision_envelope_digest")
        actual = value.get("decision_envelope_digest") if isinstance(value, Mapping) else None
        typed_result = {
            "contract_present": value is not None,
            "digest_matches": bool(actual and expected == actual),
            "overall_outcome": value.get("overall_outcome") if isinstance(value, Mapping) else None,
            "claim_ceiling": value.get("claim_ceiling") if isinstance(value, Mapping) else None,
            "verdict_changed_by_tool": False,
        }
    else:  # pragma: no cover - construction prevents this branch
        raise ValueError(f"registered_non_spend_tool_not_implemented:{tool_id}")
    if not typed_result.get("contract_present") or not typed_result.get("digest_matches"):
        raise ValueError(f"registered_tool_bound_artifact_mismatch:{tool_id}")
    return typed_result, produced_artifact_references


def non_spend_tool_bindings(
    *,
    capability: str,
    context: Any,
    registry: ToolRegistry,
    authority: Mapping[str, Any],
    observation_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[RegisteredToolBinding, ...]:
    """Bind only capability-scoped read tools in execute_non_spend mode."""

    validated_authority = AuthorityEnvelope.from_mapping(authority).to_mapping()
    if validated_authority.get("mode") not in {
        AutonomyMode.EXECUTE_NON_SPEND.value,
        AutonomyMode.EXECUTE_PREAUTHORIZED.value,
    }:
        return ()
    bindings: list[RegisteredToolBinding] = []
    for tool_id in _CAPABILITY_TOOL_IDS.get(capability, ()):
        if tool_id == "compile_frozen_frame_dataset" and not callable(
            getattr(context, "reconstruction_dataset_compiler", None)
        ):
            continue
        if tool_id == "compile_arkit_metric_scaffold" and not callable(
            getattr(context, "arkit_metric_scaffold_compiler", None)
        ):
            continue
        if tool_id == "export_arkit_reconstruction_dataset" and (
            not callable(getattr(context, "arkit_reconstruction_dataset_exporter", None))
            or not isinstance(
                getattr(context, "arkit_reconstruction_dataset_request", None), Mapping
            )
        ):
            continue
        if tool_id == "normalize_native_360_capture" and not callable(
            getattr(context, "native_360_normalizer", None)
        ):
            continue
        if tool_id == "compile_equirectangular_virtual_rig" and not callable(
            getattr(context, "equirectangular_virtual_rig_compiler", None)
        ):
            continue
        if tool_id == "run_pose_estimation" and (
            not callable(getattr(context, "pose_estimator", None))
            or not isinstance(getattr(context, "pose_estimation_request", None), Mapping)
        ):
            continue
        if tool_id == "train_gaussian_reconstruction" and (
            not callable(getattr(context, "gaussian_reconstruction_trainer", None))
            or not isinstance(
                getattr(context, "reconstruction_training_request", None), Mapping
            )
        ):
            continue
        if tool_id in _GEOMETRY_TOOL_CONFIG:
            source_attr, runtime_attr, *_ = _GEOMETRY_TOOL_CONFIG[tool_id]
            if not isinstance(getattr(context, source_attr, None), Mapping) or not callable(
                getattr(context, runtime_attr, None)
            ):
                continue
        if tool_id == "diagnose_reconstruction_failure" and not isinstance(
            getattr(context, "reconstruction_failure_diagnosis_request", None), Mapping
        ):
            continue
        descriptor = registry.resolve(tool_id)
        if descriptor is None:
            raise ValueError(f"registered_non_spend_tool_missing:{tool_id}")
        descriptor_value = descriptor.to_mapping()
        if validated_authority["mode"] not in set(descriptor_value.get("allowed_modes") or []):
            continue

        def invoke(
            arguments: Mapping[str, Any],
            *,
            selected_tool_id: str = tool_id,
            selected_descriptor_value: Mapping[str, Any] = descriptor_value,
        ) -> Mapping[str, Any]:
            try:
                proposal = ActionProposal.from_mapping(
                    {
                        "schema_version": "task_evaluation_supervisor_action_proposal.v1",
                        "proposal_id": f"{context.run_id}-{selected_tool_id}-sdk-call",
                        "run_id": context.run_id,
                        "capability": capability,
                        "action_type": (
                            "registered_read_only_tool_call"
                            if selected_descriptor_value["mutability"] == "read_only"
                            else "registered_scoped_tool_call"
                        ),
                        "tool_id": selected_tool_id,
                        "parameters": dict(arguments),
                        "reasons": ["agents_sdk_requested_registered_observation"],
                        "evidence_refs": [],
                        "estimated_cost_usd": 0.0,
                        "requested_proof_effect": "none",
                        "disposition": "shadow_only",
                    }
                )
                disposition, blockers = registry.disposition(
                    proposal.to_mapping(), validated_authority
                )
                if disposition != "eligible" or blockers:
                    raise ValueError(
                        f"registered_tool_call_refused:{selected_tool_id}:{','.join(blockers)}"
                    )
                typed_result, produced_artifact_references = _bound_artifact(
                    context,
                    tool_id=selected_tool_id,
                    arguments=arguments,
                )
                status = (
                    "completed"
                    if selected_tool_id != "execute_preauthorized_recovery"
                    or typed_result.get("status") == "completed"
                    else "failed"
                )
                typed_failure = typed_result.get("typed_failure") if status == "failed" else None
            except ValueError as exc:
                typed_result = {}
                produced_artifact_references = []
                status = "refused"
                typed_failure = {
                    "failure_type": "deterministic_tool_refusal",
                    "reason": str(exc),
                    "retryable": False,
                }
            observation: dict[str, Any] = {
                "schema_version": TOOL_OBSERVATION_SCHEMA_VERSION,
                "run_id": context.run_id,
                "capability": capability,
                "tool_id": selected_tool_id,
                "tool_version": "1",
                "status": status,
                "typed_result": typed_result,
                "typed_failure": typed_failure,
                "produced_artifact_references": produced_artifact_references,
                "input_digest": canonical_digest(dict(arguments)),
                "output_digest": canonical_digest(typed_result),
                "runtime_identity": (
                    "blueprint_preauthorized_recovery_controller"
                    if selected_tool_id == "execute_preauthorized_recovery"
                    else "blueprint_local_deterministic_non_spend"
                ),
                "mutability": selected_descriptor_value["mutability"],
                "cost_usd": float(typed_result.get("actual_cost_usd") or 0.0),
                "duration_seconds": float(typed_result.get("duration_seconds") or 0.0),
                "retries": max(0, int(typed_result.get("attempt_number") or 1) - 1),
                "authority_digest": validated_authority["authority_digest"],
                "proof_effect": "none",
                "warnings": ["tool_observation_is_not_accepted_evidence"],
                "suggested_next_legal_actions": list(
                    typed_result.get("suggested_next_legal_actions") or []
                ),
            }
            observation["observation_digest"] = canonical_digest(
                observation, digest_field="observation_digest"
            )
            validated_observation = validate_tool_observation_binding(
                observation,
                run_id=context.run_id,
                capability=capability,
                registry=registry,
                authority=validated_authority,
            )
            if observation_sink is not None:
                observation_sink(validated_observation)
            return validated_observation

        bindings.append(
            RegisteredToolBinding(
                tool_id=tool_id,
                description=(
                    f"Blueprint registered read-only tool {tool_id}. Returns a typed "
                    "non-authoritative observation with proof_effect=none."
                ),
                input_schema=dict(descriptor_value["input_schema"]),
                timeout_seconds=float(descriptor_value["timeout_seconds"]),
                invoke=invoke,
            )
        )
    return tuple(bindings)


__all__ = [
    "RegisteredToolBinding",
    "TOOL_OBSERVATION_SCHEMA_VERSION",
    "TOOL_REGISTRY_SCHEMA_VERSION",
    "ToolRegistry",
    "default_tool_descriptors",
    "non_spend_tool_bindings",
    "validate_tool_observation_binding",
]
