"""Lane 3: routed development execution plus the scene preflight gate."""

from __future__ import annotations

import copy
import hashlib
import os
import sys
from datetime import date
from pathlib import Path

import pytest

from blueprint_pipeline import measurement_mujoco_worker
from blueprint_pipeline.decision_evidence_contracts import EvidencePlan
from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)
from blueprint_pipeline.measurement_routed_execution import (
    RoutedExecutionError,
    attach_routed_development_evidence,
    build_routed_cross_engine_development_report,
    execute_routed_development_stage,
)
from blueprint_pipeline.simready_asset_lane import (
    build_simready_asset_request,
    generate_simready_asset_draft,
    preflight_simready_scene,
    validate_simready_asset_manifest,
)
from blueprint_pipeline.task_site_measurement_routing import (
    ALL_CAPABILITY_FIELDS,
    derive_task_measurement_requirements,
    route_task_site_measurement,
    validate_measurement_qualification,
    validate_method_capability_profile,
    validate_site_evidence_profile,
)


mujoco = pytest.importorskip("mujoco")


SHA_A = "sha256:" + "a" * 64
SHA_DEV = "sha256:" + "1" * 64
SHA_QUAL = "sha256:" + "2" * 64

_COMPARISON_CASE_SHAPE = {
    "protocol_family": "rigid_body_drop",
    "body_shape": "box",
    "half_size_m": [0.05, 0.05, 0.05],
    "mass_kg": 0.25,
    "initial_height_m": 0.25,
    "friction": 0.8,
    "gravity_m_s2": -9.81,
    "timestep_s": 0.002,
    "duration_s": 0.8,
}

_BOX_VERTICES = [
    [-0.05, -0.05, 0.0], [0.05, -0.05, 0.0], [0.05, 0.05, 0.0], [-0.05, 0.05, 0.0],
    [-0.05, -0.05, 0.1], [0.05, -0.05, 0.1], [0.05, 0.05, 0.1], [-0.05, 0.05, 0.1],
]
_BOX_FACES = [
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
]


def _routed_decision() -> dict:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "routed-collision", "claim_type": "collision_contact"},
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )
    caps = set(requirements["required_capabilities"])
    values: dict = {field: False for field in ALL_CAPABILITY_FIELDS}
    for field in (
        "plugin_versions", "robot_model_formats", "supported_embodiments",
        "supported_end_effectors", "action_representation_types",
        "qualification_record_ids", "qualified_task_classes",
        "qualified_material_regimes", "qualified_robot_ids",
        "qualified_end_effector_ids", "qualified_controller_ids",
        "qualified_sensor_ids", "qualified_site_classes", "qualified_metric_ids",
        "known_failure_modes", "prohibited_extrapolations", "asset_license_ids",
        "model_license_ids", "subprocessor_regions", "output_formats",
    ):
        values[field] = []
    for field in caps:
        values[field] = True
    values.update({
        "method_id": "fixture-routed-development-engine",
        "method_family": "traditional_simulation", "version": "1",
        "release_date": "2026-08-01", "commit_hash": "fixture",
        "container_digest": SHA_A, "solver_backend": "fixture",
        "numeric_precision": "float64", "deterministic_mode": "strict",
        "operating_system": "linux", "gpu_model": "none", "driver_version": "none",
        "random_seed_policy": "frozen", "contact_formulation": "fixture",
        "maximum_control_rate_hz": 1000, "qualified_parameter_ranges": {},
        "qualified_claim_ceiling": "C3", "qualification_expiration": "2027-08-01",
        "harmful_false_negative_bound": 0.01, "maximum_latency_class": "interactive",
        "maximum_compute_class": "cpu", "estimated_cost_class": "low",
        "data_retention_days": 0, "source_available": True,
        "local_offline_supported": True, "api_only": False,
        "commercial_use_allowed": True, "redistribution_allowed": True,
        "provider_training_use_allowed": False, "deletion_right_supported": True,
        "output_export_supported": True,
    })
    profile = validate_method_capability_profile({
        "schema_version": "method_capability_profile.v1",
        "method_id": "fixture-routed-development-engine",
        "capabilities": values,
        "evidence_quality": {"source": "development_fixture"},
        "expected_cost_usd": 1.0,
        "expected_latency_seconds": 1.0,
    })
    qualification = validate_measurement_qualification({
        "schema_version": "measurement_qualification_record.v1",
        "qualification_id": "development-fixture-routed",
        "method_id": "fixture-routed-development-engine",
        "method_version": "1",
        "capability_profile_digest": profile["capability_profile_digest"],
        "admission_record_digest": SHA_A,
        "admission_stage": "R7",
        "status": "approved",
        "qualified_capabilities": sorted(caps),
        "claim_ceiling": "C3",
        "scope": {
            "task_classes": ["rigid_pick_place"], "material_regimes": ["none"],
            "robot_ids": [], "end_effector_ids": [], "controller_ids": [],
            "sensor_ids": [], "metric_ids": [], "parameter_ranges": {},
        },
        "metrics": {
            "physical_accuracy_error": 0.01, "uncertainty": 0.02,
            "scope_distance": 0.0, "harmful_false_negative_rate": 0.001,
            "reproducibility_score": 1.0, "privacy_preference": 1.0,
        },
        "approval": {
            "signature_status": "verified",
            "signature_id": "development-fixture-signature",
            "approved_by": ["benchmark-owner", "independent-reviewer"],
            "agent_approved": False,
        },
        "expiration_date": "2027-08-01",
        "self_grading": False,
    })
    site = validate_site_evidence_profile({
        "schema_version": "site_evidence_profile.v1",
        "profile_id": "routed-site",
        "bundle_id": "capture-1",
        "bundle_hash": SHA_A,
        "provenance_record_id": "provenance-fixture",
        "rights": {"commercial_evaluation_allowed": True},
        "privacy": {"external_processing_allowed": False},
        "coordinate_system": {"metric_scale_verified": True},
        "evidence": {
            evidence_id: {
                "available": True, "validated": True,
                "record_id": f"record-{evidence_id}",
            }
            for evidence_id in (
                "metric_scale", "robot_site_registration", "validated_collider",
                "mass_inertia", "friction_contact", "material_parameters",
            )
        },
        "limitations": {"known_missing_regions": [], "forbidden_claims": []},
    })
    decision = route_task_site_measurement(
        requirements, site, [profile], [qualification],
        catalog_snapshot_hash=SHA_A, as_of=date(2026, 8, 2),
    )
    assert decision["status"] == "route_selected"
    return decision


def _execution_binding() -> dict:
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="routed-development-1",
        method_ids=["mujoco-3"],
        development_split_digest=SHA_DEV,
        qualification_split_digest=SHA_QUAL,
        capture_bundle_digests=[SHA_A],
        robot_controller_digests=[SHA_A],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 1.0,
            "maximum_harmful_false_negative_rate": 1.0,
            "minimum_coverage": 0.0,
        },
        compute_budget={"usd": 0.0},
    )
    case = build_benchmark_case_manifest(
        spec,
        case_id="routed-box-settle",
        split="development",
        input_artifact_digests=[SHA_A],
        task_class="rigid_pick_place",
        material_regime="rigid",
        operating_point={
            "scene": "box_settle",
            "drop_height_m": 0.25,
            "box_half_extent_m": 0.05,
            "friction": 0.8,
            "comparison_case_shape": _COMPARISON_CASE_SHAPE,
        },
    )
    source = Path(measurement_mujoco_worker.__file__).read_bytes()
    return {
        "descriptor": build_measurement_adapter_descriptor("mujoco-3"),
        "benchmark_spec": spec,
        "case_manifest": case,
        "worker_argv": [sys.executable, "-m", "blueprint_pipeline.measurement_mujoco_worker"],
        "implementation_id": measurement_mujoco_worker.MUJOCO_WORKER_ID,
        "implementation_version": measurement_mujoco_worker.MUJOCO_WORKER_VERSION,
        "implementation_digest": "sha256:" + hashlib.sha256(source).hexdigest(),
        "backend_id": "mujoco_cpu",
        "precision": "float64",
        "seed": 0,
        "solver_settings": {
            "timestep": 0.002,
            "steps": 400,
            "engine_version_policy": "record_actual_development_only",
        },
    }


def _evidence_plan(decision: dict) -> dict:
    return EvidencePlan.from_mapping({
        "schema_version": "evidence_plan.v1",
        "plan_id": "routed-development-plan",
        "request_id": "routed-development-request",
        "decision_id": "routed-development-decision",
        "request_digest": SHA_DEV,
        "testbed_id": "routed-development-testbed",
        "testbed_version": "1",
        "testbed_digest": SHA_QUAL,
        "claim_plans": [{
            "claim_id": "routed-collision",
            "measurement_routing_decision": decision,
        }],
        "execution_order": [],
        "stop_conditions": [],
        "escalation_conditions": [],
        "physical_evidence_requests": [],
        "compiled_evaluation_run_specs": [],
        "non_evaluation_run_steps": [],
        "prohibited_claims": [
            "physical_task_success", "deployment_readiness", "safety_certification",
        ],
        "shared_dependency_warnings": [],
        "budget_status": {"max_cost_usd": 0.0},
    }).to_mapping()


def test_routed_stage_executes_through_the_boundary_as_development_evidence() -> None:
    decision = _routed_decision()
    outcome = execute_routed_development_stage(
        decision,
        stage_index=0,
        execution_id="routed-development-execution-1",
        execute=True,
        **_execution_binding(),
    )
    record, bundle = outcome["record"], outcome["bundle"]
    assert record["routed_method_id"] == "fixture-routed-development-engine"
    assert record["executing_candidate_id"] == "mujoco-3"
    assert record["binding_kind"] == "development_demonstration_only"
    assert record["execution_status"] == "completed"
    assert record["evidence_class"] == "development_execution"
    assert record["development_evidence_only"] is True
    assert record["route_authorized_execution"] is False
    assert record["qualification_created"] is False
    assert record["physical_success_established"] is False
    assert record["routing_decision_digest"] == decision["routing_decision_digest"]
    assert bundle["prediction"] is not None
    assert record["prediction_digest"] == bundle["prediction"]["prediction_digest"]
    plan = _evidence_plan(decision)
    attachment = attach_routed_development_evidence(
        plan,
        claim_id="routed-collision",
        routed_outcome=outcome,
    )
    assert attachment["plan_digest"] == plan["plan_digest"]
    assert attachment["prediction"] == bundle["prediction"]
    assert attachment["execution_receipt_digest"] == record["execution_receipt_digest"]
    assert attachment["case_binding"]["comparison_case_shape"] == (
        _COMPARISON_CASE_SHAPE
    )
    assert attachment["development_evidence_only"] is True
    assert attachment["plan_mutated"] is False


def test_routed_execution_requires_a_selected_route() -> None:
    decision = _routed_decision()
    abstention = copy.deepcopy(decision)
    abstention["status"] = "abstention"
    with pytest.raises(RoutedExecutionError, match="requires_selected_route"):
        execute_routed_development_stage(
            abstention,
            stage_index=0,
            execution_id="routed-development-execution-blocked",
            **_execution_binding(),
        )
    with pytest.raises(RoutedExecutionError, match="stage_index_invalid"):
        execute_routed_development_stage(
            decision,
            stage_index=5,
            execution_id="routed-development-execution-index",
            **_execution_binding(),
        )


def test_plan_attachment_rejects_receipt_or_route_drift() -> None:
    decision = _routed_decision()
    outcome = execute_routed_development_stage(
        decision,
        stage_index=0,
        execution_id="routed-development-attachment-drift",
        execute=True,
        **_execution_binding(),
    )
    drifted = copy.deepcopy(outcome)
    drifted["record"]["execution_receipt_digest"] = SHA_A
    with pytest.raises(RoutedExecutionError, match="receipt_digest_mismatch"):
        attach_routed_development_evidence(
            _evidence_plan(decision),
            claim_id="routed-collision",
            routed_outcome=drifted,
        )


def _drake_execution_binding(worker_python: Path) -> dict:
    from blueprint_pipeline import measurement_drake_rigid_adapter

    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-geometry-and-contact",
        benchmark_version="routed-development-drake-1",
        method_ids=["drake-1-55"],
        development_split_digest=SHA_DEV,
        qualification_split_digest=SHA_QUAL,
        capture_bundle_digests=[SHA_A],
        robot_controller_digests=[SHA_A],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 1.0,
            "maximum_harmful_false_negative_rate": 1.0,
            "minimum_coverage": 0.0,
        },
        compute_budget={"usd": 0.0},
    )
    case = build_benchmark_case_manifest(
        spec,
        case_id="routed-box-settle-drake",
        split="development",
        input_artifact_digests=[SHA_A],
        task_class="rigid_pick_place",
        material_regime="rigid",
        operating_point={
            "adapter_protocol": measurement_drake_rigid_adapter.PROTOCOL_ID,
            "protocol_family": "rigid_body_drop",
            "body_shape": "box",
            "half_size_m": [0.05, 0.05, 0.05],
            "mass_kg": 0.25,
            "initial_height_m": 0.25,
            "gravity_m_s2": -9.81,
            "timestep_s": 0.002,
            "duration_s": 0.8,
            "friction": [0.8, 0.005, 0.0001],
            "penetration_unsafe_threshold_m": 0.001,
            "comparison_case_shape": _COMPARISON_CASE_SHAPE,
        },
    )
    return {
        "descriptor": build_measurement_adapter_descriptor("drake-1-55"),
        "benchmark_spec": spec,
        "case_manifest": case,
        "worker_argv": [
            str(worker_python), str(measurement_drake_rigid_adapter.WORKER_SCRIPT),
        ],
        "implementation_id": measurement_drake_rigid_adapter.IMPLEMENTATION_ID,
        "implementation_version": measurement_drake_rigid_adapter.IMPLEMENTATION_VERSION,
        "implementation_digest": measurement_drake_rigid_adapter.implementation_digest(),
        "backend_id": "drake-multibody-cpu-sap-point",
        "precision": "float64",
        "seed": 47,
        "solver_settings": {
            "discrete_contact_approximation": "sap",
            "contact_model": "point",
            "penetration_allowance_m": 0.001,
            "stiction_tolerance_m_s": 0.001,
        },
    }


@pytest.mark.external_runtime
@pytest.mark.slow
def test_routed_drake_worker_produces_first_plan_bound_cross_engine_report() -> None:
    raw = os.environ.get("BLUEPRINT_DRAKE_PYTHON", "").strip()
    if not raw or not Path(raw).is_file():
        pytest.skip("BLUEPRINT_DRAKE_PYTHON exact external runtime is not configured")
    worker_python = Path(raw).absolute()
    decision = _routed_decision()
    plan = _evidence_plan(decision)
    mujoco_outcome = execute_routed_development_stage(
        decision,
        stage_index=0,
        execution_id="routed-cross-engine-mujoco",
        execute=True,
        **_execution_binding(),
    )
    drake_outcome = execute_routed_development_stage(
        decision,
        stage_index=0,
        execution_id="routed-cross-engine-drake",
        execute=True,
        **_drake_execution_binding(worker_python),
    )
    attachments = [
        attach_routed_development_evidence(
            plan, claim_id="routed-collision", routed_outcome=outcome
        )
        for outcome in (mujoco_outcome, drake_outcome)
    ]
    report = build_routed_cross_engine_development_report(attachments)
    assert {row["executing_candidate_id"] for row in report["engine_rows"]} == {
        "mujoco-3", "drake-1-55",
    }
    assert report["comparison_case_shape"] == _COMPARISON_CASE_SHAPE
    assert "penetration" in report["numeric_metric_ranges"]
    assert report["development_evidence_only"] is True
    assert report["engine_agreement_is_qualification"] is False
    assert report["physical_accuracy_established"] is False


def _draft_manifest() -> dict:
    return generate_simready_asset_draft(
        build_simready_asset_request(
            request_id="preflight-request",
            object_id="counter-mug",
            bundle_id="capture-1",
            bundle_hash=SHA_A,
            source_references={
                "segmentation_record_id": "record-object_segmentation",
                "mesh_record_id": "record-mesh",
                "provenance_record_id": "provenance-fixture",
            },
            density_class="ceramic_glass",
        ),
        vertices=_BOX_VERTICES,
        faces=_BOX_FACES,
        generated_on="2026-08-02",
    )


def test_scene_preflight_passes_a_sound_draft_without_granting_validity() -> None:
    report = preflight_simready_scene([_draft_manifest()], settle_steps=80)
    assert report["passed"] is True
    row = report["assets"][0]
    assert row["loaded"] is True
    assert row["structural_checks"]["mass_positive"] is True
    assert row["stability"]["state_finite"] is True
    assert row["stability"]["exploded"] is False
    assert report["preflight_pass_means_loadable_and_stable_only"] is True
    assert report["physical_validity_established"] is False
    assert report["qualification_created"] is False


def test_scene_preflight_blocks_a_corrupted_export_with_typed_codes() -> None:
    manifest = _draft_manifest()
    corrupted = copy.deepcopy(manifest)
    corrupted.pop("simready_asset_digest")
    exports = dict(corrupted["exports"])
    mjcf = dict(exports["mjcf"])
    mjcf["content"] = mjcf["content"][: len(mjcf["content"]) // 2]
    import hashlib as _hashlib

    mjcf["content_digest"] = (
        "sha256:" + _hashlib.sha256(mjcf["content"].encode("utf-8")).hexdigest()
    )
    exports["mjcf"] = mjcf
    corrupted["exports"] = exports
    corrupted = validate_simready_asset_manifest(corrupted)
    report = preflight_simready_scene([corrupted])
    assert report["passed"] is False
    row = report["assets"][0]
    assert row["loaded"] is False
    assert any(
        code.startswith("preflight_mjcf_load_failed:") for code in row["failure_codes"]
    )
