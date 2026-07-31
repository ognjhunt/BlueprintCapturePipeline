from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_terminal_report import (
    RECONSTRUCTION_REPORT_REQUEST_SCHEMA_VERSION,
    ReconstructionTerminalReportError,
    build_reconstruction_terminal_report_request,
    generate_reconstruction_terminal_report,
)

from blueprint_pipeline.task_evaluation_supervisor import (
    AutonomyMode,
    SupervisorContext,
    ToolRegistry,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import non_spend_tool_bindings


RECORDED_ARKITSCENES_TERMINAL_REPORT = (
    Path(__file__).parents[1]
    / "docs/evidence/arkitscenes_reconstruction_terminal_report_40958756_27faf763.json"
)


def _ceilings(**overrides: bool) -> dict[str, bool]:
    values = {
        "decoded_observation_availability": True,
        "calibrated_camera_trajectory": True,
        "appearance_reconstruction": False,
        "metric_scale": False,
        "metric_reference_geometry": False,
        "collision_geometry": False,
        "physics_readiness": False,
        "isaac_load_render_compatibility": False,
        "simulator_task_evidence": False,
        "physical_task_success": False,
        "deployment_readiness": False,
    }
    values.update(overrides)
    return values


def _request() -> dict:
    return build_reconstruction_terminal_report_request(
        {
            "schema_version": RECONSTRUCTION_REPORT_REQUEST_SCHEMA_VERSION,
            "stable_run_identity": "arkitscenes-proxy-40958756",
            "original_capture_location": "public-dataset://ARKitScenes/40958756",
            "source_capture_digest": "sha256:" + "1" * 64,
            "implementation_digest": "sha256:" + "4" * 64,
            "input_digests": ["sha256:" + "1" * 64, "sha256:" + "2" * 64],
            "recorded_output_digests": [],
            "validated_capture_profile": "iphone_arkit_lidar_public_proxy",
            "original_customer_request": "Test the local reconstruction intake route.",
            "rights_and_permitted_use": {"status": "cleared", "remote_upload": False},
            "selected_frames": [{"frame_id": "frame-1"}],
            "rejected_frames": [{"frame_id": "frame-2", "reason": "frozen_heldout"}],
            "frozen_split_digest": "sha256:" + "2" * 64,
            "calibration_and_coordinate_status": {
                "status": "dataset_proxy_bound",
                "raw_contract_3_2": False,
            },
            "camera_calibration_binding": {"status": "dataset_proxy_bound"},
            "coordinate_frame_declaration": {
                "frame": "arkit_world",
                "handedness": "right_handed",
            },
            "units_and_metric_scale_status": {
                "declared_units": "meters",
                "independently_validated": False,
            },
            "pose_methods_attempted": [{"method": "recorded_arkit_trajectory"}],
            "registered_observations": ["frame-1"],
            "rejected_observations": ["frame-2"],
            "scale_validation": {"status": "not_independently_validated"},
            "reconstruction_methods_attempted": [],
            "failed_methods": [
                {
                    "method_id": "gaussian_training",
                    "status": "not_executed",
                    "failed_evidence_preserved": True,
                }
            ],
            "skipped_methods": [{"method_id": "isaac", "reason": "appearance_missing"}],
            "recovered_methods": [],
            "appearance_asset": {"status": "missing"},
            "metric_reference_asset": {"status": "sensor_scaffold_only"},
            "collision_candidate": {"status": "missing"},
            "independent_visual_metrics": {"status": "not_executed"},
            "independent_geometric_metrics": {"status": "not_executed"},
            "collider_qualification": {"status": "not_executed"},
            "nurec_openusd_package": {"status": "not_executed"},
            "isaac_verification": {"status": "not_executed"},
            "fixed_camera_render_references": [],
            "physics_collision_verification": {"status": "not_executed"},
            "provider_execution": {"provider": None, "status": "not_used"},
            "provider_runtime_identity": {"provider": "local", "runtime": "python"},
            "source_commit_sha": "3" * 40,
            "container_image_digests": [],
            "runtime_and_spend": {
                "total_runtime_seconds": 0.0,
                "total_spend_usd": 0.0,
            },
            "agent_proposals_and_actions": [
                {"action": "preserve_and_abstain", "proof_effect": "none"}
            ],
            "deterministic_validations": [
                {"validation": "frozen_split_replay", "status": "passed"}
            ],
            "decision": "abstention",
            "evidence_ceilings": _ceilings(),
            "what_could_change_result": ["run pinned appearance trainer"],
            "what_blueprint_cannot_claim": [
                "real Blueprint Raw Contract 3.2 iPhone success",
                "metric geometry",
                "collision readiness",
                "physical success",
                "deployment readiness",
            ],
            "warnings": ["public iPad proxy is not a Blueprint iPhone capture"],
            "blockers": ["appearance_training_not_executed"],
            "teardown_and_provider_zero": {
                "status": "not_applicable_no_provider_allocation",
                "live_provider_inventory": 0,
            },
            "authority_used": {"local_non_spend": True},
            "timestamp": "2026-07-30T19:00:00Z",
        }
    )


def test_terminal_report_replays_recorded_proxy_abstention_without_claim_upgrade() -> None:
    request = _request()
    first = generate_reconstruction_terminal_report(request)
    second = generate_reconstruction_terminal_report(request)
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_terminal_report.v1.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(request)
    validator.validate(first)

    assert first == second
    assert first["decision"] == "abstention"
    assert first["evidence_ceilings"]["calibrated_camera_trajectory"] is True
    assert first["evidence_ceilings"]["appearance_reconstruction"] is False
    assert first["evidence_ceilings"]["metric_scale"] is False
    assert first["evidence_ceilings"]["collision_geometry"] is False
    assert first["evidence_ceilings"]["physical_task_success"] is False
    assert first["agent_output_authoritative"] is False
    assert first["proof_state_mutated_by_report"] is False
    assert first["proof_effect"] == "deterministic_reconstruction_explanation_only"
    assert "abstained" in first["customer_summary"].lower()


def test_terminal_report_rejects_claim_escalation_and_failed_evidence_suppression() -> None:
    escalation = dict(_request())
    escalation.pop("reconstruction_terminal_report_request_digest")
    escalation["evidence_ceilings"] = _ceilings(
        collision_geometry=True,
        physics_readiness=True,
    )
    with pytest.raises(ReconstructionTerminalReportError, match="without_metric_geometry"):
        build_reconstruction_terminal_report_request(escalation)

    suppressed = dict(_request())
    suppressed.pop("reconstruction_terminal_report_request_digest")
    suppressed["failed_methods"] = []
    suppressed["blockers"] = ["failed_attempt_removed"]
    # Removing a failure cannot be inferred from an otherwise self-contained request,
    # so the strict contract requires every supplied failure to carry preservation.
    suppressed["failed_methods"] = [{"failed_evidence_preserved": False}]
    with pytest.raises(ReconstructionTerminalReportError, match="suppressed"):
        build_reconstruction_terminal_report_request(suppressed)


def test_registered_terminal_report_tool_accepts_only_request_digest(tmp_path: Path) -> None:
    request = _request()
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="reconstruction-report-tool",
        customer_question="Explain the final reconstruction status.",
        supervisor_output_dir=str(tmp_path / "run"),
        reconstruction_terminal_report_request=request,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            request["reconstruction_terminal_report_request_digest"]
        ],
    ).to_mapping()
    binding = next(
        binding
        for binding in non_spend_tool_bindings(
            capability="post_run_diagnostician",
            context=context,
            registry=registry,
            authority=authority,
        )
        if binding.tool_id == "generate_reconstruction_report"
    )

    assert set(binding.input_schema["properties"]) == {
        "reconstruction_terminal_report_request_digest"
    }
    observation = binding.invoke(
        {
            "reconstruction_terminal_report_request_digest": request[
                "reconstruction_terminal_report_request_digest"
            ]
        }
    )

    assert observation["status"] == "completed"
    assert observation["typed_result"]["decision"] == "abstention"
    assert observation["typed_result"]["failed_method_count"] == 1
    assert observation["typed_result"]["agent_output_authoritative"] is False
    assert observation["proof_effect"] == "none"


def test_recorded_arkitscenes_terminal_report_abstains_at_exact_missing_gates() -> None:
    report = json.loads(
        RECORDED_ARKITSCENES_TERMINAL_REPORT.read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_terminal_report.v1.schema.json"
        ).read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(report)
    assert report["reconstruction_terminal_report_digest"] == canonical_digest(
        report, digest_field="reconstruction_terminal_report_digest"
    )
    assert report["decision"] == "abstention"
    assert len(report["selected_frames"]) == 40
    assert report["evidence_ceilings"]["decoded_observation_availability"] is True
    assert report["evidence_ceilings"]["calibrated_camera_trajectory"] is True
    for ceiling in (
        "appearance_reconstruction",
        "metric_scale",
        "metric_reference_geometry",
        "collision_geometry",
        "physics_readiness",
        "isaac_load_render_compatibility",
        "simulator_task_evidence",
        "physical_task_success",
        "deployment_readiness",
    ):
        assert report["evidence_ceilings"][ceiling] is False
    assert report["runtime_and_spend"]["total_spend_usd"] == 0.0
    assert report["teardown_and_provider_zero"]["live_provider_inventory"] == 0
    assert report["agent_output_authoritative"] is False
    assert report["proof_state_mutated_by_report"] is False
