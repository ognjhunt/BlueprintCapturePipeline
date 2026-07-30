from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_validation_contracts import (
    CAMERA_RIG_VALIDATION_REQUEST_SCHEMA_VERSION,
    METRIC_SCALE_ANCHOR_SCHEMA_VERSION,
    METRIC_SCALE_VALIDATION_REQUEST_SCHEMA_VERSION,
    ReconstructionValidationContractError,
    build_camera_rig_validation_request,
    build_metric_scale_anchor,
    build_metric_scale_validation_request,
    validate_camera_rig,
    validate_metric_scale,
)
from blueprint_pipeline.task_evaluation_supervisor import (
    AutonomyMode,
    SupervisorContext,
    ToolRegistry,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import default_authority_envelope
from blueprint_pipeline.task_evaluation_supervisor.tools import non_spend_tool_bindings


CAPTURE_DIGEST = "sha256:" + "1" * 64


def _camera_rig_request(*, synchronized: bool = True) -> dict:
    rig = {
        "schema_version": "camera_360_rig_declaration.v1",
        "capture_digest": CAPTURE_DIGEST,
        "calibration_status": "valid",
        "rig_is_fixed": True,
        "blockers": [],
    }
    rig["rig_declaration_digest"] = canonical_digest(
        rig, digest_field="rig_declaration_digest"
    )
    binding = {
        "schema_version": "dual_fisheye_stream_binding.v1",
        "capture_digest": CAPTURE_DIGEST,
        "all_segments_synchronized": synchronized,
        "original_distorted_pixels_preserved": True,
        "blockers": [] if synchronized else ["native_360_lens_streams_unsynchronized"],
    }
    binding["dual_fisheye_binding_digest"] = canonical_digest(
        binding, digest_field="dual_fisheye_binding_digest"
    )
    return build_camera_rig_validation_request(
        {
            "schema_version": CAMERA_RIG_VALIDATION_REQUEST_SCHEMA_VERSION,
            "source_capture_digest": CAPTURE_DIGEST,
            "native_360_normalization_digest": "sha256:" + "2" * 64,
            "rig_declaration": rig,
            "dual_fisheye_binding": binding,
            "agent_may_change_calibration": False,
            "timestamp": "2026-07-30T20:00:00Z",
        }
    )


def _scale_request(*, estimated: float = 2.01, learned_only: bool = False) -> dict:
    anchor = build_metric_scale_anchor(
        {
            "schema_version": METRIC_SCALE_ANCHOR_SCHEMA_VERSION,
            "source_capture_digest": CAPTURE_DIGEST,
            "anchor_id": "site-wall-1",
            "anchor_type": "measured_site_anchor",
            "measured_distance_m": 2.0,
            "evidence_digest": "sha256:" + "3" * 64,
            "independently_verified": True,
            "learned_or_monocular_depth_only": learned_only,
            "coordinate_frame_declaration": {"frame": "capture_world", "units": "meters"},
        }
    )
    return build_metric_scale_validation_request(
        {
            "schema_version": METRIC_SCALE_VALIDATION_REQUEST_SCHEMA_VERSION,
            "source_capture_digest": CAPTURE_DIGEST,
            "reconstruction_result_digest": "sha256:" + "4" * 64,
            "frozen_split_digest": "sha256:" + "5" * 64,
            "anchor": anchor,
            "estimated_anchor_distance_units": estimated,
            "maximum_relative_error": 0.02,
            "threshold_frozen_before_validation": True,
            "candidate_may_change_anchor": False,
            "timestamp": "2026-07-30T20:00:00Z",
        }
    )


def test_camera_rig_validation_accepts_only_fixed_calibrated_synchronized_rig() -> None:
    request = _camera_rig_request()
    accepted = validate_camera_rig(request)
    rejected = validate_camera_rig(_camera_rig_request(synchronized=False))

    assert accepted["status"] == "validated"
    assert accepted["claim_ceiling"] == "calibrated_camera_rig"
    assert accepted["metric_scale_proven"] is False
    assert accepted["camera_trajectory_proven"] is False
    assert rejected["status"] == "rejected"
    assert rejected["claim_ceiling"] == "decoded_native_container"
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_camera_rig_and_scale_validation.v1.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(request)
    validator.validate(accepted)


def test_metric_scale_requires_independent_anchor_and_frozen_residual_threshold() -> None:
    request = _scale_request()
    accepted = validate_metric_scale(request)
    rejected = validate_metric_scale(_scale_request(estimated=2.2))

    assert accepted["status"] == "validated"
    assert accepted["relative_error"] == pytest.approx(0.005)
    assert accepted["claim_ceiling"] == "metric_scale"
    assert accepted["learned_or_monocular_depth_established_scale"] is False
    assert rejected["status"] == "rejected"
    assert rejected["blockers"] == ["scale_anchor_rejection"]
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_camera_rig_and_scale_validation.v1.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(request["anchor"])
    validator.validate(request)
    validator.validate(accepted)

    with pytest.raises(ReconstructionValidationContractError, match="independent_evidence"):
        _scale_request(learned_only=True)


def test_registered_rig_and_scale_tools_are_digest_only_and_non_authoritative(
    tmp_path: Path,
) -> None:
    rig_request = _camera_rig_request()
    scale_request = _scale_request()
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="reconstruction-validation-tools",
        customer_question="Validate camera rig and metric scale.",
        supervisor_output_dir=str(tmp_path / "run"),
        camera_rig_validation_request=rig_request,
        metric_scale_validation_request=scale_request,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            rig_request["camera_rig_validation_request_digest"],
            scale_request["metric_scale_validation_request_digest"],
        ],
    ).to_mapping()
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }
    rig_observation = bindings["validate_camera_rig"].invoke(
        {
            "camera_rig_validation_request_digest": rig_request[
                "camera_rig_validation_request_digest"
            ]
        }
    )
    scale_observation = bindings["validate_metric_scale"].invoke(
        {
            "metric_scale_validation_request_digest": scale_request[
                "metric_scale_validation_request_digest"
            ]
        }
    )

    assert rig_observation["status"] == "completed"
    assert rig_observation["typed_result"]["metric_scale_proven"] is False
    assert scale_observation["status"] == "completed"
    assert scale_observation["typed_result"]["status"] == "validated"
    assert scale_observation["typed_result"]["agent_changed_anchor_or_threshold"] is False
    assert rig_observation["proof_effect"] == scale_observation["proof_effect"] == "none"
