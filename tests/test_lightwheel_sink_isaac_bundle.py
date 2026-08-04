from __future__ import annotations

import hashlib
import json
import base64
import zipfile
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.lightwheel_sink_isaac_bundle import (
    LightwheelSinkIsaacBundleError,
    compile_lightwheel_sink_isaac_input_bundle,
    derivative_wrapper_usda,
    validate_lightwheel_sink_isaac_input_bundle_receipt,
    validate_lightwheel_sink_isaac_runtime_result,
)
from blueprint_pipeline.lightwheel_sink_isaac_worker import (
    damped_least_squares_delta,
    handle_tangent,
)
from blueprint_pipeline.measurement_isaac_runtime_release import RUNTIME_IMAGE
from blueprint_pipeline.reconstruction_gpu_admission import (
    build_reconstruction_gpu_canary_request,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_derivative_wrapper_references_source_and_authors_required_physics() -> None:
    wrapper = derivative_wrapper_usda()
    assert "@../asset/model.usd@" in wrapper
    assert 'def PhysicsScene "physicsScene"' in wrapper
    assert '"PhysicsArticulationRootAPI"' in wrapper
    assert 'def PhysicsFixedJoint "BlueprintFixedRoot"' in wrapper
    assert "rel physics:body1 = </World/Sink/RootCom>" in wrapper
    assert "RootCom_to_Com_001_0" not in wrapper


def test_bundle_is_deterministic_and_preserves_asset_bytes(tmp_path, monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    model = tmp_path / "model.usd"
    textures = tmp_path / "textures"
    textures.mkdir()
    model.write_bytes(b"PXR-USDC-test-model")
    (textures / "T_BC001.png").write_bytes(b"test-texture")
    monkeypatch.setattr(
        "blueprint_pipeline.lightwheel_sink_isaac_bundle._git_identity",
        lambda _root: "a" * 40,
    )
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_receipt = compile_lightwheel_sink_isaac_input_bundle(
        repo_root=repo,
        model_path=model,
        textures_path=textures,
        output_path=first,
    )
    second_receipt = compile_lightwheel_sink_isaac_input_bundle(
        repo_root=repo,
        model_path=model,
        textures_path=textures,
        output_path=second,
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["input_bundle_digest"] == second_receipt["input_bundle_digest"]
    assert first_receipt["source_model_digest"] == _digest(model)
    with zipfile.ZipFile(first) as archive:
        assert archive.read("asset/model.usd") == model.read_bytes()
        assert archive.read("asset/textures/T_BC001.png") == b"test-texture"
        manifest = json.loads(archive.read("bundle_manifest.json"))
        assert manifest["source_asset_modified"] is False
        assert manifest["paid_execution_authorized_by_bundle"] is False
        assert manifest["bundle_manifest_digest"] == canonical_digest(
            manifest, digest_field="bundle_manifest_digest"
        )


def test_receipt_rejects_tampering(tmp_path, monkeypatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    model = tmp_path / "model.usd"
    textures = tmp_path / "textures"
    textures.mkdir()
    model.write_bytes(b"model")
    (textures / "texture.png").write_bytes(b"texture")
    monkeypatch.setattr(
        "blueprint_pipeline.lightwheel_sink_isaac_bundle._git_identity",
        lambda _root: "b" * 40,
    )
    receipt = compile_lightwheel_sink_isaac_input_bundle(
        repo_root=repo,
        model_path=model,
        textures_path=textures,
        output_path=tmp_path / "bundle.zip",
    )
    receipt["source_model_digest"] = "sha256:" + "0" * 64
    with pytest.raises(LightwheelSinkIsaacBundleError, match="digest_mismatch"):
        validate_lightwheel_sink_isaac_input_bundle_receipt(receipt)


def test_handle_tangent_is_unit_x_axis_rotation_tangent() -> None:
    tangent = handle_tangent([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    assert tangent == pytest.approx([0.0, 0.0, 1.0])


def test_damped_least_squares_delta_is_bounded_and_reduces_error() -> None:
    jacobian = np.eye(3, 7)
    error = np.asarray([0.2, -0.1, 0.05])
    delta = np.asarray(
        damped_least_squares_delta(jacobian, error, damping=0.05, max_norm=0.03)
    )
    assert np.linalg.norm(delta) == pytest.approx(0.03)
    assert np.linalg.norm(error - jacobian @ delta) < np.linalg.norm(error)


def test_external_generated_sink_request_uses_real_asset_bindings() -> None:
    digest = "sha256:" + "1" * 64
    request = build_reconstruction_gpu_canary_request(
        {
            "schema_version": "reconstruction_gpu_canary_request.v1",
            "operation": "measurement_isaac_canary",
            "capture_profile": "external_generated_asset",
            "source_commit_sha": "a" * 40,
            "worker_image_digest": RUNTIME_IMAGE,
            "worker_stack_manifest_digest": digest,
            "deterministic_configuration_digest": digest,
            "operation_request_digest": digest,
            "operation_input_bundle_digest": digest,
            "source_model_digest": digest,
            "texture_manifest_digest": digest,
            "wrapper_digest": digest,
            "test_configuration_digest": digest,
            "expected_runtime_result_schema": "lightwheel_sink_isaac_runtime_result.v1",
            "source_relationship_to_blueprint_raw_capture": "none",
            "external_derived_support_asset": True,
            "blueprint_raw_capture_truth": False,
            "remote_upload_authorized": True,
            "paid_compute_authorized": True,
            "candidate_may_read_hidden_heldout": False,
            "trainer_may_grade_heldout": False,
            "max_spend_usd": 1.0,
            "hard_ttl_seconds": 1800,
            "retry_cap": 0,
            "authority_id": "user-requested-lightwheel-sink-canary",
            "proof_effect": "none",
        }
    )
    assert request["capture_profile"] == "external_generated_asset"
    assert "reconstruction_dataset_digest" not in request
    assert request["request_digest"] == canonical_digest(
        request, digest_field="request_digest"
    )


def test_runtime_result_validator_rehashes_embedded_rgb_frames() -> None:
    digest = "sha256:" + "1" * 64
    payload = b"bounded-png-fixture"
    frame = {
        "png_base64": base64.b64encode(payload).decode("ascii"),
        "png_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }
    gates = {
        key: True
        for key in (
            "source_digests_preserved",
            "wrapper_composed",
            "asset_sweep_completed",
            "asset_sweep_targets_reached",
            "root_anchor_stable",
            "limit_behavior_stable",
            "capsule_contact_observed",
            "capsule_handle_motion_observed",
            "franka_contact_observed",
            "franka_handle_motion_observed",
            "franka_fixed_base_verified",
            "joint_effort_readback_available",
            "numerical_state_finite",
            "omnipbr_materials_present",
            "texture_bindings_present",
            "rgb_frames_valid",
        )
    }
    receipt = {
        "input_bundle_digest": digest,
        "bundle_manifest_digest": digest,
        "source_model_digest": digest,
        "texture_manifest_digest": digest,
        "wrapper_digest": digest,
        "test_configuration_digest": digest,
    }
    bound = {
        "source_commit_sha": "a" * 40,
        "measurement_isaac_runtime_release_digest": digest,
    }
    result = {
        "schema_version": "lightwheel_sink_isaac_runtime_result.v1",
        "status": "passed",
        "source_commit_sha": bound["source_commit_sha"],
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": digest,
        **receipt,
        "runtime": {"rendering": {"frames": [dict(frame) for _ in range(7)]}},
        "gates": gates,
        "blockers": [],
        "development_only": True,
        "external_generated_support_asset": True,
        "blueprint_raw_capture_truth": False,
        "physical_success_established": False,
        "production_route_eligible": False,
        "qualification_created": False,
        "raw_secret_values_recorded": False,
        "proof_effect": "development_execution_only",
        "claim_ceiling": "isaac_articulation_and_scripted_franka_contact_development",
    }
    result["runtime_result_digest"] = canonical_digest(
        result, digest_field="runtime_result_digest"
    )
    assert validate_lightwheel_sink_isaac_runtime_result(
        result, bound_request=bound, bundle_receipt=receipt
    )["status"] == "passed"
    result["runtime"]["rendering"]["frames"][0]["png_base64"] = "dGFtcGVyZWQ="
    result["runtime_result_digest"] = canonical_digest(
        result, digest_field="runtime_result_digest"
    )
    with pytest.raises(LightwheelSinkIsaacBundleError, match="frame_digest_mismatch"):
        validate_lightwheel_sink_isaac_runtime_result(
            result, bound_request=bound, bundle_receipt=receipt
        )
