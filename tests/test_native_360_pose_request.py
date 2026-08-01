from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_360_pose_request import (
    Native360PoseRequestCompilationError,
    compile_native_360_pose_estimation_request,
)
from blueprint_pipeline.reconstruction_validation_contracts import (
    build_camera_rig_validation_request,
    validate_camera_rig,
)
from blueprint_pipeline.reconstruction_worker_contracts import (
    PINNED_MODEL_ASSETS,
    build_worker_build_receipt,
    build_worker_smoke_receipt,
    build_worker_stack_manifest,
)


CAPTURE_DIGEST = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
D3 = "sha256:" + "3" * 64
D4 = "sha256:" + "4" * 64
SHA = "a" * 40
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64


def _rig(*, timeline_valid: bool = True) -> tuple[dict, dict]:
    declaration = {
        "schema_version": "camera_360_rig_declaration.v1",
        "capture_digest": CAPTURE_DIGEST,
        "calibration_status": "valid",
        "rig_is_fixed": True,
        "blockers": [],
    }
    declaration["rig_declaration_digest"] = canonical_digest(
        declaration, digest_field="rig_declaration_digest"
    )
    binding = {
        "schema_version": "dual_fisheye_stream_binding.v1",
        "capture_digest": CAPTURE_DIGEST,
        "all_segments_synchronized": True,
        "capture_timeline_valid": timeline_valid,
        "original_distorted_pixels_preserved": True,
        "blockers": [],
    }
    binding["dual_fisheye_binding_digest"] = canonical_digest(
        binding, digest_field="dual_fisheye_binding_digest"
    )
    request = build_camera_rig_validation_request(
        {
            "schema_version": "camera_rig_validation_request.v1",
            "source_capture_digest": CAPTURE_DIGEST,
            "native_360_normalization_digest": D2,
            "rig_declaration": declaration,
            "dual_fisheye_binding": binding,
            "agent_may_change_calibration": False,
            "timestamp": "2026-08-01T12:00:00Z",
        }
    )
    return request, validate_camera_rig(request)


def _dataset(rig_result: dict, **overrides: object) -> dict:
    value = {
        "schema_version": "reconstruction_dataset_manifest.v1",
        "source_capture_identity": "native-capture-1",
        "source_capture_digest": CAPTURE_DIGEST,
        "original_file_references": [
            {"relative_path": "native/capture.insv", "digest": D3},
            {"relative_path": "native/capture_001.insv", "digest": D4},
        ],
        "capture_authority_profile": "camera_360_native",
        "train_heldout_split_digest": D2,
        "candidate_dataset_contains_hidden_heldout_pixels": False,
        "candidate_can_modify_split": False,
        "raw_capture_bytes_remain_authoritative": True,
        "camera_calibration_binding": {
            "camera_360_rig_declaration_digest": rig_result[
                "rig_declaration_digest"
            ]
        },
        "stream_metadata": {
            "dual_fisheye_binding_digest": rig_result[
                "dual_fisheye_binding_digest"
            ]
        },
        "coordinate_frame_declaration": {
            "units": "meters",
            "rig_frame": "front_lens_optical_center",
        },
        "authority_used": {
            "provider_upload_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    value.update(overrides)
    value["dataset_manifest_digest"] = canonical_digest(
        value, digest_field="dataset_manifest_digest"
    )
    return value


def _worker() -> tuple[dict, dict, dict]:
    stack = build_worker_stack_manifest(
        {
            "worker_family": "blueprint-reconstruction-worker",
            "runnable_platform": "linux/amd64",
            "headless_required": True,
            "display_required": False,
            "source_commit_sha": SHA,
            "qualification_status": "candidate_unbuilt",
            "minimum_vram_gb": 24,
            "supported_compute_capabilities": [75, 80, 86, 89],
            "tested_driver_range": {"status": "not_yet_tested"},
            "model_assets": list(PINNED_MODEL_ASSETS),
            "hidden_heldout_access": False,
            "trainer_self_grading": False,
        }
    )
    build = build_worker_build_receipt(
        {
            "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
            "status": "built",
            "resolved_image_digest": IMAGE,
            "source_commit_sha": SHA,
            "build_context_digest": D2,
            "duration_seconds": 10.0,
            "cost_usd": 0.0,
            "logs": [{"artifact_id": "build.log", "digest": D3}],
            "blockers": [],
            "scientific_qualification_inferred": False,
        }
    )
    smoke = build_worker_smoke_receipt(
        {
            "build_receipt_digest": build["build_receipt_digest"],
            "resolved_image_digest": IMAGE,
            "source_commit_sha": SHA,
            "provider_runtime_identity": {"provider": "local", "runtime": "fixture"},
            "status": "passed",
            "checks": [
                {"check_id": "colmap-headless", "status": "passed", "output_digest": D4}
            ],
            "display_attached": False,
            "scientific_qualification_inferred": False,
        }
    )
    return stack, build, smoke


def _compile(**overrides: object) -> dict:
    rig_request, rig_result = _rig()
    stack, build, smoke = _worker()
    arguments = {
        "stable_run_identity": "native-pose-request-1",
        "reconstruction_dataset": _dataset(rig_result),
        "camera_rig_validation_request": rig_request,
        "camera_rig_validation_result": rig_result,
        "worker_stack_manifest": stack,
        "worker_build_receipt": build,
        "worker_smoke_receipt": smoke,
        "execution_configuration": {
            "provider_runtime_identity": {"provider": "local", "runtime": "fixture"},
            "method_profile_id": "colmap_sift_bruteforce_v1",
            "feature_extractor": "SIFT",
            "feature_matcher": "SIFT_BRUTEFORCE",
            "camera_model": "OPENCV_FISHEYE",
            "model_asset_digest": None,
            "matcher_model_asset_digest": None,
            "random_seed": 17,
            "resource_request": {"gpu_count": 1, "minimum_vram_gb": 16},
            "timeout_seconds": 900,
            "spend_cap_usd": 0.0,
        },
        "execution_authority": {
            "authority_id": "local-fixture-authority",
            "max_spend_usd": 0.0,
            "hard_ttl_seconds": 1200,
            "retry_cap": 0,
            "paid_compute_authorized": False,
            "provider_processing_authorized": False,
            "provider_upload_authorized": False,
        },
        "timestamp": "2026-08-01T12:00:00Z",
    }
    arguments.update(overrides)
    return compile_native_360_pose_estimation_request(**arguments)


def test_native_pose_request_is_replayable_and_binds_all_native_evidence() -> None:
    first = _compile()
    second = _compile()

    assert first == second
    assert first["camera_model"] == "OPENCV_FISHEYE"
    assert first["metric_scale_status"] == "anchor_required"
    assert first["candidate_may_read_hidden_heldout"] is False
    assert len(first["original_file_references"]) == 2
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/pose_estimation_request.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(first, schema)


def test_native_pose_request_rejects_hidden_pixels_and_invalid_timeline() -> None:
    rig_request, rig_result = _rig()
    hidden = _dataset(
        rig_result, candidate_dataset_contains_hidden_heldout_pixels=True
    )
    with pytest.raises(
        Native360PoseRequestCompilationError,
        match="native_pose_dataset_isolation_invalid",
    ):
        _compile(reconstruction_dataset=hidden)

    invalid_request, invalid_result = _rig(timeline_valid=False)
    with pytest.raises(
        Native360PoseRequestCompilationError,
        match="native_pose_camera_rig_not_accepted",
    ):
        _compile(
            camera_rig_validation_request=invalid_request,
            camera_rig_validation_result=invalid_result,
            reconstruction_dataset=_dataset(invalid_result),
        )


def test_native_pose_request_rejects_remote_execution_without_capture_authority() -> None:
    configuration = copy.deepcopy(_compile()["provider_runtime_identity"])
    assert configuration["provider"] == "local"
    remote_configuration = {
        "provider_runtime_identity": {"provider": "vast", "runtime": "gpu-canary"},
        "method_profile_id": "colmap_sift_bruteforce_v1",
        "feature_extractor": "SIFT",
        "feature_matcher": "SIFT_BRUTEFORCE",
        "camera_model": "OPENCV_FISHEYE",
        "model_asset_digest": None,
        "matcher_model_asset_digest": None,
        "random_seed": 17,
        "resource_request": {"gpu_count": 1, "minimum_vram_gb": 16},
        "timeout_seconds": 900,
        "spend_cap_usd": 1.0,
    }
    stack, build, smoke = _worker()
    smoke = build_worker_smoke_receipt(
        {
            "build_receipt_digest": build["build_receipt_digest"],
            "resolved_image_digest": IMAGE,
            "source_commit_sha": SHA,
            "provider_runtime_identity": {"provider": "vast", "runtime": "gpu-canary"},
            "status": "passed",
            "checks": [
                {"check_id": "colmap-headless", "status": "passed", "output_digest": D4}
            ],
            "display_attached": False,
            "scientific_qualification_inferred": False,
        }
    )
    with pytest.raises(
        Native360PoseRequestCompilationError,
        match="native_pose_remote_authority_missing",
    ):
        _compile(
            worker_stack_manifest=stack,
            worker_build_receipt=build,
            worker_smoke_receipt=smoke,
            execution_configuration=remote_configuration,
            execution_authority={
                "authority_id": "remote-authority",
                "max_spend_usd": 2.0,
                "hard_ttl_seconds": 1200,
                "retry_cap": 1,
                "paid_compute_authorized": True,
                "provider_processing_authorized": True,
                "provider_upload_authorized": True,
            },
        )


def test_native_pose_request_rejects_unpinned_learned_models() -> None:
    request = _compile()
    configuration = {
        "provider_runtime_identity": request["provider_runtime_identity"],
        "method_profile_id": "colmap_aliked_lightglue_v1",
        "feature_extractor": "ALIKED_N16ROT",
        "feature_matcher": "ALIKED_LIGHTGLUE",
        "camera_model": "OPENCV_FISHEYE",
        "model_asset_digest": "sha256:" + "9" * 64,
        "matcher_model_asset_digest": PINNED_MODEL_ASSETS[1]["digest"],
        "random_seed": 17,
        "resource_request": {"gpu_count": 1, "minimum_vram_gb": 16},
        "timeout_seconds": 900,
        "spend_cap_usd": 0.0,
    }
    with pytest.raises(
        Native360PoseRequestCompilationError,
        match="native_pose_feature_model_not_pinned",
    ):
        _compile(execution_configuration=configuration)
