from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_gpu_operation_bundle import (
    ReconstructionGpuOperationBundleError,
    build_canary_request_from_operation_bundle,
    compile_reconstruction_gpu_operation_bundle,
)
from blueprint_pipeline.reconstruction_worker_contracts import (
    build_pose_estimation_request,
    build_training_request,
)


SHA = "a" * 40
D = ["sha256:" + str(index) * 64 for index in range(1, 7)]
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64


def _common() -> dict:
    return {
        "stable_run_identity": "gpu-operation-run-001",
        "source_capture_identity": "capture-001",
        "source_capture_digest": D[0],
        "original_file_references": [{"artifact_id": "capture.mov", "digest": D[1]}],
        "producing_method": "blueprint.gpu_operation_bundle_test",
        "implementation_version": "1.0.0",
        "container_image_digest": IMAGE,
        "source_commit_sha": SHA,
        "deterministic_configuration_digest": D[2],
        "input_digests": [{"artifact_id": "dataset", "digest": D[3]}],
        "output_digests": [],
        "train_heldout_split_digest": D[4],
        "camera_calibration_binding": {"calibration_digest": D[1]},
        "coordinate_frame_declaration": {"frame": "world", "handedness": "right"},
        "units": "meters",
        "metric_scale_status": "sensor_metric_unvalidated",
        "provider_runtime_identity": {"provider": "vast", "runtime": "candidate"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": {"authority_id": "fixture", "paid_compute": False},
        "warnings": [],
        "blockers": [],
        "proof_effect": "none",
        "claim_ceiling": "request_only",
        "parent_artifact_or_event": {"capture_build_digest": D[0]},
        "timestamp": "2026-08-01T12:00:00Z",
    }


def _pose_request() -> dict:
    return build_pose_estimation_request(
        {
            **_common(),
            "method_profile_id": "colmap_sift_bruteforce_v1",
            "feature_extractor": "SIFT",
            "feature_matcher": "SIFT_BRUTEFORCE",
            "camera_model": "OPENCV",
            "reconstruction_dataset_digest": D[0],
            "camera_rig_digest": D[1],
            "calibration_digest": D[2],
            "model_asset_digest": None,
            "matcher_model_asset_digest": None,
            "deterministic_matching": True,
            "random_seed": 17,
            "resource_request": {"gpu_count": 1, "minimum_vram_gb": 16},
            "timeout_seconds": 1800,
            "spend_cap_usd": 0,
            "candidate_dataset_contains_hidden_heldout_pixels": False,
            "candidate_can_change_split": False,
            "candidate_may_read_hidden_heldout": False,
        }
    )


def _training_request() -> dict:
    return build_training_request(
        {
            **_common(),
            "method_profile_id": "gsplat_3dgut_mcmc_v1",
            "reconstruction_dataset_digest": D[0],
            "calibration_digest": D[1],
            "initialization_geometry_digest": D[2],
            "pose_result_digest": D[3],
            "worker_stack_manifest_digest": D[4],
            "evaluation_contract_digest": D[5],
            "camera_model": "OPENCV",
            "densification_configuration": {"strategy": "mcmc"},
            "random_seed": 23,
            "iteration_budget": 30_000,
            "resource_request": {"gpu_count": 1, "minimum_vram_gb": 24},
            "timeout_seconds": 7200,
            "spend_cap_usd": 0,
            "output_contract": {"appearance_asset": "standard_3dgs_ply"},
            "candidate_dataset_contains_hidden_heldout_pixels": False,
            "candidate_can_change_split": False,
            "candidate_may_read_hidden_heldout": False,
            "trainer_may_grade_heldout": False,
        }
    )


def _write(root: Path, relative: str, payload: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": relative,
        "digest": "sha256:" + __import__("hashlib").sha256(payload).hexdigest(),
        "contains_hidden_heldout_pixels": False,
    }


def _pose_bindings(root: Path) -> list[dict]:
    return [
        {
            **_write(root, "plans/pose-plan.json", b'{"plan":"fixture"}\n'),
            "artifact_id": "pose-plan",
            "role": "pose_execution_plan",
        },
        {
            **_write(root, "images/frame-0001.png", b"candidate-pixels"),
            "artifact_id": "frame-0001",
            "role": "candidate_observation",
        },
    ]


def _training_bindings(root: Path) -> list[dict]:
    return [
        {
            **_write(root, "dataset/export.json", b'{"export":"fixture"}\n'),
            "artifact_id": "dataset-export",
            "role": "dataset_export",
        },
        {
            **_write(root, "dataset/images/frame-0001.png", b"candidate-pixels"),
            "artifact_id": "frame-0001",
            "role": "candidate_dataset_member",
        },
    ]


@pytest.mark.parametrize(
    ("operation", "request_factory", "bindings_factory", "expected_schema"),
    [
        (
            "pose_canary",
            _pose_request,
            _pose_bindings,
            "pose_estimation_result.v1",
        ),
        (
            "trainer_canary",
            _training_request,
            _training_bindings,
            "reconstruction_training_result.v1",
        ),
    ],
)
def test_operation_bundle_is_deterministic_typed_and_non_authorizing(
    tmp_path: Path,
    operation,
    request_factory,
    bindings_factory,
    expected_schema,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bindings = bindings_factory(artifacts)
    first = compile_reconstruction_gpu_operation_bundle(
        operation=operation,
        operation_request=request_factory(),
        artifact_root=artifacts,
        artifact_bindings=bindings,
        output_root=tmp_path / "first",
    )
    second = compile_reconstruction_gpu_operation_bundle(
        operation=operation,
        operation_request=request_factory(),
        artifact_root=artifacts,
        artifact_bindings=list(reversed(bindings)),
        output_root=tmp_path / "second",
    )

    assert first["operation_input_bundle_digest"] == second[
        "operation_input_bundle_digest"
    ]
    assert first["expected_runtime_result_schema"] == expected_schema
    assert first["candidate_may_read_hidden_heldout"] is False
    assert first["trainer_may_grade_heldout"] is False
    assert first["provider_allocation_performed"] is False
    assert first["paid_execution_authorized_by_bundle"] is False
    assert first["proof_effect"] == "none"
    assert first["receipt_digest"] == canonical_digest(
        first, digest_field="receipt_digest"
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_gpu_operation_bundle.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(first, schema)


def test_operation_bundle_rejects_hidden_heldout_and_secret_members(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bindings = _training_bindings(artifacts)
    hidden = _write(artifacts, "hidden_heldout/frame.png", b"forbidden")
    hidden.update(artifact_id="hidden-frame", role="candidate_dataset_member")
    with pytest.raises(
        ReconstructionGpuOperationBundleError,
        match="forbidden_member_name",
    ):
        compile_reconstruction_gpu_operation_bundle(
            operation="trainer_canary",
            operation_request=_training_request(),
            artifact_root=artifacts,
            artifact_bindings=[*bindings, hidden],
            output_root=tmp_path / "out-hidden",
        )

    secret = _write(artifacts, "provider-token.json", b"forbidden")
    secret.update(artifact_id="provider-token", role="candidate_dataset_member")
    with pytest.raises(
        ReconstructionGpuOperationBundleError,
        match="forbidden_member_name",
    ):
        compile_reconstruction_gpu_operation_bundle(
            operation="trainer_canary",
            operation_request=_training_request(),
            artifact_root=artifacts,
            artifact_bindings=[*bindings, secret],
            output_root=tmp_path / "out-secret",
        )


def test_operation_bundle_rejects_symlink_escape_and_digest_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bindings = _pose_bindings(artifacts)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    link = artifacts / "images/escape.png"
    link.symlink_to(outside)
    escape = {
        "artifact_id": "escape",
        "role": "candidate_observation",
        "relative_path": "images/escape.png",
        "digest": "sha256:" + __import__("hashlib").sha256(b"outside").hexdigest(),
        "contains_hidden_heldout_pixels": False,
    }
    with pytest.raises(
        ReconstructionGpuOperationBundleError,
        match="symlink_forbidden",
    ):
        compile_reconstruction_gpu_operation_bundle(
            operation="pose_canary",
            operation_request=_pose_request(),
            artifact_root=artifacts,
            artifact_bindings=[*bindings, escape],
            output_root=tmp_path / "out-symlink",
        )

    drifted = [dict(row) for row in bindings]
    drifted[0]["digest"] = D[0]
    with pytest.raises(
        ReconstructionGpuOperationBundleError,
        match="artifact_digest_mismatch",
    ):
        compile_reconstruction_gpu_operation_bundle(
            operation="pose_canary",
            operation_request=_pose_request(),
            artifact_root=artifacts,
            artifact_bindings=drifted,
            output_root=tmp_path / "out-digest",
        )


def test_operation_bundle_replay_rejects_tampered_archive(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    request = _training_request()
    receipt = compile_reconstruction_gpu_operation_bundle(
        operation="trainer_canary",
        operation_request=request,
        artifact_root=artifacts,
        artifact_bindings=_training_bindings(artifacts),
        output_root=tmp_path / "out",
    )
    archive = (
        tmp_path
        / "out"
        / request["reconstruction_training_request_digest"].removeprefix("sha256:")
        / "reconstruction_gpu_operation_bundle.zip"
    )
    archive.write_bytes(archive.read_bytes() + b"tamper")

    with pytest.raises(
        ReconstructionGpuOperationBundleError,
        match="existing_output_tampered",
    ):
        compile_reconstruction_gpu_operation_bundle(
            operation="trainer_canary",
            operation_request=request,
            artifact_root=artifacts,
            artifact_bindings=_training_bindings(artifacts),
            output_root=tmp_path / "out",
        )
    assert receipt["operation_input_bundle_digest"] != (
        "sha256:" + __import__("hashlib").sha256(archive.read_bytes()).hexdigest()
    )


def test_canary_request_is_derived_from_exact_bundle_receipt(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    request = _training_request()
    receipt = compile_reconstruction_gpu_operation_bundle(
        operation="trainer_canary",
        operation_request=request,
        artifact_root=artifacts,
        artifact_bindings=_training_bindings(artifacts),
        output_root=tmp_path / "out",
    )
    canary = build_canary_request_from_operation_bundle(
        request_fields={
            "schema_version": "reconstruction_gpu_canary_request.v1",
            "capture_profile": "iphone_arkit_lidar",
            "worker_stack_manifest_digest": D[4],
            "deterministic_configuration_digest": D[2],
            "candidate_may_read_hidden_heldout": False,
            "trainer_may_grade_heldout": False,
            "max_spend_usd": 2.0,
            "hard_ttl_seconds": 1800,
            "retry_cap": 1,
            "authority_id": "fixture-human-authority",
            "proof_effect": "none",
        },
        operation_bundle=receipt,
    )

    assert canary["operation_request_digest"] == request[
        "reconstruction_training_request_digest"
    ]
    assert canary["operation_input_bundle_digest"] == receipt[
        "operation_input_bundle_digest"
    ]
    assert canary["expected_runtime_result_schema"] == (
        "reconstruction_training_result.v1"
    )
    assert canary["request_digest"] == canonical_digest(
        canary, digest_field="request_digest"
    )

    tampered = dict(receipt)
    tampered["operation_input_bundle_digest"] = D[0]
    with pytest.raises(
        ReconstructionGpuOperationBundleError,
        match="receipt_digest_mismatch",
    ):
        build_canary_request_from_operation_bundle(
            request_fields={"operation": "trainer_canary"},
            operation_bundle=tampered,
        )
