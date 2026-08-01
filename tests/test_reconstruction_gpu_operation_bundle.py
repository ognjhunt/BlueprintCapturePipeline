from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import zipfile

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_gpu_operation_bundle import (
    ReconstructionGpuOperationBundleError,
    build_canary_request_from_operation_bundle,
    compile_reconstruction_gpu_operation_bundle,
    extract_reconstruction_gpu_operation_bundle,
)
from blueprint_pipeline.reconstruction_gpu_operation_worker import (
    ReconstructionGpuOperationWorkerError,
    execute_reconstruction_gpu_operation_bundle,
)
from blueprint_pipeline.reconstruction_worker_contracts import (
    build_pose_estimation_request,
    build_pose_estimation_result,
    build_training_request,
    build_training_result,
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
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "contains_hidden_heldout_pixels": False,
    }


def _pose_bindings(root: Path) -> list[dict]:
    return [
        {
            **_write(
                root,
                "plans/pose-plan.json",
                json.dumps(
                    {"native_360_colmap_execution_plan_digest": D[5]},
                    sort_keys=True,
                ).encode()
                + b"\n",
            ),
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
        "digest": "sha256:" + hashlib.sha256(b"outside").hexdigest(),
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
        "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
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


def test_bundle_extraction_is_bounded_replayable_and_schema_valid(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    request = _training_request()
    receipt = compile_reconstruction_gpu_operation_bundle(
        operation="trainer_canary",
        operation_request=request,
        artifact_root=artifacts,
        artifact_bindings=_training_bindings(artifacts),
        output_root=tmp_path / "bundles",
    )
    archive = (
        tmp_path
        / "bundles"
        / request["reconstruction_training_request_digest"].removeprefix("sha256:")
        / "reconstruction_gpu_operation_bundle.zip"
    )
    first = extract_reconstruction_gpu_operation_bundle(
        bundle_path=archive,
        bundle_receipt=receipt,
        output_root=tmp_path / "extracted",
    )
    second = extract_reconstruction_gpu_operation_bundle(
        bundle_path=archive,
        bundle_receipt=receipt,
        output_root=tmp_path / "extracted",
    )

    assert first == second
    assert first["status"] == "extracted"
    assert first["candidate_may_read_hidden_heldout"] is False
    assert first["trainer_may_grade_heldout"] is False
    assert first["provider_allocation_inferred"] is False
    assert first["extraction_receipt_digest"] == canonical_digest(
        first, digest_field="extraction_receipt_digest"
    )
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_gpu_operation_bundle_extraction.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(first, schema)

    root = (
        tmp_path
        / "extracted"
        / receipt["operation_input_bundle_digest"].removeprefix("sha256:")
    )
    (root / "inputs/dataset/images/frame-0001.png").write_bytes(b"tampered")
    with pytest.raises(
        ReconstructionGpuOperationBundleError,
        match="extraction_member_tampered",
    ):
        extract_reconstruction_gpu_operation_bundle(
            bundle_path=archive,
            bundle_receipt=receipt,
            output_root=tmp_path / "extracted",
        )


@pytest.mark.parametrize("attack", ["traversal", "symlink", "compressed"])
def test_bundle_extraction_rejects_adversarial_zip_metadata(
    tmp_path: Path, attack: str
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    request = _pose_request()
    receipt = compile_reconstruction_gpu_operation_bundle(
        operation="pose_canary",
        operation_request=request,
        artifact_root=artifacts,
        artifact_bindings=_pose_bindings(artifacts),
        output_root=tmp_path / "bundles",
    )
    source = (
        tmp_path
        / "bundles"
        / request["pose_estimation_request_digest"].removeprefix("sha256:")
        / "reconstruction_gpu_operation_bundle.zip"
    )
    malicious = tmp_path / f"{attack}.zip"
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(
        malicious, "w"
    ) as rewritten:
        for member in original.infolist():
            payload = original.read(member.filename)
            if attack == "compressed" and member.filename == "operation_request.json":
                rewritten.writestr(
                    member.filename,
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            else:
                rewritten.writestr(member, payload)
        if attack == "traversal":
            rewritten.writestr("../escape.txt", b"escape")
        elif attack == "symlink":
            info = zipfile.ZipInfo("inputs/symlink")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            rewritten.writestr(info, b"../../escape")
    drifted = dict(receipt)
    drifted["operation_input_bundle_digest"] = (
        "sha256:" + hashlib.sha256(malicious.read_bytes()).hexdigest()
    )
    drifted["receipt_digest"] = canonical_digest(
        drifted, digest_field="receipt_digest"
    )

    with pytest.raises(
        ReconstructionGpuOperationBundleError,
        match="archive_(inventory_invalid|member_unsafe)",
    ):
        extract_reconstruction_gpu_operation_bundle(
            bundle_path=malicious,
            bundle_receipt=drifted,
            output_root=tmp_path / "extract-attack",
        )


def _result_common(request: dict, *, output_digests: list[dict]) -> dict:
    return {
        "stable_run_identity": request["stable_run_identity"],
        "source_capture_identity": request["source_capture_identity"],
        "source_capture_digest": request["source_capture_digest"],
        "original_file_references": request["original_file_references"],
        "producing_method": "fixture-gpu-operation-worker",
        "implementation_version": "1.0.0",
        "container_image_digest": request["container_image_digest"],
        "source_commit_sha": request["source_commit_sha"],
        "deterministic_configuration_digest": request[
            "deterministic_configuration_digest"
        ],
        "input_digests": request["input_digests"],
        "output_digests": output_digests,
        "train_heldout_split_digest": request["train_heldout_split_digest"],
        "camera_calibration_binding": request["camera_calibration_binding"],
        "coordinate_frame_declaration": request["coordinate_frame_declaration"],
        "units": request["units"],
        "metric_scale_status": request["metric_scale_status"],
        "provider_runtime_identity": request["provider_runtime_identity"],
        "cost_usd": 0.0,
        "duration_seconds": 1.0,
        "authority_used": request["authority_used"],
        "warnings": [],
        "blockers": [],
        "parent_artifact_or_event": {
            "operation_request_digest": request.get("pose_estimation_request_digest")
            or request.get("reconstruction_training_request_digest")
        },
        "timestamp": request["timestamp"],
    }


@pytest.mark.parametrize("operation", ["pose_canary", "trainer_canary"])
def test_operation_worker_dispatches_only_typed_pose_or_trainer(
    tmp_path: Path, operation: str
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    if operation == "pose_canary":
        request = _pose_request()
        bindings = _pose_bindings(artifacts)
    else:
        request = _training_request()
        bindings = _training_bindings(artifacts)
    receipt = compile_reconstruction_gpu_operation_bundle(
        operation=operation,
        operation_request=request,
        artifact_root=artifacts,
        artifact_bindings=bindings,
        output_root=tmp_path / "bundles",
    )
    request_digest = request.get("pose_estimation_request_digest") or request.get(
        "reconstruction_training_request_digest"
    )
    archive = (
        tmp_path
        / "bundles"
        / str(request_digest).removeprefix("sha256:")
        / "reconstruction_gpu_operation_bundle.zip"
    )

    def pose_executor(**kwargs):
        assert kwargs["input_root"].joinpath("images/frame-0001.png").is_file()
        root = kwargs["artifact_root"] / ("native_colmap_execution_" + D[5][7:23])
        artifact = root / "workspace/model.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"pose-output")
        return build_pose_estimation_result(
            {
                **_result_common(
                    request,
                    output_digests=[
                        {
                            "artifact_id": "workspace/model.txt",
                            "digest": "sha256:"
                            + hashlib.sha256(b"pose-output").hexdigest(),
                        }
                    ],
                ),
                "pose_estimation_request_digest": request[
                    "pose_estimation_request_digest"
                ],
                "status": "succeeded",
                "failure_code": None,
                "registered_observation_ids": ["frame-0001"],
                "rejected_observation_ids": [],
                "heldout_labels_included": False,
                "candidate_self_graded": False,
                "proof_effect": "calibrated_trajectory_candidate_only",
                "claim_ceiling": "calibrated_camera_trajectory",
                "legal_next_actions": ["request_metric_anchor"],
            }
        )

    def trainer_executor(**kwargs):
        assert kwargs["artifact_root"].joinpath(
            "dataset/images/frame-0001.png"
        ).is_file()
        root = kwargs["output_root"] / request[
            "reconstruction_training_request_digest"
        ][7:23]
        root.mkdir(parents=True)
        outputs = {
            "training.log": b"training-log",
            "appearance_candidate.ply": b"ply-output",
            "checkpoint_last.pt": b"checkpoint",
        }
        for relative, payload in outputs.items():
            (root / relative).write_bytes(payload)
        return build_training_result(
            {
                **_result_common(
                    request,
                    output_digests=[
                        {
                            "artifact_id": name,
                            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                        }
                        for name, payload in outputs.items()
                        if name != "checkpoint_last.pt"
                    ],
                ),
                "reconstruction_training_request_digest": request[
                    "reconstruction_training_request_digest"
                ],
                "status": "succeeded",
                "failure_code": None,
                "checkpoint_references": [
                    {
                        "artifact_id": "checkpoint_last.pt",
                        "digest": "sha256:"
                        + hashlib.sha256(outputs["checkpoint_last.pt"]).hexdigest(),
                    }
                ],
                "training_metrics": {"heldout_metrics_computed": False},
                "heldout_labels_included": False,
                "candidate_self_graded": False,
                "registered_observation_ids": ["frame-0001"],
                "rejected_observation_ids": [],
                "peak_resource_use": {"gpu_count": 1},
                "legal_next_actions": ["preserve_evidence_and_stop"],
                "proof_effect": "appearance_asset_candidate_only",
                "claim_ceiling": "appearance_reconstruction",
            }
        )

    result = execute_reconstruction_gpu_operation_bundle(
        bundle_path=archive,
        bundle_receipt=receipt,
        materialization_root=tmp_path / "materialized",
        output_root=tmp_path / "results",
        pose_executor=pose_executor,
        trainer_executor=trainer_executor,
    )

    assert result["schema_version"] == receipt["expected_runtime_result_schema"]
    assert result["heldout_labels_included"] is False
    assert result["candidate_self_graded"] is False


def test_operation_worker_rejects_unbound_result_before_acceptance(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    request = _training_request()
    receipt = compile_reconstruction_gpu_operation_bundle(
        operation="trainer_canary",
        operation_request=request,
        artifact_root=artifacts,
        artifact_bindings=_training_bindings(artifacts),
        output_root=tmp_path / "bundles",
    )
    archive = (
        tmp_path
        / "bundles"
        / request["reconstruction_training_request_digest"].removeprefix("sha256:")
        / "reconstruction_gpu_operation_bundle.zip"
    )

    with pytest.raises(
        ReconstructionGpuOperationWorkerError,
        match="result_invalid",
    ):
        execute_reconstruction_gpu_operation_bundle(
            bundle_path=archive,
            bundle_receipt=receipt,
            materialization_root=tmp_path / "materialized",
            output_root=tmp_path / "results",
            trainer_executor=lambda **_kwargs: {
                **_result_common(request, output_digests=[]),
                "reconstruction_training_request_digest": D[0],
                "status": "succeeded",
                "failure_code": None,
                "checkpoint_references": [],
                "training_metrics": {},
                "heldout_labels_included": False,
                "candidate_self_graded": False,
                "registered_observation_ids": [],
                "rejected_observation_ids": [],
                "peak_resource_use": {},
                "legal_next_actions": [],
                "proof_effect": "appearance_asset_candidate_only",
                "claim_ceiling": "appearance_reconstruction",
            },
        )
