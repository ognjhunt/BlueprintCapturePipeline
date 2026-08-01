from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_training_request_compiler import (
    ReconstructionTrainingRequestCompilationError,
    compile_gaussian_training_supervisor_bindings,
    compile_reconstruction_training_request,
)
from blueprint_pipeline.reconstruction_worker_contracts import (
    PINNED_MODEL_ASSETS,
    build_worker_build_receipt,
    build_worker_smoke_receipt,
    build_worker_stack_manifest,
)


D = ["sha256:" + f"{index:x}" * 64 for index in range(1, 10)]
SHA = "a" * 40
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64
REAL_PROXY_PATH = (
    Path(__file__).parents[1] / "docs/evidence/arkitscenes_raw_proxy_40958756_b2d7297f.json"
)
REAL_COLMAP_EXPORT_PATH = (
    Path(__file__).parents[1]
    / "docs/evidence/arkitscenes_colmap_training_dataset_40958756_cb96cbfc.json"
)


def _stack() -> dict:
    return build_worker_stack_manifest(
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


def _worker_receipts(stack: dict) -> tuple[dict, dict]:
    build = build_worker_build_receipt(
        {
            "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
            "status": "built",
            "resolved_image_digest": IMAGE,
            "source_commit_sha": SHA,
            "build_context_digest": D[0],
            "duration_seconds": 300.0,
            "cost_usd": 1.0,
            "logs": [{"artifact_id": "build.log", "digest": D[1]}],
            "blockers": [],
            "scientific_qualification_inferred": False,
        }
    )
    smoke = build_worker_smoke_receipt(
        {
            "build_receipt_digest": build["build_receipt_digest"],
            "resolved_image_digest": IMAGE,
            "source_commit_sha": SHA,
            "provider_runtime_identity": {
                "provider": "vast",
                "runtime": "gpu-canary",
            },
            "status": "passed",
            "checks": [{"check_id": "gpu-runtime", "status": "passed", "output_digest": D[2]}],
            "display_attached": False,
            "scientific_qualification_inferred": False,
        }
    )
    return build, smoke


def _dataset() -> dict:
    value = {
        "schema_version": "colmap_training_dataset_export_result.v1",
        "stable_run_identity": "dataset-export",
        "status": "exported_candidate_only_colmap_text_dataset",
        "source_capture_digest": D[3],
        "source_commit_sha": SHA,
        "frozen_split_digest": D[4],
        "camera_observation_digest": D[5],
        "initialization_surface_digest": D[6],
        "colmap_training_dataset_digest": D[7],
        "relative_path": "candidate-colmap",
        "hidden_heldout_pixels_included": False,
        "trainer_self_grading_permitted": False,
        "raw_input_poses_modified": False,
        "observation_ids": ["frame-1", "frame-2"],
        "rejected_observation_ids": [],
        "coordinate_frame_declaration": {"frame": "arkit-world", "units": "meters"},
        "units": "meters",
        "metric_scale_status": "sensor_metric_unvalidated",
        "warnings": ["metric_scale_independent_validation_required"],
    }
    value["colmap_training_dataset_export_result_digest"] = canonical_digest(
        value, digest_field="colmap_training_dataset_export_result_digest"
    )
    return value


def _capture() -> dict:
    return {
        "source_capture_identity": "capture-1",
        "source_capture_digest": D[3],
        "original_file_references": [{"artifact_id": "capture.mov", "digest": D[8]}],
        "authority_used": {
            "rights_valid": True,
            "provider_upload_authorized": True,
        },
    }


def _pose() -> dict:
    return {
        "binding_kind": "unrefined_camera_observations",
        "pose_result_digest": D[5],
        "source_capture_digest": D[3],
        "train_heldout_split_digest": D[4],
        "raw_capture_poses_modified": False,
    }


def _evaluation() -> dict:
    return {
        "evaluation_contract_digest": D[8],
        "source_capture_digest": D[3],
        "train_heldout_split_digest": D[4],
        "candidate_hidden_pixel_access_permitted": False,
        "candidate_self_grading_permitted": False,
        "split_mutation_permitted": False,
    }


def _configuration() -> dict:
    return {
        "method_profile_id": "nvidia_3dgrut_3dgut_mcmc_v1",
        "camera_model": "PINHOLE",
        "densification_configuration": {"strategy": "mcmc"},
        "random_seed": 23,
        "iteration_budget": 30_000,
        "resource_request": {"gpu_count": 1, "minimum_vram_gb": 24},
        "timeout_seconds": 3600,
        "spend_cap_usd": 10.0,
        "output_contract": {"appearance_asset": "standard_3dgs_ply"},
        "provider_runtime_identity": {"provider": "vast", "runtime": "gpu-canary"},
    }


def _authority() -> dict:
    return {
        "authority_id": "user-explicit-fixture",
        "paid_compute_authorized": True,
        "max_spend_usd": 18.0,
        "hard_ttl_seconds": 7200,
        "retry_cap": 1,
        "provider_processing_authorized": True,
        "provider_upload_authorized": True,
    }


def _compiler_arguments(**overrides) -> dict:
    stack = overrides.pop("worker_stack_manifest", _stack())
    build, smoke = _worker_receipts(stack)
    return {
        "stable_run_identity": "training-run-1",
        "capture_evidence": overrides.pop("capture_evidence", _capture()),
        "dataset_export": overrides.pop("dataset_export", _dataset()),
        "worker_stack_manifest": stack,
        "worker_build_receipt": overrides.pop("worker_build_receipt", build),
        "worker_smoke_receipt": overrides.pop("worker_smoke_receipt", smoke),
        "pose_binding": overrides.pop("pose_binding", _pose()),
        "evaluation_contract": overrides.pop("evaluation_contract", _evaluation()),
        "execution_configuration": overrides.pop("execution_configuration", _configuration()),
        "execution_authority": overrides.pop("execution_authority", _authority()),
        "timestamp": "2026-07-30T23:00:00Z",
        **overrides,
    }


def _compile(**overrides) -> dict:
    return compile_reconstruction_training_request(**_compiler_arguments(**overrides))


def test_compiler_binds_dataset_worker_smoke_authority_and_claim_ceiling() -> None:
    request = _compile()
    assert request["container_image_digest"] == IMAGE
    assert request["reconstruction_dataset_digest"] == D[7]
    assert request["train_heldout_split_digest"] == D[4]
    assert request["calibration_digest"] == D[5]
    assert request["initialization_geometry_digest"] == D[6]
    assert request["candidate_may_read_hidden_heldout"] is False
    assert request["trainer_may_grade_heldout"] is False
    assert request["cost_usd"] == 0.0
    assert request["proof_effect"] == "none"
    assert request["claim_ceiling"] == "request_only"
    assert request["worker_build_receipt_digest"]
    assert request["worker_smoke_receipt_digest"]


def test_compiler_rejects_contract_only_trainer_before_paid_runtime() -> None:
    configuration = _configuration()
    configuration["method_profile_id"] = "gsplat_3dgs_mcmc_v1"
    with pytest.raises(
        ReconstructionTrainingRequestCompilationError,
        match="training_method_not_executable_by_3dgrut_adapter",
    ):
        _compile(execution_configuration=configuration)


def test_compiler_prepares_trusted_digest_only_supervisor_runtime(tmp_path: Path) -> None:
    arguments = _compiler_arguments()
    dataset = arguments["dataset_export"]
    artifact_root = tmp_path / "artifacts"
    candidate = artifact_root / dataset["relative_path"]
    (candidate / "images").mkdir(parents=True)
    (candidate / "images/frame.png").write_bytes(b"image")
    (candidate / "sparse/0").mkdir(parents=True)
    (candidate / "sparse/0/cameras.txt").write_text("camera\n", encoding="utf-8")
    (candidate / "sparse/0/images.txt").write_text("image\n", encoding="utf-8")

    def runner(command, timeout):
        assert timeout == 3600
        values = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in command if "=" in item}
        run_dir = Path(values["out_dir"])
        checkpoint = run_dir / "training/candidate-colmap-fixture/ckpt_last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        Path(values["export_ply.path"]).write_bytes(b"ply\n")
        return subprocess.CompletedProcess(command, 0, stdout="trained", stderr="")

    bindings = compile_gaussian_training_supervisor_bindings(
        compiler_arguments=arguments,
        dataset_export=dataset,
        artifact_root=artifact_root,
        command_runner=runner,
        python_executable="python",
        threedgrut_root="/fixture/3dgrut",
    )
    request = bindings["reconstruction_training_request"]
    result = bindings["gaussian_reconstruction_trainer"](
        request=request,
        output_root=tmp_path / "outputs",
    )
    assert result["status"] == "succeeded"
    assert (
        result["reconstruction_training_request_digest"]
        == request["reconstruction_training_request_digest"]
    )
    assert result["registered_observation_ids"] == ["frame-1", "frame-2"]
    assert result["heldout_labels_included"] is False
    assert result["candidate_self_graded"] is False


def test_compiler_rejects_unaccepted_worker_and_cross_receipt_image_drift() -> None:
    stack = _stack()
    build, smoke = _worker_receipts(stack)
    smoke["resolved_image_digest"] = "registry.example/other@sha256:" + "c" * 64
    smoke["smoke_test_receipt_digest"] = canonical_digest(
        smoke, digest_field="smoke_test_receipt_digest"
    )
    with pytest.raises(
        ReconstructionTrainingRequestCompilationError,
        match="training_worker_smoke_not_accepted",
    ):
        _compile(
            worker_stack_manifest=stack,
            worker_build_receipt=build,
            worker_smoke_receipt=smoke,
        )


def test_compiler_rejects_provider_runtime_claim_not_bound_to_smoke_receipt() -> None:
    configuration = _configuration()
    configuration["provider_runtime_identity"] = {
        "provider": "runpod",
        "runtime": "gpu-canary",
    }
    with pytest.raises(
        ReconstructionTrainingRequestCompilationError,
        match="training_provider_runtime_binding_invalid",
    ):
        _compile(execution_configuration=configuration)


def test_compiler_rejects_hidden_dataset_pose_and_capture_binding_drift() -> None:
    dataset = _dataset()
    dataset["hidden_heldout_pixels_included"] = True
    dataset["source_capture_digest"] = D[2]
    dataset["colmap_training_dataset_export_result_digest"] = canonical_digest(
        dataset, digest_field="colmap_training_dataset_export_result_digest"
    )
    pose = _pose()
    pose["raw_capture_poses_modified"] = True
    with pytest.raises(ReconstructionTrainingRequestCompilationError) as error:
        _compile(dataset_export=dataset, pose_binding=pose)
    assert {
        "training_dataset_candidate_isolation_invalid",
        "training_source_capture_binding_invalid",
        "training_pose_truth_mutation_forbidden",
    } <= set(error.value.codes)


def test_remote_execution_requires_explicit_processing_and_capture_upload_authority() -> None:
    authority = _authority()
    authority["provider_processing_authorized"] = False
    capture = _capture()
    capture["authority_used"]["provider_upload_authorized"] = False
    with pytest.raises(ReconstructionTrainingRequestCompilationError) as error:
        _compile(execution_authority=authority, capture_evidence=capture)
    assert {
        "training_remote_processing_authority_missing",
        "training_capture_provider_upload_authority_missing",
    } <= set(error.value.codes)


def test_recorded_arkitscenes_export_stays_blocked_from_remote_gpu_upload() -> None:
    capture = json.loads(REAL_PROXY_PATH.read_text(encoding="utf-8"))
    dataset = json.loads(REAL_COLMAP_EXPORT_PATH.read_text(encoding="utf-8"))
    pose = {
        "binding_kind": "unrefined_camera_observations",
        "pose_result_digest": dataset["camera_observation_digest"],
        "source_capture_digest": dataset["source_capture_digest"],
        "train_heldout_split_digest": dataset["frozen_split_digest"],
        "raw_capture_poses_modified": False,
    }
    evaluation = {
        "evaluation_contract_digest": D[8],
        "source_capture_digest": dataset["source_capture_digest"],
        "train_heldout_split_digest": dataset["frozen_split_digest"],
        "candidate_hidden_pixel_access_permitted": False,
        "candidate_self_grading_permitted": False,
        "split_mutation_permitted": False,
    }
    with pytest.raises(ReconstructionTrainingRequestCompilationError) as error:
        _compile(
            capture_evidence=capture,
            dataset_export=dataset,
            pose_binding=pose,
            evaluation_contract=evaluation,
        )
    assert "training_capture_provider_upload_authority_missing" in error.value.codes


@pytest.mark.parametrize(
    ("field", "value"),
    [("spend_cap_usd", 19.0), ("timeout_seconds", 7201)],
)
def test_execution_cannot_exceed_explicit_spend_or_ttl(field: str, value: float) -> None:
    configuration = copy.deepcopy(_configuration())
    configuration[field] = value
    with pytest.raises(
        ReconstructionTrainingRequestCompilationError,
        match="training_execution_authority_invalid",
    ):
        _compile(execution_configuration=configuration)
