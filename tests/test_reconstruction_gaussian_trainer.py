from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.reconstruction_gaussian_trainer import (
    GaussianTrainerRuntimeError,
    bind_gaussian_reconstruction_trainer,
    run_gaussian_reconstruction_training,
)
from blueprint_pipeline.reconstruction_worker_contracts import build_training_request


D = ["sha256:" + str(index) * 64 for index in range(1, 7)]
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "a" * 64


def _request() -> dict:
    return build_training_request(
        {
            "stable_run_identity": "training-fixture",
            "source_capture_identity": "capture-fixture",
            "source_capture_digest": D[0],
            "original_file_references": [{"artifact_id": "capture.mov", "digest": D[1]}],
            "producing_method": "fixture-request",
            "implementation_version": "1.0.0",
            "container_image_digest": IMAGE,
            "source_commit_sha": "b" * 40,
            "deterministic_configuration_digest": D[2],
            "input_digests": [{"artifact_id": "dataset", "digest": D[3]}],
            "output_digests": [],
            "train_heldout_split_digest": D[4],
            "camera_calibration_binding": {"calibration_digest": D[1]},
            "coordinate_frame_declaration": {"frame": "world"},
            "units": "meters",
            "metric_scale_status": "sensor_metric_unvalidated",
            "provider_runtime_identity": {"provider": "vast", "runtime": "candidate"},
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "authority_used": {"authority_id": "fixture"},
            "warnings": [],
            "blockers": [],
            "proof_effect": "none",
            "claim_ceiling": "request_only",
            "parent_artifact_or_event": {"dataset_export_digest": D[5]},
            "timestamp": "2026-07-30T23:00:00Z",
            "method_profile_id": "gsplat_3dgut_mcmc_v1",
            "reconstruction_dataset_digest": D[3],
            "calibration_digest": D[1],
            "initialization_geometry_digest": D[2],
            "pose_result_digest": D[3],
            "worker_stack_manifest_digest": D[4],
            "evaluation_contract_digest": D[5],
            "camera_model": "PINHOLE",
            "densification_configuration": {"strategy": "mcmc"},
            "random_seed": 23,
            "iteration_budget": 30,
            "resource_request": {"gpu_count": 1, "minimum_vram_gb": 24},
            "timeout_seconds": 60,
            "spend_cap_usd": 1.0,
            "output_contract": {"appearance_asset": "standard_3dgs_ply"},
            "candidate_dataset_contains_hidden_heldout_pixels": False,
            "candidate_can_change_split": False,
            "candidate_may_read_hidden_heldout": False,
            "trainer_may_grade_heldout": False,
        }
    )


def _dataset(root: Path) -> dict:
    dataset = root / "candidate_colmap"
    (dataset / "images").mkdir(parents=True)
    (dataset / "images/frame.png").write_bytes(b"png")
    (dataset / "sparse/0").mkdir(parents=True)
    (dataset / "sparse/0/cameras.txt").write_text("camera\n")
    (dataset / "sparse/0/images.txt").write_text("image\n")
    return {
        "schema_version": "colmap_training_dataset_export_result.v1",
        "colmap_training_dataset_digest": D[3],
        "relative_path": "candidate_colmap",
        "hidden_heldout_pixels_included": False,
        "trainer_self_grading_permitted": False,
        "observation_ids": ["frame-1"],
        "rejected_observation_ids": [],
    }


def _successful_runner(command, timeout):
    assert timeout == 60
    values = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in command if "=" in item}
    run_dir = Path(values["out_dir"])
    upstream_run = run_dir / "training/candidate_colmap-3007_010203"
    upstream_run.mkdir(parents=True)
    (upstream_run / "ckpt_last.pt").write_bytes(b"checkpoint")
    Path(values["export_ply.path"]).write_bytes(b"ply\n")
    assert "dataset.test_split_interval=-1" in command
    assert "test_last=false" in command
    assert "compute_extra_metrics=false" in command
    return subprocess.CompletedProcess(command, 0, stdout="trained", stderr="")


def test_trainer_runs_pinned_candidate_only_command_and_hashes_outputs(tmp_path: Path) -> None:
    calls = 0
    dataset_export = _dataset(tmp_path / "artifacts")

    def runner(command, timeout):
        nonlocal calls
        calls += 1
        return _successful_runner(command, timeout)

    result = run_gaussian_reconstruction_training(
        training_request=_request(),
        dataset_export=dataset_export,
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
        command_runner=runner,
        python_executable="python",
        threedgrut_root="/fixture/3dgrut",
    )
    assert result["status"] == "succeeded"
    assert result["failure_code"] is None
    assert result["registered_observation_ids"] == ["frame-1"]
    assert result["checkpoint_references"]
    assert result["checkpoint_references"][0]["artifact_id"] == "checkpoint_last.pt"
    assert result["heldout_labels_included"] is False
    assert result["candidate_self_graded"] is False
    assert "heldout_metrics" not in result
    replay = run_gaussian_reconstruction_training(
        training_request=_request(),
        dataset_export=dataset_export,
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
        command_runner=runner,
        python_executable="python",
        threedgrut_root="/fixture/3dgrut",
    )
    assert replay == result
    assert calls == 1

    run_dir = tmp_path / "outputs" / _request()["reconstruction_training_request_digest"][7:23]
    (run_dir / "appearance_candidate.ply").write_bytes(b"tampered")
    with pytest.raises(GaussianTrainerRuntimeError, match="artifact_digest_mismatch"):
        run_gaussian_reconstruction_training(
            training_request=_request(),
            dataset_export=dataset_export,
            artifact_root=tmp_path / "artifacts",
            output_root=tmp_path / "outputs",
            command_runner=runner,
        )
    assert calls == 1


@pytest.mark.parametrize(
    ("behavior", "expected_status", "expected_failure"),
    [
        ("oom", "failed", "gpu_out_of_memory"),
        ("nan", "failed", "nan_output"),
        ("timeout", "timed_out", "training_timeout"),
        ("missing", "failed", "checkpoint_acquisition_failure"),
    ],
)
def test_trainer_preserves_typed_failures(
    tmp_path: Path, behavior: str, expected_status: str, expected_failure: str
) -> None:
    def runner(command, timeout):
        if behavior == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if behavior == "missing":
            return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")
        message = "CUDA out of memory" if behavior == "oom" else "loss became NaN"
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=message)

    result = run_gaussian_reconstruction_training(
        training_request=_request(),
        dataset_export=_dataset(tmp_path / "artifacts"),
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
        command_runner=runner,
    )
    assert result["status"] == expected_status
    assert result["failure_code"] == expected_failure
    assert result["output_digests"]
    assert result["warnings"] == ["failed_training_evidence_preserved"]


def test_trainer_rejects_hidden_tree_and_dataset_digest_drift(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    export = _dataset(root)
    hidden = root / "candidate_colmap/evaluator_hidden/held_out"
    hidden.mkdir(parents=True)
    (hidden / "secret.png").write_bytes(b"hidden")
    with pytest.raises(GaussianTrainerRuntimeError, match="hidden_heldout_present"):
        run_gaussian_reconstruction_training(
            training_request=_request(),
            dataset_export=export,
            artifact_root=root,
            output_root=tmp_path / "outputs",
            command_runner=_successful_runner,
        )

    export = _dataset(tmp_path / "second")
    export["colmap_training_dataset_digest"] = D[2]
    with pytest.raises(GaussianTrainerRuntimeError, match="binding_or_isolation"):
        run_gaussian_reconstruction_training(
            training_request=_request(),
            dataset_export=export,
            artifact_root=tmp_path / "second",
            output_root=tmp_path / "outputs-2",
            command_runner=_successful_runner,
        )


def test_bound_trainer_matches_registered_tool_call_shape(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    trainer = bind_gaussian_reconstruction_trainer(
        dataset_export=_dataset(artifacts),
        artifact_root=artifacts,
        command_runner=_successful_runner,
        python_executable="python",
        threedgrut_root="/fixture/3dgrut",
    )
    result = trainer(request=_request(), output_root=tmp_path / "outputs")
    assert result["status"] == "succeeded"
    assert result["registered_observation_ids"] == ["frame-1"]


def test_trainer_rejects_missing_observation_ledger(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    export = _dataset(root)
    export.pop("observation_ids")
    with pytest.raises(GaussianTrainerRuntimeError, match="observation_ledger_invalid"):
        run_gaussian_reconstruction_training(
            training_request=_request(),
            dataset_export=export,
            artifact_root=root,
            output_root=tmp_path / "outputs",
            command_runner=_successful_runner,
        )

    export = _dataset(tmp_path / "malformed")
    export["rejected_observation_ids"] = [{"malicious": "object"}]
    with pytest.raises(GaussianTrainerRuntimeError, match="observation_ledger_invalid"):
        run_gaussian_reconstruction_training(
            training_request=_request(),
            dataset_export=export,
            artifact_root=tmp_path / "malformed",
            output_root=tmp_path / "outputs-malformed",
            command_runner=_successful_runner,
        )
