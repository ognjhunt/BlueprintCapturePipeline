from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.reconstruction_appearance_asset import (
    AppearanceAssetContractError,
    build_appearance_asset_manifest,
    compile_particlefield_appearance_asset,
)
from blueprint_pipeline.reconstruction_worker_contracts import build_training_result


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="OpenUSD is required for ParticleField authoring",
)

D = ["sha256:" + character * 64 for character in "abcdef"]
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "1" * 64


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _training_result(root: Path, *, up_axis: str = "Z", status: str = "succeeded") -> dict:
    root.mkdir(parents=True)
    rng = np.random.default_rng(17)
    splat = SplatData(
        count=5,
        xyz=rng.normal(size=(5, 3)).astype(np.float32),
        opacity=rng.normal(size=5).astype(np.float32),
        f_dc=rng.normal(size=(5, 3)).astype(np.float32),
        scales=rng.normal(size=(5, 3)).astype(np.float32),
        quats=rng.normal(size=(5, 4)).astype(np.float32),
        properties=(),
        sh_rest=rng.normal(size=(5, 45)).astype(np.float32),
    )
    ply = write_standard_3dgs_ply(splat, root / "appearance_candidate.ply")
    log = root / "training.log"
    log.write_text("trained\n", encoding="utf-8")
    checkpoint = root / "checkpoint_last.pt"
    checkpoint.write_bytes(b"checkpoint")
    succeeded = status == "succeeded"
    return build_training_result(
        {
            "stable_run_identity": "appearance-fixture",
            "source_capture_identity": "capture-fixture",
            "source_capture_digest": D[0],
            "original_file_references": [{"artifact_id": "capture.mov", "digest": D[1]}],
            "producing_method": "blueprint.pinned_3dgrut_3dgut_mcmc_trainer",
            "implementation_version": "1.0.0",
            "container_image_digest": IMAGE,
            "source_commit_sha": "2" * 40,
            "deterministic_configuration_digest": D[2],
            "input_digests": [{"artifact_id": "dataset", "digest": D[3]}],
            "output_digests": (
                [
                    {"artifact_id": "training.log", "digest": _sha256(log)},
                    {"artifact_id": "appearance_candidate.ply", "digest": _sha256(ply)},
                ]
                if succeeded
                else [{"artifact_id": "training.log", "digest": _sha256(log)}]
            ),
            "train_heldout_split_digest": D[4],
            "camera_calibration_binding": {"calibration_digest": D[1]},
            "coordinate_frame_declaration": {"frame": "world", "up_axis": up_axis},
            "units": "meters",
            "metric_scale_status": "sensor_metric_unvalidated",
            "provider_runtime_identity": {"provider": "vast", "runtime": "candidate"},
            "cost_usd": 0.5,
            "duration_seconds": 2.0,
            "authority_used": {"authority_id": "fixture"},
            "warnings": [],
            "blockers": [] if succeeded else ["training_divergence"],
            "proof_effect": "appearance_asset_candidate_only",
            "claim_ceiling": "appearance_reconstruction",
            "parent_artifact_or_event": {"digest": D[5]},
            "timestamp": "2026-07-30T23:00:00Z",
            "reconstruction_training_request_digest": D[5],
            "status": status,
            "failure_code": None if succeeded else "training_divergence",
            "checkpoint_references": (
                [{"artifact_id": "checkpoint_last.pt", "digest": _sha256(checkpoint)}]
                if succeeded
                else []
            ),
            "training_metrics": {"heldout_metrics_computed": False},
            "heldout_labels_included": False,
            "candidate_self_graded": False,
            "registered_observation_ids": ["frame-1"],
            "rejected_observation_ids": [],
            "peak_resource_use": {"gpu_memory_bytes": 1},
            "legal_next_actions": ["preserve_evidence_and_stop"],
        }
    )


def test_compile_particlefield_appearance_preserves_sh_and_lineage(tmp_path: Path) -> None:
    from pxr import Usd

    training_root = tmp_path / "training"
    training = _training_result(training_root)
    first = compile_particlefield_appearance_asset(
        training_result=training,
        training_artifact_root=training_root,
        output_root=tmp_path / "appearance",
    )
    replay = compile_particlefield_appearance_asset(
        training_result=training,
        training_artifact_root=training_root,
        output_root=tmp_path / "appearance",
    )
    assert replay == first
    assert first["schema_version"] == "appearance_asset_manifest.v1"
    assert first["reconstruction_training_result_digest"] == training[
        "reconstruction_training_result_digest"
    ]
    assert first["sh_degree"] == 3
    assert first["captured_observation"] is False
    assert first["metric_geometry_proven"] is False
    assert first["collision_geometry_proven"] is False
    assert first["heldout_evaluated"] is False
    output = tmp_path / "appearance" / first["appearance_asset_reference"]
    assert _sha256(output) == first["appearance_asset_digest"]
    stage = Usd.Stage.Open(str(output))
    prim = stage.GetPrimAtPath("/World/Appearance")
    assert str(prim.GetTypeName()) == "ParticleField3DGaussianSplat"
    assert prim.GetAttribute("radiance:sphericalHarmonicsDegree").Get() == 3


def test_compile_particlefield_appearance_fails_closed_on_lineage_and_frame(
    tmp_path: Path,
) -> None:
    failed_root = tmp_path / "failed"
    failed = _training_result(failed_root, status="failed")
    with pytest.raises(AppearanceAssetContractError, match="successful_training_result_required"):
        compile_particlefield_appearance_asset(
            training_result=failed,
            training_artifact_root=failed_root,
            output_root=tmp_path / "out-failed",
        )

    y_root = tmp_path / "y-up"
    y_up = _training_result(y_root, up_axis="Y")
    with pytest.raises(AppearanceAssetContractError, match="z_up_frame_unqualified"):
        compile_particlefield_appearance_asset(
            training_result=y_up,
            training_artifact_root=y_root,
            output_root=tmp_path / "out-y",
        )

    valid_root = tmp_path / "tampered"
    valid = _training_result(valid_root)
    (valid_root / "appearance_candidate.ply").write_bytes(b"tampered")
    with pytest.raises(AppearanceAssetContractError, match="digest_mismatch"):
        compile_particlefield_appearance_asset(
            training_result=valid,
            training_artifact_root=valid_root,
            output_root=tmp_path / "out-tampered",
        )


def test_appearance_manifest_rejects_claim_promotion(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    manifest = compile_particlefield_appearance_asset(
        training_result=_training_result(training_root),
        training_artifact_root=training_root,
        output_root=tmp_path / "appearance",
    )
    promoted = dict(manifest)
    promoted.pop("appearance_asset_manifest_digest")
    promoted["collision_geometry_proven"] = True
    with pytest.raises(AppearanceAssetContractError, match="appearance_cannot_promote_geometry"):
        build_appearance_asset_manifest(promoted)
