from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_splat_import import (
    IMPORT_REQUEST_SCHEMA_VERSION,
    ProviderSplatImportError,
    align_provider_reconstruction,
    build_provider_splat_import_request,
    import_provider_splat,
)


CAPTURE = "sha256:" + "1" * 64
SPLIT = "sha256:" + "2" * 64
DATASET = "sha256:" + "3" * 64
EXECUTION = "sha256:" + "4" * 64
COMMIT = "5" * 40


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_standard_3dgs_ply(path: Path, positions: np.ndarray, *, corrupt_nan: bool = False) -> None:
    count = positions.shape[0]
    properties = [
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    )
    rows = []
    for index in range(count):
        opacity = float("nan") if corrupt_nan and index == 0 else 0.5
        rows.append(
            struct.pack(
                "<14f",
                float(positions[index, 0]),
                float(positions[index, 1]),
                float(positions[index, 2]),
                0.2, 0.3, 0.4,
                opacity,
                -4.0, -4.1, -4.2,
                1.0, 0.0, 0.0, 0.0,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + b"".join(rows))


def _request(root: Path, splat: Path, **overrides) -> dict:
    value = {
        "schema_version": IMPORT_REQUEST_SCHEMA_VERSION,
        "stable_run_identity": "provider-import-fixture",
        "provider_identity": "teleport",
        "provider_job_identity": "teleport-job-001",
        "provider_execution_receipt_digest": EXECUTION,
        "source_capture_digest": CAPTURE,
        "frozen_split_digest": SPLIT,
        "consumed_candidate_dataset_digest": DATASET,
        "source_commit_sha": COMMIT,
        "asset_bindings": [
            {
                "asset_id": "splat",
                "artifact_kind": "splat_ply",
                "relative_path": splat.relative_to(root).as_posix(),
                "digest": _digest(splat),
            }
        ],
        "provider_had_hidden_access": False,
        "hidden_heldout_pixels_included": False,
        "authority_used": {"provider_upload_authorized": True},
        "proof_effect": "provider_output_import_request_only",
        "claim_ceiling": "none",
        "timestamp": "2026-08-01T00:00:00Z",
    }
    value.update(overrides)
    return value


def test_import_preserves_provider_bytes_and_inventories_splat(tmp_path: Path) -> None:
    root = tmp_path / "provider_out"
    splat = root / "result" / "scene.ply"
    positions = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    _write_standard_3dgs_ply(splat, positions)
    request = _request(root, splat)

    receipt = import_provider_splat(
        source_artifact=request, artifact_root=root, output_root=tmp_path / "imports"
    )
    replay = import_provider_splat(
        source_artifact=request, artifact_root=root, output_root=tmp_path / "imports"
    )

    assert receipt == replay
    assert receipt["status"] == "imported_provider_appearance_candidate_only"
    assert receipt["splat_inventory"]["splat_count"] == 3
    assert receipt["splat_inventory"]["sh_degree"] == 0
    assert receipt["provider_native_output_preserved_unchanged"] is True
    assert receipt["provider_success_is_blueprint_qualification"] is False
    imported = tmp_path / "imports" / receipt["imported_assets"][0]["relative_path"]
    assert imported.read_bytes() == splat.read_bytes()
    assert receipt["provider_splat_import_receipt_digest"] == canonical_digest(
        receipt, digest_field="provider_splat_import_receipt_digest"
    )


def test_import_rejects_nan_tamper_traversal_and_missing_splat(tmp_path: Path) -> None:
    root = tmp_path / "provider_out"
    splat = root / "scene.ply"
    _write_standard_3dgs_ply(splat, np.zeros((2, 3)), corrupt_nan=True)
    with pytest.raises(ProviderSplatImportError, match="nonfinite"):
        import_provider_splat(
            source_artifact=_request(root, splat),
            artifact_root=root,
            output_root=tmp_path / "imports-nan",
        )

    _write_standard_3dgs_ply(splat, np.zeros((2, 3)))
    tampered = _request(root, splat)
    tampered["asset_bindings"][0]["digest"] = "sha256:" + "f" * 64
    with pytest.raises(ProviderSplatImportError, match="digest_mismatch"):
        import_provider_splat(
            source_artifact=tampered,
            artifact_root=root,
            output_root=tmp_path / "imports-tamper",
        )

    hostile = _request(root, splat)
    hostile["asset_bindings"][0]["relative_path"] = "../outside.ply"
    with pytest.raises(ProviderSplatImportError, match="asset_path_invalid"):
        import_provider_splat(
            source_artifact=hostile,
            artifact_root=root,
            output_root=tmp_path / "imports-hostile",
        )

    log_only = _request(root, splat)
    log_only["asset_bindings"][0]["artifact_kind"] = "training_log"
    with pytest.raises(ProviderSplatImportError, match="exactly_one_splat"):
        build_provider_splat_import_request(log_only)


def _candidate_observations() -> list[dict]:
    rng = np.random.default_rng(3)
    observations = []
    for index in range(12):
        matrix = np.eye(4)
        matrix[:3, 3] = rng.uniform(-2.0, 2.0, size=3)
        observations.append(
            {
                "observation_id": f"frame_{index:05d}",
                "camera": {
                    "T_world_camera": matrix.tolist(),
                    "rgb_intrinsics": {"width": 8, "height": 6, "fx": 7.0, "fy": 7.0, "cx": 4.0, "cy": 3.0},
                },
            }
        )
    return observations


def _import_receipt(tmp_path: Path) -> dict:
    root = tmp_path / "provider_out"
    splat = root / "scene.ply"
    _write_standard_3dgs_ply(splat, np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
    return import_provider_splat(
        source_artifact=_request(root, splat),
        artifact_root=root,
        output_root=tmp_path / "imports",
    )


def test_alignment_recovers_similarity_and_fails_closed(tmp_path: Path) -> None:
    receipt = _import_receipt(tmp_path)
    observations = _candidate_observations()
    angle = np.deg2rad(40.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale, translation = 0.5, np.array([3.0, -1.0, 2.0])
    name_map = {}
    provider_cameras = []
    for index, observation in enumerate(observations):
        target = np.asarray(observation["camera"]["T_world_camera"])[:3, 3]
        source = (rotation.T @ (target - translation)) / scale
        image_name = f"{index + 1:06d}_{observation['observation_id']}.jpg"
        name_map[image_name] = observation["observation_id"]
        provider_cameras.append({"image_name": image_name, "position": source.tolist()})
    provider_cameras.append({"image_name": "unknown_extra.jpg", "position": [9.0, 9.0, 9.0]})

    alignment = align_provider_reconstruction(
        import_receipt=receipt,
        provider_cameras=provider_cameras,
        candidate_observations=observations,
        image_name_to_observation_id=name_map,
        alignment_thresholds={"maximum_rms_residual": 0.01, "maximum_max_residual": 0.05},
        timestamp="2026-08-01T00:00:00Z",
    )
    assert alignment["status"] == "aligned_provider_frame_to_candidate_frame"
    assert alignment["estimated_scale_factor"] == pytest.approx(scale, abs=1e-9)
    assert np.asarray(alignment["rotation_matrix"]) == pytest.approx(rotation, abs=1e-9)
    assert alignment["rms_residual"] <= 1e-9
    assert alignment["pair_count"] == 12
    assert alignment["unmatched_provider_camera_count"] == 1
    assert alignment["hidden_cameras_used"] is False
    assert alignment["provider_reconstruction_alignment_digest"] == canonical_digest(
        alignment, digest_field="provider_reconstruction_alignment_digest"
    )

    mirrored = [
        {
            "image_name": camera["image_name"],
            "position": (np.asarray(camera["position"]) * np.array([1.0, 1.0, -1.0])).tolist(),
        }
        for camera in provider_cameras
    ]
    with pytest.raises(ProviderSplatImportError, match="reflection|residual"):
        align_provider_reconstruction(
            import_receipt=receipt,
            provider_cameras=mirrored,
            candidate_observations=observations,
            image_name_to_observation_id=name_map,
            alignment_thresholds={"maximum_rms_residual": 0.01, "maximum_max_residual": 0.05},
            timestamp="2026-08-01T00:00:00Z",
        )

    hostile_map = dict(name_map)
    hostile_map["unknown_extra.jpg"] = "hidden_frame_99999"
    with pytest.raises(ProviderSplatImportError, match="noncandidate"):
        align_provider_reconstruction(
            import_receipt=receipt,
            provider_cameras=provider_cameras,
            candidate_observations=observations,
            image_name_to_observation_id=hostile_map,
            alignment_thresholds={"maximum_rms_residual": 0.01, "maximum_max_residual": 0.05},
            timestamp="2026-08-01T00:00:00Z",
        )

    with pytest.raises(ProviderSplatImportError, match="insufficient_pairs"):
        align_provider_reconstruction(
            import_receipt=receipt,
            provider_cameras=provider_cameras[:4],
            candidate_observations=observations,
            image_name_to_observation_id=name_map,
            alignment_thresholds={"maximum_rms_residual": 0.01, "maximum_max_residual": 0.05},
            timestamp="2026-08-01T00:00:00Z",
        )

    tampered_receipt = dict(receipt)
    tampered_receipt["provider_had_hidden_access"] = True
    with pytest.raises(ProviderSplatImportError, match="import_receipt_invalid"):
        align_provider_reconstruction(
            import_receipt=tampered_receipt,
            provider_cameras=provider_cameras,
            candidate_observations=observations,
            image_name_to_observation_id=name_map,
            alignment_thresholds={"maximum_rms_residual": 0.01, "maximum_max_residual": 0.05},
            timestamp="2026-08-01T00:00:00Z",
        )
