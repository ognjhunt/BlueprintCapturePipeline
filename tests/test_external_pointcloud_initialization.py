from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.external_pointcloud_initialization import (
    ExternalPointcloudInitializationError,
    REQUEST_SCHEMA_VERSION,
    compile_external_pointcloud_initialization,
    read_pointcloud_ply_positions,
)
from blueprint_pipeline.external_reconstruction_import import (
    build_external_reconstruction_import_request,
    import_external_reconstruction,
)
from blueprint_pipeline.reconstruction_colmap_dataset import (
    ColmapTrainingDatasetError,
    REQUEST_SCHEMA_VERSION as COLMAP_REQUEST_SCHEMA_VERSION,
    bind_colmap_initialization_points,
    export_colmap_training_dataset,
)


CAPTURE = "sha256:" + "a" * 64
SPLIT = "sha256:" + "b" * 64
DATASET = "sha256:" + "c" * 64
OBSERVATION = "sha256:" + "d" * 64
COMMIT = "e" * 40


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _rotation() -> np.ndarray:
    angle_z, angle_x = np.deg2rad(30.0), np.deg2rad(15.0)
    rz = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0.0],
            [np.sin(angle_z), np.cos(angle_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle_x), -np.sin(angle_x)],
            [0.0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )
    return rz @ rx


SCALE = 2.0
ROTATION = _rotation()
TRANSLATION = np.array([1.0, -2.0, 3.0])
TARGET_CENTERS = np.array(
    [
        [0.0, 0.0, 1.4],
        [1.0, 0.2, 1.5],
        [2.0, 1.1, 1.3],
        [1.8, 2.2, 1.6],
        [0.7, 2.9, 1.5],
        [-0.6, 2.4, 1.2],
        [-1.1, 1.0, 1.7],
        [-0.4, -0.6, 1.8],
        [0.9, 1.3, 2.4],
        [1.4, 1.9, 0.6],
    ]
)


def _to_source(target: np.ndarray) -> np.ndarray:
    return ((target - TRANSLATION) @ ROTATION) / SCALE


def _write_trajectory(path: Path, centers_by_id: dict[str, np.ndarray]) -> None:
    frames = []
    for frame_id, center in sorted(centers_by_id.items()):
        matrix = np.eye(4)
        matrix[:3, 3] = center
        frames.append(
            {
                "file_path": f"./images/{frame_id}.jpg",
                "depth_file_path": f"./depth/{frame_id}.png",
                "transform_matrix": matrix.tolist(),
                "w": 8,
                "h": 6,
                "fl_x": 7.0,
                "fl_y": 7.0,
                "cx": 4.0,
                "cy": 3.0,
            }
        )
    path.write_text(json.dumps({"camera_model": "OPENCV", "frames": frames}), encoding="utf-8")


def _write_ply(path: Path, positions: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by Polycam\n"
        f"element vertex {positions.shape[0]}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for row in positions:
            stream.write(struct.pack("<dddBBB", row[0], row[1], row[2], 128, 128, 128))


def _import_pointcloud(tmp_path: Path, positions: np.ndarray) -> tuple[dict, Path]:
    source = tmp_path / "provider_export"
    source.mkdir(parents=True, exist_ok=True)
    ply = source / "polycam_pointcloud.ply"
    _write_ply(ply, positions)
    digest = _digest(ply)
    declaration = {
        "provider_identity": "polycam",
        "product_tier": "user_managed_export",
        "terms_version": "user-attested-2026-08-01",
        "provider_scan_or_job_identity": "mushroom-koivu-polycam",
        "export_created_at": "2026-08-01T00:00:00Z",
        "export_performed_by": "dataset-publisher",
        "source_capture_identity": "fixture-capture",
        "source_capture_digest": CAPTURE,
        "ownership_or_license_confirmed": True,
        "commercial_use_status": "permitted",
        "intended_uses": ["reconstruction_initialization"],
        "consent_status": "not_required",
        "privacy_status": "cleared",
        "confidentiality_terms_status": "public_cc_by_dataset",
        "retention_status": "user_managed_known",
        "deletion_status": "not_requested_user_managed",
        "model_training_terms_status": "acknowledged",
        "competitive_use_status": "acknowledged",
        "resale_status": "acknowledged",
        "benchmarking_status": "acknowledged",
        "user_managed_provider_processing_attested": True,
        "blueprint_remote_upload_performed": False,
    }
    declaration["declaration_digest"] = canonical_digest(
        declaration, digest_field="declaration_digest"
    )
    request = build_external_reconstruction_import_request(
        {
            "stable_run_identity": "pointcloud-import-fixture",
            "source_capture_identity": "fixture-capture",
            "source_capture_digest": CAPTURE,
            "original_file_references": [{"artifact_id": "pointcloud", "digest": digest}],
            "producing_method": "external_import_request_compiler",
            "implementation_version": "1",
            "source_commit_sha": COMMIT,
            "deterministic_configuration_digest": "sha256:" + "9" * 64,
            "input_digests": [{"artifact_id": "pointcloud", "digest": digest}],
            "output_digests": [],
            "train_heldout_split_digest": SPLIT,
            "camera_calibration_binding": {"status": "external_unverified"},
            "coordinate_frame_declaration": {"status": "external_unverified"},
            "units": "meters",
            "provider_runtime_identity": {"provider": "local", "source_provider": "polycam"},
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "authority_used": {"mode": "execute_non_spend"},
            "warnings": [],
            "blockers": [],
            "parent_artifact_or_event": {"digest": CAPTURE},
            "timestamp": "2026-08-01T00:00:00Z",
            "provider_identity": "polycam",
            "import_lane": "local_external_import",
            "asset_bindings": [
                {
                    "asset_id": "pointcloud",
                    "relative_path": ply.name,
                    "digest": digest,
                }
            ],
            "provenance_rights_declaration": declaration,
            "remote_calls_authorized": False,
            "remote_calls_performed": False,
            "proof_effect": "external_import_request_only",
            "claim_ceiling": "none",
        }
    )
    import_root = tmp_path / "imports"
    receipt = import_external_reconstruction(
        source_artifact=request, artifact_root=source, output_root=import_root
    )
    return receipt, import_root


def _initialization_request(
    tmp_path: Path,
    *,
    thresholds: dict | None = None,
    candidate_ids: list[str] | None = None,
) -> dict:
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "stable_run_identity": "pointcloud-init-fixture",
        "source_capture_digest": CAPTURE,
        "frozen_split_digest": SPLIT,
        "camera_observation_digest": OBSERVATION,
        "source_commit_sha": COMMIT,
        "candidate_observation_ids": candidate_ids
        or [f"frame_{index:05d}" for index in range(1, 11)],
        "pointcloud_asset_id": "pointcloud",
        "source_trajectory_relative_path": "src_traj.json",
        "target_trajectory_relative_path": "tgt_traj.json",
        "alignment_thresholds": thresholds
        or {
            "maximum_rms_residual": 0.05,
            "maximum_max_residual": 0.2,
            "minimum_in_bounds_ratio": 0.9,
            "bounds_inflation_factor": 3.0,
            "minimum_bounds_margin": 1.0,
        },
        "thresholds_frozen_before_alignment": True,
        "hidden_heldout_access_requested": False,
        "maximum_points": 100000,
        "units": "target_frame_units_not_independently_validated",
        "metric_scale_status": "not_independently_validated",
        "coordinate_frame_declaration": {
            "declaration": "fixture_target_frame",
            "handedness": "not_independently_declared",
            "gravity_alignment": "not_independently_validated",
        },
        "authority_used": {"local_processing_authorized": True},
        "timestamp": "2026-08-01T00:00:00Z",
    }
    request["external_pointcloud_initialization_request_digest"] = canonical_digest(
        request, digest_field="external_pointcloud_initialization_request_digest"
    )
    return request


def _write_trajectories(root: Path, *, mirror: bool = False, corrupt: bool = False,
                        drop_last: bool = False) -> None:
    ids = [f"frame_{index:05d}" for index in range(1, 11)]
    target = {frame_id: TARGET_CENTERS[index] for index, frame_id in enumerate(ids)}
    source = {}
    for index, frame_id in enumerate(ids):
        center = TARGET_CENTERS[index]
        if mirror:
            mirrored = center * np.array([1.0, 1.0, -1.0])
            source[frame_id] = _to_source(mirrored)
        else:
            source[frame_id] = _to_source(center)
    if corrupt:
        source[ids[0]] = source[ids[0]] + np.array([1.0, 0.0, 0.0])
    source_ids = ids[:-1] if drop_last else ids
    _write_trajectory(root / "src_traj.json", {i: source[i] for i in source_ids})
    _write_trajectory(root / "tgt_traj.json", target)


def test_compile_recovers_known_similarity_and_transforms_points(tmp_path: Path) -> None:
    target_points = np.array(
        [
            [0.5, 0.5, 1.0],
            [1.5, 1.0, 1.5],
            [-0.5, 2.0, 1.2],
            [1.0, 2.5, 2.0],
        ]
    )
    receipt, import_root = _import_pointcloud(tmp_path, _to_source(target_points))
    trajectory_root = tmp_path / "trajectories"
    trajectory_root.mkdir()
    _write_trajectories(trajectory_root)
    output_root = tmp_path / "initialization_out"

    result = compile_external_pointcloud_initialization(
        source_artifact=_initialization_request(tmp_path),
        import_receipt=receipt,
        import_output_root=import_root,
        source_trajectory_root=trajectory_root,
        output_root=output_root,
    )
    replay = compile_external_pointcloud_initialization(
        source_artifact=_initialization_request(tmp_path),
        import_receipt=receipt,
        import_output_root=import_root,
        source_trajectory_root=trajectory_root,
        output_root=output_root,
    )

    assert result == replay
    assert result["status"] == "compiled_external_initialization_points"
    alignment = result["alignment"]
    assert alignment["estimated_scale_factor"] == pytest.approx(SCALE, abs=1e-9)
    assert np.asarray(alignment["rotation_matrix"]) == pytest.approx(ROTATION, abs=1e-9)
    assert alignment["rms_residual"] <= 1e-9
    assert alignment["in_bounds_ratio"] == 1.0
    assert result["reflection_preferred_by_alignment"] is False
    assert result["alignment_residual_gates_passed"] is True
    assert result["emitted_point_count"] == 4
    vertices_path = output_root / result["surface_asset"]["relative_path"]
    assert _digest(vertices_path) == result["surface_asset"]["digest"]
    vertices = json.loads(vertices_path.read_text(encoding="utf-8"))
    assert vertices["generated_fill_used"] is False
    recovered = np.asarray([row["position_m"] for row in vertices["vertices"]])
    assert recovered == pytest.approx(target_points, abs=1e-5)
    assert result["external_pointcloud_initialization_result_digest"] == canonical_digest(
        result, digest_field="external_pointcloud_initialization_result_digest"
    )


def test_compile_rejects_reflected_residual_and_out_of_bounds_geometry(tmp_path: Path) -> None:
    receipt, import_root = _import_pointcloud(tmp_path, _to_source(TARGET_CENTERS[:4]))

    mirror_root = tmp_path / "mirror"
    mirror_root.mkdir()
    _write_trajectories(mirror_root, mirror=True)
    with pytest.raises(ExternalPointcloudInitializationError, match="reflection_detected"):
        compile_external_pointcloud_initialization(
            source_artifact=_initialization_request(tmp_path),
            import_receipt=receipt,
            import_output_root=import_root,
            source_trajectory_root=mirror_root,
            output_root=tmp_path / "mirror_out",
        )

    corrupt_root = tmp_path / "corrupt"
    corrupt_root.mkdir()
    _write_trajectories(corrupt_root, corrupt=True)
    with pytest.raises(ExternalPointcloudInitializationError, match="residual_threshold"):
        compile_external_pointcloud_initialization(
            source_artifact=_initialization_request(tmp_path),
            import_receipt=receipt,
            import_output_root=import_root,
            source_trajectory_root=corrupt_root,
            output_root=tmp_path / "corrupt_out",
        )

    coverage_root = tmp_path / "coverage"
    coverage_root.mkdir()
    _write_trajectories(coverage_root, drop_last=True)
    with pytest.raises(
        ExternalPointcloudInitializationError, match="candidate_coverage_incomplete"
    ):
        compile_external_pointcloud_initialization(
            source_artifact=_initialization_request(tmp_path),
            import_receipt=receipt,
            import_output_root=import_root,
            source_trajectory_root=coverage_root,
            output_root=tmp_path / "coverage_out",
        )

    far_receipt, far_import_root = _import_pointcloud(
        tmp_path / "far", _to_source(TARGET_CENTERS[:4] + 10000.0)
    )
    bounds_root = tmp_path / "bounds"
    bounds_root.mkdir()
    _write_trajectories(bounds_root)
    with pytest.raises(ExternalPointcloudInitializationError, match="out_of_bounds_ratio"):
        compile_external_pointcloud_initialization(
            source_artifact=_initialization_request(tmp_path),
            import_receipt=far_receipt,
            import_output_root=far_import_root,
            source_trajectory_root=bounds_root,
            output_root=tmp_path / "bounds_out",
        )


def _colmap_request(root: Path) -> dict:
    frames = []
    observations = []
    for index in (1, 2):
        relative = f"candidate_dataset/training/frame-{index}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color=(index, 20, 30)).save(path)
        digest = _digest(path)
        pose = np.eye(4)
        pose[0, 3] = float(index)
        frames.append({"frame_id": f"frame-{index}", "split": "training", "frame_digest": digest})
        observations.append(
            {
                "observation_id": f"frame-{index}",
                "split": "training",
                "image_relative_path": relative,
                "image_digest": digest,
                "T_world_camera": pose.tolist(),
                "camera": {
                    "T_world_camera": pose.tolist(),
                    "rgb_intrinsics": {
                        "width": 8,
                        "height": 6,
                        "fx": 7.0,
                        "fy": 7.5,
                        "cx": 4.0,
                        "cy": 3.0,
                    },
                },
            }
        )
    candidate = {
        "schema_version": "candidate_reconstruction_dataset_manifest.v1",
        "capture_digest": CAPTURE,
        "split_digest": SPLIT,
        "heldout_pixels_included": False,
        "frames": frames,
    }
    candidate["candidate_dataset_digest"] = canonical_digest(
        candidate, digest_field="candidate_dataset_digest"
    )
    manifest = {
        "schema_version": "camera_observation_manifest.v1",
        "source_capture_digest": CAPTURE,
        "hidden_heldout_pixels_included": False,
        "observations": observations,
    }
    manifest["camera_observation_digest"] = canonical_digest(
        manifest, digest_field="camera_observation_digest"
    )
    request = {
        "schema_version": COLMAP_REQUEST_SCHEMA_VERSION,
        "stable_run_identity": "colmap-points-fixture",
        "source_capture_digest": CAPTURE,
        "source_commit_sha": COMMIT,
        "reconstruction_dataset_digest": DATASET,
        "frozen_split_digest": SPLIT,
        "camera_observation_manifest": manifest,
        "candidate_dataset_manifest": candidate,
        "coordinate_frame_declaration": {
            "declaration": "fixture_target_frame",
            "handedness": "not_independently_declared",
            "gravity_alignment": "not_independently_validated",
        },
        "units": "publisher_pose_units_not_independently_validated",
        "metric_scale_status": "not_independently_validated",
        "authority_used": {"local_processing_authorized": True},
        "timestamp": "2026-08-01T00:00:00Z",
        "blockers": ["initialization_points_not_bound"],
    }
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    return request


def test_bind_and_export_point_seeded_dataset_without_touching_candidate_pixels(
    tmp_path: Path,
) -> None:
    target_points = np.array([[0.5, 0.5, 1.0], [1.5, 1.0, 1.5], [-0.5, 2.0, 1.2]])
    receipt, import_root = _import_pointcloud(tmp_path, _to_source(target_points))
    trajectory_root = tmp_path / "trajectories"
    trajectory_root.mkdir()
    _write_trajectories(trajectory_root)
    proxy_root = tmp_path / "proxy"
    raw_request = _colmap_request(proxy_root)
    initialization_request = _initialization_request(tmp_path)
    initialization_request["camera_observation_digest"] = raw_request[
        "camera_observation_manifest"
    ]["camera_observation_digest"]
    initialization_request["external_pointcloud_initialization_request_digest"] = canonical_digest(
        {
            key: value
            for key, value in initialization_request.items()
            if key != "external_pointcloud_initialization_request_digest"
        },
        digest_field="external_pointcloud_initialization_request_digest",
    )
    result = compile_external_pointcloud_initialization(
        source_artifact=initialization_request,
        import_receipt=receipt,
        import_output_root=import_root,
        source_trajectory_root=trajectory_root,
        output_root=proxy_root,
    )

    raw_digest = raw_request["colmap_training_dataset_export_request_digest"]
    bound = bind_colmap_initialization_points(
        source_artifact=raw_request, initialization_result=result
    )
    assert raw_request["colmap_training_dataset_export_request_digest"] == raw_digest
    assert bound["parent_colmap_training_dataset_export_request_digest"] == raw_digest
    assert bound["initialization_points_result_digest"] == result[
        "external_pointcloud_initialization_result_digest"
    ]
    assert "initialization_points_not_bound" not in bound["blockers"]

    export = export_colmap_training_dataset(
        source_artifact=bound,
        artifact_root=proxy_root,
        output_root=tmp_path / "trainer_input",
        initialization_artifact_root=proxy_root,
    )
    assert export["initialization_point_count"] == 3
    assert export["initialization_surface_digest"] == result["surface_asset"]["digest"]
    points = (
        tmp_path
        / "trainer_input"
        / export["relative_path"]
        / "sparse/0/points3D.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert len(points) == 4
    first_point = [float(value) for value in points[1].split()[1:4]]
    assert first_point == pytest.approx(target_points[0], abs=1e-5)

    tampered = dict(result)
    tampered["alignment_residual_gates_passed"] = False
    with pytest.raises(ColmapTrainingDatasetError, match="points_binding_result_invalid"):
        bind_colmap_initialization_points(
            source_artifact=raw_request, initialization_result=tampered
        )
    tampered["external_pointcloud_initialization_result_digest"] = canonical_digest(
        {
            key: value
            for key, value in tampered.items()
            if key != "external_pointcloud_initialization_result_digest"
        },
        digest_field="external_pointcloud_initialization_result_digest",
    )
    with pytest.raises(ColmapTrainingDatasetError, match="points_binding_truth_boundary_invalid"):
        bind_colmap_initialization_points(
            source_artifact=raw_request, initialization_result=tampered
        )


def test_ply_reader_rejects_ascii_list_properties_and_truncation(tmp_path: Path) -> None:
    ascii_ply = tmp_path / "ascii.ply"
    ascii_ply.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
    with pytest.raises(ExternalPointcloudInitializationError, match="format_unsupported"):
        read_pointcloud_ply_positions(ascii_ply)

    list_ply = tmp_path / "list.ply"
    list_ply.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property list uchar int vertex_indices\nend_header\n\x00"
    )
    with pytest.raises(ExternalPointcloudInitializationError, match="property_unsupported"):
        read_pointcloud_ply_positions(list_ply)

    truncated = tmp_path / "truncated.ply"
    truncated.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property double x\nproperty double y\nproperty double z\nend_header\n"
        + struct.pack("<ddd", 0.0, 0.0, 0.0)
    )
    with pytest.raises(ExternalPointcloudInitializationError, match="body_truncated"):
        read_pointcloud_ply_positions(truncated)
