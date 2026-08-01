from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_colmap_dataset import (
    ColmapTrainingDatasetError,
    REQUEST_SCHEMA_VERSION,
    bind_colmap_refined_poses,
    export_colmap_training_dataset,
)
from blueprint_pipeline.reconstruction_pose_refinement import (
    POSE_REFINEMENT_EXECUTION_REQUEST_SCHEMA_VERSION,
    POSE_REFINEMENT_RESULT_SCHEMA_VERSION,
    REFINED_CAMERA_POSE_MANIFEST_SCHEMA_VERSION,
    build_pose_refinement_execution_request,
    build_pose_refinement_result,
    build_refined_camera_pose_manifest,
)


CAPTURE = "sha256:" + "a" * 64
DATASET = "sha256:" + "b" * 64
SPLIT = "sha256:" + "c" * 64
COMMIT = "d" * 40
RECORDED_REAL_EXPORT = (
    Path(__file__).parents[1]
    / "docs/evidence/arkitscenes_colmap_training_dataset_40958756_cb96cbfc.json"
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _request(root: Path) -> dict:
    frames = []
    observations = []
    for index, split in enumerate(("training", "validation"), start=1):
        relative = f"frozen/candidate_dataset/{split}/frame-{index}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color=(index, 2, 3)).save(path)
        digest = _digest(path)
        frames.append(
            {
                "frame_id": f"frame-{index}",
                "split": split,
                "frame_digest": digest,
            }
        )
        pose = np.eye(4)
        pose[0, 3] = float(index)
        observations.append(
            {
                "observation_id": f"frame-{index}",
                "split": split,
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
    observation_manifest = {
        "schema_version": "camera_observation_manifest.v1",
        "source_capture_digest": CAPTURE,
        "hidden_heldout_pixels_included": False,
        "observations": observations,
    }
    observation_manifest["camera_observation_digest"] = canonical_digest(
        observation_manifest, digest_field="camera_observation_digest"
    )
    calibration = {
        "schema_version": "camera_calibration_manifest.v1",
        "capture_digest": CAPTURE,
        "camera_model": "PINHOLE",
        "intrinsics": {
            "width": 8,
            "height": 6,
            "fx": 7.0,
            "fy": 7.5,
            "cx": 4.0,
            "cy": 3.0,
        },
    }
    calibration["calibration_digest"] = canonical_digest(
        calibration, digest_field="calibration_digest"
    )
    for observation in observation_manifest["observations"]:
        observation["calibration_digest"] = calibration["calibration_digest"]
    observation_manifest["camera_observation_digest"] = canonical_digest(
        observation_manifest, digest_field="camera_observation_digest"
    )
    surface = {
        "schema_version": "observed_surface_mesh.v1",
        "source_capture_digest": CAPTURE,
        "train_heldout_split_digest": SPLIT,
        "generated_fill_used": False,
        "vertices": [
            {"position_m": [0.0, 0.0, 0.0]},
            {"position_m": [1.0, 2.0, 3.0]},
        ],
    }
    surface_path = root / "surface/observed.json"
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    surface_path.write_text(json.dumps(surface), encoding="utf-8")
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "stable_run_identity": "colmap-export-fixture",
        "source_capture_digest": CAPTURE,
        "source_commit_sha": COMMIT,
        "reconstruction_dataset_digest": DATASET,
        "frozen_split_digest": SPLIT,
        "camera_observation_manifest": observation_manifest,
        "camera_calibration_manifest": calibration,
        "candidate_dataset_manifest": candidate,
        "initialization_surface": {
            "relative_path": "surface/observed.json",
            "digest": _digest(surface_path),
        },
        "maximum_initialization_points": 10,
        "coordinate_frame_declaration": {"frame": "fixture_world", "up_axis": "unknown"},
        "units": "meters",
        "metric_scale_status": "sensor_metric_unvalidated",
        "authority_used": {"local_processing_authorized": True},
        "timestamp": "2026-07-30T18:00:00Z",
        "blockers": ["pose_refinement_not_executed"],
    }
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    return request


def _pose_artifacts(request: dict, *, maximum_translation_m: float = 0.02) -> tuple[dict, dict, dict]:
    observations = request["camera_observation_manifest"]
    calibration = request["camera_calibration_manifest"]
    initial_digest = observations["camera_observation_digest"]
    refinement_request = build_pose_refinement_execution_request(
        {
            "schema_version": POSE_REFINEMENT_EXECUTION_REQUEST_SCHEMA_VERSION,
            "stable_run_identity": "colmap-refinement-fixture",
            "source_capture_digest": CAPTURE,
            "capture_profile": "iphone_arkit_lidar",
            "reconstruction_dataset_digest": DATASET,
            "frozen_split_digest": SPLIT,
            "camera_observation_digest": initial_digest,
            "camera_calibration_digest": calibration["calibration_digest"],
            "initial_pose_manifest_digest": initial_digest,
            "initial_pose_source": "verified_arkit_raw_contract_3_2",
            "method_id": "arkit_anchored_bundle_adjustment_v1",
            "drift_thresholds": {
                "maximum_translation_m": 0.05,
                "maximum_rotation_degrees": 2.0,
            },
            "thresholds_frozen_before_execution": True,
            "raw_arkit_poses_may_be_modified": False,
            "candidate_may_read_hidden_heldout": False,
            "coordinate_frame_declaration": request["coordinate_frame_declaration"],
            "implementation_digest": "sha256:" + "1" * 64,
            "container_image_digest": "sha256:" + "2" * 64,
            "source_commit_sha": COMMIT,
            "random_seed": 7,
            "resource_request": {"gpu_count": 0, "cpu_count": 4},
            "timeout_seconds": 600.0,
            "spend_cap_usd": 0.0,
            "authority_used": {"local_non_spend": True},
            "timestamp": "2026-07-30T18:00:01Z",
        }
    )
    refined = []
    for observation in observations["observations"]:
        matrix = json.loads(json.dumps(observation["T_world_camera"]))
        matrix[0][3] += 0.01
        refined.append(
            {"observation_id": observation["observation_id"], "T_world_camera": matrix}
        )
    manifest = build_refined_camera_pose_manifest(
        {
            "schema_version": REFINED_CAMERA_POSE_MANIFEST_SCHEMA_VERSION,
            "stable_run_identity": "colmap-refinement-fixture",
            "source_capture_identity": "fixture-capture",
            "source_capture_digest": CAPTURE,
            "original_file_references": [
                {"artifact_id": "raw-video", "digest": "sha256:" + "3" * 64}
            ],
            "producing_method": refinement_request["method_id"],
            "implementation_version": "1.0.0",
            "implementation_digest": refinement_request["implementation_digest"],
            "container_image_digest": refinement_request["container_image_digest"],
            "source_commit_sha": refinement_request["source_commit_sha"],
            "input_digests": [
                refinement_request["pose_refinement_execution_request_digest"],
                initial_digest,
            ],
            "output_digests": [],
            "frozen_split_digest": SPLIT,
            "camera_calibration_digest": calibration["calibration_digest"],
            "initial_pose_manifest_digest": initial_digest,
            "pose_refinement_execution_request_digest": refinement_request[
                "pose_refinement_execution_request_digest"
            ],
            "method_id": refinement_request["method_id"],
            "coordinate_frame_declaration": request["coordinate_frame_declaration"],
            "units": "meters",
            "metric_scale_status": "sensor_metric_unvalidated",
            "provider_runtime_identity": {"provider": "local", "runtime": "fixture"},
            "cost_usd": 0.0,
            "duration_seconds": 1.0,
            "authority_used": refinement_request["authority_used"],
            "warnings": [],
            "blockers": [],
            "parent_artifact_or_event": {
                "pose_refinement_execution_request_digest": refinement_request[
                    "pose_refinement_execution_request_digest"
                ],
                "initial_pose_manifest_digest": initial_digest,
            },
            "observations": refined,
            "raw_arkit_poses_modified": False,
            "hidden_heldout_observations_included": False,
            "proof_effect": "bounded_refined_trajectory_candidate_only",
            "claim_ceiling": "calibrated_camera_trajectory",
            "timestamp": "2026-07-30T18:00:02Z",
        }
    )
    result = build_pose_refinement_result(
        {
            "schema_version": POSE_REFINEMENT_RESULT_SCHEMA_VERSION,
            "source_capture_digest": CAPTURE,
            "pose_refinement_execution_request_digest": refinement_request[
                "pose_refinement_execution_request_digest"
            ],
            "frozen_split_digest": SPLIT,
            "camera_calibration_digest": calibration["calibration_digest"],
            "initial_pose_manifest_digest": initial_digest,
            "implementation_digest": refinement_request["implementation_digest"],
            "container_image_digest": refinement_request["container_image_digest"],
            "status": "succeeded",
            "failure_code": None,
            "refined_pose_manifest_digest": manifest[
                "refined_camera_pose_manifest_digest"
            ],
            "drift_metrics": {
                "maximum_translation_m": maximum_translation_m,
                "mean_translation_m": 0.01,
                "maximum_rotation_degrees": 0.0,
                "mean_rotation_degrees": 0.0,
            },
            "registered_observation_ids": ["frame-1", "frame-2"],
            "rejected_observation_ids": [],
            "warnings": [],
            "blockers": [],
            "raw_arkit_poses_modified": False,
            "heldout_labels_included": False,
            "candidate_self_graded": False,
            "cost_usd": 0.0,
            "duration_seconds": 1.0,
            "proof_effect": "bounded_refined_trajectory_candidate_only",
            "claim_ceiling": "calibrated_camera_trajectory",
        }
    )
    return refinement_request, result, manifest


def test_export_is_candidate_only_idempotent_and_converts_pose(tmp_path: Path) -> None:
    source = tmp_path / "source"
    request = _request(source)
    output = tmp_path / "output"

    first = export_colmap_training_dataset(
        source_artifact=request, artifact_root=source, output_root=output
    )
    second = export_colmap_training_dataset(
        source_artifact=request, artifact_root=source, output_root=output
    )

    assert first == second
    assert first["image_count"] == 2
    assert first["observation_ids"] == ["frame-1", "frame-2"]
    assert first["rejected_observation_ids"] == []
    assert first["initialization_point_count"] == 2
    assert first["hidden_heldout_pixels_included"] is False
    assert first["raw_input_poses_modified"] is False
    root = output / first["relative_path"]
    images = (root / "sparse/0/images.txt").read_text(encoding="utf-8").splitlines()
    first_pose = images[1].split()
    assert [float(value) for value in first_pose[2:5]] == pytest.approx([0.0, 0.0, 0.0])
    assert [float(value) for value in first_pose[5:8]] == pytest.approx([-1.0, 0.0, 0.0])
    assert len(list((root / "images").glob("*.png"))) == 2
    assert "held_out" not in "\n".join(path.as_posix() for path in root.rglob("*"))
    assert first["colmap_training_dataset_export_result_digest"] == canonical_digest(
        first, digest_field="colmap_training_dataset_export_result_digest"
    )
    result_schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/colmap_training_dataset_export_result.v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(first, result_schema)


def test_qualified_refined_poses_create_new_request_without_mutating_raw_parent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    raw_request = _request(source)
    raw_digest = raw_request["colmap_training_dataset_export_request_digest"]
    raw_matrix = json.loads(
        json.dumps(
            raw_request["camera_observation_manifest"]["observations"][0][
                "T_world_camera"
            ]
        )
    )
    refinement_request, refinement_result, manifest = _pose_artifacts(raw_request)

    refined_request = bind_colmap_refined_poses(
        source_artifact=raw_request,
        pose_refinement_request=refinement_request,
        pose_refinement_result=refinement_result,
        refined_pose_manifest=manifest,
    )

    assert raw_request["colmap_training_dataset_export_request_digest"] == raw_digest
    assert (
        raw_request["camera_observation_manifest"]["observations"][0]["T_world_camera"]
        == raw_matrix
    )
    assert refined_request["parent_colmap_training_dataset_export_request_digest"] == raw_digest
    assert refined_request["pose_refinement_executed"] is True
    assert refined_request["raw_arkit_poses_modified"] is False
    assert "pose_refinement_not_executed" not in refined_request["blockers"]
    assert (
        refined_request["camera_observation_manifest"]["observations"][0][
            "T_world_camera"
        ][0][3]
        == pytest.approx(raw_matrix[0][3] + 0.01)
    )

    export = export_colmap_training_dataset(
        source_artifact=refined_request,
        artifact_root=source,
        output_root=tmp_path / "refined-output",
    )
    assert export["pose_refinement_executed"] is True
    assert export["raw_input_poses_modified"] is False
    images = (
        tmp_path / "refined-output" / export["relative_path"] / "sparse/0/images.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert float(images[1].split()[5]) == pytest.approx(-(raw_matrix[0][3] + 0.01))


def test_refined_pose_binding_rejects_drift_above_frozen_threshold(tmp_path: Path) -> None:
    request = _request(tmp_path / "source")
    refinement_request, refinement_result, manifest = _pose_artifacts(
        request, maximum_translation_m=0.5
    )

    with pytest.raises(ColmapTrainingDatasetError, match="drift_threshold_exceeded"):
        bind_colmap_refined_poses(
            source_artifact=request,
            pose_refinement_request=refinement_request,
            pose_refinement_result=refinement_result,
            refined_pose_manifest=manifest,
        )


def test_export_accepts_proxy_capture_digest_alias_without_weakening_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    request = _request(source)
    observations = request["camera_observation_manifest"]
    observations["capture_digest"] = observations.pop("source_capture_digest")
    observations["camera_observation_digest"] = canonical_digest(
        observations, digest_field="camera_observation_digest"
    )
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )

    result = export_colmap_training_dataset(
        source_artifact=request, artifact_root=source, output_root=tmp_path / "output"
    )
    assert result["source_capture_digest"] == CAPTURE


def test_export_rejects_hidden_paths_digest_spoofing_and_nonrigid_pose(tmp_path: Path) -> None:
    source = tmp_path / "source"
    request = _request(source)
    hidden = source / "frozen/evaluator_hidden/held_out/frame-1.png"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes((source / "frozen/candidate_dataset/training/frame-1.png").read_bytes())
    request["camera_observation_manifest"]["observations"][0]["image_relative_path"] = (
        "frozen/evaluator_hidden/held_out/frame-1.png"
    )
    request["camera_observation_manifest"]["camera_observation_digest"] = canonical_digest(
        request["camera_observation_manifest"], digest_field="camera_observation_digest"
    )
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    with pytest.raises(ColmapTrainingDatasetError, match="unsafe_or_hidden"):
        export_colmap_training_dataset(
            source_artifact=request, artifact_root=source, output_root=tmp_path / "out-hidden"
        )

    request = _request(source)
    request["camera_observation_manifest"]["observations"][0]["image_digest"] = "sha256:" + "f" * 64
    request["camera_observation_manifest"]["camera_observation_digest"] = canonical_digest(
        request["camera_observation_manifest"], digest_field="camera_observation_digest"
    )
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    with pytest.raises(ColmapTrainingDatasetError, match="manifest_mismatch"):
        export_colmap_training_dataset(
            source_artifact=request, artifact_root=source, output_root=tmp_path / "out-digest"
        )

    request = _request(source)
    request["camera_observation_manifest"]["observations"][0]["camera"]["T_world_camera"][0][0] = (
        2.0
    )
    request["camera_observation_manifest"]["observations"][0]["T_world_camera"][0][0] = 2.0
    request["camera_observation_manifest"]["camera_observation_digest"] = canonical_digest(
        request["camera_observation_manifest"], digest_field="camera_observation_digest"
    )
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    with pytest.raises(ColmapTrainingDatasetError, match="camera_binding_invalid"):
        export_colmap_training_dataset(
            source_artifact=request, artifact_root=source, output_root=tmp_path / "out-pose"
        )


def test_recorded_real_export_is_candidate_only_and_self_digesting() -> None:
    receipt = json.loads(RECORDED_REAL_EXPORT.read_text(encoding="utf-8"))
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/colmap_training_dataset_export_result.v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(receipt, schema)
    assert receipt["colmap_training_dataset_export_result_digest"] == canonical_digest(
        receipt, digest_field="colmap_training_dataset_export_result_digest"
    )
    assert receipt["image_count"] == 32
    assert len(receipt["observation_ids"]) == 32
    assert len(set(receipt["observation_ids"])) == 32
    assert receipt["rejected_observation_ids"] == []
    assert receipt["initialization_point_count"] == 83757
    assert receipt["hidden_heldout_pixels_included"] is False
    assert receipt["raw_input_poses_modified"] is False
    assert receipt["pose_refinement_executed"] is False
    assert receipt["trainer_self_grading_permitted"] is False
    assert receipt["claim_ceiling"] == "reconstruction_training_request"
    assert receipt["cost_usd"] == 0.0
    assert "resolved_worker_image_missing" in receipt["blockers"]
