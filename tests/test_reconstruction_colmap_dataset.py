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
    export_colmap_training_dataset,
)


CAPTURE = "sha256:" + "a" * 64
DATASET = "sha256:" + "b" * 64
SPLIT = "sha256:" + "c" * 64
COMMIT = "d" * 40


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
        "candidate_dataset_manifest": candidate,
        "initialization_surface": {
            "relative_path": "surface/observed.json",
            "digest": _digest(surface_path),
        },
        "maximum_initialization_points": 10,
        "blockers": ["pose_refinement_not_executed"],
    }
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    return request


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
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    with pytest.raises(ColmapTrainingDatasetError, match="unsafe_or_hidden"):
        export_colmap_training_dataset(
            source_artifact=request, artifact_root=source, output_root=tmp_path / "out-hidden"
        )

    request = _request(source)
    request["camera_observation_manifest"]["observations"][0]["image_digest"] = "sha256:" + "f" * 64
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
    request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        request, digest_field="colmap_training_dataset_export_request_digest"
    )
    with pytest.raises(ColmapTrainingDatasetError, match="camera_binding_invalid"):
        export_colmap_training_dataset(
            source_artifact=request, artifact_root=source, output_root=tmp_path / "out-pose"
        )
