from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.adp009d_live_hybrid_frames import (
    BLOCKER_APPROVED_OBJECT_ABSENT,
    BLOCKER_APPROVED_OBJECT_OCCLUDED,
    BLOCKER_SEMANTIC_OVERRIDE_MISSING,
    FRAME_MANIFEST_SCHEMA_VERSION,
    LiveHybridFrameError,
    materialize_live_hybrid_observation_frames,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

RESOLUTION = (4, 4)
TIMESTAMP_NS = 1786088709228774736
SIM_TIME_S = 2.6666666666666665
FRAME_INDEX = 40

# Identity OpenGL orientation -> OpenCV basis is diag(1, -1, -1).
QUATERNION_XYZW = [0.0, 0.0, 0.0, 1.0]
# Chosen so the object surface lands ~0.10 m above the support plane,
# i.e. on the object body rather than in the shelf-contact band.
POSITION_WORLD_M = [0.25, -0.5, 1.6112]
INTRINSIC = [[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]
WORLD_FROM_CAMERA = [
    [1.0, 0.0, 0.0, POSITION_WORLD_M[0]],
    [0.0, -1.0, 0.0, POSITION_WORLD_M[1]],
    [0.0, 0.0, -1.0, POSITION_WORLD_M[2]],
    [0.0, 0.0, 0.0, 1.0],
]


def _sha(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration() -> dict:
    return {
        "camera_coordinate_convention": "OpenCV_right_down_forward",
        "camera_model": "pinhole",
        "intrinsic_matrix": INTRINSIC,
        "resolution": [RESOLUTION[1], RESOLUTION[0]],
        "world_from_camera": WORLD_FROM_CAMERA,
    }


def _build(
    tmp_path: Path,
    *,
    aura_depth_value: float = 5.0,
    include_approved_can: bool = True,
    declare_override: bool = True,
    quaternion: list[float] | None = None,
    timestamp_ns: int = TIMESTAMP_NS,
) -> tuple[Path, Path]:
    """Write a minimal but structurally exact Aura + Isaac evidence pair."""

    aura_root = tmp_path / "aura_exec"
    isaac_root = tmp_path / "isaac_exec"
    (aura_root / "external_camera").mkdir(parents=True)
    (isaac_root / "camera_frames" / "external_camera").mkdir(parents=True)

    height, width = RESOLUTION
    aura_rgb = np.full((height, width, 3), 10, dtype=np.uint8)
    aura_depth = np.full((height, width), aura_depth_value, dtype=np.float32)
    np.save(aura_root / "external_camera/rgb.npy", aura_rgb)
    np.save(aura_root / "external_camera/depth_m.npy", aura_depth)

    segmentation = np.zeros((height, width), dtype=np.int32)
    segmentation[0, 0] = 2  # robot
    if include_approved_can:
        segmentation[1, 1] = 3  # approved_can
    depth = np.full((height, width, 1), np.inf, dtype=np.float32)
    depth[segmentation > 0] = 1.0
    np.save(
        isaac_root / "camera_frames/external_camera/000040.distance_to_camera.npy", depth
    )
    np.save(
        isaac_root / "camera_frames/external_camera/000040.semantic.npy", segmentation
    )
    Image.fromarray(np.full((height, width, 3), 100, dtype=np.uint8)).save(
        isaac_root / "camera_frames/external_camera/000040.png"
    )

    id_to_labels = {"0": {"class": "BACKGROUND"}, "2": {"class": "robot"}}
    if include_approved_can:
        id_to_labels["3"] = {"class": "approved_can"}

    rgb_sha = _sha(isaac_root / "camera_frames/external_camera/000040.png")
    depth_sha = _sha(
        isaac_root / "camera_frames/external_camera/000040.distance_to_camera.npy"
    )
    semantic_sha = _sha(isaac_root / "camera_frames/external_camera/000040.semantic.npy")

    isaac_result = {
        "schema_version": "adp009d_native_microcheck.v1",
        "status": "completed",
        "sealed_source_mutated": False,
        "semantic_override_layer_digest": ("sha256:" + "d" * 64)
        if declare_override
        else None,
        "camera_frames": [
            {
                "camera_id": "external_camera",
                "frame_index": FRAME_INDEX,
                "timestamp_ns": timestamp_ns,
                "sim_time_seconds": SIM_TIME_S,
                "intrinsic_matrix": INTRINSIC,
                "resolution_hw": [height, width],
                "position_world_m": POSITION_WORLD_M,
                "quaternion_world_opengl_xyzw": quaternion or QUATERNION_XYZW,
                "metric_depth": {
                    "aov": "distance_to_camera",
                    "path": "camera_frames/external_camera/000040.distance_to_camera.npy",
                    "sha256": depth_sha,
                    "units": "meter",
                },
                "rgb_png": {
                    "path": "camera_frames/external_camera/000040.png",
                    "sha256": rgb_sha,
                },
                "semantic_segmentation": {
                    "dtype": "int32",
                    "path": "camera_frames/external_camera/000040.semantic.npy",
                    "sha256": semantic_sha,
                    "id_to_labels": {"idToLabels": id_to_labels},
                },
            }
        ],
    }
    isaac_path = isaac_root / "adp009d_native_microcheck.json"
    isaac_path.write_text(json.dumps(isaac_result), encoding="utf-8")

    calibration = _calibration()
    aura_result = {
        "schema_version": "adp009d_aura_native_live_camera_result.v1",
        "status": "completed",
        "blockers": [],
        "candidate_policy_queried": False,
        "implementation_commit": "0" * 40,
        "source_probe_manifest_digest": "sha256:" + "e" * 64,
        "camera_rows": [
            {
                "camera_id": "external_camera",
                "valid": True,
                "calibration": calibration,
                "calibration_digest": canonical_digest(calibration),
                "artifacts": [
                    {
                        "path": "external_camera/rgb.npy",
                        "sha256": _sha(aura_root / "external_camera/rgb.npy"),
                        "dtype": "uint8",
                    },
                    {
                        "path": "external_camera/depth_m.npy",
                        "sha256": _sha(aura_root / "external_camera/depth_m.npy"),
                        "dtype": "float32",
                    },
                ],
                "source_isaac_frame_index": FRAME_INDEX,
                "source_isaac_timestamp_ns": TIMESTAMP_NS,
                "source_isaac_sim_time_seconds": SIM_TIME_S,
                "source_isaac_input_artifacts": {
                    "dynamic_rgb": {"sha256": rgb_sha},
                    "dynamic_depth": {"sha256": depth_sha},
                    "dynamic_semantic": {"sha256": semantic_sha},
                },
            }
        ],
    }
    aura_path = aura_root / "adp009d_aura_native_live_camera_result.json"
    aura_path.write_text(json.dumps(aura_result), encoding="utf-8")
    return aura_path, isaac_path


def _materialize(tmp_path: Path, **kwargs) -> dict:
    aura_path, isaac_path = _build(tmp_path, **kwargs)
    return materialize_live_hybrid_observation_frames(
        aura_native_result_path=aura_path,
        isaac_native_result_path=isaac_path,
        output_dir=tmp_path / "out",
        generated_at="2026-08-07T00:00:00Z",
    )


def test_materializes_digest_bound_composed_frames(tmp_path: Path) -> None:
    manifest = _materialize(tmp_path)

    assert manifest["schema_version"] == FRAME_MANIFEST_SCHEMA_VERSION
    assert manifest["blockers"] == []
    assert manifest["admission_status"] == "frame_composition_only"
    assert manifest["candidate_policy_queried"] is False
    assert manifest["manifest_digest"] == canonical_digest(
        manifest, digest_field="manifest_digest"
    )

    camera = manifest["cameras"][0]
    assert camera["approved_task_object_present"] is True
    assert camera["approved_task_object_occluded_pixel_count"] == 0
    assert camera["frame_receipt"]["dynamic_front_pixel_count"] == 2
    assert camera["frame_receipt"]["dynamic_occluded_pixel_count"] == 0
    # Isaac ray-length depth must have been normalised before ordering.
    assert camera["frame_receipt"]["dynamic_depth_convention"] == "ray_length"
    assert camera["frame_receipt"]["depth_comparison_convention"] == "camera_z"
    # Retained outputs exist and hash to the recorded digests.
    out = tmp_path / "out"
    for entry in camera["retained_outputs"].values():
        assert _sha(out / entry["path"]) == entry["sha256"]


def test_records_blocker_when_aura_layer_occludes_approved_object(
    tmp_path: Path,
) -> None:
    manifest = _materialize(tmp_path, aura_depth_value=0.5)

    assert BLOCKER_APPROVED_OBJECT_OCCLUDED in manifest["blockers"]
    assert manifest["admission_status"] == "blocked"
    camera = manifest["cameras"][0]
    assert camera["approved_task_object_occluded_pixel_count"] == 1
    # Occlusion of the object body, not of its shelf-contact line.
    assert camera["approved_task_object_body_occluded_pixel_count"] == 1
    assert camera["approved_task_object_contact_occluded_pixel_count"] == 0


def test_records_blocker_when_approved_object_absent(tmp_path: Path) -> None:
    manifest = _materialize(tmp_path, include_approved_can=False)

    assert BLOCKER_APPROVED_OBJECT_ABSENT in manifest["blockers"]
    assert manifest["admission_status"] == "blocked"


def test_records_blocker_when_isaac_omits_semantic_override_digest(
    tmp_path: Path,
) -> None:
    manifest = _materialize(tmp_path, declare_override=False)

    assert BLOCKER_SEMANTIC_OVERRIDE_MISSING in manifest["blockers"]
    assert (
        manifest["cameras"][0]["semantic_override_layer_provenance"]
        == "derived_from_observed_isaac_id_to_labels"
    )


def test_verifies_declared_semantic_override_digest_against_its_body(
    tmp_path: Path,
) -> None:
    """A declared override digest is recomputed, never taken on trust."""

    aura_path, isaac_path = _build(tmp_path)
    isaac_result = json.loads(isaac_path.read_text())
    override = {
        "authoring": "isaac_lab_spawn_semantic_tags_runtime_override",
        "sealed_source_usd_mutated": False,
        "tags": {"robot": [["class", "robot"]]},
    }
    isaac_result["semantic_override_layer"] = override
    isaac_result["semantic_override_layer_digest"] = canonical_digest(override)
    isaac_path.write_text(json.dumps(isaac_result), encoding="utf-8")
    manifest = materialize_live_hybrid_observation_frames(
        aura_native_result_path=aura_path,
        isaac_native_result_path=isaac_path,
        output_dir=tmp_path / "out",
    )
    assert BLOCKER_SEMANTIC_OVERRIDE_MISSING not in manifest["blockers"]
    assert (
        manifest["cameras"][0]["semantic_override_layer_provenance"]
        == "declared_by_isaac_runtime"
    )

    isaac_result["semantic_override_layer"]["tags"]["robot"] = [["class", "spoofed"]]
    isaac_path.write_text(json.dumps(isaac_result), encoding="utf-8")
    with pytest.raises(
        LiveHybridFrameError, match="hybrid_frame_semantic_override_digest_mismatch"
    ):
        materialize_live_hybrid_observation_frames(
            aura_native_result_path=aura_path,
            isaac_native_result_path=isaac_path,
            output_dir=tmp_path / "out2",
        )


def test_rejects_tampered_isaac_semantic_bytes(tmp_path: Path) -> None:
    aura_path, isaac_path = _build(tmp_path)
    semantic = isaac_path.parent / "camera_frames/external_camera/000040.semantic.npy"
    tampered = np.load(semantic)
    tampered[2, 2] = 3
    np.save(semantic, tampered)

    with pytest.raises(
        LiveHybridFrameError, match="hybrid_frame_isaac_semantic_digest_mismatch"
    ):
        materialize_live_hybrid_observation_frames(
            aura_native_result_path=aura_path,
            isaac_native_result_path=isaac_path,
            output_dir=tmp_path / "out",
        )


def test_rejects_temporal_identity_mismatch(tmp_path: Path) -> None:
    aura_path, isaac_path = _build(tmp_path, timestamp_ns=TIMESTAMP_NS + 1)

    with pytest.raises(
        LiveHybridFrameError, match="hybrid_frame_temporal_identity_mismatch"
    ):
        materialize_live_hybrid_observation_frames(
            aura_native_result_path=aura_path,
            isaac_native_result_path=isaac_path,
            output_dir=tmp_path / "out",
        )


def test_rejects_camera_pose_conversion_mismatch(tmp_path: Path) -> None:
    """A probe-side OpenGL/OpenCV error must not reach a policy observation."""

    aura_path, isaac_path = _build(tmp_path, quaternion=[0.0, 0.7071068, 0.0, 0.7071068])

    with pytest.raises(
        LiveHybridFrameError, match="hybrid_frame_camera_pose_conversion_mismatch"
    ):
        materialize_live_hybrid_observation_frames(
            aura_native_result_path=aura_path,
            isaac_native_result_path=isaac_path,
            output_dir=tmp_path / "out",
        )
