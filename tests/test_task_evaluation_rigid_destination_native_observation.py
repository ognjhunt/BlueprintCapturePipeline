from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_rigid_destination_native_observation import (
    RigidDestinationNativeObservationError,
    materialize_rigid_destination_native_observation,
)


IMAGE = "registry.example/isaac@sha256:" + "a" * 64


def _request() -> dict:
    value = {
        "schema_version": "task_evaluation_rigid_destination_native_probe_request.v1",
        "execution_commit": "b" * 40,
        "runtime_identity": {"id": "isaac-arena", "version": "5.1"},
        "container_identity": {"image": IMAGE, "digest": "sha256:" + "a" * 64},
        "destination_identity": {"id": "blue-document-tray", "version": "v1"},
        "configured_scene_revision_digest": "sha256:" + "1" * 64,
        "configured_scene_collision_digest": "sha256:" + "2" * 64,
        "configured_scene_support_plane_digest": "sha256:" + "3" * 64,
        "destination_asset_digest": "sha256:" + "4" * 64,
        "destination_static_qualification_digest": "sha256:" + "5" * 64,
        "destination_native_import_qualification_digest": "sha256:" + "6" * 64,
        "destination_geometry_digest": "sha256:" + "7" * 64,
        "pose_world": {
            "position_world_m": [1.0, 2.0, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "qualification_limits": {
            "maximum_penetration_m": 0.001,
            "minimum_support_contact_force_n": 0.01,
            "maximum_forbidden_contact_force_n": 0.1,
            "settle_translation_tolerance_m": 0.002,
            "settle_rotation_tolerance_rad": 0.01,
            "reset_translation_tolerance_m": 0.002,
            "reset_rotation_tolerance_rad": 0.01,
            "minimum_camera_pixels": {
                "external": 100,
                "wrist": 100,
                "overview": 100,
            },
        },
        "settle_sample_count": 3,
        "settle_steps_per_sample": 60,
        "candidate_policy_queried": False,
        "policy_loaded": False,
        "request_digest": "",
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


def _pose() -> list[float]:
    return [1.0, 2.0, 0.3, 0.0, 0.0, 0.0, 1.0]


def test_executor_seals_exact_release_measurements_and_artifact_bytes(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "external.png"
    frame.write_bytes(b"lossless-frame")
    request = _request()
    result = materialize_rigid_destination_native_observation(
        request=request,
        execution_manifest={
            "implementation_commit": request["execution_commit"],
            "container_image": IMAGE,
            "execution_mode": "destination_qualification",
            "policy_candidate_id": None,
        },
        settle_samples=[
            {
                "sample_index": index,
                "destination_pose_world": _pose(),
                "maximum_penetration_m": 0.0001,
                "support_contact_peak_force_n": 1.0,
                "forbidden_contact_peak_force_n": 0.0,
            }
            for index in range(3)
        ],
        reset_samples=[
            {"sample_index": index, "destination_pose_world": _pose()}
            for index in range(3)
        ],
        camera_observations=[
            {
                "role": role,
                "task_support_pixel_count": 250,
                "camera_calibration": {"fx": 320.0},
                "render_receipt_digest": "sha256:" + character * 64,
            }
            for role, character in zip(
                ("external", "wrist", "overview"), ("8", "9", "a"), strict=True
            )
        ],
        raw_measurement_artifacts=[
            {"role": "external_render", "relative_path": frame.name}
        ],
        artifact_root=tmp_path,
        output_path=tmp_path / "observation.json",
    )

    assert result["status"] == "completed"
    assert result["execution_commit"] == "b" * 40
    assert result["raw_measurement_artifacts"][0]["size_bytes"] == 14
    assert result["no_policy_execution"]["policy_actions_executed"] == 0
    assert result["observation_digest"] == canonical_digest(
        result, digest_field="observation_digest"
    )
    assert json.loads((tmp_path / "observation.json").read_text()) == result


def test_executor_refuses_manifest_commit_rebinding(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"frame")
    request = _request()
    with pytest.raises(
        RigidDestinationNativeObservationError,
        match="rigid_destination_native_probe_execution_manifest_invalid",
    ):
        materialize_rigid_destination_native_observation(
            request=request,
            execution_manifest={
                "implementation_commit": "c" * 40,
                "container_image": IMAGE,
                "execution_mode": "destination_qualification",
                "policy_candidate_id": None,
            },
            settle_samples=[],
            reset_samples=[],
            camera_observations=[],
            raw_measurement_artifacts=[
                {"role": "render", "relative_path": frame.name}
            ],
            artifact_root=tmp_path,
            output_path=tmp_path / "observation.json",
        )
