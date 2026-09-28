from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_rigid_destination_placement_qualification import (
    RigidDestinationPlacementQualificationError,
    materialize_rigid_destination_placement_qualification,
)
from tests.test_task_evaluation_native_arena_episode_compiler import (
    _destination_case,
)


def _read_reference(references: dict[str, dict], key: str) -> tuple[Path, dict]:
    path = Path(references[key]["materialized_path"])
    return path, json.loads(path.read_text(encoding="utf-8"))


def _observation(tmp_path: Path) -> tuple[dict, dict[str, dict], dict]:
    _request, references, context = _destination_case(tmp_path)
    asset = Path(references["task.destination.asset"]["materialized_path"])
    static_path, _static = _read_reference(
        references, "task.destination.static_qualification"
    )
    native_path, _native = _read_reference(
        references, "task.destination.native_import_qualification"
    )
    geometry_path, geometry = _read_reference(
        references, "task.destination.geometry"
    )
    pose = geometry["pose_world"]
    pose_row = [*pose["position_world_m"], *pose["orientation_xyzw"]]
    measurements = tmp_path / "measurements.json"
    measurements.write_bytes(b'{"native":true}\n')
    placement_path, placement = _read_reference(
        references, "task.destination.placement_qualification"
    )
    del placement_path
    value = {
        "schema_version": "task_evaluation_rigid_destination_native_observation.v1",
        "status": "completed",
        "producer": "native_task_arena_destination_qualification",
        "native_isaac_executed": True,
        "execution_commit": "a" * 40,
        "runtime_identity": {"id": "isaac-arena", "version": "5.1"},
        "container_identity": {
            "image": "registry.example/isaac@sha256:" + "b" * 64,
            "digest": "sha256:" + "b" * 64,
        },
        "destination_identity": geometry["destination_identity"],
        "configured_scene_revision_digest": placement[
            "configured_scene_revision_digest"
        ],
        "configured_scene_collision_digest": placement[
            "configured_scene_collision_digest"
        ],
        "configured_scene_support_plane_digest": "sha256:" + "c" * 64,
        "destination_asset_digest": references["task.destination.asset"][
            "digest"
        ],
        "destination_static_qualification_digest": references[
            "task.destination.static_qualification"
        ]["digest"],
        "destination_native_import_qualification_digest": references[
            "task.destination.native_import_qualification"
        ]["digest"],
        "destination_geometry_digest": geometry["geometry_digest"],
        "pose_world": pose,
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
        "settle_samples": [
            {
                "sample_index": index,
                "destination_pose_world": pose_row,
                "maximum_penetration_m": 0.0001,
                "support_contact_peak_force_n": 1.0,
                "forbidden_contact_peak_force_n": 0.0,
            }
            for index in range(3)
        ],
        "reset_samples": [
            {"sample_index": index, "destination_pose_world": pose_row}
            for index in range(3)
        ],
        "camera_observations": [
            {
                "role": role,
                "task_support_pixel_count": 250,
                "camera_calibration": {"fx": 320.0},
                "render_receipt_digest": "sha256:" + character * 64,
            }
            for role, character in zip(
                ("external", "wrist", "overview"), ("d", "e", "f"), strict=True
            )
        ],
        "raw_measurement_artifacts": [
            {
                "role": "native_measurements",
                "relative_path": "measurements.json",
                "sha256": "sha256:"
                + hashlib.sha256(measurements.read_bytes()).hexdigest(),
                "size_bytes": measurements.stat().st_size,
            }
        ],
        "no_policy_execution": {
            "policy_loaded": False,
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
            "policy_actions_executed": 0,
        },
        "observation_digest": "",
    }
    value["observation_digest"] = canonical_digest(
        value, digest_field="observation_digest"
    )
    paths = {
        "asset": asset,
        "static": static_path,
        "native": native_path,
        "geometry": geometry_path,
        "collision": context["configured_collision_path"],
    }
    return value, references, paths


def test_native_samples_materialize_scene_bound_placement_qualification(
    tmp_path: Path,
) -> None:
    observation, _references, paths = _observation(tmp_path)
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")

    result = materialize_rigid_destination_placement_qualification(
        observation_path=observation_path,
        configured_scene_collision_path=paths["collision"],
        destination_asset_path=paths["asset"],
        destination_static_qualification_path=paths["static"],
        destination_native_import_qualification_path=paths["native"],
        destination_geometry_path=paths["geometry"],
        output_path=tmp_path / "qualified.json",
    )

    assert result["status"] == "qualified"
    assert result["camera_visibility"] == {
        "external": True,
        "wrist": True,
        "overview": True,
    }
    assert result["repeated_reset_readback"]["repeat_count"] == 3
    assert result["native_observation_digest"] == observation[
        "observation_digest"
    ]
    assert result["execution_commit"] == "a" * 40
    assert result["native_measurement_summary"] == {
        "maximum_penetration_m": 0.0001,
        "minimum_support_contact_force_n": 1.0,
        "maximum_forbidden_contact_force_n": 0.0,
        "raw_measurement_artifact_count": 1,
    }
    assert result["placement_qualification_digest"] == canonical_digest(
        result, digest_field="placement_qualification_digest"
    )


def test_native_placement_qualification_refuses_invisible_destination(
    tmp_path: Path,
) -> None:
    observation, _references, paths = _observation(tmp_path)
    observation["camera_observations"][1]["task_support_pixel_count"] = 0
    observation["observation_digest"] = canonical_digest(
        observation, digest_field="observation_digest"
    )
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps(observation), encoding="utf-8")

    with pytest.raises(
        RigidDestinationPlacementQualificationError,
        match="rigid_destination_placement_native_qualification_failed",
    ):
        materialize_rigid_destination_placement_qualification(
            observation_path=observation_path,
            configured_scene_collision_path=paths["collision"],
            destination_asset_path=paths["asset"],
            destination_static_qualification_path=paths["static"],
            destination_native_import_qualification_path=paths["native"],
            destination_geometry_path=paths["geometry"],
            output_path=tmp_path / "qualified.json",
        )
