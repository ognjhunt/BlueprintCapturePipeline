from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.processed_observation_dataset import (
    REQUEST_SCHEMA_VERSION,
    ProcessedObservationDatasetError,
    build_processed_observation_dataset_request,
    compile_bound_processed_observation_dataset,
)
from blueprint_pipeline.reconstruction_colmap_dataset import (
    export_colmap_training_dataset,
)
from scripts.run_public_processed_observations import (
    run_public_processed_observation_replay,
)


SOURCE_BUNDLE_BYTES = b"processed-public-dataset-source-bundle"
SOURCE_BUNDLE_DIGEST = "sha256:" + hashlib.sha256(SOURCE_BUNDLE_BYTES).hexdigest()
SOURCE_COMMIT = "b" * 40
TIMESTAMP = "2026-08-01T00:00:00Z"
SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "schemas"
        / "processed_observation_dataset.v1.schema.json"
    ).read_text(encoding="utf-8")
)


def _write_trajectory(
    root: Path,
    *,
    relative_path: str,
    count: int,
    bad_rotation_index: int | None = None,
) -> None:
    transformations_path = root / relative_path
    capture_root = transformations_path.parent
    frames = []
    for index in range(1, count + 1):
        image = capture_root / "images" / f"frame_{index:05d}.jpg"
        depth = capture_root / "depth" / f"frame_{index:05d}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        depth.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"rgb-{relative_path}-{index}".encode())
        depth.write_bytes(f"depth-{relative_path}-{index}".encode())
        rotation = [
            [1.0, 0.0, 0.0, float(index) * 0.1],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        if bad_rotation_index == index:
            rotation[0][0] = 2.0
        frames.append(
            {
                "fl_x": 80.0,
                "fl_y": 80.0,
                "cx": 32.0,
                "cy": 24.0,
                "h": 48,
                "w": 64,
                "file_path": f"./images/{image.name}",
                "depth_file_path": f"./depth/{depth.name}",
                "transform_matrix": rotation,
            }
        )
    transformations_path.parent.mkdir(parents=True, exist_ok=True)
    transformations_path.write_text(
        json.dumps({"camera_model": "OPENCV", "frames": frames}), encoding="utf-8"
    )


def _source_fixture(
    root: Path,
    *,
    heldout_name: str = "frame_00002",
    bad_rotation_index: int | None = None,
) -> None:
    _write_trajectory(
        root,
        relative_path="long_capture/transformations.json",
        count=5,
        bad_rotation_index=bad_rotation_index,
    )
    _write_trajectory(
        root,
        relative_path="short_capture/transformations.json",
        count=2,
    )
    (root / "long_capture" / "test.txt").write_text(
        heldout_name + "\n", encoding="utf-8"
    )


def _request() -> dict:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "dataset_id": "mushroom",
        "scene_id": "koivu",
        "source_bundle_digest": SOURCE_BUNDLE_DIGEST,
        "source_bundle_size_bytes": len(SOURCE_BUNDLE_BYTES),
        "source_bundle_uri": "https://doi.org/10.5281/zenodo.10230733",
        "license_id": "CC-BY-4.0",
        "long_transformations_relative_path": "long_capture/transformations.json",
        "declared_heldout_ids_relative_path": "long_capture/test.txt",
        "independent_transformations_relative_path": "short_capture/transformations.json",
        "source_commit_sha": SOURCE_COMMIT,
        "authority_used": {
            "local_processing_allowed": True,
            "external_provider_upload_allowed": False,
            "privacy_scope": "restricted_local_only",
        },
        "coordinate_frame_declaration": {
            "source": "dataset_transformations_json",
            "camera_convention": "camera_to_world",
            "world_up": "not_independently_verified",
            "metric_scale": "dataset_declared_not_independently_verified",
        },
        "timestamp": TIMESTAMP,
    }


def _compile(source: Path, output: Path) -> dict:
    source_bundle = source.parent / f"{source.name}.tar.gz"
    source_bundle.write_bytes(SOURCE_BUNDLE_BYTES)
    return compile_bound_processed_observation_dataset(
        source_artifact=_request(),
        source_bundle=source_bundle,
        dataset_root=source,
        output_root=output,
    )


def _read_artifact(output: Path, dataset: dict, name: str) -> dict:
    reference = dataset["artifact_references"][name]
    return json.loads((output / reference["relative_path"]).read_text(encoding="utf-8"))


def test_processed_observations_freeze_candidate_and_evaluator_lanes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _source_fixture(source)

    dataset = _compile(source, output)

    assert dataset["stream_metadata"] == {
        "long_trajectory_frames": 5,
        "independent_trajectory_frames": 2,
        "candidate_frames": 4,
        "hidden_heldout_frames": 3,
    }
    assert dataset["proof_effect"] == "processed_captured_observation_availability_only"
    assert dataset["claim_flags"] == {
        "processed_captured_observation": True,
        "raw_capture_authority": False,
        "decoded_video_timing": False,
        "metric_scale_verified": False,
        "collision_geometry": False,
        "physics": False,
        "physical_task_success": False,
        "deployment_readiness": False,
        "safety_certification": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    candidate = _read_artifact(output, dataset, "candidate_dataset_manifest")
    hidden = _read_artifact(output, dataset, "hidden_heldout_evaluator_manifest")
    observations = _read_artifact(
        output, dataset, "candidate_camera_observation_manifest"
    )
    jsonschema.validate(dataset, SCHEMA)
    for name in dataset["artifact_references"]:
        jsonschema.validate(_read_artifact(output, dataset, name), SCHEMA)
    assert candidate["heldout_pixels_included"] is False
    assert hidden["candidate_method_access_allowed"] is False
    assert observations["hidden_heldout_pixels_included"] is False
    assert {row["frame_id"] for row in candidate["frames"]}.isdisjoint(
        {row["frame_id"] for row in hidden["frames"]}
    )
    assert {row["heldout_reason"] for row in hidden["frames"]} == {
        "dataset_declared_long_trajectory_test_view",
        "independent_evaluation_trajectory",
    }


def test_processed_observation_replay_is_idempotent_and_colmap_compatible(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _source_fixture(source)
    first = _compile(source, output)
    second = _compile(source, output)
    assert second == first

    candidate = _read_artifact(output, first, "candidate_dataset_manifest")
    split = _read_artifact(output, first, "frozen_split_manifest")
    observations = _read_artifact(
        output, first, "candidate_camera_observation_manifest"
    )
    artifact_root = output / first["relative_path"]
    result = export_colmap_training_dataset(
        source_artifact={
            "schema_version": "colmap_training_dataset_export_request.v1",
            "stable_run_identity": first["stable_run_identity"],
            "source_capture_digest": SOURCE_BUNDLE_DIGEST,
            "reconstruction_dataset_digest": first["dataset_manifest_digest"],
            "frozen_split_digest": split["split_digest"],
            "source_commit_sha": SOURCE_COMMIT,
            "camera_observation_manifest": observations,
            "candidate_dataset_manifest": candidate,
            "coordinate_frame_declaration": first["coordinate_frame_declaration"],
            "units": first["units"],
            "metric_scale_status": first["metric_scale_status"],
            "authority_used": first["authority_used"],
            "timestamp": TIMESTAMP,
        },
        artifact_root=artifact_root,
        output_root=tmp_path / "colmap",
    )
    assert result["status"] == "exported_candidate_only_colmap_text_dataset"
    assert result["image_count"] == 4
    assert result["hidden_heldout_pixels_included"] is False
    assert result["pose_refinement_executed"] is False


def test_processed_observation_request_is_closed_digest_bound_and_secret_safe() -> None:
    request = build_processed_observation_dataset_request(_request())
    jsonschema.validate(request, SCHEMA)
    replay = build_processed_observation_dataset_request(request)
    assert replay == request

    unknown = _request()
    unknown["provider_token"] = "do-not-store"
    with pytest.raises(ProcessedObservationDatasetError) as caught:
        build_processed_observation_dataset_request(unknown)
    assert caught.value.codes == (
        "processed_request_unknown_field:provider_token",
    )

    credential_uri = _request()
    credential_uri["source_bundle_uri"] = "https://user:secret@example.com/data"
    with pytest.raises(ProcessedObservationDatasetError) as caught:
        build_processed_observation_dataset_request(credential_uri)
    assert caught.value.codes == ("processed_request_source_bundle_uri_invalid",)

    tampered = dict(request)
    tampered["scene_id"] = "different"
    with pytest.raises(ProcessedObservationDatasetError) as caught:
        build_processed_observation_dataset_request(tampered)
    assert caught.value.codes == ("processed_request_digest_mismatch",)


def test_processed_observation_execution_verifies_source_bundle_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _source_fixture(source)
    source_bundle = tmp_path / "source.tar.gz"
    source_bundle.write_bytes(b"different-bundle-with-same-size"[: len(SOURCE_BUNDLE_BYTES)])
    with pytest.raises(ProcessedObservationDatasetError) as caught:
        compile_bound_processed_observation_dataset(
            source_artifact=_request(),
            source_bundle=source_bundle,
            dataset_root=source,
            output_root=tmp_path / "output",
        )
    assert set(caught.value.codes) <= {
        "processed_source_bundle_size_mismatch",
        "processed_source_bundle_digest_mismatch",
    }


def test_processed_observations_reject_unknown_heldout_and_invalid_pose(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "unknown"
    _source_fixture(unknown, heldout_name="frame_99999")
    with pytest.raises(ProcessedObservationDatasetError) as caught:
        _compile(unknown, tmp_path / "unknown-output")
    assert caught.value.codes == (
        "declared_heldout_ids_not_subset_of_long_trajectory",
    )

    invalid = tmp_path / "invalid"
    _source_fixture(invalid, bad_rotation_index=3)
    with pytest.raises(ProcessedObservationDatasetError) as caught:
        _compile(invalid, tmp_path / "invalid-output")
    assert "processed_rotation_not_orthonormal:long:frame_00003" in caught.value.codes


def test_processed_observations_reject_symlinked_pixels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_fixture(source)
    image = source / "long_capture" / "images" / "frame_00001.jpg"
    replacement = source / "replacement.jpg"
    replacement.write_bytes(image.read_bytes())
    image.unlink()
    try:
        image.symlink_to(replacement)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ProcessedObservationDatasetError) as caught:
        _compile(source, tmp_path / "output")
    assert caught.value.codes == (
        "processed_frame_artifact_missing_or_symlink:long:0",
    )


def test_public_processed_observation_replay_is_repeatable_and_bounded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    bundle = tmp_path / "source.tar.gz"
    _source_fixture(source)
    bundle.write_bytes(SOURCE_BUNDLE_BYTES)
    kwargs = {
        "dataset_id": "mushroom",
        "scene_id": "koivu",
        "source_bundle": bundle,
        "source_bundle_sha256": SOURCE_BUNDLE_DIGEST,
        "source_bundle_uri": "https://doi.org/10.5281/zenodo.10230733",
        "license_id": "CC-BY-4.0",
        "dataset_root": source,
        "long_transformations_relative_path": "long_capture/transformations.json",
        "declared_heldout_ids_relative_path": "long_capture/test.txt",
        "independent_transformations_relative_path": "short_capture/transformations.json",
        "output_root": output,
        "operator_identity": "blueprint-test-operator",
        "source_commit": SOURCE_COMMIT,
        "timestamp": TIMESTAMP,
    }
    first = run_public_processed_observation_replay(**kwargs)
    second = run_public_processed_observation_replay(**kwargs)
    assert second == first
    assert first["counts"]["candidate_frames"] == 4
    assert first["counts"]["hidden_heldout_frames"] == 3
    assert first["colmap_candidate_image_count"] == 4
    assert first["raw_capture_gate_passed"] is False
    assert first["customer_upload_gate_passed"] is False
    assert first["claim_flags"]["processed_captured_observation"] is True
    assert (
        first["claim_flags"]["comparative_policy_ranking_verdict"]
        == "thesis_not_supported"
    )
