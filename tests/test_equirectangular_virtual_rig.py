from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.equirectangular_virtual_rig import (
    EquirectangularVirtualRigError,
    compile_equirectangular_virtual_rig,
)


CAPTURE_DIGEST = "sha256:" + "a" * 64
IMPLEMENTATION_DIGEST = "sha256:" + "b" * 64
SOURCE_COMMIT = "c" * 40
STITCH_RECEIPT_DIGEST = "sha256:" + "d" * 64
ORIGINAL_SOURCE_DIGEST = "sha256:" + "e" * 64
AUTHORITY = {
    "source_capture_rights_valid": True,
    "consent_valid": True,
    "privacy_review_valid": True,
    "retention_authorized": True,
    "local_processing_authorized": True,
    "provider_upload_authorized": False,
    "paid_compute_authorized": False,
}


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _schema() -> dict:
    return json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/equirectangular_virtual_rig.v1.schema.json"
        ).read_text(encoding="utf-8")
    )


def _recorded_public_proxy_receipt() -> dict:
    return json.loads(
        (
            Path(__file__).parents[1]
            / "docs/evidence/ricoh360_bridge_equirectangular_39e3baa9.json"
        ).read_text(encoding="utf-8")
    )


def test_recorded_real_ricoh360_proxy_preserves_claim_boundaries() -> None:
    receipt = _recorded_public_proxy_receipt()

    assert receipt["status"] == "partial"
    assert receipt["validated_capture_profile"] == (
        "camera_360_equirectangular_public_dataset_proxy"
    )
    assert receipt["media_probe"]["decoded_frame_count"] == 193
    assert receipt["media_probe"]["decoded_pts_unique"] is True
    assert receipt["media_probe"]["decoded_pts_strictly_increasing"] is True
    dataset = receipt["deterministic_frame_dataset"]
    assert dataset["selected_frame_count"] == 16
    assert dataset["training_frame_count"] == 11
    assert dataset["validation_frame_count"] == 2
    assert dataset["hidden_heldout_frame_count"] == 3
    assert dataset["candidate_contains_hidden_heldout_pixels"] is False
    assert dataset["candidate_can_modify_split"] is False
    assert dataset["exact_replay"] is True
    rigs = receipt["virtual_rig_compilation"]
    assert rigs["candidate_virtual_view_count"] == 156
    assert rigs["heldout_virtual_view_count"] == 36
    assert rigs["candidate_and_evaluator_access_scopes_separate"] is True
    assert rigs["virtual_views_are_captured_evidence"] is False
    assert rigs["virtual_views_are_independent_physical_cameras"] is False
    ceiling = receipt["claim_ceiling"]
    assert ceiling["decoded_observation_availability"] is True
    assert ceiling["equirectangular_virtual_camera_rig"] is True
    assert all(
        ceiling[claim] is False
        for claim in (
            "calibrated_physical_camera_trajectory",
            "appearance_reconstruction",
            "metric_scale",
            "metric_reference_geometry",
            "collision_geometry",
            "physics_readiness",
            "isaac_load_render_compatibility",
            "simulator_task_evidence",
            "physical_task_success",
            "deployment_readiness",
        )
    )
    assert (
        receipt["rights_and_authority"]["dataset_license_separately_stated"]
        is False
    )
    assert receipt["rights_and_authority"]["provider_upload_authorized"] is False
    assert receipt["rights_and_authority"]["paid_compute_authorized"] is False
    assert receipt["runtime"]["cost_usd"] == 0.0
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def _fixture(root: Path, *, width: int = 128, height: int = 64) -> tuple[dict, list[dict]]:
    source = root / "retained" / "panorama.png"
    source.parent.mkdir(parents=True)
    longitude = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    latitude = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = longitude
    image[..., 1] = latitude
    image[..., 2] = 127
    Image.fromarray(image).save(source)
    metadata = {
        "schema_version": "stitched_equirectangular_source.v1",
        "source_capture_digest": CAPTURE_DIGEST,
        "projection": "equirectangular_2_to_1",
        "stitching_provenance": "official_sdk_produced",
        "producer_identity": "fixture-sdk-1",
        "stitching_receipt_digest": STITCH_RECEIPT_DIGEST,
        "original_360_source_preserved": True,
        "original_360_source_digest": ORIGINAL_SOURCE_DIGEST,
        "spherical_pixel_mapping": {
            "longitude": "x maps [-pi,pi) with seam at encoded x=0",
            "latitude": "y maps [+pi/2,-pi/2]",
            "pixel_centers": True,
        },
    }
    observations = [
        {
            "observation_id": "panorama-0001",
            "relative_path": "retained/panorama.png",
            "digest": _digest(source),
            "t_video_sec": 1.25,
            "split": "training",
        }
    ]
    return metadata, observations


def _compile(
    capture_root: Path,
    output_root: Path,
    metadata: dict,
    observations: list[dict],
    *,
    authority: dict | None = None,
    timestamp: str = "2026-07-30T12:00:00-05:00",
    access_scope: str = "candidate_training_and_validation_only",
) -> dict:
    return compile_equirectangular_virtual_rig(
        capture_root=capture_root,
        output_root=output_root,
        intake_id="equirectangular-fixture",
        capture_digest=CAPTURE_DIGEST,
        stitched_source_metadata=metadata,
        panorama_observations=observations,
        source_commit_sha=SOURCE_COMMIT,
        implementation_digest=IMPLEMENTATION_DIGEST,
        authority_used=authority or AUTHORITY,
        timestamp=timestamp,
        access_scope=access_scope,
        parent_artifact_or_event={"split_digest": "sha256:" + "f" * 64},
    )


def test_virtual_rig_is_idempotent_fixed_and_shared_center(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    output_root = tmp_path / "output"
    metadata, observations = _fixture(capture_root)
    source = capture_root / observations[0]["relative_path"]
    source_before = source.read_bytes()

    first = _compile(capture_root, output_root, metadata, observations)
    second = _compile(
        capture_root,
        output_root,
        metadata,
        observations,
        timestamp="2099-01-01T00:00:00Z",
    )

    assert first == second
    assert first["timestamp"] == "2026-07-30T17:00:00Z"
    assert source.read_bytes() == source_before
    assert first["stitching_provenance"] == "official_sdk_produced"
    assert first["claim_ceiling"] == "equirectangular_virtual_camera_rig"
    assert first["virtual_views_are_captured_evidence"] is False
    assert first["camera_trajectory_proven"] is False
    assert first["metric_scale_proven"] is False

    artifact_root = next(output_root.glob("equirectangular_virtual_rig_*"))
    rig = json.loads(
        (artifact_root / "equirectangular_virtual_camera_rig.json").read_text(
            encoding="utf-8"
        )
    )
    views = rig["virtual_observations"]
    assert len(views) == 12
    assert len({row["shared_optical_center_group_id"] for row in views}) == 1
    assert {(row["yaw_degrees"], row["pitch_degrees"]) for row in views} == {
        (float(yaw), float(pitch))
        for pitch in (-45, 0, 45)
        for yaw in (0, 90, 180, 270)
    }
    assert all(row["independent_physical_camera"] is False for row in views)
    assert all(
        [matrix[3] for matrix in row["T_panorama_virtual_camera"][:3]]
        == [0.0, 0.0, 0.0]
        for row in views
    )
    assert all((artifact_root / row["relative_path"]).is_file() for row in views)
    assert len({row["digest"] for row in views}) == 12
    assert rig["candidate_may_change_virtual_view_definitions"] is False
    assert rig["rig_constrained_pose_estimation_required"] is True

    validator = jsonschema.Draft202012Validator(
        _schema(), format_checker=jsonschema.FormatChecker()
    )
    validator.validate(first)
    validator.validate(rig)
    rig_reference = first["artifact_references"][
        "equirectangular_virtual_camera_rig"
    ]
    assert _digest(artifact_root / rig_reference["relative_path"]) == rig_reference[
        "digest"
    ]


def test_virtual_rig_keeps_candidate_and_evaluator_scopes_separate(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    metadata, observations = _fixture(capture_root)
    observations[0]["split"] = "held_out"

    with pytest.raises(
        EquirectangularVirtualRigError,
        match="equirectangular_observation_scope_invalid",
    ):
        _compile(capture_root, tmp_path / "candidate", metadata, observations)

    evaluator = _compile(
        capture_root,
        tmp_path / "evaluator",
        metadata,
        observations,
        access_scope="independent_evaluator_only",
    )
    artifact_root = next(
        (tmp_path / "evaluator").glob("equirectangular_virtual_rig_*")
    )
    rig = json.loads(
        (artifact_root / "equirectangular_virtual_camera_rig.json").read_text(
            encoding="utf-8"
        )
    )
    assert evaluator["claim_ceiling"] == "equirectangular_virtual_camera_rig"
    assert {row["split"] for row in rig["virtual_observations"]} == {"held_out"}
    assert all(
        row["relative_path"].startswith("virtual_views/held_out/")
        for row in rig["virtual_observations"]
    )


def test_virtual_rig_requires_stitch_provenance_and_original_source(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    metadata, observations = _fixture(capture_root)
    metadata["original_360_source_preserved"] = False

    with pytest.raises(
        EquirectangularVirtualRigError,
        match="equirectangular_source_metadata_invalid",
    ):
        _compile(capture_root, tmp_path / "output", metadata, observations)


def test_virtual_rig_requires_explicit_local_non_provider_authority(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    metadata, observations = _fixture(capture_root)
    authority = dict(AUTHORITY)
    authority["provider_upload_authorized"] = True

    with pytest.raises(
        EquirectangularVirtualRigError, match="equirectangular_authority_invalid"
    ):
        _compile(
            capture_root,
            tmp_path / "output",
            metadata,
            observations,
            authority=authority,
        )


def test_virtual_rig_rejects_digest_drift_bad_shape_and_reordered_time(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture-a"
    metadata, observations = _fixture(capture_root)
    observations[0]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(
        EquirectangularVirtualRigError,
        match="equirectangular_source_digest_mismatch",
    ):
        _compile(capture_root, tmp_path / "output-a", metadata, observations)

    capture_root = tmp_path / "capture-b"
    metadata, observations = _fixture(capture_root, width=100, height=60)
    with pytest.raises(
        EquirectangularVirtualRigError,
        match="equirectangular_source_dimensions_invalid",
    ):
        _compile(capture_root, tmp_path / "output-b", metadata, observations)

    capture_root = tmp_path / "capture-c"
    metadata, observations = _fixture(capture_root)
    observations.append(
        {
            **observations[0],
            "observation_id": "panorama-0002",
            "relative_path": "retained/panorama-copy.png",
            "t_video_sec": 1.0,
        }
    )
    source = capture_root / observations[0]["relative_path"]
    copy = capture_root / observations[1]["relative_path"]
    copy.write_bytes(source.read_bytes())
    with pytest.raises(
        EquirectangularVirtualRigError,
        match="equirectangular_observation_timing_invalid",
    ):
        _compile(capture_root, tmp_path / "output-c", metadata, observations)


def test_virtual_rig_rejects_path_traversal_symlink_and_naive_timestamp(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture-a"
    metadata, observations = _fixture(capture_root)
    observations[0]["relative_path"] = "../escape.png"
    with pytest.raises(
        EquirectangularVirtualRigError,
        match="equirectangular_source_relative_path_unsafe",
    ):
        _compile(capture_root, tmp_path / "output-a", metadata, observations)

    capture_root = tmp_path / "capture-b"
    metadata, observations = _fixture(capture_root)
    source = capture_root / observations[0]["relative_path"]
    external = tmp_path / "external.png"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)
    with pytest.raises(
        EquirectangularVirtualRigError,
        match="equirectangular_source_symlink_forbidden",
    ):
        _compile(capture_root, tmp_path / "output-b", metadata, observations)

    capture_root = tmp_path / "capture-c"
    metadata, observations = _fixture(capture_root)
    with pytest.raises(
        EquirectangularVirtualRigError,
        match="equirectangular_timestamp_invalid",
    ):
        _compile(
            capture_root,
            tmp_path / "output-c",
            metadata,
            observations,
            timestamp="2026-07-30T12:00:00",
        )
