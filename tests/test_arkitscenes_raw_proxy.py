from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import arkitscenes_raw_proxy as proxy
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_geometry_compiler import (
    ReconstructionGeometryCompilerError,
    compile_metric_geometry,
)


IMPLEMENTATION_DIGEST = "sha256:" + "a" * 64
RUNTIME_DIGEST = "sha256:" + "b" * 64
SOURCE_COMMIT = "c" * 40
RECORDED_REAL_PROXY = (
    Path(__file__).parents[1]
    / "docs/evidence/arkitscenes_raw_proxy_40958756_b2d7297f.json"
)
AUTHORITY = {
    "arkitscenes_license_accepted": True,
    "license_acceptance_authority": "explicit_test_authority",
    "local_processing_authorized": True,
    "provider_upload_authorized": False,
    "paid_compute_authorized": False,
}


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _schema(name: str) -> dict:
    return json.loads(
        (Path(__file__).parents[1] / "docs" / "schemas" / name).read_text(
            encoding="utf-8"
        )
    )


def _scene(root: Path, video_id: str = "12345678") -> Path:
    source = root / "source"
    extracted = root / "extracted"
    source.mkdir(parents=True)
    for name in (
        f"{video_id}.mov",
        "lowres_wide.zip",
        "lowres_depth.zip",
        "confidence.zip",
        "lowres_wide_intrinsics.zip",
    ):
        (source / name).write_bytes(f"fixture:{name}".encode())
    trajectory_rows: list[str] = []
    for index in range(10):
        timestamp = 1000.0 + index
        stem = f"{video_id}_{timestamp:.3f}"
        for directory in (
            "lowres_wide",
            "lowres_depth",
            "confidence",
            "lowres_wide_intrinsics",
        ):
            (extracted / directory).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 192), color=(index, index, index)).save(
            extracted / "lowres_wide" / f"{stem}.png"
        )
        depth = np.full((192, 256), 1000 + index, dtype=np.uint16)
        confidence = np.ones((192, 256), dtype=np.uint8)
        confidence[:96, :] = 2
        Image.fromarray(depth).save(extracted / "lowres_depth" / f"{stem}.png")
        Image.fromarray(confidence).save(
            extracted / "confidence" / f"{stem}.png"
        )
        (extracted / "lowres_wide_intrinsics" / f"{stem}.pincam").write_text(
            "256 192 200 201 128 96\n", encoding="utf-8"
        )
        if index > 0:
            trajectory_rows.append(f"{timestamp:.6f} 0 0 0 0 0 0")
    (source / "lowres_wide.traj").write_text(
        "\n".join(trajectory_rows) + "\n", encoding="utf-8"
    )
    return root


def _metadata_records() -> list[dict]:
    return [
        {
            "OriginalTimestampWhenWrittenToFile": {
                "value": 1_000_000 + index * 1_000,
                "timescale": 1_000,
            },
            "CameraIntrinsicMatrix": [
                1000.0,
                0.0,
                0.0,
                0.0,
                1001.0,
                0.0,
                960.0,
                720.0,
                1.0,
            ],
            "MetadataDictionary": {"ExposureTime": f"fixture-{index}"},
        }
        for index in range(10)
    ]


def _install_media_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        proxy,
        "_tool_identity",
        lambda: ("ffprobe-fixture", "ffmpeg-fixture", "fixture-runtime", RUNTIME_DIGEST),
    )
    monkeypatch.setattr(
        proxy,
        "_probe_video",
        lambda _video, _ffprobe: {
            "stream": {
                "width": 1920,
                "height": 1440,
                "codec_name": "hevc",
                "display_rotation_degrees": -90.0,
            },
            "frame_count": 9,
            "presentation_times_seconds": [float(index) for index in range(9)],
            "first_source_pts_seconds": 1.0,
            "frames": [
                {
                    "decoded_frame_index": index,
                    "source_pts_seconds": float(index + 1),
                    "t_video_sec": float(index),
                }
                for index in range(9)
            ],
        },
    )
    monkeypatch.setattr(
        proxy,
        "_extract_timed_metadata",
        lambda _ffmpeg, _video, _stream_index: _metadata_records(),
    )
    monkeypatch.setattr(
        proxy,
        "_probe_packet_pts",
        lambda _ffprobe, _video, _stream_index: [float(index) for index in range(10)],
    )

    def decode_stub(
        *,
        ffmpeg: str,
        video: Path,
        selected: list[tuple[int, str]],
        destination: Path,
    ) -> dict[int, Path]:
        del ffmpeg, video
        destination.mkdir(parents=True, exist_ok=True)
        outputs: dict[int, Path] = {}
        for decoded_index, frame_id in selected:
            path = destination / f"{frame_id}.png"
            if not path.exists():
                Image.new(
                    "RGB",
                    (1920, 1440),
                    color=(decoded_index, decoded_index, decoded_index),
                ).save(path)
            outputs[decoded_index] = path
        return outputs

    monkeypatch.setattr(proxy, "_extract_selected_frames", decode_stub)


def _compile(scene: Path, output: Path, *, timestamp: str) -> dict:
    return proxy.compile_arkitscenes_raw_proxy(
        scene_root=scene,
        output_root=output,
        video_id="12345678",
        split="Training",
        maximum_selected_frames=8,
        source_commit_sha=SOURCE_COMMIT,
        implementation_digest=IMPLEMENTATION_DIGEST,
        authority_used=AUTHORITY,
        timestamp=timestamp,
    )


def test_proxy_compiler_binds_source_pts_and_isolates_heldout_scaffold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _scene(tmp_path / "scene")
    output = tmp_path / "output"
    _install_media_stubs(monkeypatch)

    report = _compile(scene, output, timestamp="2026-07-30T12:00:00-05:00")
    artifact_root = next(output.glob("arkitscenes_proxy_*"))
    observations = json.loads(
        (artifact_root / "camera_observations_proxy.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (artifact_root / "candidate_metric_scaffold_proxy.json").read_text(
            encoding="utf-8"
        )
    )
    evaluator = json.loads(
        (artifact_root / "evaluator_hidden" / "metric_scaffold_proxy.json").read_text(
            encoding="utf-8"
        )
    )
    dataset = json.loads(
        next(artifact_root.glob("frozen_dataset_*/reconstruction_dataset_manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    selection = json.loads(
        next(artifact_root.glob("frozen_dataset_*/retained_frame_selection_manifest.json")).read_text(
            encoding="utf-8"
        )
    )

    candidate_ids = {row["frame_id"] for row in candidate["camera_frames"]}
    evaluator_ids = {row["frame_id"] for row in evaluator["camera_frames"]}
    assert candidate_ids
    assert evaluator_ids
    assert candidate_ids.isdisjoint(evaluator_ids)
    assert {row["observation_id"] for row in observations["observations"]} == candidate_ids
    assert all(
        row["source_pts_seconds"] == row["t_video_sec"] + 1.0
        for row in candidate["camera_frames"] + evaluator["camera_frames"]
    )
    assert report["metadata_samples_without_decoded_frames"] == [0.0]
    assert report["hidden_heldout_pixels_exposed_to_candidate"] is False
    assert report["raw_contract_3_2_proven"] is False
    assert report["iphone_route_proven"] is False
    assert dataset["deterministic_configuration"]["selection_rule"] == (
        "arkitscenes_exact_trajectory_timestamp_even_coverage_v1"
    )
    assert all(
        "evaluator_hidden" not in row["image_relative_path"]
        for row in observations["observations"]
    )

    proxy_validator = jsonschema.Draft202012Validator(
        _schema("arkitscenes_raw_proxy.v1.schema.json"),
        format_checker=jsonschema.FormatChecker(),
    )
    for artifact in (report, observations, candidate, evaluator):
        proxy_validator.validate(artifact)
    jsonschema.Draft202012Validator(
        _schema("reconstruction_frame_dataset.v1.schema.json")
    ).validate(selection)


def test_proxy_depth_surface_uses_official_opencv_convention_without_claim_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _scene(tmp_path / "scene")
    output = tmp_path / "output"
    _install_media_stubs(monkeypatch)
    _compile(scene, output, timestamp="2026-07-30T17:00:00Z")
    artifact_root = next(output.glob("arkitscenes_proxy_*"))

    result = proxy.compile_arkitscenes_depth_surface_proxy(
        scene_root=scene,
        proxy_artifact_root=artifact_root,
        output_root=scene / "derived" / "depth-surface",
    )

    assert result["capture_profile"] == "public_dataset_arkitscenes_proxy"
    assert result["emitted_vertex_count"] > 0
    assert result["emitted_triangle_count"] > 0
    assert result["hidden_heldout_observations_accessed"] is False
    assert result["blueprint_raw_contract_3_2_proven"] is False
    assert result["iphone_route_proven"] is False
    assert result["claim_ceiling"] == "public_dataset_arkit_depth_surface_proxy"
    request = json.loads(
        (scene / "derived/depth-surface/arkit_depth_surface_proxy_request.json").read_text()
    )
    assert request["camera_ray_convention"] == "opencv_x_right_y_down_z_forward"
    assert request["camera_calibration_binding"]["official_helper_commit"] == (
        proxy.ARKITSCENES_OFFICIAL_HELPER_COMMIT
    )
    assert request["candidate_may_read_hidden_heldout"] is False
    assert request["coordinate_frame_declaration"]["up_axis"] == (
        "not_independently_validated"
    )
    metric_request = {
        "schema_version": "metric_geometry_compilation_request.v1",
        "source_artifact_digest": None,
        "source_asset": result["surface_asset"],
        "original_file_references": [
            {
                "artifact_id": "arkitscenes-observed-surface-proxy",
                "digest": result["surface_asset"]["digest"],
            }
        ],
        "coordinate_frame_declaration": request["coordinate_frame_declaration"],
        "metric_scale_status": "sensor_metric_unvalidated",
        "minimum_confidence": 1.0,
        "declared_region_ids": ["arkitscenes-observed-frusta"],
        "unsupported_region_ids": ["arkitscenes-unobserved-regions"],
        "generated_fill_used": False,
        "appearance_asset_used_as_geometry_truth": False,
        "warnings": [],
        "blockers": ["coordinate_frame_qualification_required"],
    }
    metric_request["source_artifact_digest"] = canonical_digest(
        metric_request, digest_field="source_artifact_digest"
    )
    with pytest.raises(
        ReconstructionGeometryCompilerError,
        match="observed_surface_metric_z_up_frame_required",
    ):
        compile_metric_geometry(
            source_artifact=metric_request,
            artifact_root=scene,
            output_root=scene / "derived/metric-geometry",
        )


def test_proxy_compiler_is_idempotent_and_preserves_first_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _scene(tmp_path / "scene")
    output = tmp_path / "output"
    _install_media_stubs(monkeypatch)

    first = _compile(scene, output, timestamp="2026-07-30T17:00:00Z")
    second = _compile(scene, output, timestamp="2099-01-01T00:00:00Z")

    assert first == second
    assert first["timestamp"] == "2026-07-30T17:00:00Z"


def test_proxy_compiler_fails_closed_without_explicit_local_authority(
    tmp_path: Path,
) -> None:
    invalid_authority = dict(AUTHORITY)
    invalid_authority["arkitscenes_license_accepted"] = False

    with pytest.raises(proxy.ArkitScenesProxyError, match="arkitscenes_authority_invalid"):
        proxy.compile_arkitscenes_raw_proxy(
            scene_root=tmp_path,
            output_root=tmp_path / "output",
            video_id="12345678",
            split="Training",
            maximum_selected_frames=8,
            source_commit_sha=SOURCE_COMMIT,
            implementation_digest=IMPLEMENTATION_DIGEST,
            authority_used=invalid_authority,
            timestamp="2026-07-30T17:00:00Z",
        )


def test_proxy_compiler_rejects_naive_timestamp(
    tmp_path: Path,
) -> None:
    with pytest.raises(proxy.ArkitScenesProxyError, match="arkitscenes_timestamp_invalid"):
        proxy.compile_arkitscenes_raw_proxy(
            scene_root=tmp_path,
            output_root=tmp_path / "output",
            video_id="12345678",
            split="Training",
            maximum_selected_frames=8,
            source_commit_sha=SOURCE_COMMIT,
            implementation_digest=IMPLEMENTATION_DIGEST,
            authority_used=AUTHORITY,
            timestamp="2026-07-30T17:00:00",
        )


def test_proxy_source_digest_changes_with_authoritative_bytes(tmp_path: Path) -> None:
    scene = _scene(tmp_path / "scene")
    before = _digest(scene / "source" / "12345678.mov")

    (scene / "source" / "12345678.mov").write_bytes(b"changed-authoritative-bytes")

    assert _digest(scene / "source" / "12345678.mov") != before


def test_recorded_real_scene_proxy_preserves_reduced_authority_contract() -> None:
    receipt = json.loads(RECORDED_REAL_PROXY.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(
        _schema("arkitscenes_raw_proxy.v1.schema.json"),
        format_checker=jsonschema.FormatChecker(),
    ).validate(receipt)
    assert receipt["arkitscenes_proxy_compilation_digest"] == canonical_digest(
        receipt, digest_field="arkitscenes_proxy_compilation_digest"
    )
    assert receipt["source_commit_sha"] == "b2d7297fc3b28d2bb0a7b02ff3901137d70f51d3"
    assert receipt["source_capture_digest"] == (
        "sha256:bc493651dcc0950146e49bab91c9303a4d5f49c319c3e0b1048de1344d568e04"
    )
    assert len(receipt["original_file_references"]) == 6
    assert receipt["selected_frame_count"] == 40
    assert receipt["hidden_heldout_pixels_exposed_to_candidate"] is False
    assert receipt["raw_contract_3_2_proven"] is False
    assert receipt["iphone_route_proven"] is False
    assert receipt["metric_geometry_proven"] is False
    assert receipt["collision_or_physics_proven"] is False
    assert receipt["isaac_compatibility_proven"] is False
