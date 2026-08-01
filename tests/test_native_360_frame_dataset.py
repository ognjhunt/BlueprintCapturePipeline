from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Sequence

import jsonschema
import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.native_360_frame_dataset import (
    Native360FrameDatasetError,
    build_native_360_dataset_compiler_service,
    compile_native_360_grouped_frame_dataset,
    decode_native_360_lens_observations,
)
from blueprint_pipeline.native_360_normalization import (
    build_native_360_probe_receipt,
    normalize_native_360_capture,
)


CAPTURE_DIGEST = "sha256:" + "a" * 64
IMPLEMENTATION_DIGEST = "sha256:" + "b" * 64
RUNTIME_DIGEST = "sha256:" + "c" * 64
SOURCE_COMMIT = "d" * 40
CALIBRATION_DIGEST = "sha256:" + "e" * 64
MASK_BYTES = b"native-frame-valid-mask-fixture"
MASK_DIGEST = "sha256:cbfd172a7144a6b4c4afa7972d69780a44264f7273d7651694ed30155a12f4af"
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
            Path(__file__).parents[1] / "docs/schemas/reconstruction_frame_dataset.v1.schema.json"
        ).read_text(encoding="utf-8")
    )


def _decode_schema() -> dict:
    return json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/native_360_lens_decode.v1.schema.json"
        ).read_text(encoding="utf-8")
    )


def _lens_calibration(lens_id: str) -> dict:
    return {
        "lens_id": lens_id,
        "intrinsics": {
            "fx": 32.0,
            "fy": 32.0,
            "cx": 16.0,
            "cy": 16.0,
            "width": 32,
            "height": 32,
        },
        "distortion": {
            "model": "opencv_fisheye",
            "coefficients": [0.01, -0.001, 0.0001, -0.00001],
        },
        "valid_pixel_mask_relative_path": f"calibration/{lens_id}-mask.png",
        "valid_pixel_mask_digest": MASK_DIGEST,
        "calibration_source": "official_sdk_sidecar",
        "calibration_source_digest": CALIBRATION_DIGEST,
    }


def _normalized_artifacts(
    tmp_path: Path, pair_count: int = 5, segment_count: int = 1
) -> tuple[dict, dict, dict]:
    capture_root = tmp_path / "capture"
    for lens_id in ("front", "rear"):
        mask = capture_root / f"calibration/{lens_id}-mask.png"
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask.write_bytes(MASK_BYTES)
    pts = [round(index * 0.033333, 6) for index in range(pair_count)]
    segments: list[dict] = []
    receipts: dict[str, dict] = {}
    for sequence_index in range(segment_count):
        relative_path = (
            "native/capture.insv"
            if sequence_index == 0
            else f"native/capture_{sequence_index:03d}.insv"
        )
        source = capture_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"immutable-native-dual-fisheye-fixture-{sequence_index}".encode())
        source_digest = _digest(source)
        segment = {
            "sequence_index": sequence_index,
            "segment_id": f"segment-{sequence_index:04d}",
            "files": [
                {
                    "relative_path": relative_path,
                    "original_filename": Path(relative_path).name,
                    "size_bytes": source.stat().st_size,
                    "digest": source_digest,
                    "lens_streams": [
                        {"lens_id": "front", "stream_index": 0},
                        {"lens_id": "rear", "stream_index": 1},
                    ],
                }
            ],
        }
        if segment_count > 1:
            segment["capture_timeline_start_seconds"] = float(sequence_index)
        segments.append(segment)
        receipts[relative_path] = build_native_360_probe_receipt(
            source_file_digest=source_digest,
            runtime_identity="ffprobe-grouped-fixture",
            runtime_digest=RUNTIME_DIGEST,
            streams=[
                {
                    "stream_index": index,
                    "media_type": "video",
                    "codec_name": "hevc",
                    "width": 32,
                    "height": 32,
                    "time_base": "1/90000",
                    "pts_seconds": pts,
                    "metadata": {
                        "decoded_frame_timing": [
                            {
                                "pts_seconds": value,
                                "dts_seconds": value,
                                "duration_seconds": 0.033333,
                                "key_frame": frame_index == 0,
                            }
                            for frame_index, value in enumerate(pts)
                        ]
                    },
                }
                for index in range(2)
            ],
            format_metadata={"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        )
    metadata = {
        "schema_version": "native_360_camera_metadata.v1",
        "source_capture_digest": CAPTURE_DIGEST,
        "camera_model": "Insta360 grouped fixture",
        "capture_mode": "dual_fisheye_video",
        "firmware_version": "fixture-1",
        "coordinate_frame_declaration": {
            "units": "meters",
            "handedness": "right_handed",
            "camera_axes": "+x right, +y down, +z forward",
            "rig_frame": "front_lens_optical_center",
        },
        "segments": segments,
        "lens_calibrations": [
            _lens_calibration("front"),
            _lens_calibration("rear"),
        ],
        "rig_extrinsics": {
            "T_front_rear": [
                [1.0, 0.0, 0.0, 0.06],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "transform_semantics": "rear_camera_from_front_rig",
            "translation_units": "meters",
            "calibration_source": "official_sdk_sidecar",
            "calibration_source_digest": CALIBRATION_DIGEST,
        },
        "imu": {"status": "unavailable"},
        "gyro": {"status": "unavailable"},
    }
    result = normalize_native_360_capture(
        capture_root=capture_root,
        output_root=tmp_path / "normalization",
        intake_id="native-grouped-fixture",
        capture_digest=CAPTURE_DIGEST,
        camera_metadata=metadata,
        probe_receipts_by_path=receipts,
        source_commit_sha=SOURCE_COMMIT,
        implementation_digest=IMPLEMENTATION_DIGEST,
        authority_used=AUTHORITY,
        timestamp="2026-07-30T12:00:00Z",
        maximum_source_bytes=1024,
    )
    artifact_root = next((tmp_path / "normalization").glob("native_360_normalization_*"))
    rig = json.loads((artifact_root / "camera_360_rig_declaration.json").read_text())
    binding = json.loads((artifact_root / "dual_fisheye_stream_binding.json").read_text())
    return result, rig, binding


def _decoded_frames(root: Path, binding: dict) -> list[dict]:
    rows: list[dict] = []
    for segment in binding["segments"]:
        sequence_index = segment["sequence_index"]
        streams = {row["lens_id"]: row for row in segment["lens_streams"]}
        for pair in segment["frame_pairs"]:
            for lens_id in ("front", "rear"):
                path = (
                    root
                    / "decoded"
                    / f"segment-{sequence_index:04d}"
                    / lens_id
                    / f"{pair['pair_index']:09d}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "L",
                    (32, 32),
                    color=32 if lens_id == "front" else 192,
                ).save(path)
                stream = streams[lens_id]
                pts = pair[f"{lens_id}_pts_seconds"]
                rows.append(
                    {
                        "segment_sequence_index": sequence_index,
                        "pair_index": pair["pair_index"],
                        "lens_id": lens_id,
                        "source_relative_path": stream["source_relative_path"],
                        "source_digest": stream["source_digest"],
                        "stream_index": stream["stream_index"],
                        "decoded_frame_index": pair["pair_index"],
                        "source_pts_seconds": pts,
                        "source_dts_seconds": pts,
                        "duration_seconds": 0.033333,
                        "key_frame": pair["pair_index"] == 0,
                        "artifact_relative_path": path.relative_to(root).as_posix(),
                        "digest": _digest(path),
                        "image_metadata": {"width": 32, "height": 32},
                        "quality_signals": {"gradient_energy": 1.0},
                    }
                )
    return rows


def _recorded_decode_manifest(result: dict, binding: dict, decoded: list[dict]) -> dict:
    sources = [segment["files"][0] for segment in binding["segments"]]
    manifest = {
        "schema_version": "native_360_lens_decode_manifest.v1",
        "stable_run_identity": "native-360-recorded-decode-fixture",
        "source_capture_identity": result["source_capture_identity"],
        "source_capture_digest": CAPTURE_DIGEST,
        "original_file_references": [
            {"relative_path": source["relative_path"], "digest": source["digest"]}
            for source in sources
        ],
        "producing_method": "native_360_ffmpeg_lens_decoder.v1",
        "implementation_version": IMPLEMENTATION_DIGEST,
        "container_image_digest": None,
        "source_commit_sha": SOURCE_COMMIT,
        "deterministic_configuration": {"fixture": True},
        "deterministic_configuration_digest": "sha256:" + "1" * 64,
        "input_digests": {},
        "output_digests": {"decoded_frame_digests": [row["digest"] for row in decoded]},
        "runtime_identity": "ffmpeg-grouped-fixture",
        "runtime_digest": RUNTIME_DIGEST,
        "frames": decoded,
        "decoded_frame_count": len(decoded),
        "complete_retained_native_source_preserved": True,
        "original_distorted_pixels_preserved": True,
        "lens_identity_inferred": False,
        "calibration_inferred": False,
        "candidate_method_access_allowed": False,
        "access_scope": "trusted_dataset_compiler_only",
        "authority_used": AUTHORITY,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "warnings": ["recorded_fixture_output"],
        "blockers": [],
        "proof_effect": "decoded_native_lens_observation_availability_only",
        "claim_ceiling": "decoded_observation_availability",
        "parent_artifact_or_event": {
            "native_360_normalization_digest": result["native_360_normalization_digest"],
            "dual_fisheye_binding_digest": binding["dual_fisheye_binding_digest"],
        },
        "timestamp": "2026-07-30T12:00:00Z",
    }
    manifest["lens_decode_manifest_digest"] = canonical_digest(
        manifest, digest_field="lens_decode_manifest_digest"
    )
    return manifest


def _compile(
    root: Path,
    result: dict,
    rig: dict,
    binding: dict,
    decoded: list[dict],
    decode_manifest: dict | None = None,
) -> dict:
    manifest = decode_manifest or _recorded_decode_manifest(result, binding, decoded)
    manifest_path = (
        root
        / (f"native_360_lens_decode_{manifest['deterministic_configuration_digest'][7:23]}")
        / "native_360_lens_decode_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(manifest) + "\n"
    if manifest_path.exists():
        assert manifest_path.read_text(encoding="utf-8") == payload
    else:
        manifest_path.write_text(payload, encoding="utf-8")
    return compile_native_360_grouped_frame_dataset(
        artifact_root=root,
        intake_id="native-grouped-fixture",
        capture_digest=CAPTURE_DIGEST,
        normalization_result=result,
        rig_declaration=rig,
        dual_fisheye_binding=binding,
        lens_decode_manifest=manifest,
        decoded_lens_frames=decoded,
        runtime_identity=manifest["runtime_identity"],
        runtime_digest=manifest["runtime_digest"],
        implementation_digest=IMPLEMENTATION_DIGEST,
        source_commit_sha=SOURCE_COMMIT,
        authority_used=AUTHORITY,
        timestamp="2026-07-30T12:00:00Z",
    )


def _load_reference(root: Path, dataset: dict, name: str) -> dict:
    relative = dataset["artifact_references"][name]["relative_path"]
    return json.loads((root / relative).read_text())


def test_native_lens_decoder_executes_declared_streams_and_replays(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path)
    executable = tmp_path / "bin/ffmpeg"
    executable.parent.mkdir()
    executable.write_bytes(b"pinned-ffmpeg-fixture")
    calls: list[list[str]] = []

    def runner(
        argv: Sequence[str], _timeout: float, _maximum_output: int
    ) -> subprocess.CompletedProcess[bytes]:
        command = list(argv)
        calls.append(command)
        if "-version" in command:
            return subprocess.CompletedProcess(command, 0, b"ffmpeg version fixture-7.1.1\n", b"")
        stream_index = int(command[command.index("-map") + 1].split(":")[1])
        Image.new("L", (32, 32), color=48 if stream_index == 0 else 192).save(command[-1])
        return subprocess.CompletedProcess(command, 0, b"", b"")

    decode_root = tmp_path / "decoded-output"
    manifest = decode_native_360_lens_observations(
        capture_root=tmp_path / "capture",
        artifact_root=decode_root,
        capture_digest=CAPTURE_DIGEST,
        normalization_result=result,
        rig_declaration=rig,
        dual_fisheye_binding=binding,
        implementation_digest=IMPLEMENTATION_DIGEST,
        source_commit_sha=SOURCE_COMMIT,
        authority_used=AUTHORITY,
        timestamp="2026-07-30T12:00:00Z",
        ffmpeg_executable=executable,
        runner=runner,
    )
    replay = decode_native_360_lens_observations(
        capture_root=tmp_path / "capture",
        artifact_root=decode_root,
        capture_digest=CAPTURE_DIGEST,
        normalization_result=result,
        rig_declaration=rig,
        dual_fisheye_binding=binding,
        implementation_digest=IMPLEMENTATION_DIGEST,
        source_commit_sha=SOURCE_COMMIT,
        authority_used=AUTHORITY,
        timestamp="2099-01-01T00:00:00Z",
        ffmpeg_executable=executable,
        runner=runner,
    )

    assert manifest == replay
    assert manifest["timestamp"] == "2026-07-30T12:00:00Z"
    assert manifest["decoded_frame_count"] == 10
    assert {row["lens_id"] for row in manifest["frames"]} == {"front", "rear"}
    assert manifest["lens_identity_inferred"] is False
    assert all(row["source_dts_seconds"] is not None for row in manifest["frames"])
    assert all(row["duration_seconds"] == 0.033333 for row in manifest["frames"])
    assert sum(row["key_frame"] is True for row in manifest["frames"]) == 2
    assert manifest["warnings"] == ["decoded_exposure_metadata_not_established"]
    assert manifest["candidate_method_access_allowed"] is False
    assert len(calls) == 12  # 11 initial commands plus one replay identity check.
    jsonschema.Draft202012Validator(
        _decode_schema(), format_checker=jsonschema.FormatChecker()
    ).validate(manifest)

    dataset = _compile(
        decode_root,
        result,
        rig,
        binding,
        manifest["frames"],
        decode_manifest=manifest,
    )
    assert dataset["train_heldout_split_digest"]
    assert dataset["candidate_dataset_contains_hidden_heldout_pixels"] is False


def test_native_multisegment_decode_and_grouped_splits_preserve_declared_timeline(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path, segment_count=2)
    executable = tmp_path / "ffmpeg-multisegment"
    executable.write_bytes(b"pinned-ffmpeg-multisegment-fixture")
    decoded_sources: set[str] = set()

    def runner(
        argv: Sequence[str], _timeout: float, _maximum_output: int
    ) -> subprocess.CompletedProcess[bytes]:
        command = list(argv)
        if "-version" in command:
            return subprocess.CompletedProcess(
                command, 0, b"ffmpeg version multisegment-fixture\n", b""
            )
        decoded_sources.add(command[command.index("-i") + 1])
        stream_index = int(command[command.index("-map") + 1].split(":")[1])
        Image.new("L", (32, 32), color=40 + stream_index * 120).save(command[-1])
        return subprocess.CompletedProcess(command, 0, b"", b"")

    artifact_root = tmp_path / "multisegment-output"
    manifest = decode_native_360_lens_observations(
        capture_root=tmp_path / "capture",
        artifact_root=artifact_root,
        capture_digest=CAPTURE_DIGEST,
        normalization_result=result,
        rig_declaration=rig,
        dual_fisheye_binding=binding,
        implementation_digest=IMPLEMENTATION_DIGEST,
        source_commit_sha=SOURCE_COMMIT,
        authority_used=AUTHORITY,
        timestamp="2026-07-30T12:00:00Z",
        ffmpeg_executable=executable,
        runner=runner,
    )
    dataset = _compile(
        artifact_root,
        result,
        rig,
        binding,
        manifest["frames"],
        decode_manifest=manifest,
    )
    selection = _load_reference(artifact_root, dataset, "retained_frame_selection_manifest")
    split = _load_reference(artifact_root, dataset, "frozen_split_manifest")

    assert manifest["decoded_frame_count"] == 20
    assert len(manifest["original_file_references"]) == 2
    assert len(decoded_sources) == 2
    assert len(dataset["original_file_references"]) == 2
    assert selection["source_video_references"] == dataset["original_file_references"]
    assert sorted({row["t_video_sec"] for row in selection["frames"]}) == [
        0.0,
        0.033333,
        0.066666,
        0.099999,
        0.133332,
        1.0,
        1.033333,
        1.066666,
        1.099999,
        1.133332,
    ]
    assert sorted(
        {
            row["decoded_frame_index"]
            for row in selection["frames"]
            if row["source_camera_identity"] == "front"
        }
    ) == list(range(10))
    splits_by_group: dict[str, set[str]] = {}
    for row in split["assignments"]:
        splits_by_group.setdefault(row["observation_group_id"], set()).add(row["split"])
    assert all(len(values) == 1 for values in splits_by_group.values())
    jsonschema.Draft202012Validator(
        _decode_schema(), format_checker=jsonschema.FormatChecker()
    ).validate(manifest)
    jsonschema.Draft202012Validator(_schema()).validate(dataset)


def test_native_multisegment_compiler_rejects_decode_source_suppression(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path, segment_count=2)
    artifact_root = tmp_path / "source-suppression"
    decoded = _decoded_frames(artifact_root, binding)
    manifest = _recorded_decode_manifest(result, binding, decoded)
    manifest["original_file_references"] = manifest["original_file_references"][:1]
    manifest["lens_decode_manifest_digest"] = canonical_digest(
        manifest, digest_field="lens_decode_manifest_digest"
    )

    with pytest.raises(
        Native360FrameDatasetError,
        match="native_360_grouped_dataset_decode_source_set_mismatch",
    ):
        _compile(
            artifact_root,
            result,
            rig,
            binding,
            decoded,
            decode_manifest=manifest,
        )


def test_native_lens_decoder_maps_timeout_and_dimension_failures(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path)
    executable = tmp_path / "ffmpeg"
    executable.write_bytes(b"pinned-ffmpeg-fixture")

    def timeout_runner(
        argv: Sequence[str], timeout: float, _maximum_output: int
    ) -> subprocess.CompletedProcess[bytes]:
        if "-version" in argv:
            return subprocess.CompletedProcess(list(argv), 0, b"ffmpeg version fixture\n", b"")
        raise subprocess.TimeoutExpired(list(argv), timeout)

    with pytest.raises(Native360FrameDatasetError, match="lens_decode_timeout"):
        decode_native_360_lens_observations(
            capture_root=tmp_path / "capture",
            artifact_root=tmp_path / "timeout",
            capture_digest=CAPTURE_DIGEST,
            normalization_result=result,
            rig_declaration=rig,
            dual_fisheye_binding=binding,
            implementation_digest=IMPLEMENTATION_DIGEST,
            source_commit_sha=SOURCE_COMMIT,
            authority_used=AUTHORITY,
            timestamp="2026-07-30T12:00:00Z",
            ffmpeg_executable=executable,
            runner=timeout_runner,
        )

    def wrong_size_runner(
        argv: Sequence[str], _timeout: float, _maximum_output: int
    ) -> subprocess.CompletedProcess[bytes]:
        if "-version" in argv:
            return subprocess.CompletedProcess(list(argv), 0, b"ffmpeg version fixture\n", b"")
        Image.new("L", (16, 16), color=0).save(argv[-1])
        return subprocess.CompletedProcess(list(argv), 0, b"", b"")

    with pytest.raises(Native360FrameDatasetError, match="dimensions_invalid"):
        decode_native_360_lens_observations(
            capture_root=tmp_path / "capture",
            artifact_root=tmp_path / "wrong-size",
            capture_digest=CAPTURE_DIGEST,
            normalization_result=result,
            rig_declaration=rig,
            dual_fisheye_binding=binding,
            implementation_digest=IMPLEMENTATION_DIGEST,
            source_commit_sha=SOURCE_COMMIT,
            authority_used=AUTHORITY,
            timestamp="2026-07-30T12:00:00Z",
            ffmpeg_executable=executable,
            runner=wrong_size_runner,
        )


def test_native_dataset_service_composes_decode_and_split_with_route_binding(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path)
    executable = tmp_path / "ffmpeg-service"
    executable.write_bytes(b"pinned-ffmpeg-service-fixture")

    def runner(
        argv: Sequence[str], _timeout: float, _maximum_output: int
    ) -> subprocess.CompletedProcess[bytes]:
        command = list(argv)
        if "-version" in command:
            return subprocess.CompletedProcess(command, 0, b"ffmpeg version service-fixture\n", b"")
        stream_index = int(command[command.index("-map") + 1].split(":")[1])
        Image.new("L", (32, 32), color=32 + (stream_index * 128)).save(command[-1])
        return subprocess.CompletedProcess(command, 0, b"", b"")

    capture_build_digest = "sha256:" + "6" * 64
    route_digest = "sha256:" + "7" * 64
    service = build_native_360_dataset_compiler_service(
        capture_root=tmp_path / "capture",
        intake_id="native-grouped-fixture",
        capture_digest=CAPTURE_DIGEST,
        capture_build_digest=capture_build_digest,
        capture_reconstruction_route_digest=route_digest,
        normalization_result=result,
        rig_declaration=rig,
        dual_fisheye_binding=binding,
        implementation_digest=IMPLEMENTATION_DIGEST,
        source_commit_sha=SOURCE_COMMIT,
        authority_used=AUTHORITY,
        timestamp="2026-07-30T12:00:00Z",
        ffmpeg_executable=executable,
        runner=runner,
    )
    dataset = service(
        request={
            "capture_build_digest": capture_build_digest,
            "capture_reconstruction_route_digest": route_digest,
            "capture_authority_profile": "camera_360_native",
            "requested_claim_types": ["navigation_clearance"],
        },
        output_root=tmp_path / "service-output",
    )

    assert dataset["parent_artifact_or_event"]["capture_build_digest"] == (capture_build_digest)
    assert (
        dataset["parent_artifact_or_event"]["capture_reconstruction_route_digest"] == route_digest
    )
    assert dataset["supporting_artifact_references"][0]["artifact_type"] == (
        "native_360_lens_decode_manifest.v1"
    )

    with pytest.raises(
        Native360FrameDatasetError,
        match="dataset_service_request_binding_mismatch",
    ):
        service(
            request={
                "capture_build_digest": "sha256:" + "0" * 64,
                "capture_reconstruction_route_digest": route_digest,
                "capture_authority_profile": "camera_360_native",
                "requested_claim_types": [],
            },
            output_root=tmp_path / "refused-output",
        )


def test_native_grouped_dataset_binds_pairs_and_isolates_both_hidden_lenses(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path)
    dataset_root = tmp_path / "dataset"
    decoded = _decoded_frames(dataset_root, binding)

    dataset = _compile(dataset_root, result, rig, binding, decoded)
    replay = _compile(dataset_root, result, rig, binding, decoded)

    assert dataset == replay
    assert dataset["capture_authority_profile"] == "camera_360_native"
    assert dataset["camera_calibration_binding"] == {
        "camera_360_rig_declaration_digest": rig["rig_declaration_digest"]
    }
    assert dataset["claim_ceiling"] == "decoded_observation_availability"
    assert dataset["metric_scale_status"] == "not_established"
    assert dataset["supporting_artifact_references"][0]["artifact_type"] == (
        "native_360_lens_decode_manifest.v1"
    )
    split = _load_reference(dataset_root, dataset, "frozen_split_manifest")
    candidate = _load_reference(dataset_root, dataset, "candidate_dataset_manifest")
    heldout = _load_reference(dataset_root, dataset, "hidden_heldout_evaluator_manifest")
    group_splits: dict[str, set[str]] = {}
    group_cameras: dict[str, set[str]] = {}
    for row in split["assignments"]:
        group_splits.setdefault(row["observation_group_id"], set()).add(row["split"])
        group_cameras.setdefault(row["observation_group_id"], set()).add(
            row["source_camera_identity"]
        )
    assert all(len(values) == 1 for values in group_splits.values())
    assert all(values == {"front", "rear"} for values in group_cameras.values())
    candidate_groups = {row["observation_group_id"] for row in candidate["frames"]}
    hidden_groups = {row["observation_group_id"] for row in heldout["frames"]}
    assert candidate_groups.isdisjoint(hidden_groups)
    assert hidden_groups
    assert heldout["candidate_method_access_allowed"] is False

    validator = jsonschema.Draft202012Validator(_schema())
    for artifact in (dataset, split, candidate, heldout):
        validator.validate(artifact)


def test_native_grouped_dataset_rejects_missing_or_rebound_lens_observations(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path)
    dataset_root = tmp_path / "dataset"
    decoded = _decoded_frames(dataset_root, binding)

    with pytest.raises(Native360FrameDatasetError, match="observations_incomplete"):
        _compile(dataset_root, result, rig, binding, decoded[:-1])

    rebound = [dict(row) for row in decoded]
    rebound[0]["stream_index"] = 99
    with pytest.raises(Native360FrameDatasetError, match="decoded_binding_invalid"):
        _compile(tmp_path / "rebound", result, rig, binding, rebound)

    timing_root = tmp_path / "timing-spoof"
    timing_spoof = _decoded_frames(timing_root, binding)
    timing_spoof[0]["source_dts_seconds"] = 9.0
    with pytest.raises(Native360FrameDatasetError, match="decoded_timing_mismatch"):
        _compile(timing_root, result, rig, binding, timing_spoof)


def test_native_grouped_dataset_rejects_tampered_parent_and_missing_authority(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path)
    dataset_root = tmp_path / "dataset"
    decoded = _decoded_frames(dataset_root, binding)

    tampered = dict(binding)
    tampered["all_segments_synchronized"] = False
    with pytest.raises(Native360FrameDatasetError, match="stream_binding_invalid"):
        _compile(dataset_root, result, rig, tampered, decoded)

    authority = dict(AUTHORITY)
    authority["provider_upload_authorized"] = True
    with pytest.raises(Native360FrameDatasetError, match="authority_invalid"):
        compile_native_360_grouped_frame_dataset(
            artifact_root=dataset_root,
            intake_id="native-grouped-fixture",
            capture_digest=CAPTURE_DIGEST,
            normalization_result=result,
            rig_declaration=rig,
            dual_fisheye_binding=binding,
            lens_decode_manifest=_recorded_decode_manifest(result, binding, decoded),
            decoded_lens_frames=decoded,
            runtime_identity="ffmpeg-grouped-fixture",
            runtime_digest=RUNTIME_DIGEST,
            implementation_digest=IMPLEMENTATION_DIGEST,
            source_commit_sha=SOURCE_COMMIT,
            authority_used=authority,
            timestamp="2026-07-30T12:00:00Z",
        )


def test_native_grouped_dataset_rejects_dimension_spoof_and_symlinked_pixels(
    tmp_path: Path,
) -> None:
    result, rig, binding = _normalized_artifacts(tmp_path)

    dimension_root = tmp_path / "dimension"
    dimensions = _decoded_frames(dimension_root, binding)
    dimensions[0] = dict(dimensions[0])
    dimensions[0]["image_metadata"] = {"width": 64, "height": 32}
    with pytest.raises(Native360FrameDatasetError, match="dimensions_invalid"):
        _compile(dimension_root, result, rig, binding, dimensions)

    symlink_root = tmp_path / "symlink"
    symlinked = _decoded_frames(symlink_root, binding)
    first_path = symlink_root / symlinked[0]["artifact_relative_path"]
    replacement = symlink_root / "replacement.png"
    Image.new("L", (32, 32), color=12).save(replacement)
    first_path.unlink()
    first_path.symlink_to(replacement)
    symlinked[0] = dict(symlinked[0])
    symlinked[0]["digest"] = _digest(replacement)
    with pytest.raises(Native360FrameDatasetError, match="frame_artifact_invalid"):
        _compile(symlink_root, result, rig, binding, symlinked)
