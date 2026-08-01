from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import jsonschema
from PIL import Image

from blueprint_pipeline.arkit_reconstruction_dataset import (
    compile_arkit_reconstruction_dataset,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.equirectangular_virtual_rig import (
    compile_equirectangular_virtual_rig,
)
from blueprint_pipeline.native_360_normalization import (
    build_native_360_probe_receipt,
    normalize_native_360_capture,
)
from blueprint_pipeline.native_360_frame_dataset import (
    compile_native_360_grouped_frame_dataset,
    decode_native_360_lens_observations,
)
from blueprint_pipeline.reconstruction_frame_dataset import (
    compile_frozen_frame_dataset,
)
from blueprint_pipeline.reconstruction_terminal_report import (
    RECONSTRUCTION_REPORT_REQUEST_SCHEMA_VERSION,
    generate_reconstruction_terminal_report,
)
from blueprint_pipeline.reconstruction_validation_contracts import (
    CAMERA_RIG_VALIDATION_REQUEST_SCHEMA_VERSION,
    validate_camera_rig,
)


FIXTURE_SPEC_PATH = Path(__file__).parent / "fixtures/reconstruction_vertical_v1/fixture_spec.json"


def _spec() -> dict:
    value = json.loads(FIXTURE_SPEC_PATH.read_text(encoding="utf-8"))
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_hermetic_vertical_fixture_spec.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(value, schema)
    return value


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_artifact_digest(value: dict) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _absence_digest(*, stage: str, reason: str) -> str:
    return canonical_digest(
        {
            "schema_version": "reconstruction_fixture_absence.v1",
            "stage": stage,
            "status": "not_executed",
            "reason": reason,
            "proof_effect": "none",
        }
    )


def _ceilings(
    *,
    decoded: bool,
    calibrated: bool,
    metric_scale: bool = False,
    metric_reference: bool = False,
) -> dict[str, bool]:
    return {
        "decoded_observation_availability": decoded,
        "calibrated_camera_trajectory": calibrated,
        "appearance_reconstruction": False,
        "metric_scale": metric_scale,
        "metric_reference_geometry": metric_reference,
        "collision_geometry": False,
        "physics_readiness": False,
        "isaac_load_render_compatibility": False,
        "simulator_task_evidence": False,
        "physical_task_success": False,
        "deployment_readiness": False,
    }


def _terminal_report(
    *,
    spec: dict,
    profile: str,
    source_digest: str,
    split_digest: str,
    input_digests: list[str],
    recorded_output_digests: list[str],
    blocker: str,
    ceilings: dict[str, bool],
    calibration_status: dict,
    selected_frames: list[dict] | None = None,
    rejected_frames: list[dict] | None = None,
) -> dict:
    request = {
        "schema_version": RECONSTRUCTION_REPORT_REQUEST_SCHEMA_VERSION,
        "stable_run_identity": f"hermetic-{profile}",
        "original_capture_location": f"hermetic-fixture://{profile}",
        "source_capture_digest": source_digest,
        "implementation_digest": spec["implementation_digest"],
        "input_digests": input_digests,
        "recorded_output_digests": recorded_output_digests,
        "validated_capture_profile": profile,
        "original_customer_request": "Exercise the local capture reconstruction vertical.",
        "rights_and_permitted_use": {
            "status": "cleared",
            "local_processing": True,
            "remote_upload": False,
        },
        "selected_frames": selected_frames or [],
        "rejected_frames": rejected_frames or [],
        "frozen_split_digest": split_digest,
        "calibration_and_coordinate_status": calibration_status,
        "camera_calibration_binding": calibration_status,
        "coordinate_frame_declaration": {
            "units": "meters" if profile == "iphone_arkit_lidar" else "unknown",
            "handedness": "right_handed",
        },
        "units_and_metric_scale_status": {
            "declared_units": "meters" if profile == "iphone_arkit_lidar" else "unknown",
            "independently_validated": ceilings["metric_scale"],
        },
        "pose_methods_attempted": [],
        "registered_observations": [
            row["frame_id"] for row in (selected_frames or []) if "frame_id" in row
        ],
        "rejected_observations": [
            row["frame_id"] for row in (rejected_frames or []) if "frame_id" in row
        ],
        "scale_validation": {
            "status": "not_independently_validated",
            "learned_depth_established_scale": False,
        },
        "reconstruction_methods_attempted": [],
        "failed_methods": [
            {
                "method_id": "headless_pose_and_gaussian_worker",
                "status": "not_executed",
                "reason": blocker,
                "failed_evidence_preserved": True,
            }
        ],
        "skipped_methods": [
            {"method_id": "heldout_appearance_evaluation", "reason": "appearance_missing"},
            {"method_id": "isaac_verification", "reason": "qualified_package_missing"},
        ],
        "recovered_methods": [],
        "appearance_asset": {"status": "missing"},
        "metric_reference_asset": {
            "status": "sensor_scaffold_only" if ceilings["metric_reference_geometry"] else "missing"
        },
        "collision_candidate": {"status": "missing"},
        "independent_visual_metrics": {"status": "not_executed"},
        "independent_geometric_metrics": {"status": "not_executed"},
        "collider_qualification": {"status": "not_executed"},
        "nurec_openusd_package": {"status": "not_executed"},
        "isaac_verification": {"status": "not_executed"},
        "fixed_camera_render_references": [],
        "physics_collision_verification": {"status": "not_executed"},
        "provider_execution": {"provider": None, "status": "not_used"},
        "provider_runtime_identity": {"provider": "local", "runtime": "hermetic_fixture"},
        "source_commit_sha": spec["source_commit_sha"],
        "container_image_digests": [],
        "runtime_and_spend": {"total_runtime_seconds": 0.0, "total_spend_usd": 0.0},
        "agent_proposals_and_actions": [
            {"action": "preserve_evidence_and_abstain", "proof_effect": "none"}
        ],
        "deterministic_validations": [
            {"validation": "source_digest_binding", "status": "passed"},
            {"validation": "local_profile_stage_replay", "status": "passed"},
        ],
        "decision": "abstention",
        "evidence_ceilings": ceilings,
        "what_could_change_result": [
            "build and smoke-test the pinned worker under explicit authority",
            "run independent held-out and Isaac qualification",
        ],
        "what_blueprint_cannot_claim": [
            "appearance reconstruction",
            "qualified collision geometry",
            "Isaac compatibility",
            "physical task success",
            "deployment readiness",
        ],
        "warnings": ["hermetic fixture execution is not representative real-capture proof"],
        "blockers": [blocker],
        "teardown_and_provider_zero": {
            "status": "not_applicable_no_provider_allocation",
            "live_provider_inventory": 0,
        },
        "authority_used": {
            "local_non_spend": True,
            "provider_upload": False,
            "paid_compute": False,
        },
        "timestamp": spec["timestamp"],
    }
    return generate_reconstruction_terminal_report(request)


def _compile_iphone_vertical(root: Path, spec: dict) -> tuple[dict, dict, dict, dict]:
    profile = spec["iphone_arkit_lidar"]
    retained = root / "retained"
    frames_root = root / "frames"
    retained.mkdir(parents=True)
    frames_root.mkdir(parents=True)
    video = retained / "walkthrough.mov"
    video.write_bytes(b"hermetic-retained-video-fixture")
    selected: list[dict] = []
    pose = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for index in range(profile["frame_count"]):
        path = frames_root / f"decoded-{index:09d}.png"
        Image.new(
            "RGB",
            (profile["width"], profile["height"]),
            color=(index * 20, index * 10, index * 5),
        ).save(path)
        selected.append(
            {
                "frame_id": f"decoded-{index:09d}",
                "decoded_frame_index": index,
                "t_video_sec": round(index * profile["frame_interval_seconds"], 9),
                "source_pts_seconds": round(
                    profile["source_pts_origin_seconds"]
                    + index * profile["frame_interval_seconds"],
                    9,
                ),
                "source_dts_seconds": None,
                "duration_seconds": profile["frame_interval_seconds"],
                "key_frame": index == 0,
                "artifact_relative_path": path.relative_to(root).as_posix(),
                "digest": _file_digest(path),
                "image_metadata": {
                    "width": profile["width"],
                    "height": profile["height"],
                    "pixel_orientation": "encoded_source_no_autorotate",
                },
                "quality_signals": {"gradient_energy": float(index + 1)},
            }
        )
    dataset = compile_frozen_frame_dataset(
        artifact_root=root,
        intake_id="hermetic-iphone",
        capture_digest=profile["source_capture_digest"],
        capture_authority_profile="iphone_arkit_lidar",
        source_video_relative_path=video.relative_to(root).as_posix(),
        source_video_digest=_file_digest(video),
        decoded_frame_count=profile["frame_count"],
        selected_frames=selected,
        stream_metadata={
            "width": profile["width"],
            "height": profile["height"],
            "pixel_format": "rgb24",
            "display_rotation_degrees": 0,
        },
        runtime_identity="hermetic-ffmpeg-fixture",
        runtime_digest=spec["runtime_digest"],
        implementation_digest=spec["implementation_digest"],
        source_commit_sha=spec["source_commit_sha"],
        rights_and_retention={"external_processing": False},
        timestamp=spec["timestamp"],
    )
    dataset_root = next(root.glob("frozen_dataset_*"))
    split = json.loads((dataset_root / "frozen_split_manifest.json").read_text())
    candidate = json.loads((dataset_root / "candidate_dataset_manifest.json").read_text())
    scaffold = {
        "schema_version": "arkit_metric_scaffold.v1",
        "capture_digest": profile["source_capture_digest"],
        "coordinate_frame_session_id": profile["coordinate_frame_session_id"],
        "coordinate_system": {
            "world_frame_definition": "arkit_world_origin_at_session_start",
            "units": "meters",
            "handedness": "right_handed",
            "gravity_aligned": True,
        },
        "intrinsics": profile["intrinsics"],
        "camera_frames": [
            {
                "frame_id": row["frame_id"],
                "encoded_frame_index": row["decoded_frame_index"],
                "t_video_sec": row["t_video_sec"],
                "t_capture_sec": row["t_video_sec"],
                "T_world_camera": pose,
            }
            for row in selected
        ],
        "depth_confidence_pairs": [],
        "source_artifact_digests": {"retained_video": _file_digest(video)},
    }
    scaffold_digest = _canonical_artifact_digest(scaffold)
    export = compile_arkit_reconstruction_dataset(
        output_root=root / "arkit_export",
        intake_id="hermetic-iphone",
        capture_digest=profile["source_capture_digest"],
        dataset_manifest=dataset,
        split_manifest=split,
        candidate_manifest=candidate,
        metric_scaffold=scaffold,
        metric_scaffold_digest=scaffold_digest,
        implementation_digest=spec["implementation_digest"],
        source_commit_sha=spec["source_commit_sha"],
        authority_used={"external_processing": False},
        timestamp=spec["timestamp"],
    )
    return dataset, candidate, export, scaffold


def test_iphone_hermetic_vertical_reaches_worker_gate_and_abstains(tmp_path: Path) -> None:
    spec = _spec()
    dataset, candidate, export, scaffold = _compile_iphone_vertical(tmp_path / "iphone", spec)
    profile = spec["iphone_arkit_lidar"]
    report = _terminal_report(
        spec=spec,
        profile="iphone_arkit_lidar",
        source_digest=profile["source_capture_digest"],
        split_digest=dataset["train_heldout_split_digest"],
        input_digests=[
            dataset["dataset_manifest_digest"],
            export["arkit_reconstruction_dataset_export_digest"],
        ],
        recorded_output_digests=[export["arkit_reconstruction_dataset_export_digest"]],
        blocker=profile["expected_terminal_blocker"],
        ceilings=_ceilings(decoded=True, calibrated=True),
        calibration_status={
            "status": "arkit_sensor_bound_not_refined",
            "metric_scaffold_digest": _canonical_artifact_digest(scaffold),
            "raw_arkit_poses_modified": False,
        },
        selected_frames=[{"frame_id": row["frame_id"]} for row in candidate["frames"]],
        rejected_frames=[],
    )

    assert export["hidden_heldout_pixels_included"] is False
    assert export["raw_arkit_poses_modified"] is False
    assert report["decision"] == "abstention"
    assert report["evidence_ceilings"]["calibrated_camera_trajectory"] is True
    assert report["evidence_ceilings"]["appearance_reconstruction"] is False
    assert report["blockers"] == [profile["expected_terminal_blocker"]]


def _lens_calibration(profile: dict, lens_id: str) -> dict:
    return {
        "lens_id": lens_id,
        "intrinsics": {
            "fx": 1900.0,
            "fy": 1901.0,
            "cx": profile["width"] / 2,
            "cy": profile["height"] / 2,
            "width": profile["width"],
            "height": profile["height"],
        },
        "distortion": {
            "model": "opencv_fisheye",
            "coefficients": [0.01, -0.001, 0.0001, -0.00001],
        },
        "valid_pixel_mask_digest": profile["valid_pixel_mask_digest"],
        "calibration_source": "official_sdk_sidecar",
        "calibration_source_digest": profile["calibration_source_digest"],
    }


def test_native_360_hermetic_vertical_validates_rig_then_abstains(tmp_path: Path) -> None:
    spec = _spec()
    profile = spec["camera_360_native"]
    capture_root = tmp_path / "native360"
    source = capture_root / "native/capture.insv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable-hermetic-dual-fisheye-container")
    source_digest = _file_digest(source)
    metadata = {
        "schema_version": "native_360_camera_metadata.v1",
        "source_capture_digest": profile["source_capture_digest"],
        "camera_model": profile["camera_model"],
        "capture_mode": "dual_fisheye_video",
        "firmware_version": "fixture-1",
        "coordinate_frame_declaration": {
            "units": "meters",
            "handedness": "right_handed",
            "camera_axes": "+x right, +y down, +z forward",
            "rig_frame": "front_lens_optical_center",
        },
        "segments": [
            {
                "sequence_index": 0,
                "segment_id": "segment-0000",
                "files": [
                    {
                        "relative_path": "native/capture.insv",
                        "original_filename": "capture.insv",
                        "size_bytes": source.stat().st_size,
                        "digest": source_digest,
                        "lens_streams": [
                            {"lens_id": "front", "stream_index": 0},
                            {"lens_id": "rear", "stream_index": 1},
                        ],
                    }
                ],
            }
        ],
        "lens_calibrations": [
            _lens_calibration(profile, "front"),
            _lens_calibration(profile, "rear"),
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
            "calibration_source_digest": profile["calibration_source_digest"],
        },
        "imu": {"status": "unavailable"},
        "gyro": {"status": "unavailable"},
    }
    receipt = build_native_360_probe_receipt(
        source_file_digest=source_digest,
        runtime_identity="ffprobe-hermetic-fixture",
        runtime_digest=spec["runtime_digest"],
        streams=[
            {
                "stream_index": stream_index,
                "media_type": "video",
                "codec_name": "hevc",
                "width": profile["width"],
                "height": profile["height"],
                "time_base": "1/90000",
                "pts_seconds": profile["lens_pts_seconds"],
                "metadata": {"lens": lens_id},
            }
            for stream_index, lens_id in enumerate(("front", "rear"))
        ],
        format_metadata={"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
    )
    normalized = normalize_native_360_capture(
        capture_root=capture_root,
        output_root=tmp_path / "native360-output",
        intake_id="hermetic-native-360",
        capture_digest=profile["source_capture_digest"],
        camera_metadata=metadata,
        probe_receipts_by_path={"native/capture.insv": receipt},
        source_commit_sha=spec["source_commit_sha"],
        implementation_digest=spec["implementation_digest"],
        authority_used=spec["authority"],
        timestamp=spec["timestamp"],
        maximum_source_bytes=1024,
    )
    artifact_root = next((tmp_path / "native360-output").glob("native_360_normalization_*"))
    rig = json.loads((artifact_root / "camera_360_rig_declaration.json").read_text())
    binding = json.loads((artifact_root / "dual_fisheye_stream_binding.json").read_text())
    rig_result = validate_camera_rig(
        {
            "schema_version": CAMERA_RIG_VALIDATION_REQUEST_SCHEMA_VERSION,
            "source_capture_digest": profile["source_capture_digest"],
            "native_360_normalization_digest": normalized["native_360_normalization_digest"],
            "rig_declaration": rig,
            "dual_fisheye_binding": binding,
            "agent_may_change_calibration": False,
            "timestamp": spec["timestamp"],
        }
    )
    dataset_root = tmp_path / "native360-dataset"
    ffmpeg_fixture = tmp_path / "ffmpeg-hermetic-fixture"
    ffmpeg_fixture.write_bytes(b"ffmpeg-hermetic-fixture")

    def decode_runner(argv, _timeout, _maximum_output):
        command = list(argv)
        if "-version" in command:
            return subprocess.CompletedProcess(
                command, 0, b"ffmpeg version hermetic-fixture\n", b""
            )
        stream_index = int(command[command.index("-map") + 1].split(":")[1])
        Image.fromarray(
            np.full(
                (profile["height"], profile["width"]),
                48 if stream_index == 0 else 192,
                dtype=np.uint8,
            )
        ).save(command[-1])
        return subprocess.CompletedProcess(command, 0, b"", b"")

    decode_manifest = decode_native_360_lens_observations(
        capture_root=capture_root,
        artifact_root=dataset_root,
        capture_digest=profile["source_capture_digest"],
        normalization_result=normalized,
        rig_declaration=rig,
        dual_fisheye_binding=binding,
        implementation_digest=spec["implementation_digest"],
        source_commit_sha=spec["source_commit_sha"],
        authority_used=spec["authority"],
        timestamp=spec["timestamp"],
        ffmpeg_executable=ffmpeg_fixture,
        runner=decode_runner,
    )
    dataset = compile_native_360_grouped_frame_dataset(
        artifact_root=dataset_root,
        intake_id="hermetic-native-360",
        capture_digest=profile["source_capture_digest"],
        normalization_result=normalized,
        rig_declaration=rig,
        dual_fisheye_binding=binding,
        lens_decode_manifest=decode_manifest,
        decoded_lens_frames=decode_manifest["frames"],
        runtime_identity=decode_manifest["runtime_identity"],
        runtime_digest=decode_manifest["runtime_digest"],
        implementation_digest=spec["implementation_digest"],
        source_commit_sha=spec["source_commit_sha"],
        authority_used=spec["authority"],
        timestamp=spec["timestamp"],
    )
    report = _terminal_report(
        spec=spec,
        profile="camera_360_native",
        source_digest=profile["source_capture_digest"],
        split_digest=dataset["train_heldout_split_digest"],
        input_digests=[
            normalized["native_360_normalization_digest"],
            dataset["dataset_manifest_digest"],
        ],
        recorded_output_digests=[
            rig_result["camera_rig_validation_result_digest"],
            dataset["dataset_manifest_digest"],
        ],
        blocker=profile["expected_terminal_blocker"],
        ceilings=_ceilings(decoded=True, calibrated=False),
        calibration_status={
            "status": "fixed_native_rig_validated",
            "camera_rig_validation_result_digest": rig_result[
                "camera_rig_validation_result_digest"
            ],
            "metric_scale_established": False,
        },
    )

    assert normalized["status"] == "normalized"
    assert dataset["candidate_dataset_contains_hidden_heldout_pixels"] is False
    assert dataset["camera_calibration_binding"] == {
        "camera_360_rig_declaration_digest": rig["rig_declaration_digest"]
    }
    assert rig_result["status"] == "validated"
    assert rig_result["camera_trajectory_proven"] is False
    assert rig_result["metric_scale_proven"] is False
    assert report["decision"] == "abstention"
    assert report["evidence_ceilings"]["decoded_observation_availability"] is True
    assert report["evidence_ceilings"]["calibrated_camera_trajectory"] is False


def test_equirectangular_360_vertical_compiles_shared_center_views_then_abstains(
    tmp_path: Path,
) -> None:
    spec = _spec()
    profile = spec["camera_360_equirectangular"]
    capture_root = tmp_path / "equirectangular"
    panorama = capture_root / "retained/panorama.png"
    panorama.parent.mkdir(parents=True)
    longitude = np.linspace(0, 255, profile["width"], dtype=np.uint8)[None, :]
    latitude = np.linspace(0, 255, profile["height"], dtype=np.uint8)[:, None]
    image = np.zeros((profile["height"], profile["width"], 3), dtype=np.uint8)
    image[..., 0] = longitude
    image[..., 1] = latitude
    image[..., 2] = 127
    Image.fromarray(image).save(panorama)
    split_digest = canonical_digest(
        {
            "schema_version": "hermetic_panorama_split.v1",
            "training": ["panorama-0001"],
            "held_out": [],
            "candidate_can_change_split": False,
        }
    )
    compilation = compile_equirectangular_virtual_rig(
        capture_root=capture_root,
        output_root=tmp_path / "equirectangular-output",
        intake_id="hermetic-equirectangular-360",
        capture_digest=profile["source_capture_digest"],
        stitched_source_metadata={
            "schema_version": "stitched_equirectangular_source.v1",
            "source_capture_digest": profile["source_capture_digest"],
            "projection": "equirectangular_2_to_1",
            "stitching_provenance": "official_sdk_produced",
            "producer_identity": "hermetic-fixture-sdk",
            "stitching_receipt_digest": profile["stitching_receipt_digest"],
            "original_360_source_preserved": True,
            "original_360_source_digest": profile["original_source_digest"],
            "spherical_pixel_mapping": {
                "longitude": "x maps [-pi,pi) with seam at encoded x=0",
                "latitude": "y maps [+pi/2,-pi/2]",
                "pixel_centers": True,
            },
        },
        panorama_observations=[
            {
                "observation_id": "panorama-0001",
                "relative_path": "retained/panorama.png",
                "digest": _file_digest(panorama),
                "t_video_sec": 1.25,
                "split": "training",
            }
        ],
        source_commit_sha=spec["source_commit_sha"],
        implementation_digest=spec["implementation_digest"],
        authority_used=spec["authority"],
        timestamp=spec["timestamp"],
        access_scope="candidate_training_and_validation_only",
        parent_artifact_or_event={"split_digest": split_digest},
    )
    report = _terminal_report(
        spec=spec,
        profile="camera_360_equirectangular",
        source_digest=profile["source_capture_digest"],
        split_digest=split_digest,
        input_digests=[compilation["equirectangular_compilation_digest"]],
        recorded_output_digests=[compilation["equirectangular_compilation_digest"]],
        blocker=profile["expected_terminal_blocker"],
        ceilings=_ceilings(decoded=True, calibrated=False),
        calibration_status={
            "status": "shared_center_virtual_rig_compiled",
            "virtual_rig_digest": compilation["output_digests"]["virtual_rig_digest"],
            "camera_trajectory_proven": False,
            "metric_scale_proven": False,
        },
        selected_frames=[{"frame_id": "panorama-0001"}],
    )

    assert compilation["virtual_observation_count"] == profile["expected_virtual_view_count"]
    assert compilation["virtual_views_are_independent_physical_cameras"] is False
    assert compilation["camera_trajectory_proven"] is False
    assert report["decision"] == "abstention"
    assert report["evidence_ceilings"]["decoded_observation_availability"] is True
    assert report["evidence_ceilings"]["metric_scale"] is False
