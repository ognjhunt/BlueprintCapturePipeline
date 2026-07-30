from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from blueprint_pipeline import local_reconstruction_adapters as adapters
from blueprint_pipeline.local_reconstruction_adapters import (
    LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
    LOCAL_DECODED_OBSERVATION_ADAPTER,
    LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
    LocalArkitMetricScaffoldAdapter,
    LocalDecodedObservationAdapter,
    LocalExternalReconstructionImportAdapter,
    LocalReconstructionAdapterError,
    authorized_local_reconstruction_adapter_registry,
    decoded_observation_method_profile,
)
from blueprint_pipeline.reconstruction_capability import plan_reconstruction_methods


CAPTURE_DIGEST = "sha256:" + "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _stub_media_tools(
    monkeypatch: pytest.MonkeyPatch, *, presentation_times: list[float] | None = None
) -> None:
    times = presentation_times or [0.1, 0.2]
    monkeypatch.setattr(
        "blueprint_pipeline.local_reconstruction_adapters.shutil.which",
        lambda name: f"/fake/{name}",
    )

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "-version":
            return subprocess.CompletedProcess(
                command, 0, f"{Path(command[0]).name} version test\n", ""
            )
        if "-show_frames" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "streams": [
                            {
                                "index": 0,
                                "codec_name": "h264",
                                "width": 64,
                                "height": 48,
                                "avg_frame_rate": "10/1",
                                "time_base": "1/1000",
                                "pix_fmt": "yuv420p",
                                "color_space": "bt709",
                                "tags": {"rotate": "90"},
                            }
                        ],
                        "frames": [
                            {
                                "best_effort_timestamp": str(round(value * 1000)),
                                "best_effort_timestamp_time": str(value),
                            }
                            for value in times
                        ],
                    }
                ),
                "",
            )
        output = Path(command[-1])
        selected = next(value for value in command if value.startswith("select="))
        value = int(selected.split(",")[-1].rstrip(")"))
        Image.new("RGB", (64, 48), color=(value * 40, 20, 10)).save(output, format="PNG")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("blueprint_pipeline.local_reconstruction_adapters.subprocess.run", fake_run)


def test_ordinary_video_plans_decoded_observations_not_calibration() -> None:
    profile = decoded_observation_method_profile(execution_authorized=True)
    plan = plan_reconstruction_methods(
        intake_id="intake-1",
        capture_digest=CAPTURE_DIGEST,
        capture_authority_profile="monocular_video",
        claim_ceiling={"metric_geometry": False},
        requested_claim_types=["task_discovery", "perception_visibility"],
        permitted_provider_identities=["local"],
        method_profiles=[profile],
    )

    assert plan["status"] == "planned"
    assert plan["required_representations"] == ["decoded_observation_frames"]
    assert plan["selected_methods"][0]["representations"] == ["decoded_observation_frames"]
    assert plan["selected_methods"][0]["adapter_reference"] == (LOCAL_DECODED_OBSERVATION_ADAPTER)
    assert "calibrated_frames" not in plan["required_representations"]


def test_frame_sampling_uses_actual_presentation_timeline() -> None:
    assert adapters._sample_indexes([0.0, 0.01, 0.02, 1.0, 2.0], 3) == [0, 3, 4]
    assert adapters._sample_indexes([0.0, 0.01, 0.02, 0.03, 10.0], 4) == [0, 1, 3, 4]


def test_decoded_observation_adapter_is_deterministic_and_non_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_media_tools(monkeypatch, presentation_times=[0.1, 0.2, 0.3, 0.4])
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    (capture_root / "video.mp4").write_bytes(b"retained-video")
    output_root = tmp_path / "derived"
    kwargs = {
        "intake_id": "intake-1",
        "capture_digest": CAPTURE_DIGEST,
        "capture_authority_profile": "monocular_video",
        "capture_root": capture_root,
        "video_relative_path": "video.mp4",
        "output_root": output_root,
        "rights_and_retention": {"external_processing": False},
        "maximum_frames": 4,
    }

    first = LocalDecodedObservationAdapter().execute(**kwargs)
    second = LocalDecodedObservationAdapter().execute(**kwargs)

    assert first == second
    assert first["outputs"] == ["decoded_observation_frames"]
    assert first["camera_solution"] == {"status": "not_available", "calibrated": False}
    assert len(first["source_frames"]["sampled_frames"]) == 3
    assert {
        row["t_video_sec"] for row in first["source_frames"]["sampled_frames"]
    } < {0.0, 0.1, 0.2, 0.3}
    assert first["claim_ceiling"]["calibrated_camera_poses"] is False
    assert first["claim_ceiling"]["metric_geometry"] is False
    assert first["claim_ceiling"]["physical_task_success"] is False
    dataset_path = next(output_root.glob("**/reconstruction_dataset_manifest.json"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset["stream_metadata"]["display_rotation_degrees"] == 90.0
    assert dataset["train_heldout_split_digest"].startswith("sha256:")
    assert dataset["candidate_dataset_contains_hidden_heldout_pixels"] is False
    index = json.loads(next(output_root.glob("**/decoded_observation_index.json")).read_text())
    assert len(index["sampled_frames"]) == 3
    assert index["sampled_frames"][0]["image_metadata"] == {
        "width": 64,
        "height": 48,
        "pixel_orientation": "encoded_source_no_autorotate",
    }
    assert "hidden_heldout_evaluator_manifest" not in first["asset_references"]
    assert first["source_frames"]["hidden_heldout_frame_count"] == 1


def _arkit_bundle(root: Path) -> None:
    (root / "arkit/depth").mkdir(parents=True)
    (root / "arkit/confidence").mkdir(parents=True)
    (root / "walkthrough.mov").write_bytes(b"retained-video")
    (root / "arkit/depth/000001.png").write_bytes(b"depth")
    (root / "arkit/confidence/000001.png").write_bytes(b"confidence")
    _write_json(
        root / "manifest.json",
        {
            "capture_schema_version": "3.2.0",
            "capture_profile_id": "iphone_arkit_lidar",
            "coordinate_frame_session_id": "cfs-1",
            "capture_capabilities": {
                "camera_pose": True,
                "camera_intrinsics": True,
                "depth": True,
                "depth_confidence": True,
                "tracking_state": True,
                "tracking_state_rows": 2,
            },
        },
    )
    _write_json(
        root / "video_track.json",
        {
            "video_file": "walkthrough.mov",
            "frame_count": 2,
            "frame_count_source": "decoded_sample_presentation_timestamps",
            "decoded_pts_verified": True,
            "write_attempt_count": 3,
            "retained_frame_count": 2,
            "dropped_frame_count": 1,
        },
    )
    _write_jsonl(
        root / "video_frame_retention.jsonl",
        [
            {
                "write_attempt_index": 0,
                "frame_id": "000001",
                "retention_status": "retained",
                "drop_reason": None,
                "encoded_frame_index": 0,
                "t_video_sec": 0.0,
            },
            {
                "write_attempt_index": 1,
                "frame_id": "000001-drop",
                "retention_status": "dropped_backpressure",
                "drop_reason": "asset_writer_input_not_ready",
                "encoded_frame_index": None,
                "t_video_sec": None,
            },
            {
                "write_attempt_index": 2,
                "frame_id": "000002",
                "retention_status": "retained",
                "drop_reason": None,
                "encoded_frame_index": 1,
                "t_video_sec": 0.1,
            },
        ],
    )
    _write_jsonl(
        root / "sync_map.jsonl",
        [
            {
                "frame_id": "000001",
                "t_video_sec": 0.0,
                "t_capture_sec": 0.0,
                "sync_status": "encoded_decoded_pts_match",
                "pose_frame_id": "000001",
                "encoded_frame_index": 0,
                "write_attempt_index": 0,
            },
            {
                "frame_id": "000002",
                "t_video_sec": 0.1,
                "t_capture_sec": 0.1,
                "sync_status": "encoded_decoded_pts_match",
                "pose_frame_id": "000002",
                "encoded_frame_index": 1,
                "write_attempt_index": 2,
            },
        ],
    )
    pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    _write_jsonl(
        root / "arkit/poses.jsonl",
        [
            {"frame_id": "000001", "coordinate_frame_session_id": "cfs-1", "T_world_camera": pose},
            {"frame_id": "000002", "coordinate_frame_session_id": "cfs-1", "T_world_camera": pose},
        ],
    )
    _write_jsonl(
        root / "arkit/frames.jsonl",
        [{"frame_id": "000001"}, {"frame_id": "000002"}],
    )
    _write_json(
        root / "arkit/session_intrinsics.json",
        {
            "coordinate_frame_session_id": "cfs-1",
            "intrinsics": {"fx": 50, "fy": 50, "cx": 32, "cy": 24, "width": 64, "height": 48},
        },
    )
    _write_json(
        root / "recording_session.json",
        {
            "coordinate_frame_session_id": "cfs-1",
            "world_frame_definition": "arkit_world_origin_at_session_start",
            "units": "meters",
            "handedness": "right_handed",
            "gravity_aligned": True,
            "session_reset_count": 0,
        },
    )
    _write_json(
        root / "arkit/depth_manifest.json",
        {
            "frames": [
                {
                    "frame_id": "000001",
                    "depth_path": "arkit/depth/000001.png",
                    "paired_confidence_path": "arkit/confidence/000001.png",
                }
            ]
        },
    )
    _write_json(
        root / "arkit/confidence_manifest.json",
        {
            "frames": [
                {
                    "frame_id": "000001",
                    "confidence_path": "arkit/confidence/000001.png",
                    "paired_depth_path": "arkit/depth/000001.png",
                }
            ]
        },
    )


def test_arkit_metric_scaffold_requires_exact_v32_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_media_tools(monkeypatch)
    capture_root = tmp_path / "capture"
    _arkit_bundle(capture_root)
    adapter = LocalArkitMetricScaffoldAdapter()

    result = adapter.execute(
        intake_id="intake-1",
        capture_digest=CAPTURE_DIGEST,
        capture_root=capture_root,
        output_root=tmp_path / "derived",
        rights_and_retention={"external_processing": False},
        maximum_frames=2,
    )

    assert result["outputs"] == [
        "calibrated_frames",
        "decoded_observation_frames",
        "metric_reference_layer",
    ]
    assert result["camera_solution"]["status"] == "raw_contract_3_2_verified"
    assert result["validation_metrics"]["retained_sync_pose_count"] == 2
    assert result["validation_metrics"]["depth_confidence_pair_count"] == 1
    assert result["claim_ceiling"]["sensor_declared_metric_scale"] is True
    assert result["claim_ceiling"]["metric_scale"] is False
    assert result["claim_ceiling"]["metric_reference_layer"] is False
    assert result["validation_metrics"]["independent_metric_scale_validation_passed"] is False
    assert result["validation_metrics"]["pose_refinement_executed"] is False
    assert result["validation_metrics"]["depth_surface_compilation_ready"] is False
    assert "depth_manifest_v2_required" in result["uncertainty_map"][
        "depth_surface_source_declaration_blockers"
    ]
    assert result["asset_references"]["arkit_reconstruction_dataset_export"][
        "digest"
    ].startswith("sha256:")
    assert result["claim_ceiling"]["complete_geometry"] is False
    assert result["claim_ceiling"]["collision_geometry"] is False
    assert result["claim_ceiling"]["physical_task_success"] is False

    sync_path = capture_root / "sync_map.jsonl"
    rows = [json.loads(line) for line in sync_path.read_text().splitlines()]
    rows[1]["sync_status"] = "exact_frame_id_match"
    _write_jsonl(sync_path, rows)
    with pytest.raises(LocalReconstructionAdapterError, match="decoded_pts_mismatch"):
        adapter.execute(
            intake_id="intake-1",
            capture_digest=CAPTURE_DIGEST,
            capture_root=capture_root,
            output_root=tmp_path / "derived-unsafe",
            rights_and_retention={"external_processing": False},
        )


def test_arkit_metric_scaffold_records_explicit_depth_surface_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_media_tools(monkeypatch)
    capture_root = tmp_path / "capture"
    _arkit_bundle(capture_root)
    depth_path = capture_root / "arkit/depth_manifest.json"
    confidence_path = capture_root / "arkit/confidence_manifest.json"
    depth = json.loads(depth_path.read_text(encoding="utf-8"))
    confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
    depth.update(
        {
            "schema_version": "arkit_depth_manifest.v2",
            "depth_encoding": "uint16_png",
            "scale_to_meters": 0.001,
            "camera_ray_convention": "arkit_x_right_y_up_z_backward",
            "depth_registered_to_arkit_camera": True,
            "depth_intrinsics": {
                "fx": 50,
                "fy": 50,
                "cx": 32,
                "cy": 24,
                "width": 64,
                "height": 48,
            },
        }
    )
    confidence.update(
        {
            "schema_version": "arkit_confidence_manifest.v2",
            "confidence_encoding": "uint8_png",
            "accepted_confidence_values": [2],
        }
    )
    _write_json(depth_path, depth)
    _write_json(confidence_path, confidence)

    result = LocalArkitMetricScaffoldAdapter().execute(
        intake_id="intake-1",
        capture_digest=CAPTURE_DIGEST,
        capture_root=capture_root,
        output_root=tmp_path / "derived",
        rights_and_retention={"external_processing": False},
        maximum_frames=2,
    )

    assert result["validation_metrics"]["depth_surface_compilation_ready"] is True
    assert result["uncertainty_map"][
        "depth_surface_source_declaration_blockers"
    ] == []
    scaffold_path = next((tmp_path / "derived").glob("**/arkit_metric_scaffold.json"))
    readiness = json.loads(scaffold_path.read_text(encoding="utf-8"))[
        "depth_surface_source_readiness"
    ]
    assert readiness["status"] == "ready_for_confidence_filtered_backprojection"
    assert readiness["agent_may_override"] is False
    assert readiness["source_declaration"]["declaration_digest"].startswith(
        "sha256:"
    )
    schema_root = Path(__file__).parents[1] / "docs" / "schemas"
    depth_schema = json.loads(
        (schema_root / "arkit_depth_manifest.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    confidence_schema = json.loads(
        (schema_root / "arkit_confidence_manifest.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(depth_schema).validate(depth)
    Draft202012Validator(confidence_schema).validate(confidence)


def test_local_reconstruction_registry_is_empty_by_default() -> None:
    assert authorized_local_reconstruction_adapter_registry([]) == {}
    authorized = authorized_local_reconstruction_adapter_registry(
        [
            LOCAL_DECODED_OBSERVATION_ADAPTER,
            LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
            LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
        ]
    )
    assert list(authorized) == [
        LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
        LOCAL_DECODED_OBSERVATION_ADAPTER,
        LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
    ]
    with pytest.raises(
        LocalReconstructionAdapterError, match="local_reconstruction_adapter_not_registered"
    ):
        authorized_local_reconstruction_adapter_registry(["provider://live"])


def test_decoded_observation_rejects_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_media_tools(monkeypatch)
    with pytest.raises(LocalReconstructionAdapterError, match="artifact_relative_path:unsafe"):
        LocalDecodedObservationAdapter().execute(
            intake_id="intake-1",
            capture_digest=CAPTURE_DIGEST,
            capture_authority_profile="monocular_video",
            capture_root=tmp_path,
            video_relative_path="../private.mov",
            output_root=tmp_path / "derived",
            rights_and_retention={"external_processing": False},
        )


def test_decoded_observation_rejects_duplicate_pts_and_corrupt_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"retained-video")

    def duplicate_pts(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "streams": [{"index": 0, "width": 64, "height": 48}],
                    "frames": [
                        {"best_effort_timestamp_time": "0.0"},
                        {"best_effort_timestamp_time": "0.0"},
                    ],
                }
            ),
            "",
        )

    monkeypatch.setattr(adapters.subprocess, "run", duplicate_pts)
    with pytest.raises(LocalReconstructionAdapterError, match="duplicate_pts"):
        adapters._probe_video(video, "/fake/ffprobe")

    monkeypatch.setattr(
        adapters.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 1, "", "corrupt"),
    )
    with pytest.raises(LocalReconstructionAdapterError, match="media_not_decodable"):
        adapters._probe_video(video, "/fake/ffprobe")


def test_decoded_observation_rejects_oversized_and_symlinked_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    video = capture_root / "video.mp4"
    video.write_bytes(b"retained-video")
    kwargs = {
        "intake_id": "intake-1",
        "capture_digest": CAPTURE_DIGEST,
        "capture_authority_profile": "monocular_video",
        "capture_root": capture_root,
        "output_root": tmp_path / "derived",
        "rights_and_retention": {"external_processing": False},
    }
    with pytest.raises(LocalReconstructionAdapterError, match="video_oversized"):
        LocalDecodedObservationAdapter().execute(
            **kwargs,
            video_relative_path="video.mp4",
            maximum_source_bytes=4,
        )

    link = capture_root / "linked.mp4"
    link.symlink_to(video)
    with pytest.raises(LocalReconstructionAdapterError, match="video_symlink_forbidden"):
        LocalDecodedObservationAdapter().execute(
            **kwargs,
            video_relative_path="linked.mp4",
        )


def test_decoded_observation_treats_malicious_filename_as_opaque_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_media_tools(monkeypatch)
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    filename = "$(touch owned).mp4"
    (capture_root / filename).write_bytes(b"retained-video")

    result = LocalDecodedObservationAdapter().execute(
        intake_id="intake-1",
        capture_digest=CAPTURE_DIGEST,
        capture_authority_profile="monocular_video",
        capture_root=capture_root,
        video_relative_path=filename,
        output_root=tmp_path / "derived",
        rights_and_retention={"external_processing": False},
        maximum_frames=2,
    )

    assert result["claim_ceiling"]["captured_observation"] is True
    assert not (tmp_path / "owned").exists()


def test_external_reconstruction_import_binds_ply_without_authority_upgrade(
    tmp_path: Path,
) -> None:
    payload = (
        b"ply\n"
        b"format ascii 1.0\n"
        b"element vertex 2\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"property uchar red\n"
        b"property uchar green\n"
        b"property uchar blue\n"
        b"element face 0\n"
        b"property list uchar int vertex_indices\n"
        b"end_header\n"
        b"0 0 0 255 0 0\n1 1 1 0 255 0\n"
    )
    capture_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    asset = tmp_path / "objects" / capture_digest[7:]
    asset.parent.mkdir(parents=True)
    asset.write_bytes(payload)
    adapter = LocalExternalReconstructionImportAdapter()

    result = adapter.execute(
        intake_id="intake-external-1",
        capture_digest=capture_digest,
        source_capture_binding={
            "source_capture_digest": "sha256:" + "b" * 64,
            "provider_identity": "mushroom-polycam",
        },
        capture_root=tmp_path,
        asset_relative_path=str(asset.relative_to(tmp_path)),
        original_filename="polycam_pointcloud.ply",
        output_root=tmp_path / "derived",
        rights_and_retention={"privacy": "restricted_local_only"},
        coordinate_frame_declaration={"status": "provider_declared_unverified"},
    )
    replay = adapter.execute(
        intake_id="intake-external-1",
        capture_digest=capture_digest,
        source_capture_binding={
            "source_capture_digest": "sha256:" + "b" * 64,
            "provider_identity": "mushroom-polycam",
        },
        capture_root=tmp_path,
        asset_relative_path=str(asset.relative_to(tmp_path)),
        original_filename="polycam_pointcloud.ply",
        output_root=tmp_path / "derived",
        rights_and_retention={"privacy": "restricted_local_only"},
        coordinate_frame_declaration={"status": "provider_declared_unverified"},
    )

    assert result == replay
    assert result["outputs"] == ["appearance_layer"]
    assert result["source_capture_binding"]["source_capture_digest"] == ("sha256:" + "b" * 64)
    assert result["validation_metrics"]["ply_header"]["elements"] == {
        "face": 0,
        "vertex": 2,
    }
    assert result["claim_ceiling"]["appearance_review"] is True
    assert result["claim_ceiling"]["raw_capture_authority"] is False
    assert result["claim_ceiling"]["captured_observation"] is False
    assert result["claim_ceiling"]["metric_geometry"] is False
    assert result["claim_ceiling"]["collision_geometry"] is False
    assert result["claim_ceiling"]["physical_task_success"] is False
    assert result["claim_ceiling"]["comparative_policy_ranking_verdict"] == "thesis_not_supported"

    with pytest.raises(LocalReconstructionAdapterError, match="format_not_supported"):
        adapter.execute(
            intake_id="intake-external-1",
            capture_digest=capture_digest,
            source_capture_binding={"source_capture_digest": "sha256:" + "b" * 64},
            capture_root=tmp_path,
            asset_relative_path=str(asset.relative_to(tmp_path)),
            original_filename="scene.glb",
            output_root=tmp_path / "other-derived",
            rights_and_retention={},
            coordinate_frame_declaration={},
        )


def test_external_reconstruction_import_accepts_supersplat_compressed_3dgs(
    tmp_path: Path,
) -> None:
    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element chunk 1\n"
        b"property float min_x\nproperty float min_y\nproperty float min_z\n"
        b"property float max_x\nproperty float max_y\nproperty float max_z\n"
        b"element vertex 1\n"
        b"property uint packed_position\n"
        b"property uint packed_rotation\n"
        b"property uint packed_scale\n"
        b"property uint packed_color\n"
        b"element sh 1\n"
        b"property uchar f_rest_0\n"
        b"end_header\n"
    )
    payload = header + (b"\x00" * 41)
    capture_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    asset = tmp_path / "3dgs_compressed.ply"
    asset.write_bytes(payload)

    result = LocalExternalReconstructionImportAdapter().execute(
        intake_id="intake-interiorgs-1",
        capture_digest=capture_digest,
        source_capture_binding={
            "source_capture_digest": "sha256:" + "c" * 64,
            "provider_identity": "interiorgs",
        },
        capture_root=tmp_path,
        asset_relative_path=asset.name,
        original_filename=asset.name,
        output_root=tmp_path / "derived",
        rights_and_retention={"allowed_use": "noncommercial_research_only"},
        coordinate_frame_declaration={
            "axes": {"x": "right", "y": "back", "z": "up"},
            "units": "meters",
            "status": "dataset_declared_unverified",
        },
    )

    ply = result["validation_metrics"]["ply_header"]
    assert ply["representation_profile"] == "supersplat_compressed_3dgs"
    assert ply["elements"] == {"chunk": 1, "sh": 1, "vertex": 1}
    assert result["outputs"] == ["appearance_layer"]
    assert result["claim_ceiling"]["metric_geometry"] is False
    assert result["claim_ceiling"]["collision_geometry"] is False
    assert result["claim_ceiling"]["physical_task_success"] is False
    assert result["claim_ceiling"]["comparative_policy_ranking_verdict"] == (
        "thesis_not_supported"
    )

    colorless = tmp_path / "colorless.ply"
    colorless_payload = (
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n0 0 0\n"
    )
    colorless.write_bytes(colorless_payload)
    with pytest.raises(LocalReconstructionAdapterError, match="ply_vertex_color_missing"):
        LocalExternalReconstructionImportAdapter().execute(
            intake_id="intake-external-2",
            capture_digest="sha256:" + hashlib.sha256(colorless_payload).hexdigest(),
            source_capture_binding={"source_capture_digest": "sha256:" + "b" * 64},
            capture_root=tmp_path,
            asset_relative_path=colorless.name,
            original_filename=colorless.name,
            output_root=tmp_path / "colorless-derived",
            rights_and_retention={},
            coordinate_frame_declaration={},
        )
