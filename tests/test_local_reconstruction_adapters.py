from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest

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


def _stub_media_tools(monkeypatch: pytest.MonkeyPatch) -> None:
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
                            }
                        ],
                        "frames": [
                            {"best_effort_timestamp": "100", "best_effort_timestamp_time": "0.100"},
                            {"best_effort_timestamp": "200", "best_effort_timestamp_time": "0.200"},
                        ],
                    }
                ),
                "",
            )
        output = Path(command[-1])
        selected = next(value for value in command if value.startswith("select="))
        output.write_bytes(f"png:{selected}".encode())
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


def test_decoded_observation_adapter_is_deterministic_and_non_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_media_tools(monkeypatch)
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
        "maximum_frames": 2,
    }

    first = LocalDecodedObservationAdapter().execute(**kwargs)
    second = LocalDecodedObservationAdapter().execute(**kwargs)

    assert first == second
    assert first["outputs"] == ["decoded_observation_frames"]
    assert first["camera_solution"] == {"status": "not_available", "calibrated": False}
    assert first["source_frames"]["sampled_frames"][0]["t_video_sec"] == 0.0
    assert first["source_frames"]["sampled_frames"][1]["t_video_sec"] == 0.1
    assert first["claim_ceiling"]["calibrated_camera_poses"] is False
    assert first["claim_ceiling"]["metric_geometry"] is False
    assert first["claim_ceiling"]["physical_task_success"] is False


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
    assert result["claim_ceiling"]["metric_scale"] is True
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

    colorless = tmp_path / "colorless.ply"
    colorless_payload = (
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n0 0 0\n"
    )
    colorless.write_bytes(colorless_payload)
    with pytest.raises(LocalReconstructionAdapterError, match="ply_vertex_color_missing"):
        adapter.execute(
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
