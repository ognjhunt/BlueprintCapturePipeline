from __future__ import annotations

import json

import numpy as np
import pytest

from blueprint_pipeline.adp_prospective_design import validate_episode_evidence_contract
from blueprint_pipeline.episode_visual_evidence import (
    _encode_episode_video,
    _ffmpeg_encode_command,
    _ffprobe_command,
    finalize_manipulation_evaluation_visual_evidence,
    finalize_multicamera_visual_evidence,
    finalize_visual_evidence,
    persist_multicamera_observation,
    persist_observation_frame,
    validate_multicamera_frame_manifest,
)


def test_review_video_uses_explicit_libx264_avc1_commands(tmp_path) -> None:
    video = tmp_path / "review.mp4"
    encode = _ffmpeg_encode_command(
        executable="/fixture/ffmpeg",
        video_path=video,
        width=448,
        height=224,
        frames_per_second=15.0,
        frame_count=61,
    )
    assert encode[0] == "/fixture/ffmpeg"
    assert encode[encode.index("-c:v") + 1] == "libx264"
    assert encode[encode.index("-tag:v") + 1] == "avc1"
    assert encode[encode.index("-pix_fmt") + 1] == "yuv420p"
    assert encode[encode.index("-frames:v") + 1] == "61"
    assert _ffprobe_command(
        executable="/fixture/ffprobe", video_path=video
    )[-1] == str(video)


def test_review_video_fails_closed_without_ffmpeg_toolchain(
    tmp_path, monkeypatch
) -> None:
    from PIL import Image

    frame = tmp_path / "frame.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(frame)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg_toolchain_unavailable"):
        _encode_episode_video(
            [frame], video_path=tmp_path / "review.mp4", frames_per_second=4.0
        )


def test_media_seal_retains_lossless_inputs_terminal_manifest_and_review_video(
    tmp_path,
) -> None:
    episode_id = "episode-media-1"
    first = persist_observation_frame(
        np.full((32, 64, 3), 17, dtype=np.uint8),
        output_dir=tmp_path,
        episode_id=episode_id,
        frame_index=0,
        kind="policy-input",
    )
    second = persist_observation_frame(
        np.full((32, 64, 3), 29, dtype=np.uint8),
        output_dir=tmp_path,
        episode_id=episode_id,
        frame_index=1,
        kind="policy-input",
    )
    terminal = persist_observation_frame(
        np.full((32, 64, 3), 43, dtype=np.uint8),
        output_dir=tmp_path,
        episode_id=episode_id,
        frame_index=2,
        kind="terminal-observation",
    )
    visual, artifacts = finalize_visual_evidence(
        output_dir=tmp_path,
        episode_id=episode_id,
        identity={"policy_id": "test-policy"},
        policy_input_frames=[first, second],
        terminal_observation=terminal,
    )
    manifest_artifact = next(
        row for row in artifacts if row["role"] == "observation_frame_manifest"
    )
    manifest = json.loads((tmp_path / manifest_artifact["relative_path"]).read_text())
    video = next(row for row in artifacts if row["role"] == "episode_video")

    assert manifest["frame_manifest_digest"] == visual["frame_manifest_digest"]
    assert len(manifest["policy_input_frames"]) == 2
    assert (tmp_path / video["relative_path"]).read_bytes()[4:8] == b"ftyp"
    assert visual["video"]["codec"] == "h264"
    assert visual["video"]["fourcc"] == "avc1"
    assert visual["video"]["encoder"] == "ffmpeg_libx264"
    assert visual["video"]["decoded_frame_count"] == 3
    assert visual["video"]["decode_round_trip_passed"] is True
    assert visual["video"]["ffprobe_passed"] is True
    episode = {
        "episode_id": episode_id,
        "status": "completed",
        "policy_query_count": 2,
        "visual_evidence": visual,
        "artifacts": artifacts,
        "evaluator": {
            "owner": "environment_not_policy",
            "grader_type": "deterministic_simulator_state",
            "success_source": "frozen_object_state_predicates",
            "policy_self_report_used": False,
        },
        "success_evidence": {
            "grader_type": "deterministic_simulator_state",
            "policy_self_report_used": False,
        },
    }
    admission = validate_episode_evidence_contract(episode)
    assert admission["status"] == "admitted"
    assert admission["completed_media_contract"] is True

    with pytest.raises(FileExistsError, match="overwrite_forbidden"):
        persist_observation_frame(
            np.full((32, 64, 3), 17, dtype=np.uint8),
            output_dir=tmp_path,
            episode_id=episode_id,
            frame_index=0,
            kind="policy-input",
        )


def _calibration(width: int = 64, height: int = 32) -> dict:
    return {
        "camera_model": "pinhole",
        "intrinsic_matrix": [
            [48.0, 0.0, width / 2],
            [0.0, 48.0, height / 2],
            [0.0, 0.0, 1.0],
        ],
        "world_from_camera": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "resolution": [width, height],
        "near_m": 0.01,
        "far_m": 20.0,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def _multicamera_observation(
    tmp_path,
    *,
    episode_id: str,
    index: int,
    kind: str,
) -> dict:
    images = {
        "external": np.full((32, 64, 3), 10 + index, dtype=np.uint8),
        "wrist": np.full((32, 64, 3), 20 + index, dtype=np.uint8),
    }
    return persist_multicamera_observation(
        images,
        output_dir=tmp_path,
        episode_id=episode_id,
        observation_index=index,
        kind=kind,
        timestamp_ns=100 + index,
        simulation_time_s=index / 15.0,
        calibrations={camera_id: _calibration() for camera_id in images},
        source_devices={camera_id: "cuda:0" for camera_id in images},
        synchronizations={
            camera_id: {
                "host_bytes_ready": True,
                "method": "explicit_cuda_event_then_copy",
                "dlpack_ownership": "producer_retained_until_copy_complete",
            }
            for camera_id in images
        },
    )


def test_multicamera_media_retains_exact_views_calibration_and_timestamps(
    tmp_path,
) -> None:
    episode_id = "episode-multicamera-1"
    observations = [
        _multicamera_observation(
            tmp_path, episode_id=episode_id, index=index, kind="policy-input"
        )
        for index in range(2)
    ]
    terminal = _multicamera_observation(
        tmp_path,
        episode_id=episode_id,
        index=2,
        kind="terminal-observation",
    )

    visual, artifacts = finalize_multicamera_visual_evidence(
        output_dir=tmp_path,
        episode_id=episode_id,
        identity={
            "policy_id": "pi05_droid",
            "scenario_instance_digest": "sha256:" + "1" * 64,
        },
        policy_input_observations=observations,
        terminal_observation=terminal,
        frames_per_second=15.0,
    )
    manifest_artifact = next(
        row
        for row in artifacts
        if row["role"] == "multicamera_observation_frame_manifest"
    )
    manifest = json.loads(
        (tmp_path / manifest_artifact["relative_path"]).read_text(encoding="utf-8")
    )

    validate_multicamera_frame_manifest(
        manifest, output_dir=tmp_path, verify_files=True
    )
    assert visual["required_camera_ids"] == ["external", "wrist"]
    assert visual["policy_input_observation_count"] == 2
    assert visual["policy_input_frame_count"] == 4
    assert set(visual["videos"]) == {"external", "wrist"}
    assert all(
        row["calibration_digest"].startswith("sha256:")
        for row in artifacts
        if row["role"] == "policy_input_camera_frame"
    )
    assert [
        row["timestamp_ns"] for row in manifest["policy_input_observations"]
    ] == [100, 101]


def test_manipulation_profile_requires_review_only_overview_video(tmp_path) -> None:
    episode_id = "episode-overview-profile"
    camera_ids = ("external", "wrist", "overview")

    def observation(index: int, kind: str) -> dict:
        images = {
            camera_id: np.full((32, 64, 3), 20 + offset + index, dtype=np.uint8)
            for offset, camera_id in enumerate(camera_ids)
        }
        return persist_multicamera_observation(
            images,
            output_dir=tmp_path,
            episode_id=episode_id,
            observation_index=index,
            kind=kind,
            timestamp_ns=100 + index,
            simulation_time_s=index / 15.0,
            calibrations={camera_id: _calibration() for camera_id in camera_ids},
            source_devices={camera_id: "cuda:0" for camera_id in camera_ids},
            synchronizations={
                camera_id: {"host_bytes_ready": True, "method": "test"}
                for camera_id in camera_ids
            },
        )

    visual, _ = finalize_manipulation_evaluation_visual_evidence(
        output_dir=tmp_path,
        episode_id=episode_id,
        identity={"episode_kind": "simulator_evaluation"},
        policy_input_observations=[observation(0, "policy-input")],
        review_observations=[observation(1, "review-sample")],
        terminal_observation=observation(2, "terminal-observation"),
    )

    assert visual["required_camera_ids"] == ["external", "overview", "wrist"]
    assert visual["review_only_camera_ids"] == ["overview"]
    assert visual["policy_input_frame_count"] == 2
    assert visual["review_observation_count"] == 1
    assert visual["review_frame_count"] == 3
    assert set(visual["videos"]) == {"external", "wrist", "overview"}


def test_multicamera_media_rejects_unsynchronized_or_changed_frame(tmp_path) -> None:
    with pytest.raises(ValueError, match="host_bytes_not_synchronized"):
        persist_multicamera_observation(
            {"external": np.zeros((32, 64, 3), dtype=np.uint8)},
            output_dir=tmp_path,
            episode_id="bad-sync",
            observation_index=0,
            kind="policy-input",
            timestamp_ns=1,
            simulation_time_s=0.0,
            calibrations={"external": _calibration()},
            source_devices={"external": "cuda:0"},
            synchronizations={"external": {"host_bytes_ready": False}},
        )

    episode_id = "episode-multicamera-tamper"
    policy_input = _multicamera_observation(
        tmp_path, episode_id=episode_id, index=0, kind="policy-input"
    )
    terminal = _multicamera_observation(
        tmp_path,
        episode_id=episode_id,
        index=1,
        kind="terminal-observation",
    )
    _, artifacts = finalize_multicamera_visual_evidence(
        output_dir=tmp_path,
        episode_id=episode_id,
        identity={"policy_id": "groot_n17_droid"},
        policy_input_observations=[policy_input],
        terminal_observation=terminal,
    )
    manifest_artifact = next(
        row
        for row in artifacts
        if row["role"] == "multicamera_observation_frame_manifest"
    )
    manifest = json.loads(
        (tmp_path / manifest_artifact["relative_path"]).read_text(encoding="utf-8")
    )
    external_path = (
        tmp_path
        / manifest["policy_input_observations"][0]["views"]["external"][
            "relative_path"
        ]
    )
    from PIL import Image

    changed = np.full((32, 64, 3), 99, dtype=np.uint8)
    Image.fromarray(changed, mode="RGB").save(external_path, format="PNG")

    with pytest.raises(
        ValueError,
        match="multicamera_frame_manifest_(png|raw)_digest_mismatch",
    ):
        validate_multicamera_frame_manifest(
            manifest, output_dir=tmp_path, verify_files=True
        )
