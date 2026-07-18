from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

from blueprint_pipeline import groot_oscar_episode_review as review


def _write_trace(episode_dir: Path, rows: list[dict]) -> None:
    (episode_dir / review.TRACE_NAME).write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_isaac_execution_evidence(episode_dir: Path, *, step_count: int) -> Path:
    state_dir = episode_dir.parent / review.ISAAC_STATE_DIR_NAME
    frames_dir = state_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame_bindings: dict[str, dict] = {}
    initial_bindings: dict[str, dict] = {}
    baseline_digest = "b" * 64
    common_identity = {
        "simulator_session_id": "isaac-session-1",
        "stage_id": "kitchen-stage-1",
        "attempt_id": "attempt-1",
        "launch_nonce": "launch-1",
    }
    for role in review.ISAAC_CAMERA_ROLES:
        name = f"{role}_0000.png"
        path = frames_dir / name
        path.write_bytes(f"png:{role}:initial".encode())
        initial_bindings[name] = {
            "camera_role": role,
            "step_index": 0,
            "review_frame_index": 0,
            "control_frame_global_index": 0,
            "initial_frame": True,
            "camera_motion_model": (
                "rigid_head_local_transform"
                if role == "robot_pov"
                else "task_framed_third_person_review"
            ),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "episode_baseline_digest": baseline_digest,
            **common_identity,
        }
    (frames_dir / review.ISAAC_INITIAL_FRAME_BINDINGS_NAME).write_text(
        json.dumps(
            {
                "schema_version": review.ISAAC_INITIAL_FRAME_BINDINGS_SCHEMA_VERSION,
                "frames": initial_bindings,
            }
        ),
        encoding="utf-8",
    )

    control_frame_global_index = 0
    review_frame_index = 0
    physics_step_count = 100
    simulation_time_seconds = 2.0
    for source_step_index in range(1, step_count + 1):
        action_sha256 = hashlib.sha256(f"action-{source_step_index}".encode()).hexdigest()
        binding_fields = {
            "source_action_sha256": action_sha256,
            "before_timestamp": str(100 + (source_step_index - 1) * 10),
            "after_timestamp": str(105 + (source_step_index - 1) * 10),
            **common_identity,
        }
        controller_measurements: list[dict] = []
        sampled_review_indices: list[int] = []
        terminal_review_index = None
        for horizon_frame_index in range(5):
            control_frame_global_index += 1
            physics_step_count_before = physics_step_count
            physics_step_count += 1
            simulation_time_before_seconds = simulation_time_seconds
            simulation_time_seconds += 0.02
            terminal_frame = (
                source_step_index == step_count and horizon_frame_index == 4
            )
            scheduled_frame = control_frame_global_index % 5 == 0
            sampled_for_review = scheduled_frame or terminal_frame
            source_action_frame_sha256 = hashlib.sha256(
                f"action-frame:{control_frame_global_index}".encode()
            ).hexdigest()
            artifact_rows: list[dict] = []
            sampled_index = None
            if sampled_for_review:
                review_frame_index += 1
                sampled_index = review_frame_index
                sampled_review_indices.append(review_frame_index)
                if terminal_frame:
                    terminal_review_index = review_frame_index
                for role in review.ISAAC_CAMERA_ROLES:
                    name = f"{role}_{review_frame_index:04d}.png"
                    path = frames_dir / name
                    path.write_bytes(
                        f"png:{role}:control:{control_frame_global_index}".encode()
                    )
                    frame_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                    artifact_rows.append(
                        {
                            "camera_role": role,
                            "frame_index": review_frame_index,
                            "control_frame_global_index": control_frame_global_index,
                            "path": str(path),
                            "sha256": frame_sha256,
                        }
                    )
                    frame_bindings[name] = {
                        "camera_role": role,
                        "camera_motion_model": review.ISAAC_CAMERA_MOTION_MODELS[
                            role
                        ],
                        "step_index": review_frame_index,
                        "control_frame_global_index": control_frame_global_index,
                        "physics_step_count_before": physics_step_count_before,
                        "physics_step_count_after": physics_step_count,
                        "physics_step_delta": 1,
                        "simulation_time_before_seconds": (
                            simulation_time_before_seconds
                        ),
                        "simulation_time_after_seconds": simulation_time_seconds,
                        "simulation_time_delta_seconds": 0.02,
                        "outer_source_step_index": source_step_index,
                        "horizon_frame_index": horizon_frame_index,
                        "controller_frame_index": horizon_frame_index,
                        "source_action_frame_sha256": source_action_frame_sha256,
                        "semantic_terminal_frame": terminal_frame,
                        "sha256": frame_sha256,
                        **binding_fields,
                    }
            controller_measurements.append(
                {
                    "control_frame_global_index": control_frame_global_index,
                    "physics_step_count_before": physics_step_count_before,
                    "physics_step_count_after": physics_step_count,
                    "physics_step_delta": 1,
                    "simulation_time_before_seconds": simulation_time_before_seconds,
                    "simulation_time_after_seconds": simulation_time_seconds,
                    "simulation_time_delta_seconds": 0.02,
                    "horizon_frame_index": horizon_frame_index,
                    "controller_frame_index": horizon_frame_index,
                    "source_action_frame_sha256": source_action_frame_sha256,
                    "registered_transition_passed": terminal_frame,
                    "scheduled_review_frame": scheduled_frame,
                    "sampled_for_review": sampled_for_review,
                    "review_frame_index": sampled_index,
                    "review_frame_artifacts": artifact_rows,
                    "semantic_terminal_frame": terminal_frame,
                }
            )
        measurement = {
            "schema_version": "task_transition_measurement.v1",
            "source_step_index": source_step_index,
            "evidence_step_index": source_step_index,
            "episode_baseline_digest": baseline_digest,
            "controller_horizon_executed_frame_count": len(controller_measurements),
            "controller_review_frame_count": len(sampled_review_indices),
            "controller_review_frame_indices": sampled_review_indices,
            "controller_terminal_review_frame_index": terminal_review_index,
            "controller_horizon_terminated_on_semantic_success": (
                terminal_review_index is not None
            ),
            "controller_frame_measurements": controller_measurements,
            **binding_fields,
        }
        (state_dir / f"task_measurement_{source_step_index:04d}.json").write_text(
            json.dumps(measurement), encoding="utf-8"
        )
    (frames_dir / "frame_step_bindings.json").write_text(
        json.dumps(
            {
                "schema_version": review.ISAAC_FRAME_BINDINGS_SCHEMA_VERSION,
                "frames": frame_bindings,
            }
        ),
        encoding="utf-8",
    )
    return state_dir


def _successful_media_run(command, *, review_frame_count: int = 2):  # noqa: ANN001
    command = list(command)
    if command[0] == "ffmpeg":
        Path(command[-1]).write_bytes(b"encoded-video")
        return subprocess.CompletedProcess(command, 0, "", "")
    output = Path(command[-1])
    is_wam = output.name == review.WAM_PREDICTION_OUTPUT_NAME
    duration = "0.2" if is_wam else "2.0"
    metadata = {
        "streams": [
            {
                "codec_name": "h264",
                "width": 640,
                "height": 480,
                "nb_read_frames": (
                    "48" if is_wam else str(review_frame_count)
                ),
                "duration": duration,
            }
        ],
        "format": {"duration": duration},
    }
    return subprocess.CompletedProcess(command, 0, json.dumps(metadata), "")


def test_review_builder_returns_nonzero_when_trace_has_no_clips(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"

    assert review.main([str(episode_dir)]) == 1

    validation = json.loads((episode_dir / review.VALIDATION_NAME).read_text())
    assert validation["status"] == "blocked"
    assert "separate_wam_prediction_review_not_passed" in validation["blockers"]
    wam_validation = json.loads(
        (episode_dir / review.WAM_PREDICTION_VALIDATION_NAME).read_text()
    )
    assert wam_validation["blockers"] == ["closed_loop_trace_missing"]
    assert not (episode_dir / review.OUTPUT_NAME).exists()


def test_review_builder_rejects_duplicate_clip_and_noncontiguous_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    clip = episode_dir / "step.mp4"
    clip.write_bytes(b"clip")
    _write_trace(
        episode_dir,
        [
            {"step_index": 1, "wam_generated_video": str(clip)},
            {"step_index": 3, "wam_generated_video": str(clip)},
        ],
    )
    monkeypatch.setattr(
        review,
        "_run",
        lambda _command: (_ for _ in ()).throw(AssertionError("ffmpeg must not run")),
    )

    validation = review.build_wam_prediction_review(episode_dir)

    assert validation["trace_step_count"] == 2
    assert validation["ordered_clip_count"] == 1
    assert "closed_loop_episode_step_order_not_contiguous" in validation["blockers"]
    assert "closed_loop_step_video_path_duplicate:2" in validation["blockers"]


def test_review_builder_trims_predictions_to_executed_controller_prefixes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    clips = [episode_dir / "step-1.mp4", episode_dir / "step-2.mp4"]
    for clip in clips:
        clip.write_bytes(b"clip")
    _write_trace(
        episode_dir,
        [
            {"step_index": index, "wam_generated_video": str(clip)}
            for index, clip in enumerate(clips, start=1)
        ],
    )
    _write_isaac_execution_evidence(episode_dir, step_count=2)
    calls: list[list[str]] = []

    def fake_run(command):  # noqa: ANN001
        command = list(command)
        calls.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"reencoded-video")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 640,
                            "height": 480,
                            "nb_read_frames": "3",
                            "duration": "0.2",
                        }
                    ],
                    "format": {"duration": "0.2"},
                }
            ),
            "",
        )

    monkeypatch.setattr(review, "_run", fake_run)

    validation = review.build_wam_prediction_review(episode_dir)

    assert validation["status"] == "passed"
    assert validation["concat_mode"] == "executed_control_prefix_reencode"
    assert validation["trace_step_count"] == 2
    assert validation["ordered_clip_count"] == 2
    assert validation["ordered_step_indices"] == [1, 2]
    assert validation["episode_order_verified"] is True
    assert validation["executed_prefix_duration_seconds_by_step"] == [0.1, 0.1]
    assert validation["expected_executed_timeline_duration_seconds"] == 0.2
    assert validation["overlapping_unexecuted_prediction_tails_excluded"] is True
    encode_command = next(command for command in calls if command[0] == "ffmpeg")
    filters = encode_command[encode_command.index("-filter_complex") + 1]
    assert filters.count("trim=duration=0.100000000") == 2
    assert f"fps={review.WAM_PREDICTION_REVIEW_FPS}" in filters


def test_review_builder_returns_nonzero_when_executed_prefix_encode_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    clip = episode_dir / "step.mp4"
    clip.write_bytes(b"clip")
    _write_trace(episode_dir, [{"step_index": 1, "wam_generated_video": str(clip)}])
    _write_isaac_execution_evidence(episode_dir, step_count=1)
    monkeypatch.setattr(
        review,
        "_run",
        lambda command: subprocess.CompletedProcess(list(command), 8, "", "encode failed"),
    )

    validation = review.build_wam_prediction_review(episode_dir)

    assert validation["status"] == "blocked"
    assert validation["blockers"] == ["ffmpeg_executed_prefix_concat_failed"]
    assert not (episode_dir / review.WAM_PREDICTION_OUTPUT_NAME).exists()


def test_review_builder_returns_nonzero_when_ffprobe_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    clip = episode_dir / "step.mp4"
    clip.write_bytes(b"clip")
    _write_trace(episode_dir, [{"step_index": 1, "wam_generated_video": str(clip)}])
    _write_isaac_execution_evidence(episode_dir, step_count=1)

    def fake_run(command):  # noqa: ANN001
        command = list(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"video")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 9, "{}", "probe failed")

    monkeypatch.setattr(review, "_run", fake_run)

    validation = review.build_wam_prediction_review(episode_dir)

    assert validation["status"] == "blocked"
    assert "ffprobe_failed:9" in validation["blockers"]
    assert not (episode_dir / review.WAM_PREDICTION_OUTPUT_NAME).exists()


def test_final_review_uses_only_hash_bound_same_session_isaac_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode_dir = tmp_path / "episode_001"
    episode_dir.mkdir()
    clips = [episode_dir / "wam-step-1.mp4", episode_dir / "wam-step-2.mp4"]
    for clip in clips:
        clip.write_bytes(b"wam-prediction")
    _write_trace(
        episode_dir,
        [
            {"step_index": index, "wam_generated_video": str(clip)}
            for index, clip in enumerate(clips, start=1)
        ],
    )
    _write_isaac_execution_evidence(episode_dir, step_count=2)
    calls: list[list[str]] = []

    def fake_run(command):  # noqa: ANN001
        calls.append(list(command))
        return _successful_media_run(command, review_frame_count=3)

    monkeypatch.setattr(review, "_run", fake_run)

    result = review.build_episode_review(episode_dir)

    assert result["status"] == "passed"
    assert result["review_source"] == "persistent_same_session_isaac_execution_frames"
    assert result["execution_truth"] is True
    assert result["same_session_isaac_frames"] is True
    assert result["ordered_step_indices"] == [1, 2]
    assert result["frame_count"] == 3
    assert result["ordered_review_frame_indices"] == [0, 1, 2]
    assert result["ordered_review_control_frame_indices"] == [0, 5, 10]
    assert result["terminal_execution_frame_indices"] == [2]
    assert result["width"] == 640
    assert result["primary_camera_role"] == "robot_pov"
    assert result["overview_excluded_from_primary_review"] is True
    assert result["claim_boundary"]["oscar_wam_prediction_pixels_excluded_from_final_review"]
    assert Path(result["path"]).name == review.OUTPUT_NAME
    assert result["wam_prediction_review"]["review_source"] == (
        "oscar_wam_predicted_rollout_clips"
    )
    assert Path(result["wam_prediction_review"]["path"]).name == (
        review.WAM_PREDICTION_OUTPUT_NAME
    )
    assert set(result["isaac_role_videos"]) == set(review.ISAAC_CAMERA_ROLES)
    primary_command = next(
        command for command in calls if Path(command[-1]).name == review.OUTPUT_NAME
    )
    assert str(episode_dir / review.ISAAC_ROLE_OUTPUT_NAMES["robot_pov"]) in primary_command
    assert str(episode_dir / review.ISAAC_ROLE_OUTPUT_NAMES["overview"]) not in primary_command
    assert all(str(clip) not in primary_command for clip in clips)


def test_final_review_fails_closed_when_isaac_frame_hash_binding_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode_dir = tmp_path / "episode_001"
    episode_dir.mkdir()
    clip = episode_dir / "wam-step-1.mp4"
    clip.write_bytes(b"wam-prediction")
    _write_trace(episode_dir, [{"step_index": 1, "wam_generated_video": str(clip)}])
    state_dir = _write_isaac_execution_evidence(episode_dir, step_count=1)
    (state_dir / "frames" / "overview_0001.png").write_bytes(b"tampered-after-binding")
    monkeypatch.setattr(review, "_run", _successful_media_run)

    result = review.build_episode_review(episode_dir)

    assert result["status"] == "blocked"
    assert "same_session_isaac_frame_binding_invalid:overview_0001.png" in result[
        "blockers"
    ]
    assert result["wam_prediction_review"]["status"] == "passed"
    assert (episode_dir / review.WAM_PREDICTION_OUTPUT_NAME).is_file()
    assert not (episode_dir / review.OUTPUT_NAME).exists()
