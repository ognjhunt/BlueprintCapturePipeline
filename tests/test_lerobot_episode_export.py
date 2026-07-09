"""Tests for the LeRobot/GR00T episode export and modality config."""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.lerobot_episode_export import (
    SC3_ACTION_DIM,
    SC3_ACTION_LAYOUT_ID,
    build_lerobot_episode_export,
    build_modality_config,
)
from blueprint_pipeline.scene_placement.robot_profile import UNITREE_G1_PROFILE


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def _control_row(
    attempt_id: str,
    index: int,
    *,
    action: object | None = None,
    with_state: bool = False,
    with_timestamp: bool = False,
) -> dict:
    payload: object
    if action is not None:
        payload = action
    else:
        payload = {
            "delta_position_m": [0.01 * (index + 1), 0.0, 0.0],
            "delta_rotation_axis_angle": [0.0, 0.0, 0.02],
            "gripper": 0.0,
        }
        if with_state:
            payload["base_pose_7d"] = [0.1 * index, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0]
        if with_timestamp:
            payload["sim_time_s"] = 0.1 * index
    return {
        "stream_type": "control_action",
        "attempt_id": attempt_id,
        "action_index": index,
        "action": payload,
        "task_id": "move-tote",
        "scenario_id": "clear-path",
    }


def _attempt(attempt_id: str, *, success: bool = True) -> dict:
    return {
        "attempt_id": attempt_id,
        "episode_id": f"episode-{attempt_id}",
        "scenario_eval_run_id": f"run-{attempt_id}",
        "task_id": "move-tote",
        "scenario_id": "clear-path",
        "success": success,
    }


def _seed_job_dir(
    tmp_path: Path,
    *,
    attempts: list[dict],
    control_rows: list[dict],
) -> Path:
    job_dir = tmp_path / "job"
    _write_jsonl(
        job_dir / "simulator_command_batch_attempt_trace.jsonl", attempts
    )
    _write_jsonl(
        job_dir / "simulator_command_batch_control_stream.jsonl", control_rows
    )
    return job_dir


# ---------------------------------------------------------------------------
# modality.json from RobotProfile
# ---------------------------------------------------------------------------


def test_modality_config_declares_profile_action_slices_from_profile() -> None:
    config = build_modality_config(UNITREE_G1_PROFILE)
    action = config["action"]["unitree_g1_whole_body_arm_hand_chunks_v1"]
    assert action["start"] == 0
    assert action["end"] == 78
    assert action["absolute"] is False
    fields = action["fields"]
    assert fields["base_velocity_xy_yaw"] == {"start": 0, "end": 3}
    assert fields["left_arm_joint_delta"] == {"start": 3, "end": 10}
    assert fields["left_hand_joint_delta"] == {"start": 10, "end": 17}
    assert fields["right_arm_joint_delta"] == {"start": 17, "end": 24}
    assert fields["right_hand_joint_delta"] == {"start": 24, "end": 31}
    assert fields["whole_body_residual_or_policy_latent"] == {
        "start": 31,
        "end": 78,
    }
    assert config["action_dim"] == 78
    assert config["action_layout_id"] == "unitree_g1_whole_body_arm_hand_chunks_v1"
    assert config["action_layout"]["legacy_supported_layouts"] == [
        SC3_ACTION_LAYOUT_ID
    ]
    # humanoid default state layout: base pose (pos + wxyz quat)
    assert config["state"]["base_position_m"] == {"start": 0, "end": 3}
    assert config["state"]["base_orientation_quat_wxyz"] == {"start": 3, "end": 7}
    assert config["state_dim"] == 7
    # video keys come from the profile's camera rigs
    assert (
        config["video"]["head_rgbd"]["original_key"]
        == "observation.images.head_rgbd"
    )
    assert config["robot_id"] == "unitree_g1"


def test_modality_config_carries_claim_boundary_not_execution_proof() -> None:
    config = build_modality_config(UNITREE_G1_PROFILE)
    boundary = config["claim_boundary"]
    assert boundary["episodes_are_simulator_traces_not_physical_robot_data"] is True
    assert boundary["absent_fields_are_omitted_never_zero_filled"] is True


# ---------------------------------------------------------------------------
# per-episode export
# ---------------------------------------------------------------------------


def test_export_writes_episode_rows_meta_and_stats(tmp_path: Path) -> None:
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1")],
        control_rows=[
            _control_row("a1", 0, with_state=True, with_timestamp=True),
            _control_row("a1", 1, with_state=True, with_timestamp=True),
        ],
    )
    manifest = build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )
    assert manifest["status"] == "completed_review_required"
    assert manifest["episode_count"] == 1
    assert manifest["total_frame_count"] == 2
    assert manifest["action_layout_id"] == SC3_ACTION_LAYOUT_ID
    assert manifest["action_dim"] == SC3_ACTION_DIM

    export_root = tmp_path / "out" / "lerobot_episode_export"
    rows = [
        json.loads(line)
        for line in (export_root / "data" / "episode_000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["action"] == [0.01, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0]
    assert rows[0]["action_layout_id"] == SC3_ACTION_LAYOUT_ID
    assert rows[0]["observation.state"][2] == 0.79
    assert rows[1]["frame_index"] == 1
    assert rows[1]["timestamp"] == 0.1
    assert rows[0]["task"] == "move-tote"
    assert rows[0]["observation_source"] == "simulator_trace"
    assert rows[0]["observation_source_is_simulator_trace"] is True
    assert rows[0]["observation_source_is_model_derived"] is False
    assert rows[0]["observation_source_is_raw_capture_evidence"] is False

    episodes = [
        json.loads(line)
        for line in (export_root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episodes[0]["length"] == 2
    assert episodes[0]["state_present"] is True
    assert episodes[0]["timestamps_present"] is True
    assert episodes[0]["observation_source"] == "simulator_trace"
    assert episodes[0]["simulator_trace_frame_count"] == 2
    assert episodes[0]["model_derived_frame_count"] == 0
    # no video is materialized yet, so gr00t_ready must stay False
    assert episodes[0]["gr00t_ready"] is False
    assert episodes[0]["gr00t_ready_missing"] == ["materialized_video"]

    stats = json.loads((export_root / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == SC3_ACTION_DIM
    assert stats["action"]["max"][0] == 0.02
    modality = json.loads((export_root / "meta" / "modality.json").read_text())
    assert modality["robot_id"] == "unitree_g1"
    info = json.loads((export_root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["features"]["action"]["shape"] == [SC3_ACTION_DIM]
    assert info["features"]["observation_source"]["dtype"] == "string"
    assert manifest["observation_source_columns_written"] is True
    assert manifest["simulator_trace_frame_count"] == 2
    assert manifest["model_derived_frame_count"] == 0


def test_export_accepts_profile_whole_body_action_chunk_without_7d_exclusion(
    tmp_path: Path,
) -> None:
    action_chunk = [round((index - 39) / 100.0, 4) for index in range(78)]
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1")],
        control_rows=[
            _control_row(
                "a1",
                0,
                action={
                    "action_chunk": action_chunk,
                    "base_pose_7d": [0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0],
                    "sim_time_s": 0.0,
                },
                with_state=False,
                with_timestamp=False,
            ),
        ],
    )

    manifest = build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )

    assert manifest["status"] == "completed_review_required"
    assert manifest["episode_count"] == 1
    assert manifest["excluded_episode_count"] == 0
    assert manifest["action_layout_id"] == "unitree_g1_whole_body_arm_hand_chunks_v1"
    assert manifest["action_dim"] == 78
    export_root = tmp_path / "out" / "lerobot_episode_export"
    row = json.loads(
        (export_root / "data" / "episode_000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["action"] == action_chunk
    assert row["action_layout_id"] == "unitree_g1_whole_body_arm_hand_chunks_v1"
    stats = json.loads((export_root / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == 78
    info = json.loads((export_root / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [78]
    modality = json.loads((export_root / "meta" / "modality.json").read_text())
    assert "unitree_g1_whole_body_arm_hand_chunks_v1" in modality["action"]


def test_export_marks_episode_gr00t_ready_when_materialized_video_present(
    tmp_path: Path,
) -> None:
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1")],
        control_rows=[
            _control_row("a1", 0, with_state=True, with_timestamp=True),
        ],
    )
    video_path = job_dir / "clip-a1.mp4"
    video_path.write_bytes(b"fake-mp4")

    manifest = build_lerobot_episode_export(
        job_dir=job_dir,
        output_dir=tmp_path / "out",
        robot_id="unitree_g1",
        materialized_video_by_attempt={"a1": {"path": str(video_path), "clip_id": "clip-a1"}},
    )

    assert manifest["status"] == "completed_review_required"
    assert manifest["materialized_video_count"] == 1
    assert manifest["gr00t_ready_episode_count"] == 1
    export_root = tmp_path / "out" / "lerobot_episode_export"
    episodes = [
        json.loads(line)
        for line in (export_root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episodes[0]["video_present"] is True
    assert episodes[0]["gr00t_ready"] is True
    assert episodes[0]["gr00t_ready_missing"] == []
    assert episodes[0]["video_key"] == "observation.images.head_rgbd"
    copied_video = export_root / episodes[0]["video_path"]
    assert copied_video.is_file()
    row = json.loads(
        (export_root / "data" / "episode_000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["observation.images.head_rgbd"] == episodes[0]["video_path"]


def test_export_labels_model_derived_materialized_video_source(
    tmp_path: Path,
) -> None:
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1")],
        control_rows=[
            _control_row("a1", 0, with_state=True, with_timestamp=True),
        ],
    )
    video_path = job_dir / "generated-a1.mp4"
    video_path.write_bytes(b"fake-generated-mp4")

    manifest = build_lerobot_episode_export(
        job_dir=job_dir,
        output_dir=tmp_path / "out",
        robot_id="unitree_g1",
        materialized_video_by_attempt={
            "a1": {
                "path": str(video_path),
                "clip_id": "generated-clip-a1",
                "model_derived": True,
                "observation_source": "model_derived",
                "observation_source_detail": "world_model_generated_support_video",
            }
        },
    )

    assert manifest["status"] == "completed_review_required"
    assert manifest["model_derived_frame_count"] == 1
    assert manifest["raw_capture_frame_count"] == 0
    export_root = tmp_path / "out" / "lerobot_episode_export"
    row = json.loads(
        (export_root / "data" / "episode_000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["observation_source"] == "model_derived"
    assert row["observation_source_detail"] == "world_model_generated_support_video"
    assert row["observation_source_is_model_derived"] is True
    assert row["observation_source_is_raw_capture_evidence"] is False
    episodes = [
        json.loads(line)
        for line in (export_root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episodes[0]["observation_source"] == "model_derived"
    assert episodes[0]["model_derived_frame_count"] == 1


def test_missing_control_stream_blocks_export(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    _write_jsonl(
        job_dir / "simulator_command_batch_attempt_trace.jsonl", [_attempt("a1")]
    )
    manifest = build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )
    assert manifest["status"] == "blocked"
    assert "control_stream_missing" in manifest["blockers"]
    assert manifest["episode_count"] == 0


def test_attempt_without_control_rows_is_excluded_not_padded(tmp_path: Path) -> None:
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1"), _attempt("a2")],
        control_rows=[
            _control_row("a1", 0, with_state=True, with_timestamp=True),
        ],
    )
    manifest = build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )
    assert manifest["episode_count"] == 1
    assert manifest["excluded_episode_count"] == 1
    excluded = manifest["excluded_episodes"][0]
    assert excluded["attempt_id"] == "a2"
    assert "control_rows_missing_for_attempt" in excluded["blockers"]
    export_root = tmp_path / "out" / "lerobot_episode_export"
    assert not (export_root / "data" / "episode_000001.jsonl").exists()


def test_invalid_sc3_action_excludes_episode(tmp_path: Path) -> None:
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1")],
        control_rows=[
            _control_row("a1", 0),
            _control_row("a1", 1, action={"velocity_command": [1.0, 0.0]}),
        ],
    )
    manifest = build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )
    assert manifest["status"] == "blocked"
    assert "no_exportable_episodes" in manifest["blockers"]
    excluded = manifest["excluded_episodes"][0]
    assert excluded["attempt_id"] == "a1"
    assert any(
        blocker.startswith("sc3_7d_action_invalid_at_index:1")
        for blocker in excluded["blockers"]
    )


def test_missing_state_and_timestamps_are_omitted_never_zero_filled(
    tmp_path: Path,
) -> None:
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1")],
        control_rows=[_control_row("a1", 0)],  # no state, no timestamp
    )
    manifest = build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )
    assert manifest["status"] == "completed_review_required"
    export_root = tmp_path / "out" / "lerobot_episode_export"
    row = json.loads(
        (export_root / "data" / "episode_000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert "observation.state" not in row
    assert "timestamp" not in row
    episodes = [
        json.loads(line)
        for line in (export_root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episodes[0]["state_present"] is False
    assert episodes[0]["timestamps_present"] is False
    assert set(episodes[0]["gr00t_ready_missing"]) == {
        "per_step_state",
        "per_step_timestamps",
        "materialized_video",
    }


def test_export_reads_state_and_timestamp_from_control_row_context(
    tmp_path: Path,
) -> None:
    row = _control_row("a1", 0)
    row["base_pose_7d"] = [0.25, -0.1, 0.79, 1.0, 0.0, 0.0, 0.0]
    row["sim_time_s"] = 0.25
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1")],
        control_rows=[row],
    )

    manifest = build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )

    assert manifest["status"] == "completed_review_required"
    export_root = tmp_path / "out" / "lerobot_episode_export"
    exported_row = json.loads(
        (export_root / "data" / "episode_000000.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert exported_row["observation.state"] == [
        0.25,
        -0.1,
        0.79,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert exported_row["timestamp"] == 0.25
    episodes = [
        json.loads(line)
        for line in (export_root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episodes[0]["state_present"] is True
    assert episodes[0]["timestamps_present"] is True
    assert episodes[0]["gr00t_ready_missing"] == ["materialized_video"]


def test_parquet_status_is_honest_about_pyarrow_availability(
    tmp_path: Path,
) -> None:
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[_attempt("a1")],
        control_rows=[_control_row("a1", 0, with_state=True, with_timestamp=True)],
    )
    manifest = build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )
    try:
        import pyarrow  # noqa: F401

        expected = "written"
    except ImportError:
        expected = "blocked_missing_pyarrow"
    assert manifest["parquet_status"] == expected
    parquet_path = (
        tmp_path
        / "out"
        / "lerobot_episode_export"
        / "data"
        / "episode_000000.parquet"
    )
    assert parquet_path.exists() == (expected == "written")


def test_success_label_is_strict_boolean_or_none(tmp_path: Path) -> None:
    attempt = _attempt("a1")
    attempt["success"] = "true"  # mistyped labels must not become verdicts
    job_dir = _seed_job_dir(
        tmp_path,
        attempts=[attempt],
        control_rows=[_control_row("a1", 0)],
    )
    build_lerobot_episode_export(
        job_dir=job_dir, output_dir=tmp_path / "out", robot_id="unitree_g1"
    )
    episodes = [
        json.loads(line)
        for line in (
            tmp_path / "out" / "lerobot_episode_export" / "meta" / "episodes.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episodes[0]["attempt_success_label"] is None
