"""Contract tests for offline kinematic replay rendering.

The replay lane is what makes fast live episodes and full-rate video
compatible: run the episode with per-query rendering, then re-render offline
by scrubbing the retained step trace through a renderer with no physics,
policy, or server in the loop.  These tests pin the part most likely to be
silently wrong -- frame/state alignment -- plus provenance labeling and the
parity comparator the paid canary will use.
"""

from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_dataset_capture import DatasetCaptureRecorder
from blueprint_pipeline.adp009d_episode_step_trace import build_step_trace
from blueprint_pipeline.episode_replay_render import (
    REPLAY_RENDER_SCHEMA_VERSION,
    ReplayRenderError,
    replay_render_episode,
    replay_states,
)

CONTROL_HZ = 15
HORIZON = 4
VIEWS = (
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
)


def _trace(policy_steps: int = 4, settle_steps: int = 2) -> dict:
    total = policy_steps + settle_steps
    joint_trace = [[0.1 * step] * 7 for step in range(total + 1)]
    commanded = []
    for step in range(policy_steps):
        commanded.append(
            {
                "joint_position_target_rad": joint_trace[step + 1],
                "joint_velocity_command_rad_s": [1.5] * 7,
                "source_arm_command": [1.5] * 7,
                "source_action_space": "joint_velocity",
                "clipped_droid_action": [1.5] * 7 + [0.0],
                "observed_before_rad": joint_trace[step],
                "isaac_action": joint_trace[step + 1] + [0.0],
            }
        )
    object_samples = [
        {
            "step_index": index,
            "can_pose_world": [float(index), 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            "gripper_width_m": 0.08 - 0.001 * index,
            "grasp_frame_position_world_m": [0.4, 0.0, 0.6],
        }
        for index in range(total + 1)
    ]
    return build_step_trace(
        joint_trace=joint_trace,
        commanded_actions=commanded,
        object_samples=object_samples,
        settle_isaac_action=[0.0] * 7 + [1.0],
        open_loop_horizon=HORIZON,
        control_hz=CONTROL_HZ,
        joint_limits=[[-2.9, 2.9]] * 7,
    )


def test_replay_states_align_frames_to_pre_step_observations() -> None:
    trace = _trace()
    states = replay_states(trace)

    # One state per control step plus the terminal observation.
    assert len(states) == trace["total_steps"] + 1
    first = states[0]
    assert first["frame_index"] == 0
    assert first["joint_position_rad"] == [0.0] * 7
    # Frame 0 shows the world before step 0: the initial object sample.
    assert first["object_sample"]["step_index"] == 0
    assert first["gripper_width_m"] == pytest.approx(0.08)

    second = states[1]
    assert second["joint_position_rad"] == pytest.approx([0.1] * 7)
    # Before step 1 the world is the post-step-0 sample.
    assert second["object_sample"]["step_index"] == 1

    terminal = states[-1]
    assert terminal["kind"] == "terminal-observation"
    assert terminal["joint_position_rad"] == pytest.approx(
        [0.1 * trace["total_steps"]] * 7
    )
    assert terminal["object_sample"]["step_index"] == trace["total_steps"]
    assert terminal["sim_time_s"] == pytest.approx(trace["total_steps"] / CONTROL_HZ)


class _Writer:
    """Renders a deterministic frame derived from the written joint state."""

    def __init__(self):
        self.written: list[dict] = []
        self.render_calls = 0

    def write_step_state(self, state):
        self.written.append(dict(state))

    def render(self):
        self.render_calls += 1

    def read_views(self):
        level = int(round(self.written[-1]["joint_position_rad"][0] * 100)) % 256
        frame = np.full((32, 64, 3), level, dtype=np.uint8)
        return {view: frame for view in VIEWS}


def test_replay_render_episode_records_every_frame_with_provenance(tmp_path) -> None:
    trace = _trace()
    writer = _Writer()
    recorder = DatasetCaptureRecorder(
        output_dir=tmp_path,
        episode_id="episode-replay",
        view_keys=VIEWS,
        source="kinematic_replay",
        derived_from_step_trace_digest=trace["step_trace_digest"],
    )

    report = replay_render_episode(
        step_trace=trace, state_writer=writer, recorder=recorder
    )

    assert report["schema_version"] == REPLAY_RENDER_SCHEMA_VERSION
    assert report["frame_count"] == trace["total_steps"]
    assert report["derived_from_step_trace_digest"] == trace["step_trace_digest"]
    assert writer.render_calls == trace["total_steps"] + 1
    # State written before every render: joints of the pre-step observation.
    assert writer.written[0]["joint_position_rad"] == [0.0] * 7

    capture = report["capture"]
    assert capture["source"] == "kinematic_replay"
    assert capture["derived_from_step_trace_digest"] == trace["step_trace_digest"]
    assert capture["frame_count"] == trace["total_steps"]
    assert capture["terminal_frame_included"] is True
    for stream in capture["streams"].values():
        assert stream["video"]["decoded_frame_count"] == trace["total_steps"] + 1


def test_live_capture_and_replay_are_labeled_distinctly(tmp_path) -> None:
    live = DatasetCaptureRecorder(
        output_dir=tmp_path, episode_id="episode-live", view_keys=VIEWS
    )
    live.record_step(
        step_index=0,
        views={view: np.zeros((8, 8, 3), dtype=np.uint8) for view in VIEWS},
    )
    record = live.finalize(terminal_views=None)

    assert record["source"] == "live_capture"
    assert record["derived_from_step_trace_digest"] is None


def test_replay_refuses_a_recorder_bound_to_a_different_trace(tmp_path) -> None:
    trace = _trace()
    recorder = DatasetCaptureRecorder(
        output_dir=tmp_path,
        episode_id="episode-mismatch",
        view_keys=VIEWS,
        source="kinematic_replay",
        derived_from_step_trace_digest="sha256:" + "0" * 64,
    )

    with pytest.raises(ReplayRenderError, match="digest"):
        replay_render_episode(
            step_trace=trace, state_writer=_Writer(), recorder=recorder
        )


def test_replay_refuses_a_live_labeled_recorder(tmp_path) -> None:
    """Derived renders must never impersonate live capture."""

    trace = _trace()
    recorder = DatasetCaptureRecorder(
        output_dir=tmp_path, episode_id="episode-live-label", view_keys=VIEWS
    )

    with pytest.raises(ReplayRenderError, match="source"):
        replay_render_episode(
            step_trace=trace, state_writer=_Writer(), recorder=recorder
        )


def test_parity_comparator_measures_pixel_deltas_between_captures(tmp_path) -> None:
    from blueprint_pipeline.episode_replay_render import compare_capture_streams

    def _capture(name: str, offset: int) -> dict:
        recorder = DatasetCaptureRecorder(
            output_dir=tmp_path, episode_id=name, view_keys=VIEWS
        )
        for step in range(3):
            frame = np.full((16, 16, 3), 40 * step + offset, dtype=np.uint8)
            recorder.record_step(
                step_index=step, views={view: frame for view in VIEWS}
            )
        return recorder.finalize(terminal_views=None)

    base = _capture("episode-a", offset=0)
    shifted = _capture("episode-b", offset=6)

    report = compare_capture_streams(
        base, shifted, media_root_a=tmp_path, media_root_b=tmp_path
    )

    assert sorted(report["streams"]) == [
        "exterior_image_1_left",
        "wrist_image_left",
    ]
    for stream in report["streams"].values():
        assert stream["frame_count"] == 3
        # H.264 is lossy, so the measured delta hovers around the injected 6.
        assert 3.0 < stream["mean_abs_delta"] < 9.0
    assert report["frame_counts_match"] is True


def test_generalized_camera_streams_are_accepted(tmp_path) -> None:
    """New scenes bring new cameras; the recorder accepts any valid stream id."""

    recorder = DatasetCaptureRecorder(
        output_dir=tmp_path,
        episode_id="episode-head",
        view_keys=("observation/head_camera_left",),
    )
    recorder.record_step(
        step_index=0,
        views={
            "observation/head_camera_left": np.zeros((8, 8, 3), dtype=np.uint8)
        },
    )
    record = recorder.finalize(terminal_views=None)

    assert sorted(record["streams"]) == ["head_camera_left"]


def test_replay_states_carry_full_joint_vectors_when_traced() -> None:
    inputs_trace = _trace()
    rows = [dict(row) for row in inputs_trace["rows"]]
    for index, row in enumerate(rows):
        row["observation_full_joint_position_rad"] = [0.01 * index] * 13
    full_trace = {
        **inputs_trace,
        "rows": rows,
        "final_full_joint_position_rad": [0.99] * 13,
    }

    states = replay_states(full_trace)

    assert states[0]["full_joint_position_rad"] == [0.0] * 13
    assert states[-1]["full_joint_position_rad"] == [0.99] * 13
