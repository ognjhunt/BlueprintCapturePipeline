"""Re-render an episode offline by kinematically scrubbing its step trace.

This is what makes fast live episodes and full-rate video compatible instead
of a tradeoff.  The live episode runs with per-query rendering -- the ~88%
saving -- while retaining the per-step trace.  Afterwards, this module drives
a renderer through the same sealed states with no physics, no policy, and no
server in the loop: write the state a frame should show, render, record.  The
renderer runs flat out, batches across episodes on one warm process, and can
render at any resolution, including diagnostic 1280x720, without touching the
paid episode wall clock.

Honesty is structural, not narrative: the recorder must be labeled
``kinematic_replay`` and bound to the exact step-trace digest it derives
from, so a derived render can never impersonate live capture, and the
query-cadence policy-input PNGs remain the authoritative record of what the
policy consumed.

The state/frame alignment matches the capture contract -- frame ``i`` is the
observation before control step ``i`` -- which makes replay frames directly
comparable to a live dataset capture of the same episode.  That comparison is
the paid parity canary, and ``compare_capture_streams`` is the comparator it
uses.

Everything here is orchestration over injected seams; the Isaac-side state
writer lives with the other simulator adapters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

try:  # flat provider-bundle layout
    from adp009d_dataset_capture import CAPTURE_SOURCE_REPLAY
except ModuleNotFoundError:  # repository package
    from .adp009d_dataset_capture import CAPTURE_SOURCE_REPLAY

REPLAY_RENDER_SCHEMA_VERSION = "episode_replay_render.v1"

BLOCKER_TRACE_INVALID = "replay_render_step_trace_invalid"
BLOCKER_RECORDER_SOURCE = "replay_render_recorder_source_not_kinematic_replay"
BLOCKER_RECORDER_DIGEST = "replay_render_recorder_digest_mismatch"
BLOCKER_VIEWS_MISSING = "replay_render_state_writer_view_missing"


class ReplayRenderError(ValueError):
    """Fail-closed replay contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


class ReplayStateWriter(Protocol):
    """The renderer-side seam: set state, render, read the camera views."""

    def write_step_state(self, state: Mapping[str, Any]) -> None:
        """Kinematically apply one frame's state; never step physics."""

    def render(self) -> None:
        """Advance rendering only, refreshing every camera buffer."""

    def read_views(self) -> Mapping[str, Any]:
        """Camera RGB by view key, exactly as the live adapter serves them."""


def replay_states(step_trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-frame kinematic states aligned to the capture contract.

    Frame ``i`` shows the world *before* control step ``i``: the pre-step
    joint observation with the object sample produced by the previous step
    (the initial sample for frame 0).  One terminal state follows the last
    step.  The object sample travels verbatim -- which objects it names is the
    scene's business, and the state writer for that scene interprets it.
    """

    rows = list(step_trace.get("rows") or [])
    initial = step_trace.get("initial_object_sample")
    control_hz = step_trace.get("control_hz")
    if not rows or not isinstance(initial, Mapping) or not control_hz:
        raise ReplayRenderError([BLOCKER_TRACE_INVALID])

    states: list[dict[str, Any]] = []
    previous_sample: Mapping[str, Any] = initial
    for row in rows:
        joints = [float(value) for value in row["observation_joint_position_rad"]]
        if not all(math.isfinite(value) for value in joints):
            raise ReplayRenderError([f"{BLOCKER_TRACE_INVALID}:joints:{row['step_index']}"])
        states.append(
            {
                "kind": "policy-step",
                "frame_index": int(row["step_index"]),
                "sim_time_s": float(row["sim_time_s"]),
                "joint_position_rad": joints,
                "gripper_width_m": (
                    float(previous_sample["gripper_width_m"])
                    if previous_sample.get("gripper_width_m") is not None
                    else None
                ),
                "object_sample": dict(previous_sample),
            }
        )
        full = row.get("observation_full_joint_position_rad")
        if full is not None:
            states[-1]["full_joint_position_rad"] = [float(v) for v in full]
        previous_sample = row["object_sample"]

    last = rows[-1]
    states.append(
        {
            "kind": "terminal-observation",
            "frame_index": int(last["step_index"]) + 1,
            "sim_time_s": (int(last["step_index"]) + 1) / float(control_hz),
            "joint_position_rad": [float(v) for v in last["observed_after_rad"]],
            "gripper_width_m": (
                float(previous_sample["gripper_width_m"])
                if previous_sample.get("gripper_width_m") is not None
                else None
            ),
            "object_sample": dict(previous_sample),
        }
    )
    final_full = step_trace.get("final_full_joint_position_rad")
    if final_full is not None:
        states[-1]["full_joint_position_rad"] = [float(v) for v in final_full]
    return states


def _views_for(recorder: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    view_keys = tuple(getattr(recorder, "view_keys", ()) or ())
    missing = [view for view in view_keys if view not in source]
    if missing:
        raise ReplayRenderError([f"{BLOCKER_VIEWS_MISSING}:{','.join(missing)}"])
    return {view: source[view] for view in view_keys}


def replay_render_episode(
    *,
    step_trace: Mapping[str, Any],
    state_writer: ReplayStateWriter,
    recorder: Any,
) -> dict[str, Any]:
    """Scrub the trace through the renderer and seal a derived capture.

    The recorder must already be labeled ``kinematic_replay`` and bound to
    this exact trace's digest -- provenance is a construction-time property of
    the artifact, not a caption added afterwards.
    """

    if getattr(recorder, "source", None) != CAPTURE_SOURCE_REPLAY:
        raise ReplayRenderError([BLOCKER_RECORDER_SOURCE])
    trace_digest = str(step_trace.get("step_trace_digest") or "")
    if (
        not trace_digest
        or getattr(recorder, "derived_from_step_trace_digest", None) != trace_digest
    ):
        raise ReplayRenderError([BLOCKER_RECORDER_DIGEST])

    states = replay_states(step_trace)
    for state in states[:-1]:
        state_writer.write_step_state(state)
        state_writer.render()
        recorder.record_step(
            step_index=state["frame_index"],
            views=_views_for(recorder, state_writer.read_views()),
        )
    terminal = states[-1]
    state_writer.write_step_state(terminal)
    state_writer.render()
    capture = recorder.finalize(
        terminal_views=_views_for(recorder, state_writer.read_views())
    )

    return {
        "schema_version": REPLAY_RENDER_SCHEMA_VERSION,
        "derived_from_step_trace_digest": trace_digest,
        "frame_count": len(states) - 1,
        "terminal_frame_included": True,
        "frames_are_derived_renders_not_policy_observations": True,
        "capture": capture,
    }


def compare_capture_streams(
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    *,
    media_root_a: str | Path,
    media_root_b: str | Path,
) -> dict[str, Any]:
    """Measure per-stream pixel deltas between two captures of one episode.

    This is the parity comparator for the replay canary: decode both H.264
    streams and report mean/max absolute per-pixel deltas per stream.  It
    reports; the canary's preregistered threshold decides.
    """

    import cv2
    import numpy as np

    def _frames(record: Mapping[str, Any], root: str | Path, stream_id: str):
        stream = record["streams"][stream_id]
        path = Path(root).expanduser().resolve() / str(
            stream["video"]["relative_path"]
        )
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ReplayRenderError([f"replay_parity_video_unreadable:{path.name}"])
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame.astype(np.float64))
        finally:
            capture.release()
        return frames

    stream_ids_a = set(record_a.get("streams") or {})
    stream_ids_b = set(record_b.get("streams") or {})
    if not stream_ids_a or stream_ids_a != stream_ids_b:
        raise ReplayRenderError(
            [f"replay_parity_stream_sets_differ:{sorted(stream_ids_a)}!={sorted(stream_ids_b)}"]
        )

    streams: dict[str, dict[str, Any]] = {}
    counts_match = True
    for stream_id in sorted(stream_ids_a):
        frames_a = _frames(record_a, media_root_a, stream_id)
        frames_b = _frames(record_b, media_root_b, stream_id)
        compared = min(len(frames_a), len(frames_b))
        if len(frames_a) != len(frames_b):
            counts_match = False
        deltas = [
            float(np.abs(frames_a[index] - frames_b[index]).mean())
            for index in range(compared)
        ]
        streams[stream_id] = {
            "frame_count": compared,
            "frame_count_a": len(frames_a),
            "frame_count_b": len(frames_b),
            "mean_abs_delta": float(np.mean(deltas)) if deltas else None,
            "max_frame_mean_abs_delta": float(np.max(deltas)) if deltas else None,
        }

    return {
        "schema_version": "episode_replay_parity.v1",
        "streams": streams,
        "frame_counts_match": counts_match,
        "claim_scope": (
            "pixel_delta_measurement_only_threshold_decided_by_preregistered_canary"
        ),
    }


__all__ = [
    "REPLAY_RENDER_SCHEMA_VERSION",
    "ReplayRenderError",
    "ReplayStateWriter",
    "compare_capture_streams",
    "replay_render_episode",
    "replay_states",
]
