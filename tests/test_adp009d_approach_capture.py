from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_approach_capture import (
    APPROACH_CAPTURE_FRAME_BASE,
    APPROACH_STANDOFF_HEIGHTS_M,
    BLOCKER_APPROACH_IK_FAILED,
    BLOCKER_WRIST_NEVER_SAW_OBJECT,
    CAN_AXIS_XY_M,
    SUPPORT_HEIGHT_M,
    approach_waypoints_world,
    pose_world_to_base,
    summarize_wrist_approach_capture,
)


def test_waypoints_descend_over_the_can_axis_and_clear_its_top() -> None:
    waypoints = approach_waypoints_world()

    assert len(waypoints) == len(APPROACH_STANDOFF_HEIGHTS_M)
    heights = [w["position_world_m"][2] for w in waypoints]
    assert heights == sorted(heights, reverse=True), "waypoints must descend"
    for index, waypoint in enumerate(waypoints):
        x, y, z = waypoint["position_world_m"]
        assert (x, y) == CAN_AXIS_XY_M
        # The observed can top is 0.169 m above support; every waypoint clears it.
        assert z - SUPPORT_HEIGHT_M > 0.169
        assert waypoint["capture_frame_index"] == APPROACH_CAPTURE_FRAME_BASE + index
    # Frame indices must not collide with the 40-frame hold capture.
    assert min(w["capture_frame_index"] for w in waypoints) > 40


def test_world_to_base_conversion_matches_a_rotated_translated_base() -> None:
    """A pose expressed in the base frame must round-trip through the base pose."""

    # Base yawed 90 degrees about z at (1, 2, 0): quaternion (w, x, y, z).
    half = np.sqrt(0.5)
    base_position = [1.0, 2.0, 0.0]
    base_quaternion = [half, 0.0, 0.0, half]
    # A point 1 m along world +x from the base should read as 1 m along base -y.
    position_base, quaternion_base = pose_world_to_base(
        position_world=[2.0, 2.0, 0.0],
        quaternion_world_wxyz=[1.0, 0.0, 0.0, 0.0],
        base_position_world=base_position,
        base_quaternion_world_wxyz=base_quaternion,
    )
    assert position_base[0] == pytest.approx(0.0, abs=1e-9)
    assert position_base[1] == pytest.approx(-1.0, abs=1e-9)
    assert position_base[2] == pytest.approx(0.0, abs=1e-9)
    # Orientation is the base rotation inverted.
    assert quaternion_base[0] == pytest.approx(half, abs=1e-9)
    assert quaternion_base[3] == pytest.approx(-half, abs=1e-9)


def _wrist_frame(frame_index: int, can_pixels: int) -> dict:
    labels = {"0": {"class": "BACKGROUND"}, "2": {"class": "robot"}}
    counts = {"0": 900000, "2": 21600}
    if can_pixels:
        labels["3"] = {"class": "approved_can"}
        counts["3"] = can_pixels
    return {
        "camera_id": "wrist_camera",
        "frame_index": frame_index,
        "semantic_segmentation": {
            "id_to_labels": {"idToLabels": labels},
            "pixel_counts_by_id": counts,
        },
    }


def test_wrist_gate_passes_once_the_object_is_substantially_visible() -> None:
    report = summarize_wrist_approach_capture(
        captured_frames=[
            _wrist_frame(100, 0),
            _wrist_frame(101, 40),
            _wrist_frame(102, 5200),
        ]
    )

    assert report["status"] == "observed"
    assert report["blockers"] == []
    assert report["max_approved_task_object_pixel_count"] == 5200
    assert report["candidate_policy_queried"] is False


def test_wrist_gate_blocks_when_object_never_appears_or_ik_fails() -> None:
    never = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 0), _wrist_frame(101, 12)]
    )
    assert BLOCKER_WRIST_NEVER_SAW_OBJECT in never["blockers"]

    failed = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 5200)], ik_succeeded=False
    )
    assert BLOCKER_APPROACH_IK_FAILED in failed["blockers"]
    assert failed["status"] == "blocked"


def test_external_camera_frames_never_satisfy_the_wrist_gate() -> None:
    """Only the wrist camera can establish wrist observability."""

    external = _wrist_frame(100, 5200)
    external["camera_id"] = "external_camera"

    report = summarize_wrist_approach_capture(captured_frames=[external])
    assert BLOCKER_WRIST_NEVER_SAW_OBJECT in report["blockers"]
    assert report["wrist_frames"] == []


def test_standalone_digest_matches_the_repository_contract() -> None:
    """The bundled copy must digest identically to the package contract."""

    from blueprint_pipeline.adp009d_approach_capture import canonical_digest as bundled
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    payload = {"b": [1, 2, {"c": "é"}], "a": True, "d": None}
    assert bundled(payload) == canonical_digest(payload)
    assert bundled(payload, digest_field="a") == canonical_digest(
        payload, digest_field="a"
    )


def test_runtime_imports_helper_in_both_layouts() -> None:
    """The runtime resolves the helper as a package member and would as a script."""

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    assert runtime.approach_waypoints_world() == approach_waypoints_world()
    source = __import__("pathlib").Path(runtime.__file__).read_text(encoding="utf-8")
    # Both import layouts must remain present: the bundle is a flat directory.
    assert "from adp009d_approach_capture import" in source
    assert "from .adp009d_approach_capture import" in source


def test_runtime_uses_the_arena_pinned_isaac_lab_jacobian_api() -> None:
    """Isaac Lab e57379c exposes jacobians on root_view, not on ArticulationData.

    A live run against the pinned revision failed with
    ``'ArticulationData' object has no attribute 'body_link_jacobian_w'`` because
    that accessor only exists on newer revisions.  Pin the pinned-revision API so
    a future edit cannot silently reintroduce it.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "robot.root_view.get_jacobians()" in source
    assert "robot.data.body_pose_w" in source
    assert "robot.data.root_pose_w" in source
    # Newer-revision accessors that do not exist at the pinned revision.
    assert "body_link_jacobian_w" not in source
    assert "body_link_pose_w" not in source
    # Fixed-base articulations drop the root row from the jacobian stack.
    assert "robot.is_fixed_base" in source


def test_wrist_gate_blocks_when_the_approach_moved_the_object() -> None:
    """The approach must observe the object, never move it."""

    from blueprint_pipeline.adp009d_approach_capture import (
        APPROACH_MAX_OBJECT_DISPLACEMENT_M,
        BLOCKER_APPROACH_DISTURBED_OBJECT,
    )

    disturbed = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 5200)],
        object_displacement_m=3.418578212,
    )
    assert BLOCKER_APPROACH_DISTURBED_OBJECT in disturbed["blockers"]
    assert disturbed["status"] == "blocked"

    settled = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 5200)],
        object_displacement_m=APPROACH_MAX_OBJECT_DISPLACEMENT_M / 2,
    )
    assert settled["blockers"] == []


def test_canonical_hold_is_judged_before_the_approach_runs() -> None:
    """The canonical hold must not be scored on motion it never contained.

    A live run evaluated hold stability after the approach and reported the can
    displaced by 3.42 m, which described the approach, not the hold.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    stability = source.index("_assert_canonical_object_stability(")
    approach = source.index("--- preregistered wrist approach")
    assert stability < approach, "hold stability must be evaluated before the approach"
    # The servo must clamp per-step joint motion and abort on object disturbance.
    assert "APPROACH_MAX_JOINT_STEP_RAD," in source
    assert "approach_aborted = True" in source


def _wrist_frame_at(frame_index: int, can_pixels: int, position) -> dict:
    frame = _wrist_frame(frame_index, can_pixels)
    frame["position_world_m"] = list(position)
    return frame


def test_stale_wrist_pose_is_blocked_when_the_arm_moved() -> None:
    """A hand-mounted camera whose recorded pose never changes is mis-registered.

    A live run captured a visibly changing wrist view while every recorded wrist
    pose stayed byte-identical.  Composing an appearance layer against that pose
    would silently mis-register the entire wrist observation.
    """

    from blueprint_pipeline.adp009d_approach_capture import BLOCKER_WRIST_POSE_STALE

    stale = summarize_wrist_approach_capture(
        captured_frames=[
            _wrist_frame_at(100, 5200, (3.437, -3.096, 0.737)),
            _wrist_frame_at(101, 5200, (3.437, -3.096, 0.737)),
            _wrist_frame_at(102, 5200, (3.437, -3.096, 0.737)),
        ]
    )
    assert BLOCKER_WRIST_POSE_STALE in stale["blockers"]
    assert stale["wrist_pose_travel_m"] == pytest.approx(0.0, abs=1e-12)

    moved = summarize_wrist_approach_capture(
        captured_frames=[
            _wrist_frame_at(100, 5200, (3.437, -3.096, 0.737)),
            _wrist_frame_at(101, 5200, (3.450, -3.150, 0.700)),
            _wrist_frame_at(102, 5200, (3.468, -3.250, 0.660)),
        ]
    )
    assert BLOCKER_WRIST_POSE_STALE not in moved["blockers"]
    assert moved["wrist_pose_travel_m"] > 0.1

    # A deliberately stationary arm must not trip the gate.
    stationary = summarize_wrist_approach_capture(
        captured_frames=[
            _wrist_frame_at(100, 5200, (3.437, -3.096, 0.737)),
            _wrist_frame_at(101, 5200, (3.437, -3.096, 0.737)),
        ],
        arm_moved=False,
    )
    assert BLOCKER_WRIST_POSE_STALE not in stationary["blockers"]


def test_usd_transform_separates_a_stale_buffer_from_a_detached_prim() -> None:
    """The two causes of a frozen wrist pose need opposite repairs.

    Either the sensor pose buffer lags while the prim does follow the hand, or
    the prim is not parented to the hand at all.  Only the stage transform for
    the same prim distinguishes them.
    """

    from blueprint_pipeline.adp009d_approach_capture import (
        WRIST_POSE_CAUSE_HEALTHY,
        WRIST_POSE_CAUSE_PRIM_DETACHED,
        WRIST_POSE_CAUSE_STALE_BUFFER,
        WRIST_POSE_CAUSE_UNDETERMINED,
        classify_wrist_pose_discrepancy,
    )

    frozen = [(3.437, -3.096, 0.737)] * 3
    moved = [(3.437, -3.096, 0.737), (3.450, -3.150, 0.700), (3.468, -3.250, 0.660)]

    # Stage says the prim moved, sensor reported a constant pose -> stale buffer.
    stale = classify_wrist_pose_discrepancy(
        reported_positions=frozen, usd_positions=moved
    )
    assert stale["cause"] == WRIST_POSE_CAUSE_STALE_BUFFER
    assert stale["usd_pose_travel_m"] > 0.1
    assert stale["reported_pose_travel_m"] == pytest.approx(0.0, abs=1e-12)

    # Stage agrees nothing moved -> the camera is not attached to the hand.
    detached = classify_wrist_pose_discrepancy(
        reported_positions=frozen, usd_positions=frozen
    )
    assert detached["cause"] == WRIST_POSE_CAUSE_PRIM_DETACHED

    # Both move -> healthy.
    healthy = classify_wrist_pose_discrepancy(
        reported_positions=moved, usd_positions=moved
    )
    assert healthy["cause"] == WRIST_POSE_CAUSE_HEALTHY

    # No usable stage samples -> refuse to guess.
    for unusable in ([], [[]], [[], []], [(1.0, 2.0, 3.0)]):
        undetermined = classify_wrist_pose_discrepancy(
            reported_positions=frozen, usd_positions=unusable
        )
        assert undetermined["cause"] == WRIST_POSE_CAUSE_UNDETERMINED


def test_summary_carries_the_pose_cause_through_from_frame_diagnostics() -> None:
    """The classification must reach the report without a separate call."""

    from blueprint_pipeline.adp009d_approach_capture import (
        BLOCKER_WRIST_POSE_STALE,
        WRIST_POSE_CAUSE_STALE_BUFFER,
    )

    frames = []
    for index, usd_position in enumerate(
        [(3.437, -3.096, 0.737), (3.450, -3.150, 0.700), (3.468, -3.250, 0.660)]
    ):
        frame = _wrist_frame_at(100 + index, 5200, (3.437, -3.096, 0.737))
        frame["prim_diagnostics"] = {
            "resolved_prim_path": "/World/envs/env_0/Robot/wrist_cam",
            "prim_exists": True,
            "usd_world_translation_m": list(usd_position),
        }
        frames.append(frame)

    report = summarize_wrist_approach_capture(captured_frames=frames)
    assert BLOCKER_WRIST_POSE_STALE in report["blockers"]
    assert report["wrist_pose_discrepancy"]["cause"] == WRIST_POSE_CAUSE_STALE_BUFFER
    # The digest must cover the new field.
    from blueprint_pipeline.adp009d_approach_capture import canonical_digest

    assert report["report_digest"] == canonical_digest(
        report, digest_field="report_digest"
    )


def test_runtime_records_stage_transform_on_every_capture() -> None:
    """The diagnostic must be unconditional, and must never fail a capture."""

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert '"prim_diagnostics": _camera_prim_diagnostics(camera)' in source
    assert "ComputeLocalToWorldTransform" in source
    # Established stage accessor for this runtime.
    assert "omni.usd.get_context().get_stage()" in source
    # Collected for every capture, so a healthy run proves the diagnostic works.
    diagnostics_body = source[source.index("def _camera_prim_diagnostics(") :]
    diagnostics_body = diagnostics_body[: diagnostics_body.index("def _save_camera(")]
    assert "except Exception" in diagnostics_body
    raising = [
        line for line in diagnostics_body.splitlines() if line.strip().startswith("raise")
    ]
    assert raising == [], f"diagnostics must not raise: {raising}"


def test_arrival_gate_separates_never_arrived_from_arrived_and_saw_nothing() -> None:
    """"IK succeeded" only meant "no exception"; arriving is a separate fact.

    The servo clamps joint motion over a fixed step budget, so it can run
    cleanly to the end and still stop far short of the waypoint.  A wrist that
    never got there cannot be judged on what it did not see.
    """

    from blueprint_pipeline.adp009d_approach_capture import (
        APPROACH_WAYPOINT_TOLERANCE_M,
        BLOCKER_APPROACH_DID_NOT_REACH,
    )

    def arrival(index: int, error: float) -> dict:
        return {
            "waypoint_index": index,
            "target_position_world_m": [3.468, -3.310, 0.866],
            "achieved_position_world_m": [3.468, -3.310, 0.866 + error],
            "position_error_m": error,
        }

    # Arrived at every waypoint but the object never appeared: a real negative.
    arrived = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 0)],
        waypoint_arrivals=[arrival(0, 0.004), arrival(1, 0.011)],
    )
    assert BLOCKER_APPROACH_DID_NOT_REACH not in arrived["blockers"]
    assert BLOCKER_WRIST_NEVER_SAW_OBJECT in arrived["blockers"]
    assert arrived["worst_waypoint_position_error_m"] == pytest.approx(0.011)

    # Never got close: the wrist result is uninterpretable, so say so.
    short = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 0)],
        waypoint_arrivals=[arrival(0, 0.004), arrival(1, 0.42)],
    )
    assert BLOCKER_APPROACH_DID_NOT_REACH in short["blockers"]
    assert short["status"] == "blocked"

    # Exactly at tolerance still counts as arrived.
    boundary = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 5200)],
        waypoint_arrivals=[arrival(0, APPROACH_WAYPOINT_TOLERANCE_M)],
    )
    assert BLOCKER_APPROACH_DID_NOT_REACH not in boundary["blockers"]

    # No arrival evidence at all must not silently pass the gate.
    silent = summarize_wrist_approach_capture(captured_frames=[_wrist_frame(100, 5200)])
    assert silent["worst_waypoint_position_error_m"] is None
    assert silent["waypoint_arrivals"] == []


def test_runtime_records_the_achieved_end_effector_pose_per_waypoint() -> None:
    """The arrival evidence must come from the simulator, not be inferred."""

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "approach_arrivals.append(" in source
    assert '"achieved_position_world_m"' in source
    assert "waypoint_arrivals=approach_arrivals," in source
    # Recorded before the approach captures, so an aborted waypoint still
    # reports where it got to.  Anchor inside the approach block: the hold phase
    # captures from the same camera pair earlier in the file.
    approach = source[source.index("--- preregistered wrist approach") :]
    assert approach.index("approach_arrivals.append(") < approach.index(
        'for camera_name in ("external_camera", "wrist_camera"):'
    )
