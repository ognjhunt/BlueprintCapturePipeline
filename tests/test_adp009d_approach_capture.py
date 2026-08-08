from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_approach_capture import (
    APPROACH_CAPTURE_FRAME_BASE,
    APPROACH_STANDOFF_HEIGHTS_M,
    BLOCKER_NO_SAFE_WRIST_OBSERVABLE_EPISODE_START,
    BLOCKER_EPISODE_START_RESTORE_JOINT_MISMATCH,
    BLOCKER_EPISODE_START_RESTORE_OBJECT_MOVED,
    BLOCKER_EPISODE_START_RESTORE_OBJECT_NOT_VISIBLE,
    BLOCKER_APPROACH_IK_FAILED,
    BLOCKER_WRIST_NEVER_SAW_OBJECT,
    CAN_AXIS_XY_M,
    SUPPORT_HEIGHT_M,
    approach_waypoints_world,
    pose_world_to_base,
    select_wrist_observable_episode_start,
    semantic_label_pixel_count,
    semantic_target_observability,
    summarize_wrist_approach_capture,
    validate_wrist_observable_episode_start_restore,
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

    # Base yawed 90 degrees about z at (1, 2, 0): quaternion (x, y, z, w).
    half = np.sqrt(0.5)
    base_position = [1.0, 2.0, 0.0]
    base_quaternion = [0.0, 0.0, half, half]
    # A point 1 m along world +x from the base should read as 1 m along base -y.
    position_base, quaternion_base = pose_world_to_base(
        position_world=[2.0, 2.0, 0.0],
        quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        base_position_world=base_position,
        base_quaternion_world_xyzw=base_quaternion,
    )
    assert position_base[0] == pytest.approx(0.0, abs=1e-9)
    assert position_base[1] == pytest.approx(-1.0, abs=1e-9)
    assert position_base[2] == pytest.approx(0.0, abs=1e-9)
    # Orientation is the base rotation inverted.
    assert quaternion_base[2] == pytest.approx(-half, abs=1e-9)
    assert quaternion_base[3] == pytest.approx(half, abs=1e-9)


def test_sealed_can_axis_maps_forward_of_the_minus_90_degree_robot_base() -> None:
    """Regression for treating IsaacLab xyzw quaternions as wxyz."""

    half = np.sqrt(0.5)
    position_base, _ = pose_world_to_base(
        position_world=[3.4681748, -3.3100837, 0.75],
        quaternion_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        base_position_world=[3.4681748, -2.8100837, 0.2766791],
        base_quaternion_world_xyzw=[0.0, 0.0, -half, half],
    )

    assert position_base == pytest.approx([0.5, 0.0, 0.4733209], abs=1e-7)


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


def test_semantic_pixel_count_uses_only_the_exact_target_class() -> None:
    assert semantic_label_pixel_count(
        id_to_labels={
            "1": {"class": "robot"},
            "2": {"class": "approved_can"},
            "3": "approved_can",
        },
        pixel_counts_by_id={"1": 5000, "2": 120, "3": 80},
        target_label="approved_can",
    ) == 200


def test_semantic_observability_rejects_a_border_sliver() -> None:
    centered = np.zeros((100, 100), dtype=np.int32)
    centered[35:65, 40:60] = 7
    edge = np.zeros((100, 100), dtype=np.int32)
    edge[:30, 40:60] = 7
    labels = {"7": {"class": "approved_can"}}

    centered_receipt = semantic_target_observability(
        semantic_ids=centered,
        id_to_labels=labels,
        target_label="approved_can",
    )
    edge_receipt = semantic_target_observability(
        semantic_ids=edge,
        id_to_labels=labels,
        target_label="approved_can",
    )

    assert centered_receipt["approved_task_object_pixel_fraction"] == 0.06
    assert centered_receipt["approved_task_object_within_frame_margin"] is True
    assert edge_receipt["approved_task_object_pixel_fraction"] == 0.06
    assert edge_receipt["approved_task_object_within_frame_margin"] is False


def test_episode_start_selects_first_visible_pose_inside_canonical_hold() -> None:
    samples = [
        {
            "step": 0,
            "joint_position_rad": [0.0] * 7,
            "object_offset_m": [0.0, 0.0, 0.0],
            "approved_task_object_pixel_count": 0,
            "approved_task_object_pixel_fraction": 0.0,
            "approved_task_object_within_frame_margin": False,
        },
        {
            "step": 1,
            "joint_position_rad": [0.1] * 7,
            "object_offset_m": [0.001, -0.002, 0.003],
            "approved_task_object_pixel_count": 250,
            "approved_task_object_pixel_fraction": 0.03,
            "approved_task_object_within_frame_margin": True,
        },
        {
            "step": 2,
            "joint_position_rad": [0.2] * 7,
            "object_offset_m": [0.0, 0.0, 0.0],
            "approved_task_object_pixel_count": 5000,
            "approved_task_object_pixel_fraction": 0.1,
            "approved_task_object_within_frame_margin": True,
        },
    ]

    receipt = select_wrist_observable_episode_start(samples)

    assert receipt["status"] == "ready"
    assert receipt["selected"]["step"] == 1
    assert receipt["selected"]["joint_position_rad"] == [0.1] * 7
    assert receipt["blockers"] == []
    assert receipt["selection_digest"].startswith("sha256:")


def test_episode_start_rejects_visible_pose_after_object_was_disturbed() -> None:
    receipt = select_wrist_observable_episode_start(
        [
            {
                "step": 0,
                "joint_position_rad": [0.0] * 7,
                "object_offset_m": [0.0, 0.0, 0.006],
                "approved_task_object_pixel_count": 52000,
                "approved_task_object_pixel_fraction": 0.1,
                "approved_task_object_within_frame_margin": True,
            }
        ]
    )

    assert receipt["status"] == "blocked"
    assert receipt["selected"] is None
    assert receipt["blockers"] == [
        BLOCKER_NO_SAFE_WRIST_OBSERVABLE_EPISODE_START
    ]


def test_episode_start_rejects_a_large_but_border_clipped_target() -> None:
    receipt = select_wrist_observable_episode_start(
        [
            {
                "step": 0,
                "joint_position_rad": [0.0] * 7,
                "object_offset_m": [0.0, 0.0, 0.0],
                "approved_task_object_pixel_count": 5000,
                "approved_task_object_pixel_fraction": 0.08,
                "approved_task_object_within_frame_margin": False,
            }
        ]
    )

    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [
        BLOCKER_NO_SAFE_WRIST_OBSERVABLE_EPISODE_START
    ]


def test_episode_start_rejects_v75_scale_border_sliver() -> None:
    receipt = select_wrist_observable_episode_start(
        [
            {
                "step": 11,
                "joint_position_rad": [0.1] * 7,
                "object_offset_m": [0.0, 0.0, 4.9e-6],
                "approved_task_object_pixel_count": 219,
                "approved_task_object_pixel_fraction": 219 / (320 * 180),
                "approved_task_object_within_frame_margin": False,
            }
        ]
    )

    assert receipt["status"] == "blocked"
    assert receipt["selected"] is None


def test_episode_start_restore_requires_pose_object_hold_and_visibility() -> None:
    ready = validate_wrist_observable_episode_start_restore(
        selected_joint_position_rad=[0.1] * 7,
        restored_joint_position_rad=[0.101] * 7,
        object_offset_m=[0.001, -0.002, 0.003],
        approved_task_object_pixel_count=250,
        approved_task_object_pixel_fraction=0.03,
        approved_task_object_within_frame_margin=True,
        restore_steps=12,
    )
    assert ready["status"] == "ready"
    assert ready["blockers"] == []
    assert ready["restore_digest"].startswith("sha256:")

    blocked = validate_wrist_observable_episode_start_restore(
        selected_joint_position_rad=[0.1] * 7,
        restored_joint_position_rad=[0.109] * 7,
        object_offset_m=[0.0, 0.006, 0.0],
        approved_task_object_pixel_count=199,
        approved_task_object_pixel_fraction=0.001,
        approved_task_object_within_frame_margin=False,
        restore_steps=80,
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == sorted(
        [
            BLOCKER_EPISODE_START_RESTORE_JOINT_MISMATCH,
            BLOCKER_EPISODE_START_RESTORE_OBJECT_MOVED,
            BLOCKER_EPISODE_START_RESTORE_OBJECT_NOT_VISIBLE,
        ]
    )


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


def test_rigid_offset_round_trips_a_camera_through_a_moving_body() -> None:
    """A camera on a non-articulation prim must be driven from its parent body.

    Arena parents the wrist camera under the Robotiq gripper base, which PhysX
    never writes a pose for.  A live run measured the hand travelling 0.27 m
    while every recorded wrist pose stayed byte-identical.
    """

    from blueprint_pipeline.adp009d_approach_capture import (
        MIN_WRIST_POSE_TRAVEL_M,
        apply_rigid_offset,
        rigid_offset_in_body_frame,
    )

    half = np.sqrt(0.5)
    # Reset: body yawed -90 deg, camera offset ahead of and above it.
    body_reset_pos = [3.4107, -3.2714, 0.8660]
    body_reset_quat = [0.0, 0.0, -half, half]
    camera_reset_pos = [3.4372, -3.0958, 0.7374]
    camera_reset_quat = [0.0, 1.0, 0.0, 0.0]

    offset_pos, offset_quat = rigid_offset_in_body_frame(
        body_position_world=body_reset_pos,
        body_quaternion_world_xyzw=body_reset_quat,
        child_position_world=camera_reset_pos,
        child_quaternion_world_xyzw=camera_reset_quat,
    )

    # Re-applying at the reset pose must reproduce the authored camera exactly.
    back_pos, back_quat = apply_rigid_offset(
        body_position_world=body_reset_pos,
        body_quaternion_world_xyzw=body_reset_quat,
        offset_position_body=offset_pos,
        offset_quaternion_body_xyzw=offset_quat,
    )
    for index in range(3):
        assert back_pos[index] == pytest.approx(camera_reset_pos[index], abs=1e-9)
    for index in range(4):
        assert back_quat[index] == pytest.approx(camera_reset_quat[index], abs=1e-9)

    # Moving the body must move the camera by the same rigid displacement.
    moved_pos = [3.1578, -3.3789, 0.7823]
    live_pos, _ = apply_rigid_offset(
        body_position_world=moved_pos,
        body_quaternion_world_xyzw=body_reset_quat,
        offset_position_body=offset_pos,
        offset_quaternion_body_xyzw=offset_quat,
    )
    for index in range(3):
        expected = camera_reset_pos[index] + (moved_pos[index] - body_reset_pos[index])
        assert live_pos[index] == pytest.approx(expected, abs=1e-9)
    # The camera must actually have travelled, unlike the observed live run.
    travel = sum((a - b) ** 2 for a, b in zip(live_pos, camera_reset_pos)) ** 0.5
    assert travel > MIN_WRIST_POSE_TRAVEL_M


def test_rigid_offset_rotates_the_camera_with_the_body() -> None:
    """Pure body rotation must swing the camera around it, not translate it."""

    from blueprint_pipeline.adp009d_approach_capture import (
        apply_rigid_offset,
        rigid_offset_in_body_frame,
    )

    half = np.sqrt(0.5)
    body_pos = [0.0, 0.0, 0.0]
    identity = [0.0, 0.0, 0.0, 1.0]
    camera_pos = [1.0, 0.0, 0.0]

    offset_pos, offset_quat = rigid_offset_in_body_frame(
        body_position_world=body_pos,
        body_quaternion_world_xyzw=identity,
        child_position_world=camera_pos,
        child_quaternion_world_xyzw=identity,
    )
    assert offset_pos[0] == pytest.approx(1.0, abs=1e-9)

    # Yaw the body +90 deg about z: the camera should swing to world +y.
    yawed = [0.0, 0.0, half, half]
    live_pos, live_quat = apply_rigid_offset(
        body_position_world=body_pos,
        body_quaternion_world_xyzw=yawed,
        offset_position_body=offset_pos,
        offset_quaternion_body_xyzw=offset_quat,
    )
    assert live_pos[0] == pytest.approx(0.0, abs=1e-9)
    assert live_pos[1] == pytest.approx(1.0, abs=1e-9)
    assert live_pos[2] == pytest.approx(0.0, abs=1e-9)
    assert live_quat[2] == pytest.approx(half, abs=1e-9)
    assert live_quat[3] == pytest.approx(half, abs=1e-9)


def test_the_runtime_does_not_drive_the_wrist_camera() -> None:
    """A controlled run proved the drive was a no-op, so it must not return.

    With the drive disabled and everything else byte-identical, the run
    produced identical numbers -- displacement 0.010018832981586456, waypoint
    error 0.1911235477432695, and the approved can still observed at 52,725
    pixels.  The prim already tracks the hand for rendering; only the sensor's
    reported pose buffer lags, which is why wrist_pose_travel_m reads 0.0
    either way.  Re-adding the drive would restore dead code and re-suggest a
    cause that was measured to be false.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    code = [
        line for line in source.splitlines() if not line.strip().startswith("#")
    ]
    assert not [line for line in code if "set_world_poses(" in line]
    assert not [line for line in code if "apply_rigid_offset(" in line]
    # The stale-pose gate stays: a mis-registered pose would still corrupt any
    # appearance layer composed against it.
    from blueprint_pipeline.adp009d_approach_capture import BLOCKER_WRIST_POSE_STALE

    assert BLOCKER_WRIST_POSE_STALE


def test_one_wrist_frame_is_undetermined_not_stale() -> None:
    """An aborted approach captures one frame; that cannot prove a frozen camera.

    A live run aborted at the first waypoint on the object-displacement guard,
    leaving a single wrist frame that showed 49,758 pixels of the approved can.
    Travel across one sample is trivially zero, which previously reported that
    demonstrably working camera as stale.
    """

    from blueprint_pipeline.adp009d_approach_capture import BLOCKER_WRIST_POSE_STALE

    single = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame_at(100, 49758, (3.4372, -3.0958, 0.7374))]
    )
    assert BLOCKER_WRIST_POSE_STALE not in single["blockers"]
    assert single["wrist_pose_travel_m"] == 0.0

    # Two identical samples still prove staleness, so the gate is not defanged.
    pair = summarize_wrist_approach_capture(
        captured_frames=[
            _wrist_frame_at(100, 49758, (3.4372, -3.0958, 0.7374)),
            _wrist_frame_at(101, 49758, (3.4372, -3.0958, 0.7374)),
        ]
    )
    assert BLOCKER_WRIST_POSE_STALE in pair["blockers"]


def test_runtime_selects_the_gripper_base_that_actually_exists() -> None:
    """The articulation exposes base_link, not robotiq_base_link.

    A live run recorded the body list: panda_link0..8, base_link, and the
    knuckle/finger bodies.  The old selection list looked for a name that does
    not exist and silently fell through to panda_link7, one joint short of the
    tool the wrist camera hangs from.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    selection = source[source.index("end_effector_name = next(") :]
    selection = selection[: selection.index("body_index =")]
    assert '"base_link",' in selection
    assert "robotiq_base_link" not in selection


def test_standoffs_clear_the_can_and_the_runtime_measures_real_clearance() -> None:
    """Standoff heights were inferred and wrong; the next run must measure them.

    A run at 0.34 m displaced the approved can by 10.3 mm and aborted at the
    first waypoint, so the tool did not clear the can even though the controlled
    body was well above it -- the fingers hang below.  Rather than guess at
    gripper geometry again, the runtime records the lowest gripper body.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime
    from blueprint_pipeline.adp009d_approach_capture import (
        APPROACH_GRIPPER_BODY_NAMES,
        APPROVED_CAN_TOP_ABOVE_SUPPORT_M,
    )

    # Every standoff must clear the can top by a real margin at the controlled
    # body, which is the necessary condition; the measured clearance is what
    # establishes the sufficient one.
    for standoff in APPROACH_STANDOFF_HEIGHTS_M:
        assert standoff > APPROVED_CAN_TOP_ABOVE_SUPPORT_M + 0.2

    # The observation gate was already satisfied at the first waypoint, so the
    # sequence must not descend below the height that produced it.
    assert APPROACH_STANDOFF_HEIGHTS_M == tuple(
        sorted(APPROACH_STANDOFF_HEIGHTS_M, reverse=True)
    )
    assert min(APPROACH_STANDOFF_HEIGHTS_M) >= 0.40

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert '"gripper_clearance_over_can_m"' in source
    assert '"lowest_gripper_body_z_m"' in source
    # Clearance must come from the simulator's body poses, not from a constant.
    assert "APPROACH_GRIPPER_BODY_NAMES" in source
    assert "body_pose_w)[0, gripper_indices, 2].min()" in source
    # The gripper body list must name bodies the articulation actually has.
    assert "base_link" in APPROACH_GRIPPER_BODY_NAMES
    assert "left_inner_finger" in APPROACH_GRIPPER_BODY_NAMES


def test_runtime_names_what_actually_disturbs_the_can() -> None:
    """The gripper measured 0.095 m of clearance while the can still moved.

    So whatever pushes it is elsewhere on the arm, and a magnitude alone cannot
    say whether the can is being pushed or is falling.  Both must be recorded
    from the simulator rather than inferred a third time.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")

    # Nearest body across the whole articulation, not just the gripper.
    assert '"nearest_body_to_can"' in source
    assert '"nearest_body_distance_to_can_m"' in source
    assert '"body_distances_to_can_m"' in source
    assert "torch.argmin(body_distances)" in source

    # Direction, not only magnitude: falling and being pushed differ.
    assert '"approved_can_offset_from_hold_m"' in source
    assert "approach_object_offset_m = [" in source
    # The magnitude must still come from the same vector, not a second read.
    assert "torch.linalg.vector_norm(approach_object_offset)" in source


def test_camera_refresh_never_advances_the_whole_scene() -> None:
    """A scene-wide refresh perturbed physics and moved the sealed can.

    Runs without it displaced the approved can by 0.00093 mm.  Runs with it
    displaced it by 10.019 mm, bit-identically across two runs, straight up in
    z, with the nearest articulation body 0.258 m away -- nothing was touching
    it.  Only the camera whose pose changed may be refreshed.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    approach = source[source.index("--- preregistered wrist approach") :]

    code = [
        line for line in approach.splitlines() if not line.strip().startswith("#")
    ]
    scene_refreshes = [line for line in code if "scene.update(" in line]
    assert scene_refreshes == [], f"scene-wide refresh perturbs physics: {scene_refreshes}"
    # No camera refresh of any kind remains in the approach block.
    assert not [line for line in code if "wrist_camera.update(" in line]



def test_the_displacement_guard_admits_the_unexplained_millimetric_drift() -> None:
    """Five runs aborted at 10.02 mm against a 10 mm line chosen before measuring.

    Nothing was in contact -- nearest body 0.258 m -- and the motion was almost
    purely upward.  Aborting there destroyed the evidence needed to explain it
    while the approach had already met its purpose, observing the can at 52,725
    pixels.  The guard now triggers on a real disturbance instead.
    """

    from blueprint_pipeline.adp009d_approach_capture import (
        APPROACH_MAX_OBJECT_DISPLACEMENT_M,
        BLOCKER_APPROACH_DISTURBED_OBJECT,
    )

    # The exact value five runs aborted on is now recorded, not fatal.
    observed = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 52725)],
        object_displacement_m=0.010018832981586456,
    )
    assert BLOCKER_APPROACH_DISTURBED_OBJECT not in observed["blockers"]
    assert observed["status"] == "observed"

    # A can genuinely knocked aside still stops the probe.
    knocked = summarize_wrist_approach_capture(
        captured_frames=[_wrist_frame(100, 52725)],
        object_displacement_m=0.5,
    )
    assert BLOCKER_APPROACH_DISTURBED_OBJECT in knocked["blockers"]
    # And the threshold is about the can's own radius, not an arbitrary figure.
    assert 0.03 <= APPROACH_MAX_OBJECT_DISPLACEMENT_M <= 0.08


def test_the_runtime_traces_the_displacement_per_step() -> None:
    """A binary abort cannot say when the rise starts or whether it tracks the arm."""

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "approach_object_trace" in source
    assert '"approved_can_per_step_trace"' in source
    # The arm position is recorded beside the can, so the two can be correlated.
    assert '"ee_position_world_m"' in source
    # Bounded, so a long approach cannot produce an unbounded receipt.
    assert "len(approach_object_trace) < 400" in source


def test_the_approach_holds_the_gripper_open() -> None:
    """A zero-initialised action commands a grasp under the measured convention.

    Dimension seven defaults to 0.0, and the probe measured 0.0 as closed, so
    every approach step was closing the gripper.  The per-step trace shows the
    can still for eighteen steps, rising to 31 mm as the gripper takes it, then
    drifting in x and y as the arm carries it -- an observation probe silently
    performing a pick.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    approach = source[source.index("--- preregistered wrist approach") :]

    assert 'approach_action[:, 7] = float(gripper_probe["open_command"])' in approach
    # Set after the joint targets, so it cannot be overwritten by them.
    assert approach.index("approach_action[:, :7] = joint_target") < approach.index(
        "approach_action[:, 7] ="
    )
    # Only when the convention is measured: guessing which value opens would
    # reintroduce exactly the bug this fixes.
    assert 'gripper_probe.get("status") == "measured"' in approach
