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
