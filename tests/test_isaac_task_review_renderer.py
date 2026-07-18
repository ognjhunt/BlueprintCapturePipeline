from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import isaac_task_review_renderer as renderer_module
from blueprint_pipeline.isaac_task_review_renderer import (
    IsaacTaskReviewRenderer,
    articulated_handle_focus_from_mesh,
    look_at_quaternion,
    project_world_point,
    rigid_head_camera_mount,
    rigid_head_camera_pose,
    select_robot_pov_calibration_target,
    task_camera_plan,
)


def test_articulated_handle_focus_prefers_far_disconnected_components() -> None:
    points = [
        # Eight-vertex panel centered 0.10 m from the hinge.
        (0.0, -0.10, -0.10),
        (0.2, -0.10, -0.10),
        (0.2, 0.10, -0.10),
        (0.0, 0.10, -0.10),
        (0.0, -0.10, 0.10),
        (0.2, -0.10, 0.10),
        (0.2, 0.10, 0.10),
        (0.0, 0.10, 0.10),
        # Two disconnected handle rails centered 0.35 m from the hinge.
        (0.34, -0.01, -0.10),
        (0.36, -0.01, -0.10),
        (0.36, 0.01, 0.10),
        (0.34, 0.01, 0.10),
        (0.345, -0.008, -0.09),
        (0.355, -0.008, -0.09),
        (0.355, 0.008, 0.09),
        (0.345, 0.008, 0.09),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
    ]

    result = articulated_handle_focus_from_mesh(
        points_world=points,
        face_vertex_counts=[len(face) for face in faces],
        face_vertex_indices=[vertex for face in faces for vertex in face],
        hinge_world_xyz=(0.0, 0.0, 0.0),
    )

    assert result["status"] == "resolved_disconnected_articulated_handle"
    assert result["selected_component_count"] == 2
    assert result["target_world_xyz_m"] == pytest.approx((0.35, 0.0, 0.0))


def test_look_at_quaternion_is_finite_and_normalized() -> None:
    quaternion = look_at_quaternion((2.0, -2.0, 1.5), (0.0, 0.0, 1.0))
    assert all(math.isfinite(value) for value in quaternion)
    assert math.sqrt(sum(value * value for value in quaternion)) == pytest.approx(1.0)


def _rotate_by_quaternion(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    cross = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    doubled = tuple(2.0 * value for value in cross)
    cross_again = (
        y * doubled[2] - z * doubled[1],
        z * doubled[0] - x * doubled[2],
        x * doubled[1] - y * doubled[0],
    )
    return tuple(
        vector[index] + w * doubled[index] + cross_again[index]
        for index in range(3)
    )  # type: ignore[return-value]


def test_look_at_quaternion_points_usd_minus_z_at_target_and_keeps_y_up() -> None:
    eye = (-1.264635, 1.471274, 1.28)
    target = (-1.591312, 1.471274, 1.241574)
    quaternion = look_at_quaternion(eye, target)
    forward = _rotate_by_quaternion(quaternion, (0.0, 0.0, -1.0))
    camera_up = _rotate_by_quaternion(quaternion, (0.0, 1.0, 0.0))
    desired = np.asarray(target) - np.asarray(eye)
    desired /= np.linalg.norm(desired)

    assert forward == pytest.approx(tuple(desired), abs=1e-6)
    assert camera_up[2] > 0.99


def test_rigid_head_camera_mount_inherits_head_yaw_without_task_reaim() -> None:
    initial_head_axes = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    # USD camera -Z points along robot/head +X at calibration time.
    initial_camera_axes = (
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
    )
    mount = rigid_head_camera_mount(
        head_origin=(0.0, 0.0, 1.5),
        head_axes=initial_head_axes,
        camera_eye=(0.12, 0.0, 1.5),
        camera_axes=initial_camera_axes,
    )

    pose = rigid_head_camera_pose(
        head_origin=(0.5, 0.25, 1.5),
        # Head yawed +90 degrees: local +X now points world +Y.
        head_axes=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        mount=mount,
    )

    assert pose["camera_world_xyz_m"] == pytest.approx((0.5, 0.37, 1.5))
    camera_rows = pose["camera_xmat_row_major"]
    inherited_forward = tuple(-value for value in camera_rows[2])
    assert inherited_forward == pytest.approx((0.0, 1.0, 0.0))


def test_microwave_task_camera_plan_is_head_mounted_egocentric() -> None:
    root = (-1.229635, 1.471274, 0.84)
    target = (-1.5913117835, 1.4712743045, 1.2415738103)

    plan = task_camera_plan(robot_root=root, target=target)

    assert plan["robot_head_mount"] == pytest.approx(
        (-1.229635, 1.471274, 1.35), abs=1e-6
    )
    assert plan["robot_pov_eye"] == pytest.approx(
        (-1.309635, 1.471274, 1.35), abs=1e-6
    )
    assert plan["robot_pov_target"] == pytest.approx(target, abs=1e-6)
    forward = np.asarray(target[:2]) - np.asarray(root[:2])
    camera_from_root = np.asarray(plan["robot_pov_eye"][:2]) - np.asarray(root[:2])
    assert float(np.linalg.norm(camera_from_root)) == pytest.approx(0.08)
    assert float(np.dot(forward, camera_from_root)) > 0.0
    pitch_down_deg = math.degrees(
        math.atan2(
            plan["robot_pov_eye"][2] - plan["robot_pov_target"][2],
            math.dist(plan["robot_pov_eye"][:2], plan["robot_pov_target"][:2]),
        )
    )
    assert pitch_down_deg <= renderer_module.ROBOT_POV_MAX_PITCH_DOWN_DEG
    assert math.dist(plan["robot_pov_eye"], target) < 1.0
    assert math.dist(plan["overview_eye"], root) < 2.0
    assert plan["overview_eye"][0] > root[0]  # behind a robot facing -X
    assert plan["overview_eye"][2] > target[2]


def test_arm_aware_head_calibration_restores_failed_paid_run_forearm_framing() -> None:
    # Exact live Isaac camera/task/link coordinates retained from qualification
    # attempt 033.  The task-only gaze centered the handle but projected both
    # active-arm links below the 640x480 viewport, so OSCAR correctly refused
    # to start its WAM loop.
    eye = (-1.3539847902593685, 1.4713665182353146, 1.35)
    task = (-1.591311813938244, 1.4712743113812294, 1.2415738723575633)
    landmarks = [
        {
            "landmark_id": "right_elbow_link",
            "world_position_xyz": (
                -1.2161186933517456,
                1.6547926664352417,
                0.8687322735786438,
            ),
        },
        {
            "landmark_id": "right_wrist_yaw_link",
            "world_position_xyz": (
                -1.3344639539718628,
                1.6872481107711792,
                0.7312478423118591,
            ),
        },
    ]

    target, evidence = select_robot_pov_calibration_target(
        eye=eye,
        task_target=task,
        registration_landmarks=landmarks,
    )

    assert evidence["selected_candidate"] == "late_june_forearm_weighted"
    selected = next(
        row
        for row in evidence["candidates"]
        if row["candidate"] == evidence["selected_candidate"]
    )
    assert selected["required_geometry_in_frame"] is True
    assert selected["in_frame_names"] == [
        "right_elbow_link",
        "right_wrist_yaw_link",
        "task_target",
    ]
    assert selected["projections"]["task_target"]["u_px"] == pytest.approx(
        19.601562756969713
    )
    assert selected["projections"]["right_elbow_link"]["v_px"] == pytest.approx(
        455.01452660638387
    )
    assert selected["projections"]["right_wrist_yaw_link"]["v_px"] == pytest.approx(
        430.74198964814565
    )
    assert target != pytest.approx(task)
    assert evidence["camera_remains_rigid_head_local_after_initial_calibration"] is True
    assert evidence["policy_camera_reaims_at_task_each_frame"] is False


def test_head_pov_lens_clears_rendered_face_surface() -> None:
    root = (-1.229635, 1.471274, 0.84)
    target = (-1.5913117835, 1.4712743045, 1.2415738103)

    plan = task_camera_plan(
        robot_root=root,
        target=target,
        robot_head_forward_extent_m=0.09,
    )

    assert math.dist(plan["robot_head_mount"], plan["robot_head_front"]) == pytest.approx(
        0.09
    )
    assert math.dist(plan["robot_head_mount"], plan["robot_pov_eye"]) == pytest.approx(
        0.12
    )
    assert math.dist(plan["robot_head_front"], plan["robot_pov_eye"]) == pytest.approx(
        renderer_module.ROBOT_POV_HEAD_SURFACE_CLEARANCE_M
    )
    quaternion = look_at_quaternion(plan["robot_pov_eye"], plan["robot_pov_target"])
    camera_axes = [
        _rotate_by_quaternion(quaternion, axis)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]
    contract = {
        "camera_world_xyz_m": list(plan["robot_pov_eye"]),
        "camera_xmat_row_major": camera_axes,
        "clipping_range_m": [0.05, 50.0],
        "intrinsics": {
            "fx": 168.0,
            "fy": 168.0,
            "cx": 320.0,
            "cy": 240.0,
            "image_width": 640,
            "image_height": 480,
        },
    }
    head_front = project_world_point(contract, plan["robot_head_front"])
    root_projection = project_world_point(contract, plan["robot_root"])

    assert head_front["in_depth_range"] is False
    assert root_projection["in_depth_range"] is True
    assert root_projection["in_frame"] is False


def test_egocentric_robot_pov_is_head_forward_not_overhead_fk_framing() -> None:
    """The policy camera stays at the head even when standing arms fall below view."""

    root = (-1.229635, 1.471274, 0.84)
    target = (-1.591311813938244, 1.4712743113812294, 1.2415738723575633)
    plan = task_camera_plan(robot_root=root, target=target)
    quaternion = look_at_quaternion(plan["robot_pov_eye"], plan["robot_pov_target"])
    camera_axes = [
        _rotate_by_quaternion(quaternion, axis)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]
    contract = {
        "camera_world_xyz_m": list(plan["robot_pov_eye"]),
        "camera_xmat_row_major": camera_axes,
        "clipping_range_m": [0.05, 50.0],
        "intrinsics": {
            "fx": 168.0498118992199,
            "fy": 168.0498118992199,
            "cx": 320.0,
            "cy": 240.0,
            "image_width": 640,
            "image_height": 480,
        },
    }
    # Exact controller/FK points from the failed paid attempt, including the
    # extrema across all 1,120 unique world-space landmarks emitted over the
    # step-1 action horizon.
    landmarks = (
        (-1.265531903, 1.374080031, 1.048531131),
        (-1.261314551, 1.282450502, 0.869198172),
        (-1.31546704, 1.269162796, 0.693474464),
        (-1.266367356, 1.574497637, 1.046464642),
        (-1.268677842, 1.657737912, 0.864031369),
        (-1.328133334, 1.661841018, 0.689563767),
        (-1.373512797, 1.684491303, 0.499800628),
        (-1.490333392, 1.709959662, 0.608515617),
        (-1.216911782, 1.289142009, 0.870052317),
        (-1.418392957, 1.225915806, 0.544061259),
        (-1.44603043, 1.717171248, 0.570746644),
        (-1.245570764, 1.372859197, 1.049688874),
    )

    task_projection = project_world_point(contract, target)
    landmark_projections = [
        project_world_point(contract, landmark) for landmark in landmarks
    ]

    assert task_projection["in_frame"] is True
    assert task_projection["v_px"] == pytest.approx(240.0)
    assert not any(row["in_frame"] for row in landmark_projections)
    assert all(
        row.get("v_px") is None or row["v_px"] > 480
        for row in landmark_projections
    )


def test_task_camera_plan_rejects_wrong_frame_or_coincident_target() -> None:
    with pytest.raises(
        RuntimeError, match="review_renderer_robot_target_standoff_invalid"
    ):
        task_camera_plan(robot_root=(0.0, 0.0, 0.84), target=(0.0, 0.0, 1.2))
    with pytest.raises(
        RuntimeError, match="review_renderer_robot_target_standoff_invalid"
    ):
        task_camera_plan(robot_root=(0.0, 0.0, 0.84), target=(100.0, 0.0, 1.2))


def test_task_camera_plan_allows_large_live_head_offset_only_after_mount_calibration() -> None:
    kwargs = {
        "robot_root": (0.0, 0.0, 0.84),
        "robot_head": (0.5, 0.0, 1.10),
        "target": (1.0, 0.0, 1.2),
    }

    with pytest.raises(
        RuntimeError, match="review_renderer_robot_head_mount_offset_invalid"
    ):
        task_camera_plan(**kwargs)

    plan = task_camera_plan(**kwargs, validate_robot_head_mount=False)
    assert plan["robot_head_mount"] == pytest.approx((0.5, 0.0, 1.35))


def test_place_reads_back_usd_pose_and_target_projects_inside_near_plane() -> None:
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Camera.Define(stage, "/World/Camera").GetClippingRangeAttr().Set(
        (renderer_module.REVIEW_NEAR_CLIP_M, renderer_module.REVIEW_FAR_CLIP_M)
    )
    renderer = IsaacTaskReviewRenderer.__new__(IsaacTaskReviewRenderer)
    renderer.stage = stage
    eye = (-1.264635, 1.471274, 1.28)
    target = (-1.591312, 1.471274, 1.241574)

    contract = renderer._place("/World/Camera", eye, target)
    projection = project_world_point(contract, target)

    assert contract["pose_readback"]["status"] == "PASS"
    assert contract["clipping_range_m"] == [0.05, 50.0]
    assert projection["depth_m"] == pytest.approx(math.dist(eye, target))
    assert projection["depth_m"] < 1.0  # regression: USD default near plane was 1 m
    assert projection["in_depth_range"] is True
    assert projection["in_frame"] is True
    assert projection["u_px"] == pytest.approx(320.0)
    assert projection["v_px"] == pytest.approx(240.0)


def test_synthetic_head_height_is_cached_in_head_local_not_world_up() -> None:
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    robot = UsdGeom.Xform.Define(stage, "/World/G1")
    UsdGeom.Xform.Define(stage, "/World/G1/head_link")
    robot_xform = UsdGeom.Xformable(robot.GetPrim())
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.84))
    renderer = IsaacTaskReviewRenderer.__new__(IsaacTaskReviewRenderer)
    renderer.stage = stage
    renderer.robot_prim_path = "/World/G1"
    renderer._robot_head_origin_pose_local_m = None

    initial = renderer._head_center()
    assert initial == pytest.approx((0.0, 0.0, 1.35))
    assert renderer._robot_head_origin_pose_local_m == pytest.approx((0.0, 0.0, 0.51))

    robot_xform.ClearXformOpOrder()
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.84))
    robot_xform.AddRotateYOp().Set(90.0)
    rotated = renderer._head_center()

    # The synthetic 51 cm head offset rotates with the live robot frame. It is
    # not reapplied along world +Z after the robot pitches.
    assert rotated == pytest.approx((1.51, 0.0, 0.84), abs=1e-6)


def test_render_refreshes_live_state_around_updates_and_frame_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class App:
        def update(self) -> None:
            events.append("update")

    class Annotator:
        def get_data(self):
            events.append("get_data")
            data = np.zeros((480, 640, 4), dtype=np.uint8)
            data[:, ::2, :3] = (220, 80, 30)
            return data

    class UniformAnnotator:
        def get_data(self):
            return np.zeros((480, 640, 4), dtype=np.uint8)

    renderer = IsaacTaskReviewRenderer.__new__(IsaacTaskReviewRenderer)
    renderer.app = App()
    renderer.heartbeat_callback = lambda: events.append("heartbeat")
    renderer.frames_dir = tmp_path
    renderer.robot_prim_path = "/World/G1"
    renderer.overview_path = "/World/BlueprintReview/OverviewCamera"
    renderer.robot_pov_path = "/World/BlueprintReview/RobotPOVCamera"
    renderer.annotators = {
        "overview": Annotator(),
        "robot_pov": Annotator(),
    }
    renderer._center = lambda _prim_path: (1.0, 0.0, 1.0)
    renderer._world_translation = lambda _prim_path: (0.0, 0.0, 0.84)
    renderer._head_center = lambda: (0.0, 0.0, 1.55)
    renderer._head_rotation_rows = lambda: [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    renderer._head_forward_extent = lambda _forward_xy: 0.09
    renderer._place = lambda camera_path, eye, target: {
        "available": True,
        "camera_path": camera_path,
        "projection_token": "perspective",
        "resolution": [640, 480],
        "camera_world_xyz_m": list(eye),
        "clipping_range_m": [0.05, 50.0],
        "camera_xmat_row_major": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "intrinsics": {
            "available": True,
            "fx": 168.0,
            "fy": 168.0,
            "cx": 320.0,
            "cy": 240.0,
            "image_width": 640,
            "image_height": 480,
        },
    }
    def project_for_render(contract, point):
        overview = contract.get("camera_path") == renderer.overview_path
        visible = overview or float(point[0]) > 0.5
        return {
            "world_xyz_m": list(point),
            "u_px": 320.0,
            "v_px": 240.0,
            "depth_m": 1.0,
            "in_depth_range": visible,
            "in_frame": visible,
        }

    monkeypatch.setattr(renderer_module, "project_world_point", project_for_render)

    artifacts = renderer.render(
        step_index=7,
        target_prim_path="/root/Microwave017/Microwave017_Door",
    )

    assert [artifact["camera_role"] for artifact in artifacts] == [
        "overview",
        "robot_pov",
    ]
    assert all(Path(artifact["path"]).is_file() for artifact in artifacts)
    assert all(artifact["camera_contract"]["available"] for artifact in artifacts)
    contracts = {
        artifact["camera_role"]: artifact["camera_contract"] for artifact in artifacts
    }
    assert contracts["robot_pov"]["viewpoint_mode"] == (
        "robot_head_mounted_egocentric"
    )
    assert contracts["robot_pov"]["policy_observation_eligible"] is True
    assert contracts["overview"]["viewpoint_mode"] == (
        "task_framed_third_person_review"
    )
    assert contracts["overview"]["policy_observation_eligible"] is False
    assert all(
        artifact["camera_contract"]["framing_validation"]["status"] == "PASS"
        for artifact in artifacts
    )
    assert all(artifact["visual_signal"]["non_uniform"] for artifact in artifacts)
    assert events.count("heartbeat") == 19
    assert events.count("update") == 3
    for index, event in enumerate(events):
        if event == "update":
            assert events[index - 1 : index + 2] == [
                "heartbeat",
                "update",
                "heartbeat",
            ]

    renderer.annotators["robot_pov"] = UniformAnnotator()
    degraded = renderer.capture_current(step_index=8)
    robot_pov = next(row for row in degraded if row["camera_role"] == "robot_pov")
    assert Path(robot_pov["path"]).is_file()
    assert robot_pov["visual_signal"]["status"] == "blocked"
    assert robot_pov["visual_signal"]["non_uniform"] is False
    assert robot_pov["visual_signal"]["blockers"] == [
        "review_renderer_robot_pov_visual_signal_too_low"
    ]


def test_follow_live_robot_reauthors_head_pov_from_each_live_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = IsaacTaskReviewRenderer.__new__(IsaacTaskReviewRenderer)
    renderer.heartbeat_callback = None
    renderer.robot_prim_path = "/World/G1"
    renderer.overview_path = "/World/BlueprintReview/OverviewCamera"
    renderer.robot_pov_path = "/World/BlueprintReview/RobotPOVCamera"
    renderer._target_prim_path = None
    renderer._task_target_center = None
    roots = iter([(0.0, 0.0, 0.84), (0.2, 0.0, 0.84)])
    live_root = [next(roots)]
    renderer._center = lambda _prim_path: (1.0, 0.0, 1.0)
    renderer._world_translation = lambda _prim_path: live_root[0]
    live_head_x_offset = [0.0]
    renderer._head_center = lambda: (
        live_root[0][0] + live_head_x_offset[0],
        live_root[0][1],
        live_root[0][2] + 0.51,
    )
    live_head_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    renderer._head_rotation_rows = lambda: live_head_axes
    renderer._head_forward_extent = lambda _forward_xy: 0.09
    renderer._place = lambda camera_path, eye, target: {
        "available": True,
        "camera_path": camera_path,
        "camera_world_xyz_m": list(eye),
        "camera_xmat_row_major": [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        "look_at_target_world_xyz_m": list(target),
    }
    renderer._place_axes = lambda camera_path, eye, axes: {
        "available": True,
        "camera_path": camera_path,
        "camera_world_xyz_m": list(eye),
        "camera_xmat_row_major": [list(row) for row in axes],
    }
    def projection(contract, point):
        overview = contract.get("camera_path") == renderer.overview_path
        in_front = bool(point[0] > live_root[0][0] + 0.5)
        visible = overview or in_front
        return {
            "world_xyz_m": list(point),
            "u_px": 320.0,
            "v_px": 240.0,
            "depth_m": 1.0,
            "in_depth_range": visible,
            "in_frame": visible,
        }

    monkeypatch.setattr(renderer_module, "project_world_point", projection)

    first = renderer.follow_live_robot(target_prim_path="/root/Microwave/Door")
    live_root[0] = next(roots)
    # Simulate a tipped articulation: the live head is now farther than the
    # standing-pose calibration guard permits from the pelvis/root.  The rigid
    # camera must continue following it so the backend can measure and report
    # the unsafe stance instead of crashing inside the renderer.
    live_head_x_offset[0] = 0.5
    live_head_axes[:] = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    second = renderer.follow_live_robot()

    first_eye = first["robot_pov"]["camera_world_xyz_m"]
    second_eye = second["robot_pov"]["camera_world_xyz_m"]
    assert first_eye == pytest.approx((0.12, 0.0, 1.35))
    assert second_eye == pytest.approx((0.7, 0.12, 1.35))
    assert second["robot_pov"]["viewpoint_mode"] == (
        "robot_head_mounted_egocentric"
    )
    assert second["robot_pov"]["mount_motion_model"] == "rigid_head_local_transform"
    assert second["robot_pov"]["gaze_motion_model"] == (
        "inherits_head_orientation_no_task_reaim"
    )
    assert second["robot_pov"]["mount_calibrated_this_update"] is False


def test_prewarm_realizes_both_products_before_heartbeat_attachment() -> None:
    events: list[str] = []

    class Orchestrator:
        def step(self, **kwargs) -> None:
            events.append(f"step:{kwargs}")

    class Rep:
        orchestrator = Orchestrator()

    class Annotator:
        def __init__(self, role: str) -> None:
            self.role = role

        def get_data(self):
            events.append(f"get_data:{self.role}")
            return np.zeros((480, 640, 4), dtype=np.uint8)

    renderer = IsaacTaskReviewRenderer.__new__(IsaacTaskReviewRenderer)
    renderer.rep = Rep()
    renderer.heartbeat_callback = None
    renderer._prewarm_completed = False
    renderer._prewarm_evidence = None
    renderer.render_products = {
        "overview": object(),
        "robot_pov": object(),
    }
    renderer.annotators = {
        "overview": Annotator("overview"),
        "robot_pov": Annotator("robot_pov"),
    }

    with pytest.raises(RuntimeError, match="review_renderer_heartbeat_before_prewarm"):
        renderer.attach_heartbeat_callback(lambda: None)

    evidence = renderer.prewarm(update_count=2)

    assert events == [
        "step:{'delta_time': 0.0, 'pause_timeline': True, "
        "'wait_for_render': True}",
        "get_data:overview",
        "get_data:robot_pov",
    ]
    assert evidence["status"] == "passed"
    assert evidence["render_products_realized"] is True
    assert evidence["render_steps_executed"] == 1
    assert evidence["render_step_delta_time_seconds"] == 0.0
    assert evidence["heartbeat_callback_attached_during_prewarm"] is False
    assert evidence["heartbeat_callback_attached_after_prewarm"] is False
    assert evidence["rgb_shapes"] == {
        "overview": [480, 640, 4],
        "robot_pov": [480, 640, 4],
    }
    renderer.attach_heartbeat_callback(lambda: events.append("heartbeat"))
    renderer._heartbeat()
    assert events[-1] == "heartbeat"
    assert (
        renderer.prewarm_contract()["heartbeat_callback_attached_after_prewarm"]
        is True
    )


def test_prewarm_retries_explicit_zero_delta_capture_until_both_frames_exist() -> None:
    events: list[str] = []

    class Orchestrator:
        def step(self, **kwargs) -> None:
            events.append("step")

    class Rep:
        orchestrator = Orchestrator()

    class Annotator:
        def __init__(self, empty_reads: int) -> None:
            self.empty_reads = empty_reads
            self.reads = 0

        def get_data(self):
            self.reads += 1
            if self.reads <= self.empty_reads:
                return np.asarray([])
            return np.zeros((480, 640, 4), dtype=np.uint8)

    renderer = IsaacTaskReviewRenderer.__new__(IsaacTaskReviewRenderer)
    renderer.rep = Rep()
    renderer.heartbeat_callback = None
    renderer._prewarm_completed = False
    renderer._prewarm_evidence = None
    renderer.render_products = {
        "overview": object(),
        "robot_pov": object(),
    }
    renderer.annotators = {
        "overview": Annotator(empty_reads=1),
        "robot_pov": Annotator(empty_reads=2),
    }

    evidence = renderer.prewarm(update_count=4)

    assert events == ["step", "step", "step"]
    assert evidence["render_steps_executed"] == 3
    assert evidence["rgb_shapes"] == {
        "overview": [480, 640, 4],
        "robot_pov": [480, 640, 4],
    }
