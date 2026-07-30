from __future__ import annotations

import inspect
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.nvidia_warehouse_native_camera_canary import (
    _apply_and_measure_render_only_joint_pose,
    _apply_runtime_asset_relocations,
    _articulation_link_world_pose_matrices,
    _backend_array_to_numpy,
    _camera_quaternion_wxyz,
    _fabric_world_pose_matrix,
    _load_materialization_manifest,
    _project_required_external_entities,
    _project_world_points,
    _quaternion_wxyz_from_world_pose_matrix,
    _render_world_without_physics_advance,
    _rigid_wrist_mount_from_initial_task_framing,
    _simulation_app_launch_config,
    _synchronize_camera_to_rigid_link,
    _world_pose_matrix_from_backend_pose,
    import_simulation_app,
    isaac_sim_6_backend,
    run_native_camera_canary,
)
from blueprint_pipeline.nvidia_warehouse_workcell import CANARY_SPEC_SCHEMA_VERSION
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def _spec(path: Path) -> None:
    value = {
        "schema_version": CANARY_SPEC_SCHEMA_VERSION,
        "cameras": {
            "external": {"resolution": [640, 480]},
            "wrist": {"resolution": [640, 480]},
        },
    }
    value["spec_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _backend(*, output_dir: Path, **_kwargs):
    output_dir.mkdir(parents=True)
    views = {}
    for view_id in ("external", "wrist"):
        paths = {}
        for phase, color in (("initial", (30, 60, 90)), ("commanded", (90, 60, 30))):
            path = output_dir / f"{view_id}_{phase}.png"
            image = Image.new("RGB", (640, 480), color=color)
            for x in range(0, 640, 20):
                image.putpixel((x, x % 480), (255, 255, 255))
            image.save(path)
            paths[f"{phase}_frame_path"] = str(path)
        views[view_id] = {
            **paths,
            "required_entities_projected_in_frame": {
                "franka": True,
                "spraycan": True,
                "tray": True,
            },
        }
    return {
        "isaac_sim_major_version": 6,
        "scene_loaded": True,
        "missing_dataset_local_dependencies": [],
        "franka_dof_count": 9,
        "spraycan_collision_mesh_count": 3,
        "spraycan_runtime_rigid_body": True,
        "views": views,
        "wrist_mount_calibration": {
            "mode": "one_time_initial_task_framing_rigid_parent_local_mount",
            "target_entity_ids": ["spraycan", "tray"],
            "calibrated_after_initial_joint_hold": True,
            "per_frame_task_reaim_performed": False,
        },
        "franka_render_only_joint_state": {
            "mode": "render_only_kinematic_joint_state_transition",
            "physics_dynamics_claimed": False,
            "initial": {
                "max_abs_position_error_rad": 0.001,
                "zero_time_scene_update_requested": True,
            },
            "commanded": {
                "max_abs_position_error_rad": 0.001,
                "zero_time_scene_update_requested": True,
            },
        },
        "camera_transition_physics_steps_advanced": 0,
        "wrist_camera_world_displacement_m": 0.02,
        "wrist_camera_local_transform_delta": 0.0,
        "external_wrist_timestamp_pairs_exact": True,
    }


def test_usd_camera_convention_projects_negative_z_forward_and_builds_identity_pose() -> None:
    quaternion = _camera_quaternion_wxyz((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    assert quaternion == pytest.approx([1.0, 0.0, 0.0, 0.0])
    projected = _project_world_points(
        camera_to_world=np.eye(4),
        points={"center": [0.0, 0.0, -1.0], "behind": [0.0, 0.0, 1.0]},
        width=640,
        height=480,
        vfov_deg=60.0,
    )
    assert projected == {"center": True, "behind": False}


def test_external_projection_accepts_any_visible_franka_link_origin() -> None:
    required, evidence = _project_required_external_entities(
        camera_to_world=np.eye(4),
        task_points={"spraycan": [0.0, 0.0, -1.0], "tray": [0.1, 0.1, -1.0]},
        franka_link_points={
            "panda_link0": [10.0, 0.0, -1.0],
            "panda_link4": [0.0, 0.0, -1.0],
        },
        width=640,
        height=480,
        vfov_deg=60.0,
    )

    assert required == {"franka": True, "spraycan": True, "tray": True}
    assert evidence["franka_link_origins_projected_in_frame"] == {
        "panda_link0": False,
        "panda_link4": True,
    }


def test_external_projection_fails_closed_without_franka_link_origins() -> None:
    with pytest.raises(ValueError, match="native_warehouse_franka_link_projection_points_missing"):
        _project_required_external_entities(
            camera_to_world=np.eye(4),
            task_points={"spraycan": [0.0, 0.0, -1.0]},
            franka_link_points={},
            width=640,
            height=480,
            vfov_deg=60.0,
        )


def test_zero_time_scene_update_uses_cuda_world_render_and_preserves_step_index() -> None:
    class World:
        current_time_step_index = 7
        device = "cuda:0"
        physics_sim_view = object()

        def __init__(self):
            self.calls = []

        def is_playing(self):
            return True

        def render(self):
            self.calls.append("render")

    world = World()
    _render_world_without_physics_advance(world)

    assert world.calls == ["render"]
    assert world.current_time_step_index == 7


def test_zero_time_scene_update_rejects_cpu_world_that_cannot_refresh_kinematics() -> None:
    class World:
        current_time_step_index = 7
        device = "cpu"
        physics_sim_view = object()

        def is_playing(self):
            return True

        def render(self):
            raise AssertionError("must fail before rendering")

    with pytest.raises(
        ValueError, match="native_franka_zero_time_scene_update_cuda_backend_required"
    ):
        _render_world_without_physics_advance(World())


def test_zero_time_scene_update_fails_if_world_advances_physics() -> None:
    class World:
        current_time_step_index = 7
        device = "cuda:0"
        physics_sim_view = object()

        def is_playing(self):
            return True

        def render(self):
            self.current_time_step_index += 1

    with pytest.raises(ValueError, match="native_franka_zero_time_scene_update_advanced_physics"):
        _render_world_without_physics_advance(World())


def test_backend_array_to_numpy_moves_tensor_like_value_to_cpu() -> None:
    calls: list[str] = []

    class TensorLike:
        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def numpy(self):
            calls.append("numpy")
            return np.asarray([1.0, 2.0])

    assert np.array_equal(_backend_array_to_numpy(TensorLike()), np.asarray([1.0, 2.0]))
    assert calls == ["detach", "cpu", "numpy"]


def test_backend_pose_matrix_uses_usd_row_vector_convention() -> None:
    half_turn = math.radians(90.0) / 2.0
    matrix = _world_pose_matrix_from_backend_pose(
        [1.0, 2.0, 3.0],
        [math.cos(half_turn), 0.0, 0.0, math.sin(half_turn)],
    )

    assert np.asarray([1.0, 0.0, 0.0, 0.0]) @ matrix == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert np.asarray([0.0, 0.0, 0.0, 1.0]) @ matrix == pytest.approx([1.0, 2.0, 3.0, 1.0])


def test_fabric_world_pose_query_explicitly_bypasses_stale_usd() -> None:
    calls: list[bool] = []

    class View:
        def get_world_poses(self, *, usd):
            calls.append(usd)
            return np.asarray([[0.1, 0.2, 0.3]]), np.asarray([[1.0, 0.0, 0.0, 0.0]])

    class Prim:
        _xform_prim_view = View()

    matrix = _fabric_world_pose_matrix(Prim())

    assert calls == [False]
    assert matrix[3, :3] == pytest.approx([0.1, 0.2, 0.3])


def test_wrist_camera_prim_exists_before_reset_and_syncs_from_live_link_afterward() -> None:
    source = inspect.getsource(isaac_sim_6_backend)

    camera_definition = source.index("wrist_prim = UsdGeom.Camera.Define")
    world_reset = source.index("world.reset()")
    calibrated_world_sync = source.index("_synchronize_camera_to_rigid_link")

    assert camera_definition < world_reset < calibrated_world_sync
    assert 'wrist_path = "/World/Cameras/Wrist"' in source
    assert "get_link_transforms_after_update_articulations_kinematic" in source


def test_articulation_link_pose_uses_explicit_kinematic_update_and_xyzw_tensor_order() -> None:
    calls: list[str] = []

    class SimulationView:
        def update_articulations_kinematic(self):
            calls.append("update_articulations_kinematic")

    half_turn = math.radians(90.0) / 2.0

    class PhysicsView:
        def get_link_transforms(self):
            calls.append("get_link_transforms")
            return np.asarray(
                [
                    [
                        [1.0, 2.0, 3.0, 0.0, 0.0, math.sin(half_turn), math.cos(half_turn)],
                        [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ]
            )

    class Articulation:
        _physics_view = PhysicsView()
        body_names = ["panda_link0", "panda_hand"]

    class Robot:
        _articulation_view = Articulation()

    matrices = _articulation_link_world_pose_matrices(
        robot=Robot(),
        simulation_view=SimulationView(),
    )

    assert calls == ["update_articulations_kinematic", "get_link_transforms"]
    assert matrices["panda_link0"][3, :3] == pytest.approx([1.0, 2.0, 3.0])
    assert np.asarray([1.0, 0.0, 0.0, 0.0]) @ matrices["panda_link0"] == pytest.approx(
        [0.0, 1.0, 0.0, 0.0]
    )
    assert matrices["panda_hand"][3, :3] == pytest.approx([4.0, 5.0, 6.0])


def test_world_camera_sync_preserves_one_rigid_parent_local_mount() -> None:
    calls: list[dict[str, object]] = []

    class Camera:
        def set_world_pose(self, **kwargs):
            calls.append(kwargs)

    parent_initial = _world_pose_matrix_from_backend_pose(
        [0.4, -0.2, 1.1],
        [1.0, 0.0, 0.0, 0.0],
    )
    half_turn = math.radians(30.0) / 2.0
    parent_commanded = _world_pose_matrix_from_backend_pose(
        [0.5, -0.1, 1.2],
        [math.cos(half_turn), 0.0, 0.0, math.sin(half_turn)],
    )
    mount_translation = [0.0, 0.1, 0.03]
    mount_orientation = _camera_quaternion_wxyz([1.0, 0.0, -0.2], [0.0, 0.0, 1.0])

    initial = _synchronize_camera_to_rigid_link(
        camera=Camera(),
        parent_to_world=parent_initial,
        mount_translation_parent=mount_translation,
        mount_orientation_parent_wxyz=mount_orientation,
    )
    commanded = _synchronize_camera_to_rigid_link(
        camera=Camera(),
        parent_to_world=parent_commanded,
        mount_translation_parent=mount_translation,
        mount_orientation_parent_wxyz=mount_orientation,
    )

    assert len(calls) == 2
    assert all(call["camera_axes"] == "usd" for call in calls)
    assert np.linalg.norm(commanded[3, :3] - initial[3, :3]) > 0.001
    assert commanded @ np.linalg.inv(parent_commanded) == pytest.approx(
        initial @ np.linalg.inv(parent_initial)
    )
    roundtrip = _world_pose_matrix_from_backend_pose(
        commanded[3, :3],
        _quaternion_wxyz_from_world_pose_matrix(commanded),
    )
    assert roundtrip == pytest.approx(commanded)


def test_wrist_mount_is_calibrated_once_in_parent_coordinates_toward_task_centroid() -> None:
    parent_to_world = np.eye(4)
    angle = math.radians(37.0)
    parent_to_world[:3, :3] = [
        [math.cos(angle), math.sin(angle), 0.0],
        [-math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    parent_to_world[3, :3] = [0.4, -0.2, 1.1]
    mount = np.asarray([0.0, 0.1, 0.03])
    points = {
        "spraycan": [0.1, 0.05, 0.9],
        "tray": [-0.1, 0.35, 0.9],
    }

    quaternion, evidence = _rigid_wrist_mount_from_initial_task_framing(
        parent_to_world=parent_to_world,
        mount_translation_parent=mount,
        target_world_points=points,
    )

    expected_target = np.mean(np.asarray(list(points.values())), axis=0)
    eye_world = np.concatenate((mount, [1.0])) @ parent_to_world
    expected_forward = expected_target - eye_world[:3]
    expected_forward /= np.linalg.norm(expected_forward)
    observed_forward = np.concatenate((evidence["mount_forward_parent"], [0.0])) @ parent_to_world
    observed_forward = observed_forward[:3] / np.linalg.norm(observed_forward[:3])
    assert observed_forward == pytest.approx(expected_forward)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert evidence["target_entity_ids"] == ["spraycan", "tray"]
    assert evidence["per_frame_task_reaim_performed"] is False


def test_wrist_mount_calibration_handles_world_up_parallel_to_gaze() -> None:
    quaternion, evidence = _rigid_wrist_mount_from_initial_task_framing(
        parent_to_world=np.eye(4),
        mount_translation_parent=[0.0, 0.0, 0.0],
        target_world_points={"spraycan": [0.0, 0.0, 1.0]},
    )

    assert np.isfinite(quaternion).all()
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert abs(np.dot(evidence["mount_forward_parent"], evidence["mount_up_parent"])) < 1e-9


def test_joint_pose_is_rendered_without_requesting_physics_steps() -> None:
    calls: list[tuple[str, np.ndarray]] = []

    class Robot:
        measured = np.asarray([0.1, -0.2, 0.3])

        def set_joint_positions(self, value):
            calls.append(("set_joint_positions", np.asarray(value)))

        def set_joint_velocities(self, value):
            calls.append(("set_joint_velocities", np.asarray(value)))

        def get_joint_positions(self):
            return self.measured

    renders: list[int] = []
    result = _apply_and_measure_render_only_joint_pose(
        robot=Robot(),
        joint_positions=[0.1, -0.2, 0.3],
        phase="initial",
        render=lambda: renders.append(1),
        render_count=4,
    )

    assert [name for name, _value in calls] == [
        "set_joint_positions",
        "set_joint_velocities",
    ]
    assert np.array_equal(calls[1][1], np.zeros(3))
    assert len(renders) == 4
    assert result["physics_steps_requested"] == 0
    assert result["zero_time_scene_update_requested"] is True
    assert result["max_abs_position_error_rad"] == pytest.approx(0.0)


def test_joint_pose_is_converted_to_the_robot_cuda_backend_before_application() -> None:
    class BackendValue:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

    conversions: list[tuple[np.ndarray, str]] = []

    class BackendUtils:
        @staticmethod
        def convert(value, device):
            array = np.asarray(value)
            conversions.append((array, device))
            return BackendValue(array)

    applied: list[tuple[str, BackendValue]] = []

    class Robot:
        _backend_utils = BackendUtils()
        _device = "cuda:0"

        def set_joint_positions(self, value):
            applied.append(("set_joint_positions", value))

        def set_joint_velocities(self, value):
            applied.append(("set_joint_velocities", value))

        def get_joint_positions(self):
            return np.asarray([0.1, -0.2, 0.3])

    result = _apply_and_measure_render_only_joint_pose(
        robot=Robot(),
        joint_positions=[0.1, -0.2, 0.3],
        phase="initial",
        render=lambda: None,
        render_count=1,
    )

    assert len(conversions) == 2
    assert all(device == "cuda:0" for _value, device in conversions)
    assert np.array_equal(conversions[0][0], np.asarray([0.1, -0.2, 0.3]))
    assert np.array_equal(conversions[1][0], np.zeros(3))
    assert [name for name, _value in applied] == [
        "set_joint_positions",
        "set_joint_velocities",
    ]
    assert all(isinstance(value, BackendValue) for _name, value in applied)
    assert result["max_abs_position_error_rad"] == pytest.approx(0.0)


def test_render_only_joint_state_fails_closed_without_state_api() -> None:
    class Robot:
        def set_joint_positions(self, _value):
            pass

        def get_joint_positions(self):
            return np.zeros(2)

    with pytest.raises(
        ValueError,
        match="native_franka_render_only_joint_state_api_missing:set_joint_velocities",
    ):
        _apply_and_measure_render_only_joint_pose(
            robot=Robot(),
            joint_positions=[0.0, 0.0],
            phase="initial",
            render=lambda: None,
            render_count=1,
        )


def test_render_only_joint_state_wraps_render_failure_in_safe_code() -> None:
    class Robot:
        def set_joint_positions(self, _value):
            pass

        def set_joint_velocities(self, _value):
            pass

        def get_joint_positions(self):
            return np.zeros(2)

    with pytest.raises(
        ValueError,
        match="native_franka_render_only_joint_state_failed:RuntimeError",
    ):
        _apply_and_measure_render_only_joint_pose(
            robot=Robot(),
            joint_positions=[0.0, 0.0],
            phase="initial",
            render=lambda: (_ for _ in ()).throw(RuntimeError("opaque provider detail")),
            render_count=1,
        )


def test_simulation_app_import_falls_back_when_isaacsim_shim_is_not_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isaacsim = types.ModuleType("isaacsim")
    isaacsim.SimulationApp = None
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_isaac = types.ModuleType("omni.isaac")
    omni_isaac.__path__ = []
    omni_kit = types.ModuleType("omni.isaac.kit")

    class LegacySimulationApp:
        pass

    omni_kit.SimulationApp = LegacySimulationApp
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.isaac", omni_isaac)
    monkeypatch.setitem(sys.modules, "omni.isaac.kit", omni_kit)

    assert import_simulation_app() is LegacySimulationApp


def test_simulation_app_launch_config_disables_process_terminating_fast_shutdown() -> None:
    first = _simulation_app_launch_config()
    second = _simulation_app_launch_config()

    assert first["fast_shutdown"] is False
    assert first["headless"] is True
    assert first is not second


def test_runtime_asset_relocations_require_exact_local_binding(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    owner = assets / "Props" / "clock" / "clock.usd"
    replacement = assets / "Props" / "clock" / "Textures" / "albedo.png"
    owner.parent.mkdir(parents=True)
    replacement.parent.mkdir(parents=True)
    owner.write_bytes(b"usd")
    replacement.write_bytes(b"png")
    observed = []

    result = _apply_runtime_asset_relocations(
        assets_root=assets,
        manifest={
            "runtime_asset_relocations": [
                {
                    "owner_relative_path": "Props/clock/clock.usd",
                    "source_asset_uri": "omniverse://art/clock/Textures/albedo.png",
                    "replacement_relative_path": "Props/clock/Textures/albedo.png",
                    "replacement_authored_path": "./Textures/albedo.png",
                }
            ]
        },
        layer_relocator=lambda path, source, replacement_path: (
            observed.append((path, source, replacement_path)) or 1
        ),
    )

    assert result["relocation_count"] == 1
    assert result["authored_replacement_count"] == 1
    assert observed == [
        (
            owner,
            "omniverse://art/clock/Textures/albedo.png",
            "./Textures/albedo.png",
        )
    ]


def test_materialization_manifest_resolves_from_extracted_bundle_layout(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "input"
    assets = extracted / "assets"
    assets.mkdir(parents=True)
    manifest_path = extracted / "materialization_manifest.json"
    manifest_path.write_text(json.dumps({"runtime_asset_relocations": []}), encoding="utf-8")

    resolved_path, manifest = _load_materialization_manifest(assets)

    assert resolved_path == manifest_path
    assert manifest == {"runtime_asset_relocations": []}


def test_materialization_manifest_prefers_direct_materialization_root(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    direct = assets / "materialization_manifest.json"
    direct.write_text(
        json.dumps({"runtime_asset_relocations": [{"source": "direct"}]}),
        encoding="utf-8",
    )
    (tmp_path / "materialization_manifest.json").write_text(
        json.dumps({"runtime_asset_relocations": [{"source": "parent"}]}),
        encoding="utf-8",
    )

    resolved_path, manifest = _load_materialization_manifest(assets)

    assert resolved_path == direct
    assert manifest["runtime_asset_relocations"] == [{"source": "direct"}]


def test_native_camera_canary_requires_scene_robot_rigid_object_and_two_synced_views(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)
    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=_backend,
    )

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert (
        result["assessment"]["views"]["external"]["frames"]["initial"]["relative_path"]
        == "runtime/external_initial.png"
    )
    assert result["paid_policy_or_wam_model_invoked"] is False
    assert result["claim_boundary"]["policy_wam_loop_proven"] is False


def test_native_camera_canary_fails_static_or_slipping_wrist_mount(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def broken_backend(**kwargs):
        value = _backend(**kwargs)
        value["wrist_camera_world_displacement_m"] = 0.0
        value["wrist_camera_local_transform_delta"] = 0.01
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=broken_backend,
    )

    assert result["status"] == "failed"
    assert "native_wrist_camera_did_not_move_with_hand" in result["blockers"]
    assert "native_wrist_camera_mount_not_rigid" in result["blockers"]


def test_native_camera_canary_fails_closed_on_missing_or_reaimed_mount_calibration(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def broken_backend(**kwargs):
        value = _backend(**kwargs)
        value["wrist_mount_calibration"] = {
            "mode": "one_time_initial_task_framing_rigid_parent_local_mount",
            "calibrated_after_initial_joint_hold": True,
            "per_frame_task_reaim_performed": True,
        }
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=broken_backend,
    )

    assert result["status"] == "failed"
    assert "native_wrist_camera_per_frame_reaim_not_forbidden" in result["blockers"]

    missing_result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "missing_result",
        backend=lambda **kwargs: {
            key: value
            for key, value in _backend(**kwargs).items()
            if key != "wrist_mount_calibration"
        },
    )
    assert "native_wrist_mount_calibration_missing_or_invalid" in missing_result["blockers"]


def test_native_camera_canary_fails_closed_when_joint_state_does_not_match(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def drifting_backend(**kwargs):
        value = _backend(**kwargs)
        value["franka_render_only_joint_state"]["commanded"]["max_abs_position_error_rad"] = 0.25
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=drifting_backend,
    )

    assert result["status"] == "failed"
    assert "native_franka_joint_state_error_exceeded:commanded" in result["blockers"]


def test_native_camera_canary_requires_zero_time_scene_update_evidence(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def missing_update_evidence_backend(**kwargs):
        value = _backend(**kwargs)
        value["franka_render_only_joint_state"]["commanded"].pop("zero_time_scene_update_requested")
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=missing_update_evidence_backend,
    )

    assert "native_franka_zero_time_scene_update_not_proven:commanded" in result["blockers"]


def test_native_camera_canary_fails_closed_when_camera_transition_steps_physics(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def stepping_backend(**kwargs):
        value = _backend(**kwargs)
        value["camera_transition_physics_steps_advanced"] = 1
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=stepping_backend,
    )

    assert "native_camera_transition_advanced_physics" in result["blockers"]


def test_native_camera_canary_fails_closed_on_missing_wrist_measurements(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def incomplete_backend(**kwargs):
        value = _backend(**kwargs)
        value.pop("wrist_camera_world_displacement_m")
        value["wrist_camera_local_transform_delta"] = float("nan")
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=incomplete_backend,
    )

    assert result["status"] == "failed"
    assert result["assessment"]["wrist_camera_world_displacement_m"] is None
    assert result["assessment"]["wrist_camera_local_transform_delta"] is None
    assert result["backend_result"]["wrist_camera_local_transform_delta"] is None
    assert "native_wrist_camera_world_displacement_missing_or_invalid" in result["blockers"]
    assert "native_wrist_camera_local_transform_missing_or_invalid" in result["blockers"]
