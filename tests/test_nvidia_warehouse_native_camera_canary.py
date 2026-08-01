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
    _add_prim_semantic_label,
    _apply_and_measure_render_only_joint_pose,
    _apply_runtime_asset_relocations,
    _articulation_link_world_pose_matrices,
    _author_renderable_semantic_label_tree,
    _backend_array_to_numpy,
    _camera_pose_backend_congruence,
    _camera_sensor_annotator_frame,
    _camera_quaternion_wxyz,
    _load_materialization_manifest,
    _project_required_external_entities,
    _project_world_points,
    _quaternion_wxyz_from_world_pose_matrix,
    _render_world_without_physics_advance,
    _rigid_wrist_mount_from_initial_task_framing,
    _semantic_entity_visibility,
    _simulation_app_launch_config,
    _summarize_required_entity_projections,
    _summarize_required_entity_visibility,
    _synchronize_camera_to_rigid_link,
    _unified_world_pose_matrix,
    _world_pose_matrix_from_backend_pose,
    import_simulation_app,
    isaac_sim_6_backend,
    run_native_camera_canary,
)
from blueprint_pipeline.nvidia_warehouse_workcell import CANARY_SPEC_SCHEMA_VERSION
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def _spec(path: Path, *, three_views: bool = False) -> None:
    value = {
        "schema_version": CANARY_SPEC_SCHEMA_VERSION,
        "cameras": {
            "external": {"resolution": [640, 480]},
            "wrist": {"resolution": [640, 480]},
        },
    }
    if three_views:
        value["cameras"]["external"].update(
            {"position_m": [1.0, 0.0, 1.0], "look_at_m": [0.0, 0.0, 0.0]}
        )
        value["cameras"]["external_2"] = {
            "resolution": [640, 480],
            "position_m": [-1.0, 0.0, 1.0],
            "look_at_m": [0.0, 0.0, 0.0],
        }
        value["required_views"] = ["external", "external_2", "wrist"]
    value["spec_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _backend(*, output_dir: Path, spec=None, **_kwargs):
    output_dir.mkdir(parents=True)
    views = {}
    required_views = spec.get("required_views", ["external", "wrist"])
    for view_index, view_id in enumerate(required_views):
        paths = {}
        for phase, color in (("initial", (30, 60, 90)), ("commanded", (90, 60, 30))):
            path = output_dir / f"{view_id}_{phase}.png"
            shifted = tuple(min(255, channel + view_index * 7) for channel in color)
            image = Image.new("RGB", (640, 480), color=shifted)
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
            "required_entities_visible_pixels": {
                **(
                    {"franka": True, "spraycan": True, "tray": True}
                    if view_id == "external"
                    else {}
                ),
                **({"spraycan_at_initial_pose": True} if view_id == "wrist" else {}),
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
        "camera_timestamps_exact": True,
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


def test_projection_summary_keeps_external_grounding_across_both_phases() -> None:
    summary = _summarize_required_entity_projections(
        view_id="external",
        projections_by_phase={
            "initial": {"franka": True, "spraycan": True, "tray": True},
            "commanded": {"franka": True, "spraycan": False, "tray": True},
        },
    )

    assert summary == {"franka": True, "spraycan": False, "tray": True}


def test_projection_summary_requires_initial_task_object_but_not_commanded_reaim() -> None:
    summary = _summarize_required_entity_projections(
        view_id="wrist",
        projections_by_phase={
            "initial": {"spraycan": True, "tray": True},
            "commanded": {"spraycan": False, "tray": True},
        },
    )

    assert summary == {"spraycan_at_initial_pose": True}

    missing_initial_object = _summarize_required_entity_projections(
        view_id="wrist",
        projections_by_phase={
            "initial": {"spraycan": False, "tray": True},
            "commanded": {"spraycan": True, "tray": True},
        },
    )
    assert missing_initial_object == {"spraycan_at_initial_pose": False}


def test_semantic_visibility_counts_only_rendered_target_pixels() -> None:
    data = np.zeros((12, 12), dtype=np.uint32)
    data[2:10, 2:10] = 7
    evidence = _semantic_entity_visibility(
        semantic_frame={
            "data": data,
            "info": {"idToLabels": {"0": {"class": "UNLABELLED"}, "7": {"class": "spraycan"}}},
        },
        entity_labels={"spraycan": "spraycan", "tray": "tray"},
    )

    assert evidence["spraycan"] == {
        "semantic_class": "spraycan",
        "semantic_ids": [7],
        "visible_pixel_count": 64,
        "minimum_visible_pixel_count": 64,
        "visible": True,
        "render_derived": True,
        "observed_id_to_labels": {
            "0": {"class": "UNLABELLED"},
            "7": {"class": "spraycan"},
        },
    }
    assert evidence["tray"]["visible"] is False


def test_semantic_visibility_fails_closed_for_missing_or_colorized_payload() -> None:
    assert (
        _semantic_entity_visibility(
            semantic_frame=None,
            entity_labels={"spraycan": "spraycan"},
        )["spraycan"]["visible"]
        is False
    )
    assert (
        _semantic_entity_visibility(
            semantic_frame={
                "data": np.zeros((12, 12, 4), dtype=np.uint8),
                "info": {"idToLabels": {"7": {"class": "spraycan"}}},
            },
            entity_labels={"spraycan": "spraycan"},
        )["spraycan"]["visible"]
        is False
    )


def test_semantic_labeling_prefers_isaac_6_experimental_labels_api(monkeypatch) -> None:
    calls = []
    module = types.SimpleNamespace(
        add_labels=lambda prim, *, labels, taxonomy: calls.append((prim, labels, taxonomy))
    )

    def fake_import(name: str):
        if name == "isaacsim.core.experimental.utils.semantics":
            return module
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "blueprint_pipeline.nvidia_warehouse_native_camera_canary.importlib.import_module",
        fake_import,
    )
    prim = object()
    assert _add_prim_semantic_label(prim, "spraycan") == "isaacsim_core_experimental_labels_api"
    assert calls == [(prim, ["spraycan"], "class")]


def test_semantic_labeling_falls_back_to_isaacsim_legacy_api(monkeypatch) -> None:
    calls = []
    module = types.SimpleNamespace(
        add_update_semantics=lambda prim, semantic_label, type_label: calls.append(
            (prim, semantic_label, type_label)
        )
    )

    def fake_import(name: str):
        if name == "isaacsim.core.utils.semantics":
            return module
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "blueprint_pipeline.nvidia_warehouse_native_camera_canary.importlib.import_module",
        fake_import,
    )
    prim = object()
    assert _add_prim_semantic_label(prim, "tray") == "isaacsim_core_legacy_semantics_api"
    assert calls == [(prim, "tray", "class")]


def test_semantic_labeling_fails_closed_when_no_runtime_api_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.nvidia_warehouse_native_camera_canary.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    with pytest.raises(ImportError, match="native_semantics_api_unavailable"):
        _add_prim_semantic_label(object(), "franka")


def test_semantic_labeling_authors_every_renderable_descendant() -> None:
    class Prim:
        def __init__(self, name: str, renderable: bool):
            self.name = name
            self.renderable = renderable

    root = Prim("root", False)
    mesh_a = Prim("mesh_a", True)
    mesh_b = Prim("mesh_b", True)
    non_renderable = Prim("joint", False)
    calls = []

    evidence = _author_renderable_semantic_label_tree(
        root_prim=root,
        semantic_label="franka",
        prim_range=lambda _root: [root, mesh_a, non_renderable, mesh_b],
        is_renderable=lambda prim: prim.renderable,
        add_label=lambda prim, label: calls.append((prim.name, label)) or "api",
    )

    assert calls == [("mesh_a", "franka"), ("mesh_b", "franka")]
    assert evidence == {
        "root_label": "franka",
        "renderable_prim_count": 2,
        "api_names": ["api"],
        "root_fallback_used": False,
    }


def test_visibility_summary_requires_rendered_pixels_not_projection() -> None:
    visibility = {
        "initial": {
            "spraycan": {"visible": False},
            "tray": {"visible": True},
            "franka": {"visible": True},
        },
        "commanded": {
            "spraycan": {"visible": True},
            "tray": {"visible": True},
            "franka": {"visible": True},
        },
    }
    assert _summarize_required_entity_visibility(
        view_id="external", visibility_by_phase=visibility
    ) == {"franka": True, "spraycan": False, "tray": True}
    assert _summarize_required_entity_visibility(
        view_id="wrist", visibility_by_phase=visibility
    ) == {"spraycan_at_initial_pose": False}


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


def test_camera_sensor_annotator_frame_normalizes_data_and_info() -> None:
    class WarpLike:
        def numpy(self) -> np.ndarray:
            return np.asarray([[[7]]], dtype=np.uint32)

    class Sensor:
        def get_data(self, annotator: str):
            assert annotator == "semantic_segmentation"
            return WarpLike(), {"idToLabels": {"7": {"class": "spraycan"}}}

    frame = _camera_sensor_annotator_frame(sensor=Sensor(), annotator="semantic_segmentation")

    assert np.array_equal(frame["data"], np.asarray([[[7]]], dtype=np.uint32))
    assert frame["info"] == {"idToLabels": {"7": {"class": "spraycan"}}}


def test_camera_sensor_annotator_frame_fails_closed_when_data_is_not_ready() -> None:
    class Sensor:
        def get_data(self, _annotator: str):
            return None, {}

    with pytest.raises(
        ValueError,
        match="native_camera_annotator_data_unavailable:semantic_segmentation",
    ):
        _camera_sensor_annotator_frame(sensor=Sensor(), annotator="semantic_segmentation")


def test_backend_pose_matrix_uses_usd_row_vector_convention() -> None:
    half_turn = math.radians(90.0) / 2.0
    matrix = _world_pose_matrix_from_backend_pose(
        [1.0, 2.0, 3.0],
        [math.cos(half_turn), 0.0, 0.0, math.sin(half_turn)],
    )

    assert np.asarray([1.0, 0.0, 0.0, 0.0]) @ matrix == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert np.asarray([0.0, 0.0, 0.0, 1.0]) @ matrix == pytest.approx([1.0, 2.0, 3.0, 1.0])


def test_unified_world_pose_query_uses_supported_zero_argument_api() -> None:
    calls: list[str] = []

    class View:
        def get_world_poses(self):
            calls.append("get_world_poses")
            return np.asarray([[0.1, 0.2, 0.3]]), np.asarray([[1.0, 0.0, 0.0, 0.0]])

    matrix = _unified_world_pose_matrix(View())

    assert calls == ["get_world_poses"]
    assert matrix[3, :3] == pytest.approx([0.1, 0.2, 0.3])


def test_camera_pose_backend_congruence_compares_requested_authoring_and_usd() -> None:
    requested = np.eye(4)
    authoring = requested.copy()
    usd = requested.copy()
    authoring[3, 0] = 4e-6
    usd[3, 0] = 7e-6

    evidence = _camera_pose_backend_congruence(
        requested_camera_to_world=requested,
        authoring_camera_to_world=authoring,
        usd_camera_to_world=usd,
    )

    assert evidence["congruent"] is True
    assert evidence["requested_to_authoring_max_abs"] == pytest.approx(4e-6)
    assert evidence["requested_to_usd_max_abs"] == pytest.approx(7e-6)
    assert evidence["authoring_to_usd_max_abs"] == pytest.approx(3e-6)


def test_camera_pose_backend_congruence_fails_closed_on_divergence() -> None:
    requested = np.eye(4)
    usd = requested.copy()
    usd[3, 2] = 0.25

    evidence = _camera_pose_backend_congruence(
        requested_camera_to_world=requested,
        authoring_camera_to_world=requested,
        usd_camera_to_world=usd,
    )

    assert evidence["congruent"] is False
    assert evidence["requested_to_usd_max_abs"] == pytest.approx(0.25)


def test_wrist_camera_render_product_binds_the_same_prepositioned_rtx_authoring_object() -> None:
    source = inspect.getsource(isaac_sim_6_backend)

    camera_definition = source.index("wrist_prim = UsdGeom.Camera.Define")
    world_reset = source.index("world.reset()")
    authoring_object = source.index('wrist_pose_view = camera_authoring["wrist"]')
    calibrated_world_sync = source.index("_synchronize_camera_to_rigid_link", authoring_object)
    render_product = source.index('camera_objects["wrist"] = create_camera_sensor')

    assert (
        camera_definition < world_reset < authoring_object < calibrated_world_sync < render_product
    )
    assert 'wrist_path = "/World/Cameras/Wrist"' in source
    assert 'wrist_pose_view = camera_authoring["wrist"]' in source
    assert "sensor.authoring_object is not authored_camera" in source
    assert "from isaacsim.core.experimental.prims import XformPrim" not in source
    assert "from isaacsim.core.prims import SingleArticulation" in source
    assert "get_link_transforms_after_update_articulations_kinematic" in source
    assert "._xform_prim_view" not in source


def test_backend_uses_isaac_6_experimental_rtx_camera_sensor_for_shared_annotators() -> None:
    source = inspect.getsource(isaac_sim_6_backend)

    assert "from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera" in source
    assert 'camera_prim.GetPrim().ApplyAPI("OmniSensorAPI")' in source
    assert 'wrist_prim.GetPrim().ApplyAPI("OmniSensorAPI")' in source
    assert 'annotators=["rgb", "semantic_segmentation"]' in source
    assert "resolution=(480, 640)" in source
    assert "reset_xform_op_properties=False" in source
    assert 'annotator="rgb"' in source
    assert "native_camera_rtx_authoring_setup_failed" in source
    assert "native_camera_rtx_runtime_setup_failed" in source
    assert "native_camera_rtx_authoring_identity_lost" in source
    assert "native_wrist_camera_pose_backend_divergence" in source
    assert "native_camera_frame_shape_invalid" in source
    assert 'annotators=["rgba", "semantic_segmentation"]' not in source
    assert "resolution=(640, 480)" not in source
    assert "camera.get_current_frame()" not in source
    assert "camera.add_semantic_segmentation_to_frame" not in source


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

    class PublicXFormPrim:
        def set_world_poses(self, **kwargs):
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
        pose_view=PublicXFormPrim(),
        parent_to_world=parent_initial,
        mount_translation_parent=mount_translation,
        mount_orientation_parent_wxyz=mount_orientation,
    )
    commanded = _synchronize_camera_to_rigid_link(
        pose_view=PublicXFormPrim(),
        parent_to_world=parent_commanded,
        mount_translation_parent=mount_translation,
        mount_orientation_parent_wxyz=mount_orientation,
    )

    assert len(calls) == 2
    assert all("usd" not in call for call in calls)
    assert all("camera_axes" not in call for call in calls)
    assert all(np.asarray(call["positions"]).shape == (1, 3) for call in calls)
    assert all(np.asarray(call["orientations"]).shape == (1, 4) for call in calls)
    assert np.linalg.norm(commanded[3, :3] - initial[3, :3]) > 0.001
    assert commanded @ np.linalg.inv(parent_commanded) == pytest.approx(
        initial @ np.linalg.inv(parent_initial)
    )
    roundtrip = _world_pose_matrix_from_backend_pose(
        commanded[3, :3],
        _quaternion_wxyz_from_world_pose_matrix(commanded),
    )
    assert roundtrip == pytest.approx(commanded)


def test_world_camera_sync_requires_public_pose_view_api() -> None:
    class Camera:
        def set_world_pose(self, **_kwargs):
            raise AssertionError("single-pose USD writer must not be used")

    with pytest.raises(ValueError, match="native_wrist_camera_unified_pose_write_api_missing"):
        _synchronize_camera_to_rigid_link(
            pose_view=Camera(),
            parent_to_world=np.eye(4),
            mount_translation_parent=[0.0, 0.1, 0.03],
            mount_orientation_parent_wxyz=[1.0, 0.0, 0.0, 0.0],
        )


def test_world_camera_sync_wraps_public_pose_write_failure_in_safe_code() -> None:
    class PublicXFormPrim:
        def set_world_poses(self, **_kwargs):
            raise AttributeError("opaque runtime detail")

    with pytest.raises(
        ValueError,
        match="native_wrist_camera_unified_pose_write_failed:AttributeError",
    ):
        _synchronize_camera_to_rigid_link(
            pose_view=PublicXFormPrim(),
            parent_to_world=np.eye(4),
            mount_translation_parent=[0.0, 0.1, 0.03],
            mount_orientation_parent_wxyz=[1.0, 0.0, 0.0, 0.0],
        )


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


def test_wrist_mount_world_clearance_resolves_to_one_fixed_parent_mount() -> None:
    quaternion, evidence = _rigid_wrist_mount_from_initial_task_framing(
        parent_to_world=np.eye(4),
        mount_translation_parent=[0.0, 0.1, 0.03],
        target_world_points={"spraycan": [0.0, 1.0, 0.0]},
        camera_eye_world_offset=[0.0, 0.0, 0.25],
    )

    assert evidence["base_camera_eye_world_m"] == pytest.approx([0.0, 0.1, 0.03])
    assert evidence["camera_eye_world_m"] == pytest.approx([0.0, 0.1, 0.28])
    assert evidence["resolved_mount_translation_parent_m"] == pytest.approx([0.0, 0.1, 0.28])
    assert evidence["camera_eye_world_offset_m"] == [0.0, 0.0, 0.25]
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)


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


def test_native_camera_canary_accepts_three_distinct_synchronized_views(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path, three_views=True)

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "three-view-canary",
        backend=_backend,
    )

    assert result["status"] == "passed"
    assert result["assessment"]["required_views"] == ["external", "external_2", "wrist"]


def test_native_camera_canary_rejects_duplicated_second_external_frame(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path, three_views=True)

    def duplicated_backend(**kwargs):
        result = _backend(**kwargs)
        for phase in ("initial", "commanded"):
            source = Path(result["views"]["external"][f"{phase}_frame_path"])
            target = Path(result["views"]["external_2"][f"{phase}_frame_path"])
            target.write_bytes(source.read_bytes())
        return result

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "duplicate-view-canary",
        backend=duplicated_backend,
    )

    assert result["status"] == "failed"
    assert "native_camera_external_pair_frame_not_distinct:initial" in result["blockers"]


def test_native_camera_canary_rejects_projected_but_not_rendered_task_object(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def occluded_backend(**kwargs):
        value = _backend(**kwargs)
        value["views"]["wrist"]["required_entities_visible_pixels"] = {
            "spraycan_at_initial_pose": False
        }
        value["views"]["wrist"]["required_entities_visible_pixels_by_phase"] = {
            "initial": {
                "spraycan": {
                    "visible_pixel_count": 0,
                    "minimum_visible_pixel_count": 64,
                    "visible": False,
                    "render_derived": True,
                }
            }
        }
        return value

    result = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "result",
        backend=occluded_backend,
    )

    assert result["status"] == "failed"
    assert (
        result["assessment"]["views"]["wrist"]["required_entities_projected_in_frame"]["spraycan"]
        is True
    )
    assert "native_camera_required_entity_projection_failed:wrist" not in result["blockers"]
    assert "native_camera_required_entity_visibility_failed:wrist" in result["blockers"]


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


def test_native_camera_canary_accepts_float32_pose_roundoff_but_rejects_real_slip(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    _spec(spec_path)

    def rounded_backend(**kwargs):
        value = _backend(**kwargs)
        value["wrist_camera_local_transform_delta"] = 5e-7
        return value

    rounded = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "rounded",
        backend=rounded_backend,
    )
    assert rounded["status"] == "passed"

    def slipped_backend(**kwargs):
        value = _backend(**kwargs)
        value["wrist_camera_local_transform_delta"] = 2e-6
        return value

    slipped = run_native_camera_canary(
        spec_path=spec_path,
        assets_root=tmp_path / "assets",
        output_dir=tmp_path / "slipped",
        backend=slipped_backend,
    )
    assert slipped["status"] == "failed"
    assert "native_wrist_camera_mount_not_rigid" in slipped["blockers"]


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
