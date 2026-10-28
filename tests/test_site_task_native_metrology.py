"""Offline native-handle stubs prove adapter/refusals, never engine acceptance."""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

from blueprint_pipeline import site_task_native_metrology as native
from blueprint_pipeline.adp009d_isaac_episode_adapter import IsaacEpisodeAdapter
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_runtime import NativeTaskArenaEnvironment


class Scene(dict):
    env_prim_paths = ["/World/envs/env_0"]
    env_regex_ns = "/World/envs/env_.*"


class Prim:
    def __init__(self, path):
        self.path = path

    def IsValid(self):
        return True

    def GetPath(self):
        return self.path


class Adapter(IsaacEpisodeAdapter):
    def __init__(self):
        pass

    def read_control_observation_metadata(self):
        return {"timestamp_ns": 0, "simulation_time_s": 0.,
                "calibrations": {"external": {"camera_model": "pinhole", "resolution": [5, 5]}},
                "synchronizations": {"external": {"host_bytes_ready": True,
                    "render": {"reset_generation": 3, "control_step_index": 0}}}}


def fixture(monkeypatch):
    k = np.array([[2., 0, 2], [0, 2., 2], [0, 0, 1.]])
    v, u = np.indices((5, 5))
    ranges = 2 * np.sqrt(((u - 2) / 2) ** 2 + ((v - 2) / 2) ** 2 + 1)
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[2, 1:4] = [255, 0, 0]
    camera = NS(frame=np.array([1]), data=NS(output={"rgb": rgb[None],
        "distance_to_camera": ranges[None, ..., None]}, intrinsic_matrices=k[None]))
    view = NS(get_masses=lambda: np.array([[2.5]]),
              get_material_properties=lambda: np.array([[[.7, .5, .1]]]))
    scene = Scene(object=NS(cfg=NS(prim_path="{ENV_REGEX_NS}/object"), root_view=view,
                           data=NS(root_pose_w=np.array([[0., 0, .7, 0, 0, 0, 1]]))),
                  floor=NS(cfg=NS(prim_path="{ENV_REGEX_NS}/floor")), camera=camera)
    plan = {"schema_version": "native_task_arena_scene_plan.v1", "task_kind": "rigid_pick_place",
            "robot": {"robot_id": "franka_panda"}, "task_spec": {"requested_mass_kg": 9000}}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    built = NativeTaskArenaEnvironment(env=NS(unwrapped=NS(scene=scene)), cfg=NS(), plan=plan,
        scene_asset_names={"task_object": "object", "floor": "floor"},
        contact_sensor_names={}, camera_scene_names={"external": "camera"})
    selectors = {"schema_version": native.SELECTOR_SCHEMA, "scene_plan_digest": plan["plan_digest"],
        "geometry_points": {"start": {"asset_role": "task_object", "corner": [0, 0, 0]},
                            "end": {"asset_role": "task_object", "corner": [1, 0, 0]}},
        "floor_plane": {"asset_role": "floor"},
        "collision_bounds": {"object": {"asset_role": "task_object"}},
        "collision_rays": {"front": {"asset_role": "task_object", "origin_anchor": "start",
            "direction": [1., 0, 0], "maximum_distance_m": 2.}},
        "rigid_bodies": {"object": {"asset_role": "task_object"}},
        "cameras": {"policy": {"camera_role": "external", "appearance_feature": {
            "kind": "rgb_band_endpoints", "lower_rgb": [255, 0, 0], "upper_rgb": [255, 0, 0],
            "scan_axis": "u", "scan_line": 2}}}}
    stage = NS(GetPrimAtPath=lambda path: Prim(path))
    monkeypatch.setattr(native, "_stage", lambda: stage)
    monkeypatch.setattr(native, "_bounds", lambda prim, collision: (
        np.array([0., 0, .7]), np.array([.2 if collision else .4, .1, .8])))
    monkeypatch.setattr(native, "_floor_plane", lambda prim: [0., 0, 1., 0.])
    monkeypatch.setattr(native, "_raycast", lambda origin, direction, distance: {
        "hit": True, "position": [.1, 0, .7], "collision": "/World/envs/env_0/object/mesh"})
    return built, selectors, camera


def test_off_axis_range_is_converted_with_actual_intrinsics_without_mutation():
    k = np.array([[2., 0, 2], [0, 2., 2], [0, 0, 1.]])
    v, u = np.indices((5, 5))
    ranges = 3 * np.sqrt(((u - 2) / 2) ** 2 + ((v - 2) / 2) ** 2 + 1)
    before = ranges.copy()
    assert ranges[0, 0] > ranges[2, 2]
    assert np.allclose(native.optical_axis_depth(ranges, k), 3)
    assert np.array_equal(ranges, before)


@pytest.mark.parametrize("case", ["millimetres_label_only", "negative", "skew", "bad_matrix"])
def test_depth_conversion_refuses_invalid_calibration_or_samples(case):
    distance, k = np.ones((3, 3)), np.eye(3)
    if case == "negative":
        distance[0, 0] = -1
    elif case == "skew":
        k[0, 1] = .1
    elif case == "bad_matrix":
        k[2, 2] = 2
    else:
        distance = distance.astype(np.int32)
    with pytest.raises(native.NativeMetrologyError, match="depth_calibration_invalid"):
        native.optical_axis_depth(distance, k)


def test_six_readbacks_keep_representations_and_native_values_distinct(monkeypatch):
    built, selectors, camera = fixture(monkeypatch)
    before = camera.data.output["distance_to_camera"].copy()
    capture = built.read_site_task_metrology(selectors=selectors, episode_adapter=Adapter())
    report, values = capture.report, capture.report["values"]
    assert set(report["dimensions"]) == {"scale", "registration", "collision", "task_state", "observation", "task_dynamics"}
    assert report["complete_native_dimensions"] is True
    assert values["geometry_points_m"]["end"][0] == .4
    assert values["collision_bounds_m"]["object"]["max"][0] == .2
    assert values["rigid_body_masses_kg"]["object"] == 2.5
    assert values["rigid_body_masses_kg"]["object"] != built.plan["task_spec"]["requested_mass_kg"]
    assert values["rendered_rgbd"]["policy"]["pixels"] == [[1, 2], [3, 2]]
    assert np.allclose(capture.rgbd["policy"]["depth_m"], 2)
    assert np.array_equal(camera.data.output["distance_to_camera"], before)
    assert report["runtime_open_proven"] is report["qualification_claimed"] is False
    assert report["model_open_status"]["status"] == "unproven"


def test_rgb_only_shift_changes_observed_association_with_collision_and_depth_fixed(monkeypatch):
    built, selectors, camera = fixture(monkeypatch)
    first = native.read_rigid_native_metrology(built, selectors=selectors, episode_adapter=Adapter())
    camera.data.output["rgb"][:] = 0
    camera.data.output["rgb"][0, 2, 2:5] = [255, 0, 0]
    second = native.read_rigid_native_metrology(built, selectors=selectors, episode_adapter=Adapter())
    assert first.report["values"]["collision_bounds_m"] == second.report["values"]["collision_bounds_m"]
    assert np.array_equal(first.rgbd["policy"]["depth_m"], second.rgbd["policy"]["depth_m"])
    assert first.report["values"]["rendered_rgbd"]["policy"]["pixels"] == [[1, 2], [3, 2]]
    assert second.report["values"]["rendered_rgbd"]["policy"]["pixels"] == [[2, 2], [4, 2]]
    # A changed observed association is evidence, never an invented fidelity verdict.
    assert second.report["qualification_claimed"] is False


def test_missing_native_mechanisms_are_typed_unproven_without_remote_error_content(monkeypatch):
    built, selectors, _ = fixture(monkeypatch)
    monkeypatch.setattr(native, "_stage", lambda: (_ for _ in ()).throw(ImportError("private remote body")))
    built.env.unwrapped.scene["object"].root_view.get_masses = lambda: (_ for _ in ()).throw(RuntimeError("secret-token"))
    capture = native.read_rigid_native_metrology(built, selectors=selectors)
    assert capture.report["complete_native_dimensions"] is False
    assert all(row["status"] == "unproven" for row in capture.report["dimensions"].values())
    assert "private remote" not in json.dumps(capture.report)
    assert "secret-token" not in json.dumps(capture.report)


def test_mapping_snapshot_or_scene_plan_drift_refuses_before_native_read(monkeypatch):
    built, selectors, _ = fixture(monkeypatch)
    monkeypatch.setattr(native, "_stage", lambda: pytest.fail("native reader must not run"))
    with pytest.raises(native.NativeMetrologyError, match="native_environment_handle_required"):
        native.read_rigid_native_metrology({"source_kind": "native_engine_readback"}, selectors=selectors)
    built.plan["task_spec"]["changed"] = True
    with pytest.raises(native.NativeMetrologyError, match="selector_binding_invalid"):
        native.read_rigid_native_metrology(built, selectors=selectors)


def test_foreign_closest_collider_cannot_supply_requested_scene_clearance(monkeypatch):
    built, selectors, _ = fixture(monkeypatch)
    monkeypatch.setattr(native, "_raycast", lambda *_: {"hit": True, "position": [1, 2, 3],
                                                       "collision": "/foreign/robot"})
    report = native.read_rigid_native_metrology(built, selectors=selectors).report
    assert report["dimensions"]["collision"] == {"status": "unproven", "reason": "native_metrology_collision_ray_unexpected_collider"}


def test_model_path_and_loaded_layer_identity_never_manufacture_byte_open_proof(tmp_path, monkeypatch):
    from tests.test_blueprint_robot_model_delivery import fixture as operator_fixture
    profile, _, _ = operator_fixture(tmp_path, monkeypatch)
    selected = native.prepare_default_droid_model(operator_profile=profile, target_directory=tmp_path / "staged")
    built, _, _ = fixture(monkeypatch)
    built.env.unwrapped.scene["robot"] = NS(cfg=NS(spawn=NS(usd_path=str(selected.model_path))), root_view=NS(count=1))
    monkeypatch.setattr(native, "_stage", lambda: NS(GetUsedLayers=lambda: [NS(realPath=str(selected.model_path))]))
    observation = native.read_default_droid_model_use(built, selected)
    assert observation["native_root_layer_observed"] is True
    assert observation["current_dependency_bytes_verified"] is True
    assert observation["runtime_open_proven"] is False
    assert observation["source_import_digest"] == "sha256:" + "e" * 64
    (selected.model_path.parent / "textures/body.png").write_bytes(b"changed dependency")
    with pytest.raises(ValueError, match="runtime_bytes_changed"):
        native.read_default_droid_model_use(built, selected)


def test_droid_binding_changes_only_verified_spawn_path_and_rechecks_generation(tmp_path, monkeypatch):
    from tests.test_blueprint_robot_model_delivery import fixture as operator_fixture
    profile, _, _ = operator_fixture(tmp_path, monkeypatch)
    selected = native.prepare_default_droid_model(operator_profile=profile, target_directory=tmp_path / "staged")
    embodiment = NS(scene_config=NS(robot=NS(spawn=NS(usd_path="provider-default"), control="unchanged")))
    native.bind_selected_droid_model(embodiment, selected)
    assert embodiment.scene_config.robot.spawn.usd_path == str(selected.model_path)
    assert embodiment.scene_config.robot.control == "unchanged"
    selected.delivery["source_import_digest"] = "sha256:" + "f" * 64
    with pytest.raises(native.NativeMetrologyError, match="delivery_changed"):
        native.bind_selected_droid_model(embodiment, selected)
    with pytest.raises(native.NativeMetrologyError, match="verified_default_model_required"):
        native.bind_selected_droid_model(embodiment, {"runtime_open_proven": True})


def test_retained_capture_keeps_original_range_and_derived_depth_hashes(tmp_path, monkeypatch):
    built, selectors, _ = fixture(monkeypatch)
    capture = native.read_rigid_native_metrology(built, selectors=selectors, episode_adapter=Adapter())
    report = native.retain_native_metrology(capture, tmp_path / "capture")
    record = report["render_artifacts"]["policy"]
    data = (tmp_path / "capture" / record["path"]).read_bytes()
    assert record["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()
    with np.load(tmp_path / "capture" / record["path"], allow_pickle=False) as frame:
        assert set(frame.files) == {"rgb", "depth_m", "intrinsics", "original_depth", "valid_depth"}
        assert frame["original_depth"][0, 0] > frame["depth_m"][0, 0]
    assert report["capture_digest"] == canonical_digest(report, digest_field="capture_digest")
    assert json.loads((tmp_path / "capture/capture.json").read_text()) == report
    with pytest.raises(native.NativeMetrologyError, match="capture_conflict"):
        native.retain_native_metrology(capture, tmp_path / "capture")


def test_retention_refuses_unsafe_render_names_before_writing(tmp_path):
    capture = native.NativeMetrologyCapture({}, {"../escape": {"rgb": np.zeros((1, 1, 3), dtype=np.uint8)}})
    with pytest.raises(native.NativeMetrologyError, match="identity_invalid"):
        native.retain_native_metrology(capture, tmp_path / "capture")
    assert not (tmp_path / "capture").exists()


def test_retained_directory_replacement_cannot_escape_anchored_writer(tmp_path, monkeypatch):
    built, selectors, _ = fixture(monkeypatch)
    capture = native.read_rigid_native_metrology(built, selectors=selectors, episode_adapter=Adapter())
    target, moved, outside = tmp_path / "capture", tmp_path / "moved", tmp_path / "outside"
    outside.mkdir()
    (outside / "policy.npz").write_bytes(b"outside-original")
    original = os.open
    def swap(name, flags, *args, **kwargs):
        if name == "policy.npz":
            target.rename(moved)
            target.symlink_to(outside, target_is_directory=True)
        return original(name, flags, *args, **kwargs)
    monkeypatch.setattr(native.os, "open", swap)
    with pytest.raises(native.NativeMetrologyError, match="directory_changed"):
        native.retain_native_metrology(capture, target)
    assert (outside / "policy.npz").read_bytes() == b"outside-original"
    assert not (outside / "capture.json").exists()


def test_native_background_missing_depth_is_retained_but_feature_depth_must_exist(monkeypatch):
    built, selectors, camera = fixture(monkeypatch)
    camera.data.output["distance_to_camera"][0, 0, 0, 0] = np.inf
    capture = native.read_rigid_native_metrology(built, selectors=selectors, episode_adapter=Adapter())
    assert capture.report["dimensions"]["observation"]["status"] == "observed"
    assert np.isinf(capture.rgbd["policy"]["original_depth"][0, 0])
    assert capture.rgbd["policy"]["depth_m"][0, 0] == 0
    assert not capture.rgbd["policy"]["valid_depth"][0, 0]
    camera.data.output["distance_to_camera"][0, 2, 1, 0] = np.inf
    refused = native.read_rigid_native_metrology(built, selectors=selectors, episode_adapter=Adapter())
    assert refused.report["dimensions"]["observation"] == {
        "status": "unproven", "reason": "native_metrology_native_depth_feature_missing"}


def test_existing_droid_factory_hook_preserves_reset_and_sets_only_selected_asset(tmp_path, monkeypatch):
    import sys
    from tests import test_native_task_arena_runtime as runtime_fixture
    from tests.test_blueprint_robot_model_delivery import fixture as operator_fixture
    from blueprint_pipeline.native_task_robot_registry import build_native_robot_embodiment
    runtime_fixture._install_fake_native_runtime(monkeypatch)
    class Droid(runtime_fixture._Embodiment):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scene_config.robot.spawn.usd_path = "provider-default"
    monkeypatch.setattr(sys.modules["isaaclab_arena.embodiments.droid.droid"],
                        "DroidAbsoluteJointPositionEmbodiment", Droid)
    profile, _, _ = operator_fixture(tmp_path, monkeypatch)
    selected = native.prepare_default_droid_model(operator_profile=profile, target_directory=tmp_path / "staged")
    robot = runtime_fixture._sealed_scene_plan()["robot"]
    embodiment = build_native_robot_embodiment(robot, enable_cameras=True,
        pose_class=runtime_fixture._Pose, default_robot_model=selected)
    assert embodiment.scene_config.robot.spawn.usd_path == str(selected.model_path)
    assert embodiment.scene_config.stand is None
    assert embodiment.event_config.init_franka_arm_pose.params["default_pose"] == list(robot["joint_reset_positions_rad"].values())
    assert embodiment.event_config.randomize_franka_joint_state.params == {"mean": 0., "std": 0.}


def test_serialized_default_model_cannot_reach_native_builder(monkeypatch):
    from blueprint_pipeline.native_task_arena_runtime import NativeTaskArenaRuntimeError, build_native_task_arena_environment
    from tests.test_native_task_arena_runtime import _sealed_scene_plan
    monkeypatch.setattr("blueprint_pipeline.native_task_arena_runtime._resolve_portable_assets",
                        lambda *_a, **_k: pytest.fail("asset/native setup must not run"))
    with pytest.raises(NativeTaskArenaRuntimeError, match="default_model_binding_invalid"):
        build_native_task_arena_environment(_sealed_scene_plan(), default_robot_model={"runtime_open_proven": True})


def test_reset_generation_change_cannot_make_complete_native_capture(monkeypatch):
    built, selectors, _ = fixture(monkeypatch)
    class Changed(Adapter):
        def __init__(self):
            self.calls = 0
        def read_control_observation_metadata(self):
            self.calls += 1
            metadata = super().read_control_observation_metadata()
            metadata["synchronizations"]["external"]["render"]["reset_generation"] = self.calls
            return metadata
    capture = native.read_rigid_native_metrology(built, selectors=selectors, episode_adapter=Changed())
    assert capture.report["complete_native_dimensions"] is False
    assert capture.report["reset_render_generation"] == {"status": "unproven",
        "reason": "native_reset_render_generation_missing_or_changed"}
    assert capture.report["runtime_open_proven"] is False
