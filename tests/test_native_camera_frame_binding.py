"""Replay pinned PhysX producer and camera-view consumer without CUDA/Isaac."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from blueprint_pipeline.native_task_arena_runtime import build_native_task_arena_environment
from tests.test_native_task_arena_runtime import _install_fake_native_runtime, _sealed_scene_plan


def test_camera_frame_reader_uses_physics_fabric_after_actual_runtime_configuration(monkeypatch):
    _install_fake_native_runtime(monkeypatch)
    settings_class = sys.modules['isaaclab.app.settings_manager'].SettingsManager
    settings = settings_class.instance()
    sim = SimpleNamespace(cfg=SimpleNamespace(use_fabric=True), set_setting=settings.set)
    interface = SimpleNamespace(update=lambda *args: None)
    extension = ModuleType('omni.physxfabric')
    extension.get_physx_fabric_interface = lambda: interface
    monkeypatch.setitem(sys.modules, 'omni.physxfabric', extension)
    extensions = SimpleNamespace(is_extension_enabled=lambda _: True)
    upstream = {'SettingsManager': settings_class,
        'PhysicsManager': SimpleNamespace(_sim=sim, _cfg=object()),
        'omni': SimpleNamespace(kit=SimpleNamespace(app=SimpleNamespace(
            get_app=lambda: SimpleNamespace(get_extension_manager=lambda: extensions)))),
        'UsdFrameView': lambda *args, **kwargs: SimpleNamespace(get_world_poses=lambda _: ('stale_usd', 'old_rotation')),
        'wp': SimpleNamespace(launch=lambda **kwargs: None, synchronize=lambda: None),
        'fabric_utils': SimpleNamespace(decompose_fabric_transformation_matrix_to_warp_arrays=object()),
    }
    fixture = json.loads((Path(__file__).parent / 'fixtures/isaaclab_camera_frame_binding/methods.json').read_text())
    for method in fixture['methods']:
        exec(compile('from __future__ import annotations\n' + method['source'], method['source_path'], 'exec'), upstream)
    manager = type('PinnedPhysicsManager', (), {'_load_fabric': upstream['_load_fabric']})
    view_class = type('PinnedFrameView', (), {name: upstream[name] for name in ('__init__', 'get_world_poses')})
    manager._load_fabric()
    assert settings.get('/isaaclab/fabric_enabled') is True
    assert settings.get('/physics/updateToUsd') is False
    old_view = view_class('/Robot/Gripper/wrist_camera')
    assert old_view.get_world_poses() == ('stale_usd', 'old_rotation')

    built = build_native_task_arena_environment(_sealed_scene_plan())
    assert built.cfg.sim.use_fabric is True
    assert settings.get('/physics/fabricEnabled') is True
    view = view_class('/Robot/Gripper/wrist_camera')
    assert view._use_fabric is True
    # External GPU edges stand in for measured native poses; the upstream
    # branch and renderer-facing pose source above are executed unchanged.
    view._fabric_initialized = view._fabric_usd_sync_done = True
    view._resolve_indices_wp = lambda _: SimpleNamespace(shape=(1,))
    for name in ('_fabric_positions_buf', '_fabric_orientations_buf', '_fabric_world_matrices', '_fabric_dummy_buffer', '_view_to_fabric'):
        setattr(view, name, object())
    view._fabric_device = 'cuda:0'
    view._fabric_positions_ta, view._fabric_orientations_ta = 'live_native_pose', 'live_native_rotation'
    assert view.get_world_poses() == ('live_native_pose', 'live_native_rotation')
    # Reinitializing physics (a rebuilt cell) keeps both sides in agreement.
    manager._load_fabric()
    assert view_class('/Robot/Gripper/wrist_camera')._use_fabric is True
