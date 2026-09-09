"""Check pinned reset call order; native pose freshness requires the frame-view binding."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from blueprint_pipeline.native_task_arena_runtime import build_native_task_arena_environment
from tests.test_native_task_arena_runtime import _install_fake_native_runtime, _sealed_scene_plan


def test_reset_requests_render_after_joint_write_and_forward_without_physics(monkeypatch):
    _install_fake_native_runtime(monkeypatch)
    cfg = build_native_task_arena_environment(_sealed_scene_plan()).cfg
    old = np.eye(4)
    target = old.copy()
    target[:3, 3] = [0.2, -0.1, 0.3]
    source = json.loads((Path(__file__).parent/'fixtures/isaaclab_reset/manager_reset_method.json').read_text())
    namespace = {}
    exec(compile('from __future__ import annotations\n'+source['reset_method'], source['source_path'], 'exec'), namespace)

    def replay(config):
        calls = []
        state = {'joints': 'old', 'camera': old.copy()}
        def reset(_ids):
            calls.append('reset')
            state['joints'] = 'task_aligned'
        def render():
            calls.append('render')
            assert state['joints'] == 'task_aligned'
            state['camera'] = target.copy()
        env = SimpleNamespace(
            cfg=config, has_rtx_sensors=True, extras={},
            recorder_manager=SimpleNamespace(record_pre_reset=lambda _:None, record_post_reset=lambda _:None),
            _reset_idx=reset,
            scene=SimpleNamespace(write_data_to_sim=lambda:calls.append('write')),
            sim=SimpleNamespace(forward=lambda:calls.append('forward'),render=render),
            observation_manager=SimpleNamespace(compute=lambda **_:state['camera'].copy()),
        )
        observed, _ = namespace['reset'](env, env_ids=[0])
        return observed,calls
    stale, old_calls = replay(SimpleNamespace(num_rerenders_on_reset=0,wait_for_textures=False))
    np.testing.assert_array_equal(stale,old)
    assert old_calls == ['reset','write','forward']
    cfg.wait_for_textures = False
    fresh, calls = replay(cfg)
    np.testing.assert_array_equal(fresh,target)
    assert not np.allclose(fresh,old,atol=.001)
    assert calls == ['reset','write','forward','render']


def test_camera_disabled_environment_does_not_render_on_reset(monkeypatch):
    _install_fake_native_runtime(monkeypatch)
    built=build_native_task_arena_environment(_sealed_scene_plan(),enable_cameras=False)
    assert built.cfg.num_rerenders_on_reset == 0
