"""CPU rehearsal uses real USD scopes and production lossless native AOV writer."""
from dataclasses import replace
from types import SimpleNamespace as NS
import json
import sys

import numpy as np
import pytest
import torch
from pxr import Usd, UsdGeom, UsdLux

from blueprint_pipeline import native_task_composition_diagnostic as diagnostic
from blueprint_pipeline import native_task_composition_worker as worker
from blueprint_pipeline import adp009d_isaac_runtime as aov


def request():
    return diagnostic.seal({'schema_version': diagnostic.REQUEST_SCHEMA, 'passes': list(diagnostic.PASSES),
        'camera_role': 'external', 'target_semantic_class': 'task_support', 'policy_queries_permitted': 0,
        'physics_steps_between_passes_permitted': 0, 'source_asset_mutation_permitted': False,
        'render_refresh_count': 2}, 'request_digest')


@pytest.fixture
def native(tmp_path, monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    root = '/World/envs/env_0'
    UsdGeom.Xform.Define(stage, root)
    field = stage.DefinePrim(root + '/scene_appearance/Gaussians', 'ParticleField3DGaussianSplat')
    # Schema is not installed on the CPU host; author the common Imageable attr.
    field.CreateAttribute('visibility', __import__('pxr').Sdf.ValueTypeNames.Token).Set('inherited')
    for name in ('task_support', 'task_object', 'Robot', 'scene_collision'):
        UsdGeom.Xform.Define(stage, root + '/' + name)
        UsdGeom.Cube.Define(stage, root + '/' + name + '/geometry')
    UsdGeom.Imageable(stage.GetPrimAtPath(root + '/scene_collision')).MakeInvisible()
    light = UsdLux.DomeLight.Define(stage, root + '/task_support/embedded_light')
    light.CreateIntensityAttr(2.)
    before_layer = stage.GetRootLayer().ExportToString()
    config = {'/rtx/rendermode': 'RayTracedLighting'}
    monkeypatch.setitem(sys.modules, 'carb', NS(settings=NS(get_settings=lambda: NS(get=config.get))))
    monkeypatch.setattr(aov, '_camera_prim_diagnostics', lambda _: {'fixture_native_camera': True})
    pose = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]])
    data = NS(output={}, info={'semantic_segmentation': {'idToLabels': {'7': {'class': 'task_support'}, '9': {'class': 'scene_appearance'}}}},
        intrinsic_matrices=torch.tensor([[[10.,0.,5.5],[0.,10.,3.5],[0.,0.,1.]]]),
        pos_w=torch.tensor([[0.,0.,2.]]), quat_w_opengl=torch.tensor([[0.,0.,0.,1.]]))
    camera = NS(data=data, frame=torch.tensor([0]), prim_path=root + '/camera')

    def visible(path):
        return UsdGeom.Imageable(stage.GetPrimAtPath(path)).ComputeVisibility() != 'invisible'

    def update(*args, **kwargs):
        assert kwargs == {'force_recompute': True}
        appearance = stage.GetPrimAtPath(root + '/scene_appearance/Gaussians').GetAttribute('visibility').Get() != 'invisible'
        meshes = visible(root + '/task_support/geometry')
        rgb = torch.full((1,8,12,3), 80, dtype=torch.uint8)
        rgb[..., 2] = 180
        if appearance:
            rgb[:,2:4,3:5,:] = torch.tensor([110,70,40], dtype=torch.uint8)
        depth = torch.full((1,8,12,1), 1.0 if meshes else 1.2, dtype=torch.float32)
        semantic = torch.full((1,8,12,1), 7 if meshes else 9, dtype=torch.int64)
        data.output = {'rgb': rgb, 'distance_to_camera': depth, 'semantic_segmentation': semantic}
        camera.frame += 1
    camera.update = update
    update(force_recompute=True)

    class Scene(dict):
        env_prim_paths = [root]
    scene = Scene(external_camera=camera, robot=NS(data=NS(root_pose_w=pose.clone(), joint_pos=torch.zeros(1,7))))
    for name in ('scene_appearance','scene_collision','task_object','task_support'):
        scene[name] = NS(data=NS(root_pose_w=pose.clone()))
    sim = NS(current_time=0., current_time_step_index=0, render=lambda: None)
    env = NS(scene=scene, sim=sim)
    env.unwrapped = env
    built = NS(env=env, plan={'objects':[{'semantic_role':'scene_appearance','prim_path':'{ENV_REGEX_NS}/scene_appearance'}]},
        camera_scene_names={'external':'external_camera'}, scene_asset_names={k:k for k in ('scene_appearance','scene_collision','task_object','task_support')})
    adapters = worker.make_native_adapters(built=built, stage=stage, request=request())
    return NS(stage=stage, root=root, before_layer=before_layer, adapters=adapters, camera=camera,
              config=config, sim=sim, output=tmp_path/'result', scene=scene)


def test_three_native_passes_retain_exact_aovs_calibration_and_restore_light_preserving_visibility(native):
    result = diagnostic.run_composition_diagnostic(request(), output_root=native.output, adapters=native.adapters)
    assert result['status'] == 'captured', result
    assert [row['pass'] for row in result['passes']] == list(diagnostic.PASSES)
    assert result['visibility_restored'] is True
    assert result['pixel_cause_proven'] is False
    comparison = result['pixel_comparison']
    assert comparison['target_pixel_count'] == 96
    assert comparison['rgb_changed_target_pixel_count'] == 4
    assert comparison['changed_pixels_appearance_distance_deeper'] == 4
    for row in result['passes']:
        assert len(row['artifacts']) == 3
        for rec in row['artifacts']:
            assert diagnostic.file_record(native.output / row['pass'] / rec['relative_path']) == {k:rec[k] for k in ('sha256','size_bytes')}
        assert row['camera']['semantic_segmentation']['id_to_labels']['idToLabels']['7']['class']=='task_support'
    assert native.stage.GetRootLayer().ExportToString() == native.before_layer
    assert UsdGeom.Imageable(native.stage.GetPrimAtPath(native.root+'/scene_collision')).ComputeVisibility()=='invisible'
    assert native.stage.GetPrimAtPath(native.root+'/task_support/embedded_light').GetAttribute('inputs:intensity').Get()==2
    assert native.sim.current_time_step_index == 0


@pytest.mark.parametrize('failure', ['capture', 'pose', 'settings', 'stale', 'physics'])
def test_scope_restores_on_capture_failure_or_fixed_state_change(native, failure):
    original = native.adapters.render_and_capture
    def capture(label, output):
        if label == 'appearance_only':
            if failure == 'capture':
                raise RuntimeError('fixture capture failure')
            if failure == 'pose':
                native.scene['robot'].data.root_pose_w[0,0] += .01
            if failure == 'settings':
                native.config['/rtx/rendermode'] = 'different'
            if failure == 'physics':
                native.sim.current_time_step_index += 1
        return original(label, output)
    adapters = replace(native.adapters, render_and_capture=capture)
    if failure == 'stale':
        adapters = replace(adapters, sensor_generation=lambda: 0)
    before = adapters.snapshot_visibility()
    result = diagnostic.run_composition_diagnostic(request(), output_root=native.output, adapters=adapters)
    assert result['status'] == 'blocked'
    assert result['visibility_restored']
    assert adapters.snapshot_visibility() == before
    assert native.stage.GetRootLayer().ExportToString() == native.before_layer
    if failure == 'physics':
        assert result['physics_steps_between_passes'] == 1


def test_invalid_native_depth_is_retained_as_gap_instead_of_fabricated(native):
    original = native.camera.update
    def invalid(*args, **kwargs):
        original(*args, **kwargs)
        native.camera.data.output['distance_to_camera'].fill_(float('inf'))
    native.camera.update = invalid
    result = diagnostic.run_composition_diagnostic(request(), output_root=native.output, adapters=native.adapters)
    assert result['status'] == 'captured', result
    assert result['metric_depth_usable_for_occlusion'] is False
    assert result['pixel_comparison']['finite_paired_distance_target_pixel_count'] == 0
    for row in result['passes']:
        array = np.load(native.output/row['pass']/row['camera']['metric_depth']['path'])
        assert np.isinf(array).all()


def test_capture_digest_mismatch_refuses_and_restores(native):
    original = native.adapters.render_and_capture
    def corrupt(label, output):
        row = original(label, output)
        row['rgb_png']['sha256'] = '0'*64
        return row
    before = native.adapters.snapshot_visibility()
    result = diagnostic.run_composition_diagnostic(request(), output_root=native.output,
        adapters=replace(native.adapters, render_and_capture=corrupt))
    assert result['status'] == 'blocked'
    assert 'aov_digest_mismatch' in result['blockers'][0]
    assert native.adapters.snapshot_visibility() == before


def test_fresh_output_required_so_prior_native_evidence_cannot_be_overwritten(native):
    native.output.mkdir()
    proof = native.output/'retained.json'
    proof.write_text(json.dumps({'unchanged':True}))
    with pytest.raises(diagnostic.CompositionDiagnosticError, match='fresh'):
        diagnostic.run_composition_diagnostic(request(), output_root=native.output, adapters=native.adapters)
    assert json.loads(proof.read_text()) == {'unchanged':True}


def test_composition_payload_has_complete_import_closure_without_loading_policy(tmp_path):
    import subprocess
    import shutil
    from pathlib import Path
    from blueprint_pipeline.native_task_composition_bundle import composition_runtime_sources
    package=tmp_path/'blueprint_pipeline'
    package.mkdir()
    (package/'__init__.py').write_text('')
    for source in composition_runtime_sources():
        shutil.copyfile(source,package/source.name)
    probe="import sys; sys.path.insert(0, sys.argv[1]); import blueprint_pipeline.native_task_composition_worker; from blueprint_pipeline.adp009d_isaac_runtime import _save_camera; assert not any('groot_n17' in k for k in sys.modules)"
    result=subprocess.run([sys.executable,'-I','-c',probe,str(tmp_path)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert Path(worker.__file__).name=='native_task_composition_worker.py'
