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
from tests.native_task_pinned_clock_fixture import PinnedSimulationClock


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
    sim = PinnedSimulationClock()
    env = NS(scene=scene, sim=sim)
    env.unwrapped = env
    built = NS(env=env, plan={'objects':[{'semantic_role':'scene_appearance','prim_path':'{ENV_REGEX_NS}/scene_appearance'}]},
        camera_scene_names={'external':'external_camera'}, scene_asset_names={k:k for k in ('scene_appearance','scene_collision','task_object','task_support')})
    adapters = worker.make_native_adapters(built=built, stage=stage, request=request())
    return NS(stage=stage, root=root, before_layer=before_layer, adapters=adapters, camera=camera, built=built,
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
    assert native.sim.get_physics_step_count() == 0


@pytest.mark.parametrize('intrusion', [False, True])
def test_prepolicy_composition_gate_uses_both_views_and_restores_policy_buffers(native, intrusion):
    from blueprint_pipeline.native_task_asset_composition_gate import run_native_asset_composition_gate
    native.built.plan['objects'].append({'semantic_role': 'task_support'})
    native.built.camera_scene_names['overview'] = 'external_camera'
    original = native.camera.update
    def update(*args, **kwargs):
        original(*args, **kwargs)
        for key, value in native.camera.data.output.items():
            native.camera.data.output[key] = value.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        appearance = native.stage.GetPrimAtPath(native.root + '/scene_appearance/Gaussians').GetAttribute('visibility').Get() != 'invisible'
        if intrusion and appearance:
            native.camera.data.output['semantic_segmentation'][0, 5, 8, 0] = 9
    native.camera.update = update
    result = run_native_asset_composition_gate(built=native.built, plan=native.built.plan,
        output_root=native.output, stage=native.stage)
    assert result['passed'] is not intrusion, result
    assert [row['camera_role'] for row in result['views']] == ['external', 'overview']
    assert result['physics_steps'] == 0
    assert result['full_scene_sensor_buffers_refreshed']
    assert native.stage.GetRootLayer().ExportToString() == native.before_layer
    # Full-scene appearance must be back in the buffers when the gate returns.
    assert native.camera.data.output['rgb'][0, 5, 8].tolist() == [110, 70, 40]
    if intrusion:
        assert all(row['assessment']['interior_pixels_occluded_by_appearance'] == 1 for row in result['views'])
        assert result['automatic_gaussian_deletion_authorized'] is False


def test_composition_gate_retains_edge_differences_without_promoting_them_to_interior_intrusion(tmp_path):
    from blueprint_pipeline.native_task_asset_composition_gate import assess_composition_pixels
    root = tmp_path / 'pixel_comparison'
    root.mkdir()
    mask = np.ones((20, 30), dtype=bool)
    missing = np.zeros_like(mask)
    missing[0, 10] = True
    np.save(root / 'native_mesh_target_semantic_mask.npy', mask)
    np.save(root / 'mesh_target_occluded_in_full.npy', missing)
    result = assess_composition_pixels({'status': 'captured', 'blockers': []}, output_root=tmp_path)
    assert result['passed']
    assert result['silhouette_pixels_occluded_by_appearance'] == 1
    missing[10, 10] = True
    np.save(root / 'mesh_target_occluded_in_full.npy', missing)
    result = assess_composition_pixels({'status': 'captured', 'blockers': []}, output_root=tmp_path)
    assert not result['passed']
    assert result['interior_pixels_occluded_by_appearance'] == 1


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
                native.sim._physics_step_count += 1
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


def test_native_semantic_ids_and_depth_dtype_are_not_narrowed_by_shared_writer(native):
    original = native.camera.update
    exact_id = 2**40 + 7
    exact_depth = np.nextafter(np.float64(1), np.float64(2))
    native.camera.data.info['semantic_segmentation']['idToLabels'][str(exact_id)] = {'class':'task_support'}
    del native.camera.data.info['semantic_segmentation']['idToLabels']['7']
    def precise(*args, **kwargs):
        original(*args, **kwargs)
        outputs = native.camera.data.output
        outputs['semantic_segmentation'][outputs['semantic_segmentation']==7] = exact_id
        outputs['distance_to_camera'] = torch.full((1,8,12,1),exact_depth,dtype=torch.float64)
    native.camera.update = precise
    result = diagnostic.run_composition_diagnostic(request(),output_root=native.output,adapters=native.adapters)
    assert result['status']=='captured',result
    assert result['pixel_comparison']['target_semantic_ids']==[exact_id]
    full=result['passes'][0]['camera']
    labels=np.load(native.output/'full'/full['semantic_segmentation']['path'])
    depth=np.load(native.output/'full'/full['metric_depth']['path'])
    assert labels.dtype==np.int64 and (labels==exact_id).all()
    assert depth.dtype==np.float64 and (depth==exact_depth).all()
    assert full['semantic_segmentation']['pixel_counts_by_id']=={str(exact_id):96}


def test_rgba_semantic_keys_reuse_native_decoder_and_measure_occluded_tray(native):
    original = native.camera.update
    tray_id = -1066469  # Retained V28f native RGBA tuple (27, 186, 239, 255).
    background_id = -16777216
    native.camera.data.info = {'semantic_segmentation': {'idToLabels': {
        '(27, 186, 239, 255)': {'class': 'task_support'},
        '(0, 0, 0, 255)': {'class': 'UNLABELLED'},
    }}}

    def rgba_update(*args, **kwargs):
        original(*args, **kwargs)
        semantic = native.camera.data.output['semantic_segmentation']
        semantic[semantic == 7] = tray_id
        semantic[semantic == 9] = background_id
        appearance = native.stage.GetPrimAtPath('/World/envs/env_0/scene_appearance/Gaussians').GetAttribute('visibility').Get() != 'invisible'
        if appearance:
            semantic[:, 2:4, 3:5, :] = background_id
    native.camera.update = rgba_update
    result = diagnostic.run_composition_diagnostic(request(), output_root=native.output, adapters=native.adapters)
    assert result['status'] == 'captured', result['blockers']
    comparison = result['pixel_comparison']
    assert comparison['native_mesh_target_pixel_count'] == 96
    assert comparison['native_mesh_target_pixels_occluded_in_full'] == 4
    assert comparison['target_pixel_count'] == 92
    assert result['physics_steps_between_passes'] == 0
