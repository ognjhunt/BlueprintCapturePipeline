"""No Isaac, GPU, policy, or model calls: drive the real settle receipt boundary."""
import copy
from types import SimpleNamespace

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_native_settle_gate import (
    NativeSettleAdapter, NativeSettleGateError, make_native_settle_adapter,
    run_native_settle_gate, validate_native_settle_spec,
)

D = 'sha256:' + 'a'*64


def _spec():
    spec = dict(schema_version='task_object_native_settle_spec.v1',required=True,
        source_packet_digest=D,scene_collision_digest=D,support_measurement_digest=D,
        gravity_settle_seconds=3.,settle_window_samples=20,control_hz=15,physics_hz=120,
        qualification_limits=dict(maximum_penetration_m=.001,minimum_support_contact_force_n=.01,
            maximum_forbidden_contact_force_n=.1,settle_translation_tolerance_m=.002,
            settle_rotation_tolerance_rad=.01,reset_translation_tolerance_m=.002,reset_rotation_tolerance_rad=.01),
        objects=[dict(role=role,asset_id=role,asset_version='new-v1',asset_sha256=D,
            collision_bounds_body_frame_m={'minimum':[-.1,-.1,0.],'maximum':[.1,.1,.02]},
            support_source_prim_paths=['/Root/table'],support_native_prim_paths=['{ENV_REGEX_NS}/scene_collision/table'],
            support_top_z_m=0.) for role in ('task_object','task_support')])
    spec['spec_digest'] = canonical_digest(spec,digest_field='spec_digest')
    return spec


class Simulator:
    def __init__(self, mode='sleeping'):
        self.mode, self.step, self.resets, self.error = mode, 0, [], False
    def reset(self, seed):
        self.resets.append(seed)
        self.step = 0
    def hold(self):
        self.step += 1
        if self.mode == 'step_error':
            raise RuntimeError('native cooking exception')
    def sample(self):
        z = .002 if self.step == 0 else 0.
        if self.mode == 'floating' and self.step:
            z = .02
        if self.mode == 'tunnel' and self.step == 4:
            z = -.02
        speed = .02 if self.mode == 'moving' and self.step > 20 else 0.
        pose = [0.,0.,z,0.,0.,0.,1.]
        force = .02 if self.step == 1 and self.mode != 'no_contact' else 0.
        return dict(measurement_authority='native_rigid_root_pose_and_filtered_contact_sensors',scene_collision_digest=D,
            asset_root_pose_world=pose,destination_pose_world=pose,
            task_initial_support_contact_peak_force_n=force,destination_scene_support_contact_peak_force_n=force,
            task_scene_collision_peak_force_n=.2 if self.mode == 'forbidden' and self.step == 3 else 0.,
            destination_scene_forbidden_contact_peak_force_n=0.,step_dt_s=1/15,physics_dt_s=1/120,
            asset_bindings={r:{'asset_id':r,'asset_sha256':D} for r in ('task_object','task_support')},
            root_linear_velocity_m_s={r:[speed,0.,0.] for r in ('task_object','task_support')},
            root_angular_velocity_rad_s={r:[0.,0.,0.] for r in ('task_object','task_support')},
            contact_bindings={r:['{ENV_REGEX_NS}/scene_collision/table'] for r in ('task_object','task_support')})
    def capture(self, label, root):
        Image.new('RGB',(2,2),(self.step,0,0)).save(root/(label+'.png'))
        if self.mode == 'media_error' and label == 'after':
            raise RuntimeError('camera measurement failed after preserving PNG')
        return {'snapshot_id':label,'native_step':self.step}
    def cooking(self):
        self.error = self.error or (self.mode == 'cooking' and self.step >= 2)
        return {'monitoring_active':self.mode != 'no_monitor','errors':['convex cooking failed'] if self.error else []}
    def adapter(self):
        return NativeSettleAdapter(self.reset,self.hold,self.sample,self.capture,self.cooking)


def _run(tmp_path, sim, **kwargs):
    return run_native_settle_gate(spec=_spec(),adapter=sim.adapter(),output_root=tmp_path,
        seed=77,scene_plan_digest=D,adoption_digest=D,**kwargs)


def test_sleeping_supported_assets_pass_and_original_unsettled_spawn_is_restored(tmp_path):
    sim = Simulator()
    result = _run(tmp_path,sim)
    assert result['status'] == 'passed'
    assert result['completed_hold_steps'] == 45
    assert sim.resets == [77,77]
    assert result['original_spawn_restored'] is True
    assert result['initial_state']['task_object']['support_gap_m'] == .002
    assert result['restored_state']['task_object']['support_gap_m'] == .002
    assert result['samples'][-1]['objects']['task_object']['support_contact_force_n'] == 0.
    assert result['observed_support_contacts'] == {'task_object':True,'task_support':True}
    assert result['continuous_support_contact_proven'] is False
    assert result['policy_start_already_settled'] is False
    assert result['candidate_policy_queried'] is False
    assert result['controls_qualified'] is False
    assert result['velocity_limits']['linear_m_s'] == pytest.approx(.002/(19/15))
    assert result['gate_digest'] == canonical_digest(result,digest_field='gate_digest')
    for phase in ('before','after'):
        assert result['frames'][phase]['status'] == 'captured'
        assert (tmp_path/result['frames'][phase]['frames'][0]['path']).is_file()


@pytest.mark.parametrize('mode,blocker',[
    ('floating','support_contact_missing'),('no_contact','support_contact_missing'),
    ('moving','velocity_not_stable'),('tunnel','penetration_tunneling'),('forbidden','forbidden_contact'),
    ('cooking','collider_or_physics_error'),('no_monitor','physics_error_monitor_missing'),
    ('step_error','runtime_exception'),('media_error','required_media_gap')])
def test_failure_preserves_both_camera_frames_restores_spawn_and_never_queries_policy(tmp_path,mode,blocker):
    sim = Simulator(mode)
    result = _run(tmp_path,sim)
    assert result['status'] == 'blocked'
    assert any(blocker in item for item in result['blockers'])
    assert result['candidate_policy_loaded'] is False
    assert result['candidate_policy_queried'] is False
    assert sim.resets == [77,77]
    assert result['frames']['before']['frames'] and result['frames']['after']['frames']
    assert (tmp_path/'task_object_native_settle_gate.v1.json').is_file()


def test_wrong_support_body_and_nonfinite_state_fail_closed(tmp_path):
    sim = Simulator()
    original = sim.sample
    def wrong():
        sample = original()
        sample['contact_bindings']['task_support'] = ['{ENV_REGEX_NS}/scene_collision/other_table']
        return sample
    sim.sample = wrong
    result = _run(tmp_path/'binding',sim)
    assert result['status'] == 'blocked'
    assert any('binding_mismatch' in b for b in result['blockers'])
    sim = Simulator()
    original = sim.sample
    def nonfinite():
        sample = original()
        sample['root_linear_velocity_m_s']['task_object'][0] = float('nan')
        return sample
    sim.sample = nonfinite
    result = _run(tmp_path/'nonfinite',sim)
    assert result['status'] == 'blocked'
    assert any('nonfinite' in b for b in result['blockers'])


def test_same_seed_candidate_reset_parity_is_required(tmp_path):
    reference = _run(tmp_path/'first',Simulator())
    repeat = _run(tmp_path/'second',Simulator(),reference_receipt=reference)
    assert repeat['status'] == 'passed'
    assert repeat['reset_state_digest'] == reference['reset_state_digest']
    sim = Simulator()
    original = sim.sample
    def displaced():
        result = original()
        result['asset_root_pose_world'] = [0.01,*result['asset_root_pose_world'][1:]]
        return result
    sim.sample = displaced
    failed = _run(tmp_path/'different',sim,reference_receipt=reference)
    assert any('reset_parity_mismatch' in b for b in failed['blockers'])
    assert failed['completed_hold_steps'] == 0


def test_spec_requires_typed_two_asset_adoption_and_no_unbounded_duration():
    assert validate_native_settle_spec(_spec())['required'] is True
    for key,value in [('required',False),('gravity_settle_seconds',1000.),('objects',[_spec()['objects'][0]])]:
        spec = copy.deepcopy(_spec())
        spec[key] = value
        spec['spec_digest'] = canonical_digest(spec,digest_field='spec_digest')
        with pytest.raises(NativeSettleGateError):
            validate_native_settle_spec(spec)


def test_native_adapter_uses_real_readback_fields_exact_sensor_bindings_and_existing_hold(monkeypatch):
    from blueprint_pipeline import native_task_arena_readback as readback
    import sys
    sim = Simulator()
    monkeypatch.setattr(readback,'NativeRigidTaskArenaReadback',lambda built:SimpleNamespace(read_task_sample=sim.sample))
    calls = []
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(tensor=lambda data,**kwargs: data,float32='float32'))
    class Scene(dict):
        env_regex_ns = '/World/envs/env_.*'
        env_prim_paths = ['/World/envs/env_0']
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(joint_pos=[[.1]*7+[.04,.04]])))
    rows,sensors,names = [],[],{}
    for role,logical in [('task_object','task_initial_support_contact'),('task_support','destination_scene_support_contact')]:
        scene[role] = SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=[[0.,0.,0.]],root_ang_vel_w=[[0.,0.,0.]]))
        rows.append({'semantic_role':role,'asset_id':role,'sha256':D,'object_type':'RIGID'})
        name = logical+'__rigid_00'
        paths = ['{ENV_REGEX_NS}/scene_collision/table']
        sensors.append({'sensor_instance_id':name,'prim_path':'{ENV_REGEX_NS}/'+role,'filter_prim_paths_expr':paths})
        scene[name] = SimpleNamespace(cfg=SimpleNamespace(prim_path='/World/envs/env_0/'+role,
            filter_prim_paths_expr=['/World/envs/env_0/scene_collision/table']))
        names[logical] = (name,)
    rows.append({'semantic_role':'scene_collision','sha256':D})
    env = SimpleNamespace(scene=scene,step_dt=1/15,physics_dt=1/120,device='cpu',reset=lambda **kw:calls.append(kw),step=calls.append)
    built = SimpleNamespace(env=env,plan={'objects':rows,'articulation':{'contact_sensors':sensors}},
        scene_asset_names={r:r for r in ('task_object','task_support')},contact_sensor_names=names)
    adapter = make_native_settle_adapter(built=built,capture_frames=sim.capture,read_cooking_errors=sim.cooking)
    adapter.reset(77)
    adapter.hold_step()
    assert calls == [{'seed':77},[[.1]*7+[0.]]]
    assert adapter.read_sample()['contact_bindings'] == sim.sample()['contact_bindings']
    scene['destination_scene_support_contact__rigid_00'].cfg.prim_path = '/World/envs/env_0/wrong_body'
    with pytest.raises(NativeSettleGateError,match='sensor_body_binding_mismatch'):
        adapter.read_sample()


def test_noop_reset_cannot_leave_settled_assets_within_two_mm_reset_tolerance(tmp_path):
    sim = Simulator()
    def reset(seed):
        sim.resets.append(seed)
        if len(sim.resets) == 1:
            sim.step = 0
    sim.reset = reset
    result = _run(tmp_path,sim)
    assert result['status'] == 'blocked'
    assert result['original_spawn_restored'] is False
    assert 'native_settle_original_spawn_restore_noop' in result['blockers']
