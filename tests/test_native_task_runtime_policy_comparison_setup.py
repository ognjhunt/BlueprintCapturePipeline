from dataclasses import asdict
import json
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_bundle import _candidate_runtime_binding
from blueprint_pipeline.native_task_runtime_policy_comparison_setup import prepare_retained_policy_comparison
from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import _scene_plan
from tests.test_adp_task_scoring import _articulated_spec


def inputs(tmp_path, monkeypatch):
    plan = _scene_plan()
    plan['scene_id'] = 'interiorgs-840920'
    plan['task_kind'] = 'articulated_open_close'
    plan['task_spec'] = {**_articulated_spec(), 'maximum_action_steps': 240, 'control_frequency_hz': 15.0}
    plan['task_spec']['prompt'] = 'Push the retained mug across the table.'
    plan['plan_digest'] = canonical_digest(plan, digest_field='plan_digest')
    packet = tmp_path / 'packet'
    packet.mkdir()
    write_json(packet / 'native_task_arena_scene_plan.v1.json', plan)
    write_json(packet / 'native_task_arena_packet_receipt.v1.json', {'fixture': True})
    write_json(packet / 'native_task_arena_packet_request.v1.json', {})
    monkeypatch.setattr('blueprint_pipeline.native_task_arena_bundle.verify_native_task_arena_packet',
                        lambda _: (packet, {}, {}))
    readiness_path = Path(__file__).resolve().parents[1] / 'docs/arm_decision_proof_v1/manifests/adp009d_scene_840920_policy_readiness.v1.json'
    readiness = json.loads(readiness_path.read_text())
    candidates = {x['candidate_id']: x for x in readiness['candidates']}
    specs = {}
    for name, candidate in candidates.items():
        policy, _, _ = _candidate_runtime_binding(name)
        spec = {'candidate_id': name, 'scene_plan_digest': plan['plan_digest'],
                'prompt': plan['task_spec']['prompt'], 'policy_spec': asdict(policy),
                'max_policy_queries': 30, 'open_loop_horizon': 8,
                'candidate_rights_binding': {'checkpoint_identity': candidate['checkpoint']}}
        spec['execution_spec_digest'] = canonical_digest(spec, digest_field='execution_spec_digest')
        path = tmp_path / (name + '.json')
        write_json(path, spec)
        specs[name] = path
    control = {'scene_plan_digest': plan['plan_digest'], 'status': 'blocked', 'controls_qualified': False}
    control['result_digest'] = canonical_digest(control, digest_field='result_digest')
    write_json(tmp_path / 'control.json', control)
    construction = {'schema_version': 'native_task_arena_construction_result.v1',
                    'status': 'completed', 'construction_gate_qualified': True,
                    'scene_plan_digest': plan['plan_digest']}
    construction['result_digest'] = canonical_digest(construction, digest_field='result_digest')
    write_json(tmp_path / 'construction.json', construction)
    write_json(tmp_path / 'runtime.json', {'fixture': True})
    return dict(packet_dir=packet, runtime_source_receipt=tmp_path / 'runtime.json',
                retained_spec_paths=specs, readiness_path=readiness_path,
                construction_path=tmp_path / 'construction.json', control_path=tmp_path / 'control.json',
                implementation_commit='c' * 40, confirmed_by='fixture-owner',
                authorization_reference='fixture:reuse-exact-frozen-task')


def test_both_variants_freeze_identical_inputs_and_keep_failed_controls(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path, monkeypatch)
    results = [prepare_retained_policy_comparison(**kwargs, variant=variant, output_dir=tmp_path / variant)
               for variant in ('baseline', 'isaaclab_3_ga')]
    assert results[0]['binding_digest'] == results[1]['binding_digest']
    for variant in ('baseline', 'isaaclab_3_ga'):
        prepared = json.loads((tmp_path / variant / 'runtime-inputs.json').read_text())
        assert len(prepared['cells']) == 10
        assert all(c['control_diagnostic']['mode'] == 'nonblocking_diagnostic_bound' for c in prepared['cells'])
        assert json.loads(Path(prepared['cells'][0]['control_diagnostic']['receipt']['path']).read_text())['controls_qualified'] is False
        for name, source in kwargs['retained_spec_paths'].items():
            old = json.loads(source.read_text())
            new = json.loads((tmp_path / variant / (name + '.json')).read_text())
            for field in ('prompt', 'policy_spec', 'max_policy_queries', 'open_loop_horizon'):
                assert new[field] == old[field]
            assert new['qualification_permitted'] is False


@pytest.mark.parametrize('field,value', [('prompt', 'Different task'), ('policy_spec', {}), ('scene_plan_digest', 'sha256:' + 'f' * 64)])
def test_refuses_changed_retained_policy_input(tmp_path, monkeypatch, field, value):
    kwargs = inputs(tmp_path, monkeypatch)
    path = kwargs['retained_spec_paths']['pi05_droid']
    old = json.loads(path.read_text())
    old[field] = value
    old['execution_spec_digest'] = canonical_digest(old, digest_field='execution_spec_digest')
    write_json(path, old)
    with pytest.raises(ValueError, match='policy_inputs_changed'):
        prepare_retained_policy_comparison(**kwargs, variant='baseline', output_dir=tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()
