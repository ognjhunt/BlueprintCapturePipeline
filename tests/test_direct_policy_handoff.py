"""Website policy admission uses compiled scene evidence, not scripted success."""
from copy import deepcopy
import json

import pytest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_control_omission_handoff import prepared
from tests import test_task_evaluation_policy_canary_handoff as rehearsal
from tests.test_task_evaluation_launch_activation_contract import request, ref
from blueprint_pipeline.task_evaluation_launch_activation_contract import validate_launch_activation_request


def source_closeout(tmp_path):
    from tests.test_task_evaluation_configured_controls_progression_worker import _source
    root, _ = _source(tmp_path / 'source-closure')
    old = root / 'scene-839873-qualifying'
    target = tmp_path / 'launch-runs' / rehearsal.SOURCE_LAUNCH_ID
    profile = json.loads((target / 'launch_profile.json').read_bytes())
    receipt = json.loads((old / 'launch_receipt.json').read_bytes())
    receipt.update(launch_id=rehearsal.SOURCE_LAUNCH_ID, launch_profile_digest=profile['profile_digest'])
    receipt['receipt_digest'] = canonical_digest(receipt, digest_field='receipt_digest')
    (target / 'launch_receipt.json').write_text(json.dumps(receipt))
    for name, digest in [('webapp_sync_succeeded.json', 'sync_result_digest'),
                         ('post_teardown_provider_zero_receipt.json', 'provider_zero_receipt_digest')]:
        row = json.loads((old / name).read_bytes())
        for key in ('launch_id', 'launch_profile_digest', 'receipt_digest'):
            if key in row:
                row[key] = receipt[key]
        if 'response' in row:
            row['response'].update(launch_id=receipt['launch_id'], receipt_digest=receipt['receipt_digest'])
        row[digest] = canonical_digest(row, digest_field=digest)
        (target / name).write_text(json.dumps(row))


def direct_case(tmp_path, monkeypatch):
    state, directory, directive, contract = prepared(tmp_path, monkeypatch)
    directive.unlink()  # Platform default, not a hand-created per-site opt-in.
    (state / 'construction_launch_progression.json').unlink()
    source_closeout(tmp_path)
    plan_path = tmp_path / 'compiled-episodes' / rehearsal.PREPARATION_ID / 'native-task-packet/native_task_arena_scene_plan.v1.json'
    plan = json.loads(plan_path.read_bytes())
    from tests.test_camera_start_construction_materializer import inputs
    plan['robot']['joint_reset_positions_rad'] = inputs()['source_binding']['joint_reset_positions_rad']
    plan['plan_digest'] = canonical_digest(plan, digest_field='plan_digest')
    plan_path.write_text(json.dumps(plan))
    packet_path = plan_path.with_name('native_task_arena_packet_receipt.v1.json')
    packet = json.loads(packet_path.read_bytes())
    packet['arena_scene_plan_digest'] = plan['plan_digest']
    packet_path.write_text(json.dumps(packet))
    return state, directory, {'kind': 'initial_project', 'project_spend_reconciliation': ref(21), 'initial_provider_zero': ref(22)}


def test_direct_handoff_submits_policy_without_scripted_run(tmp_path, monkeypatch):
    state, directory, lineage = direct_case(tmp_path, monkeypatch)
    before = (directory / 'intent.json').read_bytes()
    webapp, publisher = rehearsal._WebApp(), rehearsal._Publisher()
    result = rehearsal._advance(tmp_path, state=state, webapp=webapp, publisher=publisher,
        profile_calls=[], diagnostic_initial_lineage=lineage)
    assert result['status'] == 'canary_launch_submitted'
    assert len(webapp.calls) == 1
    params = json.loads((state / 'policy-canary-inputs/presubmission_parameters.json').read_bytes())
    assert params['activation_lineage']['kind'] == 'initial_project'
    construction = json.loads(publisher.published[params['activation_lineage']['construction_result']['uri']])
    assert construction['schema_version'] == 'task_evaluation_episode_compilation_result.v1'
    assert 'construction_gate_qualified' not in construction
    binding = params['diagnostic_control_omission_authority']['policy_canary_camera_start_configuration']
    assert binding['native_qualification_claimed'] is False
    assert 'construction_result_digest' not in binding
    assert (directory / 'intent.json').read_bytes() == before
    assert rehearsal._advance(tmp_path, state=state, webapp=webapp, publisher=publisher,
        profile_calls=[], diagnostic_initial_lineage=lineage)['status'] == 'canary_launch_submitted'
    assert len(webapp.calls) == 1


@pytest.mark.parametrize('missing', ['launch_receipt.json', 'webapp_sync_succeeded.json', 'post_teardown_provider_zero_receipt.json'])
def test_direct_handoff_still_requires_scene_preparation_closeout(tmp_path, monkeypatch, missing):
    state, _, lineage = direct_case(tmp_path, monkeypatch)
    (tmp_path / 'launch-runs' / rehearsal.SOURCE_LAUNCH_ID / missing).unlink()
    webapp = rehearsal._WebApp()
    with pytest.raises((ValueError, RuntimeError)):
        rehearsal._advance(tmp_path, state=state, webapp=webapp, publisher=rehearsal._Publisher(),
            profile_calls=[], diagnostic_initial_lineage=lineage)
    assert not webapp.calls


def test_initial_policy_lineage_is_diagnostic_only_and_budget_bound():
    value = request(lane='native_task_arena_policy_evaluation')
    value.update(run_kind='internal_policy_canary', capture_session_id='source-scene', intake_id='configured-task')
    value['lineage'] = {'kind': 'initial_project', 'project_spend_reconciliation': ref(21),
                        'initial_provider_zero': ref(22), 'construction_result': ref(23)}
    assert validate_launch_activation_request(value) == value
    for key in ('run_kind',):
        bad = deepcopy(value)
        bad.pop(key)
        with pytest.raises(ValueError):
            validate_launch_activation_request(bad)
    for key in ('project_spend_reconciliation', 'initial_provider_zero', 'construction_result'):
        bad = deepcopy(value)
        bad['lineage'].pop(key)
        with pytest.raises(ValueError):
            validate_launch_activation_request(bad)


@pytest.mark.parametrize("compiled", [False, True])
def test_controller_skips_scripted_construction_after_cpu_compilation(tmp_path, monkeypatch, compiled):
    from tests import test_task_evaluation_configured_controls_progression_worker as fixture
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker
    from blueprint_pipeline import task_evaluation_scene_control_omission as omission
    launch_root, _ = fixture._source(tmp_path)
    plan_path = fixture._plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    state = tmp_path / 'progressions' / plan['source_launch_id'] / ('franka-controls-' + plan['expected_production_commit'][:12])
    fixture._write(state / 'configured_controls_progression.v1.json', fixture._sealed_progression(
        'episode_preparation_queued', episode_preparation_request={'preparation_id': 'prep'}))
    fixture._write(tmp_path / 'preparations/identities/prep.json', {'identity': 'prep'})
    preparation = {'run_id': 'run', 'episode_compilation_id': 'compile', 'episode_compilation_queue_envelope_digest': 'sha256:' + 'b' * 64,
                   'status': 'queued_for_production_episode_compilation'}
    fixture._write(tmp_path / 'preparations/results/prep-a.json', preparation)
    if compiled:
        result = {'schema_version': 'task_evaluation_episode_compilation_result.v1',
                  'status': 'compiled_for_production_launch', 'run_id': 'run', 'compilation_id': 'compile',
                  'source_commit': plan['expected_production_commit'], 'blockers': [],
                  'provider_mutation_performed': False, 'paid_execution_requested': False}
        result['result_digest'] = canonical_digest(result, digest_field='result_digest')
        fixture._write(tmp_path / 'compilations/results' / ('compile-' + 'b' * 64 + '.json'), result)
    monkeypatch.setattr(omission, 'load_for_run', lambda **_: {'directive_digest': 'sha256:' + 'c' * 64})
    def no_launch(**_):
        pytest.fail('scripted construction must not be activated')
    monkeypatch.setattr(worker, 'stage_configured_controls_activation', no_launch)
    result = worker.advance_configured_controls_plan(plan_path=plan_path, launch_state_root=launch_root,
        progression_root=tmp_path / 'progressions', preparation_queue_root=tmp_path / 'preparations',
        activation_queue_root=tmp_path / 'activations', episode_compilation_queue_root=tmp_path / 'compilations',
        publisher_factory=lambda: object())
    assert result['status'] == ('controls_omitted_for_diagnostic_policy' if compiled else 'awaiting_construction_compilation')
    if compiled:
        assert result['construction_rehearsal_performed'] is False
        assert result['qualified_comparison_permitted'] is False
    assert not (state / 'construction_activation_progression.json').exists()
