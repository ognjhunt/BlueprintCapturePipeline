"""ADP-009D/day-21: controller reuses scene bytes with independent team budgets."""
import copy
import json
from functools import partial
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_controls_autoprovision as worker
from blueprint_pipeline import task_evaluation_controls_terminal_adoption as adoption
from blueprint_pipeline import task_evaluation_configured_controls_continuation_provisioning as producer
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as progression
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_team_run_controller as controller
from tests.test_task_evaluation_controls_autoprovision import setup, COMMIT
from tests.test_task_evaluation_scene_intake import request


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _case(tmp_path, monkeypatch):
    args = setup(tmp_path, purpose='scene_preparation', robot_binding_id=None)
    root = args['scene_root']
    monkeypatch.setenv(intake.ROOT_ENV, str(root))
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    monkeypatch.delenv(worker.CONFIG_ENV, raising=False)
    original_path = next(root.glob('scene-*/intent.json'))
    original = intake._read(original_path, 'intent_digest')
    profile = worker._seal({'scene_intent_digest':original['intent_digest'],
        'task_evaluation_run':{'task_id':original['request']['task']['task_id']}}, 'profile_digest')
    run = tmp_path/'launches'/'source-launch'
    _put(run/'launch_profile.json', profile)
    _put(run/'webapp_sync_succeeded.json', {'sync_result_digest':'sha256:'+'4'*64})
    terminal = {'result_digest':'sha256:'+'1'*64,'configured_scene_revision_digest':'sha256:'+'2'*64,
        'publication_result_digest':'sha256:'+'3'*64}
    monkeypatch.setattr(progression, '_validate_source', lambda p:(terminal,
        {'receipt_digest':'sha256:'+'5'*64, 'launch_profile_digest':profile['profile_digest'], 'source_commit':COMMIT},
        {'provider_zero_receipt_digest':'sha256:'+'6'*64}))
    config = {k:str(args[k]) for k in ('scene_root','preparation_queue_root','controls_root','intent_root','profile_dir')}
    config.update(launch_state_root=str(run.parent), trusted_clients=['webapp'], service_group=None)
    binding = args['catalog']['bindings']['franka-droid']
    pointer = _put(tmp_path/'current.json', {'path':binding['project_spend_reconciliation']['path']})
    binding['project_spend_current_path'] = str(pointer)
    args['catalog'] = worker._seal(args['catalog'], 'catalog_digest')
    def selected(run_id):
        value = copy.deepcopy(original['request'])
        value['submission_id'] = run_id
        value['execution'] = {**request()['execution'], 'expires_at_epoch':args['now']+7200,
            'allowed_providers':['vast','openai'], 'max_total_spend_usd':20, 'max_paid_attempts':8}
        value['task']['robot_binding_id'] = 'franka-droid'
        value['task']['evaluation_source'] = {'evaluation_run_id':run_id,'source_launch_id':run.name,
            'source_profile_digest':profile['profile_digest'],
            'configured_scene_revision_digest':terminal['configured_scene_revision_digest']}
        return intake.stage_scene_intent(value=value, queue_root=root, authenticated_client='webapp',
            trusted_clients={'webapp'}, now=args['now'])
    return args, config, original_path, selected


def test_controller_provisions_two_independent_runs_without_changing_preparation(tmp_path, monkeypatch):
    args, config, original_path, selected = _case(tmp_path, monkeypatch)
    original_bytes = original_path.read_bytes()
    observed=[]
    real_producer = args['provisioner']
    def provision(**kw):
        observed.append(kw)
        return real_producer(**kw)
    monkeypatch.setattr(producer, 'provision_configured_controls_continuation', provision)
    # Own real publication metadata/registry, replace only provider/object storage with fixture adapters.
    for run_id in ('eval-one','eval-two'):
        owner = selected(run_id)
        result = adoption.provision_terminal_controls_adoption(config=config, catalog=args['catalog'],
            intent_id=owner['intent_id'], expected_production_commit=COMMIT, now=args['now'])
        installed = json.loads(Path(result['installation']['registry_path']).read_text())
        assert installed['evaluation_run_id'] == run_id
        assert installed['evaluation_authority']['scene_intent_digest'] == owner['intent_digest']
        assert all(v['scene_intent_digest'] == owner['intent_digest'] for v in observed[-1]['scene_phase_attempts'].values())
        before = len(observed)
        assert adoption.provision_terminal_controls_adoption(config=config, catalog=args['catalog'],
            intent_id=owner['intent_id'], expected_production_commit=COMMIT, now=args['now']) == result
        assert len(observed) == before
    assert len(list(Path(config['intent_root']).glob('adoption-*.json'))) == 2
    assert original_path.read_bytes() == original_bytes
    assert not list((original_path.parent/'attempts').glob('*.json'))
    assert len(list(args['scene_root'].glob('scene-*/attempts/*.json'))) == 6
    monkeypatch.setattr(controller.authority, 'evaluation_owner',
        partial(controller.authority.evaluation_owner, now=args['now']))
    from blueprint_pipeline import task_evaluation_release_identity as release
    monkeypatch.setattr(release, 'running_release_commit', lambda: COMMIT)
    monkeypatch.setenv(worker.CONFIG_ENV, str(_put(tmp_path/'controls-config.json', config)))
    materialized=[]
    def materialize(**kw):
        selected_intent=json.loads(kw['intent_path_override'].read_text())
        materialized.append(selected_intent['evaluation_run_id'])
        return {'status':'materialized','plan_digest':'sha256:'+'a'*64}
    rows = controller.materialize_selected_evaluations(intent_root=config['intent_root'],
        launch_state_root=config['launch_state_root'], progression_root=tmp_path/'progress',
        plan_root=tmp_path/'plans', release=COMMIT, materializer=materialize)
    assert len(rows)==2 and set(materialized)=={'eval-one','eval-two'}
    selected_intent=json.loads(next(Path(config['intent_root']).glob('adoption-*.json')).read_text())
    plan={'source_launch_id':'source-launch', 'source_launch_receipt_digest':'sha256:'+'5'*64,
        'evaluation_run_id':selected_intent['evaluation_run_id'],
        'evaluation_authority':selected_intent['evaluation_authority']}
    scope=worker.ProgressionOwnerScope([], {('team','scene','task')}, False)
    assert scope.plan_blocker(_put(tmp_path/'plan.json',plan), Path(config['launch_state_root'])) is None
    assert controller.plan_authority_blocker({**plan,'source_launch_receipt_digest':'sha256:'+'9'*64},
        config['launch_state_root']) == 'controls_autoprovision_team_evaluation_source_changed'

    # A revoked request cannot materialize; the other request still advances.
    revoked=next(p for p in args['scene_root'].glob('scene-*/intent.json')
        if intake._read(p,'intent_digest')['request']['submission_id']=='eval-one')
    _put(revoked.parent/'revoked.json', {})
    materialized.clear()
    rows = controller.materialize_selected_evaluations(intent_root=config['intent_root'],
        launch_state_root=config['launch_state_root'], progression_root=tmp_path/'progress',
        plan_root=tmp_path/'plans', release=COMMIT, materializer=materialize)
    assert materialized==['eval-two']
    assert sum(r['status']=='team_evaluation_refused' for r in rows)==1


@pytest.mark.parametrize('mutation', ['revoked','wrong_profile'])
def test_refuses_before_reserving_or_provisioning(tmp_path, monkeypatch, mutation):
    args, config, original_path, selected = _case(tmp_path, monkeypatch)
    owner = selected('eval-one')
    directory = args['scene_root']/owner['intent_id']
    if mutation == 'revoked':
        _put(directory/'revoked.json', {})
    else:
        path = Path(config['launch_state_root'])/'source-launch'/'launch_profile.json'
        profile = json.loads(path.read_text())
        if mutation == 'wrong_profile':
            profile['untrusted'] = True
        else:
            profile['profile_digest'] = 'sha256:'+'9'*64
        _put(path, profile)
    with pytest.raises(ValueError):
        adoption.provision_terminal_controls_adoption(config=config, catalog=args['catalog'],
            intent_id=owner['intent_id'], expected_production_commit=COMMIT, now=args['now'])
    assert not list(args['scene_root'].glob('scene-*/attempts/*.json'))


def test_unavailable_policy_refuses_before_source_read_or_reservation(tmp_path, monkeypatch):
    args, config, original_path, selected = _case(tmp_path, monkeypatch)
    owner=selected('eval-one')
    intent=intake._read(args['scene_root']/owner['intent_id']/'intent.json','intent_digest')
    intent['request']['execution']['policy_candidates'][0]['artifact_digest']='sha256:'+'9'*64
    def forbidden(*a, **kw):
        raise AssertionError('source or provider must not be inspected')
    monkeypatch.setattr(progression, '_validate_source', forbidden)
    with pytest.raises(ValueError, match='team_evaluation_policy_unavailable'):
        controller.source_for_evaluation(config=config, intent=intent, now=args['now'])
    assert not list(args['scene_root'].glob('scene-*/attempts/*.json'))
