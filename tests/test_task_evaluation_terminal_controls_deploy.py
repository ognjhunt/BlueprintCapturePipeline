from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_terminal_controls_deploy as prepare
from tests.test_task_evaluation_terminal_adoption_retirement import adopted as adopted
from tests.test_task_evaluation_unstarted_controls_reservations import reserved as reserved, put


def setup(adopted, tmp_path, monkeypatch):
    config, source, owner, intent, path, reserve, run = adopted
    config['robot_catalog_path'] = str(tmp_path/'catalog.json')
    put(Path(config['robot_catalog_path']), {'catalog_digest': 'fixture'})
    p = tmp_path/'config.json'
    put(p, config)
    sealed = prepare.worker._sealed
    monkeypatch.setattr(prepare.worker, '_sealed', lambda path, field:
        {} if Path(path).name == 'catalog.json' else sealed(path, field))
    monkeypatch.setattr(prepare.worker, 'resolve_robot_catalog', lambda *args, **kwargs: {})
    monkeypatch.setattr(prepare.adoption, 'terminal_adoption_source', lambda **kwargs: source)
    monkeypatch.setattr(prepare.team_runs, 'source_for_evaluation', lambda **kwargs: None)
    calls = []
    def provision(**kwargs):
        calls.append(kwargs)
        return {'status':'installed_terminal_adoption', 'receipt_digest':'sha256:'+'a'*64}
    monkeypatch.setattr(prepare.adoption, 'provision_terminal_controls_adoption', provision)
    return p, config, source, owner, calls


def test_deploy_prepares_only_retained_unstarted_adoption(adopted, tmp_path, monkeypatch):
    p, config, source, owner, calls = setup(adopted, tmp_path, monkeypatch)
    result = prepare.prepare(config_path=p, expected_commit='b'*40, now=102)
    assert result['rows'][0]['status'] == 'installed_terminal_adoption'
    assert len(calls) == 1 and calls[0]['intent_id'] == owner['intent_id']
    assert result['provider_mutation_performed'] is False
    assert result['model_called'] is False and result['placement_materialized'] is False


def test_exhausted_source_attempt_is_retained_without_aborting_deploy(adopted, tmp_path, monkeypatch):
    p, config, source, owner, calls = setup(adopted, tmp_path, monkeypatch)
    def exhausted(**kwargs):
        raise prepare.intake.SceneIntakeError('scene_intake_dependent_source_attempt_required')
    monkeypatch.setattr(prepare.adoption, 'provision_terminal_controls_adoption', exhausted)
    result = prepare.prepare(config_path=p, expected_commit='b'*40, now=102)
    assert result['rows'] == [{
        'intent_id': owner['intent_id'],
        'status': 'retained_exhausted_source_attempt',
        'blocker': 'scene_intake_dependent_source_attempt_required',
    }]
    assert result['provider_mutation_performed'] is False
    assert (Path(config['scene_root'])/owner['intent_id']/'intent.json').is_file()


def test_other_source_attempt_error_still_aborts_deploy(adopted, tmp_path, monkeypatch):
    p, config, source, owner, calls = setup(adopted, tmp_path, monkeypatch)
    def invalid(**kwargs):
        raise prepare.intake.SceneIntakeError('scene_intake_another_failure')
    monkeypatch.setattr(prepare.adoption, 'provision_terminal_controls_adoption', invalid)
    with pytest.raises(prepare.intake.SceneIntakeError, match='scene_intake_another_failure'):
        prepare.prepare(config_path=p, expected_commit='b'*40, now=102)


@pytest.mark.parametrize('state', ['started', 'revoked', 'expired', 'same_release'])
def test_deploy_does_not_restart_or_refresh_existing_execution(adopted, tmp_path, monkeypatch, state):
    p, config, source, owner, calls = setup(adopted, tmp_path, monkeypatch)
    if state == 'started':
        (Path(config['progression_root'])/source['launch_id']/'cpu-robot-binding').mkdir(parents=True)
    if state == 'revoked':
        put(Path(config['scene_root'])/owner['intent_id']/'revoked.json', {})
    result = prepare.prepare(config_path=p, expected_commit=('c' if state == 'same_release' else 'b')*40,
                             now=1001 if state == 'expired' else 102)
    assert calls == []
    assert result['rows'] == ([{'intent_id':owner['intent_id'], 'status':'retained_started_materialization'}] if state == 'started' else [])


@pytest.mark.parametrize('materialization', ['absent', 'deferred_only', 'paid_started'])
def test_deploy_refreshes_team_registration_before_worker_restarts(adopted, tmp_path, monkeypatch, materialization):
    p, config, source, owner, calls = setup(adopted, tmp_path, monkeypatch)
    monkeypatch.setattr(prepare.team_runs, 'source_for_evaluation', lambda **kwargs: source)
    monkeypatch.setattr(prepare.adoption, 'terminal_adoption_source',
        lambda **kwargs: pytest.fail('selected evaluation must retain its own authority'))
    intent = prepare.worker._scene_intent(Path(config['scene_root'])/owner['intent_id']/'intent.json')
    binding = Path(config['progression_root'])/source['launch_id']/prepare.scoped_identity(
        'cpu-robot-binding', intent['request']['submission_id'])
    if materialization != 'absent':
        (binding/'deferred-inputs').mkdir(parents=True)
    if materialization == 'paid_started':
        put(binding/'placement_inference.json', {})
    result = prepare.prepare(config_path=p, expected_commit='b'*40, now=102)
    assert len(calls) == (0 if materialization == 'paid_started' else 1)
    assert result['rows'][0]['status'] == (
        'retained_started_materialization' if materialization == 'paid_started' else 'installed_terminal_adoption')
    assert result['provider_mutation_performed'] is False
    assert result['model_called'] is False


@pytest.mark.parametrize('paid_file', [None, 'openai_official_cost_run_reservation.v1.json',
    'openai_official_cost_run_completion.v1.json', 'openai_key_rotation_binding.json'])
def test_cpu_candidates_and_released_lock_do_not_count_as_paid_execution(tmp_path, paid_file):
    binding=tmp_path/'binding'
    put(binding/'cpu-placement-checkpoints'/'digest'/'cpu-placement-checkpoint.v2.json', {})
    (binding/'agent-placement-attempts-digest'/'attempt_000').mkdir(parents=True)
    cost=binding/'agent-official-openai-cost'/'agent-placement-attempts-digest'/'attempt_000'
    put(cost/'openai_scope_lock_acquired.v1.json', {})
    put(cost/'openai_scope_lock_released.v1.json', {})
    if paid_file:
        put(cost/paid_file, {})
    assert prepare._only_unpaid_preparation(binding) is (paid_file is None)


def test_cpu_preparation_symlink_is_not_refreshable(tmp_path):
    binding=tmp_path/'binding'
    (binding/'cpu-placement-checkpoints').mkdir(parents=True)
    (binding/'cpu-placement-checkpoints'/'foreign').symlink_to(tmp_path, target_is_directory=True)
    assert not prepare._only_unpaid_preparation(binding)


def test_selected_evaluation_reuses_verified_placement_instead_of_calling_model_again(adopted, tmp_path, monkeypatch):
    p, config, source, owner, calls = setup(adopted, tmp_path, monkeypatch)
    monkeypatch.setattr(prepare.team_runs, 'source_for_evaluation', lambda **kwargs: source)
    intent = prepare.worker._scene_intent(Path(config['scene_root'])/owner['intent_id']/'intent.json')
    binding = Path(config['progression_root'])/source['launch_id']/prepare.scoped_identity(
        'cpu-robot-binding', intent['request']['submission_id'])
    put(binding/'paid_placement.json', {})
    from blueprint_pipeline import task_evaluation_completed_placement_adoption as completed
    discovered=[]
    def discover(**kwargs):
        discovered.append(kwargs)
        return {'verified':'placement with no native submission'}
    monkeypatch.setattr(completed, 'discover', discover)
    result=prepare.prepare(config_path=p,expected_commit='b'*40,now=102)
    assert len(discovered)==1 and discovered[0]['intent_id']==owner['intent_id']
    assert len(calls)==1 and result['rows'][0]['status']=='installed_terminal_adoption'


def test_invalid_completed_placement_is_retained_without_aborting_deploy(adopted, tmp_path, monkeypatch):
    p, config, source, owner, calls = setup(adopted, tmp_path, monkeypatch)
    monkeypatch.setattr(prepare.team_runs, 'source_for_evaluation', lambda **kwargs: source)
    intent = prepare.worker._scene_intent(Path(config['scene_root'])/owner['intent_id']/'intent.json')
    binding = Path(config['progression_root'])/source['launch_id']/prepare.scoped_identity(
        'cpu-robot-binding', intent['request']['submission_id'])
    put(binding/'paid_placement.json', {})
    from blueprint_pipeline import task_evaluation_completed_placement_adoption as completed
    def invalid(**kwargs):
        raise ValueError('completed_placement_adoption_checkpoint_lineage_invalid')
    monkeypatch.setattr(completed, 'discover', invalid)
    result = prepare.prepare(config_path=p, expected_commit='b'*40, now=102)
    assert result['rows'] == [{
        'intent_id': owner['intent_id'],
        'status': 'retained_invalid_completed_placement_adoption',
        'blocker': 'completed_placement_adoption_checkpoint_lineage_invalid',
    }]
    assert calls == []
    assert (binding/'paid_placement.json').is_file()
