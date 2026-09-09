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
