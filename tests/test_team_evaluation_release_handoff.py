"""ADP-009D/day-21: real selected-run producers survive a release update."""
import json
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline import task_evaluation_terminal_controls_deploy as deploy
from blueprint_pipeline import task_evaluation_intent_registry as registry
from blueprint_pipeline import task_evaluation_controls_terminal_adoption as adoption
from blueprint_pipeline import task_evaluation_controls_autoprovision as worker
from blueprint_pipeline import task_evaluation_configured_controls_continuation_provisioning as producer
from tests.test_team_evaluation_controller import _case, _put, COMMIT


def test_selected_evaluation_crosses_real_provisioning_registry_and_release_refresh(tmp_path, monkeypatch):
    args, config, original_path, selected = _case(tmp_path, monkeypatch)
    owner = selected('selected-rehearsal')
    unchanged = original_path.read_bytes()
    # Real request, authority, provisioning, reservation, registry and deploy
    # functions. Replace external object storage/inventory and host identity only.
    monkeypatch.setattr(producer, 'provision_configured_controls_continuation', args['provisioner'])
    first = adoption.provision_terminal_controls_adoption(config=config, catalog=args['catalog'],
        intent_id=owner['intent_id'], expected_production_commit=COMMIT, now=args['now'])
    registration = Path(first['installation']['registry_path'])
    old_bytes = registration.read_bytes()
    new_commit = 'e' * 40
    content = args['catalog']
    content['schema_version'] = worker.CONTENT_CATALOG_SCHEMA
    content['bindings']['franka-droid'].pop('expected_production_commit')
    config['robot_catalog_path'] = str(_put(tmp_path/'catalog.json', worker._seal(content, 'catalog_digest')))
    config_path = _put(tmp_path/'config.json', config)
    units = []
    def systemctl(argv, **kwargs):
        assert argv[:2] == ['systemctl', 'show']
        units.append(argv[2])
        return SimpleNamespace(stdout='LoadState=loaded\nActiveState=inactive\nMainPID=0\n')
    monkeypatch.setattr(registry, '_verified_checkout_head', lambda: new_commit)
    monkeypatch.setattr(registry.subprocess, 'run', systemctl)
    result = deploy.prepare(config_path=config_path, expected_commit=new_commit, now=args['now']+1)
    assert result['rows'][0]['status'] == 'installed_terminal_adoption'
    current = json.loads(registration.read_text())
    assert current['expected_production_commit'] == new_commit
    assert current['evaluation_run_id'] == 'selected-rehearsal'
    assert current['evaluation_authority']['scene_intent_digest'] == owner['intent_digest']
    assert registration.with_name(f'{registration.stem}.superseded-{COMMIT}.json').read_bytes() == old_bytes
    assert set(units) == {'blueprint-task-evaluation-configured-controls-progression.'+s
                         for s in ('service', 'path', 'timer')}
    assert original_path.read_bytes() == unchanged
    attempts = list((args['scene_root']/owner['intent_id']/'attempts').glob('*.json'))
    before = {p:p.read_bytes() for p in attempts}
    deploy.prepare(config_path=config_path, expected_commit=new_commit, now=args['now']+2)
    assert {p:p.read_bytes() for p in attempts} == before
    assert len(list((args['scene_root']/owner['intent_id']/'attempts').glob('*.json'))) == len(attempts)
