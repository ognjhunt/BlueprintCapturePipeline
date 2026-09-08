from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_autostart as auto
from blueprint_pipeline import task_evaluation_controls_terminal_adoption as adoption
from blueprint_pipeline import task_evaluation_terminal_adoption_retirement as retirement
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_unstarted_controls_reservations as cancellation
from blueprint_pipeline import task_evaluation_release_identity as identity
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_unstarted_controls_reservations import reserved as reserved, put


@pytest.fixture
def adopted(reserved, tmp_path, monkeypatch):
    scene_root, run, owner, reserve, originals = reserved
    cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=scene_root)
    phases = {}
    for phase, cap, provider in [('construction', .45, 'vast'), ('controls', .45, 'vast'), ('placement', 2.56, 'openai')]:
        attempt = reserve('controls-adopted-'+phase, cap, provider, commit='c')
        if phase != 'placement':
            p = tmp_path/'adopted'/f'{phase}.json'
            put(p, {'scene_owner_attempt': authority.bind_scene_attempt(attempt)})
            phases[phase] = {'authorization_path': str(p)}
    source = {'launch_id': run.name, 'source_commit': 'd'*40,
        'adoption': {'mode': 'explicit_terminal_adoption', 'source_launch_id': run.name}}
    intent = {'expected_production_commit': 'c'*40, 'configuration_adoption': source['adoption'], 'phases': phases}
    intent['intent_digest'] = canonical_digest(intent, digest_field='intent_digest')
    p = tmp_path/'adopted'/'intent.json'
    put(p, intent)
    controls = tmp_path/'controls'
    record = {'execution_source_commit': 'c'*40, 'adoption': source['adoption'], 'provisioning': {'intent_path': str(p)}}
    record['receipt_digest'] = canonical_digest(record, digest_field='receipt_digest')
    put(controls/'terminal-adoptions'/owner['intent_id']/'old'/'terminal_adoption_provisioning.json', record)
    config = {'scene_root': str(scene_root), 'controls_root': str(controls), 'progression_root': str(tmp_path/'progression')}
    monkeypatch.setattr(identity, 'running_release_commit', lambda: 'b'*40)
    return config, source, owner, intent, p, reserve, run


def test_unstarted_adoption_rollover_preserves_holds_and_does_not_refund_started_work(adopted):
    config, source, owner, intent, path, reserve, run = adopted
    directory = Path(config['scene_root'])/owner['intent_id']
    originals = {p: p.read_bytes() for p in (directory/'attempts').glob('*.json')}
    kwargs = dict(config=config, intent_id=owner['intent_id'], source=source, expected_production_commit='b'*40)
    dry = retirement.retire_unmaterialized_adoptions(**kwargs, dry_run=True)
    assert len(dry) == 3
    assert len(list((directory/cancellation.DIRECTORY).glob('*.json'))) == 3
    rows = retirement.retire_unmaterialized_adoptions(**kwargs)
    assert rows == dry
    assert all(p.read_bytes() == b for p, b in originals.items())
    for row in rows:
        attempt = intake._read(directory/'attempts'/(row['attempt_id']+'.json'), 'attempt_digest')
        assert cancellation.validated_cancellation(directory, attempt) == row
        assert authority.scene_execution_authority_blockers(authority.bind_scene_attempt(attempt),
            source_commit='c'*40, provider=attempt['provider'], maximum_spend_usd=attempt['maximum_spend_usd'], now=102) == ['scene_execution_owner_attempt_cancelled_before_execution']
    for phase, cap, provider in [('construction', .45, 'vast'), ('controls', .45, 'vast'), ('placement', 2.56, 'openai')]:
        reserve('controls-final-'+phase, cap, provider, commit='b')
    reserve('policies', 4, commit='b')
    with pytest.raises(ValueError, match='spend_cap_exhausted'):
        reserve('extra', .10, commit='b')
    # A later current-release materialization does not invalidate old retirement
    # receipts or make the cancelled release admissible again.
    (Path(config['progression_root'])/run.name/'cpu-robot-binding').mkdir(parents=True)
    assert retirement.retire_unmaterialized_adoptions(**kwargs) == rows


@pytest.mark.parametrize('corruption', ['started', 'symlink', 'foreign_owner', 'publication', 'active_release', 'source_intent_digest'])
def test_no_retirement_when_execution_or_source_cannot_be_proven(adopted, monkeypatch, corruption):
    config, source, owner, intent, path, reserve, run = adopted
    state = Path(config['progression_root'])/run.name/'cpu-robot-binding'
    if corruption == 'started':
        state.mkdir(parents=True)
    elif corruption == 'symlink':
        state.parent.mkdir(parents=True)
        state.symlink_to('/missing')
    elif corruption == 'publication':
        source['adoption'] = {**source['adoption'], 'extra': True}
    elif corruption == 'active_release':
        monkeypatch.setattr(identity, 'running_release_commit', lambda: 'c'*40)
    elif corruption == 'source_intent_digest':
        intent['intent_digest'] = 'sha256:'+'0'*64
        put(path, intent)
    else:
        owner = {**owner, 'intent_id': 'foreign'}
        config['controls_root'] = str(Path(config['controls_root'])/'..'/'controls')
        # The supplied owner must match the discovered adoption, not just its directory.
        old = Path(config['controls_root'])/'terminal-adoptions'
        (old/owner['intent_id']).symlink_to(next(old.iterdir()), target_is_directory=True)
    with pytest.raises((ValueError, OSError, KeyError)):
        retirement.retire_unmaterialized_adoptions(config=config, intent_id=owner['intent_id'],
            source=source, expected_production_commit='b'*40)
    directory = Path(config['scene_root'])/('scene-'+owner['intent_id'].removeprefix('scene-'))
    assert not any(p.name.startswith('controls-adopted') for p in (directory/cancellation.DIRECTORY).glob('*.json'))


def test_embedded_profile_override_requires_exact_cancelled_owner_source(adopted, monkeypatch):
    config, source, owner, intent, path, reserve, run = adopted
    monkeypatch.setattr(adoption, 'terminal_adoption_source', lambda **kwargs: source)
    original = (run/'launch_profile.json').read_bytes()
    profile, selected = auto._profile_intent(run, intent_path_override=path)
    assert selected == path and (run/'launch_profile.json').read_bytes() == original
    source['adoption'] = {**source['adoption'], 'extra': True}
    with pytest.raises(auto.TaskEvaluationConfiguredControlsAutostartError, match='adoption_profile_conflict'):
        auto._profile_intent(run, intent_path_override=path)
    source['adoption'] = intent['configuration_adoption']
    original_intent = Path(profile['immutable_inputs'][0]['path'])
    original_intent.write_text('{}')
    with pytest.raises(auto.TaskEvaluationConfiguredControlsAutostartError, match='intent_binding_invalid'):
        auto._profile_intent(run, intent_path_override=path)
