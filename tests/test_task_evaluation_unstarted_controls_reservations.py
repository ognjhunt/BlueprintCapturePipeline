import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline import task_evaluation_configured_controls_autostart as autostart
from blueprint_pipeline import task_evaluation_unstarted_controls_reservations as cancellation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_task_evaluation_scene_intake import request, stage


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def reserved(tmp_path, monkeypatch):
    root = tmp_path / 'scene-intents'
    value = request()
    value['execution'].update(max_total_spend_usd=19.528870, max_paid_attempts=6, allowed_providers=['vast', 'openai'])
    owner = stage(root, value)
    def reserve(name, cost, provider='vast', commit='d'):
        return intake.reserve_scene_attempt(queue_root=root, intent_id=owner['intent_id'], attempt_id=name,
            source_commit=commit*40, runtime_digest='sha256:'+'e'*64, input_digest='sha256:'+'f'*64,
            maximum_spend_usd=cost, provider=provider, now=101)
    reserve('scene-configuration', 12)
    phases = {}
    originals = []
    for name, cost, provider in [('construction', .45, 'vast'), ('controls', .45, 'vast'), ('placement', 2.56, 'openai')]:
        attempt = reserve('controls-first-'+name, cost, provider)
        originals.append(attempt)
        if name != 'placement':
            auth_path = tmp_path / (name+'.json')
            put(auth_path, {'scene_owner_attempt': {'phase': name, **authority.bind_scene_attempt(attempt)}})
            phases[name] = {'authorization_path': str(auth_path)}
    auto = {'expected_production_commit': 'd'*40, 'configuration_adoption': {'mode': 'same_commit_automatic'}, 'phases': phases}
    auto_path = tmp_path / 'auto.json'
    auto_sha = put(auto_path, auto)
    # The production intent validator has its own complete inventory suite;
    # this fixture isolates reservation retirement and subsequent paid admission.
    monkeypatch.setattr(autostart, 'validate_configured_controls_autostart_intent', lambda v: v)
    run = tmp_path / 'launch'
    profile = {'source_commit': 'd'*40, 'task_evaluation_run': {'run_mode': 'scene_configuration'},
        'immutable_inputs': [{'name': 'configured_controls_autostart_intent', 'path': str(auto_path), 'digest': auto_sha}]}
    profile['profile_digest'] = canonical_digest(profile, digest_field='profile_digest')
    put(run / 'launch_profile.json', profile)
    launch = {'schema_version': 'task_evaluation_launch_receipt.v1', 'status': 'blocked', 'source_commit': 'd'*40,
        'launch_profile_digest': profile['profile_digest'], 'terminal_evidence': {'status': 'blocked'}}
    launch['receipt_digest'] = cross_runtime_canonical_digest(launch, digest_field='receipt_digest')
    put(run / 'launch_receipt.json', launch)
    monkeypatch.setenv(intake.ROOT_ENV, str(root))
    monkeypatch.setenv(intake.CLIENTS_ENV, 'webapp')
    return root, run, owner, reserve, originals


def test_retirement_preserves_originals_frees_only_unused_caps_and_rejects_stale_launch(reserved):
    root, run, owner, reserve, originals = reserved
    directory = root / owner['intent_id']
    before = {p: p.read_bytes() for p in (directory/'attempts').glob('*.json')}
    plan = cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root, dry_run=True)
    assert plan['status'] == 'would_cancel_unstarted_controls'
    assert not (directory/cancellation.DIRECTORY).exists()
    result = cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root)
    assert result['status'] == 'cancelled_unstarted_controls'
    assert len(result['cancellations']) == 3
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root) == result
    for old in originals:
        assert cancellation.validated_cancellation(directory, old)
        profile = {**authority.bind_scene_attempt(old), 'source_commit': old['source_commit']}
        assert authority.scene_execution_authority_blockers(profile, provider=old['provider'], maximum_spend_usd=old['maximum_spend_usd'], now=102) == ['scene_execution_owner_attempt_cancelled_before_execution']
    with pytest.raises(intake.SceneIntakeError, match='attempt_cancelled'):
        reserve('controls-first-construction', .45)
    reserve('controls-second-construction', .45, commit='c')
    reserve('controls-second-controls', .45, commit='c')
    reserve('controls-second-placement', 2.56, 'openai', commit='c')
    reserve('policy', 4, commit='c')
    with pytest.raises(intake.SceneIntakeError, match='spend_cap_exhausted'):
        reserve('extra', .10, commit='c')


@pytest.mark.parametrize('corruption', ['completed', 'profile_digest', 'intent_bytes'])
def test_refuses_eligible_or_changed_inputs_without_retiring_any_hold(reserved, corruption):
    root, run, owner, _reserve, _originals = reserved
    if corruption == 'intent_bytes':
        profile = json.loads((run/'launch_profile.json').read_text())
        Path(profile['immutable_inputs'][0]['path']).write_text('{}')
    else:
        path = run / ('launch_receipt.json' if corruption == 'completed' else 'launch_profile.json')
        d = json.loads(path.read_text())
        d['status' if corruption == 'completed' else 'profile_digest'] = 'completed' if corruption == 'completed' else 'sha256:'+'0'*64
        if corruption == 'completed':
            d['receipt_digest'] = cross_runtime_canonical_digest(d, digest_field='receipt_digest')
        put(path, d)
    with pytest.raises(ValueError):
        cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root)
    assert not list((root/owner['intent_id']/cancellation.DIRECTORY).glob('*.json'))


def test_tampered_cancellation_cannot_release_budget(reserved):
    root, run, owner, reserve, originals = reserved
    cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root)
    p = root/owner['intent_id']/cancellation.DIRECTORY/(originals[0]['attempt_id']+'.json')
    d = json.loads(p.read_text())
    d['maximum_spend_usd'] = 99
    p.chmod(0o640)
    put(p, d)
    with pytest.raises(ValueError, match='cancellation_invalid'):
        reserve('new', .45, commit='c')


def test_spend_refresh_releases_only_the_proven_unstarted_holds(reserved, tmp_path):
    from tests.test_task_evaluation_scene_spend import seed
    from blueprint_pipeline.task_evaluation_scene_spend import publish_current_scene_project_spend
    root, run, _owner, _reserve, _originals = reserved
    args = dict(scene_root=root, seed_reconciliation_path=seed(tmp_path), output_root=tmp_path/'spend', current_path=tmp_path/'current.json')
    before = publish_current_scene_project_spend(**args, now=102)
    cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root)
    after = publish_current_scene_project_spend(**args, now=103)
    assert before['total_cost_usd'] - after['total_cost_usd'] == pytest.approx(3.46)
    assert after['total_cost_usd'] == pytest.approx(43.197914 + 12)
    inventory = json.loads((Path(after['pointer']['path']).parent/'source_inventory.json').read_text())
    assert len(inventory['cancelled_before_controls_eligibility']) == 3


def test_registered_adoption_reopens_owner_authority_and_refuses_cancelled_holds(reserved, tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_controls_autoprovision as provision
    root, run, owner, _reserve, _originals = reserved
    monkeypatch.setattr(authority.time, 'time', lambda: 102)
    auto = json.loads((tmp_path/'auto.json').read_text())
    for phase in ('construction', 'controls'):
        p = tmp_path/(phase+'-launch-authority.json')
        put(p, {'max_spend_usd': .45})
        auto['phases'][phase]['launch_authority_path'] = str(p)
    auto.update(configuration_adoption={'mode': 'explicit_terminal_adoption'}, placement={'max_inference_cost_usd': 2.56}, intent_digest='sha256:'+'9'*64)
    registry = tmp_path/'registry'
    put(registry/'adoption-one.json', auto)
    args = dict(config={'scene_root': str(root), 'intent_root': str(registry)}, intent_id=owner['intent_id'], expected_production_commit='d'*40)
    assert provision._registered_terminal_adoption(**args)['status'] == 'installed_terminal_adoption'
    assert provision._registered_terminal_adoption(**{**args, 'expected_production_commit': 'c'*40}) is None
    cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root)
    with pytest.raises(ValueError, match='owner_authority_refused'):
        provision._registered_terminal_adoption(**args)
