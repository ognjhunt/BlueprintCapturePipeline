"""ADP-009D/day28: authenticated expired experiments reach actual storage GC.

Tiny development-only Mac fixtures exercise the real issuer, native birth and
existing timer entrypoint. Root metadata is the existing hermetic fixture; it
does not satisfy the separate real foreign-UID/Linux acceptance gate. No G1
completion, descendant closure or archive proof is fabricated here.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   src/blueprint_pipeline/control_plane_lane_experiment_birth.py

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation, issue, encoded  # noqa: F401
from tests.test_registered_experiment_birth import birth, prepare


def _selector(path):
    raw = Path(path).read_bytes()
    return {'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def _payload_snapshot(target):
    return {str(path.relative_to(target)): (
        path.stat().st_dev, path.stat().st_ino, path.stat().st_mode,
        path.read_bytes() if path.is_file() else None)
        for path in (target, *target.rglob('*'))}


def _current_entry(installation, intent_id):  # noqa: F811
    public = installation[2].parents[1] / 'experiment-authority'
    head = json.loads((public / 'HEAD.json').read_bytes())
    version = public / head['record_name']
    assert _selector(version) == head['record']
    authority = json.loads(version.read_bytes())
    return next(row for row in authority['enrollments'] if row['intent_id'] == intent_id)


def _selected_private_record(store, selector):
    assert set(selector) == {'sha256', 'size_bytes'}
    matches = [path for path in store.rglob('*.json') if _selector(path) == selector]
    assert len(matches) == 1
    return json.loads(matches[0].read_bytes())


@pytest.fixture
def retirement_installation(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as birth_code
    from blueprint_pipeline import control_plane_lane_scratch_retention as observer
    config, settings, _, policy_path = installation
    settings['experiment_retirement_enabled'] = True
    gc_environment = config.parent / 'gc.env'
    gc_environment.write_text('BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=' + str(config.parent / 'pins') + '\n')
    gc_environment.chmod(0o600)
    settings['experiment_gc_environment_file'] = str(gc_environment)
    config.write_bytes(encoded(settings))
    policy = json.loads(policy_path.read_bytes())
    # Preserve the existing policy grammar: owner_review selects KEEP, not a
    # newly invented fifth action in the shared owner policy.
    policy['principals'][0]['allowed_actions'] = ['register', 'delete', 'offload', 'keep']
    policy_path.write_bytes(encoded(policy))
    prepare(installation)
    monkeypatch.setattr(birth_code, '_blueprint_identity', lambda: (0, 0))
    monkeypatch.setattr(os, 'fchown', lambda *args: None)
    roots = (settings['lane_scratch_work_root'], settings['lane_scratch_inputs_root'])
    monkeypatch.setattr(observer, '_ALLOWED_ROOTS', frozenset(roots))
    return installation


def _born_scratch(installation):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_scratch as native
    grant = issue(installation)
    born = birth(installation, grant)
    target = Path(born['path'])
    (target / 'intermediate.bin').write_bytes(b'tiny disposable result')
    (target / 'nested').mkdir()
    (target / 'nested' / 'log.txt').write_bytes(b'local log\n')
    lease = json.loads((target / native.LEASE_FILE).read_bytes())
    assert lease['class_intent'] == 'scratch' and lease['cleanup'] == 'delete'
    assert lease['expires_at_epoch'] == 2800
    entry = _current_entry(installation, grant['intent_id'])
    assert entry['generation'] == born['generation'] and entry['state'] == 'active'
    assert entry['completion'] is None  # Authentic disposable profile needs no G1 proof.
    return grant, born, target


def _issue_action(installation, grant, *, action='delete'):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    return root.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
        action=action, expires_at_epoch=3500, installed_config_path=installation[0], now=lambda: 2900)


def _gc(installation, *, enabled=True, at=2900):  # noqa: F811
    from blueprint_pipeline.control_plane_storage_gc import run_storage_gc, RUN_ACK
    pins = installation[0].parent / 'pins'
    pins.mkdir(exist_ok=True)
    # The timer selects protected current-entry operation IDs itself. No action
    # IDs, target paths or caller safety Boolean are forwarded to the action.
    return run_storage_gc(content_store_roots=(), derived_roots=(), queue_roots=(), pins_root=pins,
        apply=True, ack=RUN_ACK, lane_scratch_roots=(installation[1]['lane_scratch_work_root'],
                                                  installation[1]['lane_scratch_inputs_root']),
        lane_scratch_enabled=enabled, _experiment_config_path=installation[0], now=lambda: at)


def _no_archive(monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_scene_object_store as remote
    monkeypatch.setattr(remote, 'publish_configured_scene_stream',
        lambda *args, **kwargs: pytest.fail('scratch/unknown/owner_review attempted archive upload'))


def test_authentic_expired_scratch_reaches_existing_gc_and_durable_retired_receipt(
        retirement_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    from blueprint_pipeline import control_plane_lane_scratch as native
    installation = retirement_installation  # noqa: F811
    grant, born, target = _born_scratch(installation)
    original_inode = target.stat().st_ino
    metadata = {name: (target / name).read_bytes()
                for name in (native.LEASE_FILE, '.registered-experiment.v1.json')}
    action = _issue_action(installation, grant)
    assert action['action_id'] and action['action_intent']['size_bytes'] > 0
    _no_archive(monkeypatch)
    report = _gc(installation)
    assert report['opt_in']['lane_scratch'] is True
    phase = report['registered_experiments']
    assert phase['enabled'] is True
    outcomes = [row for row in phase['outcomes'] if row['action_id'] == action['action_id']]
    assert len(outcomes) == 1 and outcomes[0]['decision'] == 'retired'
    assert outcomes[0]['intent_id'] == grant['intent_id']
    assert not (target / 'intermediate.bin').exists(), report
    assert not (target / 'nested').exists(), report
    assert target.stat().st_ino == original_inode  # The original tombstone survives.
    assert {name: (target / name).read_bytes() for name in metadata} == metadata
    entry = _current_entry(installation, grant['intent_id'])
    assert entry['state'] == 'retired' and entry['generation'] == born['generation']
    retired_events = [json.loads(path.read_bytes()) for path in installation[2].rglob('*.json')
        if json.loads(path.read_bytes()).get('event_kind') == 'retired']
    assert len(retired_events) == 1
    event = retired_events[0]
    assert _selected_private_record(installation[2], outcomes[0]['receipt']) == event
    assert event['intent_id'] == grant['intent_id'] and event['generation'] == born['generation']
    assert event['body']['partial'] is False
    assert event['body']['removed_logical_bytes'] == len(b'tiny disposable result') + len(b'local log\n')
    assert event['body']['preservation'] is None  # Delete has no restoration archive.
    # Exact replay must select the durable outcome rather than unlink again or
    # invent another owner action. Its receipt alone is not a disk-free reading.
    before = _payload_snapshot(target)
    receipt = root.run_registered_experiment_action(action['action_id'],
        expected_action_intent=action['action_intent'], installed_config_path=installation[0], now=lambda: 2901)
    assert receipt['decision'] == 'retired' and receipt['receipt'] == outcomes[0]['receipt']
    assert _payload_snapshot(target) == before


@pytest.mark.parametrize('disabled', ['gc_opt_in', 'installed_retirement'])
def test_existing_gc_never_applies_when_either_enablement_is_off(
        retirement_installation, monkeypatch, disabled):
    installation = retirement_installation  # noqa: F811
    grant, _, target = _born_scratch(installation)
    _issue_action(installation, grant)
    if disabled == 'installed_retirement':
        config, settings, _, _ = installation
        config.write_bytes(encoded(settings | {'experiment_retirement_enabled': False}))
    before = _payload_snapshot(target)
    _no_archive(monkeypatch)
    report = _gc(installation, enabled=disabled != 'gc_opt_in')
    assert report['registered_experiments']['enabled'] is False
    assert _payload_snapshot(target) == before
    assert _current_entry(installation, grant['intent_id'])['state'] != 'retired'


def test_unknown_and_legacy_expired_folders_are_kept_by_actual_enabled_gc(
        retirement_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_scratch as native
    installation = retirement_installation  # noqa: F811
    root = Path(installation[1]['lane_scratch_work_root'])
    unknown = root / 'g1' / 'unowned-old-output'
    unknown.mkdir()
    (unknown / 'payload.bin').write_bytes(b'unknown retained bytes')
    legacy = native.create_lane_scratch('g1', 'old-disposable', root=root, owner='owner',
        reason='historical_scratch', class_intent='scratch', cleanup='delete', ttl_seconds=10,
        run_ref='old-run', now=lambda: 1000)
    (legacy / 'old.log').write_bytes(b'expired does not imply authenticated birth')
    before = {path: _payload_snapshot(path) for path in (unknown, legacy)}
    _no_archive(monkeypatch)
    report = _gc(installation)
    assert report['registered_experiments']['outcomes'] == []
    assert {path: _payload_snapshot(path) for path in before} == before
    assert not list(installation[2].glob('operations/*/e-*.json'))


def test_authentic_expired_experiment_without_owner_action_is_kept(
        retirement_installation, monkeypatch):
    installation = retirement_installation  # noqa: F811
    grant, _, target = _born_scratch(installation)
    before = _payload_snapshot(target)
    _no_archive(monkeypatch)
    report = _gc(installation)
    assert report['registered_experiments']['outcomes'] == []
    assert _payload_snapshot(target) == before
    assert _current_entry(installation, grant['intent_id'])['state'] == 'active'


def test_owner_review_has_no_payload_or_provider_mutation(
        retirement_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    installation = retirement_installation  # noqa: F811
    grant, _, target = _born_scratch(installation)
    action = _issue_action(installation, grant, action='owner_review')
    before = _payload_snapshot(target)
    _no_archive(monkeypatch)
    receipt = root.run_registered_experiment_action(action['action_id'],
        expected_action_intent=action['action_intent'], installed_config_path=installation[0], now=lambda: 2900)
    assert receipt['decision'] == 'kept' and receipt['removed_logical_bytes'] == 0
    assert receipt['removed_allocated_bytes'] == 0
    assert _payload_snapshot(target) == before
    _gc(installation)
    assert _payload_snapshot(target) == before


@pytest.mark.parametrize('review_expiry', [3000, 3900])
def test_owner_review_actions_do_not_starve_later_valid_delete(retirement_installation, review_expiry):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    installation = retirement_installation  # noqa: F811
    candidates = sorted((_born_scratch(installation) for _ in range(3)),
                        key=lambda value: value[0]['intent_id'])
    reviews = []
    for grant, _, _ in candidates[:2]:
        reviews.append(root.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
            action='owner_review', expires_at_epoch=review_expiry, installed_config_path=installation[0],
            now=lambda: 2900))
    grant, _, target = candidates[2]
    action = root.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
        action='delete', expires_at_epoch=3900, installed_config_path=installation[0], now=lambda: 2900)
    report = _gc(installation, at=3501)
    outcomes = report['registered_experiments']['outcomes']
    assert any(row['action_id'] == action['action_id'] and row['intent_id'] == grant['intent_id']
               and row['decision'] == 'retired' and row['reason'] == 'disposable_expired'
               for row in outcomes)
    if review_expiry == 3000:
        assert {row['action_id'] for row in outcomes if row['reason'] == 'experiment_action_expired'} == {
            review['action_id'] for review in reviews}
    else:
        assert len(outcomes) == 2  # One delete and at most one live owner review.
    assert not (target / 'intermediate.bin').exists()


@pytest.mark.parametrize('bad_index', [1, 2])
def test_malformed_unrelated_action_does_not_hide_valid_cleanup(retirement_installation, bad_index):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    installation = retirement_installation  # noqa: F811
    candidates = sorted((_born_scratch(installation) for _ in range(3)),
                        key=lambda value: value[0]['intent_id'])
    expected = []
    for index, (grant, _, target) in enumerate(candidates):
        action = root.issue_experiment_action_intent(grant['intent_id'], principal='operator', owner='owner',
            action='owner_review' if index == bad_index else 'delete', expires_at_epoch=3500,
            installed_config_path=installation[0], now=lambda: 2900)
        expected.append((action, target, _payload_snapshot(target)))
    invalid, invalid_target, before = expected[bad_index]
    (installation[2] / (invalid['action_id'] + '.action.json')).write_bytes(b'{')
    report = _gc(installation, at=2901)
    outcomes = report['registered_experiments']['outcomes']
    for index, (action, target, _) in enumerate(expected):
        matches = [row for row in outcomes if row['action_id'] == action['action_id']]
        assert len(matches) == 1
        if index == bad_index:
            assert matches[0]['decision'] == 'kept' and matches[0]['reason'] == 'experiment_action_invalid'
        else:
            assert matches[0]['decision'] == 'retired' and matches[0]['reason'] == 'disposable_expired'
            assert not (target / 'intermediate.bin').exists()
    assert _payload_snapshot(invalid_target) == before


def test_malformed_current_authority_still_aborts_all_registered_actions(retirement_installation):
    installation = retirement_installation  # noqa: F811
    grant, _, target = _born_scratch(installation)
    _issue_action(installation, grant)
    before = _payload_snapshot(target)
    public = installation[2].parents[1] / 'experiment-authority'
    (public / 'HEAD.json').write_bytes(b'{')
    report = _gc(installation, at=2901)
    assert report['registered_experiments']['enabled'] is False
    assert report['registered_experiments']['outcomes'] == []
    assert _payload_snapshot(target) == before


@pytest.mark.parametrize('bad_field', ['manifest', 'issuer_uid', 'issued_at_epoch', 'policy', 'controller', 'principal'])
def test_parseable_invalid_actions_do_not_starve_valid_cleanup(retirement_installation, bad_field):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    installation = retirement_installation  # noqa: F811
    candidates = sorted((_born_scratch(installation) for _ in range(3)),
                        key=lambda value: value[0]['intent_id'])
    actions = []
    for index, (grant, _, _) in enumerate(candidates):
        actions.append(root.issue_experiment_action_intent(grant['intent_id'], principal='operator',
            owner='owner', action='delete', expires_at_epoch=3400 if index < 2 else 3500,
            installed_config_path=installation[0], now=lambda: 2900))
    for action in actions[:2]:
        path = installation[2] / (action['action_id'] + '.action.json')
        document = json.loads(path.read_bytes())
        if bad_field == 'manifest':
            document['manifest'] = None
        elif bad_field == 'issuer_uid':
            document['issuer_uid'] = 1
        elif bad_field == 'issued_at_epoch':
            document['issued_at_epoch'] = 3300
        elif bad_field == 'policy':
            document['policy']['sha256'] = 'sha256:' + '0' * 64
        elif bad_field == 'principal':
            document['principal'] = 'unmapped-operator'
        else:
            document['controller']['boot_id'] = '00000000-0000-0000-0000-000000000000'
        document['action_digest'] = canonical_digest(document, digest_field='action_digest')
        path.write_bytes(encoded(document))
    report = _gc(installation, at=2901)
    outcomes = report['registered_experiments']['outcomes']
    last_action = actions[2]
    assert any(row['action_id'] == last_action['action_id'] and row['decision'] == 'retired'
               and row['reason'] == 'disposable_expired' for row in outcomes)
    assert not (candidates[2][2] / 'intermediate.bin').exists()
    assert all((target / 'intermediate.bin').exists() for _, _, target in candidates[:2])


def test_valid_multi_principal_policy_does_not_exhaust_gc_selection_budget(retirement_installation):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    installation = retirement_installation  # noqa: F811
    policy_path = installation[3]
    policy = json.loads(policy_path.read_bytes())
    policy['principals'].extend(policy['principals'][0] | {'principal': f'operator-{index}'}
                                for index in range(1, 32))
    policy_path.write_bytes(encoded(policy))
    candidates = [_born_scratch(installation) for _ in range(3)]
    issued = [root.issue_experiment_action_intent(grant['intent_id'], principal='operator',
              owner='owner', action='delete', expires_at_epoch=3500,
              installed_config_path=installation[0], now=lambda: 2900)
              for grant, _, _ in candidates]
    report = _gc(installation, at=2901)
    outcomes = report['registered_experiments']['outcomes']
    assert sum(row['decision'] == 'retired' for row in outcomes) == 2, report['registered_experiments']
    assert {row['action_id'] for row in outcomes} <= {action['action_id'] for action in issued}


def test_valid_maximum_owner_policy_still_allows_actual_retirement(retirement_installation):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    installation = retirement_installation  # noqa: F811
    policy_path = installation[3]
    policy = json.loads(policy_path.read_bytes())
    policy['principals'].extend(policy['principals'][0] | {'principal': f'operator-{index}'}
                                for index in range(1, 64))
    policy_path.write_bytes(encoded(policy))
    grant, _, target = _born_scratch(installation)
    action = root.issue_experiment_action_intent(grant['intent_id'], principal='operator',
        owner='owner', action='delete', expires_at_epoch=3500,
        installed_config_path=installation[0], now=lambda: 2900)
    report = _gc(installation, at=2901)
    assert any(row['action_id'] == action['action_id'] and row['decision'] == 'retired'
               for row in report['registered_experiments']['outcomes']), report['registered_experiments']
    assert not (target / 'intermediate.bin').exists()


@pytest.mark.parametrize('fault', ['alternate_empty_root', 'duplicate', 'relative'])
def test_actual_action_refuses_wrong_or_ambiguous_installed_reference_authority(
        retirement_installation, monkeypatch, fault):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    installation = retirement_installation  # noqa: F811
    grant, _, target = _born_scratch(installation)
    action = _issue_action(installation, grant)
    before = _payload_snapshot(target)
    pins = installation[0].parent / 'pins'
    pins.mkdir()
    environment = Path(installation[1]['experiment_gc_environment_file'])
    if fault == 'alternate_empty_root':
        pins = installation[0].parent / 'wrong-empty-pins'
        pins.mkdir()
    elif fault == 'duplicate':
        with environment.open('a') as output:
            output.write('BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=' + str(pins) + '\n')
    else:
        environment.write_text('BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=relative/path\n')
    with pytest.raises(ValueError, match='experiment_reference_configuration'):
        root.run_registered_experiment_action(action['action_id'], expected_action_intent=action['action_intent'],
            installed_config_path=installation[0], now=lambda: 2900, _pins_root=pins)
    assert _payload_snapshot(target) == before
