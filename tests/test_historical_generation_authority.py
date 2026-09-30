# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_authority.py
#   src/blueprint_pipeline/control_plane_lane_historical_generation.py
"""ADP-009D/day28: original historical bytes need a distinct owner decision."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_registered_experiment_issuer import installation, encoded  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_lane_scratch_decisions import _annotations, _inventory, _json


def selector(raw):
    return {'sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


@pytest.fixture
def historical_installation(installation):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    config, settings, _, policy = installation
    settings['owner_census_decisions_enabled'] = 1
    config.write_bytes(encoded(settings))
    policy.write_bytes(encoded(dict(schema_version=owners.POLICY_SCHEMA, enabled=True,
        principals=[dict(principal='operator', owners=['owner'],
            allowed_actions=['keep', 'register', 'delete', 'offload'], max_consent_seconds=3600)])))
    work = Path(settings['lane_scratch_work_root']).parent
    inputs = Path(settings['lane_scratch_inputs_root']).parent
    work.mkdir(mode=0o755)
    inputs.mkdir(mode=0o755)
    target = work / 'old-diagnostics'
    target.mkdir()
    (target / 'one.log').write_bytes(b'original owner diagnostics\n')
    state = Path(settings['state_root']) / 'requests'
    for name, lock in [('owner-consents', '.owner-consents.lock'),
                       ('historical-generation-actions', '.historical-generation.lock')]:
        directory = state / name
        directory.mkdir(mode=0o700)
        (directory / lock).write_bytes(b'')
        (directory / lock).chmod(0o600)
    row = dict(path=str(target), family='other', owner_guess='unowned',
        owner_guess_basis='no_owner_evidence', allocated_bytes=target.stat().st_blocks * 512,
        newest_mtime_epoch=None, age_seconds=None, unreadable=0, shared_names=0,
        references=[], owner_decision=None, approved_expiry=None)
    census = _json(_inventory([row]))
    annotations = _annotations(census, [dict(path=str(target), action='register', owner='owner',
        lane='diagnostics', name='old-diagnostics', reason='finished-diagnostic-run',
        class_intent='scratch', cleanup='owner_review', ttl_seconds=900, run_ref='run-1')])
    census_path, annotation_path = config.parent / 'census.json', config.parent / 'annotations.json'
    census_path.write_bytes(census)
    annotation_path.write_bytes(annotations)
    census_path.chmod(0o600)
    annotation_path.chmod(0o600)
    from blueprint_pipeline.control_plane_lane_scratch_decisions import validate_census_annotations
    assert validate_census_annotations(census, annotations, now=1000,
        allowed_roots=(str(work), str(inputs)))['status'] == 'validated'
    owners.issue_owner_consent(census_path, annotation_path,
        census_sha256=selector(census)['sha256'], census_size_bytes=len(census),
        annotations_sha256=selector(annotations)['sha256'], annotations_size_bytes=len(annotations),
        principal='operator', selected_paths=(str(target),), expires_at_epoch=1800,
        installed_config_path=config, now=1000, monotonic=lambda: 0)
    consent = next((state / 'owner-consents').glob('*.json'))
    return config, target, policy, state / 'historical-generation-actions', consent


def packet(installed):
    from blueprint_pipeline.control_plane_lane_historical_authority import issue_historical_packet
    config, target, _, _, consent = installed
    return issue_historical_packet(installed_config_path=config, selected_path=str(target),
        consent_id=consent.stem, consent_sha256=selector(consent.read_bytes())['sha256'],
        consent_size_bytes=consent.stat().st_size, now=1010, monotonic=lambda: 0)


def decision(installed, selected, **changes):
    from blueprint_pipeline.control_plane_lane_historical_authority import approve_historical_decommission
    options = dict(installed_config_path=installed[0],
        packet_id=selected['packet_id'], ack_packet_digest=selected['packet_digest'],
        principal='operator', owner='owner', action='delete', finished_run_ref='run-1',
        no_future_writers=True, no_future_readers=True, expires_at_epoch=1100,
        now=1020, monotonic=lambda: 0)
    return approve_historical_decommission(**(options | changes))


def test_original_packet_and_separate_owner_decision_preserve_payload(historical_installation):
    config, target, _, store, _ = historical_installation
    original = (target / 'one.log').read_bytes()
    selected = packet(historical_installation)
    assert selected['execution_authorized'] is False and selected['owner'] == 'owner'
    manifest = store / (selected['packet_id'] + '.manifest.json')
    raw = manifest.read_bytes()
    assert selected['manifest'] == selector(raw)
    inventory = json.loads(raw)
    assert inventory['members'][1]['sha256'] == selector(original)['sha256']
    approved = decision(historical_installation, selected)
    assert approved['decommission_approved'] is True and approved['action'] == 'delete'
    assert approved['packet_digest'] == selected['packet_digest']
    assert approved['generation_digest'] == inventory['generation_digest']
    assert approved['execution_authorized'] is False  # Action fencing remains required.
    assert (target / 'one.log').read_bytes() == original
    assert not (target / '.lane-scratch.v1.json').exists()
    assert (store / (approved['action_id'] + '.json')).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('change', ['payload', 'policy', 'packet', 'consent'])
def test_changed_generation_or_protected_authority_refuses_before_decision(historical_installation, change):
    _, target, policy, store, consent = historical_installation
    selected = packet(historical_installation)
    destination = {'payload': target / 'one.log', 'policy': policy,
                   'packet': store / (selected['packet_id'] + '.json'), 'consent': consent}[change]
    destination.write_bytes(destination.read_bytes() + b'changed')
    before = set(store.iterdir())
    with pytest.raises(ValueError):
        decision(historical_installation, selected)
    assert set(store.iterdir()) == before


@pytest.mark.parametrize('options', [
    {'no_future_writers': False}, {'no_future_readers': False},
    {'expires_at_epoch': 1020}, {'principal': 'another-operator'}, {'owner': 'another-owner'},
    {'action': 'keep'}, {'ack_packet_digest': 'sha256:' + 'f' * 64}, {'finished_run_ref': ''},
])
def test_decision_requires_exact_owner_ack_action_and_future_lifetime(historical_installation, options):
    selected = packet(historical_installation)
    store = historical_installation[3]
    before = {path.name: path.read_bytes() for path in store.iterdir()}
    with pytest.raises(ValueError):
        decision(historical_installation, selected, **options)
    assert {path.name: path.read_bytes() for path in store.iterdir()} == before


def test_owner_review_decision_grants_no_execution(historical_installation):
    selected = packet(historical_installation)
    approved = decision(historical_installation, selected, action='owner_review')
    assert approved['action'] == 'owner_review' and approved['execution_authorized'] is False
    assert (historical_installation[1] / 'one.log').read_bytes() == b'original owner diagnostics\n'


def test_decision_collision_preserves_the_original_packet(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_authority as feature
    selected = packet(historical_installation)
    store = historical_installation[3]
    before = {path.name: path.read_bytes() for path in store.iterdir()}
    monkeypatch.setattr(feature.secrets, 'token_hex', lambda _: selected['packet_id'])
    with pytest.raises(ValueError, match='publication_destination_exists'):
        decision(historical_installation, selected)
    assert {path.name: path.read_bytes() for path in store.iterdir()} == before


@pytest.mark.parametrize('elapsed', [6, 100])
def test_long_generation_hash_retains_original_expiry_and_separate_metadata_budgets(
        historical_installation, monkeypatch, elapsed):
    from blueprint_pipeline import control_plane_lane_historical_authority as feature
    selected = packet(historical_installation)
    value = [0.0]
    original = feature.generation.inventory_historical_generation

    def slow_read(*args, **kwargs):
        result = original(*args, **kwargs)
        value[0] += elapsed
        return result

    monkeypatch.setattr(feature.generation, 'inventory_historical_generation', slow_read)
    before = set(historical_installation[3].iterdir())
    if elapsed == 100:
        with pytest.raises(ValueError, match='decision_expired'):
            decision(historical_installation, selected, monotonic=lambda: value[0])
        assert set(historical_installation[3].iterdir()) == before
    else:
        approved = decision(historical_installation, selected, monotonic=lambda: value[0])
        assert approved['issued_at_epoch'] == 1026 and approved['expires_at_epoch'] == 1100


@pytest.mark.parametrize('unsafe', ['writable', 'missing', 'linked'])
def test_private_authority_store_cannot_be_rebound_or_invented(historical_installation, unsafe):
    store = historical_installation[3]
    if unsafe == 'writable':
        store.chmod(0o777)
    else:
        (store / '.historical-generation.lock').unlink()
        if unsafe == 'linked':
            (store / '.historical-generation.lock').symlink_to(historical_installation[4])
    with pytest.raises(ValueError):
        packet(historical_installation)
    assert not list(store.glob('*.json'))


def test_historical_records_use_guarded_publication_without_replacement(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    import os
    monkeypatch.setattr(owners, '_publish', lambda *a, **kw: pytest.fail('old publisher'))
    monkeypatch.setattr(os, 'replace', lambda *a, **kw: pytest.fail('replacement'))
    selected = packet(historical_installation)
    assert decision(historical_installation, selected)['decommission_approved'] is True


def test_configuration_drift_during_payload_read_refuses_final_publication(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_authority as feature
    selected = packet(historical_installation)
    original = feature.generation.inventory_historical_generation

    def drifting(*args, **kwargs):
        result = original(*args, **kwargs)
        config = historical_installation[0]
        config.write_bytes(config.read_bytes() + b' ')
        return result

    monkeypatch.setattr(feature.generation, 'inventory_historical_generation', drifting)
    before = set(historical_installation[3].iterdir())
    with pytest.raises(ValueError, match='source_changed'):
        decision(historical_installation, selected)
    assert set(historical_installation[3].iterdir()) == before
