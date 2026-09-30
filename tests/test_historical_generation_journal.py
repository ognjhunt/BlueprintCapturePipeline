"""ADP-009D/day28: immutable original historical operation recovery records."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_journal.py
import json

import pytest

from tests.test_historical_generation_authority import decision, historical_installation, packet  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401

# Imported pytest fixtures intentionally retain their dependent names.
# ruff: noqa: F811


def journal_call(installed, action, callback, *, now=1030, monotonic=lambda: 0):
    from blueprint_pipeline import control_plane_lane_historical_authority as authority
    from blueprint_pipeline.control_plane_lane_historical_dispatch import _selection
    from blueprint_pipeline.control_plane_lane_historical_journal import HistoricalActionJournal
    operation = authority._Operation(now, monotonic)
    with authority._session(installed[0], operation) as (files, config, store):
        selected = _selection(files, config, store, installed[0], action['action_id'], operation.moment())
        journal = HistoricalActionJournal(files, config, selected, operation)
        return callback(journal)


def provision(installed):
    root = installed[3].parent / 'historical-generation-journals'
    assert root.is_dir()
    return root


def test_original_operation_survives_reopen_without_rewriting_or_touching_payload(historical_installation):
    root = provision(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    original = (historical_installation[1] / 'one.log').read_bytes()
    started = journal_call(historical_installation, approved, lambda journal: journal.head)
    assert started['kind'] == 'intent' and started['sequence'] == 0
    assert started['body']['started_at_epoch'] == 1030
    assert started['execution_authorized'] is False
    saved = {path.name: path.read_bytes() for path in (root / approved['action_id']).iterdir()}
    observed = journal_call(historical_installation, approved, lambda journal: journal.head, now=1040)
    assert observed == started
    assert saved == {path.name: path.read_bytes() for path in (root / approved['action_id']).iterdir()}
    assert (historical_installation[1] / 'one.log').read_bytes() == original


def test_event_publication_is_no_replace_and_binds_original_action_chain(historical_installation, monkeypatch):
    root = provision(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    monkeypatch.setattr('os.replace', lambda *args, **kwargs: pytest.fail('replacement forbidden'))
    intent = journal_call(historical_installation, approved, lambda journal: journal.head)
    event = journal_call(historical_installation, approved,
        lambda journal: journal.append('fence_intent', {'member_index': 0}, previous=intent['event_digest']))
    assert event['sequence'] == 1 and event['previous_event_digest'] == intent['event_digest']
    assert event['scope_digest'] == intent['scope_digest']
    path = root / approved['action_id'] / 'e-00001.json'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='journal_changed'):
        journal_call(historical_installation, approved,
            lambda journal: journal.append('fenced', {'member_index': 0}, previous=intent['event_digest']))
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['mode', 'symlink', 'unknown', 'gap', 'digest'])
def test_changed_or_unsafe_journal_keeps_every_historical_byte(historical_installation, change):
    root = provision(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    journal_call(historical_installation, approved, lambda journal: journal.head)
    directory = root / approved['action_id']
    path = directory / 'e-00000.json'
    if change == 'mode':
        path.chmod(0o644)
    elif change == 'symlink':
        content = path.read_bytes()
        path.unlink()
        (directory / 'unexpected').write_bytes(content)
        path.symlink_to(directory / 'unexpected')
    elif change == 'unknown':
        (directory / 'unexpected').write_bytes(b'unknown')
    elif change == 'gap':
        path.rename(directory / 'e-00002.json')
    else:
        value = json.loads(path.read_bytes())
        value['body']['started_at_epoch'] = 1040
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        journal_call(historical_installation, approved, lambda journal: journal.head)
    assert (historical_installation[1] / 'one.log').read_bytes() == b'original owner diagnostics\n'


def test_reopen_does_not_restart_original_monotonic_operation(historical_installation):
    provision(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    journal_call(historical_installation, approved, lambda journal: journal.head)
    with pytest.raises(ValueError, match='journal_deadline'):
        journal_call(historical_installation, approved, lambda journal: journal.head,
                     monotonic=lambda: 4 * 3600)


def test_reopened_operation_hash_budget_is_bound_to_original_start(historical_installation):
    approved = decision(historical_installation, packet(historical_installation))
    journal_call(historical_installation, approved, lambda journal: journal.head,
                 monotonic=lambda: 100)
    remaining, moment = journal_call(historical_installation, approved,
        lambda journal: (journal.operation.remaining(), journal.operation.moment()),
        now=1040, monotonic=lambda: 120)
    assert remaining == 4 * 3600 - 20
    assert moment == 1040  # Binding the old timer must not roll back current time.


def test_recovery_reads_every_link_not_only_the_last_two(historical_installation):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    approved = decision(historical_installation, packet(historical_installation))
    def add(journal):
        head = journal.head
        for index in range(4):
            head = journal.append('fence_intent', {'member_index': index}, previous=head['event_digest'])
        return head
    head = journal_call(historical_installation, approved, add)
    directory = provision(historical_installation) / approved['action_id']
    early = directory / 'e-00001.json'
    value = json.loads(early.read_bytes())
    value['body']['member_index'] = 99
    value['event_digest'] = canonical_digest(value, digest_field='event_digest')
    early.write_text(json.dumps(value))
    def replay(journal):
        return journal.replay_batch(0, previous=None, observed_at=None,
                                    expected_head=head['event_digest'])
    with pytest.raises(ValueError, match='journal_changed'):
        journal_call(historical_installation, approved, replay)


def test_published_event_survives_interruption_without_replacement(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_journal as journal_code
    root = provision(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    intent = journal_call(historical_installation, approved, lambda journal: journal.head)
    real = journal_code._publish

    def interrupted(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError('simulated interruption after durable publication')

    with monkeypatch.context() as interrupted_context:
        interrupted_context.setattr(journal_code, '_publish', interrupted)
        with pytest.raises(RuntimeError):
            journal_call(historical_installation, approved,
                lambda journal: journal.append('fence_intent', {'member_index': 0},
                                              previous=intent['event_digest']))
    path = root / approved['action_id'] / 'e-00001.json'
    before = path.read_bytes()
    head = journal_call(historical_installation, approved, lambda journal: journal.head)
    assert head['sequence'] == 1 and head['kind'] == 'fence_intent'
    assert path.read_bytes() == before
    assert (historical_installation[1] / 'one.log').is_file()


def test_reboot_or_current_namespace_change_cannot_adopt_old_journal(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_work as work
    provision(historical_installation)
    approved = decision(historical_installation, packet(historical_installation))
    journal_call(historical_installation, approved, lambda journal: journal.head)
    with monkeypatch.context() as rebooted:
        rebooted.setattr(work, '_controller_boot_id', lambda files: 'different-boot')
        with pytest.raises(ValueError, match='journal_deadline'):
            journal_call(historical_installation, approved, lambda journal: journal.head)

    def changed(journal):
        before = journal.head
        (journal.root / 'unknown').write_bytes(b'concurrent namespace change')
        return journal.append('fence_intent', {'member_index': 0}, previous=before['event_digest'])

    with pytest.raises(ValueError, match='journal_changed'):
        journal_call(historical_installation, approved, changed)
