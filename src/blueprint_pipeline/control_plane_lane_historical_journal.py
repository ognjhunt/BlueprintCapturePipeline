"""Private immutable original-operation records for historical action recovery.

The authenticated action holds the original authority-store lock while using
this journal. No record grants execution or substitutes for native unit rights,
current owner authority, reader checks or the payload generation fence.
"""
from __future__ import annotations

import math
import json
import os
import re
import stat
from pathlib import Path

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from . import control_plane_lane_experiment_work as work
from .control_plane_lane_historical_publication import _publish
from .decision_evidence_contracts import canonical_digest

MAX_OPERATIONS = 512
MAX_EVENTS = 4 * generation.MAX_MEMBERS + 64
MAX_EVENT_BYTES = 4096
MAX_SUPPORTED_EVENT_BYTES = MAX_EVENT_BYTES + 3 * (6 * 1024 + 54)
MAX_JOURNAL_BYTES = 64 * 1024**2
_EVENT_NAME = re.compile(r'e-([0-9]{5})\.json\Z')
_KINDS = frozenset({'intent', 'fence_intent', 'fenced', 'preservation', 'removal_intent',
    'removed', 'removal_uncertain', 'final', 'restore_intent', 'restore_directory',
    'restore_member', 'restore_final', 'access_reopened'})
_FIELDS = frozenset({'schema_version', 'action_id', 'scope_digest', 'sequence', 'kind',
    'body', 'previous_event_digest', 'observed_at_epoch', 'execution_authorized', 'event_digest'})


def _require(value, code):
    generation._require(value, 'journal_' + code)


def journal_root(config):
    return Path(config.owner_consent_store).parent / 'historical-generation-journals'


def _scope(selected):
    packet, decision, manifest, records = selected
    return dict(action_id=decision['action_id'], action=decision['action'], owner=decision['owner'],
        packet=records['packet'], decision=records['decision'], manifest=decision['manifest'],
        generation_digest=manifest['generation_digest'], target_path=manifest['target_path'],
        policy=decision['policy'], installed_config=(decision['installed_config']
            if decision['action'] == 'restore' else packet['installed_config']),
        expires_at_epoch=decision['expires_at_epoch'])


def event_byte_limit(selected):
    """Restore birth records contain path, stage_path and parent_path.

    JSON escapes a control byte to six ASCII bytes. Keep an explicit finite
    envelope allowance, plus the exact longest selected path and stage prefix.
    This changes no journal count, aggregate byte or acquisition deadline cap.
    """
    paths = [row['path'] for row in selected[2]['members']]
    largest = max(len(json.dumps(path, ensure_ascii=False).encode('utf-8')) for path in paths)
    cap = MAX_EVENT_BYTES + 3 * (largest + 52)
    _require(MAX_EVENT_BYTES <= cap <= MAX_SUPPORTED_EVENT_BYTES, 'invalid')
    return cap


class HistoricalActionJournal:
    """One fixed protected action ID; append-only no-replace records and deadline."""

    def __init__(self, files, config, selected, operation):
        self.files, self.operation = files, operation
        packet, decision, manifest, records = selected
        self.action_id = decision['action_id']
        _require(authority._ID.fullmatch(self.action_id), 'invalid')
        self.scope = _scope(selected)
        self.event_bytes = event_byte_limit(selected)
        self.scope_digest = canonical_digest(self.scope)
        root = journal_root(config)
        parent, _ = files.parent(root / self.action_id, protected=True)
        owners._protected(os.fstat(parent), directory=True, mode=0o700)
        count = 0
        with os.scandir(parent) as stream:
            for row in stream:
                files.budget.charge('entries')
                count += 1
                _require(count <= MAX_OPERATIONS and authority._ID.fullmatch(row.name), 'store_unsafe')
                owners._protected(os.stat(row.name, dir_fd=parent, follow_symlinks=False),
                                  directory=True, mode=0o700)
        try:
            os.stat(self.action_id, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            _require(count < MAX_OPERATIONS, 'store_full')
            files.location(parent)
            os.mkdir(self.action_id, mode=0o700, dir_fd=parent)
            files.location(parent)
            os.fsync(parent)
        self.root = root / self.action_id
        self.directory, _ = files.parent(self.root / 'e-00000.json', protected=True)
        owners._protected(os.fstat(self.directory), directory=True, mode=0o700)
        self._count, self._size = self._scan()
        if self._count == 0:
            moment = operation.moment()
            body = dict(started_at_epoch=moment, started_monotonic=operation.started,
                        boot_id=work._controller_boot_id(files))
            self._publish('intent', body, 0, None)
        self._load()
        # Anonymous records were fully fsynced before their one-link birth.
        # A killed publisher may not have synced the directory. After exact
        # original intent/head validation, make that namespace durable before
        # this mutating journal can permit any subsequent effect. Read-only
        # HistoricalJournalObservation has its own constructor and does none.
        files.location(self.directory)
        _require(owners._metadata(os.fstat(self.directory)) == self._namespace, 'changed')
        os.fsync(self.directory)
        files.location(self.directory)
        _require(owners._metadata(os.fstat(self.directory)) == self._namespace, 'changed')

    def _scan(self):
        self.files.location(self.directory)
        before = owners._metadata(os.fstat(self.directory))
        count, size, maximum = 0, 0, -1
        with os.scandir(self.directory) as stream:
            for entry in stream:
                self.files.budget.charge('entries')
                match = _EVENT_NAME.fullmatch(entry.name)
                if entry.name == 'restore.snapshot.json':
                    _require(self.scope['action'] == 'restore', 'store_unsafe')
                    info = os.stat(entry.name, dir_fd=self.directory, follow_symlinks=False)
                    owners._protected(info, mode=0o600)
                    _require(0 < info.st_size <= generation.MAX_MANIFEST_BYTES, 'store_unsafe')
                    size += info.st_size
                    _require(size <= MAX_JOURNAL_BYTES, 'store_full')
                    continue
                _require(match and count < MAX_EVENTS, 'store_unsafe')
                info = os.stat(entry.name, dir_fd=self.directory, follow_symlinks=False)
                owners._protected(info, mode=0o600)
                _require(0 < info.st_size <= self.event_bytes, 'store_unsafe')
                count, size = count + 1, size + info.st_size
                maximum = max(maximum, int(match[1]))
                _require(size <= MAX_JOURNAL_BYTES, 'store_full')
        _require(maximum == count - 1, 'changed')
        self.files.location(self.directory)
        _require(owners._metadata(os.fstat(self.directory)) == before, 'changed')
        self._namespace = before
        return count, size

    def _read(self, index):
        path = self.root / f'e-{index:05d}.json'
        raw, record = self.files.read(path, cap=self.event_bytes, protected=True, mode=0o600)
        value = retained._document(raw, self.event_bytes, _work_budget=self.files.budget)
        _require(type(value) is dict and set(value) == _FIELDS
            and value['schema_version'] == 'control_plane_historical_action_event.v1'
            and value['action_id'] == self.action_id and value['scope_digest'] == self.scope_digest
            and type(value['sequence']) is int and value['sequence'] == index
            and isinstance(value['kind'], str) and value['kind'] in _KINDS and type(value['body']) is dict
            and type(value['observed_at_epoch']) in (int, float)
            and math.isfinite(value['observed_at_epoch']) and value['observed_at_epoch'] >= 0
            and (value['previous_event_digest'] is None if index == 0 else
                 isinstance(value['previous_event_digest'], str)
                 and re.fullmatch(r'sha256:[0-9a-f]{64}', value['previous_event_digest']))
            and value['execution_authorized'] is False
            and value['event_digest'] == canonical_digest(value, digest_field='event_digest'), 'changed')
        self.files.verify_record(record)
        return value

    def _load(self):
        # The authenticated caller retains the real authority EX lock. One
        # bounded namespace scan per metadata checkpoint leaves enough of the
        # original 20k-entry budget for the declared 16k immutable events.
        self.files.location(self.directory)
        _require(self.operation.moment() < self.scope['expires_at_epoch'], 'expired')
        _require(owners._metadata(os.fstat(self.directory)) == self._namespace, 'changed')
        count = self._count
        _require(count > 0, 'changed')
        intent = self._read(0)
        _require(intent['kind'] == 'intent' and intent['previous_event_digest'] is None
            and set(intent['body']) == {'started_at_epoch', 'started_monotonic', 'boot_id'}, 'changed')
        seed = intent['body']
        current = self.operation.monotonic()
        _require(all(type(seed[key]) in (int, float) and math.isfinite(seed[key])
                     for key in ('started_at_epoch', 'started_monotonic'))
            and type(current) in (int, float) and math.isfinite(current)
            and seed['started_monotonic'] <= current < seed['started_monotonic'] + generation.MAX_SECONDS
            and seed['started_at_epoch'] <= self.operation.moment()
            and seed['boot_id'] == work._controller_boot_id(self.files), 'deadline')
        self.operation.resume_original(seed['started_monotonic'])
        head = self._read(count - 1) if count > 1 else intent
        if count > 1:
            previous = self._read(count - 2)
            _require(head['kind'] != 'intent'
                and head['previous_event_digest'] == previous['event_digest']
                and previous['observed_at_epoch'] <= head['observed_at_epoch']
                <= self.operation.moment(), 'changed')
        return head

    @property
    def head(self):
        return self._load()

    def replay_batch(self, start, *, previous, observed_at, expected_head):
        """Read the whole chain in finite metadata checkpoints on the worker.

        A fresh checkpoint gets a new five-second acquisition budget while the
        verified original operation timer keeps running. The worker must start
        at zero and carry the observed link/time through every batch. No batch
        alone establishes recovery authority or clears current references.
        """
        _require(type(start) is int and 0 <= start < self._count
            and (previous is None and observed_at is None if start == 0 else
                 isinstance(previous, str) and re.fullmatch(r'sha256:[0-9a-f]{64}', previous)
                 and type(observed_at) in (int, float) and math.isfinite(observed_at)), 'invalid')
        _require(self.head['event_digest'] == expected_head, 'changed')
        values = []
        for index in range(start, min(self._count, start + 32)):
            self.operation.remaining()
            value = self._read(index)
            _require(value['previous_event_digest'] == previous
                and (value['kind'] == 'intent' if index == 0 else value['kind'] != 'intent')
                and (observed_at is None or observed_at <= value['observed_at_epoch'])
                and value['observed_at_epoch'] <= self.operation.moment(), 'changed')
            previous, observed_at = value['event_digest'], value['observed_at_epoch']
            values.append(value)
        end = start + len(values)
        _require(self.head['event_digest'] == expected_head, 'changed')
        return dict(events=values, next_start=end, previous=previous, observed_at=observed_at,
                    complete=end == self._count, event_count=self._count)

    def _publish(self, kind, body, sequence, previous, *, observation_expires_at_epoch=None):
        moment = self.operation.moment()
        _require(moment < self.scope['expires_at_epoch'], 'expired')
        if observation_expires_at_epoch is not None:
            _require(type(observation_expires_at_epoch) in (int, float)
                and math.isfinite(observation_expires_at_epoch)
                and moment < observation_expires_at_epoch <= self.scope['expires_at_epoch'],
                'observation_expired')
        value = dict(schema_version='control_plane_historical_action_event.v1', action_id=self.action_id,
            scope_digest=self.scope_digest, sequence=sequence, kind=kind, body=body,
            previous_event_digest=previous, observed_at_epoch=moment,
            execution_authorized=False)
        value['event_digest'] = canonical_digest(value, digest_field='event_digest')
        raw = owners._encoded(value, self.files.budget, cap=self.event_bytes)
        _publish(self.files, self.directory, f'e-{sequence:05d}.json', raw, kind='event')
        _require(self._read(sequence) == value, 'changed')
        self._count += 1
        self._size += len(raw)
        self._namespace = owners._metadata(os.fstat(self.directory))
        return value

    def append(self, kind, body, *, previous, observation_expires_at_epoch=None):
        _require(kind in _KINDS - {'intent'} and type(body) is dict, 'invalid')
        head = self._load()
        _require(previous == head['event_digest'], 'changed')
        count, size = self._count, self._size
        _require(count < MAX_EVENTS and size <= MAX_JOURNAL_BYTES - self.event_bytes, 'store_full')
        return self._publish(kind, body, count, previous,
                             observation_expires_at_epoch=observation_expires_at_epoch)

    def publish_restore_snapshot(self, snapshot):
        """Retain an actual private restored inventory; this grants no access."""
        self._load()
        _require(self.scope['action'] == 'restore' and type(snapshot) is dict
            and snapshot.get('schema_version') == 'control_plane_historical_generation.v1'
            and snapshot.get('target_path') == self.scope['target_path']
            and snapshot.get('execution_authorized') is False
            and snapshot.get('generation_digest') == canonical_digest(snapshot, digest_field='generation_digest')
            and snapshot['members'][0]['version'][2:5] == [stat.S_IFDIR | 0o700, 0, 0], 'snapshot_invalid')
        raw = owners._encoded(snapshot, self.files.budget, cap=generation.MAX_MANIFEST_BYTES)
        _require(self._size + len(raw) <= MAX_JOURNAL_BYTES, 'store_full')
        _publish(self.files, self.directory, 'restore.snapshot.json', raw, kind='historical_restore_snapshot')
        self._namespace = owners._metadata(os.fstat(self.directory))
        self._size += len(raw)
        selector = authority._selector(raw)
        _require(self.read_restore_snapshot(selector) == snapshot, 'changed')
        return selector

    def read_restore_snapshot(self, selector):
        self._load()
        _require(self.scope['action'] == 'restore', 'snapshot_invalid')
        raw, record = self.files.read(self.root / 'restore.snapshot.json',
            cap=generation.MAX_MANIFEST_BYTES, protected=True, mode=0o600)
        _require(authority._selector(raw) == selector, 'snapshot_changed')
        value = retained._document(raw, generation.MAX_MANIFEST_BYTES, _work_budget=self.files.budget)
        _require(type(value) is dict and value.get('execution_authorized') is False
            and value.get('target_path') == self.scope['target_path']
            and value.get('generation_digest') == canonical_digest(value, digest_field='generation_digest'),
            'snapshot_changed')
        self.files.verify_record(record)
        return value

    def select_restore_snapshot(self):
        """Observe fixed protected snapshot bytes before a final can select them.

        This supplies an immutable byte selector, never authority or completion.
        The worker still proves original member creations, current owner/clock,
        exact private inode inventory and full original cloud/local readback.
        """
        self._load()
        _require(self.scope['action'] == 'restore', 'snapshot_invalid')
        raw, record = self.files.read(self.root / 'restore.snapshot.json',
            cap=generation.MAX_MANIFEST_BYTES, protected=True, mode=0o600)
        selector = authority._selector(raw)
        self.files.verify_record(record)
        snapshot = self.read_restore_snapshot(selector)
        self.files.verify_record(record)
        return selector, snapshot


class HistoricalJournalObservation:
    """Read past immutable facts without adopting their expired write authority.

    The caller authenticates the original selected records under the current
    protected-store lock. This object cannot create, append or resume an old
    operation. Restore must obtain a distinct current owner approval and fence.
    Every batch uses the new observer's original bounded read deadline.
    """

    _scan = HistoricalActionJournal._scan
    _read = HistoricalActionJournal._read
    replay_batch = HistoricalActionJournal.replay_batch

    def __init__(self, files, config, selected, operation):
        self.files, self.operation = files, operation
        self.action_id = selected[1]['action_id']
        _require(isinstance(self.action_id, str) and authority._ID.fullmatch(self.action_id), 'invalid')
        self.scope = _scope(selected)
        self.event_bytes = event_byte_limit(selected)
        self.scope_digest = canonical_digest(self.scope)
        self.root = journal_root(config) / self.action_id
        self.directory, _ = files.parent(self.root / 'e-00000.json', protected=True)
        owners._protected(os.fstat(self.directory), directory=True, mode=0o700)
        self._count, self._size = self._scan()
        _require(self._count > 0, 'changed')
        self.head

    @property
    def head(self):
        self.operation.remaining()
        self.files.location(self.directory)
        _require(owners._metadata(os.fstat(self.directory)) == self._namespace, 'changed')
        intent = self._read(0)
        seed = intent['body']
        _require(intent['kind'] == 'intent' and intent['previous_event_digest'] is None
            and set(seed) == {'started_at_epoch', 'started_monotonic', 'boot_id'}
            and all(type(seed[key]) in (int, float) and math.isfinite(seed[key]) and seed[key] >= 0
                    for key in ('started_at_epoch', 'started_monotonic'))
            and isinstance(seed['boot_id'], str) and re.fullmatch(
                r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', seed['boot_id'])
            and seed['started_at_epoch'] <= intent['observed_at_epoch'] <= self.operation.moment(), 'changed')
        head = self._read(self._count - 1) if self._count > 1 else intent
        _require(intent['observed_at_epoch'] <= head['observed_at_epoch'] <= self.operation.moment(), 'changed')
        return head

    def read_restore_snapshot(self, selector):
        """Read protected bytes without loading or adopting a mutation clock."""
        self.head
        _require(self.scope['action'] == 'restore', 'snapshot_invalid')
        raw, record = self.files.read(self.root / 'restore.snapshot.json',
            cap=generation.MAX_MANIFEST_BYTES, protected=True, mode=0o600)
        _require(authority._selector(raw) == selector, 'snapshot_changed')
        value = retained._document(raw, generation.MAX_MANIFEST_BYTES, _work_budget=self.files.budget)
        _require(type(value) is dict and value.get('execution_authorized') is False
            and value.get('target_path') == self.scope['target_path']
            and value.get('generation_digest') == canonical_digest(value, digest_field='generation_digest'),
            'snapshot_changed')
        self.files.verify_record(record)
        self.head
        return value
