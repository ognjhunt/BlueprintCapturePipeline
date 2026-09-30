"""Private immutable original-operation records for historical action recovery.

The authenticated action holds the original authority-store lock while using
this journal. No record grants execution or substitutes for native unit rights,
current owner authority, reader checks or the payload generation fence.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

from . import control_plane_lane_historical_authority as authority
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from . import control_plane_lane_experiment_work as work
from .control_plane_lane_experiment_publication import _publish
from .decision_evidence_contracts import canonical_digest

MAX_OPERATIONS = 512
MAX_EVENTS = 4 * generation.MAX_MEMBERS + 64
MAX_EVENT_BYTES = 4096
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


class HistoricalActionJournal:
    """One fixed protected action ID; append-only no-replace records and deadline."""

    def __init__(self, files, config, selected, operation):
        self.files, self.operation = files, operation
        packet, decision, manifest, records = selected
        self.action_id = decision['action_id']
        _require(authority._ID.fullmatch(self.action_id), 'invalid')
        self.scope = dict(action_id=self.action_id, action=decision['action'], owner=decision['owner'],
            packet=records['packet'], decision=records['decision'], manifest=decision['manifest'],
            generation_digest=manifest['generation_digest'], target_path=manifest['target_path'],
            policy=decision['policy'], installed_config=packet['installed_config'],
            expires_at_epoch=decision['expires_at_epoch'])
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

    def _scan(self):
        self.files.location(self.directory)
        before = owners._metadata(os.fstat(self.directory))
        count, size, maximum = 0, 0, -1
        with os.scandir(self.directory) as stream:
            for entry in stream:
                self.files.budget.charge('entries')
                match = _EVENT_NAME.fullmatch(entry.name)
                _require(match and count < MAX_EVENTS, 'store_unsafe')
                info = os.stat(entry.name, dir_fd=self.directory, follow_symlinks=False)
                owners._protected(info, mode=0o600)
                _require(0 < info.st_size <= MAX_EVENT_BYTES, 'store_unsafe')
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
        raw, record = self.files.read(path, cap=MAX_EVENT_BYTES, protected=True, mode=0o600)
        value = retained._document(raw, MAX_EVENT_BYTES, _work_budget=self.files.budget)
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

    def _publish(self, kind, body, sequence, previous):
        _require(self.operation.moment() < self.scope['expires_at_epoch'], 'expired')
        value = dict(schema_version='control_plane_historical_action_event.v1', action_id=self.action_id,
            scope_digest=self.scope_digest, sequence=sequence, kind=kind, body=body,
            previous_event_digest=previous, observed_at_epoch=self.operation.moment(),
            execution_authorized=False)
        value['event_digest'] = canonical_digest(value, digest_field='event_digest')
        raw = owners._encoded(value, self.files.budget, cap=MAX_EVENT_BYTES)
        _publish(self.files, self.directory, f'e-{sequence:05d}.json', raw, kind='event')
        _require(self._read(sequence) == value, 'changed')
        self._count += 1
        self._size += len(raw)
        self._namespace = owners._metadata(os.fstat(self.directory))
        return value

    def append(self, kind, body, *, previous):
        _require(kind in _KINDS - {'intent'} and type(body) is dict, 'invalid')
        head = self._load()
        _require(previous == head['event_digest'], 'changed')
        count, size = self._count, self._size
        _require(count < MAX_EVENTS and size <= MAX_JOURNAL_BYTES - MAX_EVENT_BYTES, 'store_full')
        return self._publish(kind, body, count, previous)
