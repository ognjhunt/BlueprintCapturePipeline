"""Protected original-generation packet and distinct historical owner decision.

ADP-009D/day28. These immutable records never enroll a registered producer or
clear readers. The later installed action fence must independently authorize
each mutation; issuing either record leaves every historical byte in place.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_legacy_owner as legacy
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_historical_publication import _publish
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

_ID = re.compile(r'[0-9a-f]{32}\Z')
_NAME = re.compile(r'[0-9a-f]{32}(?:\.manifest)?\.json\Z')
_LOCK = '.historical-generation.lock'
_MAX_RECORDS, _MAX_STORE_BYTES = 512, 64 * 1024**2


def _require(value, code):
    generation._require(value, 'authority_' + code)


def _selector(raw):
    return dict(sha256='sha256:' + hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _historical_reference_budget(**options):
    """Bound repeated root observations within one held historical transaction.

    Authority is read before/after; pins and reference configuration are read
    initially and at every held effect boundary. These count the same roots
    repeatedly. The reference fence still admits at most 16 configured table
    roots; the acquisition counter permits 32 observations, never a reset.
    The original five-second and all other resource bounds are unchanged.
    """
    budget = ReferenceCollectionBudget(**options)
    budget._limits['roots'] = 32
    return budget


class _Operation:
    def __init__(self, now, monotonic):
        _require(type(now) in (int, float) and math.isfinite(now) and now >= 0
                 and callable(monotonic), 'options_invalid')
        self.now, self.monotonic = now, monotonic
        self.started = self.last = monotonic()
        _require(type(self.started) in (int, float) and math.isfinite(self.started), 'deadline')

    def resume_original(self, started):
        """Retain the original monotonic deadline without rolling back wall time.

        Only the verified protected journal calls this; a retry cannot gain a
        fresh payload-read allowance from constructing another worker object.
        """
        _require(type(started) in (int, float) and math.isfinite(started)
                 and 0 <= started <= self.started, 'deadline')
        self.now -= self.started - started
        self.started = started
        self.remaining()

    def remaining(self):
        observed = self.monotonic()
        _require(type(observed) in (int, float) and math.isfinite(observed)
                 and self.last <= observed < self.started + generation.MAX_SECONDS, 'deadline')
        self.last = observed
        return self.started + generation.MAX_SECONDS - observed

    def moment(self):
        self.remaining()
        return self.now + self.last - self.started


class _Store:
    def __init__(self, files, config):
        self.files = files
        self.root = Path(config.owner_consent_store).parent / 'historical-generation-actions'
        self.parent, _ = files.parent(self.root / _LOCK, protected=True)
        owners._protected(os.fstat(self.parent), directory=True, mode=0o700)
        raw, acquired = files.read(self.root / _LOCK, cap=1, protected=True, mode=0o600)
        _require(raw == b'', 'store_unsafe')
        try:
            fcntl.flock(acquired.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise generation.HistoricalGenerationError('historical_generation_authority_store_busy') from None
        self.scan()

    def scan(self):
        count, size = 0, 0
        self.files.location(self.parent)
        with os.scandir(self.parent) as stream:
            for entry in stream:
                self.files.budget.charge('entries')
                count += 1
                _require(count <= _MAX_RECORDS + 1, 'store_full')
                info = os.stat(entry.name, dir_fd=self.parent, follow_symlinks=False)
                owners._protected(info, mode=0o600)
                _require(entry.name == _LOCK or _NAME.fullmatch(entry.name), 'store_unsafe')
                cap = generation.MAX_MANIFEST_BYTES if entry.name.endswith('.manifest.json') else 32768
                _require(0 <= info.st_size <= cap and (entry.name != _LOCK or info.st_size == 0), 'store_unsafe')
                size += info.st_size
                _require(size <= _MAX_STORE_BYTES, 'store_full')
        self.files.verify()
        return count - 1, size

    def read(self, identifier, *, manifest=False):
        _require(isinstance(identifier, str) and _ID.fullmatch(identifier), 'record_invalid')
        cap = generation.MAX_MANIFEST_BYTES if manifest else 32768
        name = identifier + ('.manifest.json' if manifest else '.json')
        raw, _ = self.files.read(self.root / name, cap=cap, protected=True, mode=0o600)
        value = self.files.document(raw, cap=cap)
        _require(type(value) is dict, 'record_invalid')
        return value, raw

    def publish(self, identifier, value, *, manifest=False):
        _require(isinstance(identifier, str) and _ID.fullmatch(identifier), 'record_invalid')
        cap = generation.MAX_MANIFEST_BYTES if manifest else 32768
        payload = owners._encoded(value, self.files.budget, cap=cap)
        count, size = self.scan()
        _require(count < _MAX_RECORDS and size + len(payload) <= _MAX_STORE_BYTES, 'store_full')
        name = identifier + ('.manifest.json' if manifest else '.json')
        _publish(self.files, self.parent, name, payload, kind='manifest' if manifest else 'private')
        observed, raw = self.read(identifier, manifest=manifest)
        _require(observed == value and raw == payload, 'record_changed')
        return _selector(raw)


class _HistoricalFiles(_BirthFiles):
    def __init__(self, budget):
        super().__init__(budget)
        # One original selection plus two complete publication comparisons and
        # caller readback can each consume a declared 1 MiB manifest. Reserve
        # another four such units for protected config/consent/event rereads.
        # The same actual counters, global 20 MiB cap and five-second deadline
        # remain conserved. Generic registered metadata retains its 2 MiB cap.
        self.raw_cap = 8 * generation.MAX_MANIFEST_BYTES
        self.metadata_records = {}
        self.document_nodes = {}
        self.document_bytes = 0

    def document(self, raw, *, cap):
        """Reuse only intrinsic validation of identical immutable JSON bytes.

        Store.read always acquires fresh protected bytes before this call. Each
        hit charges the complete allocation before creating a fresh graph;
        current selection, owner, expiry and inode checks still run normally.
        Neither a decoded object nor authorization survives a call.
        """
        self.budget.tick()
        _require(type(raw) is bytes and len(raw) <= cap, 'record_invalid')
        proof = self.document_nodes.get(raw)
        if proof is not None:
            nodes, encoded_bytes = proof
            _require(encoded_bytes <= cap, 'record_invalid')
            self.budget.charge('values', nodes)
            value = json.loads(raw)
            self.budget.tick()
            return value
        value = retained._document(raw, cap, _work_budget=self.budget)
        before = self.budget.counts['values']
        encoded_bytes = self.budget.measure(value, cap=cap)
        nodes = self.budget.counts['values'] - before
        if self.document_bytes + len(raw) <= self.raw_cap:
            self.document_nodes[raw] = (nodes, encoded_bytes)
            self.document_bytes += len(raw)
        self.budget.tick()
        return value

    def finish(self):
        try:
            super().finish()
        finally:
            self.document_nodes.clear()
            self.document_bytes = 0

    def read(self, path, *, cap, protected=False, mode=None):
        """Fresh bytes on the original retained FD, with unchanged full metadata.

        Repeated authority/head checks must not consume another live descriptor
        for the same inode. Raw bytes and observation entries are still charged;
        no decoded value or authorization is cached here.
        """
        parent, name = self.parent(path, protected=protected)
        record = self.metadata_records.get((parent, name))
        if record is None:
            raw, record = super().read(path, cap=cap, protected=protected, mode=mode)
            self.metadata_records[(parent, name)] = record
            return raw, record
        self.location(parent)
        self.verify_record(record)
        if protected:
            owners._protected(record.info, mode=mode)
        owners._require(0 <= record.info.st_size <= cap, 'owner_consent_resource_exhausted')
        os.lseek(record.fd, 0, os.SEEK_SET)
        raw = self.read_bytes(record.fd, cap)
        self.verify_record(record)
        owners._require(len(raw) == record.info.st_size, 'owner_consent_record_changed')
        self.budget.charge('entries')
        self.location(parent)
        return raw, record

    def close(self, fd):
        super().close(fd)
        if fd not in self.owned:
            for key in [key for key, record in self.metadata_records.items() if record.fd == fd]:
                del self.metadata_records[key]


@contextmanager
def _session(path, operation):
    _require(os.geteuid() == 0, 'root_required')
    budget = _historical_reference_budget(monotonic=operation.monotonic,
                                       time_budget_seconds=min(5, operation.remaining()))
    files = _HistoricalFiles(budget)
    try:
        config = owners._installed_config(files, path)
        store = _Store(files, config)
        yield files, config, store
        operation.remaining()
        files.verify()
    except OSError:
        raise generation.HistoricalGenerationError('historical_generation_authority_io_unavailable') from None
    finally:
        try:
            files.finish()
        finally:
            budget.close()


def _authority(files, config, path, consent, now):
    current = legacy._load_old_consent(files, files.budget, config,
        consent_id=consent['consent_id'], expected_sha256=consent['record_sha256'],
        expected_size_bytes=consent['record_size_bytes'], now=now)
    config_raw, _ = files.read(path, cap=owners.MAX_POLICY_BYTES, protected=True)
    policy = legacy._policy_bytes(files, config)
    return current, _selector(config_raw), _selector(policy), policy


def _old_selector(consent_id, sha256, size):
    return dict(consent_id=consent_id, record_sha256=sha256, record_size_bytes=size)


def issue_historical_packet(*, installed_config_path, selected_path, consent_id,
                            consent_sha256, consent_size_bytes, now, monotonic=time.monotonic):
    operation = _Operation(now, monotonic)
    selected = _old_selector(consent_id, consent_sha256, consent_size_bytes)
    with _session(installed_config_path, operation) as (files, config, _):
        old, config_selector, policy_selector, _ = _authority(
            files, config, installed_config_path, selected, operation.moment())
        decisions = [pair['decision'] for pair in old['decisions']
                     if pair['census_row']['path'] == selected_path]
        _require(len(decisions) == 1 and decisions[0]['action'] == 'register', 'owner_unproven')
        selected_decision = decisions[0]
        owner = selected_decision['owner']
        roots = owners._roots(config, files.budget)
    inventory = generation.inventory_historical_generation(selected_path, allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    identifier = secrets.token_hex(16)
    with _session(installed_config_path, operation) as (files, current, store):
        renewed, current_config, current_policy, _ = _authority(
            files, current, installed_config_path, selected, operation.moment())
        _require(renewed == old and current_config == config_selector
                 and current_policy == policy_selector, 'source_changed')
        manifest = store.publish(identifier, inventory, manifest=True)
        _require(operation.moment() < old['expires_at_epoch'], 'packet_expired')
        packet = dict(schema_version='control_plane_historical_generation_packet.v1',
            packet_id=identifier, owner=owner, principal=old['principal'], selected_path=selected_path,
            old_consent=selected | dict(consent_digest=old['consent_digest'],
                census=old['census'], annotations=old['annotations']),
            installed_config=config_selector, policy=policy_selector, manifest=manifest,
            generation_digest=inventory['generation_digest'], observed_at_epoch=operation.moment(),
            expires_at_epoch=min(old['expires_at_epoch'], operation.moment() + 900),
            execution_authorized=False, approval_required=True)
        packet['packet_digest'] = canonical_digest(packet, digest_field='packet_digest')
        store.publish(identifier, packet)
        return packet


def approve_historical_decommission(*, installed_config_path, packet_id, ack_packet_digest,
        principal, owner, action, finished_run_ref, no_future_writers, no_future_readers,
        expires_at_epoch, now, monotonic=time.monotonic):
    operation = _Operation(now, monotonic)
    _require(action in ('delete', 'offload', 'owner_review')
             and isinstance(finished_run_ref, str) and 0 < len(finished_run_ref) <= 256
             and finished_run_ref.isprintable()
             and no_future_writers is True and no_future_readers is True, 'decision_invalid')
    with _session(installed_config_path, operation) as (files, config, store):
        packet, _ = store.read(packet_id)
        _require(packet.get('schema_version') == 'control_plane_historical_generation_packet.v1'
            and packet.get('packet_id') == packet_id and packet.get('execution_authorized') is False
            and packet.get('packet_digest') == ack_packet_digest
            == canonical_digest(packet, digest_field='packet_digest')
            and packet['owner'] == owner and packet['principal'] == principal, 'packet_invalid')
        old, configured, policy_selector, policy = _authority(
            files, config, installed_config_path, packet['old_consent'], operation.moment())
        _require(configured == packet['installed_config'] and policy_selector == packet['policy']
                 and old['consent_digest'] == packet['old_consent']['consent_digest'], 'source_changed')
        _require(type(expires_at_epoch) in (int, float) and math.isfinite(expires_at_epoch)
                 and operation.moment() < expires_at_epoch <= min(packet['expires_at_epoch'], now + 900),
                 'decision_expired')
        approved_policy = owners._policy(policy, principal, files.budget)
        owners._authorize(dict(owner=owner, action='register' if action == 'owner_review' else action,
                              ttl_seconds=900), approved_policy, expires_at_epoch, now)
        inventory, raw = store.read(packet_id, manifest=True)
        _require(_selector(raw) == packet['manifest']
                 and inventory['generation_digest'] == packet['generation_digest']
                 == canonical_digest(inventory, digest_field='generation_digest'), 'manifest_changed')
        roots = owners._roots(config, files.budget)
    observed = generation.inventory_historical_generation(packet['selected_path'], allowed_roots=roots,
        max_seconds=operation.remaining(), monotonic=monotonic)
    _require(observed == inventory, 'generation_changed')
    with _session(installed_config_path, operation) as (files, config, store):
        current_old, current_config, current_policy, _ = _authority(
            files, config, installed_config_path, packet['old_consent'], operation.moment())
        _require(current_old == old and current_config == configured and current_policy == policy_selector,
                 'source_changed')
        current_packet, _ = store.read(packet_id)
        _, current_manifest = store.read(packet_id, manifest=True)
        _require(current_packet == packet and _selector(current_manifest) == packet['manifest'], 'record_changed')
        _require(operation.moment() < expires_at_epoch, 'decision_expired')
        identifier = secrets.token_hex(16)
        decision = dict(schema_version='control_plane_historical_decommission.v1', action_id=identifier,
            packet_id=packet_id, packet_digest=packet['packet_digest'],
            generation_digest=packet['generation_digest'], manifest=packet['manifest'],
            principal=principal, owner=owner, action=action, finished_run_ref=finished_run_ref,
            no_future_writers=True, no_future_readers=True, issued_at_epoch=operation.moment(),
            expires_at_epoch=expires_at_epoch, policy=policy_selector,
            decommission_approved=True, execution_authorized=False)
        decision['decision_digest'] = canonical_digest(decision, digest_field='decision_digest')
        store.publish(identifier, decision)
        return decision
