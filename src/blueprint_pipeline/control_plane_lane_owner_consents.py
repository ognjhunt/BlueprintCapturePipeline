"""Protected administrative owner intent; no target-generation or action authority."""
from __future__ import annotations

import errno
import math
import os
import re
import secrets
import stat
from pathlib import Path
from typing import NamedTuple

from . import control_plane_lane_scratch_decisions as retained
from .decision_evidence_contracts import canonical_digest

POLICY_SCHEMA = 'control_plane_lane_owner_policy.v1'
CONSENT_SCHEMA = 'control_plane_lane_owner_consent.v1'
REPORT_SCHEMA = 'control_plane_lane_owner_decision_report.v1'
MAX_POLICY_BYTES = 64 * 1024
MAX_RECORD_BYTES = 512 * 1024
MAX_SELECTED = 100
MAX_PRINCIPALS = 64
MAX_STORE_RECORDS = 256
MAX_STORE_BYTES = 64 * 1024 * 1024
MAX_DESCRIPTOR_COUNT = 128
_PRINCIPAL = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z')
_OWNER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z')
_DIGEST = re.compile(r'sha256:[0-9a-f]{64}\Z')
_CONSENT_ID = re.compile(r'[0-9a-f]{32}\Z')
_ROW_FIELDS = frozenset({'path', 'family', 'owner_guess', 'owner_guess_basis', 'allocated_bytes',
                        'newest_mtime_epoch', 'age_seconds', 'unreadable', 'shared_names',
                        'references', 'owner_decision', 'approved_expiry'})
_FAMILIES = frozenset({'g1', 'drawer', 'arena', 'content-agent', 'gaussian-excision', 'scene', 'other'})
_REFERENCES = frozenset({'process', 'queue', 'pin', 'live_release', 'active_run'})
_SCOPE = 'selected_owner_decisions_from_retained_census'


class OwnerCensusConsentError(ValueError):
    """A fixed public code; input, policy and exception text remain private."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnerCensusConsentError(code)


def _matches(value, pattern):
    return isinstance(value, str) and len(value) <= 80 and pattern.fullmatch(value) is not None


def _number(value):
    return (type(value) in (int, float) and (type(value) is not int or value.bit_length() <= 63)
            and math.isfinite(value) and value >= 0)


def _counter(value, *, positive=False):
    return type(value) is int and (0 < value if positive else 0 <= value) and value <= 2**63 - 1


def _identity(raw, digest, size, budget):
    _require(isinstance(raw, bytes) and _matches(digest, _DIGEST) and _counter(size, positive=True),
             'owner_consent_options_invalid')
    budget.tick()
    _require(len(raw) == size and retained._digest(raw, _work_budget=budget) == digest,
             'owner_consent_input_identity_mismatch')


def _policies(raw, budget):
    try:
        value = retained._document(raw, MAX_POLICY_BYTES, _work_budget=budget)
    except retained.CensusDecisionError:
        raise OwnerCensusConsentError('owner_consent_policy_invalid') from None
    _require(set(value) == {'schema_version', 'enabled', 'principals'}
             and value['schema_version'] == POLICY_SCHEMA and type(value['enabled']) is bool,
             'owner_consent_policy_invalid')
    _require(value['enabled'], 'owner_consent_disabled')
    rows = value['principals']
    _require(isinstance(rows, list) and len(rows) <= MAX_PRINCIPALS, 'owner_consent_policy_invalid')
    selected = {}
    for row in rows:
        budget.charge('groups')
        _require(isinstance(row, dict) and set(row) == {'principal', 'owners', 'allowed_actions', 'max_consent_seconds'},
                 'owner_consent_policy_invalid')
        name, owners, actions, duration = (row[k] for k in ('principal', 'owners', 'allowed_actions', 'max_consent_seconds'))
        _require(_matches(name, _PRINCIPAL) and name not in selected, 'owner_consent_policy_invalid')
        _require(isinstance(owners, list) and 0 < len(owners) <= MAX_PRINCIPALS,
                 'owner_consent_policy_invalid')
        budget.charge('groups')
        budget.charge('entries', len(owners))
        _require(all(_matches(owner, _OWNER) for owner in owners) and len(set(owners)) == len(owners),
                 'owner_consent_policy_invalid')
        _require(isinstance(actions, list) and 0 < len(actions) <= 4
                 and all(isinstance(action, str) and action in retained.ACTIONS for action in actions)
                 and len(set(actions)) == len(actions)
                 and type(duration) is int and 1 <= duration <= 1209600, 'owner_consent_policy_invalid')
        selected[name] = row
    return selected


def _policy(raw, principal, budget):
    selected = _policies(raw, budget).get(principal)
    _require(selected is not None, 'owner_consent_principal_unmapped')
    return selected


def _row(row, roots, budget, code):
    budget.charge('entries')
    _require(isinstance(row, dict) and set(row) == _ROW_FIELDS, code)
    _require(isinstance(row['family'], str) and row['family'] in _FAMILIES
             and isinstance(row['owner_guess_basis'], str)
             and row['owner_guess_basis'] in {'name_prefix', 'no_owner_evidence'}, code)
    guess = row['owner_guess']
    _require(isinstance(guess, str) and 0 < len(guess) <= 256 and guess.isprintable(), code)
    budget.measure(guess, cap=258)
    _require(len(guess.encode('utf-8')) <= 256
             and all(_counter(row[k]) for k in ('allocated_bytes', 'shared_names', 'unreadable'))
             and row['unreadable'] == 0
             and all(row[k] is None or _number(row[k]) for k in ('newest_mtime_epoch', 'age_seconds'))
             and row['owner_decision'] is row['approved_expiry'] is None, code)
    try:
        path = retained._path(row['path'], _work_budget=budget)
    except retained.CensusDecisionError:
        raise OwnerCensusConsentError(code) from None
    _require(path not in roots and any(root in path.parents for root in roots), code)
    refs = row['references']
    _require(isinstance(refs, list) and len(refs) <= 5
             and all(isinstance(ref, str) and ref in _REFERENCES for ref in refs)
             and len(set(refs)) == len(refs), code)
    budget.charge('entries', len(refs))


def _decision(decision, row, roots, issued, budget, code):
    _require(isinstance(decision, dict) and decision.get('path') == row['path']
             and decision.get('references') == row['references'], code)
    budget.measure(decision, cap=MAX_RECORD_BYTES)
    # Normalized consent metadata adds only the retained references field.
    metadata = {key: value for key, value in decision.items() if key != 'references'}
    _require(isinstance(metadata.get('action'), str) and metadata['action'] in retained.ACTIONS, code)
    if 'size_budget_bytes' in metadata:
        _require(_counter(metadata['size_budget_bytes'], positive=True), code)
    try:
        retained._decision_metadata(metadata, roots, issued, _work_budget=budget)
    except retained.CensusDecisionError:
        raise OwnerCensusConsentError(code) from None
    _require(not row['references'] or metadata['action'] not in ('offload', 'delete'), code)


def _authorize(decision, policy, expiry, issued):
    _require(decision['owner'] in policy['owners'], 'owner_consent_owner_unmapped')
    _require(decision['action'] in policy['allowed_actions'], 'owner_consent_action_unapproved')
    ceiling = issued + policy['max_consent_seconds']
    if decision['action'] == 'keep':
        ceiling = min(ceiling, decision['expires_at_epoch'])
    elif decision['action'] == 'register':
        ceiling = min(ceiling, issued + decision['ttl_seconds'])
    _require(_number(expiry) and issued < expiry <= ceiling, 'owner_consent_expiry_invalid')


def _seal(record, budget):
    budget.available('output_bytes', budget.measure(record, cap=MAX_RECORD_BYTES - 100) + 100)
    budget.tick()
    digest = canonical_digest(record, digest_field='consent_digest')
    budget.tick()
    return record | {'consent_digest': digest}


def _build_consent(*, census_bytes, annotation_bytes, census_sha256, census_size_bytes,
                   annotations_sha256, annotations_size_bytes, policy_bytes, principal,
                   selected_paths, expires_at_epoch, now, allowed_roots, consent_id, budget):
    """Owned acquisition adapter supplies charged bytes; no publication or targets."""
    budget.tick()
    _require(_number(now) and _matches(principal, _PRINCIPAL) and _matches(consent_id, _CONSENT_ID),
             'owner_consent_options_invalid')
    for raw, cap in ((census_bytes, retained.MAX_JSON_BYTES),
                     (annotation_bytes, retained.MAX_JSON_BYTES), (policy_bytes, MAX_POLICY_BYTES)):
        _require(isinstance(raw, bytes) and 0 < len(raw) <= cap,
                 "owner_consent_inventory_invalid")
        budget.tick()
        try:
            budget.preflight(raw.decode("utf-8"))
        except UnicodeError:
            raise OwnerCensusConsentError("owner_consent_inventory_invalid") from None
    _identity(census_bytes, census_sha256, census_size_bytes, budget)
    _identity(annotation_bytes, annotations_sha256, annotations_size_bytes, budget)
    _require(isinstance(selected_paths, (list, tuple)) and 0 < len(selected_paths) <= MAX_SELECTED
             and all(isinstance(path, str) for path in selected_paths), 'owner_consent_selection_invalid')
    for path in selected_paths:
        budget.charge('entries')
        try:
            _require(len(path) <= retained.MAX_PATH_BYTES, 'owner_consent_selection_invalid')
            _require(len(path.encode('utf-8')) <= retained.MAX_PATH_BYTES, 'owner_consent_selection_invalid')
            retained._path(path, _work_budget=budget)
        except (retained.CensusDecisionError, UnicodeError):
            raise OwnerCensusConsentError('owner_consent_selection_invalid') from None
    _require(len(set(selected_paths)) == len(selected_paths), 'owner_consent_selection_invalid')
    policy = _policy(policy_bytes, principal, budget)
    try:
        validation = retained._validate_census_annotations(census_bytes, annotation_bytes, now=now,
                        allowed_roots=allowed_roots, _work_budget=budget)
        census = retained._document(census_bytes, retained.MAX_JSON_BYTES, _work_budget=budget)
        roots = tuple(retained._path(str(root), _work_budget=budget) for root in allowed_roots)
        for row in census['rows']:
            _row(row, roots, budget, 'owner_consent_inventory_invalid')
    except retained.CensusDecisionError:
        raise OwnerCensusConsentError('owner_consent_inventory_invalid') from None
    indexed = {}
    for row in census['rows']:
        budget.charge('entries')
        indexed[row['path']] = row
    decisions = {}
    for decision in validation['decisions']:
        budget.charge('entries')
        decisions[decision['path']] = decision
    _require(all(path in indexed for path in selected_paths), 'owner_consent_selection_invalid')
    pairs = []
    retained_size = 1024  # Bounded fixed framing, digests, counters and seal.
    _require(retained_size <= MAX_RECORD_BYTES, "owner_consent_resource_exhausted")
    budget.tick()
    for path in sorted(selected_paths):
        budget.charge('facts')
        decision, row = decisions[path], indexed[path]
        _decision(decision, row, roots, now, budget, 'owner_consent_inventory_invalid')
        _authorize(decision, policy, expires_at_epoch, now)
        pair = {"decision": decision, "census_row": row}
        retained_size += budget.measure(pair, cap=MAX_RECORD_BYTES - retained_size) + 2
        _require(retained_size <= MAX_RECORD_BYTES, "owner_consent_resource_exhausted")
        budget.retain(decision)
        budget.retain(row)
        budget.charge("output_bytes", 30)
        pairs.append(pair)
    budget.tick()
    record = dict(schema_version=CONSENT_SCHEMA, consent_id=consent_id,
        issuer_kind='local_root_administrative_attestation', issuer_uid=0, principal=principal,
        policy_sha256=retained._digest(policy_bytes, _work_budget=budget), policy_size_bytes=len(policy_bytes),
        issued_at_epoch=now, expires_at_epoch=expires_at_epoch,
        census={'sha256': census_sha256, 'size_bytes': census_size_bytes},
        annotations={'sha256': annotations_sha256, 'size_bytes': annotations_size_bytes},
        inventory_count=validation['decision_count'], selected_count=len(pairs), scope=_SCOPE, decisions=pairs,
        execution_authorized=False, target_generation_bound=False, requires_fresh_reference_check=True, mutations=0)
    return _seal(record, budget)


class _Acquired(NamedTuple):
    fd: int
    parent: int
    name: str
    info: object


def _metadata(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _security(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)


def _protected(info, *, directory=False, mode=None):
    _require(info.st_uid == 0 and (mode is None or info.st_gid == 0) and not info.st_mode & 0o022
             and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
             and (directory or info.st_nlink == 1)
             and (mode is None or stat.S_IMODE(info.st_mode) == mode), 'owner_consent_store_unsafe')


class _Files:
    """Invocation-owned descriptors; no finalizer consults an expired budget."""
    def __init__(self, budget, *, raw_cap=None):
        from .control_plane_reference_budget import ReferenceCollectionBudget
        _require(type(budget) is ReferenceCollectionBudget, "owner_consent_options_invalid")
        self.anchors = []
        self.budget, self.raw_cap = budget, raw_cap
        self.owned, self.edges, self.records = {}, [], []
        self.unresolved = 0

    def slot(self):
        self.budget.tick()
        _require(len(self.owned) < MAX_DESCRIPTOR_COUNT, 'owner_consent_resource_exhausted')

    def adopt(self, fd):
        # Register ownership before the first identity call; do not guess on failure.
        self.owned[fd] = None
        try:
            info = os.fstat(fd)
        except OSError:
            self.owned.pop(fd, None)
            self.unresolved += 1
            raise OwnerCensusConsentError('owner_consent_descriptor_ownership_unproven') from None
        self.owned[fd] = (info.st_dev, info.st_ino)
        return info

    def open(self, name, flags, *, parent=None, mode=0o600):
        self.slot()
        try:
            fd = os.open(name, flags, mode, dir_fd=parent)
        except OSError as exc:
            code = "owner_consent_store_unsafe" if exc.errno in (errno.ELOOP, errno.ENOTDIR) else "owner_consent_io_failed"
            raise OwnerCensusConsentError(code) from None
        self.adopt(fd)
        return fd

    def close(self, fd):
        expected = self.owned.get(fd)
        if expected is None:
            return
        for _ in range(2):
            try:
                info = os.fstat(fd)
            except OSError as exc:
                if exc.errno == errno.EBADF:
                    self.owned.pop(fd, None)
                    return
                continue
            if (info.st_dev, info.st_ino) != expected:
                self.owned.pop(fd, None)
                self.unresolved += 1
                return
            try:
                os.close(fd)
            except OSError:
                continue
            self.owned.pop(fd, None)
            return

    def finish(self):
        for fd in list(self.owned):
            self.close(fd)
        _require(not self.unresolved, 'owner_consent_descriptor_ownership_unproven')
        _require(not self.owned, 'owner_consent_descriptor_cleanup_failed')

    def parent(self, path, *, protected=False):
        text = os.fspath(path)
        try:
            retained._path(text, _work_budget=self.budget)
        except (retained.CensusDecisionError, UnicodeError):
            raise OwnerCensusConsentError('owner_consent_options_invalid') from None
        self.budget.charge('roots')
        fd = self.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_info = os.fstat(fd)
        if protected:
            _protected(root_info, directory=True)
        self.anchors.append((fd, _security(root_info), protected))
        for name in Path(text).parts[1:-1]:
            self.budget.charge('entries')
            child = self.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, parent=fd)
            info = os.fstat(child)
            if protected:
                _protected(info, directory=True)
            self.edges.append((fd, name, child, _security(info), protected))
            fd = child
        return fd, Path(text).name

    def read_bytes(self, fd, cap):
        pieces, size = [], 0
        while True:
            self.budget.tick()
            remaining = self.budget.limits['raw_bytes'] - self.budget.counts['raw_bytes']
            if self.raw_cap is not None:
                remaining = min(remaining, self.raw_cap - self.budget.counts['raw_bytes'])
            _require(remaining > 0, 'owner_consent_resource_exhausted')
            amount = min(65536, cap + 1 - size, remaining)
            _require(amount > 0, 'owner_consent_resource_exhausted')
            self.budget.available('raw_bytes', amount)
            part = os.read(fd, amount)
            self.budget.tick()
            if not part:
                break
            self.budget.charge('raw_bytes', len(part))
            size += len(part)
            _require(size <= cap, 'owner_consent_resource_exhausted')
            pieces.append(part)
        self.budget.tick()
        return b''.join(pieces)

    def read(self, path, *, cap, protected=False, mode=None):
        parent, name = self.parent(path, protected=protected)
        fd = self.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, parent=parent)
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), 'owner_consent_store_unsafe')
        if protected:
            _protected(before, mode=mode)
        _require(0 <= before.st_size <= cap, 'owner_consent_resource_exhausted')
        record = _Acquired(fd, parent, name, before)
        raw = self.read_bytes(fd, cap)
        _require(len(raw) == before.st_size and _metadata(os.fstat(fd)) == _metadata(before),
                 'owner_consent_record_changed')
        self.budget.charge('entries')
        self.records.append(record)
        self.verify_record(record)
        return raw, record

    def verify_record(self, record):
        self.budget.tick()
        _require(_metadata(os.fstat(record.fd)) == _metadata(record.info)
                 and _metadata(os.stat(record.name, dir_fd=record.parent, follow_symlinks=False))
                 == _metadata(record.info), 'owner_consent_record_changed')
        self.budget.tick()

    def verify(self):
        for fd, expected, protected in self.anchors:
            self.budget.charge("entries")
            opened = os.fstat(fd)
            named_root = os.stat("/", follow_symlinks=False)
            _require(_security(named_root) == expected == _security(opened), "owner_consent_record_changed")
            if protected:
                _protected(opened, directory=True)
        for parent, name, fd, expected, protected in self.edges:
            self.budget.charge('entries')
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(fd)
            _require(_security(named) == expected == _security(opened)
                     and stat.S_ISDIR(named.st_mode), 'owner_consent_record_changed')
            if protected:
                _protected(named, directory=True)
                _protected(opened, directory=True)
        for record in self.records:
            self.budget.charge('entries')
            self.verify_record(record)


INSTALLED_PACKAGE_ROOT = Path('/opt/blueprint/operator-door')
_STORE_LOCK = '.owner-consents.lock'
MAX_CONSUMPTION_RAW = 576 * 1024


def _roots(config, budget):
    roots = []
    for text in (config.lane_scratch_work_root, config.lane_scratch_inputs_root):
        budget.charge('roots')
        try:
            path = retained._path(text, _work_budget=budget)
        except retained.CensusDecisionError:
            raise OwnerCensusConsentError('owner_consent_config_invalid') from None
        _require(path.name == 'lanes', 'owner_consent_config_invalid')
        roots.append(path.parent)
    _require(roots[0] != roots[1] and roots[0] not in roots[1].parents
             and roots[1] not in roots[0].parents, 'owner_consent_config_invalid')
    return tuple(roots)


def _installed_config(files, path):
    """Load only acquired protected bytes from the fixed installed config bridge."""
    import sys
    import types
    config_raw, _ = files.read(path, cap=MAX_POLICY_BYTES, protected=True)
    package = INSTALLED_PACKAGE_ROOT / 'operator_door'
    try:
        files.read(package / '__init__.py', cap=MAX_POLICY_BYTES, protected=True)
        source, acquired = files.read(package / 'config.py', cap=MAX_POLICY_BYTES, protected=True)
    except OwnerCensusConsentError as error:
        if error.code in ('owner_consent_io_failed', 'owner_consent_store_unsafe'):
            raise OwnerCensusConsentError('owner_consent_installed_bridge_invalid') from None
        raise
    files.verify()
    name = '_blueprint_owner_installed_' + secrets.token_hex(8)
    module = types.ModuleType(name)
    module.__file__ = str(package / 'config.py')
    sys.modules[name] = module
    try:
        files.budget.tick()
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        files.budget.tick()
        _require(module.__file__ == str(package / 'config.py')
                 and callable(getattr(module, 'config_from_mapping', None)),
                 'owner_consent_installed_bridge_invalid')
        mapping = retained._document(config_raw, MAX_POLICY_BYTES, _work_budget=files.budget)
        config = module.config_from_mapping(mapping, _work_budget=files.budget)
    except (SyntaxError, UnicodeError, ImportError, AttributeError):
        raise OwnerCensusConsentError('owner_consent_installed_bridge_invalid') from None
    except retained.CensusDecisionError:
        raise OwnerCensusConsentError('owner_consent_config_invalid') from None
    except ValueError as error:
        if type(error).__name__ == 'DoorConfigError':
            raise OwnerCensusConsentError('owner_consent_config_invalid') from None
        raise
    finally:
        sys.modules.pop(name, None)
    files.verify_record(acquired)
    _require(config.owner_census_decisions_enabled == 1, 'owner_consent_disabled')
    _roots(config, files.budget)
    return config


def _store(files, config, consent_id, *, lock=False):
    parent, _ = files.parent(Path(config.owner_consent_store) / (consent_id + '.json'), protected=True)
    _protected(os.fstat(parent), directory=True, mode=0o700)
    if lock:
        import fcntl
        fd = files.open(_STORE_LOCK, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, parent=parent)
        info = os.fstat(fd)
        _protected(info, mode=0o600)
        _require(info.st_size == 0, 'owner_consent_store_unsafe')
        files.records.append(_Acquired(fd, parent, _STORE_LOCK, info))
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OwnerCensusConsentError('owner_consent_store_busy') from None
        # Close-only release: all owned descriptors finish after publication.
    return parent


def _capacity(files, parent):
    count, records, occupied = 0, 0, 0
    files.slot()  # scandir owns one temporary descriptor while its context is open.
    with os.scandir(parent) as entries:
        for entry in entries:
            files.budget.charge('entries')
            count += 1
            _require(count <= MAX_STORE_RECORDS + 1, 'owner_consent_store_full')
            info = os.stat(entry.name, dir_fd=parent, follow_symlinks=False)
            _protected(info, mode=0o600)
            _require(entry.name == _STORE_LOCK or (entry.name.endswith('.json')
                     and _matches(entry.name[:-5], _CONSENT_ID)), 'owner_consent_store_unsafe')
            if entry.name != _STORE_LOCK:
                records += 1
                _require(records < MAX_STORE_RECORDS, 'owner_consent_store_full')
            else:
                _require(info.st_size == 0, 'owner_consent_store_unsafe')
            _require(0 <= info.st_size <= MAX_RECORD_BYTES, 'owner_consent_store_unsafe')
            occupied += info.st_size
            _require(occupied <= MAX_STORE_BYTES, 'owner_consent_store_full')
    files.budget.tick()
    return occupied


def _encoded(record, budget, *, cap=MAX_RECORD_BYTES):
    budget.available('output_bytes', budget.measure(record, cap=cap - 1) + 1)
    result = retained.encode_validation_report(record, _work_budget=budget)
    _require(len(result) <= cap, 'owner_consent_resource_exhausted')
    budget.charge('output_bytes', len(result))
    return result


def _clean_temp(parent, name, identity):
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    _require(stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == identity,
             'owner_consent_publication_failed')
    os.unlink(name, dir_fd=parent)


def _publish(files, parent, name, payload, *, mode=0o600, immutable=True):
    temporary = '.consent-' + secrets.token_hex(16) + '.tmp'
    fd = files.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    parent=parent, mode=mode)
    initial = os.fstat(fd)
    identity = (initial.st_dev, initial.st_ino)
    owned = True
    try:
        os.fchmod(fd, mode)
        _protected(os.fstat(fd), mode=mode if immutable else None)
        offset = 0
        while offset < len(payload):
            files.budget.tick()
            count = os.write(fd, memoryview(payload)[offset:])
            _require(count > 0, 'owner_consent_publication_failed')
            offset += count
        files.budget.tick()
        os.fsync(fd)
        files.verify()
        current = os.stat(temporary, dir_fd=parent, follow_symlinks=False)
        _require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                 and (current.st_dev, current.st_ino) == identity
                 and (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == identity,
                 'owner_consent_publication_failed')
        files.budget.tick()
        if immutable:
            os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            _clean_temp(parent, temporary, identity)
        else:
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        owned = False
        os.fsync(parent)
        files.verify()
        published = os.stat(name, dir_fd=parent, follow_symlinks=False)
        _require((published.st_dev, published.st_ino) == identity and published.st_nlink == 1
                 and stat.S_IMODE(published.st_mode) == mode, 'owner_consent_publication_failed')
        _protected(published, mode=mode if immutable else None)
    except OSError:
        raise OwnerCensusConsentError('owner_consent_publication_failed') from None
    finally:
        if owned:
            _clean_temp(parent, temporary, identity)


def _issue(census_path, annotations_path, *, census_sha256, census_size_bytes,
           annotations_sha256, annotations_size_bytes, principal, selected_paths,
           expires_at_epoch, installed_config_path, now, budget, files):
    _require(os.geteuid() == 0, 'owner_consent_issuer_required')
    config = _installed_config(files, installed_config_path)
    policy_raw, _ = files.read(config.lane_owner_policy_file, cap=MAX_POLICY_BYTES, protected=True, mode=0o600)
    census = retained.read_census_input(census_path, _work_budget=budget, _descriptors=files)
    annotations = retained.read_census_input(annotations_path, _work_budget=budget, _descriptors=files)
    consent_id = secrets.token_hex(16)
    record = _build_consent(census_bytes=census, annotation_bytes=annotations,
        census_sha256=census_sha256, census_size_bytes=census_size_bytes,
        annotations_sha256=annotations_sha256, annotations_size_bytes=annotations_size_bytes,
        policy_bytes=policy_raw, principal=principal, selected_paths=selected_paths,
        expires_at_epoch=expires_at_epoch, now=now, allowed_roots=_roots(config, budget),
        consent_id=consent_id, budget=budget)
    parent = _store(files, config, consent_id, lock=True)
    occupied = _capacity(files, parent)
    _require(MAX_STORE_RECORDS > 0, 'owner_consent_store_full')
    payload = _encoded(record, budget)
    _require(occupied + len(payload) <= MAX_STORE_BYTES, 'owner_consent_store_full')
    files.verify()
    _publish(files, parent, consent_id + '.json', payload)
    return dict(schema_version=CONSENT_SCHEMA, status='owner_consent_issued', consent_id=consent_id,
                expected_sha256=retained._digest(payload, _work_budget=budget), expected_size_bytes=len(payload),
                expires_at_epoch=expires_at_epoch, selected_count=record['selected_count'],
                consent_metadata_published=True, execution_authorized=False,
                target_generation_bound=False, requires_fresh_reference_check=True, mutations=0)


def _run_issue(census_path, annotations_path, *, census_sha256, census_size_bytes,
                        annotations_sha256, annotations_size_bytes, principal, selected_paths,
                        expires_at_epoch, installed_config_path, now, monotonic, _encoded_stdout=False):
    """Root-admin attests finite intent; immutable metadata grants no target action."""
    from .control_plane_reference_budget import ReferenceCollectionBudget, ReferenceCollectionBudgetError
    try:
        budget = ReferenceCollectionBudget(monotonic=monotonic)
    except ReferenceCollectionBudgetError as error:
        raise OwnerCensusConsentError("owner_consent_resource_exhausted") from error
    files = _Files(budget)
    try:
        budget.tick()
        result = _issue(census_path, annotations_path, census_sha256=census_sha256,
            census_size_bytes=census_size_bytes, annotations_sha256=annotations_sha256,
            annotations_size_bytes=annotations_size_bytes, principal=principal,
            selected_paths=selected_paths, expires_at_epoch=expires_at_epoch,
            installed_config_path=installed_config_path, now=now, budget=budget, files=files)
        budget.measure(result, cap=8192)
        return _encoded(result, budget, cap=8192) if _encoded_stdout else result
    except ReferenceCollectionBudgetError as error:
        raise OwnerCensusConsentError('owner_consent_resource_exhausted') from error
    except (OSError, TypeError, UnicodeError):
        raise OwnerCensusConsentError('owner_consent_io_failed') from None
    finally:
        try:
            files.finish()
        finally:
            budget.close()


def issue_owner_consent(census_path, annotations_path, *, census_sha256, census_size_bytes,
                        annotations_sha256, annotations_size_bytes, principal, selected_paths,
                        expires_at_epoch, installed_config_path, now, monotonic):
    """Root-admin attests finite intent; immutable metadata grants no target action."""
    return _run_issue(census_path, annotations_path, census_sha256=census_sha256,
        census_size_bytes=census_size_bytes, annotations_sha256=annotations_sha256,
        annotations_size_bytes=annotations_size_bytes, principal=principal,
        selected_paths=selected_paths, expires_at_epoch=expires_at_epoch,
        installed_config_path=installed_config_path, now=now, monotonic=monotonic)


_CONSENT_FIELDS = frozenset({'schema_version', 'consent_id', 'issuer_kind', 'issuer_uid', 'principal',
    'policy_sha256', 'policy_size_bytes', 'issued_at_epoch', 'expires_at_epoch', 'census', 'annotations',
    'inventory_count', 'selected_count', 'scope', 'decisions', 'execution_authorized',
    'target_generation_bound', 'requires_fresh_reference_check', 'mutations', 'consent_digest'})


def _record(raw, consent_id, policy_raw, roots, now, budget):
    code = 'owner_consent_record_invalid'
    try:
        value = retained._document(raw, MAX_RECORD_BYTES, _work_budget=budget)
    except retained.CensusDecisionError:
        raise OwnerCensusConsentError(code) from None
    _require(set(value) == _CONSENT_FIELDS and value['schema_version'] == CONSENT_SCHEMA
             and value['consent_id'] == consent_id
             and value['issuer_kind'] == 'local_root_administrative_attestation'
             and type(value['issuer_uid']) is int and value['issuer_uid'] == 0
             and value['scope'] == _SCOPE and _matches(value['principal'], _PRINCIPAL)
             and value['execution_authorized'] is False and value['target_generation_bound'] is False
             and value['requires_fresh_reference_check'] is True
             and type(value['mutations']) is int and value['mutations'] == 0, code)
    issued, expires = value['issued_at_epoch'], value['expires_at_epoch']
    _require(_number(issued) and _number(expires) and issued < expires and issued <= now, code)
    _require(now < expires, 'owner_consent_record_expired')
    for key in ('census', 'annotations'):
        budget.charge('entries')
        identity = value[key]
        _require(isinstance(identity, dict) and set(identity) == {'sha256', 'size_bytes'}
                 and _matches(identity['sha256'], _DIGEST) and _counter(identity['size_bytes'], positive=True)
                 and identity['size_bytes'] <= retained.MAX_JSON_BYTES, code)
    _require(_matches(value['policy_sha256'], _DIGEST)
             and _counter(value['policy_size_bytes'], positive=True)
             and value['policy_size_bytes'] <= MAX_POLICY_BYTES, code)
    _require(value['policy_size_bytes'] == len(policy_raw)
             and value['policy_sha256'] == retained._digest(policy_raw, _work_budget=budget),
             'owner_consent_policy_changed')
    policy = _policy(policy_raw, value['principal'], budget)
    rows = value['decisions']
    _require(isinstance(rows, list) and 0 < len(rows) <= MAX_SELECTED
             and _counter(value['selected_count'], positive=True) and value['selected_count'] == len(rows)
             and _counter(value['inventory_count'], positive=True)
             and len(rows) <= value['inventory_count'] <= retained.MAX_ROWS, code)
    seen = set()
    for pair in rows:
        budget.charge('facts')
        _require(isinstance(pair, dict) and set(pair) == {'decision', 'census_row'}, code)
        row, decision = pair['census_row'], pair['decision']
        _row(row, roots, budget, code)
        _decision(decision, row, roots, issued, budget, code)
        _require(row['path'] not in seen, code)
        seen.add(row['path'])
        _authorize(decision, policy, expires, issued)
    _require(_matches(value['consent_digest'], _DIGEST), code)
    budget.available('output_bytes', budget.measure(value, cap=MAX_RECORD_BYTES))
    budget.tick()
    _require(canonical_digest(value, digest_field='consent_digest') == value['consent_digest'], code)
    budget.tick()
    return value


def _reread(files, acquired, original, cap):
    files.verify_record(acquired)
    os.lseek(acquired.fd, 0, os.SEEK_SET)
    raw = files.read_bytes(acquired.fd, cap)
    files.verify_record(acquired)
    _require(raw == original, 'owner_consent_record_changed')


def _report(consent_id, *, expected_sha256, expected_size_bytes, installed_config_path, now, budget, files, _publication=None):
    _require(os.geteuid() == 0, 'owner_consent_issuer_required')
    _require(_matches(consent_id, _CONSENT_ID) and _matches(expected_sha256, _DIGEST)
             and _counter(expected_size_bytes, positive=True) and expected_size_bytes <= MAX_RECORD_BYTES
             and _number(now), 'owner_consent_options_invalid')
    config = _installed_config(files, installed_config_path)
    policy_raw, policy_input = files.read(config.lane_owner_policy_file, cap=MAX_POLICY_BYTES,
                                          protected=True, mode=0o600)
    _store(files, config, consent_id)
    raw, record_input = files.read(Path(config.owner_consent_store) / (consent_id + '.json'),
                                    cap=MAX_RECORD_BYTES, protected=True, mode=0o600)
    _require(len(raw) == expected_size_bytes and retained._digest(raw, _work_budget=budget) == expected_sha256,
             'owner_consent_record_changed')
    record = _record(raw, consent_id, policy_raw, _roots(config, budget), now, budget)
    rows, measured = [], 1024
    for pair in record['decisions']:
        budget.charge('facts')
        requirements = ['target_generation_unbound', 'fresh_reference_inventory_required',
                        'consumer_participation_unproven', 'retirement_admission_unproven']
        action = pair['decision']['action']
        if action in ('offload', 'delete'):
            requirements.append('offload_restore_receipts_required')
        if action == 'register':
            requirements.append('registration_not_applied')
        projected = pair | {'unmet_requirements': requirements}
        measured += budget.measure(projected, cap=MAX_RECORD_BYTES - measured) + 2
        _require(measured <= MAX_RECORD_BYTES, 'owner_consent_resource_exhausted')
        budget.retain(projected)
        rows.append(projected)
    files.verify()
    _reread(files, policy_input, policy_raw, MAX_POLICY_BYTES)
    _reread(files, record_input, raw, MAX_RECORD_BYTES)
    files.verify()
    result = dict(schema_version=REPORT_SCHEMA, status='owner_consent_observed', consent_id=consent_id,
        consent_sha256=expected_sha256, consent_size_bytes=expected_size_bytes, principal=record['principal'],
        principal_source='protected_root_consent', requestor_context_verified=False, expires_at_epoch=record['expires_at_epoch'],
        inventory_count=record['inventory_count'], selected_count=record['selected_count'], decisions=rows,
        scope=_SCOPE, execution_authorized=False, target_generation_bound=False,
        requires_fresh_reference_check=True, general_reference_inventory_complete=False,
        consumer_fence_checked=False, references_clear=False, retirement_admission_checked=False, candidate_bytes=None,
        estimated_reclaimable_bytes=None, eta_contribution_bytes=None, eta_seconds=None, mutations=0, blockers=[])
    budget.measure(result, cap=MAX_RECORD_BYTES)
    if _publication is not None:
        return _publish_reports(files, config, result, _publication)
    return result


def _run_report(consent_id, *, expected_sha256, expected_size_bytes, installed_config_path, now, monotonic, publication=None, _encoded_stdout=False):
    """One budget from installed acquisition through optional public publication."""
    from .control_plane_reference_budget import ReferenceCollectionBudget, ReferenceCollectionBudgetError
    try:
        budget = ReferenceCollectionBudget(monotonic=monotonic, values_limit=10_000)
    except ReferenceCollectionBudgetError as error:
        raise OwnerCensusConsentError("owner_consent_resource_exhausted") from error
    files = _Files(budget, raw_cap=MAX_CONSUMPTION_RAW)
    try:
        budget.tick()
        result = _report(consent_id, expected_sha256=expected_sha256, expected_size_bytes=expected_size_bytes,
                       installed_config_path=installed_config_path, now=now, budget=budget, files=files, _publication=publication)
        if _encoded_stdout:
            return _encoded(result, budget, cap=8192 if publication else MAX_RECORD_BYTES)
        return result
    except ReferenceCollectionBudgetError as error:
        raise OwnerCensusConsentError('owner_consent_resource_exhausted') from error
    except (OSError, TypeError, UnicodeError):
        raise OwnerCensusConsentError('owner_consent_io_failed') from None
    finally:
        try:
            files.finish()
        finally:
            budget.close()



def report_owner_consent(consent_id, *, expected_sha256, expected_size_bytes, installed_config_path, now, monotonic):
    """Observe retained root consent against current policy, without target IO."""
    return _run_report(consent_id, expected_sha256=expected_sha256, expected_size_bytes=expected_size_bytes,
                       installed_config_path=installed_config_path, now=now, monotonic=monotonic)


def _public_parent(files, config, path):
    first_anchor, first_edge = len(files.anchors), len(files.edges)
    parent, name = files.parent(path, protected=True)
    # A protected 0700 ancestor is valid for private records, but public report
    # transport needs the independently installed 0755 path at every level.
    for _, security, _ in files.anchors[first_anchor:]:
        files.budget.tick()
        _require(stat.S_IMODE(security[2]) == 0o755, 'owner_consent_publication_failed')
    for _, _, _, security, _ in files.edges[first_edge:]:
        files.budget.tick()
        _require(stat.S_IMODE(security[2]) == 0o755, 'owner_consent_publication_failed')
    _protected(os.fstat(parent), directory=True)
    _require(stat.S_IMODE(os.fstat(parent).st_mode) == 0o755, 'owner_consent_publication_failed')
    try:
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        current = None
    if current is not None:
        _protected(current)
        _require(current.st_nlink == 1 and stat.S_IMODE(current.st_mode) == 0o644,
                 'owner_consent_publication_failed')
        for source in files.records:
            files.budget.charge('entries')
            _require((current.st_dev, current.st_ino) != (source.info.st_dev, source.info.st_ino),
                     'owner_consent_publication_failed')
    return parent, name


def _publish_reports(files, config, report, publication):
    directory, request_id = publication
    _require(isinstance(request_id, str) and len(request_id) <= 80
             and re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-owner-census-decision-[0-9a-f]{8}', request_id)
             and directory == str(Path(config.spool_root) / 'results'), 'owner_consent_options_invalid')
    report_path = Path(directory) / (request_id + '.owner-census.json')
    parent, name = _public_parent(files, config, report_path)
    payload = _encoded(report, files.budget)
    _publish(files, parent, name, payload, mode=0o644, immutable=False)
    summary = dict(schema='blueprint_operator_door_outcome.v1', status=report['status'], code=None, exit_code=0,
        consent_id=report['consent_id'], consent_sha256=report['consent_sha256'],
        consent_size_bytes=report['consent_size_bytes'], selected_count=report['selected_count'],
        inventory_count=report['inventory_count'], expires_at_epoch=report['expires_at_epoch'], blockers=[],
        result=str(report_path), result_sha256=retained._digest(payload, _work_budget=files.budget),
        result_size_bytes=len(payload), execution_authorized=False, target_generation_bound=False,
        general_reference_inventory_complete=False, consumer_fence_checked=False, references_clear=False, mutations=0,
        candidate_bytes=None, eta_contribution_bytes=None, eta_seconds=None)
    parent, name = _public_parent(files, config, Path(directory) / (request_id + '.outcome.json'))
    _publish(files, parent, name, _encoded(summary, files.budget, cap=8192), mode=0o644, immutable=False)
    return summary


def _refusal(error):
    blockers = [error.code]
    cause = error.__cause__
    from .control_plane_reference_budget import ReferenceCollectionBudgetError
    if isinstance(cause, ReferenceCollectionBudgetError):
        blockers.append(cause.code)
    return dict(schema_version=REPORT_SCHEMA, status='refused', blockers=blockers, complete=False,
                execution_authorized=False, target_generation_bound=False,
                general_reference_inventory_complete=False, consumer_fence_checked=False,
                requires_fresh_reference_check=True, references_clear=False, candidate_bytes=None,
                eta_contribution_bytes=None, eta_seconds=None, mutations=0)


def main(argv=None):
    """Fixed report CLI; issuance belongs to the existing census command."""
    import argparse
    import json
    import time
    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise OwnerCensusConsentError('owner_consent_options_invalid')
    parser = Parser(allow_abbrev=False)
    parser.add_argument('mode', choices=['report'])
    parser.add_argument('--consent-id', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-size-bytes', required=True, type=int)
    parser.add_argument('--door-config', default='/etc/blueprint-operator-door/door.json')
    parser.add_argument('--results-dir')
    parser.add_argument('--request-id')
    try:
        args = parser.parse_args(argv)
        _require((args.results_dir is None) == (args.request_id is None), 'owner_consent_options_invalid')
        publication = None if args.results_dir is None else (args.results_dir, args.request_id)
        report = _run_report(args.consent_id, expected_sha256=args.expected_sha256,
            expected_size_bytes=args.expected_size_bytes, installed_config_path=args.door_config,
            now=time.time(), monotonic=time.monotonic, publication=publication, _encoded_stdout=True)
    except OwnerCensusConsentError as error:
        print(json.dumps(_refusal(error), sort_keys=True, separators=(',', ':')))
        return 1
    print(report.decode("utf-8"), end="")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
