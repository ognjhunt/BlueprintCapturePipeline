"""Fixed ordinary disk diagnostic request; metadata supplies no cleanup grant.

The finite producer observes only the two installed lane-root directory FDs.
Its capacity rows are sequential observations, never reference clearance.
"""
from __future__ import annotations

import os
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_owner_target_versions import _require
from .control_plane_lane_owner_target_versions import _epoch
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

REQUEST_SCHEMA = 'control_plane_lane_disk_diagnostic_request.v1'
SOURCE_MODULES = frozenset({
    'control_plane_lane_disk_diagnostic', 'control_plane_lane_experiment_retirement',
    'control_plane_lane_experiment_birth', 'control_plane_lane_experiment_authority',
    'control_plane_lane_experiment_actions', 'control_plane_lane_experiment_completion',
    'control_plane_lane_experiment_consumer',
    'control_plane_lane_experiment_publication', 'control_plane_lane_owner_consents',
    'control_plane_lane_owner_target_io', 'control_plane_lane_owner_target_versions',
    'control_plane_lane_owner_target_publication', 'control_plane_lane_scratch_decisions',
    'control_plane_lane_scratch', 'control_plane_scratch_lifetime',
    'control_plane_reference_budget', 'decision_evidence_contracts',
})
_REQUEST_FIELDS = frozenset({'schema_version', 'run_ref', 'config', 'roots',
                             'root_identities', 'installed_sources', 'request_digest'})
PROFILES = frozenset({'root_disk_diagnostic_disposable.v1', 'root_disk_diagnostic_evidence.v1'})
REPORT_NAME = 'disk-capacity-report.v1.json'
REPORT_SCHEMA = 'control_plane_lane_disk_diagnostic_report.v1'
INVOCATION_SCHEMA = 'control_plane_lane_disk_diagnostic_invocation.v1'
RESERVED_STORE_BYTES = 65536
_INVOCATION_FIELDS = frozenset({'schema_version', 'intent_id', 'generation', 'intent', 'birth',
    'target_identity', 'lease', 'authority', 'request', 'reserved_bytes', 'started_at_epoch', 'invocation_digest'})
_STAT_FIELDS = ('f_bsize', 'f_frsize', 'f_blocks', 'f_bfree', 'f_bavail', 'f_files',
                'f_ffree', 'f_favail', 'f_flag', 'f_namemax')
_CLOSED_EXECUTIONS = weakref.WeakKeyDictionary()


class _DiagnosticFiles(_BirthFiles):
    """The original authenticated expiry stays held through native publication."""
    def __init__(self, budget, now):
        super().__init__(budget)
        self.clock, self.expiry, self.last_epoch = now, None, None

    def bind_lifetime(self, issued, expiry):
        _require(_epoch(issued) and _epoch(expiry) and issued < expiry,
                 'diagnostic_producer_inactive')
        if self.expiry is None:
            self.last_epoch, self.expiry = issued, expiry
        else:
            _require(expiry == self.expiry and self.last_epoch <= issued, 'diagnostic_producer_inactive')
        self._live()

    def _live(self):
        self.budget.tick()
        if self.expiry is not None:
            observed = self.clock()
            _require(_epoch(observed) and self.last_epoch <= observed < self.expiry,
                     'diagnostic_producer_inactive')
            self.last_epoch = observed

    def publication_checkpoint(self, *, cleanup=False):
        if not cleanup:
            self._live()

    def creation_checkpoint(self):
        self._live()


def _sources(files):
    root = Path(__file__).parent
    selected = {}
    for name in sorted(SOURCE_MODULES):
        raw, record = files.read(root / (name + '.py'), cap=1024 * 1024, protected=True)
        selected[name] = issuance._selector(raw, files.budget)['sha256']
        files.verify_record(record)
    for name in ('__init__', 'config'):
        raw, record = files.read(owners.INSTALLED_PACKAGE_ROOT / 'operator_door' / (name + '.py'),
                                 cap=owners.MAX_POLICY_BYTES, protected=True)
        selected['operator_door.' + name] = issuance._selector(raw, files.budget)['sha256']
        files.verify_record(record)
    files.verify()
    return selected


def _root_selection(files, config):
    paths = {'work': config.lane_scratch_work_root, 'inputs': config.lane_scratch_inputs_root}
    identities = {}
    for key, path in paths.items():
        fd, _ = files.parent(Path(path) / '.lane-scratch.lock', protected=True)
        info = os.fstat(fd)
        files.location(fd)
        files.proof(fd)
        identities[key] = dict(dev=info.st_dev, ino=info.st_ino, type='directory')
    return paths, identities


def _configuration_selector(files, path):
    raw, record = files.read(path, cap=owners.MAX_POLICY_BYTES, protected=True)
    result = issuance._selector(raw, files.budget)
    files.verify_record(record)
    return result


def build_request(*, installed_config_path, run_ref):
    """Read protected current selectors; create no payload, intent or authority."""
    _require(os.geteuid() == 0 and owners._matches(run_ref, owners._OWNER),
             'diagnostic_request_invalid')
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        config = issuance._configuration(files, installed_config_path)
        _require(config.experiment_creation_enabled is True, 'experiment_creation_disabled')
        roots, identities = _root_selection(files, config)
        value = dict(schema_version=REQUEST_SCHEMA, run_ref=run_ref,
            config=_configuration_selector(files, installed_config_path), roots=roots,
            root_identities=identities, installed_sources=_sources(files))
        value['request_digest'] = canonical_digest(value, digest_field='request_digest')
        files.budget.measure(value, cap=32768)
        files.verify()
        return value
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


def validate_request(files, raw, *, config, installed_config_path, run_ref):
    """Identity plus fixed semantics; caller bytes select no path or callable."""
    value = retained._document(raw, 32768, _work_budget=files.budget)
    _require(type(value) is dict and set(value) == _REQUEST_FIELDS
             and value['schema_version'] == REQUEST_SCHEMA
             and value['run_ref'] == run_ref and owners._matches(run_ref, owners._OWNER)
             and value['request_digest'] == canonical_digest(value, digest_field='request_digest'),
             'diagnostic_request_invalid')
    roots, identities = _root_selection(files, config)
    _require(type(value['root_identities']) is dict
             and set(value['root_identities']) == {'work', 'inputs'}
             and all(type(row) is dict and set(row) == {'dev', 'ino', 'type'}
                     and all(type(row[key]) is int and row[key] >= 0 for key in ('dev', 'ino'))
                     and row['type'] == 'directory' for row in value['root_identities'].values())
             and value['roots'] == roots and value['root_identities'] == identities
             and value['config'] == _configuration_selector(files, installed_config_path)
             and value['installed_sources'] == _sources(files), 'diagnostic_request_changed')
    files.verify()
    return value


def _reserved_bytes(value):
    _require(type(value) is dict and set(value) == _INVOCATION_FIELDS
             and value['schema_version'] == INVOCATION_SCHEMA
             and owners._matches(value['intent_id'], owners._CONSENT_ID)
             and owners._matches(value['generation'], owners._CONSENT_ID)
             and type(value['reserved_bytes']) is int and value['reserved_bytes'] == RESERVED_STORE_BYTES
             and _epoch(value['started_at_epoch'])
             and value['invocation_digest'] == canonical_digest(value, digest_field='invocation_digest'),
             'diagnostic_invocation_invalid')
    return value['reserved_bytes']


def _admission(files, intent_id, expected_intent, config_path, request_path, issued, *, closed_context=None):
    from . import control_plane_lane_experiment_actions as actions
    from . import control_plane_lane_experiment_birth as birth
    from .control_plane_lane_experiment_authority import _current
    if closed_context is None:
        config, gid = actions._context(files, config_path, issued)
    else:
        # Reuse the actual invocation's parsed immutable config only after its
        # complete protected input/source bytes are freshly acquired unchanged.
        # This avoids executing the same loader twice under the original budget.
        _require(os.geteuid() == 0 and _epoch(issued), 'experiment_issuer_required')
        _require(_configuration_selector(files, config_path) == closed_context.request['config']
                 and _sources(files) == closed_context.request['installed_sources'],
                 'diagnostic_request_changed')
        config, gid = closed_context.config, closed_context.gid
    _require(config.experiment_creation_enabled is True, 'experiment_creation_disabled')
    intent, _ = birth._read_intent(files, config, intent_id, expected_intent, issued)
    _require(intent['participant_profile'] in PROFILES and intent['lane'] == 'diagnostics',
             'diagnostic_producer_inactive')
    public = birth._authority_lock(files, config.experiment_authority_root, gid)
    current = _current(files, public, gid)
    _require(current is not None and current[1]['state'] == 'enabled'
             and current[1]['issued_at_epoch'] <= issued < current[1]['expires_at_epoch']
             and current[1]['policy'] == intent['policy'], 'diagnostic_producer_inactive')
    rows = [row for row in current[1]['enrollments'] if row['intent_id'] == intent_id]
    _require(len(rows) == 1, 'diagnostic_producer_inactive')
    entry = rows[0]
    _require(entry['state'] == 'active' and entry['completion'] is None and entry['restoration'] is None
             and entry['operation_id'] is None and issued < entry['expires_at_epoch']
             and all(entry[key] == intent[key] for key in ('intent_id', 'generation', 'owner', 'root', 'lane', 'name')),
             'diagnostic_producer_inactive')
    files.bind_lifetime(issued, min(intent['expires_at_epoch'], entry['expires_at_epoch'],
                                   current[1]['expires_at_epoch']))
    original = actions._birth(files, public, entry, gid)
    _require(original['participant_profile'] == intent['participant_profile']
             and original['writer_scope'] == 'fixed_root_disk_diagnostic.v1'
             and original['lease'] == entry['lease'], 'diagnostic_producer_inactive')
    raw, _ = files.read(request_path, cap=32768, protected=True, mode=0o600)
    _require(intent['request_records'] == [issuance._selector(raw, files.budget)], 'diagnostic_request_changed')
    # Acquire the authenticated deepest target edge first. Its retained ancestry
    # includes this lane root, avoiding a second root walk in the same budget.
    target, target_fd = actions._target(files, config, entry)
    owners._protected(os.fstat(target_fd), directory=True, mode=0o700)
    request = validate_request(files, raw, config=config, installed_config_path=config_path,
                               run_ref=intent['reference_value'])
    lease, _ = actions._lease(files, target, entry)
    _require(lease['released_at_epoch'] is None and issued < lease['expires_at_epoch'] == entry['expires_at_epoch']
             and lease['run_ref'] == intent['reference_value'], 'diagnostic_producer_inactive')
    marker_raw, _ = files.read(target / birth._MARKER, cap=4096, protected=True, mode=0o600)
    _require(issuance._selector(marker_raw, files.budget) == original['marker'], 'diagnostic_marker_changed')
    return config, gid, public, current, entry, intent, request, target, target_fd


def _tree(files, target, target_fd, *, report=False):
    from . import control_plane_lane_scratch as scratch
    from . import control_plane_lane_experiment_birth as birth
    names = {scratch.LEASE_FILE, birth._MARKER} | ({REPORT_NAME} if report else set())
    files.location(target_fd)
    owners._protected(os.fstat(target_fd), directory=True, mode=0o700)
    seen = set()
    files.slot()
    with os.scandir(target_fd) as members:
        for member in members:
            files.budget.charge('entries')
            _require(member.name in names and member.name not in seen, 'diagnostic_tree_changed')
            seen.add(member.name)
            files.proof(target_fd)
            owners._protected(os.stat(member.name, dir_fd=target_fd, follow_symlinks=False), mode=0o600)
    _require(seen == names, 'diagnostic_tree_changed')
    files.location(target_fd)


@dataclass(frozen=True, eq=False)
class _ClosedDiagnostic:
    files: _BirthFiles
    config: object
    gid: int
    request: dict
    entry: dict
    authority: dict
    invocation: dict
    report_selector: dict
    report: dict


def _consume_closed(proof):
    _require(type(proof) is _ClosedDiagnostic and _CLOSED_EXECUTIONS.pop(proof, None) is proof.files
             and not proof.files.owned and not proof.files.unresolved
             and not proof.files.budget.closed and proof.files.budget.failure is None,
             'diagnostic_producer_closure_unproven')
    return proof.files.budget


def _execute(files, intent_id, expected, config_path, request_path, now):
    from . import control_plane_lane_experiment_actions as actions
    config, gid, _, current, entry, intent, request, target, target_fd = _admission(
        files, intent_id, expected, config_path, request_path, now())
    _tree(files, target, target_fd)
    store = issuance._store(files, config.experiment_record_store)
    occupied = issuance._capacity(files, store, adding_registration=False)
    _require(occupied + RESERVED_STORE_BYTES + 32768 <= issuance.MAX_EXPERIMENT_STORE_BYTES, 'experiment_store_full')
    started = now()
    _require(_epoch(started) and intent['issued_at_epoch'] <= started < intent['expires_at_epoch'],
             'diagnostic_producer_inactive')
    invocation = dict(schema_version=INVOCATION_SCHEMA, intent_id=intent_id, generation=entry['generation'],
        intent=expected, birth=entry['birth'], target_identity=entry['target_identity'], lease=entry['lease'],
        authority=current[0]['record'], request=intent['request_records'][0],
        reserved_bytes=RESERVED_STORE_BYTES, started_at_epoch=started)
    raw = actions._encoded(invocation, 'invocation_digest', 32768)
    files.verify()
    invocation_selector = actions._publish(files, store, intent_id + '.producer-invocation.json', raw, kind='private')
    observations = []
    last = started
    for key in ('work', 'inputs'):
        path = request['roots'][key]
        fd, _ = files.parent(Path(path) / '.lane-scratch.lock', protected=True)
        before = now()
        _require(_epoch(before) and last <= before < intent['expires_at_epoch'], 'diagnostic_producer_inactive')
        files.budget.tick()
        files.location(fd)
        values = os.fstatvfs(fd)
        files.budget.tick()
        files.location(fd)
        after = now()
        _require(_epoch(after) and before <= after < intent['expires_at_epoch'], 'diagnostic_producer_inactive')
        last = after
        row = {name: getattr(values, name) for name in _STAT_FIELDS}
        _require(all(type(value) is int and value >= 0 for value in row.values())
                 and row['f_frsize'] > 0 and row['f_bsize'] > 0
                 and row['f_bavail'] <= row['f_bfree'] <= row['f_blocks'], 'diagnostic_observation_invalid')
        observations.append(dict(root=key, root_identity=request['root_identities'][key],
                                 started_at_epoch=before, finished_at_epoch=after, statvfs=row))
    value = dict(schema_version=REPORT_SCHEMA, intent_id=intent_id, generation=entry['generation'],
        intent=expected, birth=entry['birth'], lease=entry['lease'], target_identity=entry['target_identity'],
        authority=current[0]['record'], request=intent['request_records'][0],
        invocation=invocation_selector, observations=observations)
    raw = actions._encoded(value, 'report_digest', 8192)
    files.verify()
    issued = now()
    _require(_epoch(issued) and last <= issued < intent['expires_at_epoch'], 'diagnostic_producer_inactive')
    selected = actions._publish(files, target_fd, REPORT_NAME, raw, kind='diagnostic_report')
    _tree(files, target, target_fd, report=True)
    files.verify()
    result = retained._document(raw, 8192, _work_budget=files.budget)
    return _ClosedDiagnostic(files, config, gid, request, entry, current[0], invocation_selector, selected, result)


def run_registered_disk_diagnostic(intent_id, *, expected_intent, request_path,
        installed_config_path='/etc/blueprint-operator-door/door.json', now=time.time):
    """Run exactly two FD observations, close them, then authenticate closure."""
    budget = ReferenceCollectionBudget(values_limit=10000)
    files = _DiagnosticFiles(budget, now)
    try:
        try:
            proof = _execute(files, intent_id, expected_intent, installed_config_path, request_path, now)
        finally:
            files.finish()
        budget.tick()
        _CLOSED_EXECUTIONS[proof] = files
        from .control_plane_lane_experiment_completion import complete_disk_diagnostic
        selected = complete_disk_diagnostic(proof, expected_intent=expected_intent,
            request_path=request_path, installed_config_path=installed_config_path, now=now)
        return dict(report=proof.report, completion=selected)
    finally:
        budget.close()
