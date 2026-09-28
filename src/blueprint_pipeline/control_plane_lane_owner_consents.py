"""Protected administrative owner intent; no target-generation or action authority."""
from __future__ import annotations

import math
import re

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


def _policy(raw, principal, budget):
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
    seen, selected = set(), None
    for row in rows:
        budget.charge('groups')
        _require(isinstance(row, dict) and set(row) == {'principal', 'owners', 'allowed_actions', 'max_consent_seconds'},
                 'owner_consent_policy_invalid')
        name, owners, actions, duration = (row[k] for k in ('principal', 'owners', 'allowed_actions', 'max_consent_seconds'))
        _require(_matches(name, _PRINCIPAL) and name not in seen, 'owner_consent_policy_invalid')
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
        seen.add(name)
        if name == principal:
            selected = row
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
             and all(isinstance(path, str) for path in selected_paths)
             and len(set(selected_paths)) == len(selected_paths), 'owner_consent_selection_invalid')
    budget.charge('entries', len(selected_paths))
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
