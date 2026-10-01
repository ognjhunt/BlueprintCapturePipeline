"""Fixed ordinary disk diagnostic request; metadata supplies no cleanup grant.

The finite producer observes only the two installed lane-root directory FDs.
Its capacity rows are sequential observations, never reference clearance.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_publication import _BirthFiles
from .control_plane_lane_owner_target_versions import _require
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

REQUEST_SCHEMA = 'control_plane_lane_disk_diagnostic_request.v1'
SOURCE_MODULES = frozenset({
    'control_plane_lane_disk_diagnostic', 'control_plane_lane_experiment_retirement',
    'control_plane_lane_experiment_birth', 'control_plane_lane_experiment_authority',
    'control_plane_lane_experiment_actions', 'control_plane_lane_experiment_completion',
    'control_plane_lane_experiment_publication', 'control_plane_lane_owner_consents',
    'control_plane_lane_owner_target_io', 'control_plane_lane_owner_target_versions',
    'control_plane_lane_owner_target_publication', 'control_plane_lane_scratch_decisions',
    'control_plane_lane_scratch', 'control_plane_scratch_lifetime',
    'control_plane_reference_budget', 'decision_evidence_contracts',
})
_REQUEST_FIELDS = frozenset({'schema_version', 'run_ref', 'config', 'roots',
                             'root_identities', 'installed_sources', 'request_digest'})


def _sources(files):
    root = Path(__file__).parent
    selected = {}
    for name in sorted(SOURCE_MODULES):
        raw, record = files.read(root / (name + '.py'), cap=1024 * 1024, protected=True)
        selected[name] = issuance._selector(raw, files.budget)['sha256']
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
