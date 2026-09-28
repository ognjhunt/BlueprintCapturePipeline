"""Root-only publication of the finite readable registered-producer bootstrap."""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

from . import control_plane_lane_experiment_actions as actions
from . import control_plane_lane_experiment_birth as birth
from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_authority import _current
from .control_plane_lane_experiment_publication import _BirthFiles, _publish
from .control_plane_lane_owner_target_versions import OwnerTargetVersionError, _epoch, _require
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest

SCHEMA = 'control_plane_lane_experiment_producer_bootstrap.v1'
SOURCE_MODULES = frozenset({'control_plane_lane_experiment_consumer', 'control_plane_lane_experiment_completion',
    'native_g1_development_pair', 'native_g1_development_worker', 'native_g1_registered_containment',
    'native_g1_runtime_assembly', 'native_g1_policy_server_supervisor', 'native_g1_shared_scene_episode',
    'control_plane_scratch_lifetime', 'control_plane_g1_lifetime_adapter'})


def _sources(files):
    try:
        from . import native_g1_registered_containment as contained
    except ImportError:
        raise OwnerTargetVersionError('experiment_producer_source_unavailable') from None
    expected = contained.producer_source_identities()
    _require(type(expected) is dict and set(expected) == SOURCE_MODULES, 'experiment_producer_source_invalid')
    root = Path(__file__).parent
    result = {}
    for name in sorted(SOURCE_MODULES):
        raw, record = files.read(root/(name+'.py'), cap=1024*1024, protected=True)
        _require(not stat.S_IMODE(record.info.st_mode) & 0o022 and record.info.st_uid == 0,
                 'experiment_producer_source_unsafe')
        digest = issuance._selector(raw, files.budget)['sha256']
        _require(expected[name] == digest, 'experiment_producer_source_changed')
        result[name] = digest
    files.verify()
    return result


def issue_bootstrap(intent_id, *, expected_intent_sha256, expected_intent_size_bytes, request_paths,
                    installed_config_path, now=time.time):
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        issued = now()
        _require(os.geteuid() == 0 and _epoch(issued), 'experiment_issuer_required')
        config, gid = actions._context(files, installed_config_path, issued)
        _require(config.experiment_creation_enabled is True and owners._matches(intent_id, owners._CONSENT_ID)
                 and isinstance(request_paths, (tuple, list)) and len(request_paths) == 2,
                 'experiment_producer_bootstrap_invalid')
        public, current, entry = actions._selected(files, config, intent_id, issued, gid)
        _require(entry['state'] == 'active' and entry['completion'] is None and entry['operation_id'] is None
                 and issued < entry['expires_at_epoch'], 'experiment_producer_bootstrap_invalid')
        target, _ = actions._target(files, config, entry)
        lease, _ = actions._lease(files, target, entry)
        origin = actions._birth(files, public, entry, gid)
        _require(lease['released_at_epoch'] is None and issued < lease['expires_at_epoch']
                 and origin['participant_profile'] == 'g1_local_contained_completed.v1',
                 'experiment_producer_bootstrap_invalid')
        raw, _ = files.read(Path(config.experiment_record_store)/(intent_id+'.json'), cap=32768, protected=True, mode=0o600)
        owners._identity(raw, expected_intent_sha256, expected_intent_size_bytes, files.budget)
        intent = retained._document(raw, 32768, _work_budget=files.budget)
        _require(type(intent) is dict and set(intent) == birth._INTENT_FIELDS
                 and intent['schema_version'] == issuance.CREATION_SCHEMA and type(intent['issuer_uid']) is int
                 and intent['issuer_uid'] == 0 and intent['intent_digest'] == canonical_digest(intent, digest_field='intent_digest')
                 and all(intent[key] == entry[key] for key in ('intent_id', 'generation', 'root', 'lane', 'name', 'owner'))
                 and intent['participant_profile'] == origin['participant_profile']
                 and intent['policy'] == current[1]['policy'] and issued < intent['expires_at_epoch']
                 and type(intent['request_records']) is list and len(intent['request_records']) == 2,
                 'experiment_producer_bootstrap_invalid')
        marker_raw, _ = files.read(target/birth._MARKER, cap=4096)
        _require(issuance._selector(marker_raw, files.budget) == origin['marker'], 'experiment_birth_changed')
        marker = retained._document(marker_raw, 4096, _work_budget=files.budget)
        intent_selector = issuance._selector(raw, files.budget)
        _require(marker['generation'] == entry['generation'] and marker['intent'] == intent_selector
                 and marker['marker_digest'] == canonical_digest(marker, digest_field='marker_digest'), 'experiment_birth_changed')
        policy_raw, _ = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES, protected=True, mode=0o600)
        _require(issuance._selector(policy_raw, files.budget) == current[1]['policy'], 'experiment_policy_changed')
        policy = owners._policy(policy_raw, intent['principal'], files.budget)
        owners._authorize(dict(action='register', owner=intent['owner'], ttl_seconds=intent['lease_ttl_seconds']),
                          policy, intent['expires_at_epoch'], issued)
        requests = []
        for path, selector in zip(request_paths, intent['request_records'], strict=True):
            _require(isinstance(path, Path) and path.is_absolute() and not path.is_relative_to(target),
                     'experiment_producer_request_changed')
            request_raw, original = files.read(path, cap=65536, protected=True)
            _require(original.info.st_uid == 0 and original.info.st_gid == gid and original.info.st_nlink == 1
                     and stat.S_IMODE(original.info.st_mode) == 0o640, 'experiment_producer_request_unsafe')
            owners._identity(request_raw, selector['sha256'], selector['size_bytes'], files.budget)
            request = retained._document(request_raw, 65536, _work_budget=files.budget)
            _require(request['request_digest'] == canonical_digest(request, digest_field='request_digest')
                     and owners._matches(request['candidate_id'], owners._OWNER), 'experiment_producer_request_changed')
            requests.append(dict(path=str(path), raw=selector, candidate_id=request['candidate_id'], request_digest=request['request_digest']))
        _require(requests[0]['path'] != requests[1]['path'] and requests[0]['candidate_id'] != requests[1]['candidate_id'],
                 'experiment_producer_request_changed')
        installed_sources = _sources(files)
        public = birth._authority_lock(files, config.experiment_authority_root, gid)
        refreshed = _current(files, public, gid)
        _require(refreshed[0] == current[0], 'experiment_producer_authority_changed')
        record = dict(schema_version=SCHEMA, intent_id=intent_id, generation=entry['generation'], birth=entry['birth'],
            target_identity=entry['target_identity'], lease=entry['lease'], participant_profile=origin['participant_profile'],
            intent=intent_selector, requests=requests, installed_sources=installed_sources, issued_at_epoch=issued,
            expires_at_epoch=min(entry['expires_at_epoch'], intent['expires_at_epoch'], current[1]['expires_at_epoch']))
        payload = actions._encoded(record, 'bootstrap_digest', 32768)
        files.verify()
        return _publish(files, public, intent_id+'.producer-bootstrap.json', payload, kind='bootstrap', blueprint_gid=gid)
    except (OSError, KeyError, TypeError, AttributeError):
        raise OwnerTargetVersionError('experiment_producer_bootstrap_failed') from None
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()
