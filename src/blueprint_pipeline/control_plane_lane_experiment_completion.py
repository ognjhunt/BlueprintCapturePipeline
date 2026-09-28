"""Authentic native prelaunch closure, after every owned producer lifetime closes.

This finite profile started no simulator or policy child. Successful execution
requires the separately proved containment profile; status text cannot enroll it.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import control_plane_lane_experiment_actions as actions
from . import control_plane_lane_experiment_birth as birth
from . import control_plane_lane_experiment_retirement as issuance
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_experiment_authority import _current
from .control_plane_lane_experiment_publication import _BirthFiles, _publish
from .control_plane_lane_owner_target_versions import _require
from .control_plane_reference_budget import ReferenceCollectionBudget
from .decision_evidence_contracts import canonical_digest


def complete_native_pair(use, result):
    from . import native_g1_development_pair as pair
    from . import native_g1_development_worker as worker
    from .control_plane_lane_experiment_consumer import RegisteredExperimentUse
    _require(type(use) is RegisteredExperimentUse and os.geteuid() == 0 and use._closed
             and not use.files.owned and not use.files.unresolved and len(use._producer_requests) == 2,
             'experiment_producer_closure_unproven')
    if use.birth['participant_profile'] != 'g1_local_prelaunch_block.v1':
        # No fabricated success/containment downgrade or guessed child closure.
        return None
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    try:
        now = use.now()
        config, gid = actions._context(files, use._producer_config_path, now)
        _require(config.experiment_creation_enabled is True
                 and Path(config.experiment_authority_root) == use._public_root,
                 'experiment_producer_authority_changed')
        public, current, entry = actions._selected(files, config, use.entry['intent_id'], now, gid)
        _require(entry == use.entry and entry['state'] == 'active' and entry['completion'] is None,
                 'experiment_producer_authority_changed')
        target, _ = actions._target(files, config, entry)
        lease, _ = actions._lease(files, target, entry)
        _require(now < lease['expires_at_epoch'], 'experiment_producer_authority_changed')
        original = actions._birth(files, public, entry, gid)
        _require(original['participant_profile'] == 'g1_local_prelaunch_block.v1',
                 'experiment_producer_closure_unproven')
        raw, _ = files.read(target / (pair.SCHEMA + '.json'), cap=65536)
        observed = retained._document(raw, 65536, _work_budget=files.budget)
        _require(observed == result and observed['schema_version'] == pair.SCHEMA
                 and observed['result_digest'] == canonical_digest(observed, digest_field='result_digest')
                 and observed['mode'] == 'local' and observed['status'] == 'blocked'
                 and len(observed['attempts']) == 1 and len(observed['not_attempted_candidate_ids']) == 1,
                 'experiment_producer_closure_unproven')
        attempt = observed['attempts'][0]
        _require(attempt['candidate_id'] == observed['candidate_ids'][0]
                 and attempt['worker_status'] == 'blocked' and attempt['status'] == 'blocked'
                 and attempt['score'] is None and attempt['review_media'] is None,
                 'experiment_producer_closure_unproven')
        worker_path = target / attempt['candidate_id'] / worker.RESULT_FILENAME
        _require(attempt['worker_result_path'] == str(worker_path), 'experiment_producer_closure_unproven')
        worker_raw, _ = files.read(worker_path, cap=65536)
        receipt = retained._document(worker_raw, 65536, _work_budget=files.budget)
        _require(receipt['schema_version'] == worker.RESULT_SCHEMA
                 and receipt['result_digest'] == canonical_digest(receipt, digest_field='result_digest')
                 and receipt['result_digest'] == attempt['worker_result_digest']
                 and receipt['candidate_id'] == attempt['candidate_id'] and receipt['status'] == 'blocked'
                 and receipt['phase_reached'] in {'preflight', 'packet_verification', 'rights_review',
                     'navigation_goal_validation', 'navigation_goal_authority_validation'}
                 and receipt['teardown'] == {'environment': 'not_started', 'simulator': 'not_started'}
                 and all(receipt[key] is None for key in ('isaaclab_launch', 'device_binding',
                     'supervised_episode', 'native_dependency_matrix'))
                 and receipt['request_digest'] == use._producer_requests[0][3],
                 'experiment_producer_closure_unproven')
        policy_raw, _ = files.read(config.lane_owner_policy_file, cap=owners.MAX_POLICY_BYTES,
                                  protected=True, mode=0o600)
        _require(issuance._selector(policy_raw, files.budget) == current[1]['policy'],
                 'experiment_policy_changed')
        public = birth._authority_lock(files, config.experiment_authority_root, gid)
        refreshed = _current(files, public, gid)
        _require(refreshed[0] == current[0], 'experiment_producer_authority_changed')
        store = issuance._store(files, config.experiment_record_store)
        value = dict(schema_version='control_plane_lane_experiment_producer_completion.v1',
            intent_id=entry['intent_id'], generation=entry['generation'], birth=entry['birth'],
            target_identity=entry['target_identity'], lease=entry['lease'],
            participant_profile='g1_local_prelaunch_block.v1', pair=issuance._selector(raw, files.budget),
            worker=issuance._selector(worker_raw, files.budget), lifetime_closed=True,
            child_execution_started=False, completed_at_epoch=now)
        payload = actions._encoded(value, 'completion_digest', 32768)
        occupied = issuance._capacity(files, store, adding_registration=False)
        _require(occupied + len(payload) + 65536 <= issuance.MAX_EXPERIMENT_STORE_BYTES, 'experiment_store_full')
        files.verify()
        selected = _publish(files, store, entry['intent_id'] + '.producer-completion.json', payload, kind='private')
        prepared, previous = actions._version(files, public, refreshed, entry | {'completion': selected},
                                              gid, current[1]['policy'], now)
        _publish(files, store, entry['intent_id'] + '.completion-head.json', prepared, kind='private')
        files.verify()
        actions._install_head(files, public, prepared, gid, previous)
        return selected
    finally:
        try:
            files.finish()
        finally:
            files.budget.close()


def selected_completion(files, config, entry):
    """Historical closure is checked against the current authenticated entry."""
    _require(entry['completion'] is not None, 'experiment_completion_required')
    raw, _ = files.read(Path(config.experiment_record_store) / (entry['intent_id'] + '.producer-completion.json'),
                        cap=32768, protected=True, mode=0o600)
    _require(issuance._selector(raw, files.budget) == entry['completion'], 'experiment_completion_changed')
    value = retained._document(raw, 32768, _work_budget=files.budget)
    _require(set(value) == {'schema_version', 'intent_id', 'generation', 'birth', 'target_identity', 'lease',
                 'participant_profile', 'pair', 'worker', 'lifetime_closed', 'child_execution_started',
                 'completed_at_epoch', 'completion_digest'}
             and value['schema_version'] == 'control_plane_lane_experiment_producer_completion.v1'
             and value['completion_digest'] == canonical_digest(value, digest_field='completion_digest')
             and all(value[key] == entry[key] for key in ('intent_id', 'generation', 'birth', 'target_identity', 'lease'))
             and value['participant_profile'] == 'g1_local_prelaunch_block.v1'
             and value['lifetime_closed'] is True and value['child_execution_started'] is False,
             'experiment_completion_changed')
    return value
