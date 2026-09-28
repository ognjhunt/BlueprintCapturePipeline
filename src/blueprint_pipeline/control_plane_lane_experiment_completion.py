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
    _require(type(use) is RegisteredExperimentUse and use._closed
             and not use.files.owned and not use.files.unresolved and len(use._producer_requests) == 2,
             'experiment_producer_closure_unproven')
    if use.birth['participant_profile'] != 'g1_local_prelaunch_block.v1':
        # No fabricated success/containment downgrade or guessed child closure.
        return None
    _require(os.geteuid() == 0, 'experiment_producer_closure_unproven')
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
    if value.get('schema_version') == 'control_plane_lane_experiment_producer_completion.v2':
        fields = {'schema_version', 'intent_id', 'generation', 'birth', 'target_identity', 'lease',
                  'participant_profile', 'pair', 'workers', 'supervised', 'kernel_unit', 'installed_sources',
                  'lifetime_closed', 'child_execution_started', 'completed_at_epoch', 'completion_digest'}
        _require(set(value) == fields
                 and value['completion_digest'] == canonical_digest(value, digest_field='completion_digest')
                 and all(value[key] == entry[key] for key in ('intent_id','generation','birth','target_identity','lease'))
                 and value['participant_profile'] == 'g1_local_contained_completed.v1'
                 and value['lifetime_closed'] is True and value['child_execution_started'] is True
                 and type(value['workers']) is list and len(value['workers']) == 2
                 and type(value['supervised']) is list and len(value['supervised']) == 2
                 and value['kernel_unit']['kernel']['tasks'] == 0
                 and value['kernel_unit']['finished']['ActiveState'] == 'inactive', 'experiment_completion_changed')
        return value
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


def publish_contained_completion(intent_id, *, expected_intent, proof, installed_config_path, now):
    """The root's actual native unit observer selects this historical closure."""
    from . import native_g1_registered_containment as contained
    from . import native_g1_development_pair as pair
    from . import native_g1_development_worker as worker
    _require(os.geteuid() == 0 and type(proof) is contained._TerminatedUnitProof,
             'experiment_producer_closure_unproven')
    files = _BirthFiles(ReferenceCollectionBudget(values_limit=10000))
    reader = None
    entered_reader = False
    try:
        issued = now()
        config, gid = actions._context(files, installed_config_path, issued)
        _require(config.experiment_creation_enabled is True, 'experiment_creation_disabled')
        public, current, entry = actions._selected(files, config, intent_id, issued, gid)
        _require(entry['state'] == 'active' and entry['completion'] is None and entry['operation_id'] is None
                 and issued < entry['expires_at_epoch'] and proof.observed_at <= issued,
                 'experiment_producer_authority_changed')
        # Upgrade the genuine exact reader's original SH descriptor to EX.
        # Native score/media readers then borrow that same live authority rather
        # than attempting a second incompatible lock under the finalizer.
        from .control_plane_lane_experiment_consumer import RegisteredExperimentUse
        import fcntl
        root = config.lane_scratch_work_root if entry['root'] == 'work' else config.lane_scratch_inputs_root
        selected_target = Path(root)/'g1'/entry['name']
        reader = RegisteredExperimentUse.admit(selected_target,expected_birth=entry['birth'],
            expected_generation=entry['generation'],now=now)
        reader.__enter__()
        entered_reader = True
        reader.check()
        reader.files.proof(reader.fd)
        fcntl.flock(reader.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target, _ = actions._target(files, config, entry, lock=False)
        lease, _ = actions._lease(files, target, entry)
        original = actions._birth(files, public, entry, gid)
        _require(original['participant_profile'] == 'g1_local_contained_completed.v1'
                 and lease['released_at_epoch'] is None and issued < lease['expires_at_epoch'],
                 'experiment_producer_authority_changed')
        raw, _ = files.read(Path(config.experiment_record_store)/(intent_id+'.json'), cap=32768, protected=True, mode=0o600)
        owners._identity(raw, expected_intent['sha256'], expected_intent['size_bytes'], files.budget)
        intent = retained._document(raw,32768,_work_budget=files.budget)
        _require(intent['generation'] == entry['generation'] and intent['policy'] == current[1]['policy']
                 and issued < intent['expires_at_epoch'], 'experiment_producer_authority_changed')
        sources = contained.producer_source_identities()
        from .control_plane_lane_experiment_authority import _read
        boot_raw, _ = _read(files,public,intent_id+'.producer-bootstrap.json',32768,gid)
        bootstrap = retained._document(boot_raw,32768,_work_budget=files.budget)
        _require(bootstrap['intent'] == expected_intent and bootstrap['generation'] == entry['generation']
                 and bootstrap['birth'] == entry['birth'] and bootstrap['installed_sources'] == sources
                 and bootstrap['issued_at_epoch'] <= proof.observed_at <= issued < bootstrap['expires_at_epoch'],
                 'experiment_producer_authority_changed')
        command = contained._unit_arguments(intent_id,target,Path(config.experiment_authority_root)/(intent_id+'.producer-bootstrap.json'))
        _require(proof.command == command, 'experiment_unit_command_changed')
        contained._check_unit(proof.started,command,intent_id)
        _require(proof.finished['InvocationID'] == proof.started['InvocationID']
                 and proof.finished['ActiveState'] == 'inactive' and proof.finished['Result'] == 'success'
                 and proof.finished['ExecMainStatus'] == '0' and proof.kernel['tasks'] == 0,
                 'experiment_unit_closure_unproven')
        pair_raw, _ = files.read(target/(pair.SCHEMA+'.json'),cap=65536)
        result = retained._document(pair_raw,65536,_work_budget=files.budget)
        _require(result['schema_version'] == pair.SCHEMA and result['result_digest'] == canonical_digest(result,digest_field='result_digest')
                 and result['status'] == 'completed_development_only' and result['mode'] == 'local'
                 and len(result['attempts']) == 2 and result['not_attempted_candidate_ids'] == [],
                 'experiment_producer_closure_unproven')
        workers, supervised = [], []
        for index, attempt in enumerate(result['attempts']):
            candidate = result['candidate_ids'][index]
            worker_path = target/candidate/worker.RESULT_FILENAME
            _require(attempt['candidate_id'] == candidate and attempt['status'] == attempt['worker_status'] == 'completed_development_only'
                     and attempt['worker_result_path'] == str(worker_path) and attempt['score'] is not None
                     and attempt['review_media'] is not None, 'experiment_producer_closure_unproven')
            worker_raw, _ = files.read(worker_path,cap=65536)
            receipt = retained._document(worker_raw,65536,_work_budget=files.budget)
            _require(receipt['schema_version'] == worker.RESULT_SCHEMA
                     and receipt['result_digest'] == attempt['worker_result_digest'] == canonical_digest(receipt,digest_field='result_digest')
                     and receipt['candidate_id'] == candidate and receipt['request_digest'] == bootstrap['requests'][index]['request_digest']
                     and receipt['status'] == 'completed_development_only'
                     and receipt['teardown'] == {'environment':'closed','simulator':'closed'}, 'experiment_producer_closure_unproven')
            episode_path = target/candidate/'episode/native_g1_supervised_built_scene_episode.v1.json'
            episode_raw, _ = files.read(episode_path,cap=65536)
            episode = retained._document(episode_raw,65536,_work_budget=files.budget)
            _require(episode == receipt['supervised_episode'] and episode['status'] == 'completed_development_only'
                     and episode['result_digest'] == canonical_digest(episode,digest_field='result_digest')
                     and episode['server_teardown']['status'] == 'child_exited', 'experiment_producer_closure_unproven')
            child = episode['server_teardown']['registered_child_lifetime']
            _require(child['schema_version'] == 'registered_g1_child_lifetime.v1'
                     and child['lifetime_digest'] == canonical_digest(child,digest_field='lifetime_digest')
                     and child['intent_id'] == intent_id and child['generation'] == entry['generation']
                     and child['direct_child_exited'] is True and child['log_closed'] is True
                     and child['descendant_clearance'] is False
                     and child['handshake']['pid'] == child['pid'] == episode['server_teardown']['pid']
                     == episode['server_lease_receipt']['pid'], 'experiment_producer_closure_unproven')
            # Re-run actual native score/media validation under target EX.
            episode_result = target/candidate/'episode/native_g1_built_scene_policy_episode.v1.json'
            _require(pair._score_from_episode(episode_result,worker=receipt,objective_id=result['objective_id']) == attempt['score'],
                     'experiment_producer_closure_unproven')
            with episode_result.open('rb') as stream:
                episode_receipt = retained._document(stream.read(65537),65536,_work_budget=files.budget)
            _require(pair._verified_review_media(episode_result,episode=episode_receipt,pair_root=target) == attempt['review_media'],
                     'experiment_producer_closure_unproven')
            workers.append(issuance._selector(worker_raw,files.budget))
            supervised.append(issuance._selector(episode_raw,files.budget))
            reader.check()
            _require(len(files.owned)+len(reader.files.owned)<=104,'experiment_producer_resource_exhausted')
        policy_raw, _ = files.read(config.lane_owner_policy_file,cap=owners.MAX_POLICY_BYTES,protected=True,mode=0o600)
        _require(issuance._selector(policy_raw,files.budget) == current[1]['policy'],'experiment_policy_changed')
        public = birth._authority_lock(files,config.experiment_authority_root,gid)
        refreshed = _current(files,public,gid)
        _require(refreshed[0] == current[0],'experiment_producer_authority_changed')
        store = issuance._store(files,config.experiment_record_store)
        value = dict(schema_version='control_plane_lane_experiment_producer_completion.v2',intent_id=intent_id,
            generation=entry['generation'],birth=entry['birth'],target_identity=entry['target_identity'],lease=entry['lease'],
            participant_profile='g1_local_contained_completed.v1',pair=issuance._selector(pair_raw,files.budget),
            workers=workers,supervised=supervised,installed_sources=sources,
            kernel_unit=dict(started=proof.started,finished=proof.finished,kernel=proof.kernel,observed_at=proof.observed_at),
            lifetime_closed=True,child_execution_started=True,completed_at_epoch=issued)
        payload=actions._encoded(value,'completion_digest',32768)
        _require(issuance._capacity(files,store,adding_registration=False)+len(payload)+65536
                 <= issuance.MAX_EXPERIMENT_STORE_BYTES,'experiment_store_full')
        files.verify()
        selected=_publish(files,store,intent_id+'.producer-completion.json',payload,kind='private')
        prepared, previous = actions._version(files,public,refreshed,entry|{'completion':selected},gid,current[1]['policy'],issued)
        _publish(files,store,intent_id+'.completion-head.json',prepared,kind='private')
        files.verify()
        actions._install_head(files,public,prepared,gid,previous)
        return selected
    finally:
        try:
            if reader is not None:
                if entered_reader:
                    reader.__exit__(None,None,None)
                else:
                    reader.close()
        finally:
            try:
                files.finish()
            finally:
                files.budget.close()
