"""Consent-bound scene actions; real exclusive admission lasts through completion.

A metadata KEEP plan supplies target evidence only. Protected installed policy,
current authenticated generations, actual supported lifetime closure and fresh
reference observations must independently admit an action. Missing evidence
never becomes permission. No production caller or enabled cleanup switch here.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import secrets
import stat
import time
import sys
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import load_authority, selected_document, load_document
from .task_evaluation_scene_retirement_generations import _write, _sealed
from .task_evaluation_scene_retirement_journal import SceneJournal
from .task_evaluation_scene_retirement_mutation import detach_and_remove, inventory_digest, removal_records
from .task_evaluation_scene_retirement_preservation import ActionAllowance, preserve_members
from .task_evaluation_scene_retirement_restore import restore_preserved_members, restore_records, _consume
from .task_evaluation_scene_retirement_metadata import retain_metadata_closure
from .task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt, publish_terminal_receipt, publish_progress_receipt
from .task_evaluation_scene_lifecycle_plan import build_scene_lifecycle_plan
from . import task_evaluation_scene_retirement_recovery as recovery
from .task_evaluation_scene_retirement_declared_bytes import verify_declared_bytes as _verify_declared_bytes, verify_publication_rows
from .task_evaluation_scene_retirement_reference_transfer import validate_current_reference_transfer
from .task_evaluation_scene_lineage_budget import _Rows


_LIFETIME='scene_retirement_lifetime.v1'
_TERMINAL={'completed','revoked_grace_elapsed','expired_grace_elapsed'}

# Finite actual participating consumers. Source hashes for an arbitrary subset
# do not establish closure; unknown future callsites require a reviewed update.
_COHORT_CALLS={
    'task_evaluation_scene_intake':'reserve_scene_attempt stage_scene_intent',
    'task_evaluation_scene_progression':'process_scene_intents',
    'task_evaluation_launch_preparation_worker':'materialize_preparation_references materialize_recipe_configuration_references materialize_recipe_supplemental_destination_references process_launch_preparation_queue',
    'task_evaluation_launch_activation_worker':'process_launch_activation_queue',
    'task_evaluation_episode_compilation_worker':'process_episode_compilation_queue',
    'task_evaluation_sam31_prefix_adoption':'materialize_completed_prefix_adoption publish_adoption_release_binding validate_completed_prefix_adoption',
    'task_evaluation_sam31_preparation_execution':'process_sam31_phase_queue',
    'task_evaluation_scene_configuration_sam31_preparation_driver':'advance_sam31_preparation',
    'task_evaluation_launch_dispatcher':'dispatch_launch_request process_launch_queue',
    'task_evaluation_policy_canary_dispatcher':'dispatch_policy_canary_activation process_policy_canary_activation_results process_policy_canary_dispatch_queue',
    'task_evaluation_scene_configuration_submission_publication':'publish_scene_configuration_submission',
    'website_scene_dispatch':'materialize_website_attempt register_website_preparation resolve_website_source',
    'website_native_submission':'materialize_website_submission validate_website_publication verified_submission_inputs',
    'artifixer_completed_training_reuse':'stage_completed_review stage_completed_training',
    'control_plane_storage_pins':'release_storage_pin write_storage_pin',
    'pubsub_handoff_listener':'process_handoff_payload pull_and_process stage_handoff_capture',
    'task_evaluation_terminal_scene_attempt_settlement':'budget_retained_hold retained_hold settle_retired_attempt_rows sweep_retired_attempts validate_terminal_settlement',
    'task_evaluation_scene_owner_authority':'reopen_scene_intent validate_task_scene_owner',
    'live_pipeline_result_artifact_resolution':'resolve_live_pipeline_result_artifact',
    'live_pipeline_result_artifact_response':'ResultArtifactFileResponse.__call__ result_artifact_response',
}
_REQUIRED_COHORT=frozenset('blueprint_pipeline.'+module+':'+name
    for module,names in _COHORT_CALLS.items() for name in names.split())


def _kept(reason, *, journal=None, members=()):
    result=dict(schema_version='scene_retirement_action_result.v1',status='kept',action='KEEP',
                reason=reason,mutations=0,members=list(members))
    if journal is not None:
        result.update(status='incomplete',mutations=None,journal_initial_raw_ref=journal.initial_ref,
                      last_event_raw_ref=journal.prior_ref)
    return result


def _allowance(authority, now, monotonic):
    limits=authority['policy']['limits']
    _require(type(limits) is dict and set(limits) <= {
        'logical_payload_bytes','archive_bytes','remote_bytes','elapsed_seconds'},'scene_retirement_limits_invalid')
    return ActionAllowance(expires_at=authority['consent']['expires_at'],now=now,monotonic=monotonic,
        **{('local_bytes' if key=='logical_payload_bytes' else key):value for key,value in limits.items()})


def _bind_transport(transport,allowance):
    bind=getattr(transport,'bind_allowance',None)
    if bind is not None:
        allowance.tick()
        bind(allowance)
        allowance.tick()


def _path_relation(a,b):
    first,second=Path(a),Path(b)
    return first.is_relative_to(second) or second.is_relative_to(first)


def _generation(policy, member, *, expected_states, retired_token=None):
    key=hashlib.sha256(member['canonical_path'].encode()).hexdigest()+'.json'
    value,ref=load_document(Path(policy['generation_store'])/key,maximum=65536)
    _require(value.get('schema_version')=='scene_member_generation.v1'
             and value.get('state_digest')==canonical_digest(value,digest_field='state_digest')
             and value.get('canonical_path')==member['canonical_path']
             and value.get('generation_id')==member['generation_id']
             and value.get('owner_intent_id')==member['owner_intent_id']
             and value.get('owner_raw_ref')==member['owner_raw_ref']
             and value.get('state') in expected_states,'scene_retirement_generation_unavailable')
    if retired_token is not None:
        _require(value.get('retirement_token')==retired_token,'scene_retirement_generation_unavailable')
    return value,ref


def _transition(policy, prior, *, state, token, journal_ref, inventory_sha256=None, identity=None):
    store=Path(policy['generation_store'])
    key=hashlib.sha256(prior['canonical_path'].encode()).hexdigest()
    current=access._read(store/(key+'.json'))
    _require(current==prior,'scene_retirement_generation_changed')
    value=dict(prior,state=state,retirement_token=token,journal_sha256=journal_ref['sha256'],
               state_sequence=prior['state_sequence']+1)
    if inventory_sha256 is not None:
        value['inventory_sha256']=inventory_sha256
    if identity is not None:
        value.update(zip(('dev','ino','mode'),identity))
    value=_sealed(value)
    with _opened(store,directory=True) as (fd,info):
        _require(stat.S_IMODE(info.st_mode)==0o700
                 and (info.st_uid,info.st_gid)==access._service_identity(),
                 'scene_retirement_service_identity_unproven')
        # Global EX excludes every enrolled SH birth/publisher; immutable history
        # and the exact prior value are checked before updating this projection.
        _write(fd,key+'.'+value['generation_id']+'.'+str(value['state_sequence'])+'.'+state+'.json',
               value,parent_identity=_identity(info))
        _write(fd,key+'.json',value,parent_identity=_identity(info),replace=True)
    return value


def _installed_cohort(policy, allowance):
    rows=policy['consumer_cohort']
    _require(type(rows) is list and 1<=len(rows)<=256,'scene_retirement_cohort_unproven')
    names=set()
    for row in rows:
        allowance.tick()
        _require(type(row) is dict and type(row.get('entrypoint')) is str
                 and len(row['entrypoint'])<=256 and row['entrypoint'] not in names,
                 'scene_retirement_cohort_unproven')
        names.add(row['entrypoint'])
    _require(names==_REQUIRED_COHORT,'scene_retirement_cohort_unproven')
    for row in rows:
        allowance.tick()
        entrypoint=row['entrypoint']
        _require(type(entrypoint) is str and entrypoint.startswith('blueprint_pipeline.')
                 and entrypoint.count(':')==1,'scene_retirement_cohort_unproven')
        module_name,name=entrypoint.split(':')
        _require(module_name.replace('_','').replace('.','').isalnum()
                 and name.replace('_','').replace('.','').isalnum(),'scene_retirement_cohort_unproven')
        module=importlib.import_module(module_name)
        path=_canonical(module.__file__)
        with _opened(path) as (fd,info):
            _require(info.st_size<=1024*1024,'scene_retirement_cohort_unproven')
            digest=hashlib.sha256()
            remaining=info.st_size
            while remaining:
                allowance.tick()
                _require(_identity(os.fstat(fd))==_identity(info),'scene_retirement_cohort_unproven')
                raw=os.read(fd,min(65536,remaining))
                _require(raw,'scene_retirement_cohort_unproven')
                digest.update(raw)
                remaining-=len(raw)
            _require(_identity(os.fstat(fd))==_identity(info),'scene_retirement_cohort_unproven')
        _require('sha256:'+digest.hexdigest()==row['installed_source_sha'],'scene_retirement_cohort_unproven')
        target=module
        for part in name.split('.'):
            target=getattr(target,part,None)
        _require(getattr(target,'__scene_retirement_lifetime__',None)==_LIFETIME,
                 'scene_retirement_cohort_unproven')
    # This verifies actual installed/admitted callsites, never absence of an
    # unknown older process. Such lifetimes require their own retained evidence.


def _plan_members(plan, consent, allowance=None):
    rows=plan.get('measured_members')
    _require(type(rows) in (list,_Rows) and len(rows)<=10000,'scene_retirement_members_unproven')
    selected=consent['members']
    _require(selected,'scene_retirement_members_unproven')
    roots=[Path(member['canonical_path']) for member in selected]
    _require(not any(a!=b and a.is_relative_to(b) for a in roots for b in roots),
             'scene_retirement_members_unproven')
    matched=set()
    observed={row['path'] for row in rows if type(row) is dict and row.get('status')=='observed_scoped_metadata'}
    for row in rows:
        if allowance is not None:
            allowance.tick()
        _require(type(row) is dict and type(row.get('path')) is str,'scene_retirement_members_unproven')
        path=_canonical(row['path'])
        owners=[index for index,root in enumerate(roots) if path.is_relative_to(root)]
        _require(len(owners)==1,'scene_retirement_members_unproven')
        if row.get('status')=='coalesced_descendant_member':
            attributed=_canonical(row.get('attributed_root'))
            _require(str(attributed) in observed and path!=attributed and path.is_relative_to(attributed)
                     and attributed.is_relative_to(roots[owners[0]]) and not row.get('keeps'),
                     'scene_retirement_members_unproven')
            continue
        _require(row.get('status')=='observed_scoped_metadata','scene_retirement_members_unproven')
        # Shared scratch/release or an unselected external alias is not owned by
        # a consented parent. Positive keep reasons remain independent blockers.
        keeps=row.get('keeps',[])
        _require(type(keeps) is list and all(type(reason) is str for reason in keeps),
                 'scene_retirement_shared_or_unresolved_member')
        # The actual authenticated owner scope supplies the explicit retention
        # decision; private preservation is still proved before any mutation.
        # No metadata flag is cleared, and no other keep reason is waived.
        retention={'sam_evidence_retention_policy_required'} if (
            selected[owners[0]]['class'] in consent['private_archive_classes']
            and row.get('storage_class')!='cache') else set()
        _require(set(keeps)<=retention,'scene_retirement_shared_or_unresolved_member')
        matched.add(owners[0])
    _require(len(matched)==len(selected),'scene_retirement_members_unproven')


def _current_plan(policy, consent, retained, allowance, now, monotonic):
    context=retained.get('planner_context')
    _require(type(context) is dict and policy.get('reference_context')==context,
             'scene_retirement_installed_context_unproven')
    _require(retained.get('schema_version')=='task_evaluation_scene_lifecycle_plan.v1'
             and retained.get('intent_id')==consent['intent_id'],'scene_retirement_plan_invalid')
    allowance.tick()
    fresh=build_scene_lifecycle_plan(intent_id=consent['intent_id'],context=context,
                                    observed_at_epoch=now(),monotonic=monotonic)
    allowance.tick()
    _require(fresh.get('finished_observation',{}).get('status') in _TERMINAL
             and 'historical_lineage' in fresh,'scene_retirement_not_finished')
    provenance=fresh.get('selected_intent_provenance')
    _require(type(provenance) is dict and provenance.get('role')=='intent'
             and {key:provenance.get(key) for key in ('path','sha256','size_bytes')}==consent['intent_raw_ref'],
             'scene_retirement_owner_changed')
    _plan_members(fresh,consent,allowance)
    _installed_cohort(policy,allowance)
    _require(not fresh.get('reference_keeps') and not fresh.get('other_owner_capture_keeps'),
             'scene_retirement_reference_protected')
    fresh['reference_transfer']=validate_current_reference_transfer(fresh,allowance)
    _current_readers(policy,allowance)
    return fresh


def _current_readers(policy,allowance):
    from . import task_evaluation_scene_retirement_supervisor as native
    allowance.tick()
    callback=getattr(native,'require_current_reader_closure',None)
    _require(callable(callback),'scene_retirement_reader_closure_unproven')
    # Native admission must raise for every unknown/manual/old/HTTP gap.
    # No source catalogue, policy flag or diagnostic observation can replace it.
    callback(policy,allowance)
    allowance.tick()


def _resume_current_references(policy,consent,retained,allowance,now,monotonic):
    from .control_plane_reference_budget import ReferenceCollectionBudget
    from .task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from .task_evaluation_scene_lifecycle_plan import _context
    from .task_evaluation_scene_lifecycle_references import observe
    context=retained.get('planner_context')
    _require(type(context) is dict and policy.get('reference_context')==context,
             'scene_retirement_installed_context_unproven')
    _require(retained.get('schema_version')=='task_evaluation_scene_lifecycle_plan.v1'
             and retained.get('intent_id')==consent['intent_id'], 'scene_retirement_plan_invalid')
    _installed_cohort(policy,allowance)
    budget=ReferenceCollectionBudget._for_scene_lifecycle_plan(monotonic=monotonic,time_budget_seconds=30)
    sink=RetainedEmissionBudget(max_bytes=16*1024*1024,max_rows=10000,max_refs=10000,work_budget=budget)
    try:
        _context(context,budget)
        observation=observe(context,now(),budget,sink)
        allowance.tick()
        validate_current_reference_transfer(dict(retained,reference_observation=observation),allowance)
        _current_readers(policy,allowance)
    finally:
        budget.close()


def _partial_result(reason,journal,outcomes,policy,consent,pending,allowance,restore_context=None):
    result=_kept(reason,journal=journal,members=outcomes)
    if journal is None or pending is None:
        return result
    # The action context has unwound. Reacquire actual EX before advancing a
    # readable projection; a refused clock or a live reader leaves the already
    # durable private chain and prior public version untouched.
    try:
        with access.exclusive_scene_access() as locked:
            _require(locked==policy,'scene_retirement_policy_changed')
            progress=dict(status='incomplete',
                token=journal.token,intent_id=consent['intent_id'],members=outcomes,
                last_event_raw_ref=journal.prior_ref)
            if restore_context is not None:
                progress.update(restore_context)
            reference=publish_progress_receipt(policy,consent,pending,progress,allowance)
        result['intent_receipt_raw_ref']=reference
    except (ValueError,OSError,KeyError,TypeError,AttributeError):
        result['receipt_finalization']='unavailable_existing_durable_evidence_retained'
    return result


def _finish_retirement(policy,consent,initial,journal,pending,generations,outcomes,allowance,*,resumed=False):
    preserved=initial['preserved']
    closure=initial['metadata_closure_raw_ref']
    token=journal.token
    removed=recovery.removed_inode_counts(journal) if resumed else {}
    for index,generation in enumerate(generations):
        allowance.tick()
        outcome=detach_and_remove(preserved,member_index=index,generation_id=generation['generation_id'],
                  journal=journal,removed_inodes=removed)
        outcomes.append(outcome)
        if generation['state']!='retired':
            generations[index]=_transition(policy,generation,state='retired',token=token,
                    journal_ref=outcome['event_raw_ref'])
        pending=publish_progress_receipt(policy,consent,pending,dict(status='retiring',token=token,
            intent_id=consent['intent_id'],members=list(outcomes),last_event_raw_ref=journal.prior_ref),allowance)
    snapshot=journal.retired_snapshot(dict(initial,status='retired',members=consent['members'],
                   outcomes=outcomes,generations=generations))
    receipt=dict(schema_version='scene_retirement_receipt.v1',status='retired',intent_id=consent['intent_id'],
        token=token,members=outcomes,retired_journal_raw_ref=snapshot,fresh_remote_readback_verified=True,
        removed_allocated_bytes=sum(row['removed_allocated_bytes'] for row in outcomes),
        logical_bytes=sum(row['logical_bytes'] for row in outcomes),
        planned_unique_allocated_bytes=preserved['unique_allocated_bytes'],
        metadata_closure_raw_ref=closure)
    receipt['intent_receipt_raw_ref']=publish_terminal_receipt(policy,consent,pending,receipt,allowance)
    receipt['intent_receipt_path']=receipt['intent_receipt_raw_ref']['path']
    return receipt


def retire_scene(plan_path, consent_path, *, transport, now=time.time, monotonic=time.monotonic):
    journal=None
    pending=None
    policy=consent=allowance=None
    outcomes=[]
    try:
        authority=load_authority(consent_path,action='retire',now=now)
        policy,consent=authority['policy'],authority['consent']
        allowance=_allowance(authority,now,monotonic)
        _bind_transport(transport,allowance)
        _require(str(_canonical(str(plan_path)))==consent['plan_raw_ref']['path'],
                 'scene_retirement_raw_reference_changed')
        with access.exclusive_scene_access() as locked:
            _require(locked==policy,'scene_retirement_policy_changed')
            # Reload both authorities after EX; none of the earlier observation
            # can grant action if the installed records changed while waiting.
            current=load_authority(consent_path,action='retire',now=now)
            _require(current==authority,'scene_retirement_policy_changed')
            retained=selected_document(consent['plan_raw_ref'],maximum=16*1024*1024)
            resumed=recovery.select_retirement(policy,authority,allowance)
            if resumed is not None:
                journal,pending,initial=resumed
                recovery.bind_original_allowance(journal,initial,allowance)
                _resume_current_references(policy,consent,retained,allowance,now,monotonic)
                generations=recovery.resumed_generations(sys.modules[__name__],policy,consent,journal,initial)
                published_objects=initial.get('declared_byte_verification',{}).get('published_objects',[])
                recovery.reserve_phase(journal,initial['preserved'],readback=True,published_objects=published_objects)
                _consume(initial['preserved'],transport,allowance)
                verify_publication_rows(published_objects,transport,allowance)
                recovery.reserve_phase(journal,initial['preserved'])
                for index,generation in enumerate(generations):
                    if generation['state'] in {'active','restored-active'}:
                        event=journal.append('retiring',member_key=str(index),evidence={
                            'generation_id':generation['generation_id'],
                            'inventory_sha256':consent['members'][index]['inventory_sha256']})
                        generations[index]=_transition(policy,generation,state='retiring',token=journal.token,
                            journal_ref=event,inventory_sha256=consent['members'][index]['inventory_sha256'])
                return _finish_retirement(policy,consent,initial,journal,pending,generations,outcomes,allowance,resumed=True)
            fresh=_current_plan(policy,consent,retained,allowance,now,monotonic)
            generations=[]
            for member in consent['members']:
                allowance.tick()
                generation,_=_generation(policy,member,expected_states={'active','restored-active'})
                with _opened(member['canonical_path'],directory=True) as (_,info):
                    _require(_identity(info)==(member['dev'],member['ino'],member['mode'])
                             ==(generation['dev'],generation['ino'],generation['mode']),
                             'scene_retirement_generation_changed')
                generations.append(generation)
            token=secrets.token_hex(32)[:32]
            preserved=preserve_members([member['canonical_path'] for member in consent['members']],
                transport=transport,allowance=allowance,token=token)
            verified_bytes=_verify_declared_bytes(fresh,preserved,transport,allowance)
            for index,member in enumerate(consent['members']):
                _require(inventory_digest(preserved,index)==member['inventory_sha256'],
                         'scene_retirement_inventory_changed')
                _require(member['class'] in consent['private_archive_classes'],
                         'scene_retirement_private_archive_denied')
            closure=retain_metadata_closure(preserved,policy,generations,token,allowance)
            initial=dict(schema_version='scene_retirement_journal.v1',intent_id=consent['intent_id'],
                intent_raw_ref=consent['intent_raw_ref'],plan_raw_ref=consent['plan_raw_ref'],
                consent_raw_ref=authority['consent_raw_ref'],policy_sha256=consent['policy_sha256'],
                cohort_sha256=consent['cohort_sha256'],status='pending',members=consent['members'],
                generations=generations,preserved=preserved,metadata_closure_raw_ref=closure,
                action_allowance=allowance.checkpoint(),declared_byte_verification=verified_bytes)
            journal=SceneJournal.create(policy['journal_store'],token=token,initial=initial,allowance=allowance)
            pending=publish_pending_receipt(policy,consent,journal,preserved,allowance)
            def complete_records():
                for index,generation in enumerate(generations):
                    yield 'retiring',str(index),{
                        'generation_id':generation['generation_id'],
                        'inventory_sha256':consent['members'][index]['inventory_sha256']}
                    with _opened(consent['members'][index]['canonical_path'],directory=True) as (_,info):
                        _require(_identity(info)==tuple(preserved['members'][index]['physical_identity']),
                                 'scene_retirement_generation_changed')
                    with _opened(Path(consent['members'][index]['canonical_path']).parent,directory=True) as (_,info):
                        yield from removal_records(preserved,index,generation['generation_id'],journal,_identity(info))
            journal.preflight(complete_records())
            recovery.reserve_phase(journal,preserved)
            for index,generation in enumerate(generations):
                event=journal.append('retiring',member_key=str(index),evidence={
                    'generation_id':generation['generation_id'],'inventory_sha256':consent['members'][index]['inventory_sha256']})
                generations[index]=_transition(policy,generation,state='retiring',token=token,journal_ref=event,
                    inventory_sha256=consent['members'][index]['inventory_sha256'])
            return _finish_retirement(policy,consent,initial,journal,pending,generations,outcomes,allowance)
    except access.SceneRetirementAccessError as error:
        code=str(error) if str(error).startswith('scene_retirement_') and len(str(error))<=128 else 'scene_retirement_action_unproven'
        return _partial_result(code,journal,outcomes,policy,consent,pending,allowance)
    except (ValueError,OSError,KeyError,TypeError,AttributeError,OverflowError,RecursionError):
        return _partial_result('scene_retirement_action_unproven',journal,outcomes,policy,consent,pending,allowance)


def _finish_restore(policy,consent,retired,reference,journal,pending,restore_context,generations,outcomes,allowance,transport,*,was_restored=False):
    token=journal.token
    restored=restore_preserved_members(retired['preserved'],transport=transport,journal=journal)
    for index,(generation,outcome) in enumerate(zip(generations,restored)):
        if generation['state']!='restored-active':
            event=journal.append('restored-active',member_key=str(index),evidence=outcome)
            _transition(policy,generation,state='restored-active',token=retired['token'],journal_ref=event,
                identity=outcome['restore_identity'])
        else:
            _require(outcome['restore_identity']==[generation['dev'],generation['ino'],generation['mode']],
                     'scene_retirement_generation_changed')
        if index<len(outcomes):
            _require(outcomes[index]==outcome,'scene_retirement_restore_journal_unproven')
        else:
            outcomes.append(outcome)
        if not was_restored:
            pending=publish_progress_receipt(policy,consent,pending,dict(status='restoring',token=token,
                intent_id=consent['intent_id'],members=list(outcomes),last_event_raw_ref=journal.prior_ref,
                **restore_context),allowance)
    receipt=dict(schema_version='scene_restore_receipt.v1',status='restored',intent_id=consent['intent_id'],
         token=token,members=outcomes,retired_journal_raw_ref=reference)
    receipt['intent_receipt_raw_ref']=publish_progress_receipt(policy,consent,pending,dict(status='restored',
        token=token,intent_id=consent['intent_id'],members=outcomes,last_event_raw_ref=journal.prior_ref,
        **restore_context),allowance)
    receipt['intent_receipt_path']=receipt['intent_receipt_raw_ref']['path']
    return receipt


def restore_scene(retired_journal_path, consent_path, *, transport, now=time.time, monotonic=time.monotonic):
    journal=None
    pending=restore_context=None
    policy=consent=allowance=None
    outcomes=[]
    try:
        authority=load_authority(consent_path,action='restore',now=now)
        policy,consent=authority['policy'],authority['consent']
        allowance=_allowance(authority,now,monotonic)
        _bind_transport(transport,allowance)
        reference=consent['retired_journal_raw_ref']
        _require(str(_canonical(str(retired_journal_path)))==reference['path'],'scene_retirement_raw_reference_changed')
        with access.exclusive_scene_access() as locked:
            _require(locked==policy and load_authority(consent_path,action='restore',now=now)==authority,
                     'scene_retirement_policy_changed')
            retired=selected_document(reference,maximum=16*1024*1024,protected=True)
            _require(retired.get('schema_version')=='scene_retirement_journal.v1' and retired.get('status')=='retired'
                     and retired.get('journal_digest')==canonical_digest(retired,digest_field='journal_digest')
                     and retired.get('intent_id')==consent['intent_id'] and retired.get('members')==consent['members'],
                     'scene_retirement_restore_snapshot_invalid')
            resumed=recovery.select_restore(policy,authority,allowance,reference)
            if resumed is not None:
                journal,pending,initial,projection=resumed
                recovery.bind_original_allowance(journal,initial,allowance,restoring=True)
                _installed_cohort(policy,allowance)
                generations=recovery.resumed_restore_generations(sys.modules[__name__],policy,consent,journal,initial)
                outcomes.extend(recovery.restored_prefix(journal,generations))
                recovery.reserve_phase(journal,retired['preserved'],restoring=True)
                restore_context=dict(original_retirement_token=retired['token'],restore_journal_initial_raw_ref=journal.initial_ref)
                was_restored=projection['status']=='restored'
                if not was_restored:
                    pending=publish_progress_receipt(policy,consent,pending,dict(status='restoring',token=journal.token,
                        intent_id=consent['intent_id'],members=list(outcomes),last_event_raw_ref=journal.prior_ref,**restore_context),allowance)
                for index,generation in enumerate(generations):
                    if generation['state']=='retired':
                        event=journal.append('restoring',member_key=str(index),evidence={'generation_id':generation['generation_id']})
                        generations[index]=_transition(policy,generation,state='restoring',token=retired['token'],journal_ref=event)
                return _finish_restore(policy,consent,retired,reference,journal,pending,restore_context,
                                       generations,outcomes,allowance,transport,was_restored=was_restored)
            generations=[]
            for member in consent['members']:
                allowance.tick()
                generation,_=_generation(policy,member,expected_states={'retired'},retired_token=retired['token'])
                generations.append(generation)
                try:
                    os.lstat(member['canonical_path'])
                except FileNotFoundError:
                    pass
                else:
                    _require(False,'scene_retirement_restore_conflict')
            _installed_cohort(policy,allowance)
            token=secrets.token_hex(32)[:32]
            initial=dict(schema_version='scene_restore_journal.v1',status='restoring',intent_id=consent['intent_id'],
                intent_raw_ref=consent['intent_raw_ref'],members=consent['members'],generations=generations,
                original_retirement_token=retired['token'],retired_journal_raw_ref=reference,
                consent_raw_ref=authority['consent_raw_ref'],action_allowance=allowance.checkpoint())
            journal=SceneJournal.create(policy['journal_store'],token=token,initial=initial,allowance=allowance)
            def complete_restore_records():
                for index,generation in enumerate(generations):
                    yield 'restoring',str(index),{'generation_id':generation['generation_id']}
                    yield 'restored-active',str(index),dict(canonical_path=consent['members'][index]['canonical_path'],
                        outcome='restored',restore_identity=[2**64-1]*3)
                yield from restore_records(retired['preserved'],journal)
            journal.preflight(complete_restore_records())
            recovery.reserve_phase(journal,retired['preserved'],restoring=True)
            receipt_path=Path(policy['reference_context']['roots']['intent_root'])/consent['intent_id']/'scene-retired.v1.json'
            allowance.tick()
            _,pending=load_document(receipt_path,maximum=16*1024*1024)
            restore_context=dict(original_retirement_token=retired['token'],restore_journal_initial_raw_ref=journal.initial_ref)
            pending=publish_progress_receipt(policy,consent,pending,dict(status='restoring',token=token,
                intent_id=consent['intent_id'],members=[],last_event_raw_ref=journal.prior_ref,**restore_context),allowance)
            for index,generation in enumerate(generations):
                event=journal.append('restoring',member_key=str(index),evidence={'generation_id':generation['generation_id']})
                generations[index]=_transition(policy,generation,state='restoring',token=retired['token'],journal_ref=event)
            return _finish_restore(policy,consent,retired,reference,journal,pending,restore_context,
                                   generations,outcomes,allowance,transport)
    except access.SceneRetirementAccessError as error:
        code=str(error) if str(error).startswith('scene_retirement_') and len(str(error))<=128 else 'scene_retirement_action_unproven'
        return _partial_result(code,journal,outcomes,policy,consent,pending,allowance,restore_context)
    except (ValueError,OSError,KeyError,TypeError,AttributeError,OverflowError,RecursionError):
        return _partial_result('scene_retirement_action_unproven',journal,outcomes,policy,consent,pending,allowance,restore_context)
