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
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import load_authority, selected_document, load_document
from .task_evaluation_scene_retirement_generations import _write, _sealed
from .task_evaluation_scene_retirement_journal import SceneJournal
from .task_evaluation_scene_retirement_mutation import detach_and_remove, inventory_digest, removal_records
from .task_evaluation_scene_retirement_preservation import ActionAllowance, preserve_members
from .task_evaluation_scene_retirement_restore import restore_preserved_members
from .task_evaluation_scene_retirement_metadata import retain_metadata_closure
from .task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt, publish_terminal_receipt, publish_progress_receipt
from .task_evaluation_scene_lifecycle_plan import build_scene_lifecycle_plan


_LIFETIME='scene_retirement_lifetime.v1'
_TERMINAL={'completed','revoked_grace_elapsed','expired_grace_elapsed'}


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
    _require(rows,'scene_retirement_cohort_unproven')
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


def _plan_members(plan, consent):
    rows=plan.get('measured_members')
    _require(type(rows) is list and len(rows)<=10000,'scene_retirement_members_unproven')
    selected=consent['members']
    _require(selected,'scene_retirement_members_unproven')
    roots=[Path(member['canonical_path']) for member in selected]
    _require(not any(a!=b and a.is_relative_to(b) for a in roots for b in roots),
             'scene_retirement_members_unproven')
    matched=set()
    for row in rows:
        _require(type(row) is dict and type(row.get('path')) is str,'scene_retirement_members_unproven')
        path=_canonical(row['path'])
        owners=[index for index,root in enumerate(roots) if path.is_relative_to(root)]
        _require(len(owners)==1 and row.get('status')=='observed_scoped_metadata',
                 'scene_retirement_members_unproven')
        # Shared scratch/release or an unselected external alias is not owned by
        # a consented parent. Positive keep reasons remain independent blockers.
        _require(not row.get('keeps'),'scene_retirement_shared_or_unresolved_member')
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
    _require(fresh.get('selected_intent_provenance')==consent['intent_raw_ref'],
             'scene_retirement_owner_changed')
    _plan_members(fresh,consent)
    _installed_cohort(policy,allowance)
    observation=fresh.get('reference_observation',{})
    _require(not observation.get('blockers') and observation.get('child_scopes')
             and all(row.get('complete') is True for row in observation['child_scopes']),
             'scene_retirement_reference_scope_unproven')
    _require(not fresh.get('reference_keeps') and not fresh.get('other_owner_capture_keeps'),
             'scene_retirement_reference_protected')
    # The scoped historical observer intentionally cannot clear unsupported
    # runtime/settlement/reopen lifetimes. Those protections cannot be waived by
    # consent, source hashes, expiry or an empty queue.
    _require(not observation.get('record_dispositions') and not observation.get('protections'),
             'scene_retirement_reference_closure_unproven')
    return fresh


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
            _current_plan(policy,consent,retained,allowance,now,monotonic)
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
                generations=generations,preserved=preserved,metadata_closure_raw_ref=closure)
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
            for index,generation in enumerate(generations):
                event=journal.append('retiring',member_key=str(index),evidence={
                    'generation_id':generation['generation_id'],'inventory_sha256':consent['members'][index]['inventory_sha256']})
                generations[index]=_transition(policy,generation,state='retiring',token=token,journal_ref=event,
                    inventory_sha256=consent['members'][index]['inventory_sha256'])
            removed={}
            for index,generation in enumerate(generations):
                allowance.tick()
                outcome=detach_and_remove(preserved,member_index=index,generation_id=generation['generation_id'],
                                          journal=journal,removed_inodes=removed)
                outcomes.append(outcome)
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
    except access.SceneRetirementAccessError as error:
        code=str(error) if str(error).startswith('scene_retirement_') and len(str(error))<=128 else 'scene_retirement_action_unproven'
        return _partial_result(code,journal,outcomes,policy,consent,pending,allowance)
    except (ValueError,OSError,KeyError,TypeError,AttributeError,OverflowError,RecursionError):
        return _partial_result('scene_retirement_action_unproven',journal,outcomes,policy,consent,pending,allowance)


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
            journal=SceneJournal.create(policy['journal_store'],token=token,initial=dict(
                schema_version='scene_restore_journal.v1',status='restoring',intent_id=consent['intent_id'],
                intent_raw_ref=consent['intent_raw_ref'],members=consent['members'],
                original_retirement_token=retired['token'],
                retired_journal_raw_ref=reference,consent_raw_ref=authority['consent_raw_ref']),allowance=allowance)
            receipt_path=Path(policy['reference_context']['roots']['intent_root'])/consent['intent_id']/'scene-retired.v1.json'
            allowance.tick()
            _,pending=load_document(receipt_path,maximum=16*1024*1024)
            restore_context=dict(original_retirement_token=retired['token'],restore_journal_initial_raw_ref=journal.initial_ref)
            pending=publish_progress_receipt(policy,consent,pending,dict(status='restoring',token=token,
                intent_id=consent['intent_id'],members=[],last_event_raw_ref=journal.prior_ref,**restore_context),allowance)
            for index,generation in enumerate(generations):
                event=journal.append('restoring',member_key=str(index),evidence={'generation_id':generation['generation_id']})
                generations[index]=_transition(policy,generation,state='restoring',token=retired['token'],journal_ref=event)
            restored=restore_preserved_members(retired['preserved'],transport=transport,journal=journal)
            for index,(generation,outcome) in enumerate(zip(generations,restored)):
                event=journal.append('restored-active',member_key=str(index),evidence=outcome)
                _transition(policy,generation,state='restored-active',token=retired['token'],journal_ref=event,
                            identity=outcome['restore_identity'])
                outcomes.append(outcome)
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
    except access.SceneRetirementAccessError as error:
        code=str(error) if str(error).startswith('scene_retirement_') and len(str(error))<=128 else 'scene_retirement_action_unproven'
        return _partial_result(code,journal,outcomes,policy,consent,pending,allowance,restore_context)
    except (ValueError,OSError,KeyError,TypeError,AttributeError,OverflowError,RecursionError):
        return _partial_result('scene_retirement_action_unproven',journal,outcomes,policy,consent,pending,allowance,restore_context)
