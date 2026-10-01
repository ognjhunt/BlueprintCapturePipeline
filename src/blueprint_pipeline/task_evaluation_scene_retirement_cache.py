"""Normal owned preparation publication and exact digest-file generation evidence.

These records authenticate publication identity, never cache ownership or action
permission. Legacy bytes remain unadopted; retirement needs fresh whole-scene proof.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import stat
import time
import re
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import load_document, selected_document
from .task_evaluation_scene_retirement_generations import _birth_gate, _guard, _write, _sealed

_SIDE_FIELDS={'schema_version','intent_raw_ref','attempt_raw_ref','factory_raw_ref',
              'submission_request_raw_ref','request_digest','source_commit','authority_digest'}
_ERROR='scene_retirement_storage_authority_unproven'


def _raw(path):
    _,ref=load_document(path,maximum=65536)
    return ref


def _directory(path):
    path=_canonical(str(path))
    with _opened(path.parent,directory=True) as (fd,info):
        _guard(fd,_identity(info))
        try:
            os.mkdir(path.name,0o750,dir_fd=fd)
        except FileExistsError:
            pass
        _guard(fd,_identity(info))
        os.fsync(fd)
    with _opened(path,directory=True) as (_,info):
        _require(info.st_uid==os.geteuid() and not stat.S_IMODE(info.st_mode)&0o022,_ERROR)
    return path


def _sidecar(value):
    _require(type(value) is dict and set(value)==_SIDE_FIELDS
             and value['schema_version']=='scene_preparation_storage_authority.v1'
             and value['authority_digest']==canonical_digest(value,digest_field='authority_digest'),_ERROR)


def _attempt_identity(value,intent,attempt):
    parent=Path(value['intent_raw_ref']['path']).parent
    preparation_only=attempt.get('schema_version')=='task_evaluation_scene_preparation_attempt.v1'
    expected=parent/('preparation-attempts' if preparation_only else 'attempts')
    _require(attempt.get('schema_version') in {'task_evaluation_scene_preparation_attempt.v1',
                 'task_evaluation_scene_attempt.v1'}
             and Path(value['attempt_raw_ref']['path']).parent==expected
             and Path(value['attempt_raw_ref']['path']).name==attempt.get('attempt_id','')+'.json'
             and attempt.get('intent_id')==intent['intent_id'] and attempt.get('intent_digest')==intent['intent_digest']
             and attempt.get('attempt_digest')==canonical_digest(attempt,digest_field='attempt_digest'),_ERROR)
    if preparation_only:
        _require(type(attempt.get('maximum_spend_usd')) is int and attempt['maximum_spend_usd']==0
                 and attempt.get('provider_allocation_permitted') is False
                 and attempt.get('paid_authority_granted') is False,_ERROR)
    return parent,preparation_only


def _factory_identity(value,request,intent,attempt,factory,supplied):
    from .task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    _require(supplied==request and factory.get('schema_version') in {
        'website_scene_attempt_factory.v1','task_evaluation_public_scene_attempt_factory.v1',
        'task_evaluation_completed_scene_attempt_factory.v1'}
        and factory.get('status')=='publication_ready' and factory.get('provider_mutation_performed') is False
        and factory.get('intent_digest')==intent['intent_digest'] and factory.get('attempt_digest')==attempt['attempt_digest']
        and factory.get('submission_request')==value['submission_request_raw_ref']
        and factory.get('factory_digest')==canonical_digest(factory,digest_field='factory_digest')
        and factory.get('source_commit')==attempt.get('source_commit')==request.get('expected_production_commit')==value['source_commit']
        and request.get('scene_intent_digest')==intent['intent_digest']
        and request.get('task',{}).get('identity',{}).get('id')==intent['request']['task']['task_id']
        and value['request_digest']==launch_preparation_request_digest(request),_ERROR)
    policy=access._policy()
    _require(policy is not None and any(Path(value['factory_raw_ref']['path']).is_relative_to(Path(r['root']))
             for r in policy['roots']),_ERROR)


def _validate(value,request,*,now):
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    _sidecar(value)
    intent=selected_document(value['intent_raw_ref'],maximum=65536)
    _require(reopen_scene_intent(value['intent_raw_ref'],now=now)==intent,_ERROR)
    attempt=selected_document(value['attempt_raw_ref'],maximum=65536)
    parent,preparation_only=_attempt_identity(value,intent,attempt)
    if not preparation_only:
        from .task_evaluation_scene_execution_budget import validate_attempt_execution_budget
        validate_attempt_execution_budget(parent,intent,attempt)
    factory=selected_document(value['factory_raw_ref'],maximum=65536)
    supplied=selected_document(value['submission_request_raw_ref'],maximum=65536)
    _factory_identity(value,request,intent,attempt,factory,supplied)
    return intent,attempt


def _storage_history(value,allowance):
    """Authenticate original publication identity, without execution permission.

    Fresh protected retirement consent, generation and current-reference checks
    are separate. Expiry/revocation prevents producers from reopening authority;
    it does not rewrite the original accepted actor or immutable request bytes.
    """
    from .task_evaluation_scene_intake import validate_request
    _sidecar(value)
    records=[]
    for key in ('intent_raw_ref','attempt_raw_ref','factory_raw_ref','submission_request_raw_ref'):
        reference=value[key]
        _require(type(reference) is dict and type(reference.get('size_bytes')) is int
                 and 0<reference['size_bytes']<=65536,_ERROR)
        allowance.charge('local_bytes',reference['size_bytes'])
        records.append(selected_document(reference,maximum=65536))
        allowance.tick()
    intent,attempt,factory,request=records
    path=_canonical(value['intent_raw_ref']['path'])
    root=os.getenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT','')
    _require(root and path.name=='intent.json' and path.parent.parent==_canonical(root)
             and intent.get('schema_version')=='task_evaluation_scene_intent.v1'
             and path.parent.name==intent.get('intent_id')
             and intent.get('intent_digest')==canonical_digest(intent,digest_field='intent_digest'),_ERROR)
    trusted={item.strip() for item in os.getenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS',
                                               'blueprint-webapp').split(',') if item.strip()}
    _require(intent.get('authenticated_issuer') in trusted,_ERROR)
    accepted=validate_request(intent['request'],now=intent['accepted_at_epoch'])
    _require(accepted['consent']['accepted_by']==accepted['owner']['user_id'],_ERROR)
    _attempt_identity(value,intent,attempt)
    _factory_identity(value,request,intent,attempt,factory,request)
    return request


def publish_preparation_storage_authority(*,queue_root,request,intent_raw_ref,attempt_raw_ref,
                                         factory_raw_ref,submission_request_raw_ref,now=None):
    from .task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    if access._policy() is None:
        return None
    now=time.time() if now is None else now
    digest=launch_preparation_request_digest(request)
    value=dict(schema_version='scene_preparation_storage_authority.v1',intent_raw_ref=intent_raw_ref,
        attempt_raw_ref=attempt_raw_ref,factory_raw_ref=factory_raw_ref,
        submission_request_raw_ref=submission_request_raw_ref,request_digest=digest,
        source_commit=request['expected_production_commit'],authority_digest='')
    value['authority_digest']=canonical_digest(value,digest_field='authority_digest')
    with access.scene_access():
        _validate(value,request,now=now)
        store=_directory(Path(queue_root)/'scene-authorities')
        name=request['preparation_id']+'-'+digest[7:]+'.json'
        with _opened(store,directory=True) as (fd,info):
            try:
                _write(fd,name,value,parent_identity=_identity(info))
            except FileExistsError:
                existing,_=load_document(store/name,maximum=65536)
                _require(existing==value,_ERROR)
        return _raw(store/name)


def enroll_preparation_storage(*,queue_path,input_root,now=None):
    if access._policy() is None:
        return None
    path=_canonical(str(queue_path))
    sidecar=path.parent.parent/'scene-authorities'/path.name
    try:
        value,ref=load_document(sidecar,maximum=65536)
    except FileNotFoundError:
        return None
    envelope,_=load_document(path,maximum=65536)
    _require(envelope.get('schema_version')=='task_evaluation_launch_preparation_envelope.v1'
        and envelope.get('envelope_digest')==canonical_digest(envelope,digest_field='envelope_digest'),_ERROR)
    request=envelope['request']
    intent,attempt=_validate(value,request,now=time.time() if now is None else now)
    _require(path.name==request['preparation_id']+'-'+value['request_digest'][7:]+'.json'
             and envelope.get('request_digest')==value['request_digest'],_ERROR)
    target=Path(input_root)/request['preparation_id']
    generation=access.birth_scene_member(target,owner_intent_id=intent['intent_id'],
        owner_raw_ref=value['intent_raw_ref'],birth_request_raw_ref=value['attempt_raw_ref'],now=now)
    if generation is None:
        return None
    store=Path(access._policy()['generation_store'])
    key=hashlib.sha256(str(target).encode()).hexdigest()
    with _opened(store,directory=True) as (fd,info),_birth_gate(fd,key,parent_identity=_identity(info)):
        current,_=load_document(store/(key+'.json'),maximum=65536)
        _require(current==generation or current.get('source_storage_authority_raw_ref')==ref,_ERROR)
        if current.get('source_storage_authority_raw_ref')!=ref:
            current=_sealed(dict(current,source_storage_authority_raw_ref=ref,state_sequence=current['state_sequence']+1))
            _write(fd,key+'.json',current,parent_identity=_identity(info),replace=True)
    return current


def publication_authority(root):
    policy=access._policy()
    if policy is None:
        return None
    path=_canonical(str(root))
    roots=[Path(row['root']) for row in policy['roots'] if path.is_relative_to(Path(row['root']))]
    if not roots:
        return None
    for candidate in (path,*path.parents):
        if not any(candidate.is_relative_to(anchor) for anchor in roots):
            break
        key=hashlib.sha256(str(candidate).encode()).hexdigest()
        try:
            generation,_=load_document(Path(policy['generation_store'])/(key+'.json'),maximum=65536)
        except FileNotFoundError:
            continue
        _require(generation.get('canonical_path')==str(candidate)
                 and generation.get('schema_version')=='scene_member_generation.v1',_ERROR)
        with access.scene_access(path):
            ref=generation.get('source_storage_authority_raw_ref')
            if ref is None:
                return None
            selected_document(ref,maximum=65536)
            return ref
    return None


def publish_content_generation(path,temporary,*,digest,size_bytes,authority):
    """Publish one NEW verified cache name behind an exact regular-file generation."""
    if authority is None:
        return False
    policy=access._policy()
    path,temporary=_canonical(str(path)),_canonical(str(temporary))
    _require(path.parent==temporary.parent and path.name==digest[7:]
             and any(path.is_relative_to(Path(r['root'])) for r in policy['roots']),_ERROR)
    store=Path(policy['generation_store'])
    key=hashlib.sha256(str(path).encode()).hexdigest()
    with _opened(store,directory=True) as (ledger,ledger_info),_birth_gate(ledger,key,parent_identity=_identity(ledger_info)):
        with _opened(path.parent,directory=True) as (parent,parent_info),_opened(temporary) as (fd,info):
            identity=_identity(info)
            _require(info.st_size==size_bytes and info.st_nlink>=1,_ERROR)
            if info.st_nlink!=1:
                from . import task_evaluation_scene_retirement_generated as generated
                source=selected_document(authority,maximum=65536)
                _require(type(source) is dict and source.get('schema_version')==generated.SCHEMA,_ERROR)
                generated.validate_external_publication_source(source,source_path=temporary,
                    digest=digest,size_bytes=size_bytes)
                _guard(fd,identity)
                _guard(parent,_identity(parent_info))
                _require(_identity(os.stat(temporary.name,dir_fd=parent,follow_symlinks=False))==identity,_ERROR)
            prior=None
            try:
                prior,_=load_document(store/(key+'.json'),maximum=65536)
            except FileNotFoundError:
                pass
            _require(prior is None and not os.path.lexists(path),_ERROR)
            value=_sealed(dict(schema_version='scene_content_generation.v1',canonical_path=str(path),
                digest=digest,size_bytes=size_bytes,generation_id=secrets.token_hex(16),state='birth',
                dev=info.st_dev,ino=info.st_ino,mode=info.st_mode,uid=info.st_uid,gid=info.st_gid,
                source_publication_raw_ref=authority,state_sequence=0,retirement_token=None,journal_sha256=None))
            _write(ledger,key+'.json',value,parent_identity=_identity(ledger_info))
            _guard(parent,_identity(parent_info))
            _guard(fd,identity)
            _require(_identity(os.stat(temporary.name,dir_fd=parent,follow_symlinks=False))==identity,_ERROR)
            os.link(temporary.name,path.name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
            _guard(parent,_identity(parent_info))
            _guard(fd,identity)
            _require(_identity(os.stat(path.name,dir_fd=parent,follow_symlinks=False))==identity,_ERROR)
            _guard(parent,_identity(parent_info))
            os.fsync(parent)
            value=_sealed(dict(value,state='active',state_sequence=1))
            _write(ledger,key+'.json',value,parent_identity=_identity(ledger_info),replace=True)
    return True


def project_content(path,destination,*,authority):
    """Guard the actual cache-to-projection hardlink publication, including reuse."""
    if authority is None:
        return False
    path,destination=_canonical(str(path)),_canonical(str(destination))
    with access.scene_access(path),_opened(path) as (source,source_info):
        with _opened(path.parent,directory=True) as (source_parent,source_parent_info), \
                _opened(destination.parent,directory=True) as (parent,parent_info):
            parent_identity,identity=_identity(parent_info),_identity(source_info)
            original_parent=_identity(source_parent_info)
            _guard(source,identity)
            _guard(source_parent,original_parent)
            _guard(parent,parent_identity)
            _require(_identity(os.stat(path.name,dir_fd=source_parent,follow_symlinks=False))==identity,_ERROR)
            try:
                os.link(path.name,destination.name,src_dir_fd=source_parent,dst_dir_fd=parent,follow_symlinks=False)
            except FileExistsError:
                pass
            _guard(source,identity)
            _guard(source_parent,original_parent)
            _guard(parent,parent_identity)
            _require(_identity(os.stat(destination.name,dir_fd=parent,follow_symlinks=False))==identity,_ERROR)
            _guard(parent,parent_identity)
            os.fsync(parent)
    return True


def _cache_event(journal,event,key):
    rows=[row for row in journal.events if row['event']==event and row['member_key']==key]
    _require(len(rows)<=1,'scene_retirement_cache_journal_unproven')
    return rows[0] if rows else None


def _current_parent(path,fd,identity):
    _guard(fd,identity)
    with _opened(path,directory=True) as (_,named):
        _require(_identity(named)==identity,'scene_retirement_cache_parent_changed')
    _guard(fd,identity)


def remove_unused_content_for_gc(path,*,digest,size_bytes,minimum_age_seconds=0):
    """Existing blob-GC phase: registered names need owner-wide closure.

    Missing or disabled installation returns the original native path. This
    does not supply scene ownership, a private action grant or a new birth.
    The generic GC manifest observes hardlinks, bytes and age, but cannot prove
    that a request or retained lineage no longer selects a name at nlink one.
    """
    policy=access._policy()
    if policy is None:
        return None
    path=_canonical(str(path))
    _require(path.parent.name=='sha256' and path.name==digest and len(digest)==64
        and all(char in '0123456789abcdef' for char in digest)
        and type(size_bytes) is int and size_bytes>=0
        and type(minimum_age_seconds) is int and minimum_age_seconds>=0,
        'scene_retirement_cache_candidate_changed')
    # An ordinary reader or publisher owns the shared scene lifetime.  Taking
    # another shared lifetime here would let GC unlink its last name while that
    # reader still uses it, so mutation needs the same coarse exclusive fence
    # as the scene retirement engine.
    with access.exclusive_scene_access():
        _require(access._policy()==policy,'scene_retirement_policy_binding_unproven')
        generation_path=Path(policy['generation_store'])/(hashlib.sha256(str(path).encode()).hexdigest()+'.json')
        generation=access._read(generation_path)
        _require(generation.get('schema_version')=='scene_content_generation.v1'
                 and generation.get('canonical_path')==str(path)
                 and generation.get('digest')=='sha256:'+digest
                 and generation.get('size_bytes')==size_bytes
                 and generation.get('state') in {'active','restored-active'}
                 and generation.get('state_digest')==canonical_digest(generation,digest_field='state_digest'),
                 'scene_retirement_cache_generation_unavailable')
        # A current generation is publication identity, not cleanup authority.
        # Only the scene retirement action has the authenticated whole-scene
        # reference closure, consent, archive and journal needed to unlink it.
        raise access.SceneRetirementAccessError('scene_retirement_cache_reference_closure_unproven')


def remove_preserved_cache_aliases(preserved,*,journal,removed_inodes):
    """Internal EX action: remove only the separately inventoried LAST union aliases."""
    from .task_evaluation_scene_retirement_preservation import _snapshot,_payload
    aliases=preserved.get('cache_aliases',[])
    _require(type(aliases) is list and len(aliases)<=256,'scene_retirement_inventory_limit')
    remaining={}
    for index,alias in enumerate(aliases):
        journal.allowance.tick()
        if _cache_event(journal,'cache_unlinked','cache-'+str(index)) is None:
            inode=tuple(alias['physical_identity'][:2])
            remaining[inode]=remaining.get(inode,0)+1
    outcomes=[]
    for index,alias in enumerate(aliases):
        journal.allowance.tick()
        key='cache-'+str(index)
        path=_canonical(alias['path'])
        completed=_cache_event(journal,'cache_unlinked',key)
        if completed is not None:
            _require(completed['evidence']['canonical_path']==str(path) and not os.path.lexists(path),
                     'scene_retirement_cache_journal_unproven')
            outcomes.append(dict(completed['evidence'],event_raw_ref=completed['raw_ref']))
            continue
        planned=_cache_event(journal,'cache_unlink_planned',key)
        with _opened(path.parent,directory=True) as (parent,parent_info):
            parent_identity=_identity(parent_info)
            _require(list(parent_identity)==alias['parent_identity'],'scene_retirement_cache_parent_changed')
            if not os.path.lexists(path):
                _require(planned is not None and planned['evidence']['canonical_path']==str(path)
                         and planned['evidence']['original_identity']==alias['physical_identity'],
                         'scene_retirement_cache_journal_unproven')
                outcome=planned['evidence']['outcome']
                _current_parent(path.parent,parent,parent_identity)
                journal.allowance.tick()
                os.fsync(parent)
            else:
                with _opened(path) as (fd,info):
                    original=alias['snapshot']
                    observed=list(_snapshot(info))
                    removed=removed_inodes.get(tuple(alias['physical_identity'][:2]),0)
                    _require(observed[:-2]==original[:-2] and observed[-1]==original[-1]-removed
                             ==remaining[tuple(alias['physical_identity'][:2])] and observed[-1]>=1,
                             'scene_retirement_shared_inode')
                    digest=hashlib.sha256()
                    row=dict(alias,snapshot=observed)
                    for chunk in _payload(path,row,journal.allowance):
                        digest.update(chunk)
                    _require('sha256:'+digest.hexdigest()==alias['digest'],'scene_retirement_cache_alias_changed')
                    outcome=dict(outcome='removed',canonical_path=str(path),digest=alias['digest'],
                        size_bytes=alias['size_bytes'],removed_allocated_bytes=info.st_blocks*512 if info.st_nlink==1 else 0,
                        allocation_method='observed_file_st_blocks_512_last_union_link_unlinked')
                    evidence=dict(canonical_path=str(path),original_identity=alias['physical_identity'],
                        parent_identity=list(parent_identity),snapshot=observed,outcome=outcome)
                    if planned is None:
                        journal.append('cache_unlink_planned',member_key=key,evidence=evidence)
                    else:
                        _require(planned['evidence']==evidence,'scene_retirement_cache_journal_unproven')
                    journal.allowance.tick()
                    _current_parent(path.parent,parent,parent_identity)
                    _guard(fd,_identity(info))
                    _require(_identity(os.stat(path.name,dir_fd=parent,follow_symlinks=False))==_identity(info),
                             'scene_retirement_cache_alias_changed')
                    journal.allowance.tick()
                    os.unlink(path.name,dir_fd=parent)
                    journal.allowance.tick()
                    _current_parent(path.parent,parent,parent_identity)
                    journal.allowance.tick()
                    os.fsync(parent)
            inode=tuple(alias['physical_identity'][:2])
            removed_inodes[inode]=removed_inodes.get(inode,0)+1
            remaining[inode]-=1
            reference=journal.append('cache_unlinked',member_key=key,evidence=outcome)
            outcomes.append(dict(outcome,event_raw_ref=reference))
    return outcomes


def require_cache_restore_destinations(preserved,journal):
    aliases=preserved.get('cache_aliases',[])
    _require(type(aliases) is list and len(aliases)<=256,'scene_retirement_inventory_limit')
    for index,alias in enumerate(aliases):
        journal.allowance.tick()
        path=_canonical(alias['path'])
        if os.path.lexists(path):
            event=_cache_event(journal,'cache_restore_planned','cache-'+str(index))
            _require(event is not None and event['evidence']['canonical_path']==str(path),
                     'scene_retirement_cache_restore_conflict')
            with _opened(path) as (_,info):
                _require(list(_identity(info))==event['evidence']['restore_identity'],
                         'scene_retirement_cache_restore_conflict')


def restore_preserved_cache_aliases(preserved,roots,file_identities,journal):
    """Restore absent cache names as links to their verified restored projection bytes."""
    for index,alias in enumerate(preserved.get('cache_aliases',[])):
        journal.allowance.tick()
        key='cache-'+str(index)
        path=_canonical(alias['path'])
        source=roots[alias['member_index']]/alias['relative_path']
        identity=file_identities[(alias['member_index'],alias['relative_path'])]
        with _opened(source) as (fd,info),_opened(source.parent,directory=True) as (source_parent,source_info), \
                _opened(path.parent,directory=True) as (parent,parent_info):
            parent_identity,source_parent_identity=_identity(parent_info),_identity(source_info)
            _require(_identity(info)==identity and info.st_size==alias['size_bytes']
                     and info.st_uid==alias['uid'] and info.st_gid==alias['gid']
                     and stat.S_IMODE(info.st_mode)==alias['mode'],'scene_retirement_cache_restore_conflict')
            evidence=dict(canonical_path=str(path),restore_identity=list(identity),
                source_path=str(source),parent_identity=list(parent_identity),digest=alias['digest'],
                size_bytes=alias['size_bytes'])
            planned=_cache_event(journal,'cache_restore_planned',key)
            if planned is None:
                _require(not os.path.lexists(path),'scene_retirement_cache_restore_conflict')
                journal.append('cache_restore_planned',member_key=key,evidence=evidence)
            else:
                _require(planned['evidence']==evidence,'scene_retirement_cache_restore_conflict')
            _guard(fd,identity)
            _current_parent(source.parent,source_parent,source_parent_identity)
            _current_parent(path.parent,parent,parent_identity)
            _require(_identity(os.stat(source.name,dir_fd=source_parent,follow_symlinks=False))==identity,
                     'scene_retirement_cache_restore_conflict')
            if not os.path.lexists(path):
                journal.allowance.tick()
                os.link(source.name,path.name,src_dir_fd=source_parent,dst_dir_fd=parent,follow_symlinks=False)
            _guard(fd,identity)
            _current_parent(path.parent,parent,parent_identity)
            _require(_identity(os.stat(path.name,dir_fd=parent,follow_symlinks=False))==identity,
                     'scene_retirement_cache_restore_conflict')
            journal.allowance.tick()
            _current_parent(path.parent,parent,parent_identity)
            journal.allowance.tick()
            os.fsync(parent)
            done=_cache_event(journal,'cache_alias_restored',key)
            if done is None:
                journal.append('cache_alias_restored',member_key=key,evidence=evidence)
            else:
                _require(done['evidence']==evidence,'scene_retirement_cache_restore_conflict')


def _selected_queue_document(fresh,path,role,seal,allowance):
    matches={}
    for member in fresh.get('measured_members',[]):
        allowance.tick()
        for proof in member.get('source_provenance',[]):
            allowance.tick()
            if (proof.get('path')==str(path) and proof.get('role')==role
                    and proof.get('seal_field')==seal):
                ref={key:proof[key] for key in ('path','sha256','size_bytes')}
                matches[(ref['sha256'],ref['size_bytes'])]=(ref,proof)
    _require(len(matches)==1,_ERROR)
    ref,proof=next(iter(matches.values()))
    _require(0<ref['size_bytes']<=4*1024*1024,_ERROR)
    allowance.charge('local_bytes',ref['size_bytes'])
    value=selected_document(ref,maximum=4*1024*1024)
    _require(value.get(seal)==proof['seal_digest']
             and value[seal]==canonical_digest(value,digest_field=seal),_ERROR)
    return value,ref


def _recipe_stage_authority(source,request,source_ref,fresh,consent,allowance):
    """Derive transitive stage identities only from selected original recipe bytes."""
    from .task_evaluation_scene_construction_recipe import validate_scene_construction_recipe, CAPABILITY_ORDER
    from .task_evaluation_launch_preparation_worker import validate_recipe_request_binding
    context=fresh.get('planner_context',{}).get('roots',{})
    prep=request.get('preparation_id')
    recipe_ref=request.get('construction',{}).get('recipe')
    _require(type(prep) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}',prep)
             and type(recipe_ref) is dict and set(recipe_ref)=={'uri','digest','size_bytes'}
             and type(recipe_ref['digest']) is str and re.fullmatch(r'sha256:[0-9a-f]{64}',recipe_ref['digest'])
             and type(recipe_ref['size_bytes']) is int and 0<recipe_ref['size_bytes']<=4*1024*1024,_ERROR)
    queue=Path(context.get('preparation_queue_root',''))
    input_root=Path(context.get('preparation_input_root',''))
    cache_root=Path(context.get('content_store_root',''))
    _require(queue.is_absolute() and input_root.is_absolute() and cache_root.is_absolute()
             and Path(source_ref['path']).parent==queue/'scene-authorities',_ERROR)
    name=prep+'-'+source['request_digest'][7:]+'.json'
    _require(Path(source_ref['path']).name==name,_ERROR)
    envelope,envelope_ref=_selected_queue_document(fresh,queue/'materialized'/name,
        'preparation_envelopes','envelope_digest',allowance)
    result,result_ref=_selected_queue_document(fresh,queue/'results'/name,
        'preparation_results','result_digest',allowance)
    _require(envelope.get('schema_version')=='task_evaluation_launch_preparation_envelope.v1'
             and envelope.get('request')==request and envelope.get('request_digest')==source['request_digest']
             and result.get('schema_version')=='task_evaluation_launch_preparation_result.v1'
             and result.get('status')=='queued_for_production_scene_configuration'
             and result.get('preparation_id')==prep and result.get('run_id')==request.get('run_id')
             and result.get('source_commit')==source['source_commit']
             and result.get('full_byte_service_account_readback_passed') is True
             and result.get('provider_mutation_performed') is False
             and result.get('paid_execution_requested') is False,_ERROR)
    refs=result.get('references')
    _require(type(refs) is list and len(refs)<=256,_ERROR)
    parent=[row for row in refs if type(row) is dict and row.get('contract_path')=='construction.recipe']
    recipe_path=input_root/prep/recipe_ref['digest'][7:]
    _require(len(parent)==1 and all(parent[0].get(k)==recipe_ref[k] for k in ('uri','digest','size_bytes'))
             and parent[0].get('materialized_path')==str(recipe_path)
             and parent[0].get('full_byte_service_account_readback_passed') is True,_ERROR)
    allowance.charge('local_bytes',recipe_ref['size_bytes'])
    recipe_raw=dict(path=str(recipe_path),sha256=recipe_ref['digest'],size_bytes=recipe_ref['size_bytes'])
    recipe=selected_document(recipe_raw,maximum=4*1024*1024)
    recipe=validate_scene_construction_recipe(recipe)
    validate_recipe_request_binding(request=request,recipe=recipe)
    _require(len(recipe['stage_sequence'])==len(CAPABILITY_ORDER),_ERROR)
    stage_rows={}
    for row in refs:
        allowance.tick()
        if type(row) is not dict or not str(row.get('contract_path','')).startswith('construction.recipe.stage_sequence.'):
            continue
        match=re.fullmatch(r'construction\.recipe\.stage_sequence\.([0-9]+)\.configuration',row['contract_path'])
        _require(match is not None and int(match.group(1))<len(CAPABILITY_ORDER),_ERROR)
        index=int(match.group(1))
        _require(index not in stage_rows,_ERROR)
        stage_rows[index]=row
    _require(len(stage_rows)==len(CAPABILITY_ORDER),_ERROR)
    stages=[]
    supplemental=[]
    cache_paths={}
    for index,stage in enumerate(recipe['stage_sequence']):
        allowance.tick()
        row=stage_rows[index]
        reference=stage['configuration']
        digest=reference['digest']
        path=input_root/prep/'construction-stage-configurations'/digest[7:]
        _require(all(row.get(k)==reference[k] for k in ('uri','digest','size_bytes'))
                 and row.get('materialized_path')==str(path)
                 and row.get('full_byte_service_account_readback_passed') is True
                 and type(row.get('content_addressed_reuse')) is bool,_ERROR)
        alias=cache_root/digest[7:]
        prior=cache_paths.setdefault(str(alias),reference)
        _require(prior==reference,_ERROR)
        stages.append(dict(index=index,contract_path=row['contract_path'],reference=reference,
                           projected_path=str(path),cache_path=str(alias)))
    destination=recipe.get('supplemental_destination')
    if destination is not None:
        for field in ('authoring_receipt','simready_result'):
            allowance.tick()
            contract_path='construction.recipe.supplemental_destination.'+field
            matches=[row for row in refs if type(row) is dict and row.get('contract_path')==contract_path]
            _require(len(matches)==1,_ERROR)
            row=matches[0]
            reference=destination[field]
            digest=reference['digest']
            path=input_root/prep/'construction-supplemental-destination'/digest[7:]
            _require(all(row.get(k)==reference[k] for k in ('uri','digest','size_bytes'))
                     and row.get('materialized_path')==str(path)
                     and row.get('full_byte_service_account_readback_passed') is True
                     and type(row.get('content_addressed_reuse')) is bool,_ERROR)
            alias=cache_root/digest[7:]
            prior=cache_paths.setdefault(str(alias),reference)
            _require(prior==reference,_ERROR)
            supplemental.append(dict(contract_path=contract_path,reference=reference,
                                     projected_path=str(path),cache_path=str(alias)))
    targets={row['canonical_path']:row for row in consent['cache_objects']}
    _require(all(path in targets and targets[path]['digest']==ref['digest']
                 and targets[path]['size_bytes']==ref['size_bytes'] for path,ref in cache_paths.items()),_ERROR)
    return dict(kind='selected_recipe_stage_authority.v1',source_raw_ref=source_ref,
                envelope_raw_ref=envelope_ref,result_raw_ref=result_ref,recipe_raw_ref=recipe_raw,
                recipe_digest=recipe['recipe_digest'],stages=stages,supplemental=supplemental,
                cache_objects=[dict(targets[path]) for path in sorted(cache_paths)])


def validate_cache_objects(policy,consent,allowance,*,fresh=None):
    """Exact target/publication proof only; native current-reference closure is separate."""
    from .control_plane_storage_gc import DEFAULT_MINIMUM_AGE_SECONDS
    from .task_evaluation_launch_preparation_worker import collect_preparation_references
    objects=consent.get('cache_objects',[])
    _require(type(objects) is list and len(objects)<=256,'scene_retirement_inventory_limit')
    selected=[]
    recipe_proofs={}
    for row in objects:
        allowance.tick()
        for reference in (row['generation_raw_ref'],row['source_raw_ref']):
            allowance.charge('local_bytes',reference['size_bytes'])
        generation=selected_document(row['generation_raw_ref'],maximum=65536)
        source=selected_document(row['source_raw_ref'],maximum=65536)
        _require(set(generation)=={'schema_version','canonical_path','digest','size_bytes','generation_id',
            'state','dev','ino','mode','uid','gid','source_publication_raw_ref','state_sequence',
            'retirement_token','journal_sha256','state_digest'}
            and generation['schema_version']=='scene_content_generation.v1'
            and generation['state'] in {'active','restored-active'}
            and generation['state_digest']==canonical_digest(generation,digest_field='state_digest')
            and all(row[key]==generation[key] for key in ('canonical_path','digest','size_bytes','generation_id'))
            and generation['source_publication_raw_ref']==row['source_raw_ref']
            and source.get('intent_raw_ref')==consent['intent_raw_ref'],_ERROR)
        from . import task_evaluation_scene_retirement_generated as generated
        if type(source) is dict and source.get('schema_version')==generated.SCHEMA:
            verified=generated.validate_publication(source,policy=policy,consent=consent,allowance=allowance)
            _require(verified['digest']==row['digest'] and verified['size_bytes']==row['size_bytes']
                and verified['intent_raw_ref']==consent['intent_raw_ref'],_ERROR)
        else:
            _require(type(source) is dict and set(source)==_SIDE_FIELDS,_ERROR)
            request=_storage_history(source,allowance)
            allowance.tick()
            refs=collect_preparation_references(request)
            direct=any(ref['digest']==row['digest'] and ref['size_bytes']==row['size_bytes'] for ref in refs)
            if not direct:
                _require(fresh is not None,_ERROR)
                key=tuple(row['source_raw_ref'][field] for field in ('path','sha256','size_bytes'))
                if key not in recipe_proofs:
                    recipe_proofs[key]=_recipe_stage_authority(source,request,row['source_raw_ref'],fresh,consent,allowance)
                _require(any(stage['cache_path']==row['canonical_path']
                             and stage['reference']['digest']==row['digest']
                             and stage['reference']['size_bytes']==row['size_bytes']
                             for stage in [*recipe_proofs[key]['stages'],*recipe_proofs[key]['supplemental']]),_ERROR)
        path=_canonical(row['canonical_path'])
        with _opened(path) as (_,info):
            allowance.tick()
            _require(_identity(info)==(generation['dev'],generation['ino'],generation['mode'])
                     and info.st_size==row['size_bytes'] and info.st_uid==generation['uid']
                     and info.st_gid==generation['gid'],_ERROR)
            _require(allowance.last_wall-info.st_mtime>=DEFAULT_MINIMUM_AGE_SECONDS,
                     'scene_retirement_cache_idle_grace_unproven')
        selected.append(dict(row))
    if fresh is not None:
        return selected,list(recipe_proofs.values())
    return selected
