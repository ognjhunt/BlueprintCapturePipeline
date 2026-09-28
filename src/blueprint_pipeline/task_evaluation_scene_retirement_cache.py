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


def _validate(value,request,*,now):
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    from .task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    _require(type(value) is dict and set(value)==_SIDE_FIELDS
             and value['schema_version']=='scene_preparation_storage_authority.v1'
             and value['authority_digest']==canonical_digest(value,digest_field='authority_digest'),_ERROR)
    intent=selected_document(value['intent_raw_ref'],maximum=65536)
    _require(reopen_scene_intent(value['intent_raw_ref'],now=now)==intent,_ERROR)
    attempt=selected_document(value['attempt_raw_ref'],maximum=65536)
    parent=Path(value['intent_raw_ref']['path']).parent
    preparation_only=attempt.get('schema_version')=='task_evaluation_scene_preparation_attempt.v1'
    expected=parent/('preparation-attempts' if preparation_only else 'attempts')
    _require(Path(value['attempt_raw_ref']['path']).parent==expected
             and Path(value['attempt_raw_ref']['path']).name==attempt.get('attempt_id','')+'.json'
             and attempt.get('intent_id')==intent['intent_id'] and attempt.get('intent_digest')==intent['intent_digest']
             and attempt.get('attempt_digest')==canonical_digest(attempt,digest_field='attempt_digest'),_ERROR)
    if preparation_only:
        _require(attempt.get('maximum_spend_usd')==0 and attempt.get('provider_allocation_permitted') is False
                 and attempt.get('paid_authority_granted') is False,_ERROR)
    else:
        from .task_evaluation_scene_execution_budget import validate_attempt_execution_budget
        validate_attempt_execution_budget(parent,intent,attempt)
    factory=selected_document(value['factory_raw_ref'],maximum=65536)
    supplied=selected_document(value['submission_request_raw_ref'],maximum=65536)
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
    return intent,attempt


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
            _require(info.st_size==size_bytes and info.st_nlink==1,_ERROR)
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
