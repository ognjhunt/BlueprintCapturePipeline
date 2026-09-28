"""Exact local metadata closure for existing retained accounting/reopen readers.

Original logical identity and raw bytes are preserved. This read-only lookup
never grants execution, ownership, remote access or a new generation.
"""
from __future__ import annotations

import hashlib
import os
import json
import secrets
import re
from itertools import chain
import stat
import sys
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _close_owned, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import raw_digest, raw_reference
from .task_evaluation_scene_retirement_generations import _named, _new_file
from .task_evaluation_scene_retirement_journal import _parent
from .task_evaluation_scene_retirement_preservation import _payload

MAX_METADATA_BYTES=16*1024*1024
MAX_ROW_BYTES=1024*1024


def _store(policy):
    # Fixed installer sibling, never a plan/receipt-supplied clone directory.
    store=_canonical(policy['journal_store']+'.metadata')
    with _opened(store,directory=True,protected=True) as (_,info):
        _require(info.st_uid==access._POLICY_UID and stat.S_IMODE(info.st_mode)==0o750,
                 'scene_retirement_metadata_permissions')
        return store,info.st_gid


def _read_raw(path,maximum,gid,*,audience=None):
    expected=audience or dict(uid=access._POLICY_UID,gid=gid,mode=0o640)
    _require(type(expected) is dict and set(expected)=={'uid','gid','mode'} and all(
        type(expected[key]) is int and 0<=expected[key]<2**32-1 for key in ('uid','gid'))
        and type(expected['mode']) is int and 0<=expected['mode']<=0o777,
        'scene_retirement_metadata_audience_unproven')
    # The containing index/store stays root protected. A known SAM clone keeps
    # its original leaf audience; OS access and exact mode/owner checks apply.
    with _opened(Path(path).parent,directory=True,protected=True) as (parent,parent_info), _opened(path) as (fd,info):
        _parent(Path(path).parent,parent,_identity(parent_info))
        _require(info.st_uid==expected['uid'] and info.st_gid==expected['gid']
                 and stat.S_IMODE(info.st_mode)==expected['mode'] and 0<info.st_size<=maximum,
                 'scene_retirement_metadata_changed')
        identity=_identity(info)
        remaining=info.st_size
        chunks=[]
        while remaining:
            _require(_identity(os.fstat(fd))==identity,'scene_retirement_metadata_changed')
            chunk=os.read(fd,min(65536,remaining))
            _require(chunk,'scene_retirement_metadata_changed')
            chunks.append(chunk)
            remaining-=len(chunk)
        _parent(Path(path).parent,parent,_identity(parent_info))
        after=os.fstat(fd)
        _require((_identity(after),after.st_size,after.st_mtime_ns,after.st_ctime_ns)==(
            identity,info.st_size,info.st_mtime_ns,info.st_ctime_ns),'scene_retirement_metadata_changed')
    return b''.join(chunks)


def _metadata_document(raw):
    # Bound lexical value occurrences before allocating the decoded graph.
    quoted=escaped=False
    values=0
    for character in raw:
        if quoted:
            if escaped:
                escaped=False
            elif character==92:
                escaped=True
            elif character==34:
                quoted=False
        elif character==34:
            quoted=True
            values+=1
        elif character in (44,91,123):
            values+=1
        _require(values<=100000,'scene_retirement_metadata_limit')
    return access._document(raw)




def _progress_directory(path):
    return path.parent.name=='source-progress' and len(path.name)<=255 and re.fullmatch(
        r'[A-Za-z0-9][A-Za-z0-9._-]*-[0-9a-f]{64}',path.name) is not None

def _sam_kind(path):
    name=path.name
    if re.fullmatch(r'sam31-[0-9a-f]{64}\.json',name):
        if path.parent.name in {'completed','failed','pending','processing','waiting_external'}:
            return ('task_evaluation_sam31_preparation_execution_job.v1','job_digest')
        if path.parent.name=='results':
            return ('task_evaluation_sam31_preparation_execution_result.v1','result_digest')
    if name=='phase_execution_receipt.v1.json' and re.fullmatch(r'sam31-[0-9a-f]{64}',path.parent.name) and re.fullmatch(r'[0-9a-f]{64}',path.parent.parent.name):
        return ('task_evaluation_sam31_phase_execution_receipt.v1','receipt_digest')
    if re.fullmatch(r'sam31-prefix-[0-9a-f]{64}\.json',name):
        return ('task_evaluation_sam31_completed_prefix_adoption.v1','adoption_digest')
    if _progress_directory(path.parent) and re.fullmatch(r'[0-9]{6}-[0-9a-f]{64}\.json',name):
        return ('task_evaluation_sam31_preparation_progress.v1','progress_digest')
    return None


def _known_sam(path,value):
    kind=_sam_kind(path)
    if kind is None:
        return False
    schema,seal=kind
    _require(value.get('schema_version')==schema and value.get(seal)==canonical_digest(value,digest_field=seal),
             'scene_retirement_metadata_unavailable')
    return True

def _reviewed_record(path,value):
    schema=value.get('schema_version')
    if path.name=='launch_receipt.json':
        return schema=='task_evaluation_launch_receipt.v1'
    if path.name in {'factory_receipt.json','website_scene_attempt_factory.v1.json'} and schema in {'website_scene_attempt_factory.v1','task_evaluation_public_scene_attempt_factory.v1',
                  'task_evaluation_completed_scene_attempt_factory.v1'}:
        _require(value.get('factory_digest')==canonical_digest(value,digest_field='factory_digest'),
                 'scene_retirement_metadata_changed')
        return True
    return path.name=='scene_configuration_preparation_request.v1.json' and (
        schema=='task_evaluation_launch_preparation_request.v1')


def _retained_bytes(reference,gid,*,audience=None):
    reference=raw_reference(reference)
    raw=_read_raw(reference['path'],MAX_ROW_BYTES,gid,audience=audience)
    _require(len(raw)==reference['size_bytes'] and raw_digest(raw)==reference['sha256'],
             'scene_retirement_metadata_changed')
    return raw


def _selected_closure(path):
    policy=access._policy()
    if policy is None:
        return None
    path=_canonical(str(path))
    roots=[Path(row['root']) for row in policy['roots'] if path.is_relative_to(Path(row['root']))]
    if not roots:
        return None
    selected=None
    for candidate in (path,*path.parents):
        if not any(candidate.is_relative_to(root) for root in roots):
            continue
        key=hashlib.sha256(str(candidate).encode()).hexdigest()+'.json'
        try:
            generation=access._read(Path(policy['generation_store'])/key)
        except FileNotFoundError:
            continue
        _require(generation.get('state_digest')==canonical_digest(generation,digest_field='state_digest')
                 and generation.get('canonical_path')==str(candidate),'scene_retirement_generation_unavailable')
        if generation.get('state') in {'active','restored-active'}:
            access._admit(policy,[path])
            return None
        _require(generation.get('state')=='retired','scene_retirement_generation_unavailable')
        _require(selected is None,'scene_retirement_metadata_ambiguous')
        selected=generation
    if selected is None:
        return None
    # A new named entry never gets hidden by historical closure bytes.
    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    else:
        _require(False,'scene_retirement_metadata_changed')
    token=selected.get('retirement_token')
    _require(type(token) is str and len(token)==32 and all(c in '0123456789abcdef' for c in token),
             'scene_retirement_metadata_unavailable')
    store,gid=_store(policy)
    closure=_metadata_document(_read_raw(store/(token+'.metadata.json'),MAX_METADATA_BYTES,gid))
    _require(closure.get('schema_version')=='scene_retirement_metadata_closure.v1'
             and closure.get('token')==token
             and closure.get('closure_digest')==canonical_digest(closure,digest_field='closure_digest'),
             'scene_retirement_metadata_unavailable')
    members=closure.get('members')
    rows=closure.get('rows')
    _require(type(members) is list and len(members)<=256 and type(rows) is list and len(rows)<=10000,
             'scene_retirement_metadata_limit')
    _require(dict(canonical_path=selected['canonical_path'],generation_id=selected['generation_id']) in members,
             'scene_retirement_metadata_unavailable')
    _require(all(type(row) is dict and set(row) in (
        {'logical_path','sha256','size_bytes','retained_raw_ref'},
        {'logical_path','sha256','size_bytes','retained_raw_ref','audience'}) for row in rows),
        'scene_retirement_metadata_unavailable')
    return store,gid,closure,selected


def read_logical_metadata(path, *, expected_sha256=None, expected_size_bytes=None):
    """Return exact retired bytes, or None for the unchanged native live path."""
    path=_canonical(str(path))
    with access.scene_access():
        selected=_selected_closure(path)
        if selected is None:
            return None
        store,gid,closure,_=selected
        rows=closure['rows']
        found=[]
        for row in rows:
            _require(type(row) is dict and set(row) in ({'logical_path','sha256','size_bytes','retained_raw_ref'}, {'logical_path','sha256','size_bytes','retained_raw_ref','audience'}),
                     'scene_retirement_metadata_unavailable')
            if row['logical_path']==str(path) and (expected_sha256 is None or row['sha256']==expected_sha256):
                found.append(row)
        _require(len(found)==1,'scene_retirement_metadata_unavailable')
        row=found[0]
        reference=raw_reference(row['retained_raw_ref'])
        _require(Path(reference['path']).parent==store and reference['sha256']==row['sha256']
                 and reference['size_bytes']==row['size_bytes']
                 and (expected_size_bytes is None or row['size_bytes']==expected_size_bytes),
                 'scene_retirement_metadata_changed')
        return _retained_bytes(reference,gid,audience=row.get('audience'))



def retained_progress_paths(directory):
    """Select a complete immutable original progress listing, never infer empty."""
    directory=_canonical(str(directory))
    with access.scene_access():
        selected=_selected_closure(directory)
        if selected is None:
            return None
        _,_,closure,_=selected
        listings=closure.get('progress_listings')
        _require(type(listings) is list and len(listings)<=10000,'scene_retirement_metadata_unavailable')
        _require(all(type(row) is dict and set(row)=={'logical_directory','paths'} for row in listings),
                 'scene_retirement_metadata_unavailable')
        found=[row for row in listings if row['logical_directory']==str(directory)]
        _require(len(found)==1 and type(found[0].get('paths')) is list and len(found[0]['paths'])<=10000,
                 'scene_retirement_metadata_unavailable')
        paths=[]
        known_paths={row['logical_path'] for row in closure['rows']}
        for name in found[0]['paths']:
            path=_canonical(name)
            _require(path.parent==directory and _sam_kind(path)==('task_evaluation_sam31_preparation_progress.v1','progress_digest'),
                     'scene_retirement_metadata_changed')
            _require(name in known_paths,'scene_retirement_metadata_changed')
            paths.append(path)
        _require(len(set(paths))==len(paths),'scene_retirement_metadata_changed')
        return tuple(sorted(paths))

def _publish_raw(store,name,raw,allowance,*,maximum=MAX_ROW_BYTES,audience=None):
    _require(len(raw)<=maximum,'scene_retirement_metadata_limit')
    temporary='.'+secrets.token_hex(16)+'.metadata'
    with _opened(store,directory=True,protected=True) as (parent,info):
        _require(info.st_uid==access._POLICY_UID and stat.S_IMODE(info.st_mode)==0o750)
        audience=audience or dict(uid=access._POLICY_UID,gid=info.st_gid,mode=0o640)
        expected=_identity(info)
        allowance.tick()
        _parent(store,parent,expected)
        fd,identity=_new_file(parent,temporary,parent_identity=expected)
        placed=False
        try:
            view=memoryview(raw)
            while view:
                allowance.tick()
                _parent(store,parent,expected)
                _named(parent,expected,temporary,fd,identity)
                count=os.write(fd,view[:65536])
                _require(count>0)
                view=view[count:]
            allowance.tick()
            _parent(store,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            os.fchown(fd,audience['uid'],audience['gid'])
            _named(parent,expected,temporary,fd,identity)
            allowance.tick()
            _parent(store,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            os.fchmod(fd,audience['mode'])
            identity=(identity[0],identity[1],stat.S_IFREG|audience['mode'])
            _named(parent,expected,temporary,fd,identity)
            allowance.tick()
            _parent(store,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            os.fsync(fd)
            allowance.tick()
            _parent(store,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            os.link(temporary,name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
            allowance.tick()
            _parent(store,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            os.unlink(temporary,dir_fd=parent)
            placed=True
            allowance.tick()
            _parent(store,parent,expected)
            os.fsync(parent)
        finally:
            incoming=sys.exc_info()[1]
            if not placed:
                try:
                    _parent(store,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    os.unlink(temporary,dir_fd=parent)
                except (ValueError,OSError):
                    if incoming is not None:
                        incoming.add_note('scene_retirement_metadata_cleanup_unproven')
            failure=_close_owned(fd,identity)
            if failure:
                if incoming is None:
                    raise access.SceneRetirementAccessError('scene_retirement_descriptor_cleanup_failed')
                incoming.add_note('scene_retirement_descriptor_cleanup_failed')
    return dict(path=str(store/name),sha256=raw_digest(raw),size_bytes=len(raw))


def retain_metadata_closure(preserved,policy,generations,token,allowance):
    """Publish exact raw JSON closure before any member detaches."""
    store,gid=_store(policy)
    rows,objects,total=[],{},0
    listings={}
    # A listing is issued from the complete preserved directory inventory, not
    # inferred from whichever supported rows happened to survive filtering.
    for item in chain(preserved['members'],preserved.get('directories',[])):
        allowance.tick()
        directory=Path(item['path']) if 'path' in item else Path(
            preserved['members'][item['member_index']]['path'])/item['relative_path']
        if _progress_directory(directory):
            _require(len(listings)<10000,'scene_retirement_metadata_limit')
            listings[str(directory)]=[]
    members=[dict(canonical_path=member['path'],generation_id=generation['generation_id'])
             for member,generation in zip(preserved['members'],generations)]
    output_bytes=512+len(json.dumps(members,separators=(',',':')).encode())
    output_bytes+=sum(len(json.dumps(directory).encode())+64 for directory in listings)
    _require(output_bytes<=MAX_METADATA_BYTES,'scene_retirement_metadata_limit')
    for row in preserved['files']:
        allowance.tick()
        logical=Path(preserved['members'][row['member_index']]['path'])/row['relative_path']
        sam_kind=_sam_kind(logical)
        if _progress_directory(logical.parent) and logical.suffix=='.json':
            _require(sam_kind is not None,'scene_retirement_metadata_unavailable')
        if sam_kind is None and Path(row['relative_path']).name not in {'launch_receipt.json','factory_receipt.json',
            'website_scene_attempt_factory.v1.json','scene_configuration_preparation_request.v1.json'}:
            continue
        _require(row['size_bytes']<=MAX_ROW_BYTES and total+row['size_bytes']<=MAX_METADATA_BYTES
                 and len(rows)<10000,'scene_retirement_metadata_limit')
        total+=row['size_bytes']  # Charge each logical occurrence before allocation.
        logical=Path(preserved['members'][row['member_index']]['path'])/row['relative_path']
        raw=b''.join(_payload(logical,row,allowance))
        _require(raw_digest(raw)==row['sha256'],'scene_retirement_metadata_changed')
        value=_metadata_document(raw)
        sam=_known_sam(logical,value) if sam_kind else False
        if not sam and not _reviewed_record(logical,value):
            continue
        service_uid,service_gid=access._service_identity()
        _require((sam and row['uid']==service_uid and row['gid']==service_gid and row['mode']==0o600)
                 or (row['gid']==gid and row['mode'] & 0o040) or row['mode'] & 0o004,
                 'scene_retirement_metadata_audience_unproven')
        # Every source ancestor must already permit the same service audience.
        for ancestor in (logical.parent,*logical.parent.parents):
            if not ancestor.is_relative_to(Path(preserved['members'][row['member_index']]['path'])):
                break
            with _opened(ancestor,directory=True) as (_,info):
                _require((sam and info.st_uid==row['uid'] and info.st_mode & 0o100)
                         or (info.st_gid==gid and info.st_mode & 0o010) or info.st_mode & 0o001,
                         'scene_retirement_metadata_audience_unproven')
        audience=dict(uid=row['uid'],gid=row['gid'],mode=row['mode']) if sam else None
        key=(row['sha256'],row['size_bytes'],tuple(audience.values()) if audience else None)
        suffix=hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()
        expected_ref=dict(path=str(store/(token+'.'+suffix+'.metadata.bin')),sha256=row['sha256'],size_bytes=row['size_bytes'])
        charge=dict(logical_path=str(logical),sha256=row['sha256'],size_bytes=row['size_bytes'],retained_raw_ref=expected_ref)
        if audience:
            charge['audience']=audience
        output_bytes+=len(json.dumps(charge,separators=(',',':')).encode())+1
        if sam_kind and sam_kind[0]=='task_evaluation_sam31_preparation_progress.v1':
            output_bytes+=len(json.dumps(str(logical)).encode())+1
        _require(output_bytes<=MAX_METADATA_BYTES,'scene_retirement_metadata_limit')
        if key not in objects:
            objects[key]=_publish_raw(store,token+'.'+suffix+'.metadata.bin',raw,allowance,audience=audience)
        output=dict(logical_path=str(logical),sha256=row['sha256'],size_bytes=row['size_bytes'],retained_raw_ref=objects[key])
        if sam:
            output['audience']=audience
        rows.append(output)
        if sam_kind and sam_kind[0]=='task_evaluation_sam31_preparation_progress.v1':
            listings.setdefault(str(logical.parent),[]).append(str(logical))
    closure=dict(schema_version='scene_retirement_metadata_closure.v1',token=token,
        members=members,rows=rows,
        progress_listings=[dict(logical_directory=directory,paths=sorted(paths)) for directory,paths in sorted(listings.items())])
    closure['closure_digest']=canonical_digest(closure,digest_field='closure_digest')
    raw=json.dumps(closure,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    _require(len(raw)<=MAX_METADATA_BYTES,'scene_retirement_metadata_limit')
    reference=_publish_raw(store,token+'.metadata.json',raw,allowance,maximum=MAX_METADATA_BYTES)
    return reference
