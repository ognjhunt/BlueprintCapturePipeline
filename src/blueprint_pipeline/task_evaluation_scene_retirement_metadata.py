"""Exact local metadata closure for existing retained accounting/reopen readers.

Original logical identity and raw bytes are preserved. This read-only lookup
never grants execution, ownership, remote access or a new generation.
"""
from __future__ import annotations

import hashlib
import os
import json
import secrets
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


def _read_raw(path,maximum,gid):
    with _opened(path,protected=True) as (fd,info):
        _require(info.st_uid==access._POLICY_UID and info.st_gid==gid
                 and stat.S_IMODE(info.st_mode)==0o640 and 0<info.st_size<=maximum,
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


def _retained_bytes(reference,gid):
    reference=raw_reference(reference)
    raw=_read_raw(reference['path'],MAX_ROW_BYTES,gid)
    _require(len(raw)==reference['size_bytes'] and raw_digest(raw)==reference['sha256'],
             'scene_retirement_metadata_changed')
    return raw


def read_logical_metadata(path, *, expected_sha256=None, expected_size_bytes=None):
    """Return exact retired bytes, or None for the unchanged native live path."""
    policy=access._policy()
    if policy is None:
        return None
    path=_canonical(str(path))
    with access.scene_access():
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
        found=[]
        for row in rows:
            _require(type(row) is dict and set(row)=={'logical_path','sha256','size_bytes','retained_raw_ref'},
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
        return _retained_bytes(reference,gid)


def _publish_raw(store,name,raw,allowance,*,maximum=MAX_ROW_BYTES):
    _require(len(raw)<=maximum,'scene_retirement_metadata_limit')
    temporary='.'+secrets.token_hex(16)+'.metadata'
    with _opened(store,directory=True,protected=True) as (parent,info):
        _require(info.st_uid==access._POLICY_UID and stat.S_IMODE(info.st_mode)==0o750)
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
            os.fchown(fd,access._POLICY_UID,info.st_gid)
            _named(parent,expected,temporary,fd,identity)
            allowance.tick()
            _parent(store,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            os.fchmod(fd,0o640)
            identity=(identity[0],identity[1],stat.S_IFREG|0o640)
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
    for row in preserved['files']:
        allowance.tick()
        if Path(row['relative_path']).name not in {'launch_receipt.json','factory_receipt.json',
            'website_scene_attempt_factory.v1.json','scene_configuration_preparation_request.v1.json'}:
            continue
        _require(row['size_bytes']<=MAX_ROW_BYTES and total+row['size_bytes']<=MAX_METADATA_BYTES
                 and len(rows)<10000,'scene_retirement_metadata_limit')
        total+=row['size_bytes']  # Charge each logical occurrence before allocation.
        logical=Path(preserved['members'][row['member_index']]['path'])/row['relative_path']
        raw=b''.join(_payload(logical,row,allowance))
        _require(raw_digest(raw)==row['sha256'],'scene_retirement_metadata_changed')
        value=_metadata_document(raw)
        if not _reviewed_record(logical,value):
            continue
        _require((row['gid']==gid and row['mode'] & 0o040) or row['mode'] & 0o004,
                 'scene_retirement_metadata_audience_unproven')
        # Every source ancestor must already permit the same service audience.
        for ancestor in (logical.parent,*logical.parent.parents):
            if not ancestor.is_relative_to(Path(preserved['members'][row['member_index']]['path'])):
                break
            with _opened(ancestor,directory=True) as (_,info):
                _require((info.st_gid==gid and info.st_mode & 0o010) or info.st_mode & 0o001,
                         'scene_retirement_metadata_audience_unproven')
        key=(row['sha256'],row['size_bytes'])
        if key not in objects:
            objects[key]=_publish_raw(store,token+'.'+row['sha256'][7:]+'.metadata.bin',raw,allowance)
        rows.append(dict(logical_path=str(logical),sha256=row['sha256'],size_bytes=row['size_bytes'],retained_raw_ref=objects[key]))
    closure=dict(schema_version='scene_retirement_metadata_closure.v1',token=token,
        members=[dict(canonical_path=member['path'],generation_id=generation['generation_id'])
                 for member,generation in zip(preserved['members'],generations)],rows=rows)
    closure['closure_digest']=canonical_digest(closure,digest_field='closure_digest')
    raw=json.dumps(closure,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    _require(len(raw)<=MAX_METADATA_BYTES,'scene_retirement_metadata_limit')
    reference=_publish_raw(store,token+'.metadata.json',raw,allowance,maximum=MAX_METADATA_BYTES)
    return reference
