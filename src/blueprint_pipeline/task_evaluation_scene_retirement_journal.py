"""Immutable fsynced operation records; a journal is evidence, never consent."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path

from . import task_evaluation_scene_retirement_access as access
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_retirement_access import _canonical, _close_owned, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import TOKEN, raw_digest, raw_reference, load_document
from .task_evaluation_scene_retirement_generations import _guard, _named, _new_file

EVENTS={'pin_release_planned','pin_released','pin_restore_planned','pin_restored','cache_unlink_planned','cache_unlinked','cache_restore_planned','cache_alias_restored','allowance_reserved','detach_planned','detached','member_removed','retiring','retired','restoring',
        'restore_directory_created','restore_file_created','member_restored','restored-active','kept',
        'leaf_unlink_planned','leaf_unlinked','directory_unlink_planned','directory_unlinked'}
MAX_EVENTS=10000
MAX_JOURNAL_BYTES=32*1024*1024


def _parent(directory,fd,expected):
    # Both retained numeric identity and the currently named protected ancestry
    # must still lead to the original directory before each publication mutation.
    _guard(fd,expected)
    with _opened(directory,directory=True,protected=True) as (_,current):
        _require(_identity(current)==expected,'scene_retirement_journal_parent_changed')
    _guard(fd,expected)


def publish_record(directory,name,value,*,maximum,allowance):
    directory=_canonical(str(directory))
    _require(type(name) is str and '/' not in name and name not in {'','.','..'})
    temporary='.'+secrets.token_hex(16)+'.pending'
    encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False)
    digest=hashlib.sha256()
    size=0
    with _opened(directory,directory=True,protected=True) as (parent,info):
        _require(stat.S_IMODE(info.st_mode)==0o700 and info.st_uid==access._POLICY_UID,
                 'scene_retirement_journal_permissions')
        expected=_identity(info)
        allowance.tick()
        _parent(directory,parent,expected)
        fd,identity=_new_file(parent,temporary,parent_identity=expected,action_guard=allowance.tick)
        placed=False
        try:
            buffer=bytearray()
            def flush():
                if not buffer:
                    return
                chunk=bytes(buffer)
                buffer.clear()
                view=memoryview(chunk)
                while view:
                    allowance.tick()
                    _parent(directory,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    allowance.tick()
                    written=os.write(fd,view)
                    _require(written>0)
                    view=view[written:]
            for text in encoder.iterencode(value):
                allowance.tick()
                raw=text.encode('utf-8')
                _require(size+len(raw)<=maximum,'scene_retirement_journal_limit')
                size+=len(raw)
                digest.update(raw)
                offset=0
                while offset<len(raw):
                    amount=min(65536-len(buffer),len(raw)-offset)
                    buffer.extend(raw[offset:offset+amount])
                    offset+=amount
                    if len(buffer)==65536:
                        flush()
            flush()
            allowance.tick()
            _parent(directory,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            allowance.tick()
            os.fsync(fd)
            allowance.tick()
            _parent(directory,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            allowance.tick()
            os.link(temporary,name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
            allowance.tick()
            _parent(directory,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            _require(_identity(os.stat(name,dir_fd=parent,follow_symlinks=False))==identity)
            allowance.tick()
            os.unlink(temporary,dir_fd=parent)
            placed=True
            allowance.tick()
            _parent(directory,parent,expected)
            _guard(fd,identity)
            allowance.tick()
            os.fsync(parent)
            allowance.tick()
        finally:
            incoming=sys.exc_info()[1]
            cleanup_failure=False
            if not placed:
                try:
                    # Cleanup has no clock dependency and never removes a
                    # substituted token, entry, parent or published record.
                    _parent(directory,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    os.unlink(temporary,dir_fd=parent)
                except FileNotFoundError:
                    pass
                except (ValueError,OSError):
                    cleanup_failure=True
            failure=_close_owned(fd,identity)
            if failure or cleanup_failure:
                if incoming is None:
                    raise access.SceneRetirementAccessError('scene_retirement_descriptor_cleanup_failed')
                incoming.add_note('scene_retirement_descriptor_cleanup_failed')
    return dict(path=str(directory/name),sha256='sha256:'+digest.hexdigest(),size_bytes=size)


class SceneJournal:
    def __init__(self,directory,token,initial_ref,allowance):
        self.directory,self.token,self.initial_ref,self.allowance=directory,token,initial_ref,allowance
        self.sequence=0
        self.prior_ref=initial_ref
        self.bytes=initial_ref['size_bytes']
        self.events=[]

    @classmethod
    def create(cls,directory,*,token,initial,allowance):
        _require(type(token) is str and TOKEN.fullmatch(token))
        _require(type(initial) is dict)
        value=dict(initial,token=token,sequence=0,prior_event_sha256=None)
        value['journal_digest']=canonical_digest(value,digest_field='journal_digest')
        reference=publish_record(directory,token+'.initial.json',value,
                                 maximum=16*1024*1024,allowance=allowance)
        return cls(Path(directory),token,reference,allowance)

    @classmethod
    def resume(cls,initial_ref,*,allowance):
        initial_ref=raw_reference(initial_ref)
        path=_canonical(initial_ref['path'])
        token=path.name.removesuffix('.initial.json')
        _require(TOKEN.fullmatch(token) and path.name==token+'.initial.json')
        allowance.tick()
        initial,observed=load_document(path,maximum=16*1024*1024,protected=True)
        _require(observed==initial_ref and initial.get('token')==token and initial.get('sequence')==0
                 and initial.get('prior_event_sha256') is None
                 and initial.get('journal_digest')==canonical_digest(initial,digest_field='journal_digest'),
                 'scene_retirement_journal_chain_unproven')
        result=cls(path.parent,token,initial_ref,allowance)
        sequences=set()
        with _opened(path.parent,directory=True,protected=True) as (fd,info):
            _require(stat.S_IMODE(info.st_mode)==0o700)
            expected=_identity(info)
            examined=0
            with os.scandir(fd) as entries:
                for entry in entries:
                    allowance.tick()
                    _parent(path.parent,fd,expected)
                    examined+=1
                    _require(examined<=100000,'scene_retirement_journal_limit')
                    name=entry.name
                    if not name.startswith(token+'.') or name==path.name:
                        continue
                    sequence=name.removeprefix(token+'.').removesuffix('.json')
                    _require(sequence.isdecimal() and str(int(sequence))==sequence and name==token+'.'+sequence+'.json'
                             and 0<int(sequence)<=MAX_EVENTS and len(sequences)<MAX_EVENTS,
                             'scene_retirement_journal_chain_unproven')
                    sequences.add(int(sequence))
            after=os.fstat(fd)
            _require((after.st_size,after.st_mtime_ns,after.st_ctime_ns)==(info.st_size,info.st_mtime_ns,info.st_ctime_ns),
                     'scene_retirement_journal_chain_unproven')
        _require(sequences==set(range(1,len(sequences)+1)),'scene_retirement_journal_chain_unproven')
        for sequence in range(1,len(sequences)+1):
            allowance.tick()
            value,reference=load_document(path.parent/(token+'.'+str(sequence)+'.json'),maximum=65536,protected=True)
            _require(value.get('schema_version')=='scene_retirement_journal_event.v1' and value.get('token')==token
                     and value.get('sequence')==sequence and value.get('prior_event_sha256')==result.prior_ref['sha256']
                     and value.get('event') in EVENTS and type(value.get('member_key')) is str
                     and len(value['member_key'])<=128 and type(value.get('evidence')) is dict
                     and value.get('event_digest')==canonical_digest(value,digest_field='event_digest'),
                     'scene_retirement_journal_chain_unproven')
            _require(result.bytes+reference['size_bytes']<=MAX_JOURNAL_BYTES,'scene_retirement_journal_limit')
            result.events.append(dict(value,raw_ref=reference))
            result.sequence=sequence
            result.prior_ref=reference
            result.bytes+=reference['size_bytes']
        allowance.tick()
        return result

    def preflight(self,records):
        """Prove the remaining complete operation fits before payload mutation.

        The private journal has one EX-protected writer. Forecasts use the
        largest supported sequence/ref widths and stream their framing rather
        than retaining an encoded document or granting a second allowance.
        Every actual append still checks these same limits independently.
        """
        count,total=self.sequence,self.bytes
        encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False)
        for event,member_key,evidence in records:
            self.allowance.tick()
            count+=1
            _require(count<=MAX_EVENTS and event in EVENTS and type(evidence) is dict,
                     'scene_retirement_journal_limit')
            value=dict(schema_version='scene_retirement_journal_event.v1',token=self.token,
                       sequence=MAX_EVENTS,prior_event_sha256='sha256:'+'f'*64,event=event,
                       member_key=member_key,evidence=evidence,event_digest='sha256:'+'f'*64)
            size=0
            for piece in encoder.iterencode(value):
                self.allowance.tick()
                size+=len(piece.encode('utf-8'))
                _require(size<=65536 and total+size<=MAX_JOURNAL_BYTES,
                         'scene_retirement_journal_limit')
            total+=size
        self.allowance.tick()

    def append(self,event,*,member_key,evidence):
        self.allowance.tick()
        _require(event in EVENTS and type(member_key) is str and len(member_key)<=128)
        _require(type(evidence) is dict and self.sequence<MAX_EVENTS,'scene_retirement_journal_limit')
        value=dict(schema_version='scene_retirement_journal_event.v1',token=self.token,
                   sequence=self.sequence+1,prior_event_sha256=self.prior_ref['sha256'],
                   event=event,member_key=member_key,evidence=evidence)
        value['event_digest']=canonical_digest(value,digest_field='event_digest')
        maximum=min(65536,MAX_JOURNAL_BYTES-self.bytes)
        reference=publish_record(self.directory,self.token+'.'+str(value['sequence'])+'.json',
                                 value,maximum=maximum,allowance=self.allowance)
        self.sequence=value['sequence']
        self.bytes+=reference['size_bytes']
        self.prior_ref=reference
        self.events.append(dict(value,raw_ref=reference))
        return reference

    def retired_snapshot(self,value):
        self.allowance.tick()
        snapshot=dict(value,token=self.token,sequence=self.sequence,
                      prior_event_sha256=self.prior_ref['sha256'],status='retired')
        snapshot['journal_digest']=canonical_digest(snapshot,digest_field='journal_digest')
        raw=json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        _require(len(raw)<=16*1024*1024,'scene_retirement_journal_limit')
        # The directory is installer-owned; no service creates authority paths.
        return publish_record(self.directory/'retired',raw_digest(raw)[7:]+'.json',snapshot,
                              maximum=16*1024*1024,allowance=self.allowance)
