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
from .task_evaluation_scene_retirement_authority import TOKEN, raw_digest
from .task_evaluation_scene_retirement_generations import _guard, _named, _new_file

EVENTS={'detach_planned','detached','member_removed','retiring','retired','restoring',
        'restore_directory_created','restore_file_created','member_restored','restored-active','kept'}


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
        fd,identity=_new_file(parent,temporary,parent_identity=expected)
        placed=False
        try:
            for text in encoder.iterencode(value):
                allowance.tick()
                raw=text.encode('utf-8')
                _require(size+len(raw)<=maximum,'scene_retirement_journal_limit')
                size+=len(raw)
                digest.update(raw)
                view=memoryview(raw)
                while view:
                    allowance.tick()
                    _parent(directory,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    written=os.write(fd,view)
                    _require(written>0)
                    view=view[written:]
            allowance.tick()
            _parent(directory,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            os.fsync(fd)
            allowance.tick()
            _parent(directory,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            os.link(temporary,name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
            allowance.tick()
            _parent(directory,parent,expected)
            _named(parent,expected,temporary,fd,identity)
            _require(_identity(os.stat(name,dir_fd=parent,follow_symlinks=False))==identity)
            os.unlink(temporary,dir_fd=parent)
            placed=True
            allowance.tick()
            _parent(directory,parent,expected)
            _guard(fd,identity)
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

    @classmethod
    def create(cls,directory,*,token,initial,allowance):
        _require(type(token) is str and TOKEN.fullmatch(token))
        _require(type(initial) is dict)
        value=dict(initial,token=token,sequence=0,prior_event_sha256=None)
        value['journal_digest']=canonical_digest(value,digest_field='journal_digest')
        reference=publish_record(directory,token+'.initial.json',value,
                                 maximum=16*1024*1024,allowance=allowance)
        return cls(Path(directory),token,reference,allowance)

    def append(self,event,*,member_key,evidence):
        self.allowance.tick()
        _require(event in EVENTS and type(member_key) is str and len(member_key)<=128)
        _require(type(evidence) is dict and self.sequence<10000,'scene_retirement_journal_limit')
        value=dict(schema_version='scene_retirement_journal_event.v1',token=self.token,
                   sequence=self.sequence+1,prior_event_sha256=self.prior_ref['sha256'],
                   event=event,member_key=member_key,evidence=evidence)
        value['event_digest']=canonical_digest(value,digest_field='event_digest')
        maximum=min(65536,32*1024*1024-self.bytes)
        reference=publish_record(self.directory,self.token+'.'+str(value['sequence'])+'.json',
                                 value,maximum=maximum,allowance=self.allowance)
        self.sequence=value['sequence']
        self.bytes+=reference['size_bytes']
        self.prior_ref=reference
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
