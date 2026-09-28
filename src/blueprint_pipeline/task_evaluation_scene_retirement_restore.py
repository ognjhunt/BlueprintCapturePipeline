"""Engine-internal verified streaming restore; existing destinations are conflicts."""
from __future__ import annotations

import hashlib
import os
import secrets
import sys
import tarfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

from .task_evaluation_scene_retirement_access import _close_owned, _identity, _opened, _open_owned, _require
from .task_evaluation_scene_retirement_generations import _named, _new_file
from .task_evaluation_scene_retirement_mutation import _current_parent
from .task_evaluation_scene_retirement_preservation import CHUNK


class _ArchiveReader:
    def __init__(self,transport,archive,allowance):
        self.archive,self.allowance=archive,allowance
        allowance.tick()
        self.source=iter(transport.read_archive(archive['uri']))
        self.buffer=bytearray()
        self.digest=hashlib.sha256()
        self.size=0
        self.done=False

    def read(self,size):
        _require(type(size) is int and 0<=size<=CHUNK,'scene_retirement_readback_unproven')
        while len(self.buffer)<size and not self.done:
            self.allowance.tick()
            try:
                chunk=next(self.source)
            except StopIteration:
                self.done=True
                break
            self.allowance.tick()
            _require(type(chunk) is bytes and 0<len(chunk)<=CHUNK,'scene_retirement_readback_unproven')
            self.allowance.charge('remote_bytes',len(chunk))
            self.size+=len(chunk)
            _require(self.size<=self.archive['size_bytes'],'scene_retirement_readback_unproven')
            self.digest.update(chunk)
            self.buffer.extend(chunk)
        result=bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def verify(self):
        while self.read(CHUNK):
            pass
        self.allowance.tick()
        _require(self.size==self.archive['size_bytes']
                 and 'sha256:'+self.digest.hexdigest()==self.archive['sha256'],'scene_retirement_readback_unproven')


def _consume(preserved,transport,allowance,file_sink=None):
    expected={str(row['member_index'])+'/'+row['relative_path']:(row,'file') for row in preserved['files']}
    expected.update({str(row['member_index'])+'/'+row['relative_path']:(row,'directory') for row in preserved['directories']})
    _require(len(expected)==len(preserved['files'])+len(preserved['directories'])<=10000)
    reader=_ArchiveReader(transport,preserved['archive'],allowance)
    seen=set()
    try:
        with tarfile.open(fileobj=reader,mode='r|',bufsize=10240) as archive:
            for item in archive:
                allowance.tick()
                _require(len(seen)<10000 and item.name in expected and item.name not in seen,
                         'scene_retirement_readback_unproven')
                row,kind=expected[item.name]
                _require(item.mode==row['mode'] and item.uid==row['uid'] and item.gid==row['gid']
                         and set(item.pax_headers)<={'path','uid','gid'},'scene_retirement_readback_unproven')
                seen.add(item.name)
                if kind=='directory':
                    _require(item.isdir() and item.size==0,'scene_retirement_readback_unproven')
                    continue
                _require(item.isfile() and item.size==row['size_bytes'],'scene_retirement_readback_unproven')
                remaining=item.size
                digest=hashlib.sha256()
                stream=archive.extractfile(item)
                _require(stream is not None,'scene_retirement_readback_unproven')
                with file_sink(row) if file_sink is not None else nullcontext(None) as write:
                    while remaining:
                        allowance.tick()
                        chunk=stream.read(min(CHUNK,remaining))
                        allowance.tick()
                        _require(type(chunk) is bytes and 0<len(chunk)<=remaining,'scene_retirement_readback_unproven')
                        digest.update(chunk)
                        remaining-=len(chunk)
                        if write is not None:
                            write(chunk)
                    _require('sha256:'+digest.hexdigest()==row['sha256'],'scene_retirement_readback_unproven')
        reader.verify()
        _require(seen==expected.keys(),'scene_retirement_readback_unproven')
    except (tarfile.TarError,EOFError,UnicodeError) as error:
        raise ValueError('scene_retirement_readback_unproven') from error


def restore_preserved_members(preserved,*,transport,journal):
    """Internal: verify full union FIRST, claim absent roots and publish no-replace."""
    allowance=journal.allowance
    _consume(preserved,transport,allowance)  # No local destination exists or is touched here.
    roots=[Path(row['path']) for row in preserved['members']]
    directories={}
    for index,member in enumerate(preserved['members']):
        root=roots[index]
        journal.append('restoring',member_key=str(index),evidence={'canonical_path':str(root)})
        with _opened(root.parent,directory=True) as (parent,info):
            expected=_identity(info)
            allowance.tick()
            _current_parent(root.parent,parent,expected)
            os.mkdir(root.name,0o700,dir_fd=parent)
            allowance.tick()
            _current_parent(root.parent,parent,expected)
            with _opened(root,directory=True) as (_,born):
                _require(born.st_uid==os.geteuid() and (born.st_mode & 0o777)==0o700)
                directories[(index,'')]=_identity(born)
            allowance.tick()
            _current_parent(root.parent,parent,expected)
            os.fsync(parent)
            journal.append('restore_directory_created',member_key=str(index),evidence={
                'canonical_path':str(root),'relative_path':'','restore_identity':list(directories[(index,'')])})
    for row in sorted(preserved['directories'],key=lambda item:len(Path(item['relative_path']).parts)):
        index=row['member_index']
        relative=Path(row['relative_path'])
        parent_relative='' if str(relative.parent)=='.' else str(relative.parent)
        parent_path=roots[index]/relative.parent
        expected=directories[(index,parent_relative)]
        with _opened(parent_path,directory=True) as (fd,info):
            _require(_identity(info)==expected)
            allowance.tick()
            _current_parent(parent_path,fd,expected)
            os.mkdir(relative.name,0o700,dir_fd=fd)
            with _opened(roots[index]/relative,directory=True) as (_,born):
                _require(born.st_uid==os.geteuid() and (born.st_mode & 0o777)==0o700)
                directories[(index,str(relative))]=_identity(born)
            allowance.tick()
            _current_parent(parent_path,fd,expected)
            os.fsync(fd)
        journal.append('restore_directory_created',member_key=str(index),evidence={
            'relative_path':str(relative),'restore_identity':list(directories[(index,str(relative))])})
    groups={}
    @contextmanager
    def file_sink(row):
        index=row['member_index']
        relative=Path(row['relative_path'])
        parent_relative='' if str(relative.parent)=='.' else str(relative.parent)
        parent_path=roots[index]/relative.parent
        expected=directories[(index,parent_relative)]
        group=row['hardlink_group']
        with _opened(parent_path,directory=True) as (parent,info):
            _require(_identity(info)==expected)
            if group is not None and group in groups:
                yield lambda chunk:None  # Bytes still verified against this exact row.
                original,identity=groups[group]
                with _opened(original.parent,directory=True) as (source,source_info):
                    source_identity=_identity(source_info)
                    allowance.tick()
                    _current_parent(original.parent,source,source_identity)
                    _current_parent(parent_path,parent,expected)
                    _require(_identity(os.stat(original.name,dir_fd=source,follow_symlinks=False))==identity)
                    os.link(original.name,relative.name,src_dir_fd=source,dst_dir_fd=parent,follow_symlinks=False)
                created=identity
            else:
                temporary='.'+secrets.token_hex(16)+'.restore'
                allowance.tick()
                _current_parent(parent_path,parent,expected)
                fd,identity=_new_file(parent,temporary,parent_identity=expected)
                placed=False
                try:
                    def write(chunk):
                        view=memoryview(chunk)
                        while view:
                            allowance.tick()
                            _current_parent(parent_path,parent,expected)
                            _named(parent,expected,temporary,fd,identity)
                            count=os.write(fd,view)
                            _require(count>0)
                            view=view[count:]
                    yield write
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    os.fchown(fd,row['uid'],row['gid'])
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    os.fchmod(fd,row['mode'])
                    # chmod changes the owned token's mode, with the SAME inode.
                    observed=os.fstat(fd)
                    _require(_identity(observed)[:2]==identity[:2])
                    identity=_identity(observed)
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    os.fsync(fd)
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    os.link(temporary,relative.name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    os.unlink(temporary,dir_fd=parent)
                    placed=True
                    created=identity
                finally:
                    incoming=sys.exc_info()[1]
                    if not placed:
                        try:
                            _current_parent(parent_path,parent,expected)
                            _named(parent,expected,temporary,fd,identity)
                            os.unlink(temporary,dir_fd=parent)
                        except (OSError,ValueError):
                            if incoming is not None:
                                incoming.add_note('scene_retirement_restore_cleanup_unproven')
                    failure=_close_owned(fd,identity)
                    if failure:
                        if incoming is None:
                            raise ValueError('scene_retirement_descriptor_cleanup_failed')
                        incoming.add_note('scene_retirement_descriptor_cleanup_failed')
                if group is not None:
                    groups[group]=(roots[index]/relative,created)
            allowance.tick()
            _current_parent(parent_path,parent,expected)
            os.fsync(parent)
            journal.append('restore_file_created',member_key=str(index),evidence={
                'relative_path':str(relative),'restore_identity':list(created),'sha256':row['sha256']})
    _consume(preserved,transport,allowance,file_sink=file_sink)
    outcomes=[]
    metadata={(row['member_index'],row['relative_path']):row for row in preserved['directories']}
    metadata.update({(index,''):row for index,row in enumerate(preserved['members'])})
    for (index,relative),identity in sorted(directories.items(),key=lambda pair:len(Path(pair[0][1]).parts),reverse=True):
        path=roots[index]/relative
        original=metadata[(index,relative)]
        # The ancestor context never owns the leaf whose mode this operation
        # changes; its cleanup keeps the independently proved original identity.
        with _opened(path.parent,directory=True) as (parent,parent_info):
            parent_identity=_identity(parent_info)
            _current_parent(path.parent,parent,parent_identity)
            fd,info=_open_owned(path.name,os.O_RDONLY|os.O_DIRECTORY,dir_fd=parent)
            expected=_identity(info)
            try:
                _require(expected==identity)
                allowance.tick()
                _current_parent(path.parent,parent,parent_identity)
                _named(parent,parent_identity,path.name,fd,expected)
                os.fchown(fd,original['uid'],original['gid'])
                allowance.tick()
                _current_parent(path.parent,parent,parent_identity)
                _named(parent,parent_identity,path.name,fd,expected)
                os.fchmod(fd,original['mode'])
                expected=(*expected[:2],(expected[2] & ~0o7777)|original['mode'])
                current=os.fstat(fd)
                _require(_identity(current)==expected and current.st_uid==original['uid'] and current.st_gid==original['gid'])
                directories[(index,relative)]=expected
                allowance.tick()
                _current_parent(path.parent,parent,parent_identity)
                _named(parent,parent_identity,path.name,fd,expected)
                os.fsync(fd)
            finally:
                failure=_close_owned(fd,expected)
                if failure:
                    incoming=sys.exc_info()[1]
                    if incoming is not None:
                        incoming.add_note('scene_retirement_descriptor_cleanup_failed')
                    else:
                        raise ValueError('scene_retirement_descriptor_cleanup_failed')
    for index,root in enumerate(roots):
        outcome=dict(canonical_path=str(root),outcome='restored',restore_identity=list(directories[(index,'')]))
        journal.append('member_restored',member_key=str(index),evidence=outcome)
        outcomes.append(outcome)
    return outcomes
