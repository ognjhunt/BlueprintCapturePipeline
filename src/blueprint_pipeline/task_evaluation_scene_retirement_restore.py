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
from .task_evaluation_scene_retirement_preservation import CHUNK, _payload, _scan, read_archive_chunks


class _ArchiveReader:
    def __init__(self,transport,archive,allowance):
        self.archive,self.allowance=archive,allowance
        allowance.tick()
        self.source=iter(read_archive_chunks(transport,archive['uri'],allowance))
        self.buffer=bytearray()
        self.digest=hashlib.sha256()
        self.size=0
        self.done=False

    def read(self,size):
        try:
            return self._read(size)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.source.close()

    def _read(self,size):
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


def _bounded_tar_info(allowance):
    """Reject unknown/oversized recursive extensions before stdlib allocation.

    The producer emits only regular/directory entries and one bounded local
    PAX header for path/uid/gid. Global, GNU, sparse and nested extensions never
    belong to its archive. Limits share the original action allowance.
    """
    count = 0
    depth = 0
    class BoundedTarInfo(tarfile.TarInfo):
        def _proc_pax(self, archive):
            # Never delegate unvalidated keys to stdlib: GNU sparse maps have
            # independent allocation counts even inside a tiny local header.
            allowance.tick()
            raw = archive.fileobj.read(self._block(self.size))
            allowance.tick()
            _require(len(raw) == self._block(self.size)
                     and not any(raw[self.size:]) and not archive.pax_headers,
                     'scene_retirement_readback_unproven')
            payload, offset, headers = raw[:self.size], 0, {}
            while offset < len(payload):
                allowance.tick()
                space = payload.find(b' ', offset, offset + 5)
                digits = payload[offset:space] if space >= 0 else b''
                _require(1 <= len(digits) <= 4 and digits.isdigit()
                         and not digits.startswith(b'0'), 'scene_retirement_readback_unproven')
                length = int(digits)
                _require(space + 3 < offset + length <= len(payload)
                         and payload[offset + length - 1:offset + length] == b'\n',
                         'scene_retirement_readback_unproven')
                key, separator, value = payload[space + 1:offset + length - 1].partition(b'=')
                _require(separator and key in (b'path', b'uid', b'gid') and key not in headers
                         and len(headers) < 3, 'scene_retirement_readback_unproven')
                if key == b'path':
                    _require(0 < len(value) <= 4096 and b'\0' not in value,
                             'scene_retirement_readback_unproven')
                    decoded = value.decode('utf-8', errors='strict')
                else:
                    _require(0 < len(value) <= 10 and value.isdigit() and int(value) < 2**32,
                             'scene_retirement_readback_unproven')
                    decoded = value.decode('ascii')
                headers[key] = decoded
                offset += length
            validated = {key.decode('ascii'): value for key, value in headers.items()}
            result = self.fromtarfile(archive)
            result._apply_pax_info(validated, archive.encoding, archive.errors)
            result.offset = self.offset
            return result

        def _proc_member(self, archive):
            nonlocal count, depth
            allowance.tick()
            if self.type == tarfile.XHDTYPE:
                _require(depth == 0 and type(self.size) is int and 0 < self.size <= 8192
                         and count < 10000, 'scene_retirement_readback_unproven')
                count += 1
                depth += 1
                try:
                    return super()._proc_member(archive)
                finally:
                    depth -= 1
            _require(self.type in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE),
                     'scene_retirement_readback_unproven')
            return super()._proc_member(archive)
    return BoundedTarInfo


def _consume(preserved,transport,allowance,file_sink=None):
    _require(type(preserved.get('files')) is list and type(preserved.get('directories')) is list
             and len(preserved['files'])+len(preserved['directories'])<=10000,
             'scene_retirement_inventory_limit')
    expected={str(row['member_index'])+'/'+row['relative_path']:(row,'file') for row in preserved['files']}
    expected.update({str(row['member_index'])+'/'+row['relative_path']:(row,'directory') for row in preserved['directories']})
    _require(len(expected)==len(preserved['files'])+len(preserved['directories'])<=10000)
    reader=_ArchiveReader(transport,preserved['archive'],allowance)
    seen=set()
    try:
        with tarfile.open(fileobj=reader,mode='r|',bufsize=10240,tarinfo=_bounded_tar_info(allowance)) as archive:
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
    finally:
        reader.close()


def _verify_restored(preserved,roots,directory_identities,file_identities,allowance):
    """No completion before the full CURRENT union matches its verified archive."""
    files,directories,members=[],[],[]
    for index,root in enumerate(roots):
        allowance.tick()
        members.append(_scan(root,index,allowance,files,directories))
    expected_files={(row['member_index'],row['relative_path']):row for row in preserved['files']}
    expected_directories={(row['member_index'],row['relative_path']):row for row in preserved['directories']}
    expected_directories.update({(index,''):row for index,row in enumerate(preserved['members'])})
    current_files={(row['member_index'],row['relative_path']):row for row in files}
    current_directories={(row['member_index'],row['relative_path']):row for row in directories}
    current_directories.update({(index,''):row for index,row in enumerate(members)})
    _require(current_files.keys()==expected_files.keys() and current_directories.keys()==expected_directories.keys(),
             'scene_retirement_restore_inventory_changed')
    inodes={}
    group_identities={}
    for key,row in current_files.items():
        allowance.tick()
        expected=expected_files[key]
        _require(tuple(row['physical_identity'])==file_identities[key]
                 and all(row[field]==expected[field] for field in ('mode','uid','gid','size_bytes')),
                 'scene_retirement_restore_inventory_changed')
        inode=tuple(row['physical_identity'][:2])
        group=expected['hardlink_group']
        identity=('independent',key) if group is None else ('hardlink',group)
        _require(inode not in inodes or inodes[inode][0]==identity,'scene_retirement_restore_inventory_changed')
        previous,count=inodes.get(inode,(identity,0))
        inodes[inode]=(previous,count+1)
        _require(identity not in group_identities or group_identities[identity]==inode,
                 'scene_retirement_restore_inventory_changed')
        group_identities[identity]=inode
        digest=hashlib.sha256()
        for chunk in _payload(roots[key[0]]/key[1],row,allowance):
            digest.update(chunk)
        _require('sha256:'+digest.hexdigest()==expected['sha256'],'scene_retirement_restore_payload_changed')
    alias_counts={}
    for alias in preserved.get('cache_aliases',[]):
        allowance.tick()
        key=(alias['member_index'],alias['relative_path'])
        expected_identity=file_identities[key]
        with _opened(alias['path']) as (_,info):
            _require(_identity(info)==expected_identity,'scene_retirement_restore_shared_inode')
        inode=expected_identity[:2]
        alias_counts[inode]=alias_counts.get(inode,0)+1
    for row in current_files.values():
        allowance.tick()
        inode=tuple(row['physical_identity'][:2])
        _require(row['snapshot'][-1]==inodes[inode][1]+alias_counts.get(inode,0),
                 'scene_retirement_restore_shared_inode')
    for key,row in current_directories.items():
        allowance.tick()
        expected=expected_directories[key]
        _require(tuple(row['physical_identity'])==directory_identities[key]
                 and all(row[field]==expected[field] for field in ('mode','uid','gid')),
                 'scene_retirement_restore_inventory_changed')
    # Reobserve exact names and every descriptor's metadata after byte reads.
    # This is bounded observation under EX, not a lifetime guarantee.
    after_files,after_directories,after_members=[],[],[]
    for index,root in enumerate(roots):
        after_members.append(_scan(root,index,allowance,after_files,after_directories))
    _require(after_files==files and after_directories==directories and after_members==members,
             'scene_retirement_restore_inventory_changed')


def restore_records(preserved,journal):
    """Complete native restore framing, bounded before the first destination."""
    _require(type(preserved.get('members')) is list and 0<len(preserved['members'])<=256
             and type(preserved.get('files')) is list and type(preserved.get('directories')) is list
             and len(preserved['files'])+len(preserved['directories'])<=10000,
             'scene_retirement_inventory_limit')
    identity=[2**64-1]*3
    for index,member in enumerate(preserved['members']):
        journal.allowance.tick()
        yield 'restoring',str(index),{'canonical_path':member['path']}
        yield 'restore_directory_created',str(index),dict(canonical_path=member['path'],relative_path='',
            restore_identity=identity,parent_identity=identity,restore_uid=2**32-1,restore_gid=2**32-1)
    for row in preserved['directories']:
        journal.allowance.tick()
        yield 'restore_directory_created',str(row['member_index']),dict(relative_path=row['relative_path'],
            restore_identity=identity,parent_identity=identity,restore_uid=2**32-1,restore_gid=2**32-1)
    for row in preserved['files']:
        journal.allowance.tick()
        yield 'restore_file_created',str(row['member_index']),dict(relative_path=row['relative_path'],restore_identity=identity,sha256=row['sha256'])
    for index,member in enumerate(preserved['members']):
        journal.allowance.tick()
        yield 'member_restored',str(index),dict(canonical_path=member['path'],outcome='restored',restore_identity=identity)


def _created_destinations(preserved,roots,journal):
    """Only the exact journal-created inode can become a resumed destination."""
    expected_files={(row['member_index'],row['relative_path']):row for row in preserved['files']}
    expected_dirs={(row['member_index'],row['relative_path']):row for row in preserved['directories']}
    expected_dirs.update({(index,''):row for index,row in enumerate(preserved['members'])})
    directories,files,directory_proofs={},{},{}
    for event in journal.events:
        journal.allowance.tick()
        kind=event['event']
        if kind not in {'restore_directory_created','restore_file_created'}:
            continue
        index=int(event['member_key'])
        proof=event['evidence']
        key=index,proof.get('relative_path')
        expected=expected_dirs if kind=='restore_directory_created' else expected_files
        target=directories if kind=='restore_directory_created' else files
        _require(key in expected and key not in target,'scene_retirement_restore_journal_unproven')
        identity=proof.get('restore_identity')
        _require(type(identity) is list and len(identity)==3
                 and all(type(value) is int and value>=0 for value in identity),
                 'scene_retirement_restore_journal_unproven')
        if key[1]=='':
            _require(proof.get('canonical_path')==str(roots[index]),'scene_retirement_restore_journal_unproven')
        if kind=='restore_file_created':
            _require(proof.get('sha256')==expected[key]['sha256'],'scene_retirement_restore_journal_unproven')
        else:
            _require(all(type(proof.get(field)) is int and 0<=proof[field]<=2**32-1
                         for field in ('restore_uid','restore_gid'))
                     and type(proof.get('parent_identity')) is list and len(proof['parent_identity'])==3,
                     'scene_retirement_restore_journal_unproven')
            directory_proofs[key]=proof
        target[key]=tuple(identity)
    inode_counts={}
    found_files=[]
    for index,root in enumerate(roots):
        if (index,'') not in directories:
            continue
        actual_files,actual_dirs=[],[]
        member=_scan(root,index,journal.allowance,actual_files,actual_dirs)
        actual_dirs.append(dict(member,relative_path=''))
        _require({(index,row['relative_path']) for row in actual_files}=={key for key in files if key[0]==index}
                 and {(index,row['relative_path']) for row in actual_dirs}=={key for key in directories if key[0]==index},
                 'scene_retirement_restore_inventory_changed')
        for row in actual_dirs:
            key=index,row['relative_path']
            old=directories[key]
            original=expected_dirs[key]
            proof=directory_proofs[key]
            with _opened((root/row['relative_path']).parent,directory=True) as (_,parent):
                _require(_identity(parent)[:2]==tuple(proof['parent_identity'][:2]),
                         'scene_retirement_restore_inventory_changed')
            _require(tuple(row['physical_identity'][:2])==old[:2]
                     and ((row['uid']==proof['restore_uid'] and row['gid']==proof['restore_gid'] and row['mode']==0o700)
                          or all(row[field]==original[field] for field in ('mode','uid','gid'))),
                     'scene_retirement_restore_inventory_changed')
            directories[key]=tuple(row['physical_identity'])
        for row in actual_files:
            key=index,row['relative_path']
            original=expected_files[key]
            _require(tuple(row['physical_identity'])==files[key]
                     and all(row[field]==original[field] for field in ('mode','uid','gid','size_bytes')),
                     'scene_retirement_restore_inventory_changed')
            digest=hashlib.sha256()
            for chunk in _payload(root/row['relative_path'],row,journal.allowance):
                digest.update(chunk)
            _require('sha256:'+digest.hexdigest()==original['sha256'],'scene_retirement_restore_payload_changed')
            inode=tuple(row['physical_identity'][:2])
            inode_counts[inode]=inode_counts.get(inode,0)+1
            found_files.append(row)
    for row in found_files:
        _require(row['snapshot'][-1]==inode_counts[tuple(row['physical_identity'][:2])],
                 'scene_retirement_restore_shared_inode')
    return directories,files


def restore_preserved_members(preserved,*,transport,journal):
    """Internal: verify full union FIRST, claim absent roots and publish no-replace."""
    allowance=journal.allowance
    journal.preflight(restore_records(preserved,journal))
    _consume(preserved,transport,allowance)  # No local destination exists or is touched here.
    roots=[Path(row['path']) for row in preserved['members']]
    from .task_evaluation_scene_retirement_cache import require_cache_restore_destinations,restore_preserved_cache_aliases
    require_cache_restore_destinations(preserved,journal)
    directories,file_identities=_created_destinations(preserved,roots,journal)
    for index,member in enumerate(preserved['members']):
        root=roots[index]
        if (index,'') in directories:
            continue
        journal.append('restoring',member_key=str(index),evidence={'canonical_path':str(root)})
        with _opened(root.parent,directory=True) as (parent,info):
            expected=_identity(info)
            allowance.tick()
            _current_parent(root.parent,parent,expected)
            allowance.tick()
            os.mkdir(root.name,0o700,dir_fd=parent)
            allowance.tick()
            _current_parent(root.parent,parent,expected)
            with _opened(root,directory=True) as (_,born):
                _require(born.st_uid==os.geteuid() and (born.st_mode & 0o777)==0o700)
                directories[(index,'')]=_identity(born)
            allowance.tick()
            _current_parent(root.parent,parent,expected)
            allowance.tick()
            os.fsync(parent)
            journal.append('restore_directory_created',member_key=str(index),evidence={
                'canonical_path':str(root),'relative_path':'','restore_identity':list(directories[(index,'')]),
                'parent_identity':list(expected),'restore_uid':born.st_uid,'restore_gid':born.st_gid})
    for row in sorted(preserved['directories'],key=lambda item:len(Path(item['relative_path']).parts)):
        index=row['member_index']
        relative=Path(row['relative_path'])
        if (index,str(relative)) in directories:
            continue
        parent_relative='' if str(relative.parent)=='.' else str(relative.parent)
        parent_path=roots[index]/relative.parent
        expected=directories[(index,parent_relative)]
        with _opened(parent_path,directory=True) as (fd,info):
            _require(_identity(info)==expected)
            allowance.tick()
            _current_parent(parent_path,fd,expected)
            allowance.tick()
            os.mkdir(relative.name,0o700,dir_fd=fd)
            with _opened(roots[index]/relative,directory=True) as (_,born):
                _require(born.st_uid==os.geteuid() and (born.st_mode & 0o777)==0o700)
                directories[(index,str(relative))]=_identity(born)
            allowance.tick()
            _current_parent(parent_path,fd,expected)
            allowance.tick()
            os.fsync(fd)
        journal.append('restore_directory_created',member_key=str(index),evidence={
            'relative_path':str(relative),'restore_identity':list(directories[(index,str(relative))]),
            'parent_identity':list(expected),'restore_uid':born.st_uid,'restore_gid':born.st_gid})
    groups={}
    for row in preserved['files']:
        key=row['member_index'],row['relative_path']
        if key not in file_identities or row['hardlink_group'] is None:
            continue
        group=row['hardlink_group']
        if group in groups:
            _require(groups[group][1][:2]==file_identities[key][:2],'scene_retirement_restore_inventory_changed')
        else:
            groups[group]=(roots[key[0]]/key[1],file_identities[key])
    @contextmanager
    def file_sink(row):
        index=row['member_index']
        relative=Path(row['relative_path'])
        parent_relative='' if str(relative.parent)=='.' else str(relative.parent)
        parent_path=roots[index]/relative.parent
        expected=directories[(index,parent_relative)]
        group=row['hardlink_group']
        if (index,str(relative)) in file_identities:
            yield lambda chunk:None  # Entire remote row remains independently verified.
            with _opened(roots[index]/relative) as (_,info):
                _require(_identity(info)==file_identities[(index,str(relative))],
                         'scene_retirement_restore_inventory_changed')
            return
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
                    allowance.tick()
                    os.link(original.name,relative.name,src_dir_fd=source,dst_dir_fd=parent,follow_symlinks=False)
                created=identity
            else:
                temporary='.'+secrets.token_hex(16)+'.restore'
                allowance.tick()
                _current_parent(parent_path,parent,expected)
                fd,identity=_new_file(parent,temporary,parent_identity=expected,action_guard=allowance.tick)
                placed=False
                try:
                    def write(chunk):
                        view=memoryview(chunk)
                        while view:
                            allowance.tick()
                            _current_parent(parent_path,parent,expected)
                            _named(parent,expected,temporary,fd,identity)
                            allowance.tick()
                            count=os.write(fd,view)
                            _require(count>0)
                            view=view[count:]
                    yield write
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    allowance.tick()
                    os.fchown(fd,row['uid'],row['gid'])
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    allowance.tick()
                    os.fchmod(fd,row['mode'])
                    # chmod changes the owned token's mode, with the SAME inode.
                    observed=os.fstat(fd)
                    _require(_identity(observed)[:2]==identity[:2])
                    identity=_identity(observed)
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    allowance.tick()
                    os.fsync(fd)
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    allowance.tick()
                    os.link(temporary,relative.name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
                    allowance.tick()
                    _current_parent(parent_path,parent,expected)
                    _named(parent,expected,temporary,fd,identity)
                    allowance.tick()
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
            allowance.tick()
            os.fsync(parent)
            file_identities[(index,str(relative))]=created
            journal.append('restore_file_created',member_key=str(index),evidence={
                'relative_path':str(relative),'restore_identity':list(created),'sha256':row['sha256']})
    _consume(preserved,transport,allowance,file_sink=file_sink)
    restore_preserved_cache_aliases(preserved,roots,file_identities,journal)
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
                allowance.tick()
                os.fchown(fd,original['uid'],original['gid'])
                allowance.tick()
                _current_parent(path.parent,parent,parent_identity)
                _named(parent,parent_identity,path.name,fd,expected)
                allowance.tick()
                os.fchmod(fd,original['mode'])
                expected=(*expected[:2],(expected[2] & ~0o7777)|original['mode'])
                current=os.fstat(fd)
                _require(_identity(current)==expected and current.st_uid==original['uid'] and current.st_gid==original['gid'])
                directories[(index,relative)]=expected
                allowance.tick()
                _current_parent(path.parent,parent,parent_identity)
                _named(parent,parent_identity,path.name,fd,expected)
                allowance.tick()
                os.fsync(fd)
            finally:
                failure=_close_owned(fd,expected)
                if failure:
                    incoming=sys.exc_info()[1]
                    if incoming is not None:
                        incoming.add_note('scene_retirement_descriptor_cleanup_failed')
                    else:
                        raise ValueError('scene_retirement_descriptor_cleanup_failed')
    _verify_restored(preserved,roots,directories,file_identities,allowance)
    for index,root in enumerate(roots):
        outcome=dict(canonical_path=str(root),outcome='restored',restore_identity=list(directories[(index,'')]))
        existing=[event for event in journal.events if event['event']=='member_restored' and event['member_key']==str(index)]
        _require(len(existing)<=1 and (not existing or existing[0]['evidence']==outcome),
                 'scene_retirement_restore_journal_unproven')
        if not existing:
            journal.append('member_restored',member_key=str(index),evidence=outcome)
        outcomes.append(outcome)
    return outcomes
