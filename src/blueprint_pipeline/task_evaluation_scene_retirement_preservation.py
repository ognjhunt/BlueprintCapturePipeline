"""Bounded exact scene preservation
no mutation or retirement authority here."""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import tarfile
import time
from pathlib import Path

from .task_evaluation_scene_retirement_access import (
    SceneRetirementAccessError, _canonical, _close_owned, _identity, _opened,
    _open_owned, _require,
)
from .task_evaluation_scene_retirement_generations import _guard

CHUNK = 1024 * 1024
_SHA = re.compile(r'sha256:[0-9a-f]{64}')


class ActionAllowance:
    """One physical transfer/deadline allowance, never reset or refunded."""
    def __init__(self, *, expires_at, now=time.time, monotonic=time.monotonic,
                 local_bytes=8*1024**3, archive_bytes=12*1024**3,
                 remote_bytes=24*1024**3, elapsed_seconds=1800):
        _require(not hasattr(self, '_initialized'), 'scene_retirement_allowance_already_initialized')
        self._initialized = True  # Partial failures cannot reset an action origin.
        self.failure = None
        _require(type(expires_at) is int)
        for value, maximum in ((local_bytes,32*1024**3),(archive_bytes,48*1024**3),
                               (remote_bytes,96*1024**3),(elapsed_seconds,3600)):
            _require(type(value) is int and 0 <= value <= maximum)
        self.now, self.monotonic, self.expires_at = now, monotonic, expires_at
        self.limits = dict(local_bytes=local_bytes,archive_bytes=archive_bytes,remote_bytes=remote_bytes)
        self.counts = {key:0 for key in self.limits}
        self.start = self.last_tick = self._sample(monotonic)
        self.last_wall = self._sample(now)
        self.elapsed_seconds = elapsed_seconds
        self.tick()

    def _sample(self, callback):
        try:
            value = callback()
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError()
            return value
        except Exception as error:
            self.failure = 'scene_retirement_clock_unproven'
            raise SceneRetirementAccessError(self.failure) from error

    def tick(self):
        if self.failure:
            raise SceneRetirementAccessError(self.failure)
        observed, wall = self._sample(self.monotonic), self._sample(self.now)
        if observed < self.last_tick or wall < self.last_wall:
            self.failure = 'scene_retirement_clock_unproven'
        elif wall >= self.expires_at:
            self.failure = 'scene_retirement_consent_expired'
        elif observed-self.start > self.elapsed_seconds:
            self.failure = 'scene_retirement_deadline'
        else:
            self.last_tick, self.last_wall = observed, wall
            return
        raise SceneRetirementAccessError(self.failure)

    def charge(self, kind, count):
        self.tick()
        _require(type(count) is int and count >= 0)
        if self.counts[kind]+count > self.limits[kind]:
            self.failure = 'scene_retirement_byte_limit'
            raise SceneRetirementAccessError(self.failure)
        self.counts[kind] += count


def _snapshot(info):
    return (*_identity(info),info.st_size,info.st_uid,info.st_gid,info.st_mtime_ns,info.st_ctime_ns,info.st_nlink)


def _relative(value):
    _require(type(value) is str and 0 < len(value) <= 4096 and len(value.encode()) <= 4096)
    path = Path(value)
    _require(not path.is_absolute() and str(path) == value
             and all(part not in {'','.','..'} for part in path.parts)
             and '\\' not in value and '\x00' not in value)
    return value


def _scan(root, member_index, allowance, files, directories, *, depth=0):
    _require(depth <= 64)
    with _opened(root,directory=True) as (fd, info):
        identity = _identity(info)
        device = info.st_dev
        def visit(parent, expected, prefix, level):
            _require(level <= 64)
            allowance.tick()
            _guard(parent,expected)
            names = []
            with os.scandir(parent) as iterator:
                for entry in iterator:
                    allowance.tick()
                    _guard(parent,expected)
                    _require(len(files)+len(directories)+len(names) < 10000,'scene_retirement_inventory_limit')
                    names.append(entry.name)
            _guard(parent,expected)
            for name in sorted(names):
                allowance.tick()
                _guard(parent,expected)
                relative = _relative(prefix+name)
                child = os.stat(name,dir_fd=parent,follow_symlinks=False)
                _require(child.st_dev == device,'scene_retirement_mount_unproven')
                _require(0 <= child.st_uid < 2**32-1 and 0 <= child.st_gid < 2**32-1
                         and not stat.S_IMODE(child.st_mode) & 0o7000,'scene_retirement_owner_metadata_unproven')
                row = dict(member_index=member_index,relative_path=relative,mode=stat.S_IMODE(child.st_mode),
                           uid=child.st_uid,gid=child.st_gid,
                           physical_identity=list(_identity(child)),snapshot=list(_snapshot(child)))
                if stat.S_ISDIR(child.st_mode):
                    _require(len(files)+len(directories) < 10000,'scene_retirement_inventory_limit')
                    directories.append(row)
                    _guard(parent,expected)
                    retained, observed = _open_owned(name,os.O_RDONLY|os.O_DIRECTORY,dir_fd=parent)
                    owned = _identity(observed)
                    try:
                        _require(owned == _identity(child))
                        visit(retained,owned,relative+'/',level+1)
                    finally:
                        failure = _close_owned(retained,owned)
                        _require(failure is None,'scene_retirement_descriptor_cleanup_failed')
                else:
                    _require(stat.S_ISREG(child.st_mode),'scene_retirement_special_file')
                    _require(len(files)+len(directories) < 10000,'scene_retirement_inventory_limit')
                    files.append(dict(row,size_bytes=child.st_size,allocated_bytes=child.st_blocks*512,
                                      sha256=None,hardlink_group=None))
            _guard(parent,expected)
        visit(fd,identity,'',depth)
        _require(0 <= info.st_uid < 2**32-1 and 0 <= info.st_gid < 2**32-1
                 and not stat.S_IMODE(info.st_mode) & 0o7000,'scene_retirement_owner_metadata_unproven')
        return dict(path=str(root),physical_identity=list(identity),snapshot=list(_snapshot(info)),
                    mode=stat.S_IMODE(info.st_mode),uid=info.st_uid,gid=info.st_gid)


def _payload(path, row, allowance):
    with _opened(path) as (fd,info):
        _require(list(_snapshot(info)) == row['snapshot'],'scene_retirement_payload_changed')
        expected = _identity(info)
        remaining = row['size_bytes']
        while remaining:
            count = min(CHUNK,remaining)
            allowance.charge('local_bytes',count)
            _guard(fd,expected)
            data = os.read(fd,count)
            allowance.tick()
            _guard(fd,expected)
            _require(0 < len(data) <= count,'scene_retirement_payload_changed')
            remaining -= len(data)
            yield data
        allowance.tick()
        _guard(fd,expected)
        _require(list(_snapshot(os.fstat(fd))) == row['snapshot'],'scene_retirement_payload_changed')


def _archive_chunks(roots, files, directories, allowance):
    rows = [dict(row,type='directory') for row in directories]+[dict(row,type='file') for row in files]
    _require(len(rows) <= 10000,'scene_retirement_inventory_limit')
    for row in sorted(rows,key=lambda value:(value['member_index'],value['relative_path'])):
        allowance.tick()
        header = tarfile.TarInfo(str(row['member_index'])+'/'+row['relative_path'])
        header.mode = row['mode']
        header.uid, header.gid = row['uid'], row['gid']
        header.mtime = 0
        header.type = tarfile.DIRTYPE if row['type'] == 'directory' else tarfile.REGTYPE
        header.size = row.get('size_bytes',0)
        raw = header.tobuf(format=tarfile.PAX_FORMAT)
        allowance.charge('archive_bytes',len(raw))
        yield raw
        if header.isfile():
            digest = hashlib.sha256()
            for chunk in _payload(roots[row['member_index']]/row['relative_path'],row,allowance):
                digest.update(chunk)
                allowance.charge('archive_bytes',len(chunk))
                yield chunk
            _require('sha256:'+digest.hexdigest() == row['sha256'],'scene_retirement_payload_changed')
            padding = (-header.size)%512
            if padding:
                allowance.charge('archive_bytes',padding)
                yield b'\x00'*padding
    allowance.charge('archive_bytes',1024)
    yield b'\x00'*1024


def read_archive_chunks(transport,uri,allowance):
    """Installed readers precharge physical reads on this exact action origin.

    Memory/test transports retain the original charge-after-yield contract.
    This interface conveys byte accounting only; it proves no archive, owner,
    current reference, cohort or action permission.
    """
    allowance.tick()
    charged=getattr(transport,'read_archive_charged',None)
    source=iter(charged(uri,allowance) if charged is not None else transport.read_archive(uri))
    while True:
        allowance.tick()
        try:
            chunk=next(source)
        except StopIteration:
            allowance.tick()
            return
        allowance.tick()
        _require(type(chunk) is bytes and 0<len(chunk)<=CHUNK,'scene_retirement_readback_unproven')
        if charged is None:
            allowance.charge('remote_bytes',len(chunk))
        yield chunk


def preserve_members(paths, *, transport, allowance, token):
    """Stream and freshly verify exactly inventoried private archive bytes."""
    allowance.tick()
    _require(type(token) is str and re.fullmatch('[0-9a-f]{32}',token))
    _require(type(paths) in (list,tuple) and 0 < len(paths) <= 256)
    roots = [_canonical(str(path)) for path in paths]
    _require(len(set(roots)) == len(roots))
    _require(not any(a != b and a.is_relative_to(b) for a in roots for b in roots))
    files, directories, members = [], [], []
    for index, root in enumerate(roots):
        members.append(_scan(root,index,allowance,files,directories))
    inodes = {}
    for row in files:
        key = tuple(row['physical_identity'][:2])
        inodes.setdefault(key,[]).append(row)
    for rows in inodes.values():
        _require(all(row['snapshot'][-1] == len(rows) for row in rows),'scene_retirement_shared_inode')
        if len(rows) > 1:
            group = 'inode-'+str(rows[0]['physical_identity'][0])+'-'+str(rows[0]['physical_identity'][1])
            for row in rows:
                row['hardlink_group'] = group
    for row in files:
        digest = hashlib.sha256()
        for chunk in _payload(roots[row['member_index']]/row['relative_path'],row,allowance):
            digest.update(chunk)
        row['sha256'] = 'sha256:'+digest.hexdigest()
    digest, sent, complete = hashlib.sha256(), [0], [False]
    def upload():
        for chunk in _archive_chunks(roots,files,directories,allowance):
            allowance.charge('remote_bytes',len(chunk))
            digest.update(chunk)
            sent[0] += len(chunk)
            yield chunk
        complete[0] = True
    allowance.tick()
    archive = transport.put_archive(token+'.tar',upload())
    allowance.tick()
    _require(complete[0], 'scene_retirement_archive_incomplete')
    _require(type(archive) is dict and set(archive) == {'uri','sha256','size_bytes'})
    _require(type(archive['uri']) is str and len(archive['uri']) <= 4096
             and re.fullmatch(r'(?:s3|gs)://[a-z0-9][a-z0-9.-]*/[A-Za-z0-9._/-]+',archive['uri']))
    _require(type(archive['size_bytes']) is int and archive['size_bytes'] == sent[0]
             and _SHA.fullmatch(archive['sha256']) and archive['sha256'] == 'sha256:'+digest.hexdigest())
    verified, received = hashlib.sha256(), 0
    allowance.tick()
    for chunk in read_archive_chunks(transport,archive['uri'],allowance):
        allowance.tick()
        _require(type(chunk) is bytes and 0 < len(chunk) <= CHUNK,'scene_retirement_readback_unproven')
        received += len(chunk)
        _require(received <= sent[0],'scene_retirement_readback_unproven')
        verified.update(chunk)
    allowance.tick()
    _require(received == sent[0] and verified.digest() == digest.digest(),'scene_retirement_readback_unproven')
    archive = dict(archive,fresh_readback_sha256='sha256:'+verified.hexdigest(),fresh_readback_size_bytes=received)
    return dict(members=members,files=files,directories=directories,archive=archive,
                unique_allocated_bytes=sum(rows[0]['allocated_bytes'] for rows in inodes.values()))
