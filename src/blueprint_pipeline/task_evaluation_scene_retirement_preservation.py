"""Bounded exact scene preservation
no mutation or retirement authority here."""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import sys
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
        self.started_wall = self.last_wall = self._sample(now)
        self._resume_origin = None
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
        elif observed-self.start > self.elapsed_seconds or (self._resume_origin is not None and (
            observed-self._resume_origin['start_monotonic']>self.elapsed_seconds
            or wall-self._resume_origin['started_wall']>self.elapsed_seconds)):
            self.failure = 'scene_retirement_deadline'
        else:
            self.last_tick, self.last_wall = observed, wall
            return
        raise SceneRetirementAccessError(self.failure)

    def checkpoint(self):
        """Pure current counters; only a protected selected journal binds them."""
        self.tick()
        origin=self._resume_origin or dict(start_monotonic=self.start,started_wall=self.started_wall)
        return dict(schema_version='scene_retirement_action_allowance.v1',
            start_monotonic=origin['start_monotonic'],started_wall=origin['started_wall'],
            last_monotonic=self.last_tick,last_wall=self.last_wall,expires_at=self.expires_at,
            elapsed_seconds=self.elapsed_seconds,limits=dict(self.limits),counts=dict(self.counts))

    def bind_resume(self,checkpoint):
        """Narrow a fresh invocation to its exact durable original work prefix.

        No native limit, expiry, current clock origin or consumed count resets.
        Unknown partial transfer is conservatively reserved before physical work
        in the selected journal, so a crash cannot refund that reservation.
        """
        self.tick()
        _require(not hasattr(self,'_resume_bound'),'scene_retirement_resume_allowance_unproven')
        self._resume_bound=True
        try:
            keys={'schema_version','start_monotonic','started_wall','last_monotonic','last_wall',
                  'expires_at','elapsed_seconds','limits','counts'}
            _require(type(checkpoint) is dict and set(checkpoint)==keys
                and checkpoint['schema_version']=='scene_retirement_action_allowance.v1',
                'scene_retirement_resume_allowance_unproven')
            for key in ('start_monotonic','started_wall','last_monotonic','last_wall'):
                _require(type(checkpoint[key]) in (int,float) and math.isfinite(checkpoint[key]),
                         'scene_retirement_resume_allowance_unproven')
            _require(type(checkpoint['expires_at']) is int and checkpoint['expires_at']==self.expires_at
                and type(checkpoint['elapsed_seconds']) is int and checkpoint['elapsed_seconds']==self.elapsed_seconds
                and type(checkpoint['limits']) is dict and checkpoint['limits']==self.limits
                and all(type(value) is int for value in checkpoint['limits'].values())
                and type(checkpoint['counts']) is dict and set(checkpoint['counts'])==set(self.counts)
                and not any(self.counts.values()),'scene_retirement_resume_allowance_unproven')
            _require(checkpoint['start_monotonic']<=checkpoint['last_monotonic']<=self.last_tick
                and checkpoint['started_wall']<=checkpoint['last_wall']<=self.last_wall,
                'scene_retirement_clock_unproven')
            self._resume_origin=dict(start_monotonic=checkpoint['start_monotonic'],started_wall=checkpoint['started_wall'])
            self.tick()
            for key,count in checkpoint['counts'].items():
                self.charge(key,count)
        except SceneRetirementAccessError as error:
            self.failure=str(error)
            raise

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


def archive_size(files,directories,allowance):
    """Exact deterministic tar framing before any payload is opened."""
    total=1024
    for kind,rows in (('directory',directories),('file',files)):
        for row in rows:
            allowance.tick()
            header=tarfile.TarInfo(str(row['member_index'])+'/'+row['relative_path'])
            header.mode,header.uid,header.gid=row['mode'],row['uid'],row['gid']
            header.mtime=0
            header.type=tarfile.DIRTYPE if kind=='directory' else tarfile.REGTYPE
            header.size=row.get('size_bytes',0)
            total+=len(header.tobuf(format=tarfile.PAX_FORMAT))
            if kind=='file':
                total+=header.size+(-header.size)%512
            _require(total<=48*1024**3,'scene_retirement_byte_limit')
    return total


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
    try:
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
    finally:
        close=getattr(source,'close',None)
        if close is not None:
            incoming=sys.exc_info()[1]
            try:
                close()  # Known remote cleanup remains independent of expiry.
            except Exception:
                if incoming is None or isinstance(incoming,GeneratorExit):
                    raise SceneRetirementAccessError('scene_retirement_remote_cleanup_unproven') from None
                incoming.add_note('scene_retirement_remote_cleanup_unproven')


def _inventory_members(paths,allowance,*,cache_aliases=None):
    """Bound exact names/inodes before payload access; no action authority."""
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
    aliases=[]
    alias_counts={}
    if cache_aliases is not None:
        _require(type(cache_aliases) in (list,tuple) and len(cache_aliases)<=256,
                 'scene_retirement_inventory_limit')
        seen=set()
        for selected in cache_aliases:
            allowance.tick()
            _require(type(selected) is dict and set(selected)=={'canonical_path','digest','size_bytes'},
                     'scene_retirement_cache_alias_unproven')
            path=_canonical(selected['canonical_path'])
            _require(str(path) not in seen and not any(path.is_relative_to(root) for root in roots)
                     and type(selected['digest']) is str and _SHA.fullmatch(selected['digest'])
                     and path.name==selected['digest'][7:] and type(selected['size_bytes']) is int
                     and selected['size_bytes']>=0,'scene_retirement_cache_alias_unproven')
            _require(len(files)+len(directories)+len(aliases)<10000,'scene_retirement_inventory_limit')
            seen.add(str(path))
            with _opened(path.parent,directory=True) as (_,parent_info),_opened(path) as (_,info):
                identity=_identity(info)
                key=identity[:2]
                matches=inodes.get(key,[])
                _require(matches and info.st_size==selected['size_bytes']
                         and all(tuple(row['physical_identity'])==identity and row['size_bytes']==info.st_size
                                 for row in matches),'scene_retirement_cache_alias_unproven')
                aliases.append(dict(path=str(path),digest=selected['digest'],size_bytes=selected['size_bytes'],
                    physical_identity=list(identity),snapshot=list(_snapshot(info)),
                    member_index=matches[0]['member_index'],relative_path=matches[0]['relative_path'],
                    mode=stat.S_IMODE(info.st_mode),uid=info.st_uid,gid=info.st_gid,
                    parent_identity=list(_identity(parent_info))))
                alias_counts[key]=alias_counts.get(key,0)+1
    for key,rows in inodes.items():
        expected=len(rows)+alias_counts.get(key,0)
        _require(all(row['snapshot'][-1] == expected for row in rows),'scene_retirement_shared_inode')
        if expected > 1:
            group = 'inode-'+str(rows[0]['physical_identity'][0])+'-'+str(rows[0]['physical_identity'][1])
            for row in rows:
                row['hardlink_group'] = group
    return dict(members=members,files=files,directories=directories,cache_aliases=aliases,
                unique_allocated_bytes=sum(rows[0]['allocated_bytes'] for rows in inodes.values()))


def preserve_members(paths, *, transport, allowance, token, before_payload=None, before_upload=None, archive_name=None, cache_aliases=None):
    """Stream and freshly verify exactly inventoried private archive bytes."""
    allowance.tick()
    _require(type(token) is str and re.fullmatch('[0-9a-f]{32}',token))
    name=token+'.tar' if archive_name is None else archive_name
    _require(type(name) is str and re.fullmatch(re.escape(token)+r'(?:\.[1-9][0-9]{0,2})?\.tar',name)
             and (name==token+'.tar' or int(name.split('.')[1])<=256),'scene_retirement_archive_name_unproven')
    inventory=_inventory_members(paths,allowance,cache_aliases=cache_aliases)
    members,files,directories,aliases=(inventory[key] for key in ('members','files','directories','cache_aliases'))
    unique_allocated_bytes=inventory['unique_allocated_bytes']
    roots=[Path(member['path']) for member in members]
    if before_payload is not None:
        before_payload(files,directories,archive_size(files,directories,allowance))
        allowance.tick()
    for row in files:
        digest = hashlib.sha256()
        for chunk in _payload(roots[row['member_index']]/row['relative_path'],row,allowance):
            digest.update(chunk)
        row['sha256'] = 'sha256:'+digest.hexdigest()
    for alias in aliases:
        allowance.tick()
        selected=next(row for row in files if row['member_index']==alias['member_index']
                      and row['relative_path']==alias['relative_path'])
        _require(selected['sha256']==alias['digest'],'scene_retirement_cache_alias_unproven')
        with _opened(alias['path']) as (_,info):
            _require(list(_snapshot(info))==alias['snapshot'],'scene_retirement_cache_alias_changed')
    if before_upload is not None:
        inventory=dict(members=members,files=files,directories=directories)
        if cache_aliases is not None:
            inventory['cache_aliases']=aliases
        before_upload(inventory)
        allowance.tick()
    digest, sent, complete = hashlib.sha256(), [0], [False]
    def upload():
        for chunk in _archive_chunks(roots,files,directories,allowance):
            allowance.charge('remote_bytes',len(chunk))
            digest.update(chunk)
            sent[0] += len(chunk)
            yield chunk
        complete[0] = True
    allowance.tick()
    archive = transport.put_archive(name,upload())
    allowance.tick()
    _require(complete[0], 'scene_retirement_archive_incomplete')
    _require(type(archive) is dict and set(archive) == {'uri','sha256','size_bytes'})
    _require(type(archive['uri']) is str and len(archive['uri']) <= 4096
             and re.fullmatch(r'(?:s3|gs)://[a-z0-9][a-z0-9.-]*/[A-Za-z0-9._/-]+',archive['uri']))
    _require(type(archive['size_bytes']) is int and archive['size_bytes'] == sent[0]
             and _SHA.fullmatch(archive['sha256']) and archive['sha256'] == 'sha256:'+digest.hexdigest())
    verified, received = hashlib.sha256(), 0
    allowance.tick()
    stream=read_archive_chunks(transport,archive['uri'],allowance)
    try:
        for chunk in stream:
            allowance.tick()
            _require(type(chunk) is bytes and 0 < len(chunk) <= CHUNK,'scene_retirement_readback_unproven')
            received += len(chunk)
            _require(received <= sent[0],'scene_retirement_readback_unproven')
            verified.update(chunk)
    finally:
        stream.close()
    allowance.tick()
    _require(received == sent[0] and verified.digest() == digest.digest(),'scene_retirement_readback_unproven')
    archive = dict(archive,fresh_readback_sha256='sha256:'+verified.hexdigest(),fresh_readback_size_bytes=received)
    result=dict(members=members,files=files,directories=directories,archive=archive,
                unique_allocated_bytes=unique_allocated_bytes)
    if cache_aliases is not None:
        result['cache_aliases']=aliases
    return result
