"""Engine-internal exact member mutation, after consent, fencing and preservation.

This module supplies no public admission route. The complete engine retains the
exclusive coordinator and all generation/reference authority across these calls.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import control_plane_lane_scratch as primitive
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_retirement_access import _identity, _opened, _require
from .task_evaluation_scene_retirement_generations import _guard
from .task_evaluation_scene_retirement_preservation import _payload, _scan, _snapshot


def inventory_digest(preserved,index):
    files=[dict(relative_path=row['relative_path'],mode=row['mode'],size_bytes=row['size_bytes'],
                sha256=row['sha256'],hardlink_group=row['hardlink_group'])
           for row in preserved['files'] if row['member_index']==index]
    directories=[dict(relative_path=row['relative_path'],mode=row['mode'])
                 for row in preserved['directories'] if row['member_index']==index]
    return canonical_digest({'files':files,'directories':directories})


def _verify_current(preserved,index,allowance):
    member=preserved['members'][index]
    files,directories=[],[]
    current=_scan(Path(member['path']),index,allowance,files,directories)
    _require(current['snapshot']==member['snapshot'],'scene_retirement_member_changed')
    expected_files={row['relative_path']:row for row in preserved['files'] if row['member_index']==index}
    expected_directories={row['relative_path']:row for row in preserved['directories'] if row['member_index']==index}
    _require({row['relative_path'] for row in files}==expected_files.keys()
             and {row['relative_path'] for row in directories}==expected_directories.keys(),
             'scene_retirement_member_changed')
    for row in directories:
        _require(row['snapshot']==expected_directories[row['relative_path']]['snapshot'],'scene_retirement_member_changed')
    for row in files:
        original=expected_files[row['relative_path']]
        _require(row['snapshot']==original['snapshot'],'scene_retirement_member_changed')
        digest=hashlib.sha256()
        for chunk in _payload(Path(member['path'])/row['relative_path'],original,allowance):
            digest.update(chunk)
        _require('sha256:'+digest.hexdigest()==original['sha256'],'scene_retirement_payload_changed')


def _current_parent(path,fd,expected):
    _guard(fd,expected)
    with _opened(path,directory=True) as (_,info):
        _require(_identity(info)==expected,'scene_retirement_member_changed')
    _guard(fd,expected)


def detach_and_remove(preserved,*,member_index,generation_id,journal,removed_inodes=None):
    """Internal: durable pre-detach proof, atomic no-replace and exact leaf union."""
    allowance=journal.allowance
    allowance.tick()
    _require(type(member_index) is int and 0<=member_index<len(preserved['members']))
    member=preserved['members'][member_index]
    source=Path(member['path'])
    destination=source.parent/('.scene-retirement-'+journal.token+'-'+str(member_index))
    _verify_current(preserved,member_index,allowance)
    with _opened(source.parent,directory=True) as (parent,parent_info):
        parent_identity=_identity(parent_info)
        evidence=dict(canonical_path=str(source),detached_path=str(destination),generation_id=generation_id,
                      pre_identity=member['physical_identity'],parent_identity=list(parent_identity),
                      inventory_sha256=inventory_digest(preserved,member_index))
        planned=journal.append('detach_planned',member_key=str(member_index),evidence=evidence)
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        _require(list(_identity(os.stat(source.name,dir_fd=parent,follow_symlinks=False)))==member['physical_identity'])
        # This is the existing supported Linux/macOS atomic NO-REPLACE primitive,
        # never a check followed by an overwriting rename or whole-tree rmtree.
        primitive._publish_no_replace(parent,source.name,destination.name)
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        _require(list(_identity(os.stat(destination.name,dir_fd=parent,follow_symlinks=False)))==member['physical_identity'])
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        os.fsync(parent)
        journal.append('detached',member_key=str(member_index),evidence=dict(evidence,detach_plan_raw_ref=planned))
        removed=removed_inodes if removed_inodes is not None else {}
        files=[row for row in preserved['files'] if row['member_index']==member_index]
        directories={row['relative_path']:row for row in preserved['directories'] if row['member_index']==member_index}
        directory_ids={'':tuple(member['physical_identity']),**{
            name:tuple(row['physical_identity']) for name,row in directories.items()}}
        for row in files:
            relative=Path(row['relative_path'])
            parent_relative='' if str(relative.parent)=='.' else str(relative.parent)
            leaf_parent=destination/relative.parent
            with _opened(leaf_parent,directory=True) as (leaf_fd,leaf_info):
                expected=directory_ids[parent_relative]
                _require(_identity(leaf_info)==expected,'scene_retirement_member_changed')
                allowance.tick()
                _current_parent(leaf_parent,leaf_fd,expected)
                info=os.stat(relative.name,dir_fd=leaf_fd,follow_symlinks=False)
                key=tuple(row['physical_identity'][:2])
                removed_count=removed.get(key,0)
                _require(list(_identity(info))==row['physical_identity'] and info.st_size==row['size_bytes']
                         and info.st_nlink==row['snapshot'][-1]-removed_count,'scene_retirement_payload_changed')
                _require(removed_count>0 or list(_snapshot(info))==row['snapshot'],'scene_retirement_payload_changed')
                current=dict(row,snapshot=list(_snapshot(info)))
                digest=hashlib.sha256()
                for chunk in _payload(destination/relative,current,allowance):
                    digest.update(chunk)
                _require('sha256:'+digest.hexdigest()==row['sha256'],'scene_retirement_payload_changed')
                allowance.tick()
                _current_parent(leaf_parent,leaf_fd,expected)
                _require(list(_snapshot(os.stat(relative.name,dir_fd=leaf_fd,follow_symlinks=False)))==current['snapshot'],
                         'scene_retirement_payload_changed')
                os.unlink(relative.name,dir_fd=leaf_fd)
                removed[key]=removed_count+1
                allowance.tick()
                _current_parent(leaf_parent,leaf_fd,expected)
                os.fsync(leaf_fd)
        for relative,row in sorted(directories.items(),key=lambda pair:len(Path(pair[0]).parts),reverse=True):
            value=Path(relative)
            parent_relative='' if str(value.parent)=='.' else str(value.parent)
            directory_parent=destination/value.parent
            with _opened(directory_parent,directory=True) as (fd,info):
                expected=directory_ids[parent_relative]
                _require(_identity(info)==expected)
                allowance.tick()
                _current_parent(directory_parent,fd,expected)
                _require(list(_identity(os.stat(value.name,dir_fd=fd,follow_symlinks=False)))==row['physical_identity'])
                os.rmdir(value.name,dir_fd=fd)  # Any unknown/new entry refuses; it is never swept.
                allowance.tick()
                _current_parent(directory_parent,fd,expected)
                os.fsync(fd)
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        _require(list(_identity(os.stat(destination.name,dir_fd=parent,follow_symlinks=False)))==member['physical_identity'])
        os.rmdir(destination.name,dir_fd=parent)
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        os.fsync(parent)
        outcome=dict(outcome='removed',canonical_path=str(source),detached_path=str(destination),
                     generation_id=generation_id,inventory_sha256=evidence['inventory_sha256'])
        outcome['event_raw_ref']=journal.append('member_removed',member_key=str(member_index),evidence=outcome)
        return outcome
