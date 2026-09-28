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
from .task_evaluation_scene_retirement_authority import selected_document
from .task_evaluation_scene_retirement_preservation import _payload, _scan, _snapshot


def inventory_digest(preserved,index):
    files=[dict(relative_path=row['relative_path'],mode=row['mode'],uid=row['uid'],gid=row['gid'],size_bytes=row['size_bytes'],
                sha256=row['sha256'],hardlink_group=row['hardlink_group'])
           for row in preserved['files'] if row['member_index']==index]
    directories=[dict(relative_path=row['relative_path'],mode=row['mode'],uid=row['uid'],gid=row['gid'])
                 for row in preserved['directories'] if row['member_index']==index]
    return canonical_digest({'files':files,'directories':directories})


def _verify_current(preserved,index,allowance,removed_inodes,*,detached_path=None,missing_files=()):
    member=preserved['members'][index]
    files,directories=[],[]
    root=Path(member['path']) if detached_path is None else detached_path
    current=_scan(root,index,allowance,files,directories)
    if missing_files:
        _require(current['snapshot'][:3]+current['snapshot'][4:6]
                 ==member['snapshot'][:3]+member['snapshot'][4:6]
                 and current['snapshot'][-1] in {member['snapshot'][-1],member['snapshot'][-1]-sum(
                     str(Path(name).parent)=='.' for name in missing_files)},
                 'scene_retirement_member_changed')
    elif detached_path is None:
        _require(current['snapshot']==member['snapshot'],'scene_retirement_member_changed')
    else:
        # A proved atomic rename can change only the root directory ctime here.
        # All remaining root attributes, exact child names and every payload
        # snapshot/hash still bind the original prejournaled inventory.
        _require(current['snapshot'][:-2]==member['snapshot'][:-2]
                 and current['snapshot'][-1]==member['snapshot'][-1],
                 'scene_retirement_member_changed')
    expected_files={row['relative_path']:row for row in preserved['files'] if row['member_index']==index}
    expected_directories={row['relative_path']:row for row in preserved['directories'] if row['member_index']==index}
    _require({row['relative_path'] for row in files}==expected_files.keys()-set(missing_files)
             and {row['relative_path'] for row in directories}==expected_directories.keys(),
             'scene_retirement_member_changed')
    for row in directories:
        original=expected_directories[row['relative_path']]
        if any(Path(name).is_relative_to(Path(row['relative_path'])) for name in missing_files):
            _require(row['snapshot'][:3]+row['snapshot'][4:6]
                     ==original['snapshot'][:3]+original['snapshot'][4:6]
                     and row['snapshot'][-1] in {original['snapshot'][-1],original['snapshot'][-1]-sum(
                         str(Path(name).parent)==row['relative_path'] for name in missing_files)},
                     'scene_retirement_member_changed')
        else:
            _require(row['snapshot']==original['snapshot'],'scene_retirement_member_changed')
    for row in files:
        original=expected_files[row['relative_path']]
        removed=removed_inodes.get(tuple(original['physical_identity'][:2]),0)
        if removed:
            _require(original['hardlink_group'] is not None
                     and row['snapshot'][:-2]==original['snapshot'][:-2]
                     and row['snapshot'][-1]==original['snapshot'][-1]-removed,
                     'scene_retirement_member_changed')
        else:
            _require(row['snapshot']==original['snapshot'],'scene_retirement_member_changed')
        checked=dict(original,snapshot=row['snapshot'])
        digest=hashlib.sha256()
        for chunk in _payload(root/row['relative_path'],checked,allowance):
            digest.update(chunk)
        _require('sha256:'+digest.hexdigest()==original['sha256'],'scene_retirement_payload_changed')


def _current_parent(path,fd,expected):
    _guard(fd,expected)
    with _opened(path,directory=True) as (_,info):
        _require(_identity(info)==expected,'scene_retirement_member_changed')
    _guard(fd,expected)


def _leaf_plans(preserved,index,journal,generation_id):
    expected={row['relative_path']:row for row in preserved['files'] if row['member_index']==index}
    plans={}
    for event in journal.events:
        journal.allowance.tick()
        if event['event']!='leaf_unlink_planned' or event['member_key']!=str(index):
            continue
        proof=selected_document(event['raw_ref'],maximum=65536,protected=True)
        value=proof.get('evidence',{})
        relative=value.get('relative_path')
        _require(type(relative) is str and relative in expected and relative not in plans,
                 'scene_retirement_journal_chain_unproven')
        row=expected[relative]
        _require(proof.get('token')==journal.token and proof.get('event')=='leaf_unlink_planned'
                 and proof.get('member_key')==str(index)
                 and value.get('generation_id')==generation_id
                 and value.get('canonical_path')==preserved['members'][index]['path']
                 and value.get('physical_identity')==row['physical_identity']
                 and value.get('sha256')==row['sha256'] and value.get('size_bytes')==row['size_bytes']
                 and type(value.get('nlink_before')) is int and 0<value['nlink_before']<=row['snapshot'][-1],
                 'scene_retirement_journal_chain_unproven')
        plans[relative]=(event['raw_ref'],value)
    return plans


def detach_and_remove(preserved,*,member_index,generation_id,journal,removed_inodes=None):
    """Internal: durable pre-detach proof, atomic no-replace and exact leaf union."""
    allowance=journal.allowance
    allowance.tick()
    _require(type(member_index) is int and 0<=member_index<len(preserved['members']))
    member=preserved['members'][member_index]
    source=Path(member['path'])
    destination=source.parent/('.scene-retirement-'+journal.token+'-'+str(member_index))
    removed=removed_inodes if removed_inodes is not None else {}
    leaf_plans=_leaf_plans(preserved,member_index,journal,generation_id)
    missing_files=set()
    plans=[row for row in journal.events if row['event']=='detach_planned' and row['member_key']==str(member_index)]
    _require(len(plans)<=1,'scene_retirement_journal_chain_unproven')
    with _opened(source.parent,directory=True) as (parent,parent_info):
        parent_identity=_identity(parent_info)
        evidence=dict(canonical_path=str(source),detached_path=str(destination),generation_id=generation_id,
                      pre_identity=member['physical_identity'],parent_identity=list(parent_identity),
                      inventory_sha256=inventory_digest(preserved,member_index))
        if plans:
            prior=selected_document(plans[0]['raw_ref'],maximum=65536,protected=True)
            _require(prior.get('event')=='detach_planned' and prior.get('token')==journal.token
                     and prior.get('member_key')==str(member_index) and prior.get('evidence')==evidence,
                     'scene_retirement_journal_chain_unproven')
            planned=plans[0]['raw_ref']
        else:
            planned=None
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        try:
            original=os.stat(source.name,dir_fd=parent,follow_symlinks=False)
        except FileNotFoundError:
            _require(planned is not None,'scene_retirement_member_changed')
            _current_parent(source.parent,parent,parent_identity)
            found=os.stat(destination.name,dir_fd=parent,follow_symlinks=False)
            _require(list(_identity(found))==member['physical_identity'],'scene_retirement_member_changed')
            for relative in leaf_plans:
                candidate=Path(relative)
                with _opened(destination/candidate.parent,directory=True) as (leaf_fd,leaf_info):
                    expected_parent=tuple(member['physical_identity']) if str(candidate.parent)=='.' else next(
                        tuple(row['physical_identity']) for row in preserved['directories']
                        if row['member_index']==member_index and row['relative_path']==str(candidate.parent))
                    _require(_identity(leaf_info)==expected_parent,'scene_retirement_member_changed')
                    _current_parent(destination/candidate.parent,leaf_fd,expected_parent)
                    try:
                        os.stat(candidate.name,dir_fd=leaf_fd,follow_symlinks=False)
                    except FileNotFoundError:
                        missing_files.add(relative)
            by_inode={}
            for row in preserved['files']:
                if row['member_index']==member_index and row['relative_path'] in missing_files:
                    key=tuple(row['physical_identity'][:2])
                    by_inode[key]=by_inode.get(key,0)+1
            for key,count in by_inode.items():
                removed[key]=max(removed.get(key,0),count)
            _verify_current(preserved,member_index,allowance,removed,detached_path=destination,missing_files=missing_files)
        else:
            _require(list(_identity(original))==member['physical_identity'],'scene_retirement_member_changed')
            _verify_current(preserved,member_index,allowance,removed)
            if planned is None:
                planned=journal.append('detach_planned',member_key=str(member_index),evidence=evidence)
            allowance.tick()
            _current_parent(source.parent,parent,parent_identity)
            _require(list(_identity(os.stat(source.name,dir_fd=parent,follow_symlinks=False)))==member['physical_identity'])
            # Atomic NO-REPLACE, never check then overwrite or orphan adoption.
            primitive._publish_no_replace(parent,source.name,destination.name)
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        _require(list(_identity(os.stat(destination.name,dir_fd=parent,follow_symlinks=False)))==member['physical_identity'])
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        os.fsync(parent)
        journal.append('detached',member_key=str(member_index),evidence=dict(evidence,detach_plan_raw_ref=planned))
        files=[row for row in preserved['files'] if row['member_index']==member_index]
        directories={row['relative_path']:row for row in preserved['directories'] if row['member_index']==member_index}
        directory_ids={'':tuple(member['physical_identity']),**{
            name:tuple(row['physical_identity']) for name,row in directories.items()}}
        removed_allocated=sum(row['allocated_bytes'] for row in files if row['relative_path'] in missing_files
            and leaf_plans[row['relative_path']][1]['nlink_before']==1)
        for row in files:
            if row['relative_path'] in missing_files:
                planned_leaf=leaf_plans[row['relative_path']][0]
                if not any(event['event']=='leaf_unlinked' and event['member_key']==str(member_index)
                           and event['evidence'].get('leaf_plan_raw_ref')==planned_leaf for event in journal.events):
                    journal.append('leaf_unlinked',member_key=str(member_index),evidence={
                        'relative_path':row['relative_path'],'leaf_plan_raw_ref':planned_leaf,
                        'absence_verified_after_interruption':True})
                continue
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
                if row['relative_path'] in leaf_plans:
                    planned_leaf=leaf_plans[row['relative_path']][0]
                else:
                    planned_leaf=journal.append('leaf_unlink_planned',member_key=str(member_index),evidence={
                        'canonical_path':str(source),'relative_path':row['relative_path'],
                        'generation_id':generation_id,'physical_identity':row['physical_identity'],
                        'size_bytes':row['size_bytes'],'sha256':row['sha256'],'nlink_before':info.st_nlink})
                allowance.tick()
                _current_parent(leaf_parent,leaf_fd,expected)
                _require(list(_snapshot(os.stat(relative.name,dir_fd=leaf_fd,follow_symlinks=False)))==current['snapshot'],
                         'scene_retirement_payload_changed')
                os.unlink(relative.name,dir_fd=leaf_fd)
                removed[key]=removed_count+1
                if info.st_nlink==1:
                    removed_allocated+=row['allocated_bytes']
                allowance.tick()
                _current_parent(leaf_parent,leaf_fd,expected)
                os.fsync(leaf_fd)
                journal.append('leaf_unlinked',member_key=str(member_index),evidence={
                    'relative_path':row['relative_path'],'leaf_plan_raw_ref':planned_leaf,
                    'absence_verified_after_interruption':False})
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
                     generation_id=generation_id,inventory_sha256=evidence['inventory_sha256'],
                     logical_bytes=sum(row['size_bytes'] for row in files),
                     apparent_bytes=sum(row['size_bytes'] for row in files),
                     unique_allocated_bytes=sum(row['allocated_bytes'] for row in {
                         tuple(item['physical_identity'][:2]):item for item in files}.values()),
                     removed_allocated_bytes=removed_allocated,removed_file_count=len(files),
                     allocation_method='observed_file_st_blocks_512_last_union_link_unlinked')
        outcome['event_raw_ref']=journal.append('member_removed',member_key=str(member_index),evidence=outcome)
        return outcome
