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


def _verify_current(preserved,index,allowance,removed_inodes,*,detached_path=None,missing_files=(),missing_directories=()):
    member=preserved['members'][index]
    files,directories=[],[]
    root=Path(member['path']) if detached_path is None else detached_path
    current=_scan(root,index,allowance,files,directories)
    missing_entries=set(missing_files)|set(missing_directories)
    if missing_entries:
        _require(current['snapshot'][:3]+current['snapshot'][4:6]
                 ==member['snapshot'][:3]+member['snapshot'][4:6]
                 and member['snapshot'][-1]-sum(str(Path(name).parent)=='.' for name in missing_entries)
                     <=current['snapshot'][-1]<=member['snapshot'][-1],
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
             and {row['relative_path'] for row in directories}==expected_directories.keys()-set(missing_directories),
             'scene_retirement_member_changed')
    for row in directories:
        original=expected_directories[row['relative_path']]
        if any(Path(name).is_relative_to(Path(row['relative_path'])) for name in missing_entries):
            _require(row['snapshot'][:3]+row['snapshot'][4:6]
                     ==original['snapshot'][:3]+original['snapshot'][4:6]
                     and original['snapshot'][-1]-sum(str(Path(name).parent)==row['relative_path'] for name in missing_entries)
                         <=row['snapshot'][-1]<=original['snapshot'][-1],
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


def _directory_plans(preserved,index,journal,generation_id,parent_identity):
    expected={'':preserved['members'][index],**{row['relative_path']:row for row in preserved['directories']
                                              if row['member_index']==index}}
    plans={}
    for event in journal.events:
        journal.allowance.tick()
        if event['event']!='directory_unlink_planned' or event['member_key']!=str(index):
            continue
        proof=selected_document(event['raw_ref'],maximum=65536,protected=True)
        value=proof.get('evidence',{})
        relative=value.get('relative_path')
        _require(type(relative) is str and relative in expected and relative not in plans,
                 'scene_retirement_journal_chain_unproven')
        parent='' if str(Path(relative).parent)=='.' else str(Path(relative).parent)
        required_parent=list(parent_identity) if relative=='' else expected[parent]['physical_identity']
        _require(proof.get('token')==journal.token and proof.get('event')=='directory_unlink_planned'
                 and proof.get('member_key')==str(index) and value.get('generation_id')==generation_id
                 and value.get('canonical_path')==preserved['members'][index]['path']
                 and value.get('physical_identity')==expected[relative]['physical_identity']
                 and value.get('parent_identity')==required_parent,
                 'scene_retirement_journal_chain_unproven')
        plans[relative]=(event['raw_ref'],value)
    return plans


def _directory_completion(journal,index,relative,reference,*,recovered):
    events=[event for event in journal.events if event['event']=='directory_unlinked'
            and event['member_key']==str(index) and event['evidence'].get('relative_path')==relative]
    _require(len(events)<=1,'scene_retirement_journal_chain_unproven')
    if events:
        _require(events[0]['evidence'].get('directory_plan_raw_ref')==reference,
                 'scene_retirement_journal_chain_unproven')
        return
    journal.append('directory_unlinked',member_key=str(index),evidence={
        'relative_path':relative,'directory_plan_raw_ref':reference,'absence_verified_after_interruption':recovered})


def _remove_directory(path,relative,identity,parent_identity,source,index,generation_id,journal,plans):
    allowance=journal.allowance
    with _opened(path.parent,directory=True) as (fd,info):
        _require(_identity(info)==parent_identity,'scene_retirement_member_changed')
        allowance.tick()
        _current_parent(path.parent,fd,parent_identity)
        with _opened(path,directory=True) as (child,child_info):
            _require(_identity(child_info)==identity,'scene_retirement_member_changed')
            allowance.tick()
            _guard(child,identity)
            with os.scandir(child) as names:
                _require(next(names,None) is None,'scene_retirement_member_changed')
            _guard(child,identity)
        evidence=dict(canonical_path=str(source),relative_path=relative,generation_id=generation_id,
                      physical_identity=list(identity),parent_identity=list(parent_identity))
        if relative in plans:
            reference,value=plans[relative]
            _require(value==evidence,'scene_retirement_journal_chain_unproven')
        else:
            reference=journal.append('directory_unlink_planned',member_key=str(index),evidence=evidence)
        allowance.tick()
        _current_parent(path.parent,fd,parent_identity)
        _require(_identity(os.stat(path.name,dir_fd=fd,follow_symlinks=False))==identity,
                 'scene_retirement_member_changed')
        allowance.tick()
        os.rmdir(path.name,dir_fd=fd)
        allowance.tick()
        _current_parent(path.parent,fd,parent_identity)
        allowance.tick()
        os.fsync(fd)
        _directory_completion(journal,index,relative,reference,recovered=False)


def _outcome(preserved,index,generation_id,journal,removed_allocated):
    member=preserved['members'][index]
    source=Path(member['path'])
    files=[row for row in preserved['files'] if row['member_index']==index]
    outcome=dict(outcome='removed',canonical_path=str(source),
        detached_path=str(source.parent/('.scene-retirement-'+journal.token+'-'+str(index))),
        generation_id=generation_id,inventory_sha256=inventory_digest(preserved,index),
        logical_bytes=sum(row['size_bytes'] for row in files),apparent_bytes=sum(row['size_bytes'] for row in files),
        unique_allocated_bytes=sum(row['allocated_bytes'] for row in {
            tuple(item['physical_identity'][:2]):item for item in files}.values()),
        removed_allocated_bytes=removed_allocated,removed_file_count=len(files),
        allocation_method='observed_file_st_blocks_512_last_union_link_unlinked')
    existing=[event for event in journal.events if event['event']=='member_removed' and event['member_key']==str(index)]
    _require(len(existing)<=1,'scene_retirement_journal_chain_unproven')
    if existing:
        _require(existing[0]['evidence']==outcome,'scene_retirement_journal_chain_unproven')
        outcome['event_raw_ref']=existing[0]['raw_ref']
    else:
        outcome['event_raw_ref']=journal.append('member_removed',member_key=str(index),evidence=outcome)
    return outcome


def removal_records(preserved,index,generation_id,journal,parent_identity):
    """Bound the complete native removal, including every future raw selector."""
    from .task_evaluation_scene_retirement_journal import MAX_EVENTS
    member=preserved['members'][index]
    source=Path(member['path'])
    destination=source.parent/('.scene-retirement-'+journal.token+'-'+str(index))
    reference=dict(path=str(journal.directory/(journal.token+'.'+str(MAX_EVENTS)+'.json')),
                   sha256='sha256:'+'f'*64,size_bytes=65536)
    evidence=dict(canonical_path=str(source),detached_path=str(destination),generation_id=generation_id,
                  pre_identity=member['physical_identity'],parent_identity=list(parent_identity),
                  inventory_sha256=inventory_digest(preserved,index))
    yield 'detach_planned',str(index),evidence
    yield 'detached',str(index),dict(evidence,detach_plan_raw_ref=reference)
    logical=allocated=count=0
    unique={}
    for row in preserved['files']:
        journal.allowance.tick()
        if row['member_index']!=index:
            continue
        logical+=row['size_bytes']
        allocated+=row['allocated_bytes']
        count+=1
        unique[tuple(row['physical_identity'][:2])]=row['allocated_bytes']
        yield 'leaf_unlink_planned',str(index),dict(canonical_path=str(source),relative_path=row['relative_path'],
            generation_id=generation_id,physical_identity=row['physical_identity'],size_bytes=row['size_bytes'],
            sha256=row['sha256'],nlink_before=row['snapshot'][-1])
        yield 'leaf_unlinked',str(index),dict(relative_path=row['relative_path'],leaf_plan_raw_ref=reference,
                                            absence_verified_after_interruption=False)
    directories={row['relative_path']:row for row in preserved['directories'] if row['member_index']==index}
    identities={'':member['physical_identity'],**{name:row['physical_identity'] for name,row in directories.items()}}
    for relative,row in {**directories,'':member}.items():
        parent='' if str(Path(relative).parent)=='.' else str(Path(relative).parent)
        yield 'directory_unlink_planned',str(index),dict(canonical_path=str(source),relative_path=relative,
            generation_id=generation_id,physical_identity=row['physical_identity'],
            parent_identity=list(parent_identity) if relative=='' else identities[parent])
        yield 'directory_unlinked',str(index),dict(relative_path=relative,directory_plan_raw_ref=reference,
                                                 absence_verified_after_interruption=False)
    yield 'member_removed',str(index),dict(outcome='removed',canonical_path=str(source),detached_path=str(destination),
        generation_id=generation_id,inventory_sha256=evidence['inventory_sha256'],logical_bytes=logical,
        apparent_bytes=logical,unique_allocated_bytes=sum(unique.values()),removed_allocated_bytes=allocated,
        removed_file_count=count,allocation_method='observed_file_st_blocks_512_last_union_link_unlinked')


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
    missing_directories=set()
    plans=[row for row in journal.events if row['event']=='detach_planned' and row['member_key']==str(member_index)]
    _require(len(plans)<=1,'scene_retirement_journal_chain_unproven')
    with _opened(source.parent,directory=True) as (parent,parent_info):
        parent_identity=_identity(parent_info)
        directory_plans=_directory_plans(preserved,member_index,journal,generation_id,parent_identity)
        journal.preflight(removal_records(preserved,member_index,generation_id,journal,parent_identity))
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
            try:
                found=os.stat(destination.name,dir_fd=parent,follow_symlinks=False)
            except FileNotFoundError:
                # Only a prejournaled empty ORIGINAL root can explain absence.
                # No new inode, detached basename or missing initial proof is adopted.
                files=[row for row in preserved['files'] if row['member_index']==member_index]
                required_dirs={row['relative_path'] for row in preserved['directories'] if row['member_index']==member_index}|{''}
                _require(required_dirs<=directory_plans.keys() and {row['relative_path'] for row in files}<=leaf_plans.keys(),
                         'scene_retirement_member_changed')
                for relative in sorted(required_dirs,key=lambda value:len(Path(value).parts),reverse=True):
                    _directory_completion(journal,member_index,relative,directory_plans[relative][0],recovered=True)
                allowance.tick()
                _current_parent(source.parent,parent,parent_identity)
                _require(not os.path.lexists(source) and not os.path.lexists(destination),
                         'scene_retirement_member_changed')
                return _outcome(preserved,member_index,generation_id,journal,sum(
                    row['allocated_bytes'] for row in files if leaf_plans[row['relative_path']][1]['nlink_before']==1))
            _require(list(_identity(found))==member['physical_identity'],'scene_retirement_member_changed')
            for relative in sorted(directory_plans,key=lambda value:len(Path(value).parts)):
                if relative=='':
                    continue
                candidate=Path(relative)
                if any(candidate.is_relative_to(Path(missing)) for missing in missing_directories):
                    missing_directories.add(relative)
                    continue
                with _opened(destination/candidate.parent,directory=True) as (directory_fd,directory_info):
                    expected_parent=tuple(directory_plans[relative][1]['parent_identity'])
                    _require(_identity(directory_info)==expected_parent,'scene_retirement_member_changed')
                    allowance.tick()
                    _current_parent(destination/candidate.parent,directory_fd,expected_parent)
                    try:
                        observed=os.stat(candidate.name,dir_fd=directory_fd,follow_symlinks=False)
                    except FileNotFoundError:
                        missing_directories.add(relative)
                    else:
                        _require(list(_identity(observed))==directory_plans[relative][1]['physical_identity'],
                                 'scene_retirement_member_changed')
            for relative in leaf_plans:
                candidate=Path(relative)
                if any(candidate.is_relative_to(Path(missing)) for missing in missing_directories):
                    missing_files.add(relative)
                    continue
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
            _verify_current(preserved,member_index,allowance,removed,detached_path=destination,
                            missing_files=missing_files,missing_directories=missing_directories)
        else:
            _require(list(_identity(original))==member['physical_identity'],'scene_retirement_member_changed')
            _verify_current(preserved,member_index,allowance,removed)
            if planned is None:
                planned=journal.append('detach_planned',member_key=str(member_index),evidence=evidence)
            allowance.tick()
            _current_parent(source.parent,parent,parent_identity)
            _require(list(_identity(os.stat(source.name,dir_fd=parent,follow_symlinks=False)))==member['physical_identity'])
            # Atomic NO-REPLACE, never check then overwrite or orphan adoption.
            allowance.tick()
            primitive._publish_no_replace(parent,source.name,destination.name)
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        _require(list(_identity(os.stat(destination.name,dir_fd=parent,follow_symlinks=False)))==member['physical_identity'])
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        allowance.tick()
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
                allowance.tick()
                os.unlink(relative.name,dir_fd=leaf_fd)
                removed[key]=removed_count+1
                if info.st_nlink==1:
                    removed_allocated+=row['allocated_bytes']
                allowance.tick()
                _current_parent(leaf_parent,leaf_fd,expected)
                allowance.tick()
                os.fsync(leaf_fd)
                journal.append('leaf_unlinked',member_key=str(member_index),evidence={
                    'relative_path':row['relative_path'],'leaf_plan_raw_ref':planned_leaf,
                    'absence_verified_after_interruption':False})
        for relative,row in sorted(directories.items(),key=lambda pair:len(Path(pair[0]).parts),reverse=True):
            if relative in missing_directories:
                _directory_completion(journal,member_index,relative,directory_plans[relative][0],recovered=True)
                continue
            value=Path(relative)
            parent_relative='' if str(value.parent)=='.' else str(value.parent)
            _remove_directory(destination/value,relative,tuple(row['physical_identity']),directory_ids[parent_relative],
                              source,member_index,generation_id,journal,directory_plans)
        allowance.tick()
        _current_parent(source.parent,parent,parent_identity)
        _require(list(_identity(os.stat(destination.name,dir_fd=parent,follow_symlinks=False)))==member['physical_identity'])
        _remove_directory(destination,'',tuple(member['physical_identity']),parent_identity,
                          source,member_index,generation_id,journal,directory_plans)
        return _outcome(preserved,member_index,generation_id,journal,removed_allocated)
