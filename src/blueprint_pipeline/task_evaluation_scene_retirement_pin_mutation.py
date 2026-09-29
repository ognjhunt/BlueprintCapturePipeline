"""Exact consent-selected pin CAS; caller holds scene EX before this pin lock.

Only protected original journal/pending selectors authorize this internal phase.
These records do not grant payload, process, owner or cleanup authority.
"""
import fcntl
import hashlib
import json
import os
import secrets
import stat
import sys
from contextlib import contextmanager

from .task_evaluation_scene_retirement_access import _canonical,_opened,_identity,_require,_close_owned
from .task_evaluation_scene_retirement_authority import selected_document
from .task_evaluation_scene_retirement_generations import _guard,_named,_new_file

_REASON='scene_retirement_terminal_pin_changed'


def _snapshot(info):
    return [info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_uid,info.st_gid,
            info.st_mtime_ns,info.st_ctime_ns,info.st_nlink]


def _reference(path,raw):
    return dict(path=str(path),sha256='sha256:'+hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


def _read(path,allowance):
    allowance.tick()
    with _opened(path) as (fd,info):
        _require(0<info.st_size<=16384 and stat.S_IMODE(info.st_mode)==0o640 and info.st_nlink==1,_REASON)
        allowance.charge('local_bytes',info.st_size)
        before=_snapshot(info)
        _guard(fd,_identity(info))
        raw=os.read(fd,info.st_size)
        _require(len(raw)==info.st_size and _snapshot(os.fstat(fd))==before,_REASON)
    return raw,before


@contextmanager
def terminal_pin_guard(policy,consent,allowance):
    """Lock the same actual inode used by native publishers, never a new store."""
    if not consent.get('terminal_pin_refs'):
        yield
        return
    root=_canonical(policy['reference_context']['pins_root'])
    allowance.tick()
    with _opened(root,directory=True) as (fd,info):
        expected=_identity(info)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            _require(False,'scene_retirement_terminal_pin_busy')
        _guard(fd,expected)
        with _opened(root,directory=True) as (_,current):
            _require(_identity(current)==expected,_REASON)
        allowance.tick()
        yield
        _guard(fd,expected)
        # Proved owned descriptor cleanup releases the lock even after expiry.


def _parent(path,parent,expected):
    _guard(parent,expected)
    with _opened(path,directory=True) as (_,current):
        _require(_identity(current)==expected,_REASON)
    _guard(parent,expected)


def _prepare(path,raw,snapshot,allowance):
    """Write a private temporary, prove first token, fsync before journaling it."""
    name='.'+secrets.token_hex(16)+'.pin-pending'
    allowance.charge('local_bytes',len(raw))
    with _opened(path.parent,directory=True) as (parent,info):
        expected=_identity(info)
        allowance.tick()
        _parent(path.parent,parent,expected)
        fd,identity=_new_file(parent,name,parent_identity=expected)
        def proof():
            allowance.tick()
            _parent(path.parent,parent,expected)
            _named(parent,expected,name,fd,identity)
        completed=False
        try:
            view=memoryview(raw)
            while view:
                proof()
                written=os.write(fd,view)
                _require(written>0,_REASON)
                view=view[written:]
            proof()
            os.fchown(fd,snapshot[4],snapshot[5])
            proof()
            os.fchmod(fd,stat.S_IMODE(snapshot[2]))
            # Mode is an intentional mutation; adopt only this owned transition.
            observed=os.fstat(fd)
            _require(_identity(observed)[:2]==identity[:2]
                and stat.S_IMODE(observed.st_mode)==stat.S_IMODE(snapshot[2]),_REASON)
            identity=_identity(observed)
            proof()
            os.fsync(fd)
            proof()
            os.fsync(parent)
            completed=True
            return dict(temporary_name=name,temporary_identity=list(identity),parent_identity=list(expected))
        finally:
            incoming=sys.exc_info()[1]
            if not completed:
                try:
                    _parent(path.parent,parent,expected)
                    _named(parent,expected,name,fd,identity)
                    os.unlink(name,dir_fd=parent)
                except (OSError,ValueError):
                    if incoming is not None:
                        incoming.add_note('scene_retirement_descriptor_cleanup_failed')
            failure=_close_owned(fd,identity)
            if failure and incoming is None:
                _require(False,failure)
            if failure and incoming is not None:
                incoming.add_note(failure)


def _replace(path,plan,journal):
    allowance=journal.allowance
    raw,observed=_read(path,allowance)
    if _reference(path,raw)==plan['after_raw_ref']:
        _require(observed[:3]==plan['temporary_identity'],_REASON)
        return observed
    _require(_reference(path,raw)==plan['before_raw_ref'] and observed==plan['before_snapshot'],_REASON)
    temporary=path.parent/plan['temporary_name']
    contents,version=_read(temporary,allowance)
    _require(_reference(path,contents)==plan['after_raw_ref'] and version[:3]==plan['temporary_identity']
        and version[4:6]==observed[4:6],_REASON)
    with _opened(path.parent,directory=True) as (parent,info),_opened(temporary) as (fd,temp):
        expected=_identity(info)
        identity=_identity(temp)
        _require(list(expected)==plan['parent_identity'] and list(identity)==plan['temporary_identity'],_REASON)
        allowance.tick()
        _parent(path.parent,parent,expected)
        _named(parent,expected,temporary.name,fd,identity)
        # Bind destination raw/stat again immediately before the atomic CAS.
        current,snapshot=_read(path,allowance)
        _require(_reference(path,current)==plan['before_raw_ref'] and snapshot==plan['before_snapshot'],_REASON)
        _parent(path.parent,parent,expected)
        _named(parent,expected,temporary.name,fd,identity)
        os.replace(temporary.name,path.name,src_dir_fd=parent,dst_dir_fd=parent)
        allowance.tick()
        _parent(path.parent,parent,expected)
        _guard(fd,identity)
        _require(_identity(os.stat(path.name,dir_fd=parent,follow_symlinks=False))==identity,_REASON)
        os.fsync(parent)
    after,observed=_read(path,allowance)
    _require(_reference(path,after)==plan['after_raw_ref'] and observed[:3]==plan['temporary_identity'],_REASON)
    return observed


def _row(row):
    _require(type(row) is dict and set(row)=={'original_raw_ref','original_value','original_raw_hex',
        'physical_identity','snapshot'},_REASON)
    encoded=row['original_raw_hex']
    _require(type(encoded) is str and 0<len(encoded)<=32768 and len(encoded)%2==0,_REASON)
    raw=bytes.fromhex(encoded)
    _require(_reference(row['original_raw_ref']['path'],raw)==row['original_raw_ref']
             and json.loads(raw)==row['original_value'],_REASON)
    return raw


def _operation(policy,rows,journal,*,restoring,outcomes=()):
    _require(type(rows) is list and len(rows)<=256 and (not restoring or len(outcomes)==len(rows)),_REASON)
    plans={}
    done={}
    prefix='pin_restore' if restoring else 'pin_release'
    for event in journal.events:
        journal.allowance.tick()
        if event['event'] in {prefix+'_planned','pin_restored' if restoring else 'pin_released'}:
            table=plans if event['event']==prefix+'_planned' else done
            _require(event['member_key'] not in table,_REASON)
            table[event['member_key']]=event['evidence']
    result=[]
    for index,row in enumerate(rows):
        journal.allowance.tick()
        original=_row(row)
        path=_canonical(row['original_raw_ref']['path'])
        root=_canonical(policy['reference_context']['pins_root'])
        _require(path.is_relative_to(root) and len(path.relative_to(root).parts)==2,_REASON)
        key='pin-'+str(index)
        if restoring:
            before_ref=outcomes[index]['released_raw_ref']
            before_snapshot=outcomes[index]['snapshot']
            after=original
        else:
            before_ref=row['original_raw_ref']
            before_snapshot=row['snapshot']
            after=(json.dumps(dict(row['original_value'],released_at_epoch=journal.allowance.last_wall),
                sort_keys=True,separators=(',',':'))+'\n').encode()
        _require(len(after)<=16384,_REASON)
        if key in done:
            value=done[key]
            raw,current=_read(path,journal.allowance)
            _require(value['pin_index']==index and value['original_raw_ref']==row['original_raw_ref']
                and _reference(path,raw)==value['restored_raw_ref' if restoring else 'released_raw_ref']
                and current==value['snapshot'],_REASON)
            result.append(value)
            continue
        plan=plans.get(key)
        if plan is None:
            raw,current=_read(path,journal.allowance)
            _require(_reference(path,raw)==before_ref and current==before_snapshot,_REASON)
            base=dict(pin_index=index,original_raw_ref=row['original_raw_ref'],before_raw_ref=before_ref,
                before_snapshot=before_snapshot,after_raw_ref=_reference(path,after))
            # Complete event framing is checked before creating/writing a temp.
            journal.preflight([(prefix+'_planned',key,dict(base,temporary_name='.'+'f'*32+'.pin-pending',
                temporary_identity=[2**64-1]*3,parent_identity=[2**64-1]*3)),
                ('pin_restored' if restoring else 'pin_released',key,dict(pin_index=index,
                 original_raw_ref=row['original_raw_ref'],released_raw_ref=_reference(path,after),
                 restored_raw_ref=_reference(path,after),snapshot=[2**64-1]*9))])
            plan=dict(base,**_prepare(path,after,before_snapshot,journal.allowance))
            journal.append(prefix+'_planned',member_key=key,evidence=plan)
        _require(plan['pin_index']==index and plan['original_raw_ref']==row['original_raw_ref']
             and plan['before_raw_ref']==before_ref and plan['before_snapshot']==before_snapshot,_REASON)
        if restoring:
            _require(plan['after_raw_ref']==row['original_raw_ref'],_REASON)
        snapshot=_replace(path,plan,journal)
        value=dict(pin_index=index,original_raw_ref=row['original_raw_ref'],snapshot=snapshot,
            **{('restored_raw_ref' if restoring else 'released_raw_ref'):plan['after_raw_ref']})
        journal.append('pin_restored' if restoring else 'pin_released',member_key=key,evidence=value)
        result.append(value)
    return result


def release_terminal_pins(policy,consent,rows,*,journal,pending_raw_ref):
    journal.allowance.charge('local_bytes',journal.initial_ref['size_bytes'])
    initial=selected_document(journal.initial_ref,maximum=16*1024*1024,protected=True)
    _require(initial.get('terminal_pin_release_rows')==rows and initial.get('intent_id')==consent['intent_id']
        and initial.get('intent_raw_ref')==consent['intent_raw_ref'],_REASON)
    journal.allowance.charge('local_bytes',pending_raw_ref['size_bytes'])
    pending=selected_document(pending_raw_ref,maximum=16*1024*1024)
    _require(pending.get('schema_version')=='scene_lifecycle_retirement_receipt.v1'
        and pending.get('status') in {'pending','retiring','incomplete'}
        and pending.get('intent_id')==consent['intent_id'] and pending.get('journal_initial_raw_ref')==journal.initial_ref,_REASON)
    return _operation(policy,rows,journal,restoring=False)


def restore_terminal_pins(policy,rows,outcomes,*,journal):
    journal.allowance.charge('local_bytes',journal.initial_ref['size_bytes'])
    initial=selected_document(journal.initial_ref,maximum=16*1024*1024,protected=True)
    _require(initial.get('terminal_pin_release_rows')==rows and initial.get('terminal_pin_outcomes')==outcomes
        and initial.get('schema_version')=='scene_restore_journal.v1',_REASON)
    return _operation(policy,rows,journal,restoring=True,outcomes=outcomes)


def pin_history(policy,consent,journal):
    """Rebind only exact private original/CAS versions; no mutation or new pin."""
    from .task_evaluation_scene_retirement_journal import SceneJournal
    _require(type(journal) is SceneJournal,_REASON)
    reference=journal.initial_ref
    journal.allowance.charge('local_bytes',reference['size_bytes'])
    initial=selected_document(reference,maximum=16*1024*1024,protected=True)
    rows=initial.get('terminal_pin_release_rows',[])
    _require(type(rows) is list and len(rows)<=256 and initial.get('intent_id')==consent['intent_id']
        and initial.get('intent_raw_ref')==consent['intent_raw_ref']
        and [row['original_raw_ref'] for row in rows]==consent.get('terminal_pin_refs',[]),_REASON)
    plans={}
    completed={}
    for event in journal.events:
        journal.allowance.tick()
        if event['event'] in {'pin_release_planned','pin_released'}:
            table=plans if event['event']=='pin_release_planned' else completed
            _require(event['member_key'] not in table,_REASON)
            table[event['member_key']]=event['evidence']
    result={}
    for index,row in enumerate(rows):
        _row(row)
        path=_canonical(row['original_raw_ref']['path'])
        _require(path.is_relative_to(_canonical(policy['reference_context']['pins_root'])),_REASON)
        raw,snapshot=_read(path,journal.allowance)
        current=_reference(path,raw)
        key='pin-'+str(index)
        plan=plans.get(key)
        if current==row['original_raw_ref']:
            _require(snapshot==row['snapshot'] and key not in completed,_REASON)
        else:
            _require(plan is not None and plan['pin_index']==index
                and plan['original_raw_ref']==row['original_raw_ref']
                and plan['before_raw_ref']==row['original_raw_ref'] and plan['before_snapshot']==row['snapshot']
                and plan['after_raw_ref']==current and plan['temporary_identity']==snapshot[:3],_REASON)
            if key in completed:
                done=completed[key]
                _require(done['released_raw_ref']==current and done['snapshot']==snapshot
                    and done['original_raw_ref']==row['original_raw_ref'] and done['pin_index']==index,_REASON)
        result[tuple(row['original_raw_ref'][key] for key in ('path','sha256','size_bytes'))]=dict(row,
            observed_raw_ref=current,observed_snapshot=snapshot)
    return result


def pin_records(rows,*,restoring=False,outcomes=()):
    """Worst-case suffix framing for the same complete operation preflight."""
    _require(type(rows) is list and len(rows)<=256 and (not restoring or len(rows)==len(outcomes)),_REASON)
    for index,row in enumerate(rows):
        _row(row)
        reference=row['original_raw_ref']
        before=outcomes[index]['released_raw_ref'] if restoring else reference
        snapshot=outcomes[index]['snapshot'] if restoring else row['snapshot']
        after=reference if restoring else dict(reference,sha256='sha256:'+'f'*64,size_bytes=16384)
        key='pin-'+str(index)
        yield ('pin_restore_planned' if restoring else 'pin_release_planned'),key,dict(pin_index=index,
            original_raw_ref=reference,before_raw_ref=before,before_snapshot=snapshot,after_raw_ref=after,
            temporary_name='.'+'f'*32+'.pin-pending',temporary_identity=[2**64-1]*3,parent_identity=[2**64-1]*3)
        yield ('pin_restored' if restoring else 'pin_released'),key,dict(pin_index=index,
            original_raw_ref=reference,snapshot=[2**64-1]*9,
            **{('restored_raw_ref' if restoring else 'released_raw_ref'):after})
