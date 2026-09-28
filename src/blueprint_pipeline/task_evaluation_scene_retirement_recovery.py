"""Resume only an exact protected action; no fresh origin or orphan adoption."""
from pathlib import Path
import os
import re

from .task_evaluation_scene_retirement_access import _opened, _identity

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_retirement_access import _require
from .task_evaluation_scene_retirement_authority import selected_document, load_document, TOKEN
from .task_evaluation_scene_retirement_intent_receipt import _projection, _resume, publish_pending_receipt
from .task_evaluation_scene_retirement_journal import publish_record, SceneJournal, _parent


def _attempt_path(policy,consent):
    _require(type(consent.get('consent_id')) is str and TOKEN.fullmatch(consent['consent_id']),
             'scene_retirement_resume_unproven')
    return Path(policy['journal_store'])/('retirement-attempt.'+consent['consent_id']+'.json')


def claim_retirement(policy,authority,token,allowance):
    """Publish a fixed protected token/origin before any preservation transfer.

    An interrupted preparation with no completed private initializer cannot
    silently spend again. It stays protected pending original-token recovery.
    """
    consent=authority['consent']
    _require(type(token) is str and TOKEN.fullmatch(token),'scene_retirement_resume_unproven')
    value=dict(schema_version='scene_retirement_attempt.v1',token=token,
        intent_id=consent['intent_id'],consent_id=consent['consent_id'],
        intent_raw_ref=consent['intent_raw_ref'],plan_raw_ref=consent['plan_raw_ref'],
        consent_raw_ref=authority['consent_raw_ref'],policy_sha256=consent['policy_sha256'],
        cohort_sha256=consent['cohort_sha256'],members=consent['members'],
        initial_path=str(Path(policy['journal_store'])/(token+'.initial.json')),
        action_allowance=allowance.checkpoint())
    value['claim_digest']=canonical_digest(value,digest_field='claim_digest')
    path=_attempt_path(policy,consent)
    reference=publish_record(path.parent,path.name,value,maximum=65536,allowance=allowance)
    return dict(claim=value,claim_raw_ref=reference,index=0,archive_index=0,prior_raw_ref=reference,ready=None)


def _claim(policy,authority):
    consent=authority['consent']
    try:
        value,reference=load_document(_attempt_path(policy,consent),maximum=65536,protected=True)
    except FileNotFoundError:
        return None
    keys={'schema_version','token','intent_id','consent_id','intent_raw_ref','plan_raw_ref',
          'consent_raw_ref','policy_sha256','cohort_sha256','members','initial_path','action_allowance','claim_digest'}
    _require(set(value)==keys and value.get('schema_version')=='scene_retirement_attempt.v1'
             and type(value.get('token')) is str and TOKEN.fullmatch(value['token'])
             and value.get('claim_digest')==canonical_digest(value,digest_field='claim_digest')
             and value.get('consent_raw_ref')==authority['consent_raw_ref']
             and all(value.get(k)==consent[k] for k in ('intent_id','consent_id','intent_raw_ref',
                 'plan_raw_ref','policy_sha256','cohort_sha256','members'))
             and value['initial_path']==str(Path(policy['journal_store'])/(value['token']+'.initial.json')),
             'scene_retirement_resume_unproven')
    return value,reference


def _preparation(policy,claim,reference,allowance):
    token=claim['token']
    directory=Path(policy['journal_store'])
    prefix='preparation.'+token+'.'
    indexes=set()
    archive_refs=[]
    with _opened(directory,directory=True,protected=True) as (fd,info):
        with os.scandir(fd) as entries:
            examined=0
            for entry in entries:
                allowance.tick()
                _parent(directory,fd,_identity(info))
                examined+=1
                _require(examined<=100000,'scene_retirement_journal_limit')
                if not entry.name.startswith(prefix) or not entry.name.endswith('.escrow.json'):
                    continue
                text=entry.name[len(prefix):-len('.escrow.json')]
                _require(re.fullmatch(r'[1-9][0-9]{0,2}',text) and int(text)<=256,
                         'scene_retirement_preparation_resume_unproven')
                indexes.add(int(text))
        _parent(directory,fd,_identity(info))
        after=os.fstat(fd)
        _require((_identity(after),after.st_size,after.st_mtime_ns,after.st_ctime_ns)==(
            _identity(info),info.st_size,info.st_mtime_ns,info.st_ctime_ns),
            'scene_retirement_preparation_resume_unproven')
    _require(indexes==set(range(1,len(indexes)+1)),'scene_retirement_preparation_resume_unproven')
    state=dict(claim=claim,claim_raw_ref=reference,index=0,archive_index=0,prior_raw_ref=reference,ready=None)
    checkpoint=claim['action_allowance']
    keys={'schema_version','token','index','archive_index','phase','claim_raw_ref','prior_raw_ref',
          'action_allowance','reserved_bytes','archive_name','escrow_digest'}
    for index in range(1,len(indexes)+1):
        value,observed=load_document(directory/(prefix+str(index)+'.escrow.json'),maximum=65536,protected=True)
        _require(set(value)==keys and value['schema_version']=='scene_retirement_preparation_escrow.v1'
            and value['token']==token and type(value['index']) is int and value['index']==index
            and value['claim_raw_ref']==reference and value['prior_raw_ref']==state['prior_raw_ref']
            and value['phase'] in {'archive','ready-revalidation','finalize'}
            and type(value['archive_index']) is int and value['archive_index']==state['archive_index']+(value['phase']=='archive')
            and value['escrow_digest']==canonical_digest(value,digest_field='escrow_digest'),
            'scene_retirement_preparation_resume_unproven')
        selected=value['action_allowance']
        reserved=value['reserved_bytes']
        _require(type(selected) is dict and type(reserved) is dict and set(reserved)=={'local_bytes','archive_bytes','remote_bytes'}
            and all(selected.get(k)==checkpoint.get(k) for k in ('start_monotonic','started_wall','expires_at','elapsed_seconds','limits'))
            and type(selected.get('counts')) is dict and set(selected['counts'])==set(reserved)
            and all(type(reserved[k]) is int and reserved[k]>=0 and type(selected['counts'][k]) is int
                and selected['counts'][k]>=checkpoint['counts'][k]+reserved[k] for k in reserved),
            'scene_retirement_preparation_resume_unproven')
        expected=token+'.'+str(value['archive_index'])+'.tar'
        _require(value['archive_name']==expected,'scene_retirement_preparation_resume_unproven')
        checkpoint=selected
        if value['phase']=='archive':
            archive_refs.append(observed)
        state.update(index=index,archive_index=value['archive_index'],prior_raw_ref=observed)
    allowance.bind_resume(checkpoint)
    try:
        ready,ready_ref=load_document(directory/(prefix+'ready.json'),maximum=16*1024*1024,protected=True)
    except FileNotFoundError:
        return state
    _require(set(ready)=={'schema_version','token','claim_raw_ref','escrow_raw_ref','preserved','ready_digest'}
        and ready['schema_version']=='scene_retirement_preparation_ready.v1' and ready['token']==token
        and ready['claim_raw_ref']==reference and ready['ready_digest']==canonical_digest(ready,digest_field='ready_digest')
        and type(ready['preserved']) is dict,'scene_retirement_preparation_resume_unproven')
    selected=ready['escrow_raw_ref']
    _require(type(selected) is dict and selected in archive_refs,
             'scene_retirement_preparation_resume_unproven')
    escrow=selected_document(selected,maximum=65536,protected=True)
    _require(escrow.get('claim_raw_ref')==reference and escrow.get('phase')=='archive'
        and 1<=escrow['index']<=state['index'] and escrow.get('token')==token
        and Path(ready['preserved']['archive']['uri']).name==escrow['archive_name'],
        'scene_retirement_preparation_resume_unproven')
    state.update(ready=ready['preserved'],ready_raw_ref=ready_ref)
    return state


def preparation_escrow(policy,state,allowance,*,phase,local_bytes=0,archive_bytes=0,remote_bytes=0):
    _require(state['index']<256 and phase in {'archive','ready-revalidation','finalize'},
             'scene_retirement_preparation_limit')
    reserved=dict(local_bytes=local_bytes,archive_bytes=archive_bytes,remote_bytes=remote_bytes)
    for key,count in reserved.items():
        allowance.charge(key,count)
    index=state['index']+1
    archive_index=state['archive_index']+(phase=='archive')
    value=dict(schema_version='scene_retirement_preparation_escrow.v1',token=state['claim']['token'],
        index=index,archive_index=archive_index,phase=phase,claim_raw_ref=state['claim_raw_ref'],
        prior_raw_ref=state['prior_raw_ref'],reserved_bytes=reserved,action_allowance=allowance.checkpoint(),
        archive_name=state['claim']['token']+'.'+str(archive_index)+'.tar')
    value['escrow_digest']=canonical_digest(value,digest_field='escrow_digest')
    name='preparation.'+value['token']+'.'+str(index)+'.escrow.json'
    reference=publish_record(policy['journal_store'],name,value,maximum=65536,allowance=allowance)
    state.update(index=index,archive_index=archive_index,prior_raw_ref=reference)
    return value['archive_name']


def complete_preparation(policy,state,preserved,allowance):
    value=dict(schema_version='scene_retirement_preparation_ready.v1',token=state['claim']['token'],
        claim_raw_ref=state['claim_raw_ref'],escrow_raw_ref=state['prior_raw_ref'],preserved=preserved)
    value['ready_digest']=canonical_digest(value,digest_field='ready_digest')
    name='preparation.'+value['token']+'.ready.json'
    reference=publish_record(policy['journal_store'],name,value,maximum=16*1024*1024,allowance=allowance)
    state.update(ready=preserved,ready_raw_ref=reference)


def select_retirement(policy, authority, allowance):
    consent = authority['consent']
    context = policy.get('reference_context')
    if not isinstance(context, dict):
        return None
    path = Path(context['roots']['intent_root']) / consent['intent_id'] / 'scene-retired.v1.json'
    selected_claim=_claim(policy,authority)
    claim=None if selected_claim is None else selected_claim[0]
    missing_projection=False
    try:
        _, reference = load_document(path, maximum=65536)
    except FileNotFoundError:
        if claim is None:
            return None
        try:
            initial,initial_ref=load_document(claim['initial_path'],maximum=16*1024*1024,protected=True)
        except FileNotFoundError:
            return _preparation(policy,claim,selected_claim[1],allowance)
        journal=SceneJournal.resume(initial_ref,allowance=allowance)
        missing_projection=True
    else:
        _, _, _, _, pending, _ = _projection(policy, consent, reference, allowance)
        journal = _resume(policy, pending, allowance)
        initial = selected_document(journal.initial_ref, maximum=16 * 1024 * 1024, protected=True)
    _require(initial.get('schema_version') == 'scene_retirement_journal.v1'
             and initial.get('status') == 'pending'
             and initial.get('intent_id') == consent['intent_id']
             and initial.get('intent_raw_ref') == consent['intent_raw_ref']
             and initial.get('plan_raw_ref') == consent['plan_raw_ref']
             and initial.get('consent_raw_ref') == authority['consent_raw_ref']
             and initial.get('policy_sha256') == consent['policy_sha256']
             and initial.get('cohort_sha256') == consent['cohort_sha256']
             and initial.get('members') == consent['members'], 'scene_retirement_resume_unproven')
    if claim is not None:
        checkpoint=initial.get('action_allowance')
        _require(initial.get('token')==claim['token'] and type(checkpoint) is dict
                 and type(claim['action_allowance']) is dict
                 and all(checkpoint.get(k)==claim['action_allowance'].get(k) for k in
                     ('start_monotonic','started_wall','expires_at','elapsed_seconds','limits')),
                 'scene_retirement_resume_allowance_unproven')
    bind_original_allowance(journal,initial,allowance)
    if missing_projection:
        # No mutation event can predate public pending publication. A deleted
        # projection after removal cannot masquerade as an interrupted publish.
        _require(journal.sequence==0,'scene_retirement_receipt_resume_unproven')
        reference=publish_pending_receipt(policy,consent,journal,initial['preserved'],allowance)
    return journal, reference, initial


def bind_original_allowance(journal, initial, allowance, *, restoring=False):
    selected = initial.get('action_allowance')
    for event in journal.events:
        if event['event'] == 'allowance_reserved':
            _require(event['member_key'] == 'action'
                     and event['evidence'].get('phase') in ({'restore'} if restoring else {'remove', 'resume-readback'}),
                     'scene_retirement_resume_allowance_unproven')
            selected = event['evidence'].get('action_allowance')
    allowance.bind_resume(selected)


def reserve_phase(journal, preserved, *, readback=False, restoring=False, published_objects=None):
    """Persist an upper bound before physical work, including a crash prefix.

    Actual reads still charge natively. This conservative reservation is never
    refunded; a retry pays its next phase against the original remaining cap.
    """
    allowance = journal.allowance
    if restoring:
        allowance.charge('remote_bytes', 2 * (preserved['archive']['size_bytes']+1))
        allowance.charge('local_bytes', 2 * sum(row['size_bytes'] for row in preserved['files']))
    elif readback:
        rows=[] if published_objects is None else published_objects
        _require(type(rows) is list and len(rows)<=10000,'scene_retirement_resume_unproven')
        remote=preserved['archive']['size_bytes']+1
        for row in rows:
            allowance.tick()
            _require(type(row) is dict and type(row.get('size_bytes')) is int and row['size_bytes']>0,
                     'scene_retirement_resume_unproven')
            remote+=row['size_bytes']+1
        allowance.charge('remote_bytes', remote)
    else:
        allowance.charge('local_bytes', 2 * sum(row['size_bytes'] for row in preserved['files']))
    evidence = dict(phase='restore' if restoring else ('resume-readback' if readback else 'remove'),
                    action_allowance=allowance.checkpoint())
    journal.preflight([('allowance_reserved', 'action', evidence)])
    journal.append('allowance_reserved', member_key='action', evidence=evidence)


def resumed_generations(engine, policy, consent, journal, initial):
    result = []
    _require(len(initial['generations']) == len(consent['members']), 'scene_retirement_resume_unproven')
    for index, member in enumerate(consent['members']):
        current, _ = engine._generation(policy, member, expected_states={'active', 'restored-active', 'retiring', 'retired'})
        original = initial['generations'][index]
        if current['state'] in {'active', 'restored-active'}:
            _require(current == original, 'scene_retirement_generation_changed')
        else:
            _require(current.get('retirement_token') == journal.token
                     and current.get('inventory_sha256') == member['inventory_sha256'],
                     'scene_retirement_generation_changed')
            events = [event for event in journal.events if event['member_key'] == str(index)
                      and event['raw_ref']['sha256'] == current.get('journal_sha256')]
            _require(len(events) == 1 and events[0]['event'] == (
                'retiring' if current['state'] == 'retiring' else 'member_removed'),
                'scene_retirement_generation_changed')
            proof = events[0]['evidence']
            _require(proof.get('generation_id') == current['generation_id']
                     and proof.get('inventory_sha256') == member['inventory_sha256'],
                     'scene_retirement_generation_changed')
        result.append(current)
    return result


def select_restore(policy, authority, allowance, retired_reference):
    consent=authority['consent']
    path=Path(policy['reference_context']['roots']['intent_root'])/consent['intent_id']/'scene-retired.v1.json'
    _,reference=load_document(path,maximum=65536)
    _,current,_,_,_,_=_projection(policy,consent,reference,allowance,restoring=True)
    if 'restore_journal_initial_raw_ref' not in current:
        _require(current['status']=='retired','scene_retirement_restore_resume_unproven')
        return None
    initial_ref=current['restore_journal_initial_raw_ref']
    token=current['restore_token']
    _require(initial_ref['path']==str(Path(policy['journal_store'])/(token+'.initial.json')),
             'scene_retirement_restore_resume_unproven')
    from .task_evaluation_scene_retirement_journal import SceneJournal
    journal=SceneJournal.resume(initial_ref,allowance=allowance)
    initial=selected_document(initial_ref,maximum=16*1024*1024,protected=True)
    _require(initial.get('schema_version')=='scene_restore_journal.v1'
             and initial.get('status')=='restoring'
             and initial.get('intent_id')==consent['intent_id']
             and initial.get('intent_raw_ref')==consent['intent_raw_ref']
             and initial.get('members')==consent['members']
             and initial.get('consent_raw_ref')==authority['consent_raw_ref']
             and initial.get('retired_journal_raw_ref')==retired_reference
             and initial.get('original_retirement_token')==current['retiring_token'],
             'scene_retirement_restore_resume_unproven')
    return journal,reference,initial,current


def resumed_restore_generations(engine,policy,consent,journal,initial):
    _require(len(initial['generations'])==len(consent['members']),'scene_retirement_restore_resume_unproven')
    result=[]
    for index,member in enumerate(consent['members']):
        current,_=engine._generation(policy,member,expected_states={'retired','restoring','restored-active'},
                                    retired_token=initial['original_retirement_token'])
        if current['state']=='retired':
            _require(current==initial['generations'][index],'scene_retirement_generation_changed')
        else:
            events=[event for event in journal.events if event['member_key']==str(index)
                    and event['raw_ref']['sha256']==current.get('journal_sha256')]
            _require(len(events)==1 and events[0]['event']==(
                'restoring' if current['state']=='restoring' else 'restored-active'),
                'scene_retirement_generation_changed')
            proof=events[0]['evidence']
            if current['state']=='restoring':
                _require(proof.get('generation_id')==current['generation_id'],'scene_retirement_generation_changed')
            else:
                _require(proof.get('canonical_path')==member['canonical_path'] and proof.get('outcome')=='restored'
                         and proof.get('restore_identity')==[current['dev'],current['ino'],current['mode']],
                         'scene_retirement_generation_changed')
        result.append(current)
    return result


def restored_prefix(journal,generations):
    result=[]
    for index,generation in enumerate(generations):
        if generation['state']!='restored-active':
            _require(not any(row['state']=='restored-active' for row in generations[index+1:]),
                     'scene_retirement_generation_changed')
            break
        rows=[event for event in journal.events if event['event']=='member_restored'
              and event['member_key']==str(index)]
        _require(len(rows)==1,'scene_retirement_restore_journal_unproven')
        result.append(rows[0]['evidence'])
    return result


def removed_inode_counts(journal):
    """Only completed native unlink events establish prior union link changes."""
    plans = {event['raw_ref']['sha256']: event for event in journal.events
             if event['event'] == 'leaf_unlink_planned'}
    result, seen = {}, set()
    for event in journal.events:
        if event['event'] != 'leaf_unlinked':
            continue
        selected = event['evidence']['leaf_plan_raw_ref']
        _require(selected['sha256'] in plans, 'scene_retirement_journal_chain_unproven')
        plan = plans[selected['sha256']]
        _require(plan['raw_ref'] == selected and plan['member_key'] == event['member_key'],
                 'scene_retirement_journal_chain_unproven')
        if selected['sha256'] not in seen:
            seen.add(selected['sha256'])
            inode = tuple(plan['evidence']['physical_identity'][:2])
            result[inode] = result.get(inode, 0) + 1
    return result
