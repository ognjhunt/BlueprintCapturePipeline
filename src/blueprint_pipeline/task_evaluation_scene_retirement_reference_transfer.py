"""Exact completed metadata transfer; independent process admission is mandatory.

Retained queue bytes stay at their original path. Selected native member proofs
bind each accepted record; an arbitrary raw version cannot grant transfer.
Unknown/active/unresolved references keep the scene. No references-clear flag or
consumer authority is changed by this read-only evidence.
"""
import math
import re
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_retirement_access import _canonical, _require, SceneRetirementAccessError
from .task_evaluation_scene_retirement_authority import selected_document
from .task_evaluation_scene_retirement_declared_bytes import _selector
from .task_evaluation_scene_lineage_budget import _Rows

_SCOPES={'pins','primary_queues','auxiliary_queues'}
_ROLE_SCHEMAS={
    ('preparation','envelope'):('task_evaluation_launch_preparation_envelope.v1',{'completed','materialized'}),
    ('activation','envelope'):('task_evaluation_launch_activation_envelope.v1',{'prepared'}),
    ('preparation','result'):('task_evaluation_launch_preparation_result.v1',{'results'}),
    ('activation','result'):('task_evaluation_launch_activation_result.v1',{'results'}),
}
_SUCCESS={
    'preparation':{'inputs_materialized_awaiting_construction_adapter',
        'native_arena_inputs_verified_awaiting_profile_authority',
        'queued_for_production_episode_compilation','queued_for_production_scene_configuration'},
    'activation':{'profile_authority_materialized_no_execution','policy_campaign_queue_materialized_no_execution'},
}
_REASON='scene_retirement_reference_closure_unproven'
_SAM_KEYS={
    'sam_jobs':{'schema_version','child_id','parent_preparation_id','parent_request_digest','plan_digest',
                'phase','inputs_digest','expected_source_commit','plan_ref','inputs','job_digest'},
    'sam_results':{'schema_version','child_id','job_digest','parent_request_digest','plan_digest',
                   'phase','source_commit','status','artifacts','executor_result','result_digest'},
}


def _rows(value):
    _require(type(value) in (list,_Rows) and len(value)<=10000,'scene_retirement_reference_limit')
    return value


def _sources(fresh,allowance):
    result={}
    occurrences=0
    for row in _rows(fresh.get('measured_members',[])):
        allowance.tick()
        _require(type(row) is dict,_REASON)
        for proof in _rows(row.get('source_provenance',[])):
            allowance.tick()
            occurrences+=1
            _require(occurrences<=10000,'scene_retirement_reference_limit')
            _require(type(proof) is dict and type(proof.get('role')) is str,_REASON)
            identity=_selector(proof)
            # Native measurement provenance is selected membership evidence.
            # Raw_versions/unselected observations are deliberately not indexed.
            result.setdefault(identity,[]).append(proof)
    return result


def _current_record(record,selected,allowance):
    allowance.tick()
    _require(type(record) is dict and record.get('disposition')=='supported',_REASON)
    source=record.get('source')
    _require(type(source) is dict and set(source)=={'family','queue_root','role','row_path',
        'raw_sha256','raw_size_bytes','observed_identity'},_REASON)
    identity=_selector(source,'row_path','raw_sha256','raw_size_bytes')
    _require(identity[2]>0 and identity in selected,_REASON)
    family,role=source.get('family'),source.get('role')
    contract=_ROLE_SCHEMAS.get((family,role))
    _require(contract is not None,_REASON)
    schema,states=contract
    root=_canonical(source.get('queue_root'))
    path=Path(identity[0])
    _require(path.is_relative_to(root) and len(path.relative_to(root).parts)==2
             and path.parent.name in states,_REASON)
    # Reselect actual named bytes; a copied record at another path or a changed
    # retained version cannot borrow the native selected proof's binding.
    reference=dict(zip(('path','sha256','size_bytes'),identity))
    try:
        value=selected_document(reference,maximum=4*1024*1024)
    except (OSError,ValueError,TypeError,UnicodeError):
        raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
    allowance.tick()
    _require(value.get('schema_version')==schema,_REASON)
    if role=='result':
        _require(value.get('status') in _SUCCESS[family],_REASON)
    seal=record.get('canonical_digest')
    _require(type(seal) is str and any(proof.get('seal_digest')==seal for proof in selected[identity]),_REASON)
    return dict(source=dict(source),canonical_digest=seal,
                disposition='exact_selected_closed_metadata_retained',action='KEEP')


def _sam_value(identity,proofs,fresh,allowance):
    """Reselect already bound terminal bytes without borrowing a raw version."""
    allowance.tick()
    _require(proofs and all(p.get('role') in _SAM_KEYS for p in proofs),_REASON)
    roles={p['role'] for p in proofs}
    _require(len(roles)==1,_REASON)
    role=next(iter(roles))
    root=_canonical(fresh.get('planner_context',{}).get('sam_queue_root'))
    path=Path(identity[0])
    _require(path.is_relative_to(root) and len(path.relative_to(root).parts)==2,_REASON)
    state='completed' if role=='sam_jobs' else 'results'
    _require(path.parent.name==state,_REASON)
    try:
        value=selected_document(dict(zip(('path','sha256','size_bytes'),identity)),maximum=4*1024*1024)
    except (OSError,ValueError,TypeError,UnicodeError):
        raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
    allowance.tick()
    field='job_digest' if role=='sam_jobs' else 'result_digest'
    schema='task_evaluation_sam31_preparation_execution_'+('job' if role=='sam_jobs' else 'result')+'.v1'
    _require(set(value)==_SAM_KEYS[role] and value.get('schema_version')==schema
             and type(value.get('child_id')) is str and re.fullmatch('sam31-[0-9a-f]{64}',value['child_id'])
             and value.get(field)==canonical_digest(value,digest_field=field)
             and all(p.get('seal_field')==field and p.get('seal_digest')==value[field] for p in proofs),_REASON)
    names={value['child_id']+'.json'}
    if role=='sam_results':
        _require(value.get('status')=='completed',_REASON)
        names.add(value['child_id']+'.conflict-'+value[field][7:]+'.json')
    _require(path.name in names,_REASON)
    return value,role,root,field


def _current_sam_results(selected,fresh,allowance):
    """A result follows a positively owned terminal job, not arbitrary history."""
    source=fresh.get('historical_lineage',{}).get('source_family_inventory',{})
    observations=_rows(source.get('sam_observations',[]))
    jobs={}
    for row in observations:
        allowance.tick()
        _require(type(row) is dict,_REASON)
        if (row.get('role')!='sam_job' or row.get('parent_binding_verified') is not True
                or row.get('result_binding_verified') is not True):
            continue
        proofs=_rows(row.get('source_provenance'))
        _require(len(proofs)==1,_REASON)
        identity=_selector(proofs[0])
        if identity not in selected:
            continue
        value,role,_,_=_sam_value(identity,selected[identity],fresh,allowance)
        _require(role=='sam_jobs',_REASON)
        key=(value['child_id'],value['job_digest'])
        _require(key not in jobs or jobs[key]==value,_REASON)
        jobs[key]=value
    occurrences=sum(len(proofs) for proofs in selected.values())
    for row in observations:
        allowance.tick()
        if row.get('role')!='sam_result' or row.get('result_status')!='completed':
            continue
        proofs=_rows(row.get('source_provenance'))
        _require(len(proofs)==1,_REASON)
        identity=_selector(proofs[0])
        if identity in selected:
            continue
        value,role,_,_=_sam_value(identity,proofs,fresh,allowance)
        job=jobs.get((value['child_id'],value['job_digest']))
        if job is None:
            continue
        _require(role=='sam_results' and all(value[k]==job[k] for k in
                 ('parent_request_digest','plan_digest','phase'))
                 and value['source_commit']==job['expected_source_commit'],_REASON)
        occurrences+=1
        _require(occurrences<=10000,'scene_retirement_reference_limit')
        selected[identity]=[proofs[0]]


def _selected_sam(protection,selected,fresh,allowance):
    """Transfer an exact prefix-selected terminal SAM record, never raw history."""
    allowance.tick()
    _require(set(protection)=={'kind','path','raw_sha256','raw_size_bytes','scope','action'}
             and protection['scope'] in {'selected_primary_queue_states_only','preparation_sam_auxiliary_layouts_only'}
             and protection['action']=='KEEP',_REASON)
    identity=_selector(protection,'path','raw_sha256','raw_size_bytes')
    value,role,root,field=_sam_value(identity,selected.get(identity,[]),fresh,allowance)
    return dict(source={'row_path':identity[0],'raw_sha256':identity[1],'raw_size_bytes':identity[2],
                        'role':role,'queue_root':str(root)},canonical_digest=value[field],
                disposition='exact_selected_closed_metadata_retained',action='KEEP')


def validate_current_reference_transfer(fresh,allowance):
    observation=fresh.get('reference_observation')
    _require(type(observation) is dict,'scene_retirement_reference_scope_unproven')
    scopes=_rows(observation.get('child_scopes'))
    _require(len(scopes)==3 and {row.get('child') for row in scopes if type(row) is dict}==_SCOPES
             and all(type(row) is dict and row.get('complete') is True for row in scopes)
             and not observation.get('blockers'),'scene_retirement_reference_scope_unproven')
    selected=_sources(fresh,allowance)
    _current_sam_results(selected,fresh,allowance)
    records=[]
    emitted=0
    for row in _rows(observation.get('record_dispositions')):
        current=_current_record(row,selected,allowance)
        # Fixed finite source shape; bound retained duplicate framing before
        # adding it. Original full observer evidence remains in the plan.
        emitted+=1024+sum(len(current['source'][key].encode('utf-8')) for key in ('row_path','queue_root'))
        _require(emitted<=1024*1024,'scene_retirement_reference_limit')
        records.append(current)
    released=0
    auxiliaries=[]
    for protection in _rows(observation.get('protections')):
        allowance.tick()
        _require(type(protection) is dict,_REASON)
        if protection.get('kind')=='unsupported_queue_observation':
            current=_selected_sam(protection,selected,fresh,allowance)
            emitted+=1024+len(current['source']['row_path'].encode('utf-8'))
            _require(emitted<=1024*1024,'scene_retirement_reference_limit')
            auxiliaries.append(current)
            continue
        # Released observations carry retained evidence but protect no live
        # consumer. Every positive/dependent/unreleased/unknown fact still keeps.
        _require(protection.get('kind')=='pin_observation',_REASON)
        pin=protection.get('observation')
        _require(type(pin) is dict and pin.get('status')=='released'
                 and type(pin.get('released_at_epoch')) in (int,float)
                 and math.isfinite(pin['released_at_epoch']) and pin['released_at_epoch']>=0,_REASON)
        released+=1
    return dict(scope='selected_closed_metadata_transfer_only',
        transferred_records=records,transferred_record_count=len(records),
        transferred_auxiliary_records=auxiliaries,
        retained_released_pin_count=released,references_clear=False,consumer_fence_checked=False,
        mutations=0,unknown_scopes_cleared=False)
