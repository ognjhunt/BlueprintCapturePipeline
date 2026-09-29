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
from .task_evaluation_scene_compilation_owner_contracts import FIELD_SETS

_SCOPES={'pins','primary_queues','auxiliary_queues'}
_ROLE_SCHEMAS={
    ('preparation','identity'):('task_evaluation_launch_preparation_identity.v1',{'identities'}),
    ('preparation','envelope'):('task_evaluation_launch_preparation_envelope.v1',{'completed','materialized'}),
    ('activation','identity'):('task_evaluation_launch_activation_identity.v1',{'identities'}),
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
            if proof['role'] in {'preparation_identity','activation_identity','queue_identities'}:
                # Identity rows only establish an exact envelope pair below;
                # their seals cannot satisfy generic selector obligations.
                continue
            identity=_selector(proof)
            # Native measurement provenance is selected membership evidence.
            # Raw_versions/unselected observations are deliberately not indexed.
            result.setdefault(identity,[]).append(proof)
    lineage=fresh.get('historical_lineage',{})
    native=lineage.get('compilation_native_owner_inventory',lineage)
    observations=_rows(native.get('preparation_handoff_observations',[]))
    fields={'configured_revisions':'revision_digest',
            'compilation_intake_receipts':'receipt_digest'}
    for row in observations:
        allowance.tick()
        if type(row) is not dict or row.get('pre_handoff_binding_verified') is not True:
            continue
        for proof in _rows(row.get('source_provenance',[])):
            allowance.tick()
            role=proof.get('role') if type(proof) is dict else None
            if role not in fields:
                continue
            occurrences+=1
            _require(occurrences<=10000 and proof.get('seal_field')==fields[role]
                     and type(proof.get('seal_digest')) is str,_REASON)
            identity=_selector(proof)
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
    if role=='identity':
        id_field=family+'_id'
        _require(set(value)=={'schema_version',id_field,'request_digest','identity_digest'}
                 and type(value.get(id_field)) is str
                 and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}',value[id_field])
                 and path.name==value[id_field]+'.json'
                 and type(value.get('request_digest')) is str
                 and re.fullmatch('sha256:[0-9a-f]{64}',value['request_digest'])
                 and value.get('identity_digest')==canonical_digest(value,digest_field='identity_digest'),_REASON)
    seal=record.get('canonical_digest')
    _require(type(seal) is str and any(proof.get('seal_digest')==seal for proof in selected[identity]),_REASON)
    return (dict(source=dict(source),canonical_digest=seal,
                 disposition='exact_selected_closed_metadata_retained',action='KEEP'),value)


def _retained_identity(record,envelopes,fresh,consent,allowance):
    """A queue identity is inert retained metadata, never selected member proof."""
    allowance.tick()
    _require(type(record) is dict and record.get('disposition')=='supported',_REASON)
    source=record.get('source')
    _require(type(source) is dict and set(source)=={'family','queue_root','role','row_path',
        'raw_sha256','raw_size_bytes','observed_identity'} and source['role']=='identity',_REASON)
    family=source['family']
    roots=fresh.get('planner_context',{}).get('roots',{})
    root=roots.get(family+'_queue_root')
    _require(family in {'preparation','activation'} and type(root) is str
        and source['queue_root']==root,_REASON)
    identity=_selector(source,'row_path','raw_sha256','raw_size_bytes')
    path=Path(identity[0])
    _require(0<identity[2]<=65536 and path.parent==Path(root)/'identities',_REASON)
    if consent is not None:
        _require(all(not path.is_relative_to(Path(member['canonical_path']))
            for member in consent['members']),_REASON)
    reference=dict(zip(('path','sha256','size_bytes'),identity))
    allowance.charge('local_bytes',identity[2])
    try:
        value=selected_document(reference,maximum=65536)
    except (OSError,ValueError,TypeError,UnicodeError):
        raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
    id_field=family+'_id'
    _require(set(value)=={'schema_version',id_field,'request_digest','identity_digest'}
        and value.get('schema_version')=='task_evaluation_launch_'+family+'_identity.v1'
        and type(value.get(id_field)) is str
        and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}',value[id_field])
        and path.name==value[id_field]+'.json'
        and type(value.get('request_digest')) is str
        and re.fullmatch(r'sha256:[0-9a-f]{64}',value['request_digest'])
        and value.get('identity_digest')==canonical_digest(value,digest_field='identity_digest')
        and record.get('canonical_digest')==value['identity_digest'],_REASON)
    key=(family,root,value[id_field],value['request_digest'])
    _require(len(envelopes.get(key,[]))==1,_REASON)
    return (dict(source=dict(source),canonical_digest=value['identity_digest'],
        disposition='exact_observed_identity_retained_only',action='KEEP'),value)


def _sam_value(identity,proofs,fresh,allowance):
    """Reselect already bound terminal bytes without borrowing a raw version."""
    allowance.tick()
    _require(proofs and all(p.get('role') in _SAM_KEYS for p in proofs),_REASON)
    roles={p['role'] for p in proofs}
    _require(len(roles)==1,_REASON)
    role=next(iter(roles))
    roots=fresh.get('planner_context',{}).get('roots')
    _require(type(roots) is dict and type(roots.get('sam_queue_root')) is str,_REASON)
    root=_canonical(roots['sam_queue_root'])
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
    return jobs


def _progress_value(identity,proof,fresh,allowance):
    """Whole durable document identity differs from its nested final seal."""
    allowance.tick()
    roots=fresh.get('planner_context',{}).get('roots')
    _require(type(roots) is dict and type(roots.get('preparation_queue_root')) is str,_REASON)
    root=_canonical(roots['preparation_queue_root'])
    path=Path(identity[0])
    _require(path.is_relative_to(root) and len(path.relative_to(root).parts)==3
        and path.parent.parent.name=='source-progress',_REASON)
    try:
        value=selected_document(dict(zip(('path','sha256','size_bytes'),identity)),maximum=4*1024*1024)
    except (OSError,ValueError,TypeError,UnicodeError):
        raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
    keys={'schema_version','preparation_id','request_digest','run_id','source_commit','status','sequence',
          'previous_progress_digest','advancement','provider_mutation_performed','paid_execution_requested','progress_digest'}
    _require(keys<=set(value)<=keys|{'resume_signal_digest'} and value.get('schema_version')=='task_evaluation_sam31_preparation_progress.v1'
        and value.get('status')=='ready' and type(value.get('sequence')) is int and 1<=value['sequence']<=999999
        and type(value.get('preparation_id')) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}',value['preparation_id'])
        and type(value.get('request_digest')) is str and re.fullmatch('sha256:[0-9a-f]{64}',value['request_digest'])
        and value.get('progress_digest')==canonical_digest(value,digest_field='progress_digest')
        and proof.get('role')=='source_progress' and proof.get('seal_field')=='progress_digest'
        and proof.get('seal_digest')==value['progress_digest']
        and value.get('provider_mutation_performed') is False and value.get('paid_execution_requested') is False,_REASON)
    _require(path.parent.name==value['preparation_id']+'-'+value['request_digest'][7:]
        and path.name==f"{value['sequence']:06d}-"+value['progress_digest'][7:]+'.json',_REASON)
    advancement=value['advancement']
    fields={'status','evidence_refs','sam31_exact_mask_inputs','sam31_preparation_result'}
    _require(type(advancement) is dict and fields<=set(advancement)<=fields|{'human_review_required','candidate_policy_queried'}
        and advancement['status']=='ready' and all(advancement.get(k,False) is False for k in ('human_review_required','candidate_policy_queried')),_REASON)
    final=advancement['sam31_preparation_result']
    fields={'schema_version','status','source_commit','plan_digest','evidence','stage_result_receipts','result_digest'}
    _require(type(final) is dict and fields<=set(final)<=fields|{'completed_prefix_adoption','review_kind','human_review_required','candidate_policy_queried'}
        and final.get('schema_version')=='task_evaluation_sam31_preparation_result.v1'
        and final.get('status')=='exact_mask_inputs_ready' and final.get('source_commit')==value['source_commit']
        and final.get('result_digest')==canonical_digest(final,digest_field='result_digest')
        and all(final.get(k,False) is False for k in ('human_review_required','candidate_policy_queried'))
        and final.get('review_kind','ai')=='ai',_REASON)
    return value,final,root


def _current_sam_progress(selected,fresh,jobs,allowance):
    source=fresh.get('historical_lineage',{}).get('source_family_inventory',{})
    observations=_rows(source.get('sam_observations',[]))
    owners={(job['parent_request_digest'],job['plan_digest'],job['expected_source_commit']) for job in jobs.values()}
    finals={}
    for row in observations:
        allowance.tick()
        if row.get('role')!='sam_final' or row.get('parent_binding_verified') is not True:
            continue
        proofs=_rows(row.get('source_provenance'))
        _require(len(proofs)==1,_REASON)
        proof=proofs[0]
        _require(proof.get('role')=='source_progress' and proof.get('json_pointer')=='/advancement/sam31_preparation_result',_REASON)
        identity=_selector(proof)
        _require(len(finals)<10000 or identity in finals,'scene_retirement_reference_limit')
        finals.setdefault(identity,[]).append(proof)
    bound={}
    occurrences=sum(len(proofs) for proofs in selected.values())
    for row in observations:
        allowance.tick()
        if row.get('role')!='sam_source_progress' or row.get('progress_status')!='ready':
            continue
        proofs=_rows(row.get('source_provenance'))
        _require(len(proofs)==1,_REASON)
        proof=proofs[0]
        identity=_selector(proof)
        nested=finals.get(identity,[])
        if not nested:
            continue
        value,final,root=_progress_value(identity,proof,fresh,allowance)
        if (value['request_digest'],final['plan_digest'],value['source_commit']) not in owners:
            continue
        _require(all(p.get('seal_field')=='result_digest' and p.get('seal_digest')==final['result_digest'] for p in nested),_REASON)
        references=_rows(final['stage_result_receipts'])
        _require(len(references)<=10,_REASON)
        adoption=final.get('completed_prefix_adoption')
        if adoption is not None:
            _require(type(adoption) is dict,_REASON)
            original=_rows(adoption.get('original_phase_result_receipts'))
            _require(len(original)<=10,_REASON)
            references=[*references,*original]
        for reference in references:
            allowance.tick()
            key=_selector(reference)
            _,role,_,_=_sam_value(key,selected.get(key,[]),fresh,allowance)
            _require(role=='sam_results',_REASON)
        if identity not in selected:
            occurrences+=1
            _require(occurrences<=10000,'scene_retirement_reference_limit')
            selected[identity]=[proof]
        _require(len(bound)<10000 or identity in bound,'scene_retirement_reference_limit')
        bound[identity]=(value,final,root)
    return bound


def _bound_compilations(fresh,selected,allowance):
    """Index selected successful output joins once; raw history never qualifies."""
    result={}
    for row in _rows(fresh.get('historical_lineage',{}).get('compilation_native_owner_observations',[])):
        allowance.tick()
        _require(type(row) is dict,_REASON)
        if (row.get('kind')!='compilation_output' or row.get('adapter_metadata_binding_verified') is not True
                or row.get('compiler_output_metadata_binding_verified') is not True):
            continue
        proofs=_rows(row.get('source_provenance'))
        envelope_names=set()
        for proof in proofs:
            allowance.tick()
            if proof.get('role')!='compilation_envelopes':
                continue
            identity=_selector(proof)
            if identity in selected and proof.get('seal_field')=='envelope_digest':
                _require(len(envelope_names)<10000,'scene_retirement_reference_limit')
                envelope_names.add(Path(identity[0]).name)
        for proof in proofs:
            allowance.tick()
            if proof.get('role') not in {'compilation_envelopes','compilation_results'}:
                continue
            identity=_selector(proof)
            if identity not in selected or Path(identity[0]).name not in envelope_names:
                continue
            _require(len(result)<10000 or identity in result,'scene_retirement_reference_limit')
            result.setdefault(identity,set()).add(proof['role'])
    return result


def _compilation_value(identity,proofs,fresh,bound,allowance):
    allowance.tick()
    _require(proofs and identity in bound,_REASON)
    roles={proof.get('role') for proof in proofs}
    _require(len(roles)==1 and roles<=bound[identity],_REASON)
    role=next(iter(roles))
    _require(role in {'compilation_envelopes','compilation_results'},_REASON)
    roots=fresh.get('planner_context',{}).get('roots')
    _require(type(roots) is dict and type(roots.get('compilation_queue_root')) is str,_REASON)
    root=_canonical(roots['compilation_queue_root'])
    path=Path(identity[0])
    state='completed' if role=='compilation_envelopes' else 'results'
    _require(path.is_relative_to(root) and len(path.relative_to(root).parts)==2 and path.parent.name==state,_REASON)
    try:
        value=selected_document(dict(zip(('path','sha256','size_bytes'),identity)),maximum=4*1024*1024)
    except (OSError,ValueError,TypeError,UnicodeError):
        raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
    field='envelope_digest' if role=='compilation_envelopes' else 'result_digest'
    schema='task_evaluation_episode_compilation_'+('envelope' if role=='compilation_envelopes' else 'result')+'.v1'
    probe_fields={'destination_native_probe_request_path','destination_native_probe_request_digest',
                  'destination_native_probe_request_document_digest'}
    required=FIELD_SETS[role]-(probe_fields if role=='compilation_results' else set())
    _require(required<=set(value)<=FIELD_SETS[role] and value.get('schema_version')==schema
        and (not (set(value)&probe_fields) or probe_fields<=set(value))
        and type(value.get('compilation_id')) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',value['compilation_id'])
        and value.get(field)==canonical_digest(value,digest_field=field)
        and all(proof.get('seal_field')==field and proof.get('seal_digest')==value[field] for proof in proofs),_REASON)
    suffix=path.name.removeprefix(value['compilation_id']+'-').removesuffix('.json')
    _require(path.name==value['compilation_id']+'-'+suffix+'.json' and re.fullmatch('[0-9a-f]{64}',suffix),_REASON)
    if role=='compilation_envelopes':
        _require(suffix==value[field][7:] and value.get('preparation_id')==value['compilation_id']
            and all(value.get(k) is True for k in ('automatic_progression_required',
                'robot_specific_episode_packet_compiled_in_production','production_compiler_owns_episode_packet')),_REASON)
    else:
        _require(value.get('status')=='compiled_for_production_launch' and value.get('blockers')==[]
            and value.get('compiled_by_production') is True and value.get('automatic_progression_required') is True,_REASON)
    _require(all(value.get(k) is False for k in ('customer_supplied_prebuilt_episode_packet',
        'provider_mutation_performed','paid_execution_requested')),_REASON)
    return value,role,root,field


def _selected_sam(protection,selected,fresh,allowance,bound,progress):
    """Transfer an exact prefix-selected terminal SAM record, never raw history."""
    allowance.tick()
    _require(set(protection)=={'kind','path','raw_sha256','raw_size_bytes','scope','action'}
             and protection['scope'] in {'selected_primary_queue_states_only','preparation_sam_auxiliary_layouts_only'}
             and protection['action']=='KEEP',_REASON)
    identity=_selector(protection,'path','raw_sha256','raw_size_bytes')
    proofs=selected.get(identity,[])
    nested=None
    if proofs and proofs[0].get('role')=='source_progress':
        _require(identity in progress and protection['scope']=='preparation_sam_auxiliary_layouts_only',_REASON)
        value,final,root=progress[identity]
        role,field='source_progress','progress_digest'
        nested=final['result_digest']
    elif proofs and proofs[0].get('role') in {'compilation_envelopes','compilation_results'}:
        _require(protection['scope']=='selected_primary_queue_states_only',_REASON)
        value,role,root,field=_compilation_value(identity,proofs,fresh,bound,allowance)
    else:
        value,role,root,field=_sam_value(identity,proofs,fresh,allowance)
    result=dict(source={'row_path':identity[0],'raw_sha256':identity[1],'raw_size_bytes':identity[2],
                        'role':role,'queue_root':str(root)},canonical_digest=value[field],
                disposition='exact_selected_closed_metadata_retained',action='KEEP')
    if nested is not None:
        result['nested_final_digest']=nested
    return result


def _sam_marker_pairs(selected, fresh, jobs, allowance):
    """Index only exact selected completed jobs and their completed results."""
    pairs={}
    for identity,proofs in selected.items():
        allowance.tick()
        roles={proof.get('role') for proof in proofs}
        if len(roles)!=1 or not roles<=set(_SAM_KEYS):
            continue
        value,role,_,_=_sam_value(identity,proofs,fresh,allowance)
        key=(value['child_id'],value['job_digest'])
        if key not in jobs or (role=='sam_jobs' and value!=jobs[key]):
            continue
        pairs.setdefault(key,{'sam_jobs':[],'sam_results':[]})[role].append(value)
    return pairs


def _selected_sam_marker(protection,fresh,allowance,pairs,parents):
    """Retain exact worker markers without promoting their metadata to authority."""
    allowance.tick()
    _require(set(protection)=={'kind','path','raw_sha256','raw_size_bytes','scope','action'}
        and protection['kind']=='unsupported_queue_observation'
        and protection['scope']=='preparation_sam_auxiliary_layouts_only'
        and protection['action']=='KEEP',_REASON)
    identity=_selector(protection,'path','raw_sha256','raw_size_bytes')
    roots=fresh.get('planner_context',{}).get('roots',{})
    _require(type(roots) is dict and type(roots.get('sam_queue_root')) is str
        and 0<identity[2]<=4096,_REASON)
    root=_canonical(roots['sam_queue_root'])
    path=Path(identity[0])
    _require(path.is_relative_to(root) and len(path.relative_to(root).parts)==2
        and path.parent.name in {'started','wake-completed'}
        and re.fullmatch(r'sam31-[0-9a-f]{64}\.json',path.name),_REASON)
    allowance.charge('local_bytes',identity[2])
    try:
        value=selected_document(dict(zip(('path','sha256','size_bytes'),identity)),maximum=4096)
    except (OSError,ValueError,TypeError,UnicodeError):
        raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
    child=path.stem
    _require(type(value) is dict and type(value.get('job_digest')) is str
        and re.fullmatch(r'sha256:[0-9a-f]{64}',value['job_digest']),_REASON)
    key=(child,value['job_digest'])
    pair=pairs.get(key,{})
    _require(len(pair.get('sam_jobs',[]))==1 and len(pair.get('sam_results',[]))==1,_REASON)
    job=pair['sam_jobs'][0]
    result=pair['sam_results'][0]
    _require(result['status']=='completed' and result['job_digest']==job['job_digest']
        and result['child_id']==job['child_id']
        and result['parent_request_digest']==job['parent_request_digest']
        and result['source_commit']==job['expected_source_commit'],_REASON)
    if path.parent.name=='started':
        _require(set(value)=={'job_digest','child_id'} and value['child_id']==child,_REASON)
    else:
        _require(set(value)=={'status','job_digest'} and value['status']=='parent_terminal',_REASON)
        parent=(job['parent_preparation_id'],job['parent_request_digest'],job['expected_source_commit'])
        _require(parents.get(parent)==1,_REASON)
    return {'source':{'row_path':identity[0],'raw_sha256':identity[1],
                      'raw_size_bytes':identity[2],'role':path.parent.name,'queue_root':str(root)},
            'disposition':'exact_sam_marker_retained_only','action':'KEEP'}


def validate_current_reference_transfer(fresh,allowance,*,preserved=None,policy=None,consent=None,pin_journal=None):
    observation=fresh.get('reference_observation')
    _require(type(observation) is dict,'scene_retirement_reference_scope_unproven')
    scopes=_rows(observation.get('child_scopes'))
    blockers=_rows(observation.get('blockers'))
    _require(len(scopes)==3 and {row.get('child') for row in scopes if type(row) is dict}==_SCOPES
             and all(type(row) is dict and row.get('complete') is True for row in scopes)
             and all(reason in {'deferred_semantic_object','deferred_downstream_document',
                                'deferred_parent_reference_proof'}
                     for reason in blockers),
             'scene_retirement_reference_scope_unproven')
    selected=_sources(fresh,allowance)
    jobs=_current_sam_results(selected,fresh,allowance)
    progress=_current_sam_progress(selected,fresh,jobs,allowance)
    compilations=_bound_compilations(fresh,selected,allowance)
    envelope_bindings={}
    for row in _rows(observation.get('record_dispositions')):
        allowance.tick()
        source=row.get('source') if type(row) is dict else None
        if type(source) is not dict or source.get('role')!='envelope':
            continue
        current,value=_current_record(row,selected,allowance)
        family=current['source']['family']
        request=value.get('request')
        _require(type(request) is dict and value.get('request_digest')==canonical_digest(request),_REASON)
        key=(family,current['source']['queue_root'],request.get(family+'_id'),value['request_digest'])
        envelope_bindings.setdefault(key,[]).append((current,value))
    records=[]
    selected_records=[]
    read_records=[]
    emitted=0
    identity_sources=set()
    for row in _rows(observation.get('record_dispositions')):
        source=row.get('source') if type(row) is dict else None
        if type(source) is dict and source.get('role')=='identity':
            key=(source.get('family'),source.get('queue_root'),source.get('row_path'))
            _require(key not in identity_sources,_REASON)
            identity_sources.add(key)
            current,value=_retained_identity(row,envelope_bindings,fresh,consent,allowance)
        else:
            current,value=_current_record(row,selected,allowance)
            selected_records.append(current)
        # Fixed finite source shape; bound retained duplicate framing before
        # adding it. Original full observer evidence remains in the plan.
        emitted+=1024+sum(len(current['source'][key].encode('utf-8')) for key in ('row_path','queue_root'))
        _require(emitted<=1024*1024,'scene_retirement_reference_limit')
        records.append(current)
        read_records.append((current,value))
    envelopes={}
    for current,value in read_records:
        allowance.tick()
        source=current['source']
        if source['role']=='envelope':
            request=value.get('request')
            _require(type(request) is dict and value.get('request_digest')==canonical_digest(request),_REASON)
            key=(source['family'],source['queue_root'],request.get(source['family']+'_id'),value['request_digest'])
            envelopes.setdefault(key,[]).append(source)
    for identity,identity_value in (pair for pair in read_records if pair[0]['source']['role']=='identity'):
        allowance.tick()
        source=identity['source']
        key=(source['family'],source['queue_root'],identity_value[source['family']+'_id'],
             identity_value['request_digest'])
        _require(len(envelopes.get(key,[]))==1,_REASON)
    parent_results={}
    parent_root=fresh.get('planner_context',{}).get('roots',{}).get('preparation_queue_root')
    for current,value in read_records:
        allowance.tick()
        source=current['source']
        if (source['family']=='preparation' and source['role']=='result'
                and source['queue_root']==parent_root):
            parent_results.setdefault(Path(source['row_path']).name,[]).append(value)
    terminal_parents={}
    for current,value in read_records:
        allowance.tick()
        source=current['source']
        if (source['family']!='preparation' or source['role']!='envelope'
                or source['queue_root']!=parent_root
                or Path(source['row_path']).parent.name not in {'completed','materialized'}):
            continue
        request=value['request']
        matches=[result for result in parent_results.get(Path(source['row_path']).name,[])
            if result.get('preparation_id')==request.get('preparation_id')
            and result.get('source_commit')==request.get('expected_production_commit')
            and result.get('team_namespace')==request.get('team_namespace')]
        if len(matches)!=1:
            continue
        key=(request['preparation_id'],value['request_digest'],request['expected_production_commit'])
        terminal_parents[key]=terminal_parents.get(key,0)+1
    from .task_evaluation_scene_retirement_reference_proofs import TerminalProofs
    facts=TerminalProofs(fresh,selected,selected_records,allowance,preserved)
    from .task_evaluation_scene_retirement_pins import select_terminal_pins, covers
    pin_documents=dict(facts.documents)
    if consent is not None and consent.get('terminal_pin_refs'):
        for identity in compilations:
            allowance.tick()
            allowance.charge('local_bytes',identity[2])
            value,role,_,_=_compilation_value(identity,selected[identity],fresh,compilations,allowance)
            if role=='compilation_results':
                pin_documents[identity]=value
    history=None
    if pin_journal is not None:
        from .task_evaluation_scene_retirement_pin_mutation import pin_history
        history=pin_history(policy,consent,pin_journal)
    terminal_pins=select_terminal_pins(fresh,policy,consent,pin_documents,allowance,history=history) if consent is not None else []
    released=0
    auxiliaries=[]
    deferred=[]
    marker_pairs=None
    marker_sources=set()
    for protection in _rows(observation.get('protections')):
        allowance.tick()
        _require(type(protection) is dict,_REASON)
        if protection.get('kind') in {'positive_pin_path','pin_observation'} and covers(protection,terminal_pins):
            continue
        if protection.get('kind')=='unsupported_queue_observation':
            path=Path(protection.get('path',''))
            if path.parent.name in {'started','wake-completed'}:
                identity=_selector(protection,'path','raw_sha256','raw_size_bytes')
                _require(identity not in marker_sources,_REASON)
                marker_sources.add(identity)
                if marker_pairs is None:
                    marker_pairs=_sam_marker_pairs(selected,fresh,jobs,allowance)
                current=_selected_sam_marker(protection,fresh,allowance,marker_pairs,terminal_parents)
            else:
                current=_selected_sam(protection,selected,fresh,allowance,compilations,progress)
            emitted+=1024+len(current['source']['row_path'].encode('utf-8'))
            _require(emitted<=1024*1024,'scene_retirement_reference_limit')
            auxiliaries.append(current)
            continue
        if protection.get('kind') in {'local_path_protections','remote_raw_references',
            'raw_digest_selector_obligations','canonical_document_selector_obligations'}:
            facts.transfer(protection)
            continue
        if protection.get('kind')=='missing_edge_obligations':
            reason=protection.get('observation',{}).get('reason')
            if reason=='deferred_semantic_object':
                facts.transfer_inline_owner(protection)
            elif reason=='deferred_downstream_document':
                if protection.get('observation',{}).get('contract_path')=='configured_scene_bundle_digest':
                    facts.transfer_native_bundle(protection)
                else:
                    facts.transfer_native_downstream(protection)
            elif reason=='deferred_parent_reference_proof':
                if re.fullmatch(r'construction\.recipe\.stage_sequence\.[0-9]+\.configuration',
                                protection.get('observation',{}).get('contract_path','')):
                    facts.transfer_recipe_stage(protection,fresh.get('recipe_stage_authority',[]))
                else:
                    facts.transfer_native_bundle(protection)
            else:
                _require(False,_REASON)
            deferred.append(reason)
            continue
        # Released observations carry retained evidence but protect no live
        # consumer. Every positive/dependent/unreleased/unknown fact still keeps.
        _require(protection.get('kind')=='pin_observation',_REASON)
        pin=protection.get('observation')
        _require(type(pin) is dict and pin.get('status')=='released'
                 and type(pin.get('released_at_epoch')) in (int,float)
                 and math.isfinite(pin['released_at_epoch']) and pin['released_at_epoch']>=0,_REASON)
        released+=1
    _require(set(deferred)==set(blockers),_REASON)
    _require(all(parts=={'deferred_downstream_document','deferred_parent_reference_proof'}
                 for parts in facts.bundle_parts.values()),_REASON)
    recipe_authorities=fresh.get('recipe_stage_authority',[])
    _require(type(recipe_authorities) is list and len(recipe_authorities)<=16,_REASON)
    for authority in recipe_authorities:
        _require(all(((_selector(authority['result_raw_ref'])),stage['contract_path']) in facts.recipe_stage_seen
                     for stage in authority['stages']),_REASON)
    return dict(scope='selected_closed_metadata_transfer_only',
        transferred_records=records,transferred_record_count=len(records),
        transferred_auxiliary_records=auxiliaries,
        retained_released_pin_count=released,transferred_obligations=facts.transferred,
        archive_inventory_verified=facts.has_inventory,covered_reference_keeps=facts.covered_keeps(terminal_pins),terminal_pin_release_rows=terminal_pins,references_clear=False,consumer_fence_checked=False,
        recipe_stage_authority=recipe_authorities,
        mutations=0,unknown_scopes_cleared=False)
