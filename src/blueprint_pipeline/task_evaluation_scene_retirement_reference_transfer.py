"""Exact completed metadata transfer; independent process admission is mandatory.

Retained queue bytes stay at their original path. Selected native member proofs
bind each accepted record; an arbitrary raw version cannot grant transfer.
Unknown/active/unresolved references keep the scene. No references-clear flag or
consumer authority is changed by this read-only evidence.
"""
import math
from pathlib import Path

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


def validate_current_reference_transfer(fresh,allowance):
    observation=fresh.get('reference_observation')
    _require(type(observation) is dict,'scene_retirement_reference_scope_unproven')
    scopes=_rows(observation.get('child_scopes'))
    _require(len(scopes)==3 and {row.get('child') for row in scopes if type(row) is dict}==_SCOPES
             and all(type(row) is dict and row.get('complete') is True for row in scopes)
             and not observation.get('blockers'),'scene_retirement_reference_scope_unproven')
    selected=_sources(fresh,allowance)
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
    for protection in _rows(observation.get('protections')):
        allowance.tick()
        _require(type(protection) is dict,_REASON)
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
        retained_released_pin_count=released,references_clear=False,consumer_fence_checked=False,
        mutations=0,unknown_scopes_cleared=False)
