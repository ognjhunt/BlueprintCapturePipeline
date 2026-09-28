"""Actual selected raw metadata transfer; not process/reference clearance.

These local boundary cases use a native retained preparation join and exact
current metadata bytes. They do not claim whole scene admission or retirement.
"""
import copy
from dataclasses import asdict
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance


def setup(tmp_path):
    from tests.test_scene_inventory_preparations import fixture, api
    from tests.scene_lifecycle_fixture_support import rebase_graph
    from blueprint_pipeline.control_plane_preparation_activation_references import (
        RawReferenceProvenance, ReferenceRecordDisposition,
    )
    args=fixture()
    args['seed_records']=args.pop('records')
    args=rebase_graph(args,tmp_path.resolve())
    args['records']=args.pop('seed_records')
    for role, rows in args['records'].items():
        rows=([rows] if rows else []) if role in {'intent','projection'} else rows
        for path,raw in rows:
            target=Path(path)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
    native=api().join_retained_scene_inventory_seed(**{key:args[key] for key in ('intent_id','records','roots')})
    proof=next(proof for row in native['members'] for proof in row['source_provenance']
               if proof['role']=='preparation_envelopes')
    root=args['roots']['preparation_queue_root']
    source=RawReferenceProvenance('preparation',root,'envelope',proof['path'],proof['sha256'],proof['size_bytes'],None)
    record=asdict(ReferenceRecordDisposition(source,'supported','supplied_integrity_only',proof['seal_digest']))
    fresh=dict(measured_members=[dict(row,status='observed_scoped_metadata') for row in native['members']],
        reference_observation={'blockers':[],'child_scopes':[{'child':name,'complete':True}
            for name in ('pins','primary_queues','auxiliary_queues')],
            'record_dispositions':[record],'protections':[]})
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    return fresh,record,proof,allowance


def check(fresh,allowance):
    from blueprint_pipeline.task_evaluation_scene_retirement_reference_transfer import validate_current_reference_transfer
    return validate_current_reference_transfer(fresh,allowance)


def test_exact_closed_selected_preparation_record_is_transferred_not_discarded(tmp_path):
    fresh,record,_,allowance=setup(tmp_path)
    result=check(fresh,allowance)
    assert result['transferred_record_count']==1
    assert result['transferred_records'][0]['source']==record['source']
    assert result['consumer_fence_checked'] is False and result['references_clear'] is False
    assert fresh['reference_observation']['record_dispositions']==[record]


@pytest.mark.parametrize('change',['active','foreign_copy','raw_drift','unsupported'])
def test_active_foreign_changed_or_unsupported_record_keeps_before_transfer(tmp_path,change):
    fresh,record,proof,allowance=setup(tmp_path)
    if change in {'active','foreign_copy'}:
        old=Path(proof['path'])
        target=old.parent.parent/('pending' if change=='active' else 'materialized')/old.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(old.read_bytes())
        record['source']['row_path']=str(target)
        if change=='active':
            for row in fresh['measured_members']:
                for source in row['source_provenance']:
                    if source['path']==str(old):
                        source['path']=str(target)
    elif change=='raw_drift':
        Path(proof['path']).write_bytes(b'{}')
    else:
        record['disposition']='unsupported'
    with pytest.raises(ValueError,match='scene_retirement_reference_'):
        check(fresh,allowance)


def test_released_pin_observation_is_retained_without_becoming_live_protection(tmp_path):
    fresh,_,_,allowance=setup(tmp_path)
    pin={'kind':'pin_observation','observation':{'status':'released','released_at_epoch':100,
        'paths':[str(tmp_path/'unrelated')],'depends_on':[]},'action':'KEEP'}
    fresh['reference_observation']['protections']=[pin]
    result=check(fresh,allowance)
    assert result['retained_released_pin_count']==1 and result['references_clear'] is False
    assert fresh['reference_observation']['protections']==[pin]


@pytest.mark.parametrize('kind',['positive_pin_path','missing_edge_obligations','future_protection'])
def test_positive_or_unresolved_protections_are_never_waived_by_selected_bytes(tmp_path,kind):
    fresh,_,_,allowance=setup(tmp_path)
    fresh['reference_observation']['protections']=[{'kind':kind,'path':fresh['measured_members'][0]['path']}]
    with pytest.raises(ValueError,match='scene_retirement_reference_'):
        check(fresh,allowance)


def test_missing_or_duplicate_observer_scope_cannot_be_treated_as_complete(tmp_path):
    fresh,_,_,allowance=setup(tmp_path)
    fresh['reference_observation']['child_scopes']=[{'child':'pins','complete':True}]*3
    with pytest.raises(ValueError,match='scene_retirement_reference_scope_unproven'):
        check(copy.deepcopy(fresh),allowance)
