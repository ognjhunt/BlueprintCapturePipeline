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


def original_sam_transfer(tmp_path):
    from tests.test_scene_source_family_adoption import fixture
    from tests.test_scene_source_family_website import api
    from tests.scene_lifecycle_fixture_support import rebase_graph
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from blueprint_pipeline.task_evaluation_scene_lifecycle_measurement import members
    args=rebase_graph(fixture(through='standard_splat_conversion'),tmp_path.resolve())
    for group in ('seed_records','downstream_records','source_records'):
        for role,rows in args[group].items():
            rows=([rows] if rows else []) if role in {'intent','projection'} else rows
            for name,raw in rows:
                target=Path(name); target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
    source=api().join_retained_scene_source_family_inventory(**args)
    budget=ReferenceCollectionBudget(monotonic=lambda:0)
    declared=members({'declared_lexical_members':[],'source_family_inventory':source},budget,
        RetainedEmissionBudget(max_bytes=100000,max_rows=100,max_references=100),args['roots'])
    proof=next(p for rows in declared.values() for row in rows for p in row['source_provenance']
               if p['role']=='sam_results')
    protection={'kind':'unsupported_queue_observation','path':proof['path'],'raw_sha256':proof['sha256'],
        'raw_size_bytes':proof['size_bytes'],'scope':'selected_preparation_sam_auxiliaries_only','action':'KEEP'}
    fresh={'planner_context':{'sam_queue_root':args['roots']['sam_queue_root']},
        'historical_lineage':{'source_family_inventory':source},
        'measured_members':[{'source_provenance':row['source_provenance']} for rows in declared.values() for row in rows],
        'reference_observation':{'blockers':[],'child_scopes':[{'child':name,'complete':True}
            for name in ('pins','primary_queues','auxiliary_queues')],
            'record_dispositions':[],'protections':[protection]}}
    return fresh,proof,ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)


def test_exact_original_sam_result_selected_by_verified_prefix_transfers_with_all_evidence_retained(tmp_path):
    fresh,proof,allowance=original_sam_transfer(tmp_path)
    before=copy.deepcopy(fresh)
    result=check(fresh,allowance)
    assert result['transferred_auxiliary_records'][0]['source']['row_path']==proof['path']
    assert result['references_clear'] is False and fresh==before


@pytest.mark.parametrize('change',['raw_drift','unselected_copy','failed','future_schema','active_job'])
def test_sam_raw_versions_active_or_failed_records_cannot_borrow_prefix_membership(tmp_path,change):
    import json
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    fresh,proof,allowance=original_sam_transfer(tmp_path)
    record=fresh['reference_observation']['protections'][0]
    if change=='unselected_copy':
        target=Path(proof['path']).with_name('foreign.json'); target.write_bytes(Path(proof['path']).read_bytes())
        record['path']=str(target)
    elif change=='raw_drift':
        Path(proof['path']).write_bytes(b'{}')
    else:
        if change=='active_job':
            candidate=next(p for row in fresh['measured_members'] for p in row['source_provenance'] if p['role']=='sam_jobs')
            target=Path(candidate['path']).parent.parent/'pending'/Path(candidate['path']).name
            target.parent.mkdir(exist_ok=True); target.write_bytes(Path(candidate['path']).read_bytes())
            candidate['path']=str(target)
            record.update(path=str(target),raw_sha256=candidate['sha256'],raw_size_bytes=candidate['size_bytes'])
        else:
            path=Path(proof['path']); value=json.loads(path.read_bytes())
            value['status']='failed' if change=='failed' else value['status']
            if change=='future_schema': value['schema_version']='future_result.v2'
            value['result_digest']=canonical_digest(value,digest_field='result_digest')
            raw=json.dumps(value,sort_keys=True).encode(); path.write_bytes(raw)
            digest='sha256:'+__import__('hashlib').sha256(raw).hexdigest()
            for row in fresh['measured_members']:
                for selected in row['source_provenance']:
                    if selected['path']==str(path): selected.update(sha256=digest,size_bytes=len(raw),seal_digest=value['result_digest'])
            record.update(raw_sha256=digest,raw_size_bytes=len(raw))
    with pytest.raises(ValueError,match='scene_retirement_reference_'):
        check(fresh,allowance)
