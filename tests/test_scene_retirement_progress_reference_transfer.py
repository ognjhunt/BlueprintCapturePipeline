"""Transfer only exact owned durable final SAM metadata, with both seals retained."""
import copy
from pathlib import Path

import pytest

from tests.test_scene_retirement_reference_transfer import check
from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance


def fixture(tmp_path):
    from tests.test_scene_source_family_sam import durable_final_fixture
    from tests.test_scene_source_family_website import api
    from tests.scene_lifecycle_fixture_support import rebase_graph
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from blueprint_pipeline.task_evaluation_scene_lifecycle_measurement import members
    args=rebase_graph(durable_final_fixture(),tmp_path.resolve())
    for group in ('seed_records','downstream_records','source_records'):
        for role,rows in args[group].items():
            rows=([rows] if rows else []) if role in {'intent','projection'} else rows
            for path,raw in rows:
                target=Path(path)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw)
    native=api().join_retained_scene_source_family_inventory(**args)
    declared=members({'declared_lexical_members':[],'source_family_inventory':native},
        ReferenceCollectionBudget(monotonic=lambda:0),RetainedEmissionBudget(max_bytes=1000000,max_rows=1000,max_references=1000),args['roots'])
    final=next(row for row in native['sam_observations'] if row['role']=='sam_final')
    assert final['parent_binding_verified'] is True
    proof=next(row for row in native['sam_observations'] if row['role']=='sam_source_progress')['source_provenance'][0]
    protection={'kind':'unsupported_queue_observation','path':proof['path'],'raw_sha256':proof['sha256'],
        'raw_size_bytes':proof['size_bytes'],'scope':'preparation_sam_auxiliary_layouts_only','action':'KEEP'}
    fresh={'planner_context':{'roots':args['roots']},'historical_lineage':{'source_family_inventory':native},
        'measured_members':[{'source_provenance':row['source_provenance']} for rows in declared.values() for row in rows],
        'reference_observation':{'blockers':[],'child_scopes':[{'child':name,'complete':True}
            for name in ('pins','primary_queues','auxiliary_queues')],'record_dispositions':[],'protections':[protection]}}
    return fresh,proof,final,ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)


def test_native_owned_ready_progress_retains_whole_raw_and_distinct_nested_final_seals(tmp_path):
    fresh,proof,final,allowance=fixture(tmp_path)
    before=copy.deepcopy(fresh)
    result=check(fresh,allowance)
    transferred=result['transferred_auxiliary_records'][0]
    assert transferred['source']['row_path']==proof['path']
    assert transferred['canonical_digest']==proof['seal_digest']
    assert transferred['nested_final_digest']==final['source_provenance'][0]['seal_digest']
    assert transferred['nested_final_digest']!=transferred['canonical_digest']
    assert fresh==before and result['references_clear'] is False


@pytest.mark.parametrize('change',['unowned_job','unbound_final','missing_result','raw_drift','foreign_copy'])
def test_partial_or_unbound_durable_progress_cannot_borrow_raw_history(tmp_path,change):
    fresh,proof,final,allowance=fixture(tmp_path)
    if change=='unowned_job':
        for member in fresh['measured_members']:
            member['source_provenance']=[p for p in member['source_provenance'] if p['role']!='sam_jobs']
    elif change=='unbound_final':
        final['parent_binding_verified']=False
    elif change=='missing_result':
        for member in fresh['measured_members']:
            member['source_provenance']=[p for p in member['source_provenance'] if p['role']!='sam_results']
        for row in fresh['historical_lineage']['source_family_inventory']['sam_observations']:
            if row['role']=='sam_job':
                row['result_binding_verified']=False
    elif change=='raw_drift':
        Path(proof['path']).write_bytes(b'{}')
    else:
        target=Path(proof['path']).with_name('foreign.json')
        target.write_bytes(Path(proof['path']).read_bytes())
        fresh['reference_observation']['protections'][0]['path']=str(target)
    with pytest.raises(ValueError,match='scene_retirement_reference_'):
        check(fresh,allowance)
