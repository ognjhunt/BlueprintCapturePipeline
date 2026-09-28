"""Terminal compiler queue bytes remain at their selected original addresses.

Native owned membership and adapter/compiler metadata joins are exercised here;
this boundary does not establish current process or global reference closure.
"""
import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
from tests.test_scene_retirement_reference_transfer import check


def fixture(tmp_path):
    from tests.test_scene_compilation_native_owner import fixture as supplied
    from tests.test_scene_compilation_owner_preparations import api
    from tests.scene_lifecycle_fixture_support import rebase_graph
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from blueprint_pipeline.task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from blueprint_pipeline.task_evaluation_scene_lifecycle_measurement import members
    args=rebase_graph(supplied(),tmp_path.resolve())
    path,raw=args['downstream_records']['compilation_envelopes'][0]
    args['downstream_records']['compilation_envelopes']=[(path.replace('/pending/','/completed/'),raw)]
    for group in ('seed_records','downstream_records','source_records','bridge_records'):
        for role,rows in args[group].items():
            rows=([rows] if rows else []) if role in {'intent','projection'} else rows
            for name,raw in rows:
                target=Path(name)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw)
    native=api().join_retained_scene_compilation_native_owner_inventory(**args)
    declared=members(native,ReferenceCollectionBudget(monotonic=lambda:0),
        RetainedEmissionBudget(max_bytes=1000000,max_rows=1000,max_references=1000),args['roots'])
    proofs={proof['role']:proof for rows in declared.values() for row in rows for proof in row['source_provenance']
            if proof['role'] in {'compilation_envelopes','compilation_results'}}
    assert set(proofs)=={'compilation_envelopes','compilation_results'}
    observed=[row for row in native['compilation_native_owner_observations'] if row.get('kind')=='compilation_output']
    assert observed[0]['adapter_metadata_binding_verified'] is True
    assert observed[0]['compiler_output_metadata_binding_verified'] is True
    protections=[dict(kind='unsupported_queue_observation',path=proof['path'],raw_sha256=proof['sha256'],
        raw_size_bytes=proof['size_bytes'],scope='selected_primary_queue_states_only',action='KEEP')
        for proof in proofs.values()]
    fresh={'planner_context':{'roots':args['roots']},'historical_lineage':native,
        'measured_members':[{'source_provenance':row['source_provenance']} for rows in declared.values() for row in rows],
        'reference_observation':{'blockers':[],'child_scopes':[{'child':name,'complete':True}
            for name in ('pins','primary_queues','auxiliary_queues')],'record_dispositions':[],'protections':protections}}
    return fresh,proofs,ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)


def test_selected_finished_compilation_bytes_transfer_with_native_binding_and_original_provenance(tmp_path):
    fresh,proofs,allowance=fixture(tmp_path)
    original=copy.deepcopy(fresh)
    result=check(fresh,allowance)
    assert {row['source']['row_path'] for row in result['transferred_auxiliary_records']}=={p['path'] for p in proofs.values()}
    assert fresh==original and result['references_clear'] is False and result['consumer_fence_checked'] is False


@pytest.mark.parametrize('change',['active','foreign_copy','raw_drift','unselected','adapter_unproven','compiler_unproven'])
def test_unproven_or_changed_compilation_cannot_borrow_a_terminal_selected_version(tmp_path,change):
    fresh,proofs,allowance=fixture(tmp_path)
    protection=fresh['reference_observation']['protections'][0]
    proof=proofs['compilation_envelopes']
    if change in {'active','foreign_copy'}:
        old=Path(proof['path'])
        target=(old.parent.parent/'pending'/old.name) if change=='active' else old.with_name('foreign.json')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(old.read_bytes())
        protection.update(path=str(target),raw_sha256=proof['sha256'],raw_size_bytes=proof['size_bytes'])
        if change=='active':
            for member in fresh['measured_members']:
                for source in member['source_provenance']:
                    if source['path']==str(old):
                        source['path']=str(target)
    elif change=='raw_drift':
        Path(proof['path']).write_bytes(b'{}')
    elif change=='unselected':
        for member in fresh['measured_members']:
            member['source_provenance']=[p for p in member['source_provenance'] if p['role']!='compilation_envelopes']
    else:
        for row in fresh['historical_lineage']['compilation_native_owner_observations']:
            if row.get('kind')=='compilation_output':
                row[('adapter' if change=='adapter_unproven' else 'compiler_output')+'_metadata_binding_verified']=False
    with pytest.raises(ValueError,match='scene_retirement_reference_'):
        check(fresh,allowance)


@pytest.mark.parametrize('change',['failed','future_schema','future_field'])
def test_known_failed_future_or_extended_selected_compilation_remains_protected(tmp_path,change):
    import hashlib
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    fresh,proofs,allowance=fixture(tmp_path)
    proof=proofs['compilation_results']
    path=Path(proof['path'])
    value=json.loads(path.read_bytes())
    if change=='failed':
        value['status']='blocked'
    elif change=='future_schema':
        value['schema_version']='future_compilation.v2'
    else:
        value['unknown_retention']='requires-owner'
    value['result_digest']=canonical_digest(value,digest_field='result_digest')
    raw=json.dumps(value,sort_keys=True).encode()
    path.write_bytes(raw)
    digest='sha256:'+hashlib.sha256(raw).hexdigest()
    for member in fresh['measured_members']:
        for source in member['source_provenance']:
            if source['path']==str(path):
                source.update(sha256=digest,size_bytes=len(raw),seal_digest=value['result_digest'])
    for protection in fresh['reference_observation']['protections']:
        if protection['path']==str(path):
            protection.update(raw_sha256=digest,raw_size_bytes=len(raw))
    with pytest.raises(ValueError,match='scene_retirement_reference_'):
        check(fresh,allowance)
