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


def test_verified_native_handoff_selects_revision_and_intake_raw_targets(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement_reference_transfer import _sources
    fresh,_,allowance=fixture(tmp_path)
    selected=_sources(fresh,allowance)
    roles={proof['role'] for rows in selected.values() for proof in rows}
    assert {'configured_revisions','compilation_intake_receipts'}<=roles
    fresh['historical_lineage']['preparation_handoff_observations'][0]['pre_handoff_binding_verified']=False
    selected=_sources(fresh,allowance)
    roles={proof['role'] for rows in selected.values() for proof in rows}
    assert not {'configured_revisions','compilation_intake_receipts'} & roles


@pytest.mark.parametrize('field',[
    'configured_scene_revision_digest','episode_compilation_queue_envelope_digest',
    'episode_compilation_queue_receipt_digest'])
def test_verified_native_handoff_transfers_only_exact_selected_downstream_selector(tmp_path,field):
    fresh,allowance,raw=downstream_selector_fact(tmp_path,field)
    result=check(fresh,allowance)
    proof=result['transferred_obligations'][-1]['preservation_proof']
    assert proof['kind']=='selected_native_downstream_document'
    assert proof['canonical_digest']==json.loads(raw)[field]


def downstream_selector_fact(tmp_path,field):
    import hashlib
    fresh,_,allowance=fixture(tmp_path)
    proof=next(p for row in fresh['measured_members'] for p in row['source_provenance']
               if p.get('role')=='native_preparation_results' and p.get('seal_field')=='result_digest'
               and field in json.loads(Path(p['path']).read_bytes()))
    raw=Path(proof['path']).read_bytes()
    source={'family':'preparation','queue_root':fresh['planner_context']['roots']['preparation_queue_root'],
            'role':'result','row_path':proof['path'],'raw_sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),
            'raw_size_bytes':len(raw),'observed_identity':None}
    fresh['reference_observation']['record_dispositions']=[{'source':source,'disposition':'supported',
        'reason':'supplied_integrity_only','canonical_digest':proof['seal_digest']}]
    fact={'source':source,'contract_path':field,'binding_status':'unresolved',
          'reason':'deferred_downstream_document','digest_meaning':'no_inferred_raw_identity',
          'digest':None,'path':None,'uri':None,'size_bytes':None,'related_sources':[]}
    fresh['reference_observation']['protections'].append({'kind':'missing_edge_obligations',
        'observation':fact,'action':'KEEP'})
    fresh['reference_observation']['blockers']=['deferred_downstream_document']
    return fresh,allowance,raw


def test_active_compilation_envelope_cannot_transfer_downstream_selector(tmp_path):
    fresh,allowance,_=downstream_selector_fact(tmp_path,'episode_compilation_queue_envelope_digest')
    proof=next(p for row in fresh['measured_members'] for p in row['source_provenance']
               if p.get('role')=='compilation_envelopes')
    old=Path(proof['path'])
    active=old.parent.parent/'pending'/old.name
    active.parent.mkdir(exist_ok=True)
    old.rename(active)
    for row in fresh['measured_members']:
        for item in row['source_provenance']:
            if item.get('path')==str(old):
                item['path']=str(active)
    for row in fresh['historical_lineage']['preparation_handoff_observations']:
        for item in row['source_provenance']:
            if item.get('path')==str(old):
                item['path']=str(active)
    with pytest.raises(ValueError,match='scene_retirement_reference_'):
        check(fresh,allowance)


def test_native_bundle_digest_and_nested_materialized_row_transfer_as_one_pair(tmp_path):
    fresh,allowance,raw=downstream_selector_fact(tmp_path,'configured_scene_bundle_digest')
    nested=copy.deepcopy(fresh['reference_observation']['protections'][-1])
    nested['observation'].update(contract_path='scene.configured_revision.configured_scene_bundle',
        reason='deferred_parent_reference_proof')
    fresh['reference_observation']['protections'].append(nested)
    fresh['reference_observation']['blockers'].append('deferred_parent_reference_proof')
    result=check(fresh,allowance)
    proofs=[row['preservation_proof'] for row in result['transferred_obligations'][-2:]]
    assert [row['kind'] for row in proofs]==['selected_native_bundle','selected_native_bundle']
    assert all(row['raw_digest']==json.loads(raw)['configured_scene_bundle_digest'] for row in proofs)
    from blueprint_pipeline.task_evaluation_scene_retirement_reference_transfer import validate_current_reference_transfer
    materialized=json.loads(raw)['references'][0]['materialized_path']
    preserved={'members':[{'path':str(Path(materialized).parent)}],'files':[]}
    with pytest.raises(ValueError,match='scene_retirement_reference_closure_unproven'):
        validate_current_reference_transfer(fresh,allowance,preserved=preserved)


def test_unpaired_native_bundle_digest_remains_keep(tmp_path):
    fresh,allowance,_=downstream_selector_fact(tmp_path,'configured_scene_bundle_digest')
    with pytest.raises(ValueError,match='scene_retirement_reference_closure_unproven'):
        check(fresh,allowance)


def inline_owner_fact(tmp_path):
    import hashlib
    fresh,_,allowance=fixture(tmp_path)
    proofs=[proof for row in fresh['measured_members'] for proof in row['source_provenance']
            if proof.get('role')=='native_activation_envelopes']
    envelope=next(proof for proof in proofs if proof.get('seal_field')=='envelope_digest')
    owner=next(proof for proof in proofs if proof.get('json_pointer')=='/request/authorization/scene_owner_attempt')
    raw=Path(envelope['path']).read_bytes()
    source={'family':'activation','queue_root':fresh['planner_context']['roots']['activation_queue_root'],
            'role':'envelope','row_path':envelope['path'],'raw_sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),
            'raw_size_bytes':len(raw),'observed_identity':None}
    fresh['reference_observation']['record_dispositions']=[{'source':source,'disposition':'supported',
        'reason':'supplied_integrity_only','canonical_digest':envelope['seal_digest']}]
    fact={'source':source,'contract_path':'authorization.scene_owner_attempt','binding_status':'unresolved',
          'reason':'deferred_semantic_object','digest_meaning':'no_inferred_raw_identity',
          'digest':None,'path':None,'uri':None,'size_bytes':None,'related_sources':[]}
    fresh['reference_observation']['protections'].append({'kind':'missing_edge_obligations',
        'observation':fact,'action':'KEEP'})
    fresh['reference_observation']['blockers']=['deferred_semantic_object']
    return fresh,allowance,envelope,owner,source


def test_selected_inline_native_owner_attempt_closes_only_its_exact_deferred_fact(tmp_path):
    fresh,allowance,_,owner,_=inline_owner_fact(tmp_path)
    result=check(fresh,allowance)
    assert result['transferred_obligations'][-1]['preservation_proof']['kind']=='selected_inline_native_owner_attempt'
    assert result['transferred_obligations'][-1]['preservation_proof']['canonical_digest']==owner['seal_digest']


def test_path_bearing_inline_owner_stays_keep_even_with_current_sealed_bytes(tmp_path):
    import hashlib
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    fresh,allowance,envelope,_,source=inline_owner_fact(tmp_path)
    path=Path(envelope['path'])
    value=json.loads(path.read_bytes())
    owner=value['request']['authorization']['scene_owner_attempt']
    owner['active_path']=str(tmp_path.resolve()/'live-consumer')
    owner['owner_attempt_digest']=canonical_digest(owner,digest_field='owner_attempt_digest')
    value['request_digest']=canonical_digest(value['request'])
    value['envelope_digest']=canonical_digest(value,digest_field='envelope_digest')
    raw=json.dumps(value,sort_keys=True).encode()
    path.write_bytes(raw)
    digest='sha256:'+hashlib.sha256(raw).hexdigest()
    for row in fresh['measured_members']:
        for proof in row['source_provenance']:
            if proof.get('path')==str(path):
                proof.update(sha256=digest,size_bytes=len(raw))
                proof['seal_digest']=(owner['owner_attempt_digest'] if proof.get('json_pointer')
                                      else value['envelope_digest'])
    source.update(raw_sha256=digest,raw_size_bytes=len(raw))
    fresh['reference_observation']['record_dispositions'][0]['canonical_digest']=value['envelope_digest']
    with pytest.raises(ValueError,match='scene_retirement_reference_closure_unproven'):
        check(fresh,allowance)


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
