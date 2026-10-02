"""Exact selected native bundle local proof boundary."""
import copy
import hashlib
import json

import pytest

from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance

def _native_bundle_fact(tmp_path):
    """One sealed result, its selected revision, and a preserved bundle byte."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_scene_retirement_reference_proofs import TerminalProofs

    root=tmp_path/'inputs'
    bundle=root/'native-prep'/'bundle.zip'
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b'bundle')
    digest='sha256:'+hashlib.sha256(bundle.read_bytes()).hexdigest()
    remote={'uri':'s3://test/bundle.zip','digest':digest,'size_bytes':bundle.stat().st_size}
    revision={'schema_version':'task_evaluation_configured_scene_revision.v1',
              'status':'configured','configured_scene_bundle':remote}
    revision['revision_digest']=canonical_digest(revision,digest_field='revision_digest')
    result={'schema_version':'task_evaluation_launch_preparation_result.v1',
            'preparation_id':'native-prep','configured_scene_revision_digest':revision['revision_digest'],
            'configured_scene_bundle_digest':digest,'references':[{**remote,
                'contract_path':'scene.configured_revision.configured_scene_bundle',
                'materialized_path':str(bundle),'content_addressed_reuse':False,
                'full_byte_service_account_readback_passed':True}]}
    result['result_digest']=canonical_digest(result,digest_field='result_digest')
    def write(path,value,role,seal_field):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True))
        raw=path.read_bytes()
        return {'path':str(path),'sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),
                'size_bytes':len(raw),'role':role,'seal_field':seal_field,
                'seal_digest':value[seal_field]}
    revision_proof=write(tmp_path/'metadata'/'revision.json',revision,'configured_revisions','revision_digest')
    result_proof=write(tmp_path/'queue'/'results'/'native-prep.json',result,'native_preparation_results','result_digest')
    source={'family':'preparation','queue_root':str(tmp_path/'queue'),'role':'result',
            'row_path':result_proof['path'],'raw_sha256':result_proof['sha256'],
            'raw_size_bytes':result_proof['size_bytes'],'observed_identity':None}
    selected={tuple(proof[key] for key in ('path','sha256','size_bytes')):[proof]
              for proof in (revision_proof,result_proof)}
    fresh={'planner_context':{'roots':{'preparation_input_root':str(root)}},
           'historical_lineage':{'compilation_native_owner_inventory':{
               'preparation_handoff_observations':[{'pre_handoff_binding_verified':True,
                   'source_provenance':[revision_proof,result_proof]}]}}}
    preserved={'members':[{'path':str(bundle.parent)}],
               'files':[{'member_index':0,'relative_path':'bundle.zip',
                         'sha256':digest,'size_bytes':bundle.stat().st_size}]}
    fact={'source':source,'contract_path':'scene.configured_revision.configured_scene_bundle',
          'binding_status':'receipt_only','reason':'deferred_parent_reference_proof',
          'digest_meaning':'declared_materialized_raw_bytes','digest':digest,'path':str(bundle),
          'uri':None,'size_bytes':bundle.stat().st_size,'related_sources':[]}
    protection={'kind':'local_path_protections','observation':fact,'action':'KEEP'}
    allowance=ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)
    return TerminalProofs(fresh,selected,[{'source':source}],allowance,preserved),protection


def test_selected_native_bundle_receipt_only_local_fact_requires_archive_identity(tmp_path):
    proofs,protection=_native_bundle_fact(tmp_path)
    proofs.transfer(protection)
    assert proofs.transferred[0]['preservation_proof']['kind']=='selected_native_bundle_local_bytes'
    assert proofs.covered


def test_same_selected_result_provenance_in_two_measured_members_is_not_a_second_source(tmp_path):
    proofs, protection = _native_bundle_fact(tmp_path)
    source = protection['observation']['source']
    identity = (source['row_path'], source['raw_sha256'], source['raw_size_bytes'])
    proofs.selected[identity].append(copy.deepcopy(proofs.selected[identity][0]))
    proofs.transfer(protection)
    assert proofs.transferred[0]['preservation_proof']['kind'] == 'selected_native_bundle_local_bytes'


@pytest.mark.parametrize('change',['fact_digest','fact_path','fact_size','result_readback',
                                    'revision_uri','duplicate_handoff','missing_archive',
                                    'missing_inventory','other_receipt_only','wrong_result_role'])
def test_native_bundle_local_fact_refuses_unproved_byte_or_source(tmp_path,change):
    proofs,protection=_native_bundle_fact(tmp_path)
    fact=protection['observation']
    source=(fact['source']['row_path'],fact['source']['raw_sha256'],fact['source']['raw_size_bytes'])
    revision=next(identity for identity,value in proofs.raw.items()
        if value.get('schema_version')=='task_evaluation_configured_scene_revision.v1')
    if change=='fact_digest':
        fact['digest']='sha256:'+'f'*64
    elif change=='fact_path':
        fact['path']=str(tmp_path/'foreign'/'bundle.zip')
    elif change=='fact_size':
        fact['size_bytes']+=1
    elif change=='result_readback':
        proofs.raw[source]['references'][0]['full_byte_service_account_readback_passed']=False
    elif change=='revision_uri':
        proofs.raw[revision]['configured_scene_bundle']['uri']='s3://test/other.zip'
    elif change=='duplicate_handoff':
        rows=proofs.fresh['historical_lineage']['compilation_native_owner_inventory'][
            'preparation_handoff_observations']
        rows.append(copy.deepcopy(rows[0]))
    elif change=='missing_archive':
        proofs.physical.pop(fact['path'])
    elif change=='missing_inventory':
        proofs.has_inventory=False
    elif change=='other_receipt_only':
        fact['contract_path']='construction.recipe'
    elif change=='wrong_result_role':
        proofs.selected[source][0]['role']='preparation_results'
    with pytest.raises(ValueError,match='scene_retirement_reference_closure_unproven'):
        proofs.transfer(protection)
