# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_compilation_owner_preparations.py
"""Use actual remote request plus locally materialized raw revision proof."""
import copy
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_compilation_owner_preparations import api, fixture
from tests.test_scene_source_family_website import pair, ref, seal
from tests.test_scene_retirement_connected_acceptance import _rebase_complete_graph


def actual_remote_revision():
    args=fixture()
    bridge=args['bridge_records']
    old_parent=json.loads(bridge['native_preparation_envelopes'][0][1])
    request=copy.deepcopy(old_parent['request'])
    revision_path,revision_raw=bridge['configured_revisions'][0]
    local=args['roots']['preparation_input_root']+'/native-prep/revision.json'
    bridge['configured_revisions']=[(local,revision_raw)]
    args['retained_metadata_roots'].append(args['roots']['preparation_input_root'])
    remote={'uri':'s3://fixture/revision.json','digest':ref((revision_path,revision_raw))['sha256'],
            'size_bytes':len(revision_raw)}
    request['scene']['configured_revision']=remote
    request['scene']['mode']='reuse_configured_revision'
    proof=dict(contract_path='scene.configured_revision',**remote,materialized_path=local,
        content_addressed_reuse=False,full_byte_service_account_readback_passed=True)
    old_final=json.loads(bridge['native_preparation_results'][0][1])
    final=copy.deepcopy(old_final)
    final['references'].insert(0,proof)
    final['reference_count']=final['unique_object_count']=len(final['references'])
    from blueprint_pipeline.task_evaluation_scene_compilation_owner_preparations import HANDOFF,PRE
    previous={k:v for k,v in final.items() if k not in HANDOFF|{'result_digest'}}
    previous['status']=PRE
    old_comp_path,old_comp_raw=args['downstream_records']['compilation_envelopes'][0]
    old_comp=json.loads(old_comp_raw)
    comp=copy.deepcopy(old_comp)
    comp['request']=request
    comp['materialized_references']=final['references']
    comp['preparation_result_digest']=canonical_digest(previous,digest_field='result_digest')
    comp=seal(comp,'envelope_digest')
    final['episode_compilation_queue_envelope_digest']=comp['envelope_digest']
    final=seal(final,'result_digest')
    args['downstream_records']['compilation_envelopes']=[pair(old_comp_path,comp)]
    bridge['native_preparation_results']=[pair(bridge['native_preparation_results'][0][0],final)]
    maps={old_comp['envelope_digest']:comp['envelope_digest'],old_final['result_digest']:final['result_digest']}
    return _rebase_complete_graph(args,'/retained',maps,requests={old_parent['request_digest']:request})


def test_actual_remote_revision_request_joins_exact_materialized_copy_without_invented_local_request():
    args=actual_remote_revision()
    before=copy.deepcopy(args)
    result=api().join_retained_scene_compilation_native_owner_inventory(**args)
    assert result['preparation_handoff_observations'][0]['pre_handoff_binding_verified']
    assert args==before and result['cleanup_authorized'] is False
    assert any(p['role']=='configured_revisions' for p in result['raw_versions'])


@pytest.mark.parametrize('change',['raw_digest','raw_size','local_path'])
def test_available_materialized_revision_contradictions_refuse(change):
    args=actual_remote_revision()
    comp_path,comp_raw=args['downstream_records']['compilation_envelopes'][0]
    comp=json.loads(comp_raw)
    final_path,final_raw=args['bridge_records']['native_preparation_results'][0]
    final=json.loads(final_raw)
    row=final['references'][0]
    if change=='raw_digest':
        row['digest']='sha256:'+'e'*64
    elif change=='raw_size':
        row['size_bytes']+=1
    else:
        row['materialized_path']=args['roots']['preparation_input_root']+'/native-prep/foreign.json'
    comp['materialized_references']=final['references']
    from blueprint_pipeline.task_evaluation_scene_compilation_owner_preparations import HANDOFF,PRE
    previous={k:v for k,v in final.items() if k not in HANDOFF|{'result_digest'}}
    previous['status']=PRE
    comp['preparation_result_digest']=canonical_digest(previous,digest_field='result_digest')
    old=comp['envelope_digest']
    comp=seal(comp,'envelope_digest')
    final['episode_compilation_queue_envelope_digest']=comp['envelope_digest']
    args['downstream_records']['compilation_envelopes']=[pair(comp_path,comp)]
    args['bridge_records']['native_preparation_results']=[pair(final_path,seal(final,'result_digest'))]
    args=_rebase_complete_graph(args,'/retained',{old:comp['envelope_digest']})
    with pytest.raises(ValueError,match='scene_compilation_owner_'):
        api().join_retained_scene_compilation_native_owner_inventory(**args)
