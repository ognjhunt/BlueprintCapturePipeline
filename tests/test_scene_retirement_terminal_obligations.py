"""Exact terminal facts stay discoverable and require preservation before action."""
import copy
import hashlib
from dataclasses import asdict
from pathlib import Path

import pytest

from blueprint_pipeline.control_plane_preparation_activation_references import (
    RawReferenceProvenance, ReferenceFact, ReferenceRecordDisposition,
)
from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance, preserve_members


def fixture(tmp_path):
    from tests.test_scene_inventory_preparations import fixture as supplied,api
    from tests.scene_lifecycle_fixture_support import rebase_graph
    digest='sha256:'+hashlib.sha256(b'x').hexdigest()
    args=supplied(request_edit=lambda v:v['task']['artifact'].update(digest=digest))
    args['seed_records']=args.pop('records')
    args=rebase_graph(args,tmp_path.resolve())
    args['records']=args.pop('seed_records')
    for role,rows in args['records'].items():
        rows=([rows] if rows else []) if role in {'intent','projection'} else rows
        for name,raw in rows:
            path=Path(name)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
    native=api().join_retained_scene_inventory_seed(**{k:args[k] for k in ('intent_id','records','roots')})
    proofs={p['role']:p for m in native['members'] for p in m['source_provenance']}
    records=[]
    sources={}
    for role,kind in (('preparation_envelopes','envelope'),('preparation_results','result')):
        p=proofs[role]
        s=RawReferenceProvenance('preparation',args['roots']['preparation_queue_root'],kind,p['path'],p['sha256'],p['size_bytes'],None)
        sources[kind]=s
        records.append(asdict(ReferenceRecordDisposition(s,'supported','supplied_integrity_only',p['seal_digest'])))
    root=Path(args['roots']['preparation_input_root'])/'prep-1'
    root.mkdir(parents=True,exist_ok=True)
    target=root/digest[7:]
    target.write_bytes(b'x')
    remote='https://example.test/input?q=1'
    local=ReferenceFact(sources['result'],'task.artifact','request_bound','exact_supplied_request_binding',
        'declared_materialized_raw_bytes',digest,str(target),None,1,(sources['envelope'],))
    remote_fact=ReferenceFact(sources['envelope'],'task.artifact','declared_remote_raw','supplied_reference_only',
        'remote_raw_bytes',digest,None,remote,1)
    intent_path,intent_raw=args['records']['intent']
    import json
    intent=json.loads(intent_raw)
    canonical=ReferenceFact(sources['envelope'],'scene_intent_digest','selector_only','declared_document_selector',
        'canonical_document_seal',intent['intent_digest'])
    protections=[{'kind':kind,'observation':asdict(fact),'action':'KEEP'} for kind,fact in (
        ('local_path_protections',local),('remote_raw_references',remote_fact),
        ('canonical_document_selector_obligations',canonical))]
    fresh={'selected_intent_provenance':{'role':'intent','path':intent_path,
        'sha256':'sha256:'+hashlib.sha256(intent_raw).hexdigest(),'size_bytes':len(intent_raw),
        'seal_field':'intent_digest','seal_digest':intent['intent_digest']},
        'measured_members':[dict(row,status='observed_scoped_metadata') for row in native['members']],
        'reference_keeps':[{'member_path':str(root),'protected_path':str(target),
            'observation':protections[0],'action':'KEEP','references_clear':False}],
        'reference_observation':{'blockers':[],'child_scopes':[{'child':x,'complete':True} for x in
            ('pins','primary_queues','auxiliary_queues')],'record_dispositions':records,'protections':protections}}
    return fresh,root,ActionAllowance(expires_at=999,now=lambda:200,monotonic=lambda:0)


class Transport:
    def put_archive(self,name,chunks):
        self.raw=b''.join(chunks)
        return {'uri':'s3://fixture/'+name,'sha256':'sha256:'+hashlib.sha256(self.raw).hexdigest(),'size_bytes':len(self.raw)}
    def read_archive(self,uri):
        yield self.raw


def test_terminal_local_remote_and_canonical_facts_require_exact_archive_then_remain_verbatim(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_retirement_reference_transfer import validate_current_reference_transfer
    fresh,root,allowance=fixture(tmp_path)
    before=copy.deepcopy(fresh)
    pending=validate_current_reference_transfer(fresh,allowance)
    assert pending['archive_inventory_verified'] is False
    preserved=preserve_members([root],transport=Transport(),allowance=allowance,token='a'*32)
    verified=validate_current_reference_transfer(fresh,allowance,preserved=preserved)
    assert verified['archive_inventory_verified'] is True
    assert [row['original_obligation'] for row in verified['transferred_obligations']]==fresh['reference_observation']['protections']
    assert verified['covered_reference_keeps']==fresh['reference_keeps']
    assert fresh==before and not verified['references_clear'] and not verified['consumer_fence_checked']


@pytest.mark.parametrize('change',['source','related','path','size','digest','canonical','unbound','unselected','blocker','archive'])
def test_terminal_facts_cannot_borrow_foreign_missing_or_contradictory_proof(tmp_path,change):
    from blueprint_pipeline.task_evaluation_scene_retirement_reference_transfer import validate_current_reference_transfer
    fresh,root,allowance=fixture(tmp_path)
    preserved=preserve_members([root],transport=Transport(),allowance=allowance,token='a'*32)
    fact=fresh['reference_observation']['protections'][0]['observation']
    if change=='source':
        fact['source']['raw_sha256']='sha256:'+'a'*64
    elif change=='related':
        fact['related_sources'][0]['row_path']=str(root/'foreign')
    elif change=='path':
        fact['path']=str(root/'absent')
    elif change=='size':
        fact['size_bytes']=2
    elif change=='digest':
        fact['digest']='sha256:'+'a'*64
    elif change=='canonical':
        fresh['reference_observation']['protections'][2]['observation']['digest']='sha256:'+'a'*64
    elif change=='unbound':
        fact['binding_status']='receipt_only'
    elif change=='unselected':
        fresh['measured_members']=[]
    elif change=='blocker':
        fresh['reference_observation']['blockers']=['supported_reference_invalid']
    else:
        preserved['files'][0]['sha256']='sha256:'+'a'*64
    with pytest.raises(ValueError,match='scene_retirement_reference_'):
        validate_current_reference_transfer(fresh,allowance,preserved=preserved)
