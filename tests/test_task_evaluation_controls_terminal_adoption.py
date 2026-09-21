import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_controls_terminal_adoption as adoption
from blueprint_pipeline import task_evaluation_controls_autoprovision as worker
from blueprint_pipeline import task_evaluation_configured_controls_continuation_provisioning as producer
from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as progression
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_unstarted_controls_reservations as cancellation
from tests.test_task_evaluation_unstarted_controls_reservations import reserved as reserved, put
from tests.test_task_evaluation_scene_spend import seed


@pytest.mark.parametrize("catalog_bucket", [None, "blueprint", "artifact-store"])
def test_delivered_scene_is_automatically_reprovisioned_without_reconstruction(reserved, tmp_path, monkeypatch, catalog_bucket):
    root, run, owner, _reserve, _old = reserved
    config = {'scene_root':str(root), 'launch_state_root':str(tmp_path), 'trusted_clients':['webapp'],
        'controls_root':str(tmp_path/'controls'), 'intent_root':str(tmp_path/'registry'), 'profile_dir':str(tmp_path/'profiles'),
        'preparation_queue_root':str(tmp_path/'preparations'), 'service_group':None}
    assert adoption.terminal_adoption_source(config=config, intent_id=owner['intent_id'], expected_production_commit='c'*40) is None
    cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root)
    assert adoption.terminal_adoption_source(config=config, intent_id=owner['intent_id'], expected_production_commit='c'*40) is None
    receipt = json.loads((run/'launch_receipt.json').read_text())
    receipt['status'] = 'completed'
    put(run/'launch_receipt.json', receipt)
    put(run/'webapp_sync_succeeded.json', {'sync_result_digest':'sha256:'+'5'*64})
    terminal = {'result_digest':'sha256:'+'1'*64, 'configured_scene_revision_digest':'sha256:'+'2'*64, 'publication_result_digest':'sha256:'+'3'*64}
    monkeypatch.setattr(progression, '_validate_source', lambda p: (terminal,receipt,{'provider_zero_receipt_digest':'sha256:'+'4'*64}))
    payload = tmp_path/'runtime'
    payload.mkdir()
    (payload/'runner.py').write_text('pass\n')
    def asset(name):
        p=tmp_path/name
        p.write_text(name)
        return {'path':str(p),'digest':'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
    request_path=root/owner['intent_id']/'intent.json'
    owner_doc=intake._read(request_path,'intent_digest')
    # Preserve the actual intent; grant its omitted operational robot explicitly.
    pointer=tmp_path/'project-current.json'
    put(pointer,{'path':str(seed(tmp_path))})
    binding={'expected_production_commit':'c'*40,'runtime_source_payload_dir':str(payload),
        'runtime_digest':worker.payload_digest(payload),'robot_asset_usd':asset('robot.usd'),
        'embodiment_camera_template':asset('camera.json'),'phase_hard_cap_usd':.45,
        'project_spend_current_path':str(pointer),'openai_project_id':'project','openai_api_key_id':'key'}
    binding['external_layer_bucket'] = catalog_bucket
    monkeypatch.setattr(producer, '_live_external_layer_bucket', lambda: 'artifact-store')
    catalog=worker._seal({'schema_version':worker.CATALOG_SCHEMA,
        'bindings':{'fixture-franka':binding}}, 'catalog_digest')
    from blueprint_pipeline import task_evaluation_scene_robot_assignment as assignment
    from tests.test_scene_robot_assignment import authorize
    catalog_path=tmp_path/'assignment-catalog.json'
    put(catalog_path,catalog)
    auth=authorize(tmp_path,owner_doc,catalog,binding_id='fixture-franka')
    assigned=assignment.assign_scene_robot(queue_root=root,intent_id=owner['intent_id'],
        intent_digest=owner['intent_digest'],owner=owner_doc['request']['owner'],authenticated_client='webapp',
        trusted_clients={'webapp'},robot_catalog_path=catalog_path,robot_binding_id='fixture-franka',
        authorization_reference=auth,ack=assignment.ACK,now=102)
    monkeypatch.setattr(worker, '_configured_scene_preparation_link', lambda **kw: {'result_filename':'retained-preparation.json'})
    monkeypatch.setattr(producer, '_preparation_context', lambda **kw: {})
    observed=[]
    def provision(**kwargs):
        observed.append(kwargs)
        p=Path(kwargs['controls_root'])/'intent.json'
        put(p,{'fixture':True})
        return {'status':'configured_controls_continuation_provisioned','intent_path':str(p),'provider_mutation_performed':False}
    monkeypatch.setattr(producer, 'provision_configured_controls_continuation', provision)
    monkeypatch.setattr(producer, 'install_intent_into_registry', lambda **kw: {'status':'installed'})
    result=adoption.provision_terminal_controls_adoption(config=config,catalog=catalog,intent_id=owner['intent_id'],expected_production_commit='c'*40,now=102)
    assert result['status']=='installed_terminal_adoption'
    assert result['robot_assignment_digest']==assigned['assignment_digest']
    args=observed[0]
    assert args['external_layer_bucket'] == 'artifact-store'
    assert binding['external_layer_bucket'] == catalog_bucket  # Never rewrite the selected binding.
    assert args['configuration_source_commit']=='d'*40 and args['expected_production_commit']=='c'*40
    assert args['configuration_adoption']['terminal_result_digest']==terminal['result_digest']
    assert args['phase_hard_cap_usd']==.45 and args['max_inference_cost_usd']==2.56
    assert all(v['scene_intent_digest']==owner['intent_digest'] for v in args['scene_phase_attempts'].values())
    assert len(list((root/owner['intent_id']/'attempts').glob('*.json')))==7
    assert adoption.provision_terminal_controls_adoption(config=config,catalog=catalog,intent_id=owner['intent_id'],expected_production_commit='c'*40,now=103)==result
    assert len(observed)==1
    with pytest.raises(ValueError,match='authority_expired'):
        adoption.provision_terminal_controls_adoption(config=config,catalog=catalog,intent_id=owner['intent_id'],expected_production_commit='c'*40,now=1001)


def _second_launch_cancellations(root, owner, reserve, *, launch_id, commit='c'):
    """Three valid unstarted-controls cancellations bound to another blocked launch."""
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
    directory = root / owner['intent_id']
    launch = {'schema_version': 'task_evaluation_launch_receipt.v1', 'launch_id': launch_id, 'status': 'blocked',
              'source_commit': commit * 40, 'launch_profile_digest': 'sha256:' + '9' * 64,
              'terminal_evidence': {'status': 'blocked'}}
    launch['receipt_digest'] = cross_runtime_canonical_digest(launch, digest_field='receipt_digest')
    for name, cost, provider in [('construction', .45, 'vast'), ('controls', .45, 'vast'), ('placement', 2.56, 'openai')]:
        attempt = reserve(f'controls-{launch_id}-{name}', cost, provider, commit=commit)
        receipt = {'schema_version': cancellation.SCHEMA, 'status': 'cancelled_before_controls_eligibility',
                   'attempt_id': attempt['attempt_id'], 'attempt_digest': attempt['attempt_digest'],
                   'intent_digest': attempt['intent_digest'], 'maximum_spend_usd': attempt['maximum_spend_usd'],
                   'provider': attempt['provider'], 'original_blocked_launch_receipt': launch,
                   'source_profile': {'path': 'x', 'digest': 'sha256:' + '8' * 64},
                   'source_autostart_intent': {'path': 'y', 'digest': 'sha256:' + '7' * 64},
                   'downstream_execution_eligible': False, 'provider_mutation_performed': False}
        receipt['receipt_digest'] = canonical_digest(receipt, digest_field='receipt_digest')
        put(directory / cancellation.DIRECTORY / (attempt['attempt_id'] + '.json'), receipt)
    return launch


def test_multiple_blocked_launches_are_judged_per_launch(reserved, tmp_path, monkeypatch):
    """2026-09-13: two blocked launches retired six controls rows and the whole intent was
    refused with terminal_adoption_retired_scope_invalid, so no continuation registry entry
    was ever provisioned for the next attempt and its activation waited forever."""
    root, run, owner, reserve, _originals = reserved
    config = {'scene_root': str(root), 'launch_state_root': str(tmp_path)}
    cancellation.cancel_unstarted_controls_reservations(launch_root=run, scene_root=root)
    _second_launch_cancellations(root, owner, reserve, launch_id='launch-two')
    assert adoption.terminal_adoption_source(config=config, intent_id=owner['intent_id'],
                                             expected_production_commit='c' * 40) is None
    receipt = json.loads((run / 'launch_receipt.json').read_text())
    receipt['status'] = 'completed'
    put(run / 'launch_receipt.json', receipt)
    put(run / 'webapp_sync_succeeded.json', {'sync_result_digest': 'sha256:' + '5' * 64})
    terminal = {'result_digest': 'sha256:' + '1' * 64, 'configured_scene_revision_digest': 'sha256:' + '2' * 64,
                'publication_result_digest': 'sha256:' + '3' * 64}
    monkeypatch.setattr(progression, '_validate_source',
                        lambda p: (terminal, receipt, {'provider_zero_receipt_digest': 'sha256:' + '4' * 64}))
    source = adoption.terminal_adoption_source(config=config, intent_id=owner['intent_id'],
                                               expected_production_commit='c' * 40)
    assert source is not None and source['launch_id'] == 'launch'
    assert sorted(a['attempt_id'] for a in source['retired_attempts']) == [
        'controls-first-construction', 'controls-first-controls', 'controls-first-placement']
    # A launch whose trio is incomplete is still a scope error.
    extra = _second_launch_cancellations(root, owner, reserve, launch_id='launch-three')
    directory = root / owner['intent_id']
    (directory / cancellation.DIRECTORY / 'controls-launch-three-placement.json').unlink()
    with pytest.raises(ValueError, match='terminal_adoption_retired_scope_invalid'):
        adoption.terminal_adoption_source(config=config, intent_id=owner['intent_id'],
                                          expected_production_commit='c' * 40)
    assert extra['launch_id'] == 'launch-three'
