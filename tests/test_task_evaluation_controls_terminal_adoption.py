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


def test_delivered_scene_is_automatically_reprovisioned_without_reconstruction(reserved, tmp_path, monkeypatch):
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
    # Preserve the actual intent. Resolve its absent fixture binding-id key to
    # the one catalog row without changing owner evidence.
    pointer=tmp_path/'project-current.json'
    put(pointer,{'path':str(seed(tmp_path))})
    binding={'expected_production_commit':'c'*40,'runtime_source_payload_dir':str(payload),
        'runtime_digest':worker.payload_digest(payload),'robot_asset_usd':asset('robot.usd'),
        'embodiment_camera_template':asset('camera.json'),'phase_hard_cap_usd':.45,
        'project_spend_current_path':str(pointer),'openai_project_id':'project','openai_api_key_id':'key'}
    catalog={'bindings':{owner_doc['request']['task'].get('robot_binding_id'):binding}}
    monkeypatch.setattr(worker, '_configured_scene_preparation_link', lambda **kw: {'result_filename':'retained-preparation.json'})
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
    args=observed[0]
    assert args['configuration_source_commit']=='d'*40 and args['expected_production_commit']=='c'*40
    assert args['configuration_adoption']['terminal_result_digest']==terminal['result_digest']
    assert args['phase_hard_cap_usd']==.45 and args['max_inference_cost_usd']==2.56
    assert all(v['scene_intent_digest']==owner['intent_digest'] for v in args['scene_phase_attempts'].values())
    assert len(list((root/owner['intent_id']/'attempts').glob('*.json')))==7
    assert adoption.provision_terminal_controls_adoption(config=config,catalog=catalog,intent_id=owner['intent_id'],expected_production_commit='c'*40,now=103)==result
    assert len(observed)==1
    with pytest.raises(ValueError,match='authority_expired'):
        adoption.provision_terminal_controls_adoption(config=config,catalog=catalog,intent_id=owner['intent_id'],expected_production_commit='c'*40,now=1001)
