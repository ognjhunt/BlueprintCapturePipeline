import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_evaluation_retained_placement_review as review
from blueprint_pipeline import task_evaluation_robot_placement_agent as agent
from blueprint_pipeline import task_evaluation_release_identity as identity
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline.robot_placement_preview_rasterizer import RENDERER
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


@pytest.mark.parametrize('verdict',['passed','uncertain'])
def test_reviewer_runs_once_without_repeating_proposal_or_overriding_veto(tmp_path,monkeypatch,verdict):
    scene={'scene':'fixture'}
    task={'robot_id':'franka_panda'}
    proposal={'candidate_id':'one','pose':{'position_world_m':[0.,0.,0.],'orientation_xyzw':[0.,0.,0.,1.]},
        'support_surface_id':'table','rationale':'Retained model proposal.','addressed_blockers':[],'uncertainty':'Native controls remain required.'}
    trajectory={'trajectory_digest':'sha256:'+'d'*64}
    inventory={'candidate_inventory_digest':'sha256:'+'e'*64,'trajectory_digest':trajectory['trajectory_digest']}
    old={'scene_binding_digest':canonical_digest(scene),'task_binding_digest':canonical_digest(task),
        'task_trajectory_digest':trajectory['trajectory_digest'],'candidate_inventory_digest':inventory['candidate_inventory_digest'],
        'rounds':[{'proposal_model':'gpt-5.6-sol','preview_images':[]}]}
    source={'execution_commit':'b'*40,'continuation_digest':'sha256:'+'f'*64,'source_placement_receipt':{'path':'retained','digest':'sha256:'+'a'*64}}
    monkeypatch.setattr(review.continuation,'validate',lambda *args,**kwargs:{'receipt':old,'proposal':proposal})
    monkeypatch.setattr(identity,'running_release_commit',lambda:'b'*40)
    monkeypatch.setattr(agent,'_reject_infeasible_orientation_slew',lambda **kwargs:kwargs['gate'])
    gate={'schema_version':'task_evaluation_robot_placement_geometry_gate.v1','status':'passed','blockers':[],
        'candidate_id':'one','declared_support_surface_id':'table',
        'support_passed':True,'collision_passed':True,'reachability_passed':True,'facing_passed':True}
    gate['geometry_gate_digest']=canonical_digest(gate,digest_field='geometry_gate_digest')
    events=[]
    monkeypatch.setattr(authority,'require_scene_execution_authority',lambda *args,**kwargs:events.append(('authority',kwargs)))
    images=[]
    for i in range(3):
        digest='sha256:'+str(i)*64
        provenance={'renderer':RENDERER,'image_digest':digest,'robot_mesh_scope':'default_prim_with_instance_proxies',
            'depth_buffer_shared_by_scene_and_robot':True}
        provenance['render_digest']=canonical_digest(provenance,digest_field='render_digest')
        p=tmp_path/f'{i}.render.json'
        p.write_text(json.dumps(provenance))
        images.append({'label':str(i),'digest':digest,'image_url':'data:image/png;base64,AA==',
            'render_provenance_path':str(p),'render_provenance_digest':provenance['render_digest']})
    class Invoker:
        def __init__(self,config):
            assert config.max_inference_cost_usd==.15
            assert config.max_input_tokens==12000 and config.max_output_tokens==4096
        def configure_reservation_audit(self,**kwargs):pass
        def invoke(self,spec,input_value):
            events.append(('invoke',spec.capability))
            assert spec.capability=='robot_placement_visual_review'
            assert spec.max_turns==1 and spec.reasoning_effort=='high'
            assert 12000*4/1_000_000+4096*20/1_000_000 < .15
            return SimpleNamespace(output=agent.RobotPlacementVisualReviewOutput(status=verdict,
                robot_supported_by_declared_surface=True,robot_not_visibly_clipping_site_geometry=True,
                robot_faces_task_workspace=True,task_workspace_visually_reachable=True,
                camera_views_are_sufficient=verdict=='passed',reason='Fixture visual verdict.',revision_guidance=[]),
                provider='openai',model='gpt-5.6-sol',sdk_version='fixture',usage={},trace_id=None)
    result=review.run(source=source,run_id='review',scene_binding=scene,task_binding=task,task_trajectory=trajectory,
        inventory=inventory,overview_images=[],output_dir=tmp_path,artifacts=[],validate_candidate=lambda p:gate,
        render_candidate=lambda *args:images,placement_scene_owner={},max_inference_cost_usd=.15,invoker_factory=Invoker)
    assert [e[0] for e in events]==['authority','invoke']
    assert result['new_provider_call_count']==1 and result['proposal_reexecuted'] is False
    assert 'proposal_usage' not in result['rounds'][0]
    assert result['status']==('accepted' if verdict=='passed' else 'blocked')
    if verdict=='passed':
        agent.validate_robot_placement_receipt(result)
