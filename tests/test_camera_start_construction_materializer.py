"""Current native reset plus retained kinematics; old scene visibility is never adopted."""
from copy import deepcopy
import json

import pytest

from blueprint_pipeline import native_task_camera_start_configuration as camera
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_runtime import _rotation_matrix_to_xyzw
from tests.test_native_task_camera_start_configuration import fixture


def inputs():
    plan=fixture()
    source=deepcopy(plan['policy_canary_camera_start_configuration'])
    matrix=source['native_reference']['world_from_wrist_camera_opengl']
    gate={'schema_version':'policy_canary_runtime_observation_integrity_gate.v1','status':'blocked',
        'candidate_policy_queried':False,'snapshot':{'cameras':[{'role':'wrist',
        'position_world_m':[row[3] for row in matrix[:3]],
        'quaternion_world_opengl_xyzw':_rotation_matrix_to_xyzw([row[:3] for row in matrix[:3]])}]}}
    gate['gate_digest']=canonical_digest(gate,digest_field='gate_digest')
    source['native_reference']['gate_digest']=gate['gate_digest']
    source['configuration_digest']=canonical_digest(source,digest_field='configuration_digest')
    joints=source['joint_reset_positions_rad']
    base=plan['robot']['base_pose_world']
    construction={'schema_version':'native_task_arena_construction_result.v1','status':'completed',
        'construction_gate_qualified':True,'candidate_policy_queried':False,'blockers':[],
        'reset_replay':{'passed':True},'initial_readback':{'robot_root_pose_world':[
            *base['position_world_m'],*base['orientation_xyzw']],
            'robot_joint_names':list(joints),'robot_joint_positions_rad':list(joints.values())}}
    construction['result_digest']=canonical_digest(construction,digest_field='result_digest')
    return dict(plan=plan,construction=construction,source_binding=source,native_reference_gate=gate,
        robot_asset_sha256=source['source_robot_asset_sha256'],runtime_digest='sha256:'+'a'*64,
        calibration_digest='sha256:'+'b'*64)


def test_materializer_uses_current_readback_and_not_old_reference_reset():
    kwargs=inputs()
    before=deepcopy(kwargs)
    result=camera.materialize_camera_start_from_construction(**kwargs)
    assert result['joint_reset_positions_rad']==kwargs['source_binding']['joint_reset_positions_rad']
    assert result['joint_reset_positions_rad']!=kwargs['source_binding']['native_reference']['joint_reset_positions_rad']
    assert result['construction_result_digest']==kwargs['construction']['result_digest']
    assert result['native_reference']==kwargs['source_binding']['native_reference']
    assert result['native_qualification_claimed'] is result['historical_scene_visibility_adopted'] is False
    assert kwargs['native_reference_gate']['status']=='blocked'
    assert kwargs==before
    assert camera.validate_camera_start_configuration(kwargs['plan'],result)==result


@pytest.mark.parametrize('fault',['robot','gate_pose','native_base','default_reset','missing_readback','failed_reset','chain'])
def test_materializer_rejects_mismatched_native_or_calibration_evidence(fault):
    kwargs=inputs()
    if fault=='robot':
        kwargs['robot_asset_sha256']='sha256:'+'0'*64
    elif fault=='gate_pose':
        kwargs['native_reference_gate']['snapshot']['cameras'][0]['position_world_m'][0]+=.1
    elif fault=='native_base':
        kwargs['construction']['initial_readback']['robot_root_pose_world'][0]+=.1
    elif fault=='default_reset':
        kwargs['construction']['initial_readback']['robot_joint_positions_rad']=list(kwargs['source_binding']['native_reference']['joint_reset_positions_rad'].values())
    elif fault=='missing_readback':
        kwargs['construction'].pop('initial_readback')
    elif fault=='failed_reset':
        kwargs['construction']['reset_replay']['passed']=False
    else:
        kwargs['source_binding']['source_joint_chain'][0]['local0'][0][3]+=.1
        kwargs['source_binding']['configuration_digest']=canonical_digest(kwargs['source_binding'],digest_field='configuration_digest')
    kwargs['construction']['result_digest']=canonical_digest(kwargs['construction'],digest_field='result_digest')
    if fault=='gate_pose':
        gate=kwargs['native_reference_gate']
        gate['gate_digest']=canonical_digest(gate,digest_field='gate_digest')
        kwargs['source_binding']['native_reference']['gate_digest']=gate['gate_digest']
        kwargs['source_binding']['configuration_digest']=canonical_digest(kwargs['source_binding'],digest_field='configuration_digest')
    with pytest.raises((ValueError,KeyError)):
        camera.materialize_camera_start_from_construction(**kwargs)


def install_fixture_calibration(tmp_path, monkeypatch, directory, plan, construction_path):
    """Retain existing numeric calibration; exercise real private-file joins."""
    from blueprint_pipeline import task_evaluation_controls_autoprovision as controls
    from blueprint_pipeline import task_evaluation_scene_robot_assignment as assignment
    from blueprint_pipeline import task_evaluation_scene_intake as intake
    from tests.test_scene_robot_assignment import authorize
    kwargs=inputs()
    robot_path=tmp_path/'robot.usd'
    robot_path.write_bytes(b'fixture robot bytes')
    def ref(path):
        import hashlib
        return {'path':str(path),'sha256':'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest(),'size_bytes':path.stat().st_size}
    robot_ref=ref(robot_path)
    runtime=tmp_path/'runtime'
    runtime.mkdir(exist_ok=True)
    (runtime/'fixture.py').write_text('pass\n')
    cam=tmp_path/'camera-template.json'
    cam.write_text('{}')
    row={'robot_asset_usd':{'path':str(robot_path),'digest':robot_ref['sha256']},
         'embodiment_camera_template':{'path':str(cam),'digest':ref(cam)['sha256']},
         'runtime_source_payload_dir':str(runtime),'runtime_digest':controls.payload_digest(runtime),
         'phase_hard_cap_usd':2.,'openai_project_id':'project','openai_api_key_id':'key'}
    catalog=controls._seal({'schema_version':controls.CATALOG_SCHEMA,'bindings':{'fixture-franka':row}},'catalog_digest')
    catalog_path=tmp_path/'catalog.json'
    catalog_path.write_text(json.dumps(catalog))
    intent=intake._read(directory/'intent.json','intent_digest')
    auth=authorize(tmp_path,intent,catalog,binding_id='fixture-franka')
    assignment.assign_scene_robot(queue_root=directory.parent,intent_id=intent['intent_id'],intent_digest=intent['intent_digest'],
        owner=intent['request']['owner'],authenticated_client='webapp',trusted_clients={'webapp'},
        robot_catalog_path=catalog_path,robot_binding_id='fixture-franka',authorization_reference=auth,ack=assignment.ACK)
    config=tmp_path/'controls-config.json'
    config.write_text(json.dumps({'scene_root':str(directory.parent),'robot_catalog_path':str(catalog_path),'trusted_clients':['webapp']}))
    monkeypatch.setenv(controls.CONFIG_ENV,str(config))
    source=kwargs['source_binding']
    source['source_robot_asset_sha256']=robot_ref['sha256']
    source['configuration_digest']=canonical_digest(source,digest_field='configuration_digest')
    records=[]
    for name,value in [('binding',source),('gate',kwargs['native_reference_gate'])]:
        path=tmp_path/(name+'.json')
        path.write_text(json.dumps(value))
        path.chmod(0o640)
        records.append(ref(path))
    registry=tmp_path/'calibrations'
    registry.mkdir()
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_POLICY_CAMERA_CALIBRATION_ROOT',str(registry))
    calibration={'schema_version':'policy_canary_robot_camera_kinematic_calibration.v1',
        'source_robot_asset_sha256':robot_ref['sha256'],'camera_start_binding':records[0],'native_reference_gate':records[1]}
    calibration['calibration_digest']=canonical_digest(calibration,digest_field='calibration_digest')
    path=registry/(robot_ref['sha256'][7:]+'.json')
    path.write_text(json.dumps(calibration))
    path.chmod(0o640)
    # Preserve the construction result's closure identity; add its actual reset
    # fixture before the surrounding terminal envelope is sealed by the caller.
    result=json.loads(construction_path.read_bytes())
    result.update({k:kwargs['construction'][k] for k in ('initial_readback','reset_replay')})
    result['result_digest']=canonical_digest(result,digest_field='result_digest')
    construction_path.write_text(json.dumps(result))
    return kwargs['plan']
