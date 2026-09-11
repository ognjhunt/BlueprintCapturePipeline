"""Offline geometry rebinding never fabricates qualification or relaxes scoring."""
import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError, file_record, save_json
from blueprint_pipeline.task_object_astra_native_adoption import (
    BOOK_DIMENSIONS_M, _asset_evidence, _geometry_task_spec,
)
from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import _subject_bounds_in_scoring_frame


def test_geometry_rebinding_fixes_bottom_center_and_scoring_duplicates_preserving_limits():
    source = {
        'start_pose_world': [-2.0039529,-3.441138,0.2860687,0,0,0,1],
        'target_position_world_m': [-2.0292786,-2.9522899860558796,.294500000291],
        'configured_success_criteria': {'target_center_xyz_m':[-2.0292786,-2.9522899860558796,.2905687],
            'per_cell_controls_required':False,'maximum_task_contact_force_n':20,'minimum_lift_m':.1},
        'destination_position_bounds_world_m': {
            'minimum':[-2.0342786,-2.9572899860558796,.2805687],
            'maximum':[-2.0242786,-2.9472899860558796,.3005687]},
        'interaction_affordance': {'contact_point_scoring_frame_m':[0,-.198848014,0],
            'allowed_contact_prim_paths':['/Asset'],'pregrasp_clearance_m':.08},
        'minimum_lift_m':.1,'release_required':True,'release_gripper_width_min_m':.06,
        'maximum_retries':0,'maximum_regrasps':0,'retreat_clearance_m':.1,
        'collision_failure_minimum_force_n':1,'settle_window_samples':20,
        'destination_position_tolerance_m':.005,'destination_orientation_tolerance_rad':.08,
        'task_success_contract':{'old':True},'task_success_contract_digest':'old',
        'configured_task_source_documents_digest':'old-asset-native-import',
    }
    source['success_criteria'] = copy.deepcopy(source['configured_success_criteria'])
    before = copy.deepcopy(source)
    support = .27650001156143844
    book_pose = {'position_world_m':[-2.0039529,-3.441138,support+.002], 'orientation_xyzw':[0,0,0,1]}
    tray_pose = {'position_world_m':[-2.0292786,-2.9522899860558796,support+.002], 'orientation_xyzw':[0,0,0,1]}
    half = BOOK_DIMENSIONS_M[2]/2
    low,high = _subject_bounds_in_scoring_frame(bounds={
        'minimum':[-BOOK_DIMENSIONS_M[0]/2,-BOOK_DIMENSIONS_M[1]/2,0],
        'maximum':[BOOK_DIMENSIONS_M[0]/2,BOOK_DIMENSIONS_M[1]/2,BOOK_DIMENSIONS_M[2]]},
        transform={'position_m':[0,0,half], 'orientation_xyzw':[0,0,0,1]})
    target_z = support+.002+.005+half
    geometry = {'subject_collision_bounds_scoring_frame_m':{'minimum':low,'maximum':high},
        'destination_position_bounds_destination_frame_m':{'minimum':[-.012347999,-.036151986,.004+half],
                                                          'maximum':[.012347999,.036151986,.035-half]},
        'destination_interior_bounds_body_frame_m':{'minimum':[-.16,-.235,.004],'maximum':[.16,.235,.035]},
        'support_height_interval_m':[target_z-.003,target_z+.003], 'intended_support_prim_paths':['/Asset']}
    result = _geometry_task_spec(source,book_pose=book_pose,tray_pose=tray_pose,geometry=geometry,
                                 support_top_z_m=support,authority_digest='sha256:'+'a'*64)
    assert source == before
    assert result['start_pose_world'][2] == pytest.approx(.28906871156143843)
    assert result['target_position_world_m'][2] == pytest.approx(.29406871156143843)
    assert result['destination_pose_world'][2] == pytest.approx(.27850001156143844)
    for field in ('configured_success_criteria','success_criteria'):
        assert result[field]['target_center_xyz_m'] == result['target_position_world_m']
        assert result[field]['per_cell_controls_required'] is False
    assert result['interaction_affordance']['asset_root_from_scoring_frame']['position_m'] == [0,0,half]
    assert result['interaction_affordance']['support_alignment']['native_settling_qualified'] is False
    assert result['interaction_affordance']['affordance_digest'] == canonical_digest(result['interaction_affordance'],digest_field='affordance_digest')
    for field in ('minimum_lift_m','release_required','release_gripper_width_min_m','maximum_retries',
                  'maximum_regrasps','retreat_clearance_m','collision_failure_minimum_force_n',
                  'settle_window_samples','destination_position_tolerance_m','destination_orientation_tolerance_rad'):
        assert result[field] == before[field]
    assert 'task_success_contract' not in result  # Must be explicitly rederived, never reused.
    assert 'configured_task_source_documents_digest' not in result
    assert result['destination_position_bounds_world_m']['maximum'][2] == pytest.approx(target_z+.01)


def test_unqualified_standin_asset_cannot_be_adopted(tmp_path):
    asset = tmp_path/'candidate.usdz'
    asset.write_bytes(b'Unqualified stand-in; this is not USD or qualified evidence.')
    package = tmp_path/'package.json'
    static = tmp_path/'static.json'
    save_json(package, {'asset':file_record(asset),'claim_ceiling':'development_only','native_qualified':False})
    save_json(static, {'status':'candidate_pending_qualification'})
    with pytest.raises(AssetAuthoringError,match='static_evidence_invalid'):
        _asset_evidence(package,static,BOOK_DIMENSIONS_M)
    asset.write_bytes(b'changed bytes')
    with pytest.raises(AssetAuthoringError,match='asset_digest_mismatch'):
        _asset_evidence(package,static,BOOK_DIMENSIONS_M)
