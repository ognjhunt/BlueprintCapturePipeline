import copy
import pytest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.native_marked_area_rehearsal import marked_area_request


def test_task_change_removes_tray_and_preserves_scene_robot_and_thresholds():
    original = {"schema_version": "native_task_arena_packet_request.v1", "task_id": "old-task", "assets": [{"semantic_role": role, "source": {"sha256": role}} for role in ["scene_appearance", "scene_collision", "task_object", "task_support"]], "robot_base_pose_world": {"position_world_m": [1,2,3]}, "task_spec": {"start_pose_world": [1,2,0.286,0,0,0,1], "target_position_world_m": [1,3,0.30], "destination_position_tolerance_m": 0.005, "destination_relation": "inside", "destination_support_asset_id": "tray", "destination_orientation_xyzw": [0,0,0,1], "destination_orientation_tolerance_rad": 0.08, "minimum_lift_m": 0.1, "release_required": True, "configured_owner_authority": {"old": True}}, "scenario": {"seed": 1, "context_document": {"resolved_parameters": {}}}}
    original["task_spec"]["configured_success_criteria"] = {"drop_events_allowed": False, "forbidden_contact_classes": ["object_background", "destination_background"], "whole_subject_containment_required": True, "maximum_task_contact_force_n": 20}
    original["task_spec"]["interaction_affordance"] = {"intended_support_prim_paths": ["/Asset"], "allowed_contact_prim_paths": ["/Book"]}
    original["task_spec"]["success_criteria"] = copy.deepcopy(original["task_spec"]["configured_success_criteria"])
    original["task_spec"]["initial_source_support"] = {"scene_prim_paths": ["/Root/Table"]}
    original["request_digest"] = canonical_digest(original, digest_field="request_digest")
    retained = copy.deepcopy(original)
    authority = {"schema_version": "task_evaluation_task_change_authority.v1", "new_task_id": "marked-area", "destination_kind": "marked_tabletop_area", "tray_required": False, "retain_sealed_scene_and_book": True, "authorized_by": "owner"}
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    result = marked_area_request(source_request=original, authority=authority, surface_z_m=0.275, support_prim_path="/Root/Table")
    assert original == retained
    assert result["assets"] == original["assets"][:3]
    assert result["robot_base_pose_world"] == original["robot_base_pose_world"]
    spec = result["task_spec"]
    assert "initial_source_support" not in spec
    assert "task_change_authority" not in result
    assert "whole_subject_containment_required" not in spec["success_criteria"]
    assert spec["success_criteria"]["forbidden_contact_classes"] == ["object_background"]
    assert "destination_relation" not in spec
    assert "destination_support_asset_id" not in spec
    assert spec["minimum_lift_m"] == 0.1 and spec["release_required"] is True
    assert spec["destination_position_tolerance_m"] == 0.005
    assert spec["interaction_affordance"]["intended_support_prim_paths"] == ["/Root/Table"]
    assert spec["interaction_affordance"]["allowed_contact_prim_paths"] == ["/Book"]
    assert spec["configured_success_criteria"]["drop_events_allowed"] is False
    assert spec["configured_success_criteria"]["maximum_task_contact_force_n"] == 20
    assert spec["configured_success_criteria"]["forbidden_contact_classes"] == ["object_background"]
    assert spec["target_position_world_m"] == [1,3,0.286]
    assert spec["visible_target_marker"]["surface_position_world_m"] == [1,3,0.275]
    assert result["scenario"]["context_document"]["resolved_parameters"]["target_z_m"] == 0.286
    authority["tray_required"] = True
    with pytest.raises(ValueError, match="authority_invalid"):
        marked_area_request(source_request=original, authority=authority, surface_z_m=0.275, support_prim_path="/Root/Table")


def test_marked_area_reseals_the_confirmed_success_contract_for_the_new_task():
    from tests.test_adp_task_scoring import _rigid_v2_spec
    from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract, validate_rigid_task_success_contract
    spec = _rigid_v2_spec()
    spec.update(target_position_world_m=[1.15,2,0.9], destination_position_tolerance_m=0.005)
    original_contract = seal_rigid_task_success_contract(task_spec=spec, site_id="site", task_id="tray-task", author_source="task_owner", author_id="owner", confirmation_status="confirmed", confirmed_by_team_id="team")
    spec["task_success_contract"] = original_contract
    request = {"schema_version": "native_task_arena_packet_request.v1", "task_id": "tray-task", "task_spec": spec, "assets": [{"semantic_role": "task_support"}], "scenario": {"seed": 1, "context_document": {"resolved_parameters": {}}}}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    authority = {"schema_version": "task_evaluation_task_change_authority.v1", "new_task_id": "marked-area", "destination_kind": "marked_tabletop_area", "tray_required": False, "retain_sealed_scene_and_book": True, "authorized_by": "owner"}
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    result = marked_area_request(source_request=request, authority=authority, surface_z_m=0.75, support_prim_path="/Root/Table")
    contract = result["task_spec"]["task_success_contract"]
    validate_rigid_task_success_contract(contract, expected_site_id="site", expected_task_id="marked-area")
    assert contract["criteria"]["motion"] == original_contract["criteria"]["motion"]
    assert contract["criteria"]["support"]["height_interval_m"] == pytest.approx([0.795,0.805])
    assert spec["task_success_contract"] == original_contract


def test_control_search_declares_many_options_without_rewriting_task():
    from blueprint_pipeline.native_marked_area_rehearsal import marked_area_control_search_request
    request = {"schema_version": "native_task_arena_packet_request.v1", "task_id": "book-to-area",
        "task_spec": {"strict": "unchanged"}, "assets": [{"semantic_role": "task_object"}],
        "robot_base_pose_world": {"position_world_m": [1,2,3], "orientation_xyzw": [0,0,0,1]},
        "robot_joint_reset_positions_rad": {f"panda_joint{i}": 0.1*i for i in range(1,8)},
        "cameras": [{"role": "external"}], "scenario": {"seed": 7}}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    original = copy.deepcopy(request)
    phase = {"phases": [{"phase_id": "pregrasp", "position_world_m": [1,2,4], "orientation_world_xyzw": [0,0,0,1]}]}
    result = marked_area_control_search_request(source_request=request, phase_plan=phase, support_prim_path="/Table")
    assert request == original
    for key in ("task_spec", "assets", "scenario", "robot_base_pose_world", "cameras"):
        assert result[key] == original[key]
    feedback = result["native_construction_feedback"]
    assert len(feedback["candidate_universe"]["candidates"]) == 16
    assert len({row["candidate_digest"] for row in feedback["candidate_universe"]["candidates"]}) == 16
    assert feedback["control_search"]["full_fidelity_replay_required"] is True
    assert feedback["control_search"]["appearance_mode"] == "omitted"
    assert feedback["allocator_retry_cap"] == 0


def test_direct_policy_omission_preserves_all_task_scoring_criteria():
    from tests.test_native_task_arena_policy_canary_session import _activation
    from blueprint_pipeline.native_marked_area_rehearsal import direct_policy_request
    original_contract = _activation()['task_success_contract']
    contract = copy.deepcopy(original_contract)
    contract['criteria']['controls'] = {'mode': 'required_per_cell',
        'control_ids': ['zero_action_negative', 'deterministic_scripted_positive']}
    contract['contract_digest'] = cross_runtime_canonical_digest(contract, digest_field='contract_digest')
    request = {'schema_version': 'native_task_arena_packet_request.v1',
        'task_spec': {'task_success_contract': contract,
            'target_position_world_m': [1.,2.,3.], 'destination_orientation_xyzw': [0.,0.,0.,1.],
            'visible_target_marker': {'schema_version': 'native_task_target_marker.v1'},
            'configured_success_criteria': {'per_cell_controls_required': True, 'minimum_lift_m': .1},
            'success_criteria': {'per_cell_controls_required': True}},
        'assets': [], 'native_construction_feedback': {'enabled': True}}
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    original = copy.deepcopy(request)
    result = direct_policy_request(source_request=request, authorized_by='owner', authorization_reference='user:skip-controls')
    assert request == original
    assert result['task_spec']['task_success_contract']['criteria'] == original_contract['criteria']
    assert result['task_spec']['destination_pose_world'] == [1.,2.,3.,0.,0.,0.,1.]
    assert result['task_spec']['configured_success_criteria']['minimum_lift_m'] == .1
    assert result['task_spec']['configured_success_criteria']['per_cell_controls_required'] is False
    assert 'native_construction_feedback' not in result
    assert result['diagnostic_control_omission_authority']['qualified_comparison_permitted'] is False
