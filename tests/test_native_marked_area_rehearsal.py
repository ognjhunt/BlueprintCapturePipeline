import copy
import pytest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_marked_area_rehearsal import marked_area_request


def test_task_change_removes_tray_and_preserves_scene_robot_and_thresholds():
    original = {"schema_version": "native_task_arena_packet_request.v1", "task_id": "old-task", "assets": [{"semantic_role": role, "source": {"sha256": role}} for role in ["scene_appearance", "scene_collision", "task_object", "task_support"]], "robot_base_pose_world": {"position_world_m": [1,2,3]}, "task_spec": {"start_pose_world": [1,2,0.286,0,0,0,1], "target_position_world_m": [1,3,0.30], "destination_position_tolerance_m": 0.005, "destination_relation": "inside", "destination_support_asset_id": "tray", "destination_orientation_xyzw": [0,0,0,1], "destination_orientation_tolerance_rad": 0.08, "minimum_lift_m": 0.1, "release_required": True, "configured_owner_authority": {"old": True}}, "scenario": {"seed": 1, "context_document": {"resolved_parameters": {}}}}
    original["task_spec"]["configured_success_criteria"] = {"drop_events_allowed": False, "forbidden_contact_classes": ["object_background", "destination_background"], "whole_subject_containment_required": True, "maximum_task_contact_force_n": 20}
    original["request_digest"] = canonical_digest(original, digest_field="request_digest")
    retained = copy.deepcopy(original)
    authority = {"schema_version": "task_evaluation_task_change_authority.v1", "new_task_id": "marked-area", "destination_kind": "marked_tabletop_area", "tray_required": False, "retain_sealed_scene_and_book": True, "authorized_by": "owner"}
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    result = marked_area_request(source_request=original, authority=authority, surface_z_m=0.275)
    assert original == retained
    assert result["assets"] == original["assets"][:3]
    assert result["robot_base_pose_world"] == original["robot_base_pose_world"]
    spec = result["task_spec"]
    assert "destination_relation" not in spec
    assert "destination_support_asset_id" not in spec
    assert spec["minimum_lift_m"] == 0.1 and spec["release_required"] is True
    assert spec["destination_position_tolerance_m"] == 0.005
    assert spec["configured_success_criteria"]["drop_events_allowed"] is False
    assert spec["configured_success_criteria"]["maximum_task_contact_force_n"] == 20
    assert spec["configured_success_criteria"]["forbidden_contact_classes"] == ["object_background"]
    assert spec["target_position_world_m"] == [1,3,0.286]
    assert spec["visible_target_marker"]["surface_position_world_m"] == [1,3,0.275]
    assert result["scenario"]["context_document"]["resolved_parameters"]["target_z_m"] == 0.286
    authority["tray_required"] = True
    with pytest.raises(ValueError, match="authority_invalid"):
        marked_area_request(source_request=original, authority=authority, surface_z_m=0.275)
