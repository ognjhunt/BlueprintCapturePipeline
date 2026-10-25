from copy import deepcopy

import pytest

from blueprint_pipeline.task_object_required_motions import (
    assembly_mounting, merge_assembly_mounting, merge_required_motions,
    motion_graph_readback, required_motions,
)


def test_mounting_is_evidence_bound_and_conflicting_reviews_remain_unknown():
    fixed = {"mode": "fixed_in_scene", "source_frame_ids": ["installed"],
             "reason": "The base is installed within a fixed surround."}
    free = {"mode": "free", "source_frame_ids": ["moving"],
            "reason": "The base moves independently."}
    assert assembly_mounting(None) is None
    assert assembly_mounting(fixed, source_frame_ids=["installed"]) == fixed
    for bad in ({**fixed, "source_frame_ids": []}, {**fixed, "mode": []}):
        with pytest.raises(ValueError, match="assembly_mounting_invalid"):
            assembly_mounting(bad)
    with pytest.raises(ValueError, match="assembly_mounting_invalid"):
        assembly_mounting(fixed, source_frame_ids=["other"])
    assert merge_assembly_mounting([fixed, free])["mode"] == "unknown"
    unknown = {"mode": "unknown", "source_frame_ids": [], "reason": "Mount is not visible."}
    assert merge_assembly_mounting([unknown, fixed]) == fixed
    assert merge_assembly_mounting([unknown])["mode"] == "unknown"


def motion(name, parts, kind="prismatic", frames=None):
    return {"motion_id": name, "part_ids": parts, "joint_type": kind,
            "task_actions": ["extend", "retract"], "source_frame_ids": frames or []}


def graph():
    def joint(name, parent, child, kind, role):
        return {"joint_id": name, "parent_link_id": parent, "child_link_id": child,
                "joint_type": kind, "role": role, "limits": [0., 1.] if kind != "fixed" else [0., 0.]}
    return {"links": [{"link_id": name} for name in ["base", "lid", "handle", "upper_tray", "lower_tray"]],
            "joints": [joint("lid_hinge", "base", "lid", "revolute", "target"),
                       joint("handle_mount", "lid", "handle", "fixed", "locked"),
                       joint("upper_slide", "base", "upper_tray", "prismatic", "passive"),
                       joint("lower_slide", "base", "lower_tray", "prismatic", "passive")]}


def test_each_requested_motion_needs_a_distinct_exported_carrier_and_fixed_features_inherit_it():
    requests = [motion("open_lid", ["lid", "handle"], "revolute"),
                motion("slide_upper", ["upper_tray"]), motion("slide_lower", ["lower_tray"])]
    readback = motion_graph_readback(requests, graph())
    assert {r["joint_id"] for r in readback["motions"]} == {"lid_hinge", "upper_slide", "lower_slide"}
    assert readback["task_success_proven"] is False and readback["physics_authority_granted"] is False


@pytest.mark.parametrize("defect", ["fixed", "locked", "wrong_kind", "no_travel"])
def test_named_mesh_parts_do_not_satisfy_a_required_motion(defect):
    asset = graph()
    row = asset["joints"][2]
    if defect == "fixed":
        row.update(joint_type="fixed", role="locked", limits=[0., 0.])
    elif defect == "locked":
        row["role"] = "locked"
    elif defect == "wrong_kind":
        row["joint_type"] = "revolute"
    else:
        row["limits"] = [0., 0.]
    with pytest.raises(ValueError, match="required_task_motion_missing:slide_upper"):
        motion_graph_readback([motion("slide_upper", ["upper_tray"])], asset)


def test_two_requested_independent_parts_cannot_share_one_slider():
    asset = graph()
    asset["joints"][3].update(parent_link_id="upper_tray", joint_type="fixed", role="locked", limits=[0., 0.])
    with pytest.raises(ValueError, match="independent_task_motions_share_joint:upper_slide"):
        motion_graph_readback([motion("upper", ["upper_tray"]), motion("lower", ["lower_tray"])], asset)


def test_visual_part_ids_bind_to_their_actual_exported_body_without_name_guessing():
    assert motion_graph_readback([motion("pull", ["tray_face"])], graph(),
        part_links={"tray_face": "upper_tray"})["motions"][0]["joint_id"] == "upper_slide"
    with pytest.raises(ValueError, match="motion_part_missing:tray_face"):
        motion_graph_readback([motion("pull", ["tray_face"])], graph())


def test_batch_merge_keeps_all_actions_parts_and_evidence_without_changing_joint_type():
    before = [motion("lid_motion", ["lid"], "revolute", ["first"])]
    after = [motion("lid_motion", ["handle"], "revolute", ["second"]), motion("tray_motion", ["upper_tray"])]
    snapshot = deepcopy(before)
    merged = merge_required_motions(before, after)
    assert merged[0]["part_ids"] == ["handle", "lid"]
    assert merged[0]["source_frame_ids"] == ["first", "second"] and before == snapshot
    after[0]["joint_type"] = "prismatic"
    with pytest.raises(ValueError, match="motion_type_changed"):
        merge_required_motions(before, after)


def test_requirement_evidence_and_part_identity_are_bound_before_authoring():
    row = motion("slide", ["upper_tray"], frames=["frame"])
    assert required_motions([row], known_parts={"upper_tray"}, source_frame_ids={"frame"})
    for kwargs in ({"known_parts": {"other"}}, {"source_frame_ids": {"other"}}):
        with pytest.raises(ValueError, match="required_motions_invalid"):
            required_motions([row], **kwargs)
