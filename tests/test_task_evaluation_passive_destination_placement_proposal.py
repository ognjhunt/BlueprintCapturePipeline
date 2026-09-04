from __future__ import annotations

import copy
import math

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_passive_destination_placement_proposal import (
    PassiveDestinationPlacementProposalError,
    derive_passive_destination_placement_proposal,
)


SUPPORT = {
    "schema_version": "task_evaluation_support_plane_input.v1",
    "status": "frozen_candidate_pending_production_validation",
    "scene_id": "841757",
    "publisher_instance_id": "85",
    "publisher_label": "TV cabinet",
    "sage_prim_path": "/Root/_J6IMDBVAV27YPTUKI888888",
    "top_z_m": 0.275,
    "bounds_min_xyz_m": [-2.2168, -4.2363, 0.0],
    "bounds_max_xyz_m": [-1.8418, 0.9322, 0.275],
}
SUBJECT = {
    "schema_version": "task_evaluation_source_object_selection.v1",
    "status": "frozen_before_scene_configuration_run",
    "publisher_instance_id": "115",
    "aabb_min_xyz_m": [-2.1516, -3.6400, 0.2755],
    "aabb_max_xyz_m": [-1.8563, -3.2423, 0.29664],
    "center_xyz_m": [-2.00395, -3.44115, 0.28607],
}
TRAY_STATIC = {
    "schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
    "status": "authored_structure_statically_qualified",
    "replacement_identity": {"id": "document-tray", "version": "v2"},
    "observed_structure": {
        "collision_bounds_body_frame_m": {
            "minimum": [-0.165, -0.24, 0.0],
            "maximum": [0.165, 0.24, 0.035],
        },
        "rigid_body_paths": ["/Asset"],
        "collision_prim_paths": ["/Asset/Colliders/Bottom"],
    },
}


def _proposal(**overrides):
    inputs = dict(
        support_plane=SUPPORT,
        subject_selection=SUBJECT,
        destination_identity={"id": "document-tray", "version": "v2"},
        destination_static_qualification=TRAY_STATIC,
        clearance_gap_m=0.05,
        support_edge_margin_m=0.02,
    )
    inputs.update(overrides)
    return derive_passive_destination_placement_proposal(**inputs)


def test_tray_is_placed_on_the_support_beside_the_subject_along_the_long_axis() -> None:
    proposal = _proposal()
    assert proposal["schema_version"] == "task_evaluation_passive_destination_placement_proposal.v1"
    assert proposal["proposal_digest"] == canonical_digest(proposal, digest_field="proposal_digest")
    pose = proposal["pose_world"]
    # Rests on the support top, centered across the cabinet's short (x) axis.
    assert pose["position_world_m"][2] == pytest.approx(0.275)
    assert pose["position_world_m"][0] == pytest.approx((-2.2168 + -1.8418) / 2.0)
    # The tray's long body axis (y, 0.48 m) already runs along the cabinet's long
    # axis (y), so no yaw is needed.
    assert pose["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    # Along y the tray sits beyond the book plus the authored gap, on the side
    # with more free cabinet length (+y).
    expected_y = -3.2423 + 0.05 + 0.24
    assert pose["position_world_m"][1] == pytest.approx(expected_y)
    assert proposal["derivation"]["long_axis"] == "y"
    assert proposal["derivation"]["side"] == "positive"
    footprint = proposal["footprint_world_m"]
    assert footprint["minimum"][0] >= SUPPORT["bounds_min_xyz_m"][0] + 0.02 - 1e-9
    assert footprint["maximum"][0] <= SUPPORT["bounds_max_xyz_m"][0] - 0.02 + 1e-9
    assert footprint["minimum"][1] >= SUBJECT["aabb_max_xyz_m"][1] + 0.05 - 1e-9
    probe = proposal["native_probe"]
    assert probe["schema_version"] == "task_evaluation_rigid_destination_native_probe_configuration.v1"
    assert probe["placement_support_scene_prim_paths"] == [SUPPORT["sage_prim_path"]]
    assert probe["settle_sample_count"] == 3
    assert proposal["claim_boundary"]["robot_reachability_established"] is False


def test_tray_falls_back_to_the_other_side_when_the_preferred_side_is_short() -> None:
    support = copy.deepcopy(SUPPORT)
    support["bounds_max_xyz_m"][1] = -3.0  # only 24 cm free on +y
    proposal = _proposal(support_plane=support)
    assert proposal["derivation"]["side"] == "negative"
    expected_y = -3.6400 - 0.05 - 0.24
    assert proposal["pose_world"]["position_world_m"][1] == pytest.approx(expected_y)


def test_tray_rotates_when_the_support_long_axis_is_x() -> None:
    support = copy.deepcopy(SUPPORT)
    support["bounds_min_xyz_m"] = [-4.2363, -2.2168, 0.0]
    support["bounds_max_xyz_m"] = [0.9322, -1.8418, 0.275]
    subject = copy.deepcopy(SUBJECT)
    subject["aabb_min_xyz_m"] = [-3.6400, -2.1516, 0.2755]
    subject["aabb_max_xyz_m"] = [-3.2423, -1.8563, 0.29664]
    subject["center_xyz_m"] = [-3.44115, -2.00395, 0.28607]
    proposal = _proposal(support_plane=support, subject_selection=subject)
    assert proposal["derivation"]["long_axis"] == "x"
    quaternion = proposal["pose_world"]["orientation_xyzw"]
    assert math.isclose(sum(v * v for v in quaternion), 1.0, abs_tol=1e-9)
    assert abs(quaternion[2]) == pytest.approx(math.sin(math.pi / 4.0))
    assert proposal["pose_world"]["position_world_m"][1] == pytest.approx((-2.2168 + -1.8418) / 2.0)


@pytest.mark.parametrize(
    ("mutate", "blocker"),
    [
        (
            lambda inputs: inputs.update(support_edge_margin_m=0.03),
            "passive_destination_placement_footprint_exceeds_support_width",
        ),
        (
            lambda inputs: inputs["support_plane"].update(bounds_max_xyz_m=[-1.8418, -3.0, 0.275], bounds_min_xyz_m=[-2.2168, -3.9, 0.0]),
            "passive_destination_placement_no_free_support_length",
        ),
        (
            lambda inputs: inputs["subject_selection"].update(status="observed_after_episode"),
            "passive_destination_placement_subject_selection_invalid",
        ),
        (
            lambda inputs: inputs["destination_static_qualification"].update(replacement_identity={"id": "document-tray", "version": "v1"}),
            "passive_destination_placement_identity_mismatch",
        ),
        (
            lambda inputs: inputs["support_plane"].update(top_z_m=0.5),
            "passive_destination_placement_support_top_invalid",
        ),
    ],
)
def test_placement_proposal_fails_closed(mutate, blocker) -> None:
    inputs = dict(
        support_plane=copy.deepcopy(SUPPORT),
        subject_selection=copy.deepcopy(SUBJECT),
        destination_identity={"id": "document-tray", "version": "v2"},
        destination_static_qualification=copy.deepcopy(TRAY_STATIC),
        clearance_gap_m=0.05,
        support_edge_margin_m=0.02,
    )
    mutate(inputs)
    with pytest.raises(PassiveDestinationPlacementProposalError, match=blocker):
        derive_passive_destination_placement_proposal(**inputs)
