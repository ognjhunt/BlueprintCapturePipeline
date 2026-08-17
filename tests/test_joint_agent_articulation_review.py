from __future__ import annotations

import copy
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.joint_agent_articulation_review import (
    JointAgentArticulationReviewError,
    NON_TASK_JOINT_MODE_EXEMPT,
    NON_TASK_JOINT_MODE_NO_EXERCISE,
    NON_TASK_JOINT_MODE_STRICT,
    collect_candidate_bounds,
    resolve_joint_agent_output,
    resolve_link_membership,
    review_joint_agent_articulation,
)


def _contract() -> dict:
    return {
        "minimum_assembly_joint_count": 1,
        "maximum_assembly_joint_count": 4,
        "commanded_task_joint_count": 1,
        "required_articulation_root_count": 1,
        "non_task_joint_mode": NON_TASK_JOINT_MODE_EXEMPT,
        "non_task_joint_motion_tolerance": 0.001,
        "allowed_joint_types": ["revolute", "prismatic"],
        "target_joint_type": "revolute",
        "target_axis_world": [0.0, 0.0, 1.0],
        "target_axis_absolute_dot_minimum": 0.99,
        "target_member_extent_ratio_band": [0.5, 1.5],
        "target_member_projection_constraints": [
            {
                "axis_world": [0.0, 0.0, 1.0],
                "interval_m": [0.94, 1.632],
                "minimum_overlap_fraction": 0.85,
            }
        ],
    }


_SHELL_AABB = ([-0.415, -0.35, 0.0], [0.415, 0.35, 1.9])
_UPPER_PANEL_AABB = ([-0.36, 0.17, 0.94], [0.36, 0.35, 1.632])
_LOWER_PANEL_AABB = ([-0.36, 0.17, 0.03], [0.36, 0.35, 0.92])
_UPPER_HANDLE_AABB = ([-0.34, 0.30, 1.10], [-0.28, 0.42, 1.40])
_LOWER_HANDLE_AABB = ([-0.34, 0.30, 0.45], [-0.28, 0.42, 0.80])
_UPPER_HINGE_TOP_AABB = ([0.30, 0.20, 1.58], [0.36, 0.30, 1.632])
_UPPER_HINGE_BOTTOM_AABB = ([0.30, 0.20, 0.94], [0.36, 0.30, 1.00])
_LOWER_HINGE_TOP_AABB = ([0.30, 0.20, 0.86], [0.36, 0.30, 0.92])
_LOWER_HINGE_BOTTOM_AABB = ([0.30, 0.20, 0.03], [0.36, 0.30, 0.10])


def _component(index: int, aabb: tuple[list[float], list[float]]) -> dict:
    return {
        "component_index": index,
        "aabb_min_asset_m": list(aabb[0]),
        "aabb_max_asset_m": list(aabb[1]),
        "aabb_extent_m": [high - low for low, high in zip(aabb[0], aabb[1])],
        "face_count": 100 - index,
        "vertex_count": 300 - index,
    }


def _fridge_components() -> list[dict]:
    """28 connected components: shell, two door panels, hardware, interior."""

    named = [
        _SHELL_AABB,
        _UPPER_PANEL_AABB,
        _LOWER_PANEL_AABB,
        _UPPER_HANDLE_AABB,
        _LOWER_HANDLE_AABB,
        _UPPER_HINGE_TOP_AABB,
        _UPPER_HINGE_BOTTOM_AABB,
        _LOWER_HINGE_TOP_AABB,
        _LOWER_HINGE_BOTTOM_AABB,
    ]
    components = [_component(index, aabb) for index, aabb in enumerate(named)]
    for extra in range(19):
        bottom = 0.05 + extra * 0.09
        components.append(
            _component(
                9 + extra,
                ([-0.38, -0.33, bottom], [0.38, 0.10, bottom + 0.08]),
            )
        )
    assert len(components) == 28
    return components


def _candidate(
    candidate_id: str,
    *,
    axis: list[float],
    kind: str = "revolute",
    status: str = "ready_for_rigger_input",
    codes: list[str] | None = None,
    moving: list[str] | None = None,
    parent: str | None = "/Asset/source_mesh_part_5",
    parent_source: str | None = "released_model_topology",
) -> dict:
    candidate = {
        "schema_version": "joint-agent-stage2-v0",
        "candidate_id": candidate_id,
        "joint_type_hint": kind,
        "motion_axis_world": axis,
        "review_status": status,
        "unresolved_reason_codes": codes or [],
        "moving_part_prims": moving or [],
    }
    if parent is not None:
        candidate["fixed_parent_prim"] = parent
    if parent_source is not None:
        candidate["parent_resolution_source"] = parent_source
    return candidate


def _bounds_record(
    aabb: tuple[list[float], list[float]],
    *,
    moving: list[str] | None = None,
    parent_aabb: tuple[list[float], list[float]] | None = _SHELL_AABB,
) -> dict:
    record: dict = {
        "status": "measured_from_optimized_usd",
        "moving_part_prims": moving or [],
        "aabb_min": list(aabb[0]),
        "aabb_max": list(aabb[1]),
    }
    if parent_aabb is not None:
        record["fixed_parent_bounds"] = {
            "status": "measured_from_optimized_usd",
            "aabb_min": list(parent_aabb[0]),
            "aabb_max": list(parent_aabb[1]),
        }
    return record


def _document(candidates: list[dict]) -> dict:
    return {
        "schema_version": "joint-agent-stage2-v0",
        "summary": {"candidate_count": len(candidates)},
        "candidates": candidates,
    }


def _fridge_fixture() -> tuple[dict, dict]:
    """Model answer that fragments each door into panel + hardware candidates."""

    candidates = [
        _candidate(
            "upper_panel",
            axis=[0.0, 0.0, -1.0],
            moving=["/Asset/source_mesh_part"],
        ),
        _candidate(
            "upper_handle_hw",
            axis=[0.0, 0.0, 1.0],
            status="hardware_attachment",
            codes=["attachment_not_a_joint"],
            moving=["/Asset/source_mesh_part_3"],
            parent="/Asset/source_mesh_part",
        ),
        _candidate(
            "lower_panel",
            axis=[0.0, 0.0, 1.0],
            moving=["/Asset/source_mesh_part_1"],
        ),
        _candidate(
            "lower_hinge_hw",
            axis=[0.0, 0.0, 1.0],
            status="hardware_attachment",
            codes=["attachment_not_a_joint"],
            moving=["/Asset/source_mesh_part_7"],
        ),
    ]
    bounds = {
        "upper_panel": _bounds_record(
            _UPPER_PANEL_AABB, moving=["/Asset/source_mesh_part"]
        ),
        "upper_handle_hw": _bounds_record(
            _UPPER_HANDLE_AABB,
            moving=["/Asset/source_mesh_part_3"],
            parent_aabb=_UPPER_PANEL_AABB,
        ),
        "lower_panel": _bounds_record(
            _LOWER_PANEL_AABB, moving=["/Asset/source_mesh_part_1"]
        ),
        "lower_hinge_hw": _bounds_record(
            _LOWER_HINGE_TOP_AABB, moving=["/Asset/source_mesh_part_7"]
        ),
    }
    return _document(candidates), bounds


@pytest.mark.parametrize(
    "non_task_mode",
    [NON_TASK_JOINT_MODE_EXEMPT, NON_TASK_JOINT_MODE_NO_EXERCISE],
)
def test_review_admits_one_task_joint_inside_bounded_multi_joint_assembly(
    non_task_mode: str,
) -> None:
    candidates = [
        _candidate("upper_door", axis=[0.0, 0.0, -1.0]),
        _candidate("lower_door", axis=[0.0, 0.0, 1.0]),
        _candidate("drawer", axis=[1.0, 0.0, 0.0], kind="prismatic"),
    ]
    document = _document(candidates)
    bounds = {
        "upper_door": _bounds_record(([-0.36, 0.17, 0.94], [0.36, 0.35, 1.632])),
        "lower_door": _bounds_record(([-0.36, 0.17, 0.03], [0.36, 0.35, 0.92])),
        "drawer": _bounds_record(([-0.2, -0.2, 0.2], [0.2, 0.2, 0.4])),
    }
    components = [
        _component(0, _SHELL_AABB),
        _component(1, _UPPER_PANEL_AABB),
        _component(2, _LOWER_PANEL_AABB),
        _component(3, ([-0.2, -0.2, 0.2], [0.2, 0.2, 0.4])),
    ]

    contract = _contract()
    contract["non_task_joint_mode"] = non_task_mode
    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=contract,
        source_components=components,
    )

    assert receipt["assembly_joint_count"] == 3
    assert receipt["raw_candidate_count"] == 3
    assert receipt["target_candidate_id"] == "upper_door"
    assert receipt["non_task_candidate_ids"] == ["drawer", "lower_door"]
    assert receipt["articulation_root_count"] == 1
    assert receipt["non_task_joint_mode"] == non_task_mode
    assert receipt["non_task_joint_motion_tolerance"] == 0.001
    assert receipt["commanded_task_joint_count"] == 1
    assert receipt["claim_boundary"]["non_task_joint_behavior_exercised"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_review_supports_prismatic_task_selected_on_arbitrary_world_axis() -> None:
    contract = _contract()
    contract["target_joint_type"] = "prismatic"
    contract["target_axis_world"] = [1.0, 0.0, 0.0]
    contract["target_member_projection_constraints"] = [
        {
            "axis_world": [0.0, 1.0, 0.0],
            "interval_m": [2.0, 2.4],
            "minimum_overlap_fraction": 0.9,
        }
    ]
    body_aabb = ([-0.1, 1.8, 0.0], [0.9, 3.6, 0.6])
    candidates = [
        _candidate(
            "left_drawer", axis=[1.0, 0.0, 0.0], kind="prismatic", parent="/Asset/body"
        ),
        _candidate(
            "right_drawer", axis=[1.0, 0.0, 0.0], kind="prismatic", parent="/Asset/body"
        ),
    ]
    document = _document(candidates)
    bounds = {
        "left_drawer": _bounds_record(
            ([0.0, 2.0, 0.1], [0.8, 2.4, 0.4]), parent_aabb=body_aabb
        ),
        "right_drawer": _bounds_record(
            ([0.0, 3.0, 0.1], [0.8, 3.4, 0.4]), parent_aabb=body_aabb
        ),
    }
    components = [
        _component(0, body_aabb),
        _component(1, ([0.0, 2.0, 0.1], [0.8, 2.4, 0.4])),
        _component(2, ([0.0, 3.0, 0.1], [0.8, 3.4, 0.4])),
    ]

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=contract,
        source_components=components,
    )

    assert receipt["target_candidate_id"] == "left_drawer"
    assert receipt["assembly_joint_count"] == 2


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("ambiguous_upper", "exactly_one_task_joint_not_resolved"),
        ("too_many", "joint_candidate_count_outside_preregistered_bounds"),
        ("unresolved", "joint_candidate_not_rigger_ready"),
        ("missing_bounds", "joint_candidate_bounds_invalid"),
    ],
)
def test_review_fails_closed_before_model_authored_topology(
    mutation: str, error: str
) -> None:
    candidates = [
        _candidate("upper_door", axis=[0.0, 0.0, 1.0]),
        _candidate("lower_door", axis=[0.0, 0.0, 1.0]),
    ]
    bounds = {
        "upper_door": _bounds_record(([-1.0, -1.0, 0.94], [1.0, 1.0, 1.632])),
        "lower_door": _bounds_record(([-1.0, -1.0, 0.03], [1.0, 1.0, 0.92])),
    }
    if mutation == "ambiguous_upper":
        # A second, geometrically distinct member also satisfying every target
        # test must stay a distinct link and fail the exactly-one selection.
        bounds["lower_door"] = _bounds_record(([-1.0, -1.0, 0.94], [1.0, 1.0, 1.532]))
    elif mutation == "too_many":
        for index, band in enumerate([(0.05, 0.3), (0.4, 0.65), (0.75, 1.0)]):
            candidate_id = f"extra_{index}"
            candidates.append(
                _candidate(candidate_id, axis=[1.0, 0.0, 0.0], kind="prismatic")
            )
            bounds[candidate_id] = _bounds_record(
                ([2.0, -0.5, band[0]], [2.5, 0.5, band[1]])
            )
    elif mutation == "unresolved":
        candidates[0]["review_status"] = "review_required"
        candidates[0]["unresolved_reason_codes"] = ["axis_unresolved"]
    elif mutation == "missing_bounds":
        bounds.pop("upper_door")
    document = _document(candidates)
    components = [
        _component(0, _SHELL_AABB),
        _component(1, _UPPER_PANEL_AABB),
        _component(2, _LOWER_PANEL_AABB),
    ]

    with pytest.raises(JointAgentArticulationReviewError, match=error):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=_contract(),
            source_components=components,
        )


def test_link_reduction_groups_fragmented_member_candidates_before_count_bound() -> None:
    document, bounds = _fridge_fixture()

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=_contract(),
        source_components=_fridge_components(),
    )

    assert receipt["raw_candidate_count"] == 4
    assert receipt["assembly_joint_count"] == 2
    members = {
        link["link_id"]: link["member_candidate_ids"] for link in receipt["links"]
    }
    assert members == {
        "link_00": ["lower_hinge_hw", "lower_panel"],
        "link_01": ["upper_handle_hw", "upper_panel"],
    }
    assert receipt["target_candidate_id"] == "upper_panel"
    assert receipt["target_link_id"] == "link_01"
    assert receipt["non_task_candidate_ids"] == [
        "lower_hinge_hw",
        "lower_panel",
        "upper_handle_hw",
    ]
    assert receipt["link_membership_basis"] == "geometry_only"
    # The shell and the 19 interior pieces stay the fixed-body complement.
    assert receipt["fixed_body"]["component_indices"] == [0, *range(9, 28)]
    assert receipt["articulation_root_count"] == 1
    by_link = {
        link["link_id"]: link["component_indices"] for link in receipt["links"]
    }
    assert by_link["link_01"] == [1, 3, 5, 6]
    assert by_link["link_00"] == [2, 4, 7, 8]


def test_non_task_candidates_keep_unresolved_codes_without_failing_the_gate() -> None:
    document, bounds = _fridge_fixture()

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=_contract(),
        source_components=_fridge_components(),
    )

    rows = {row["candidate_id"]: row for row in receipt["candidate_review"]}
    assert rows["upper_handle_hw"]["unresolved_reason_codes"] == [
        "attachment_not_a_joint"
    ]
    assert rows["upper_handle_hw"]["non_task_exemption_applied"] is True
    assert rows["upper_handle_hw"]["rigger_ready"] is False
    # The handle's model-declared parent is its own moving link: recorded as a
    # typed violation, tolerated for a locked non-task candidate.
    assert rows["upper_handle_hw"]["parent_binding_violations"] == ["parent_self"]
    assert rows["upper_panel"]["task_candidate"] is True
    assert rows["upper_panel"]["non_task_exemption_applied"] is False
    assert rows["upper_panel"]["parent_binding_violations"] == []


def test_strict_non_task_mode_still_requires_full_resolution() -> None:
    document, bounds = _fridge_fixture()
    contract = _contract()
    contract["non_task_joint_mode"] = NON_TASK_JOINT_MODE_STRICT

    with pytest.raises(
        JointAgentArticulationReviewError,
        match="joint_candidate_not_rigger_ready:upper_handle_hw",
    ):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=contract,
            source_components=_fridge_components(),
        )


def test_unmeasured_non_task_candidate_is_dropped_with_typed_reason() -> None:
    document, bounds = _fridge_fixture()
    document["candidates"].append(
        _candidate(
            "phantom_seal",
            axis=[0.0, 0.0, 1.0],
            status="hardware_attachment",
            codes=["not_measurable"],
            moving=["/Asset/source_mesh_part_27"],
        )
    )
    document["summary"]["candidate_count"] = 5
    bounds["phantom_seal"] = {
        "status": "unmeasured",
        "moving_part_prims": ["/Asset/source_mesh_part_27"],
        "unmeasured_reason": "prim_measurement_error",
        "measurement_errors": ["/Asset/source_mesh_part_27:ErrorException"],
    }

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=_contract(),
        source_components=_fridge_components(),
    )

    assert receipt["raw_candidate_count"] == 5
    assert receipt["assembly_joint_count"] == 2
    assert receipt["dropped_candidates"] == [
        {
            "candidate_id": "phantom_seal",
            "drop_reason_code": "unmeasured_candidate_bounds:prim_measurement_error",
        }
    ]

    strict = _contract()
    strict["non_task_joint_mode"] = NON_TASK_JOINT_MODE_STRICT
    with pytest.raises(
        JointAgentArticulationReviewError,
        match="joint_candidate_unmeasured:phantom_seal",
    ):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=strict,
            source_components=_fridge_components(),
        )


def test_unadmitted_joint_type_is_dropped_not_silently_admitted() -> None:
    document, bounds = _fridge_fixture()
    document["candidates"].append(
        _candidate(
            "lid_ball_joint",
            axis=[0.0, 0.0, 1.0],
            kind="ball",
            moving=["/Asset/source_mesh_part_8"],
        )
    )
    document["summary"]["candidate_count"] = 5
    bounds["lid_ball_joint"] = _bounds_record(_LOWER_HINGE_BOTTOM_AABB)

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=_contract(),
        source_components=_fridge_components(),
    )

    assert receipt["dropped_candidates"] == [
        {
            "candidate_id": "lid_ball_joint",
            "drop_reason_code": "unadmitted_joint_type:ball",
        }
    ]
    assert receipt["assembly_joint_count"] == 2


def test_assembly_shell_candidate_fails_extent_band_and_never_matches_target() -> None:
    document, bounds = _fridge_fixture()
    document["candidates"].append(
        _candidate(
            "cabinet_shell",
            axis=[0.0, 0.0, 1.0],
            moving=["/Asset/source_mesh_part_5"],
            parent="/Asset",
        )
    )
    document["summary"]["candidate_count"] = 5
    bounds["cabinet_shell"] = _bounds_record(
        _SHELL_AABB, moving=["/Asset/source_mesh_part_5"], parent_aabb=None
    )

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=_contract(),
        source_components=_fridge_components(),
    )

    rows = {row["candidate_id"]: row for row in receipt["candidate_review"]}
    shell = rows["cabinet_shell"]
    # The shell passes the raw geometric tests (type, axis, full overlap) but
    # its extent is far outside the preregistered member band.
    assert shell["target_projection_overlap_fractions"] == [1.0]
    assert shell["target_extent_ratio_within_band"] is False
    assert shell["target_match"] is False
    assert shell["parent_binding_violations"] == [
        "parent_is_articulation_root",
        "parent_unmeasured",
    ]
    shell_link = next(
        link
        for link in receipt["links"]
        if "cabinet_shell" in link["member_candidate_ids"]
    )
    assert shell_link["frame_coextensive"] is True
    assert shell_link["member_candidate_ids"] == ["cabinet_shell"]
    assert receipt["target_candidate_id"] == "upper_panel"
    assert receipt["assembly_joint_count"] == 3


def test_containment_preference_selects_the_member_inside_the_target_band() -> None:
    shroud_aabb = ([-0.4, 0.1, 0.7], [0.4, 0.45, 1.7])
    candidates = [
        _candidate("upper_panel", axis=[0.0, 0.0, -1.0], moving=["/Asset/source_mesh_part"]),
        _candidate("lower_panel", axis=[0.0, 0.0, 1.0], moving=["/Asset/source_mesh_part_1"]),
        _candidate("door_shroud", axis=[0.0, 0.0, 1.0], moving=["/Asset/source_mesh_part_9"]),
    ]
    document = _document(candidates)
    bounds = {
        "upper_panel": _bounds_record(_UPPER_PANEL_AABB, moving=["/Asset/source_mesh_part"]),
        "lower_panel": _bounds_record(_LOWER_PANEL_AABB, moving=["/Asset/source_mesh_part_1"]),
        "door_shroud": _bounds_record(shroud_aabb, moving=["/Asset/source_mesh_part_9"]),
    }

    receipt = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=_contract(),
        source_components=_fridge_components(),
    )

    rows = {row["candidate_id"]: row for row in receipt["candidate_review"]}
    # Both satisfy type, axis, overlap, and the extent band; only the panel is
    # CONTAINED by the target interval, so the shroud that CONTAINS it loses.
    assert rows["door_shroud"]["target_match"] is True
    assert rows["door_shroud"]["contained_by_target_intervals"] is False
    assert rows["upper_panel"]["target_match"] is True
    assert rows["upper_panel"]["contained_by_target_intervals"] is True
    assert receipt["target_candidate_id"] == "upper_panel"
    assert receipt["containment_disambiguation"]["applied"] is True
    assert receipt["containment_disambiguation"]["rejected_containing_link_ids"] == [
        rows["door_shroud"]["link_id"]
    ]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("parent_missing", "joint_task_candidate_parent_missing:upper_panel"),
        (
            "parent_is_root",
            "joint_task_candidate_parent_is_articulation_root:upper_panel",
        ),
        ("parent_self_prim", "joint_task_candidate_parent_self:upper_panel"),
        ("parent_unmeasured", "joint_task_candidate_parent_unmeasured:upper_panel"),
        ("parent_outside", "joint_task_candidate_parent_unresolvable:upper_panel"),
        ("parent_cycle", "joint_task_candidate_parent_cycle:upper_panel"),
    ],
)
def test_task_parent_binding_violations_fail_closed(mutation: str, error: str) -> None:
    document, bounds = _fridge_fixture()
    upper = next(
        candidate
        for candidate in document["candidates"]
        if candidate["candidate_id"] == "upper_panel"
    )
    if mutation == "parent_missing":
        upper.pop("fixed_parent_prim")
    elif mutation == "parent_is_root":
        upper["fixed_parent_prim"] = "/Asset"
    elif mutation == "parent_self_prim":
        upper["fixed_parent_prim"] = "/Asset/source_mesh_part"
    elif mutation == "parent_unmeasured":
        bounds["upper_panel"]["fixed_parent_bounds"] = {
            "status": "unmeasured",
            "unmeasured_reason": "prim_not_found",
        }
    elif mutation == "parent_outside":
        bounds["upper_panel"]["fixed_parent_bounds"] = {
            "status": "measured_from_optimized_usd",
            "aabb_min": [0.0, 0.0, 3.0],
            "aabb_max": [0.5, 0.5, 3.5],
        }
    elif mutation == "parent_cycle":
        upper["fixed_parent_prim"] = "/Asset/source_mesh_part_1"
        bounds["upper_panel"]["fixed_parent_bounds"] = {
            "status": "measured_from_optimized_usd",
            "aabb_min": list(_LOWER_PANEL_AABB[0]),
            "aabb_max": list(_LOWER_PANEL_AABB[1]),
        }
        lower = next(
            candidate
            for candidate in document["candidates"]
            if candidate["candidate_id"] == "lower_panel"
        )
        lower["fixed_parent_prim"] = "/Asset/source_mesh_part"
        bounds["lower_panel"]["fixed_parent_bounds"] = {
            "status": "measured_from_optimized_usd",
            "aabb_min": list(_UPPER_PANEL_AABB[0]),
            "aabb_max": list(_UPPER_PANEL_AABB[1]),
        }

    with pytest.raises(JointAgentArticulationReviewError, match=error):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=_contract(),
            source_components=_fridge_components(),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("minimum_assembly_joint_count", 0, "joint_review_minimum_joint_count_invalid"),
        ("minimum_assembly_joint_count", 5, "joint_review_minimum_joint_count_invalid"),
        (
            "commanded_task_joint_count",
            0,
            "joint_review_commanded_task_joint_count_invalid",
        ),
        (
            "required_articulation_root_count",
            0,
            "joint_review_required_root_count_invalid",
        ),
        (
            "non_task_joint_mode",
            "anything_goes",
            "joint_review_non_task_joint_mode_invalid",
        ),
        (
            "non_task_joint_motion_tolerance",
            -0.5,
            "joint_review_non_task_joint_motion_tolerance_invalid",
        ),
        (
            "target_member_extent_ratio_band",
            [0.0, 1.5],
            "joint_review_extent_ratio_band_invalid",
        ),
    ],
)
def test_every_preregistered_scope_field_is_read_and_validated(
    field: str, value, error: str
) -> None:
    document, bounds = _fridge_fixture()
    contract = _contract()
    contract[field] = value

    with pytest.raises(JointAgentArticulationReviewError, match=error):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=contract,
            source_components=_fridge_components(),
        )


def test_minimum_joint_count_bound_is_enforced_from_the_contract() -> None:
    document, bounds = _fridge_fixture()
    contract = _contract()
    contract["minimum_assembly_joint_count"] = 3

    with pytest.raises(
        JointAgentArticulationReviewError,
        match="joint_candidate_count_outside_preregistered_bounds",
    ):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=contract,
            source_components=_fridge_components(),
        )


def test_extent_ratio_band_is_enforced_not_just_validated() -> None:
    document, bounds = _fridge_fixture()
    contract = _contract()
    contract["target_member_extent_ratio_band"] = [1.6, 2.0]

    with pytest.raises(
        JointAgentArticulationReviewError,
        match="exactly_one_task_joint_not_resolved",
    ):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=contract,
            source_components=_fridge_components(),
        )


def test_required_root_count_is_validated_against_the_resolved_complement() -> None:
    document, bounds = _fridge_fixture()
    # Only the door-panel components: every component is claimed by a moving
    # link, so no fixed body remains and the required single root is missing.
    components = [
        _component(1, _UPPER_PANEL_AABB),
        _component(2, _LOWER_PANEL_AABB),
    ]

    with pytest.raises(
        JointAgentArticulationReviewError,
        match="joint_review_articulation_root_count_mismatch",
    ):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=_contract(),
            source_components=components,
        )


def test_review_requires_source_components_for_link_membership() -> None:
    document, bounds = _fridge_fixture()

    with pytest.raises(
        JointAgentArticulationReviewError,
        match="joint_review_source_components_missing",
    ):
        review_joint_agent_articulation(
            candidates_document=document,
            candidate_bounds=bounds,
            review_contract=_contract(),
        )


def test_link_membership_resolver_is_pure_geometry() -> None:
    membership = resolve_link_membership(
        member_aabbs={
            "panel": (tuple(_UPPER_PANEL_AABB[0]), tuple(_UPPER_PANEL_AABB[1])),
            "handle": (tuple(_UPPER_HANDLE_AABB[0]), tuple(_UPPER_HANDLE_AABB[1])),
            "second_member": (
                tuple(_LOWER_PANEL_AABB[0]),
                tuple(_LOWER_PANEL_AABB[1]),
            ),
        },
        source_component_aabbs=[
            (0, (tuple(_SHELL_AABB[0]), tuple(_SHELL_AABB[1]))),
            (1, (tuple(_UPPER_PANEL_AABB[0]), tuple(_UPPER_PANEL_AABB[1]))),
            (2, (tuple(_LOWER_PANEL_AABB[0]), tuple(_LOWER_PANEL_AABB[1]))),
            (3, (tuple(_UPPER_HANDLE_AABB[0]), tuple(_UPPER_HANDLE_AABB[1]))),
        ],
    )

    members = {
        link["link_id"]: link["member_candidate_ids"] for link in membership["links"]
    }
    assert members == {
        "link_00": ["second_member"],
        "link_01": ["handle", "panel"],
    }
    assert membership["fixed_component_indices"] == [0]
    assert membership["articulation_root_count"] == 1
    assert membership["membership_basis"] == "geometry_only"


def test_collect_candidate_bounds_quarantines_prim_errors_per_candidate() -> None:
    document = _document(
        [
            _candidate(
                "measured_with_noise",
                axis=[0.0, 0.0, 1.0],
                moving=["/Asset/part_ok", "/Asset/part_bad"],
                parent="/Asset/parent_ok",
            ),
            _candidate(
                "unmeasurable",
                axis=[0.0, 0.0, 1.0],
                moving=["/Asset/part_bad"],
                parent="/Asset/parent_bad",
            ),
            _candidate(
                "prim_absent",
                axis=[0.0, 0.0, 1.0],
                moving=["/Asset/missing"],
                parent=None,
                parent_source=None,
            ),
        ]
    )

    def measure_prim(path: str):
        if path.endswith("_bad"):
            raise RuntimeError("pxr measurement failure")
        if path.endswith("missing"):
            return None
        return ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    bounds = collect_candidate_bounds(document, measure_prim=measure_prim)

    noisy = bounds["measured_with_noise"]
    assert noisy["status"] == "measured_from_optimized_usd"
    assert noisy["aabb_min"] == [0.0, 0.0, 0.0]
    assert noisy["measurement_errors"] == ["/Asset/part_bad:RuntimeError"]
    assert noisy["fixed_parent_bounds"]["status"] == "measured_from_optimized_usd"

    broken = bounds["unmeasurable"]
    assert broken["status"] == "unmeasured"
    assert broken["unmeasured_reason"] == "prim_measurement_error"
    assert broken["measurement_errors"] == ["/Asset/part_bad:RuntimeError"]
    assert broken["fixed_parent_bounds"] == {
        "status": "unmeasured",
        "unmeasured_reason": "prim_measurement_error",
        "measurement_errors": ["/Asset/parent_bad:RuntimeError"],
    }

    absent = bounds["prim_absent"]
    assert absent["status"] == "unmeasured"
    assert absent["unmeasured_reason"] == "no_measurable_prims"
    assert "fixed_parent_bounds" not in absent


def test_resolve_joint_agent_output_follows_symlinks_and_skips_zero_byte(
    tmp_path: Path,
) -> None:
    working = tmp_path / "work"
    optimized_dir = working / "optimized"
    optimized_dir.mkdir(parents=True)
    real = optimized_dir / "asset_optimized.usdc"
    real.write_bytes(b"usd-bytes")
    (optimized_dir / "empty_optimized.usdc").write_bytes(b"")
    (optimized_dir / "link_optimized.usda").symlink_to(real)

    notes: list[str] = []
    resolved = resolve_joint_agent_output(
        working_dir=working,
        role="optimized_source",
        relative_glob="optimized/*_optimized.usd*",
        notes=notes,
    )

    assert resolved == real.resolve()
    assert "skipped_zero_byte:optimized_source:optimized/empty_optimized.usdc" in notes
    assert "resolved_symlink:optimized_source:optimized/link_optimized.usda" in notes

    (optimized_dir / "second_optimized.usdc").write_bytes(b"other")
    with pytest.raises(
        ValueError,
        match="joint_agent_output_role_not_unique:optimized_source:observed=2",
    ):
        resolve_joint_agent_output(
            working_dir=working,
            role="optimized_source",
            relative_glob="optimized/*_optimized.usd*",
        )


def test_receipt_is_deterministic_for_identical_inputs() -> None:
    document, bounds = _fridge_fixture()
    components = _fridge_components()

    first = review_joint_agent_articulation(
        candidates_document=copy.deepcopy(document),
        candidate_bounds=copy.deepcopy(bounds),
        review_contract=_contract(),
        source_components=copy.deepcopy(components),
    )
    second = review_joint_agent_articulation(
        candidates_document=document,
        candidate_bounds=bounds,
        review_contract=_contract(),
        source_components=components,
    )

    assert first == second
    assert first["receipt_digest"] == second["receipt_digest"]
