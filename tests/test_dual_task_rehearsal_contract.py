from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_rehearsal_contract import (
    FORBIDDEN_SELECTION_SIGNALS,
    REQUIRED_SELECTION_CRITERIA,
    DualTaskRehearsalContractError,
    validate_scene_freeze,
    validate_selection_preregistration,
    validate_task_freeze,
    validate_task_freeze_join,
    validate_task_freeze_set,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _seal(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def selection_preregistration() -> dict:
    return _seal(
        {
            "schema_version": "dual_task_scene_selection_preregistration.v1",
            "program_id": "arm-decision-proof-v1",
            "adp_item": "ADP-009D",
            "day_gate": "public_scene_day_28",
            "frozen_before_learned_policy_execution": True,
            "learned_policy_outcomes_accessed": False,
            "candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "criteria": [
                {
                    "criterion_id": criterion,
                    "required": True,
                    "evidence_rule": f"observed_receipt_required:{criterion}",
                }
                for criterion in sorted(REQUIRED_SELECTION_CRITERIA)
            ],
            "forbidden_selection_signals": sorted(FORBIDDEN_SELECTION_SIGNALS),
            "candidate_scenes": [
                {
                    "publisher_scene_id": scene_id,
                    "method_outcomes_consulted": False,
                }
                for scene_id in ("840874", "840920", "841151")
            ],
            "selected_scene_id": None,
            "articulation_rule": (
                "complete_joint_graph_no_universal_joint_count_cap_reject_only_if_unbounded_"
                "uncontrollable_unresettable_unscoreable_or_collision_unqualified"
            ),
            "task_count": 2,
            "claim_ceiling": "development_only_public_dataset_rehearsal",
            "preregistration_digest": "",
        },
        "preregistration_digest",
    )


def scene_freeze() -> dict:
    return _seal(
        {
            "schema_version": "dual_task_scene_freeze.v1",
            "selection_preregistration_digest": selection_preregistration()[
                "preregistration_digest"
            ],
            "learned_policy_outcomes_accessed": False,
            "selected_scene_id": "840920",
            "candidate_ledger": [
                {
                    "publisher_scene_id": "840874",
                    "decision": "rejected",
                    "reason": "no independent rigid source collider and feasible base",
                    "previously_used": False,
                    "method_outcomes_consulted": False,
                },
                {
                    "publisher_scene_id": "840920",
                    "decision": "selected",
                    "reason": "two independent observed task regions",
                    "previously_used": False,
                    "method_outcomes_consulted": False,
                },
                {
                    "publisher_scene_id": "841151",
                    "decision": "rejected",
                    "reason": "forbidden refrigerator is only explicit articulation",
                    "previously_used": False,
                    "method_outcomes_consulted": False,
                },
            ],
            "source_components": {
                "interiorgs": {
                    "repository": "spatialverse/InteriorGS",
                    "revision": "revision-a",
                    "sha256": _digest("a"),
                    "size_bytes": 10,
                    "license": "custom",
                    "rights_admitted": True,
                    "restrictions": {
                        "redistribution": "forbidden",
                        "raw_private_upload": "forbidden",
                        "derived_private_upload": "bounded_authority_only",
                        "retention": "bounded_to_goal",
                        "training": "forbidden",
                        "publication": "derived_receipts_only",
                    },
                },
                "sage_collision": {
                    "repository": "spatialverse/SAGE-3D_Collision_Mesh",
                    "revision": "revision-b",
                    "sha256": _digest("b"),
                    "size_bytes": 20,
                    "license": "CC-BY-NC-4.0",
                    "rights_admitted": True,
                    "restrictions": {
                        "redistribution": "license_and_attribution_bound",
                        "raw_private_upload": "bounded_authority_only",
                        "derived_private_upload": "bounded_authority_only",
                        "retention": "bounded_to_goal",
                        "training": "not_authorized_by_goal",
                        "publication": "noncommercial_with_attribution",
                    },
                },
            },
            "criterion_results": {
                criterion: {
                    "status": "observed_pass",
                    "evidence_digest": _digest("e"),
                    "remaining_gate": "native construction qualification still required",
                }
                for criterion in REQUIRED_SELECTION_CRITERIA
            },
            "topology_survey_digest": _digest("c"),
            "reconnaissance_render_digest": _digest("d"),
            "scene_freeze_digest": "",
        },
        "scene_freeze_digest",
    )


def _articulation_graph() -> dict:
    locked_drive = {
        "drive_type": "force",
        "stiffness": 20.0,
        "damping": 1.0,
        "maximum_force": 100.0,
    }
    target_drive = {
        **locked_drive,
        "stiffness": 0.0,
    }
    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "body", "is_root": True, "semantic_role": "fixed_body"},
            {"link_id": "door", "is_root": False, "semantic_role": "target_door"},
            {"link_id": "dial", "is_root": False, "semantic_role": "locked_dial"},
        ],
        "joints": [
            {
                "joint_id": "door_hinge",
                "parent_link_id": "body",
                "child_link_id": "door",
                "joint_type": "revolute",
                "role": "target",
                "axis": [0.0, 0.0, 1.0],
                "limits": [0.0, 1.2],
                "reset_position": 0.0,
                "reset_tolerance": 0.0001,
                "drive": target_drive,
                "dependency": None,
            },
            {
                "joint_id": "selector_axis",
                "parent_link_id": "body",
                "child_link_id": "dial",
                "joint_type": "revolute",
                "role": "locked",
                "axis": [1.0, 0.0, 0.0],
                "limits": [-3.2, 3.2],
                "reset_position": 0.0,
                "reset_tolerance": 0.0001,
                "drive": locked_drive,
                "dependency": None,
            },
        ],
        "collision_pairs": [
            {"link_a": "body", "link_b": "door", "collision_enabled": True}
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"door_hinge": [0.7, 1.0]},
        },
    }


def task_freeze(task_id: str, task_kind: str, source_id: str) -> dict:
    suffix = "a" if task_id == "task_a" else "b"
    payload = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": task_id,
        "prompt": (
            "Open the observed mechanism"
            if suffix == "a"
            else "Relocate the observed rigid object on its support"
        ),
        "task_kind": task_kind,
        "scene_freeze_digest": scene_freeze()["scene_freeze_digest"],
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "source_object": {
            "instance_id": source_id,
            "semantic_label": "washing_machine" if suffix == "a" else "notebook",
            "observed_bounds_world_m": {
                "minimum": [0.0, 0.0, 0.0],
                "maximum": [0.6, 0.6, 0.9],
            },
            "observed_pose_world": {
                "position_world_m": [0.3, 0.3, 0.45],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "support_or_attachment_id": "floor" if suffix == "a" else "desk",
            "collision_identity_receipt_digest": _digest("4"),
            "support_receipt_digest": _digest("5"),
            "franka_placement_packet_digest": _digest("6"),
            "visibility_receipt_digest": _digest("7"),
        },
        "removal_plan": {
            "removal_id": f"removal_{suffix}",
            "mask_set_id": f"mask_set_{suffix}",
            "source_collider_prim_path": f"/Root/source_{suffix}",
            "collider_deletion_id": f"collider_delete_{suffix}",
            "replacement_asset_id": f"replacement_{suffix}",
            "replacement_qualification_id": f"replacement_qualification_{suffix}",
        },
        "cameras": {
            "external": f"external_{suffix}",
            "wrist": f"wrist_{suffix}",
            "overview": f"overview_{suffix}",
        },
        "overview_camera_policy_input": False,
        "overview_camera_deterministic_scoring_input": False,
        "execution_contract": {
            "control_frequency_hz": 20,
            "maximum_steps": 400,
            "settle_window_steps": 20,
            "seeds": [1101, 1102],
            "canonical_scenario_cell_id": f"canonical_{suffix}",
            "reset_state": {"robot": "home", "object": "source_start"},
        },
        "deterministic_success_predicates": ["released", "settled", "retreated"],
        "failure_rungs": ["never_moved", "moved_below_threshold", "collision_failure"],
        "target_configuration": (
            {
                "kind": "joint_interval",
                "target_joint_ids": ["door_hinge"],
                "joint_intervals": {"door_hinge": [0.7, 1.0]},
            }
            if suffix == "a"
            else {
                "kind": "pose_volume",
                "position_bounds_world_m": {
                    "minimum": [0.4, 0.4, 0.4],
                    "maximum": [0.5, 0.5, 0.5],
                },
                "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
                "maximum_orientation_error_rad": 0.1,
                "support_id": "desk",
                "release_required": True,
            }
        ),
        "articulation_graph": _articulation_graph() if suffix == "a" else None,
        "task_freeze_digest": "",
    }
    return _seal(payload, "task_freeze_digest")


def test_selection_scene_and_two_independent_task_freezes_validate() -> None:
    assert validate_selection_preregistration(selection_preregistration())[
        "selected_scene_id"
    ] is None
    assert validate_scene_freeze(scene_freeze())["selected_scene_id"] == "840920"
    task_a = validate_task_freeze(
        task_freeze("task_a", "articulated_interaction", "165")
    )
    task_b = validate_task_freeze(
        task_freeze("task_b", "rigid_object_manipulation", "385")
    )

    joined = validate_task_freeze_join([task_a, task_b])

    assert joined["independent"] is True
    assert joined["task_ids"] == ["task_a", "task_b"]


def test_join_rejects_shared_mask_collider_or_replacement() -> None:
    task_a = task_freeze("task_a", "articulated_interaction", "165")
    task_b = task_freeze("task_b", "rigid_object_manipulation", "385")
    for field in (
        "mask_set_id",
        "source_collider_prim_path",
        "replacement_asset_id",
    ):
        mutated = copy.deepcopy(task_b)
        mutated["removal_plan"][field] = task_a["removal_plan"][field]
        _seal(mutated, "task_freeze_digest")
        with pytest.raises(DualTaskRehearsalContractError) as caught:
            validate_task_freeze_join([task_a, mutated])
        assert f"dual_task_join_shared_{field}" in caught.value.errors


def _independent_task_copy(base: dict, index: int) -> dict:
    value = copy.deepcopy(base)
    suffix = chr(ord("a") + index)
    value["task_id"] = f"task_{suffix}"
    value["source_object"]["instance_id"] = f"source_{suffix}"
    for field in (
        "removal_id",
        "mask_set_id",
        "collider_deletion_id",
        "replacement_asset_id",
        "replacement_qualification_id",
    ):
        value["removal_plan"][field] = f"{field}_{suffix}"
    value["removal_plan"]["source_collider_prim_path"] = f"/Root/source_{suffix}"
    return _seal(value, "task_freeze_digest")


@pytest.mark.parametrize("count", [1, 2, 5])
def test_general_scene_task_freeze_set_accepts_one_to_five_objects(count: int) -> None:
    base = task_freeze("task_b", "rigid_object_manipulation", "source_b")
    tasks = [_independent_task_copy(base, index) for index in range(count)]

    result = validate_task_freeze_set(tasks)

    assert result["task_count"] == count
    assert result["maximum_task_count"] == 5
    assert len(result["task_freeze_digests"]) == count


def test_general_scene_task_freeze_set_rejects_six_objects() -> None:
    base = task_freeze("task_b", "rigid_object_manipulation", "source_b")
    tasks = [_independent_task_copy(base, index) for index in range(6)]

    with pytest.raises(DualTaskRehearsalContractError) as caught:
        validate_task_freeze_set(tasks)

    assert caught.value.errors == ("scene_task_freeze_set_count_out_of_range",)


def test_learned_outcome_leakage_fails_before_scene_selection() -> None:
    value = selection_preregistration()
    value["learned_policy_outcomes_accessed"] = True
    _seal(value, "preregistration_digest")

    with pytest.raises(DualTaskRehearsalContractError) as caught:
        validate_selection_preregistration(value)

    assert "dual_task_selection_outcome_leakage" in caught.value.errors


def test_checked_in_third_scene_freezes_validate_as_one_independent_pair() -> None:
    root = Path(__file__).resolve().parents[1]
    manifests = root / "docs/arm_decision_proof_v1/manifests"

    preregistration = json.loads(
        (manifests / "third_scene_dual_task_selection_preregistration.v1.json").read_text()
    )
    scene = json.loads(
        (manifests / "third_scene_840920_dual_task_scene_freeze.v1.json").read_text()
    )
    task_a = json.loads(
        (manifests / "third_scene_840920_task_a_freeze.v1.json").read_text()
    )
    task_b = json.loads(
        (manifests / "third_scene_840920_task_b_freeze.v1.json").read_text()
    )
    expected_join = json.loads(
        (manifests / "third_scene_840920_dual_task_freeze_join.v1.json").read_text()
    )

    validate_selection_preregistration(preregistration)
    validate_scene_freeze(scene)
    assert validate_task_freeze_join([task_a, task_b]) == expected_join
