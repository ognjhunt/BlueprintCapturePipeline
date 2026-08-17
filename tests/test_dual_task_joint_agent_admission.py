from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_joint_agent_admission import (
    DualTaskJointAgentAdmissionError,
    INAPPLICABLE_STATUS,
    NON_TASK_MODE,
    READY_STATUS,
    build_dual_task_joint_agent_admission,
    main,
    validate_dual_task_joint_agent_admission,
)


SCENE_ID = "840920"
TASK_A_SOURCE_SHA = (
    "sha256:607411e0bbf2bd4850321f4ca5f635fa2095a54cdf0dd3df1c95152be8f616cc"
)
TASK_B_SOURCE_SHA = (
    "sha256:52cc7d4623f429b86ab2cff678bc9e4a2a99d3eb33ddb7ecd040bec9f65df50c"
)
COLLISION_SHA = (
    "sha256:9e51c7c4360b5071fdbbb9cebbf61a475cc88802d53b5b2b45df294ec2b7c8fd"
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _seal(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _drive(*, stiffness: float, damping: float = 2.0) -> dict:
    return {
        "drive_type": "force",
        "stiffness": stiffness,
        "damping": damping,
        "maximum_force": 100.0,
    }


def _washer_graph() -> dict:
    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "body", "is_root": True, "semantic_role": "fixed_body_generated_candidate"},
            {"link_id": "door", "is_root": False, "semantic_role": "target_door_observed_exterior_generated_mechanism"},
            {"link_id": "latch", "is_root": False, "semantic_role": "dependent_latch_generated_candidate"},
            {"link_id": "drum", "is_root": False, "semantic_role": "passive_drum_generated_candidate"},
            {"link_id": "selector", "is_root": False, "semantic_role": "locked_selector_observed_exterior"},
            {"link_id": "drawer", "is_root": False, "semantic_role": "locked_detergent_drawer_observed_exterior"},
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
                "reset_tolerance": 0.001,
                "drive": _drive(stiffness=0.0),
                "dependency": None,
            },
            {
                "joint_id": "latch_coupler",
                "parent_link_id": "door",
                "child_link_id": "latch",
                "joint_type": "revolute",
                "role": "dependent",
                "axis": [0.0, 0.0, 1.0],
                "limits": [-0.1, 0.1],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": _drive(stiffness=4.0, damping=1.0),
                "dependency": {
                    "driver_joint_id": "door_hinge",
                    "multiplier": 0.05,
                    "offset": 0.0,
                    "tolerance": 0.002,
                },
            },
            {
                "joint_id": "drum_bearing",
                "parent_link_id": "body",
                "child_link_id": "drum",
                "joint_type": "continuous",
                "role": "passive",
                "axis": [0.0, 1.0, 0.0],
                "limits": [-100.0, 100.0],
                "reset_position": 0.0,
                "reset_tolerance": 0.01,
                "drive": {
                    "drive_type": "none",
                    "stiffness": 0.0,
                    "damping": 0.2,
                    "maximum_force": 0.0,
                },
                "dependency": None,
            },
            {
                "joint_id": "selector_axis",
                "parent_link_id": "body",
                "child_link_id": "selector",
                "joint_type": "revolute",
                "role": "locked",
                "axis": [0.0, 1.0, 0.0],
                "limits": [-3.2, 3.2],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": _drive(stiffness=100.0, damping=10.0),
                "dependency": None,
            },
            {
                "joint_id": "drawer_slide",
                "parent_link_id": "body",
                "child_link_id": "drawer",
                "joint_type": "prismatic",
                "role": "locked",
                "axis": [0.0, 1.0, 0.0],
                "limits": [0.0, 0.2],
                "reset_position": 0.0,
                "reset_tolerance": 0.001,
                "drive": _drive(stiffness=100.0, damping=10.0),
                "dependency": None,
            },
        ],
        "collision_pairs": [
            {"link_a": "body", "link_b": "door", "collision_enabled": True},
            {"link_a": "door", "link_b": "latch", "collision_enabled": False},
            {"link_a": "body", "link_b": "drum", "collision_enabled": True},
            {"link_a": "body", "link_b": "selector", "collision_enabled": True},
            {"link_a": "body", "link_b": "drawer", "collision_enabled": True},
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"door_hinge": [0.7, 0.95]},
        },
    }


def _notebook_graph() -> dict:
    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": [
            {"link_id": "base", "is_root": True, "semantic_role": "rigid_task_body_observed_exterior"},
            {"link_id": "display", "is_root": False, "semantic_role": "locked_display_observed_exterior_generated_hinge"},
        ],
        "joints": [
            {
                "joint_id": "display_hinge",
                "parent_link_id": "base",
                "child_link_id": "display",
                "joint_type": "revolute",
                "role": "locked",
                "axis": [1.0, 0.0, 0.0],
                "limits": [0.0, 2.2],
                "reset_position": 1.745329252,
                "reset_tolerance": 0.001,
                "drive": _drive(stiffness=200.0, damping=20.0),
                "dependency": None,
            }
        ],
        "collision_pairs": [
            {"link_a": "base", "link_b": "display", "collision_enabled": True}
        ],
        "success_predicate": {"combination": "all", "joint_intervals": {}},
    }


def _task_freeze(*, task: str) -> dict:
    washer = task == "a"
    instance = "165" if washer else "385"
    semantic = "washing_machine" if washer else "notebook_computer"
    graph = _washer_graph() if washer else _notebook_graph()
    payload = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": (
            "task_a_washer_door_open" if washer else "task_b_notebook_relocation"
        ),
        "prompt": (
            "Open the front-loading washer door and release it in the open target interval."
            if washer
            else "Relocate the open notebook 0.15 meters along the observed desk support, preserve its locked-open orientation, release, settle, and retreat."
        ),
        "task_kind": "articulated_interaction" if washer else "rigid_object_manipulation",
        "scene_freeze_digest": _digest("8"),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "mechanism_provenance": (
            "observed washer exterior and circular door; hidden mechanisms are generated candidate content"
            if washer
            else "observed open notebook exterior; display hinge is generated candidate content and locked"
        ),
        "source_object": {
            "instance_id": instance,
            "semantic_label": semantic,
            "observed_bounds_world_m": {
                "minimum": [3.2, 9.4, 0.0] if washer else [13.4, 6.9, 0.825],
                "maximum": [3.8, 10.1, 0.848] if washer else [13.76, 7.39, 1.125],
            },
            "observed_pose_world": {
                "position_world_m": [3.515, 9.759, 0.424] if washer else [13.58, 7.172, 0.975],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "support_or_attachment_id": (
                "washer_body_observed_exterior_generated_mechanism"
                if washer
                else "interiorgs_101_wardrobe_top"
            ),
            "collision_identity_receipt_digest": _digest("a" if washer else "b"),
            "support_receipt_digest": _digest("c"),
            "franka_placement_packet_digest": _digest("d"),
            "visibility_receipt_digest": _digest("e"),
        },
        "removal_plan": {
            "removal_id": f"840920_removal_{instance}",
            "mask_set_id": f"840920_mask_set_{instance}",
            "source_collider_prim_path": f"/Root/source_{instance}",
            "collider_deletion_id": f"840920_collider_delete_{instance}",
            "replacement_asset_id": f"840920_replacement_{instance}",
            "replacement_qualification_id": f"840920_qualification_{instance}",
        },
        "cameras": {"external": "external", "wrist": "wrist", "overview": "overview"},
        "overview_camera_policy_input": False,
        "overview_camera_deterministic_scoring_input": False,
        "execution_contract": {
            "control_frequency_hz": 20,
            "maximum_steps": 400,
            "settle_window_steps": 20,
            "seeds": [3101, 3102] if washer else [3201, 3202],
            "canonical_scenario_cell_id": "840920_task_a_canonical" if washer else "840920_task_b_canonical",
            "reset_state": {"robot": "home", "object": "frozen"},
        },
        "deterministic_success_predicates": ["released", "stable", "retreated"],
        "failure_rungs": ["never_moved", "collision_failure", "reset_failure"],
        "target_configuration": (
            {
                "kind": "joint_interval",
                "target_joint_ids": ["door_hinge"],
                "joint_intervals": {"door_hinge": [0.7, 0.95]},
            }
            if washer
            else {
                "kind": "pose_volume",
                "position_bounds_world_m": {
                    "minimum": [13.55, 7.29, 0.95],
                    "maximum": [13.61, 7.35, 0.99],
                },
                "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
                "maximum_orientation_error_rad": 0.12,
                "support_id": "interiorgs_101_wardrobe_top",
                "release_required": True,
            }
        ),
        "articulation_graph": graph,
        "task_freeze_digest": "",
    }
    return _seal(payload, "task_freeze_digest")


def _source_receipt(*, task: str) -> dict:
    washer = task == "a"
    count = 71 if washer else 88
    source_sha = TASK_A_SOURCE_SHA if washer else TASK_B_SOURCE_SHA
    instance = "165" if washer else "385"
    semantic = "washing_machine" if washer else "notebook_computer"
    payload = {
        "schema_version": "articulated_source_asset.v1",
        "status": "materialized",
        "target": {
            "interiorgs_instance_id": instance,
            "semantic_label": semantic,
        },
        "source_collision_prim_path": f"/Root/source_{instance}",
        "source_collision_identity_receipt_digest": _digest("a" if washer else "b"),
        "source_files": {
            "sage_collision_usd": {
                "path": "840920_collision.usd",
                "sha256": COLLISION_SHA,
                "size_bytes": 17_929_371,
            }
        },
        "output_asset": {
            "relative_path": "articulated_source_mesh.usda",
            "sha256": source_sha,
            "size_bytes": 1_881_177 if washer else 183_207,
        },
        "connected_component_count": count,
        "connected_components": [
            {
                "component_index": index,
                "aabb_min_asset_m": [-0.3, -0.3, 0.0],
                "aabb_max_asset_m": [0.3, 0.3, 0.848],
            }
            for index in range(count)
        ],
        "joint_agent_0_5_2_input": {
            "usd_path_ready": True,
            "default_prim_valid": True,
            "connected_component_geom_subsets_authored": True,
            "predicted_split_prim_count": count,
            "topology_inference_executed": False,
        },
        "claim_boundary": {
            "connected_components_are_not_rigid_links": True,
            "joint_topology_inferred": False,
            "simready_qualified": False,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    return _seal(payload, "receipt_digest")


def _build(*, task: str = "a") -> dict:
    return build_dual_task_joint_agent_admission(
        publisher_scene_id=SCENE_ID,
        task_freeze=_task_freeze(task=task),
        source_receipt=_source_receipt(task=task),
    )


def test_scene840920_washer_exact_five_joint_freeze_is_admitted() -> None:
    admission = _build()

    assert admission["status"] == READY_STATUS
    assert admission["task"] == {
        "publisher_scene_id": "840920",
        "task_id": "task_a_washer_door_open",
        "task_kind": "articulated_interaction",
        "task_freeze_digest": admission["task_freeze"]["task_freeze_digest"],
        "scene_freeze_digest": _digest("8"),
        "source_instance_id": "165",
        "articulation_graph_digest": canonical_digest(_washer_graph()),
        "frozen_assembly_joint_count": 5,
        "target_joint_id": "door_hinge",
    }
    scope = admission["scope_amendment"]["joint_scope"]
    assert scope["maximum_assembly_joint_count"] == 5
    assert scope["frozen_assembly_joint_count"] == 5
    assert scope["non_task_joint_mode"] == NON_TASK_MODE
    assert scope["non_task_joint_roles"] == ["dependent", "locked", "passive"]
    assert admission["source"]["source_asset_sha256"] == TASK_A_SOURCE_SHA
    assert admission["source"]["connected_component_count"] == 71
    assert admission["claim_boundary"]["non_task_joint_behavior_exercised"] is False
    assert validate_dual_task_joint_agent_admission(admission) == admission


def test_notebook_locked_hinge_is_typed_inapplicable_without_execution_claim() -> None:
    admission = _build(task="b")

    assert admission["status"] == INAPPLICABLE_STATUS
    assert admission["paid_joint_agent_execution_permitted"] is False
    assert admission["reason"] == "rigid_task_with_only_locked_preexisting_joints"
    assert admission["claim_boundary"]["locked_joint_exercised"] is False
    assert admission["claim_boundary"]["joint_agent_inference_executed"] is False
    assert "normalized_freeze" not in admission
    assert validate_dual_task_joint_agent_admission(admission) == admission


def test_source_instance_or_collision_binding_mismatch_fails_closed() -> None:
    receipt = _source_receipt(task="a")
    receipt["target"]["interiorgs_instance_id"] = "385"
    _seal(receipt, "receipt_digest")

    with pytest.raises(
        DualTaskJointAgentAdmissionError,
        match="joint_agent_dual_task_source_binding_invalid",
    ):
        build_dual_task_joint_agent_admission(
            publisher_scene_id=SCENE_ID,
            task_freeze=_task_freeze(task="a"),
            source_receipt=receipt,
        )


def test_six_joint_assembly_exceeds_the_preregistered_cap() -> None:
    freeze = _task_freeze(task="a")
    graph = freeze["articulation_graph"]
    graph["links"].append(
        {"link_id": "extra", "is_root": False, "semantic_role": "extra_candidate"}
    )
    graph["joints"].append(
        {
            "joint_id": "extra_joint",
            "parent_link_id": "body",
            "child_link_id": "extra",
            "joint_type": "revolute",
            "role": "locked",
            "axis": [1.0, 0.0, 0.0],
            "limits": [0.0, 1.0],
            "reset_position": 0.0,
            "reset_tolerance": 0.001,
            "drive": _drive(stiffness=100.0),
            "dependency": None,
        }
    )
    _seal(freeze, "task_freeze_digest")

    with pytest.raises(
        DualTaskJointAgentAdmissionError,
        match="joint_agent_assembly_joint_count_out_of_range",
    ):
        build_dual_task_joint_agent_admission(
            publisher_scene_id=SCENE_ID,
            task_freeze=freeze,
            source_receipt=_source_receipt(task="a"),
        )


def test_target_scope_cannot_be_redirected_to_a_locked_joint() -> None:
    freeze = _task_freeze(task="a")
    freeze["target_configuration"] = {
        "kind": "joint_interval",
        "target_joint_ids": ["selector_axis"],
        "joint_intervals": {"selector_axis": [0.7, 0.95]},
    }
    freeze["articulation_graph"]["success_predicate"]["joint_intervals"] = {
        "selector_axis": [0.7, 0.95]
    }
    _seal(freeze, "task_freeze_digest")

    with pytest.raises(
        DualTaskJointAgentAdmissionError,
        match="articulation_graph_success_joint_set_invalid",
    ):
        build_dual_task_joint_agent_admission(
            publisher_scene_id=SCENE_ID,
            task_freeze=freeze,
            source_receipt=_source_receipt(task="a"),
        )


def test_mutated_claim_ceiling_invalidates_the_admission() -> None:
    admission = _build()
    mutated = copy.deepcopy(admission)
    mutated["claim_boundary"]["simready_qualified"] = True
    _seal(mutated, "admission_digest")

    with pytest.raises(
        DualTaskJointAgentAdmissionError,
        match="joint_agent_dual_task_admission_rebuild_mismatch",
    ):
        validate_dual_task_joint_agent_admission(mutated)


def test_no_spend_cli_binds_exact_source_asset_bytes(tmp_path: Path) -> None:
    source_asset = tmp_path / "articulated_source_mesh.usda"
    source_asset.write_bytes(b"#usda 1.0\ndef Xform \"Asset\" {}\n")
    receipt = _source_receipt(task="a")
    receipt["output_asset"].update(
        {
            "sha256": "sha256:" + hashlib.sha256(source_asset.read_bytes()).hexdigest(),
            "size_bytes": source_asset.stat().st_size,
        }
    )
    _seal(receipt, "receipt_digest")
    freeze_path = tmp_path / "task_a.json"
    receipt_path = tmp_path / "source_receipt.json"
    output_path = tmp_path / "admission.json"
    freeze_path.write_text(json.dumps(_task_freeze(task="a")), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert main(
        [
            "--publisher-scene-id",
            "840920",
            "--task-freeze",
            str(freeze_path),
            "--source-receipt",
            str(receipt_path),
            "--source-asset",
            str(source_asset),
            "--output",
            str(output_path),
        ]
    ) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == READY_STATUS
    assert written["source"]["source_asset_sha256"] == receipt["output_asset"][
        "sha256"
    ]
