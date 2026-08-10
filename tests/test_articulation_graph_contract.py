from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.articulation_graph_contract import (
    ArticulationGraphContractError,
    validate_articulation_graph,
)


def complete_graph() -> dict:
    links = [
        {"link_id": "body", "is_root": True, "semantic_role": "fixed_body"},
        {"link_id": "door", "is_root": False, "semantic_role": "target_door"},
        {"link_id": "latch", "is_root": False, "semantic_role": "dependent_latch"},
        {"link_id": "drum", "is_root": False, "semantic_role": "passive_drum"},
        {"link_id": "selector", "is_root": False, "semantic_role": "locked_selector"},
        {"link_id": "drawer", "is_root": False, "semantic_role": "locked_drawer"},
        {"link_id": "panel", "is_root": False, "semantic_role": "fixed_panel"},
    ]

    def joint(
        joint_id: str,
        child: str,
        joint_type: str,
        role: str,
        limits: list[float],
        *,
        parent: str = "body",
        dependency: dict | None = None,
    ) -> dict:
        return {
            "joint_id": joint_id,
            "parent_link_id": parent,
            "child_link_id": child,
            "joint_type": joint_type,
            "role": role,
            "axis": [0.0, 0.0, 0.0] if joint_type == "fixed" else [0.0, 0.0, 1.0],
            "limits": limits,
            "reset_position": 0.0,
            "reset_tolerance": 0.0001,
            "drive": {
                "drive_type": "none" if role == "passive" else "force",
                "stiffness": 0.0 if role == "passive" else 20.0,
                "damping": 0.1,
                "maximum_force": 0.0 if role == "passive" else 100.0,
            },
            "dependency": dependency,
        }

    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": links,
        "joints": [
            joint("door_hinge", "door", "revolute", "target", [0.0, 1.2]),
            joint(
                "latch_coupler",
                "latch",
                "revolute",
                "dependent",
                [-0.2, 0.2],
                parent="door",
                dependency={
                    "driver_joint_id": "door_hinge",
                    "multiplier": 0.1,
                    "offset": 0.0,
                    "tolerance": 0.001,
                },
            ),
            joint("drum_bearing", "drum", "continuous", "passive", [-100.0, 100.0]),
            joint("selector_axis", "selector", "revolute", "locked", [-3.2, 3.2]),
            joint("detergent_slide", "drawer", "prismatic", "locked", [0.0, 0.2]),
            joint("service_panel_weld", "panel", "fixed", "locked", [0.0, 0.0]),
        ],
        "collision_pairs": [
            {"link_a": "body", "link_b": "door", "collision_enabled": True},
            {"link_a": "door", "link_b": "latch", "collision_enabled": False},
        ],
        "success_predicate": {
            "combination": "all",
            "joint_intervals": {"door_hinge": [0.7, 1.0]},
        },
    }


def test_complete_graph_has_no_universal_joint_count_cap() -> None:
    graph = validate_articulation_graph(complete_graph())

    assert len(graph["links"]) == 7
    assert len(graph["joints"]) == 6
    assert {joint["role"] for joint in graph["joints"]} == {
        "target",
        "dependent",
        "passive",
        "locked",
    }


def test_graph_rejects_unbound_drive_and_cycle() -> None:
    graph = complete_graph()
    del graph["joints"][0]["drive"]
    graph["joints"][0]["parent_link_id"] = "latch"

    with pytest.raises(ArticulationGraphContractError) as caught:
        validate_articulation_graph(graph)

    assert "articulation_graph_cycle_detected" in caught.value.errors
    assert any("joint_drive_invalid" in error for error in caught.value.errors)


def test_success_predicate_must_name_exact_target_set() -> None:
    graph = copy.deepcopy(complete_graph())
    graph["success_predicate"]["joint_intervals"] = {"selector_axis": [0.2, 0.4]}

    with pytest.raises(ArticulationGraphContractError) as caught:
        validate_articulation_graph(graph)

    assert "articulation_graph_success_joint_set_invalid" in caught.value.errors


def test_rigid_task_subject_may_bind_an_all_locked_articulation_graph() -> None:
    graph = complete_graph()
    graph["links"] = graph["links"][:2]
    graph["links"][1]["semantic_role"] = "locked_panel"
    [joint] = graph["joints"][:1]
    joint["role"] = "locked"
    graph["joints"] = [joint]
    graph["collision_pairs"] = [
        {"link_a": "body", "link_b": "door", "collision_enabled": True}
    ]
    graph["success_predicate"]["joint_intervals"] = {}

    locked = validate_articulation_graph(graph, require_target_joint=False)

    assert locked["joints"][0]["role"] == "locked"
    assert locked["success_predicate"]["joint_intervals"] == {}

    graph["joints"][0]["role"] = "passive"
    with pytest.raises(
        ArticulationGraphContractError,
        match="articulation_graph_rigid_subject_joint_not_locked",
    ):
        validate_articulation_graph(graph, require_target_joint=False)
