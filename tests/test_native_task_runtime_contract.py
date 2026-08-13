from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_runtime_contract import (
    DROID_FRANKA_RESET_JOINT_NAMES,
    FROZEN_CANDIDATES,
    NativeTaskRuntimeContractError,
    load_native_task_runtime_contract,
    materialize_native_task_runtime_contract,
)
from blueprint_pipeline.replacement_construction_bindings import (
    GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
    MASK_SET_QUALIFICATION_SCHEMA_VERSION,
    REPLACEMENT_QUALIFICATION_SCHEMA_VERSION,
    SOURCE_COLLIDER_DELETION_SCHEMA_VERSION,
    seal_replacement_construction_bindings,
)
from blueprint_pipeline.paired_target_native_construction_bindings import (
    validate_paired_target_native_construction_bindings,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _evidence_record(name: str, digest: str, schema_version: str) -> dict:
    return {
        "path": f"/fixture/{name}.json",
        "size_bytes": 1,
        "sha256": _sha("a"),
        "schema_version": schema_version,
        "canonical_digest": digest,
    }


def _construction_row_with_evidence(row: dict) -> dict:
    row = json.loads(json.dumps(row))
    row["evidence_receipts"] = {
        "task_freeze": _evidence_record(
            "task_freeze", row["task_freeze_digest"], "dual_task_task_freeze.v1"
        ),
        "mask_set": _evidence_record(
            "mask_set",
            row["mask_set_receipt_digest"],
            MASK_SET_QUALIFICATION_SCHEMA_VERSION,
        ),
        "gaussian_removal": _evidence_record(
            "gaussian_removal",
            row["source_removal_receipt_digest"],
            GAUSSIAN_REMOVAL_QUALIFICATION_SCHEMA_VERSION,
        ),
        "source_collider_deletion": {
            "selected_deletion_id": row["collider_deletion_id"],
            "independent": _evidence_record(
                "source_collider_deletion",
                row["collider_deletion_receipt_digest"],
                SOURCE_COLLIDER_DELETION_SCHEMA_VERSION,
            ),
        },
        "replacement_qualification": _evidence_record(
            "replacement_qualification",
            row["replacement_qualification_receipt_digest"],
            REPLACEMENT_QUALIFICATION_SCHEMA_VERSION,
        ),
    }
    return row


def _materialized_construction(value: dict) -> dict:
    result = json.loads(json.dumps(value))
    result["scene_freeze_receipt"] = _evidence_record(
        "scene_freeze",
        result["scene_freeze_digest"],
        "dual_task_scene_freeze.v1",
    )
    result["construction_digest"] = canonical_digest(
        result,
        digest_field="construction_digest",
    )
    return result


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> dict:
    return {
        "position_world_m": [x, y, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _camera(role: str) -> dict:
    wrist = role == "wrist"
    return {
        "role": role,
        "policy_input": role in {"external", "wrist"},
        "scoring_input": False,
        "pose_frame": "robot_body" if wrist else "world",
        "parent_prim_path": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
            if wrist
            else "{ENV_REGEX_NS}"
        ),
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "intrinsics": {
            "fx": 300.0,
            "fy": 300.0,
            "cx": 159.5,
            "cy": 119.5,
            "width": 320,
            "height": 240,
        },
    }


def _common(task_spec: dict, *, scene_id: str, task_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "task_id": task_id,
        "task_spec": task_spec,
        "assets": [
            {
                "semantic_role": "scene_collision",
                "filename": "scene_collision.usd",
                "sha256": _sha("a"),
                "pose_world": _pose(),
            },
            {
                "semantic_role": "scene_appearance",
                "filename": "scene_appearance.usdz",
                "sha256": _sha("b"),
                "pose_world": _pose(),
            },
            {
                "semantic_role": "task_object",
                "filename": "task_object.usda",
                "sha256": _sha("c"),
                "pose_world": _pose(1.0, 2.0, 0.0),
            },
        ],
        "robot_base_pose_world": _pose(1.75, 1.99, 0.0),
        "robot_joint_reset_positions_rad": {
            name: float(index) / 100.0
            for index, name in enumerate(DROID_FRANKA_RESET_JOINT_NAMES)
        },
        "cameras": [_camera(role) for role in ("external", "wrist", "overview")],
        "scenario_cell_id": "canonical_seed_17",
        "scenario_instance_digest": _sha("d"),
        "scenario_context_kind": "evaluation_cell",
        "seed": 17,
    }


def _rigid_fixture() -> dict:
    fixture = _common(
        {
            "task_kind": "rigid_pick_place",
            "prompt": "Pick up the can and place it in the destination.",
        },
        scene_id="840313",
        task_id="840313_canned_beverage_pick_place_v1",
    )
    fixture["task_joint_bindings"] = []
    return fixture


def _articulated_fixture() -> dict:
    freeze = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/arm_decision_proof_v1/manifests"
            / "second_scene_840796_scene_task_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )
    fixture = _common(
        freeze["task_spec"],
        scene_id="840796",
        task_id="840796_refrigerator_upper_door_open_v1",
    )
    fixture["task_joint_bindings"] = [
        {
            "joint_id": "refrigerator_upper_door_hinge",
            "joint_prim_path": "/Asset/joints/upper_door_hinge",
            "native_joint_name": "upper_door_hinge",
            "role": "task_joint",
        },
        {
            "joint_id": "refrigerator_lower_door_hinge",
            "joint_prim_path": "/Asset/joints/lower_door_hinge",
            "native_joint_name": "lower_door_hinge",
            "role": "locked_joint",
        },
    ]
    fixture["task_state_binding"] = {
        "moving_link_prim_path": "/Asset/upper_door",
        "moving_link_native_body_name": "upper_door",
        "handle_prim_paths": [
            "/Asset/upper_door/component_004",
            "/Asset/upper_door/handle_post_a",
            "/Asset/upper_door/handle_post_b",
        ],
        "handle_grasp_point_link_m": [0.119962, 0.327634, 1.022997],
        "robot_gripper_contact_prim_pattern": (
            "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/.*"
        ),
        "robot_collision_prim_pattern": "{ENV_REGEX_NS}/Robot/.*",
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "retreat_minimum_separation_m": 0.10,
        "root_translation_tolerance_m": 0.002,
        "root_orientation_tolerance_rad": 0.01,
    }
    return fixture


def _graph_articulated_fixture() -> dict:
    freeze = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/arm_decision_proof_v1/manifests"
            / "third_scene_840920_task_a_freeze.v1.json"
        ).read_text(encoding="utf-8")
    )
    graph = json.loads(json.dumps(freeze["articulation_graph"]))
    task_spec = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "articulated_open_close",
        "articulation_graph": graph,
    }
    contact_link_id = next(
        link["link_id"] for link in graph["links"] if not link["is_root"]
    )
    affordance = {
        "schema_version": "native_articulated_graph_interaction_affordance.v1",
        "contact_link_id": contact_link_id,
        "contact_body_prim_paths": [f"/Asset/links/{contact_link_id}"],
        "contact_point_link_m": [0.1, 0.0, 0.0],
        "affordance_digest": "",
    }
    affordance["affordance_digest"] = canonical_digest(
        affordance, digest_field="affordance_digest"
    )
    task_spec["interaction_affordance"] = affordance
    fixture = _common(
        task_spec,
        scene_id="graph_fixture_scene",
        task_id="graph_fixture_task",
    )
    fixture["task_joint_bindings"] = [
        {
            "joint_id": joint["joint_id"],
            "joint_prim_path": f"/Asset/joints/{joint['joint_id']}",
            "native_joint_name": (
                None if joint["joint_type"] == "fixed" else joint["joint_id"]
            ),
            "role": joint["role"],
            "readback_kind": (
                "fixed_joint_static"
                if joint["joint_type"] == "fixed"
                else "native_coordinate"
            ),
            "static_qualification_digest": (
                _sha("f") if joint["joint_type"] == "fixed" else None
            ),
        }
        for joint in graph["joints"]
    ]
    fixture["task_state_binding"] = {
        "schema_version": "native_articulated_graph_task_state_binding.v1",
        "articulation_graph_digest": canonical_digest(graph),
        "interaction_affordance_digest": task_spec["interaction_affordance"][
            "affordance_digest"
        ],
        "link_native_body_names": {
            link["link_id"]: link["link_id"] for link in graph["links"]
        },
        "task_contact_minimum_force_n": 0.5,
        "collision_failure_minimum_force_n": 1.0,
        "retreat_minimum_separation_m": 0.10,
        "root_translation_tolerance_m": 0.002,
        "root_orientation_tolerance_rad": 0.01,
    }
    return fixture


def _dual_replacement_assets() -> list[dict]:
    return [
        {
            "semantic_role": "scene_collision",
            "filename": "scene_collision.usd",
            "sha256": _sha("a"),
            "pose_world": _pose(),
        },
        {
            "semantic_role": "scene_appearance",
            "filename": "scene_appearance.usdz",
            "sha256": _sha("b"),
            "pose_world": _pose(),
        },
        {
            "semantic_role": "replacement",
            "asset_id": "articulated_a",
            "filename": "articulated_a.usda",
            "sha256": _sha("c"),
            "pose_world": _pose(1.0, 2.0, 0.0),
            "object_type": "ARTICULATION",
            "reset_state": {
                "joint_positions": {
                    "upper_door_hinge": 0.0,
                    "lower_door_hinge": 0.0,
                }
            },
        },
        {
            "semantic_role": "replacement",
            "asset_id": "rigid_b",
            "filename": "rigid_b.usda",
            "sha256": _sha("e"),
            "pose_world": _pose(2.0, 3.0, 0.8),
            "object_type": "RIGID",
            "reset_state": {"joint_positions": {}},
        },
    ]


def _construction_bindings() -> dict:
    rows = [
        {
            "task_id": "840796_refrigerator_upper_door_open_v1",
            "asset_id": "articulated_a",
            "task_freeze_digest": _sha("3"),
            "source_object_instance_id": "source_a",
            "removal_id": "removal_a",
            "mask_set_id": "masks_a",
            "mask_set_receipt_digest": _sha("4"),
            "source_removal_receipt_digest": _sha("5"),
            "source_removal_qualified": True,
            "collider_deletion_id": "collider_delete_a",
            "source_collider_prim_path": "/Root/source_a",
            "collider_deletion_receipt_digest": _sha("6"),
            "collider_deletion_qualified": True,
            "replacement_qualification_id": "replacement_qualification_a",
            "replacement_qualification_receipt_digest": _sha("7"),
            "replacement_asset_sha256": _sha("c"),
            "replacement_simulator_import_qualified": True,
        },
        {
            "task_id": "840313_canned_beverage_pick_place_v1",
            "asset_id": "rigid_b",
            "task_freeze_digest": _sha("8"),
            "source_object_instance_id": "source_b",
            "removal_id": "removal_b",
            "mask_set_id": "masks_b",
            "mask_set_receipt_digest": _sha("9"),
            "source_removal_receipt_digest": _sha("f"),
            "source_removal_qualified": True,
            "collider_deletion_id": "collider_delete_b",
            "source_collider_prim_path": "/Root/source_b",
            "collider_deletion_receipt_digest": _sha("0"),
            "collider_deletion_qualified": True,
            "replacement_qualification_id": "replacement_qualification_b",
            "replacement_qualification_receipt_digest": _sha("b"),
            "replacement_asset_sha256": _sha("e"),
            "replacement_simulator_import_qualified": True,
        },
    ]
    return _materialized_construction(
        seal_replacement_construction_bindings(
            scene_freeze_digest=_sha("1"),
            task_freeze_join_digest=_sha("2"),
            bindings=[_construction_row_with_evidence(row) for row in rows],
        )
    )


@pytest.mark.parametrize(
    ("fixture", "expected_type", "expected_joints"),
    [
        (_rigid_fixture, "RIGID", []),
        (
            _articulated_fixture,
            "ARTICULATION",
            [
                "refrigerator_lower_door_hinge",
                "refrigerator_upper_door_hinge",
            ],
        ),
    ],
)
def test_original_and_second_scene_share_one_runtime_contract(
    fixture, expected_type: str, expected_joints: list[str]
) -> None:
    contract = materialize_native_task_runtime_contract(**fixture())

    objects = {row["semantic_role"]: row for row in contract["objects"]}
    assert objects["task_object"]["object_type"] == expected_type
    assert objects["scene_collision"]["visible"] is False
    assert objects["scene_appearance"]["visible"] is True
    assert contract["task_sample_binding"]["joint_ids"] == expected_joints
    assert contract["candidate_ids"] == list(FROZEN_CANDIDATES)
    assert contract["scenario"]["context_kind"] == "evaluation_cell"
    assert contract["robot"]["action_seam"]["action_dimension"] == 8
    assert set(contract["robot"]["joint_reset_positions_rad"]) == set(
        DROID_FRANKA_RESET_JOINT_NAMES
    )
    if expected_type == "ARTICULATION":
        assert contract["task_state_binding"]["measurement_authority"][
            "caller_asserted_booleans_forbidden"
        ] is True
    else:
        assert contract["task_state_binding"] is None


def test_general_graph_articulation_uses_graph_state_binding_without_handle_fields() -> None:
    contract = materialize_native_task_runtime_contract(
        **_graph_articulated_fixture()
    )

    state = contract["task_state_binding"]
    sample = contract["task_sample_binding"]
    assert state["schema_version"] == (
        "native_articulated_graph_task_state_binding.v1"
    )
    assert "moving_link_prim_path" not in state
    assert "handle_prim_paths" not in state
    assert set(state["link_native_body_names"]) == {
        link["link_id"]
        for link in contract["task_spec"]["articulation_graph"]["links"]
    }
    assert set(sample["native_coordinate_joint_ids"]).isdisjoint(
        sample["fixed_joint_ids"]
    )
    assert set(sample["native_coordinate_joint_ids"]).union(
        sample["fixed_joint_ids"]
    ) == set(sample["joint_ids"])


def test_two_tasks_select_distinct_subjects_from_one_shared_replacement_set() -> None:
    articulated = _articulated_fixture()
    articulated["assets"] = _dual_replacement_assets()
    articulated["task_spec"]["subject_asset_id"] = "articulated_a"
    articulated["construction_bindings"] = _construction_bindings()
    articulated["task_freeze_digest"] = _sha("3")
    rigid = _rigid_fixture()
    rigid["assets"] = _dual_replacement_assets()
    rigid["task_spec"]["subject_asset_id"] = "rigid_b"
    rigid["construction_bindings"] = _construction_bindings()
    rigid["task_freeze_digest"] = _sha("8")

    contract_a = materialize_native_task_runtime_contract(**articulated)
    contract_b = materialize_native_task_runtime_contract(**rigid)

    assert {row["asset_id"] for row in contract_a["objects"]} == {
        row["asset_id"] for row in contract_b["objects"]
    }
    assert contract_a["task_subject_asset_id"] == "articulated_a"
    assert contract_b["task_subject_asset_id"] == "rigid_b"
    assert next(row for row in contract_a["objects"] if row["task_subject"])[
        "semantic_role"
    ] == "task_object"
    inactive_a = next(
        row for row in contract_b["objects"] if row["asset_id"] == "articulated_a"
    )
    assert inactive_a["semantic_role"] == "replacement"
    assert inactive_a["object_type"] == "ARTICULATION"
    assert inactive_a["reset_state"]["joint_positions"] == {
        "lower_door_hinge": 0.0,
        "upper_door_hinge": 0.0,
    }
    assert set(contract_a["reset_contract"]["per_object_reset_states"]) == {
        "articulated_a",
        "rigid_b",
    }
    assert (
        contract_a["construction_bindings"]["construction_digest"]
        == contract_b["construction_bindings"]["construction_digest"]
    )


def test_runtime_accepts_paired_target_construction_binding_without_legacy_receipts() -> None:
    articulated = _articulated_fixture()
    rigid = _rigid_fixture()
    rows = []
    for task_id, asset_id, freeze, asset_sha in (
        (articulated["task_id"], "articulated_a", _sha("3"), _sha("c")),
        (rigid["task_id"], "rigid_b", _sha("8"), _sha("e")),
    ):
        registered_digest = _sha("7" if asset_id == "articulated_a" else "b")
        probe_digest = _sha("6" if asset_id == "articulated_a" else "a")
        rows.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze_digest": freeze,
                "registered_asset_receipt_digest": registered_digest,
                "replacement_asset_sha256": asset_sha,
                "native_import_probe_result_digest": probe_digest,
                "native_simulator_import_qualified": True,
                "evidence_receipts": {
                    "task_freeze": _evidence_record(
                        f"{asset_id}_freeze",
                        freeze,
                        "dual_task_task_freeze.v1",
                    ),
                    "registered_asset": _evidence_record(
                        f"{asset_id}_registered",
                        registered_digest,
                        "registered_replacement_asset.v1",
                    ),
                    "registered_usd": {
                        "path": f"/fixture/{asset_id}.usda",
                        "size_bytes": 1,
                        "sha256": asset_sha,
                    },
                    "native_import_probe": _evidence_record(
                        f"{asset_id}_probe",
                        probe_digest,
                        "simready_replacement_native_import_probe_result.v1",
                    ),
                },
            }
        )
    paired = {
        "schema_version": "paired_target_native_construction_bindings.v1",
        "status": "paired_targets_admitted_for_native_construction",
        "scene_id": articulated["scene_id"],
        "task_freeze_set_digest": _sha("2"),
        "replacement_object_count": 2,
        "bindings": sorted(rows, key=lambda row: row["asset_id"]),
        "native_camera_readback_qualified": False,
        "native_reachability_qualified": False,
        "controls_executed": False,
        "learned_policies_executed": False,
        "construction_digest": "",
    }
    paired["construction_digest"] = canonical_digest(
        paired, digest_field="construction_digest"
    )
    paired = validate_paired_target_native_construction_bindings(paired)
    articulated["assets"] = _dual_replacement_assets()
    articulated["task_spec"]["subject_asset_id"] = "articulated_a"
    articulated["construction_bindings"] = paired
    articulated["task_freeze_digest"] = _sha("3")

    contract = materialize_native_task_runtime_contract(**articulated)

    assert contract["task_subject_asset_id"] == "articulated_a"
    assert contract["construction_bindings"]["schema_version"] == (
        "paired_target_native_construction_bindings.v1"
    )


def _five_replacement_fixture() -> dict:
    fixture = _rigid_fixture()
    scene_assets = _dual_replacement_assets()[:2]
    replacement_assets = []
    binding_rows = []
    for index in range(5):
        asset_id = f"replacement_{index}"
        asset_sha = "sha256:" + f"{100 + index:064x}"
        replacement_assets.append(
            {
                "semantic_role": "replacement",
                "asset_id": asset_id,
                "filename": f"replacement_{index}.usda",
                "sha256": asset_sha,
                "pose_world": _pose(float(index), 3.0, 0.8),
                "object_type": "RIGID",
                "reset_state": {"joint_positions": {}},
            }
        )
        task_id = (
            fixture["task_id"] if index == 0 else f"inactive_task_{index}"
        )
        binding_rows.append(
            _construction_row_with_evidence(
                {
                    "task_id": task_id,
                    "asset_id": asset_id,
                    "task_freeze_digest": "sha256:" + f"{200 + index:064x}",
                    "source_object_instance_id": f"source_{index}",
                    "removal_id": f"removal_{index}",
                    "mask_set_id": f"masks_{index}",
                    "mask_set_receipt_digest": "sha256:" + f"{300 + index:064x}",
                    "source_removal_receipt_digest": "sha256:"
                    + f"{400 + index:064x}",
                    "source_removal_qualified": True,
                    "collider_deletion_id": f"collider_delete_{index}",
                    "source_collider_prim_path": f"/Root/source_{index}",
                    "collider_deletion_receipt_digest": "sha256:"
                    + f"{500 + index:064x}",
                    "collider_deletion_qualified": True,
                    "replacement_qualification_id": (
                        f"replacement_qualification_{index}"
                    ),
                    "replacement_qualification_receipt_digest": "sha256:"
                    + f"{600 + index:064x}",
                    "replacement_asset_sha256": asset_sha,
                    "replacement_simulator_import_qualified": True,
                }
            )
        )
    fixture["assets"] = [*scene_assets, *replacement_assets]
    fixture["task_spec"]["subject_asset_id"] = "replacement_0"
    fixture["task_freeze_digest"] = binding_rows[0]["task_freeze_digest"]
    fixture["construction_bindings"] = _materialized_construction(
        seal_replacement_construction_bindings(
            scene_freeze_digest=_sha("1"),
            task_freeze_set_digest=_sha("2"),
            bindings=binding_rows,
        )
    )
    return fixture


def test_runtime_preserves_five_copresent_replacements_and_one_subject() -> None:
    contract = materialize_native_task_runtime_contract(**_five_replacement_fixture())

    replacement_objects = [
        row
        for row in contract["objects"]
        if row["source_semantic_role"] == "replacement"
    ]
    assert len(replacement_objects) == 5
    assert sum(bool(row["task_subject"]) for row in replacement_objects) == 1
    assert set(contract["reset_contract"]["per_object_reset_states"]) == {
        f"replacement_{index}" for index in range(5)
    }


def test_digest_bound_replacement_asset_id_may_begin_with_scene_number() -> None:
    fixture = _five_replacement_fixture()
    asset = next(
        row
        for row in fixture["assets"]
        if row.get("asset_id") == "replacement_0"
    )
    asset["asset_id"] = "840920_replacement_0"
    fixture["task_spec"]["subject_asset_id"] = asset["asset_id"]
    fixture["construction_bindings"]["bindings"][0]["asset_id"] = asset["asset_id"]
    fixture["construction_bindings"]["construction_digest"] = canonical_digest(
        fixture["construction_bindings"], digest_field="construction_digest"
    )

    contract = materialize_native_task_runtime_contract(**fixture)

    subject = next(row for row in contract["objects"] if row["task_subject"])
    assert subject["asset_id"] == "840920_replacement_0"
    assert subject["runtime_name"] == "task_object"


def test_runtime_accepts_single_repeatable_replacement_with_construction_binding() -> None:
    fixture = _five_replacement_fixture()
    fixture["assets"] = [
        row
        for row in fixture["assets"]
        if row.get("semantic_role") != "replacement"
        or row.get("asset_id") == "replacement_0"
    ]
    fixture["construction_bindings"] = _materialized_construction(
        seal_replacement_construction_bindings(
            scene_freeze_digest=_sha("1"),
            task_freeze_set_digest=_sha("2"),
            bindings=fixture["construction_bindings"]["bindings"][:1],
        )
    )

    contract = materialize_native_task_runtime_contract(**fixture)

    replacement_objects = [
        row
        for row in contract["objects"]
        if row["source_semantic_role"] == "replacement"
    ]
    assert len(replacement_objects) == 1
    assert replacement_objects[0]["task_subject"] is True
    assert contract["task_subject_asset_id"] == "replacement_0"
    assert len(contract["construction_bindings"]["bindings"]) == 1


def test_runtime_rejects_sixth_replacement_before_scene_build() -> None:
    fixture = _five_replacement_fixture()
    fixture["assets"].append(
        {
            "semantic_role": "replacement",
            "asset_id": "replacement_5",
            "filename": "replacement_5.usda",
            "sha256": "sha256:" + f"{105:064x}",
            "pose_world": _pose(5.0, 3.0, 0.8),
            "object_type": "RIGID",
            "reset_state": {"joint_positions": {}},
        }
    )

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_replacement_asset_count_out_of_range" in (
        excinfo.value.errors
    )


def test_rigid_manipulation_preserves_locked_articulated_subject_spawn_type() -> None:
    rigid = _rigid_fixture()
    assets = _dual_replacement_assets()
    subject = next(row for row in assets if row.get("asset_id") == "rigid_b")
    subject["object_type"] = "ARTICULATION"
    subject["reset_state"] = {"joint_positions": {"display_hinge": 0.0}}
    rigid["assets"] = assets
    rigid["task_spec"]["subject_asset_id"] = "rigid_b"
    rigid["construction_bindings"] = _construction_bindings()
    rigid["task_freeze_digest"] = _sha("8")

    contract = materialize_native_task_runtime_contract(**rigid)

    planned = next(row for row in contract["objects"] if row["task_subject"])
    assert planned["object_type"] == "ARTICULATION"
    assert planned["reset_state"]["joint_positions"] == {"display_hinge": 0.0}


def test_dual_replacement_contract_rejects_shared_mask_or_swapped_asset() -> None:
    fixture = _articulated_fixture()
    fixture["assets"] = _dual_replacement_assets()
    fixture["task_spec"]["subject_asset_id"] = "articulated_a"
    bindings = _construction_bindings()
    bindings["bindings"][1]["mask_set_id"] = bindings["bindings"][0]["mask_set_id"]
    bindings["construction_digest"] = canonical_digest(
        bindings, digest_field="construction_digest"
    )
    fixture["construction_bindings"] = bindings
    fixture["task_freeze_digest"] = _sha("3")

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "replacement_construction_shared_identity:mask_set_id" in excinfo.value.errors

    fixture["construction_bindings"] = _construction_bindings()
    first, second = fixture["construction_bindings"]["bindings"]
    first["replacement_asset_sha256"], second["replacement_asset_sha256"] = (
        second["replacement_asset_sha256"],
        first["replacement_asset_sha256"],
    )
    fixture["construction_bindings"]["construction_digest"] = canonical_digest(
        fixture["construction_bindings"], digest_field="construction_digest"
    )
    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)
    assert (
        "native_task_runtime_replacement_qualification_asset_mismatch:articulated_a"
        in excinfo.value.errors
    )


def test_native_runtime_rejects_caller_authored_construction_without_evidence() -> None:
    fixture = _five_replacement_fixture()
    caller_rows = []
    for row in fixture["construction_bindings"]["bindings"]:
        caller_row = dict(row)
        caller_row.pop("evidence_receipts", None)
        caller_rows.append(caller_row)
    fixture["construction_bindings"] = seal_replacement_construction_bindings(
        scene_freeze_digest=fixture["construction_bindings"]["scene_freeze_digest"],
        task_freeze_set_digest=fixture["construction_bindings"][
            "task_freeze_set_digest"
        ],
        bindings=caller_rows,
    )

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "replacement_construction_scene_freeze_receipt_invalid" in (
        excinfo.value.errors
    )


def test_repeatable_replacements_require_qualified_construction_bindings() -> None:
    fixture = _rigid_fixture()
    fixture["assets"] = _dual_replacement_assets()
    fixture["task_spec"]["subject_asset_id"] = "rigid_b"

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_construction_bindings_missing" in excinfo.value.errors
    assert "native_task_runtime_task_freeze_digest_invalid" in excinfo.value.errors


def test_shared_replacement_set_rejects_duplicate_or_missing_subject_identity() -> None:
    fixture = _rigid_fixture()
    fixture["assets"] = _dual_replacement_assets()
    fixture["task_spec"]["subject_asset_id"] = "missing"
    fixture["assets"][3]["asset_id"] = "articulated_a"
    fixture["construction_bindings"] = _construction_bindings()
    fixture["task_freeze_digest"] = _sha("8")

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_replacement_asset_ids_invalid" in excinfo.value.errors
    assert "native_task_runtime_subject_asset_id_invalid" in excinfo.value.errors


def test_policy_and_review_camera_roles_cannot_be_swapped() -> None:
    fixture = _articulated_fixture()
    fixture["cameras"][2]["policy_input"] = True

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_camera_policy_role_invalid:overview" in excinfo.value.errors


def test_runtime_contract_distinguishes_construction_canary_from_evaluation() -> None:
    fixture = _articulated_fixture()
    fixture["scenario_context_kind"] = "construction_canary"

    contract = materialize_native_task_runtime_contract(**fixture)

    assert contract["scenario"]["context_kind"] == "construction_canary"

    fixture["scenario_context_kind"] = "caller_claimed_episode"
    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)
    assert excinfo.value.errors == (
        "native_task_runtime_scenario_context_kind_invalid",
    )


def test_camera_pose_must_be_rigid_and_opencv_calibrated() -> None:
    fixture = _articulated_fixture()
    fixture["cameras"][0]["frame_from_camera_matrix"][0] = 2.0
    fixture["cameras"][1]["optical_convention"] = "unspecified"

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_camera_pose_invalid:external" in excinfo.value.errors
    assert "native_task_runtime_camera_convention_invalid:wrist" in excinfo.value.errors


def test_wrist_camera_requires_exact_robot_body_parent() -> None:
    fixture = _articulated_fixture()
    fixture["cameras"][1]["parent_prim_path"] = "{ENV_REGEX_NS}/Robot/panda_hand"

    contract = materialize_native_task_runtime_contract(**fixture)

    assert contract["cameras"][1]["parent_prim_path"] == (
        "{ENV_REGEX_NS}/Robot/panda_hand"
    )

    fixture["cameras"][1]["parent_prim_path"] = "{ENV_REGEX_NS}/scene_collision"
    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)
    assert "native_task_runtime_camera_parent_invalid:wrist" in excinfo.value.errors


def test_runtime_contract_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    path = tmp_path / "native_task_runtime_contract.json"
    created = materialize_native_task_runtime_contract(
        **_articulated_fixture(), destination=path
    )

    assert load_native_task_runtime_contract(path) == created
    tampered = json.loads(path.read_text())
    tampered["robot"]["base_pose_world"]["position_world_m"][0] += 0.1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        load_native_task_runtime_contract(path)
    assert excinfo.value.errors == ("native_task_runtime_contract_digest_invalid",)


def test_articulated_task_without_joint_binding_fails_before_gpu() -> None:
    fixture = _articulated_fixture()
    fixture["task_joint_bindings"] = []

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert any("composition_invalid" in error for error in excinfo.value.errors)


def test_articulated_task_without_native_state_binding_fails_before_gpu() -> None:
    fixture = _articulated_fixture()
    fixture["task_state_binding"] = None

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_state_binding_missing" in excinfo.value.errors


def test_state_binding_rejects_handle_outside_the_moving_link() -> None:
    fixture = _articulated_fixture()
    fixture["task_state_binding"]["handle_prim_paths"] = [
        "/Asset/lower_door/component_005"
    ]

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_handle_prim_invalid:0" in excinfo.value.errors


def test_state_binding_requires_explicit_native_moving_body_name() -> None:
    fixture = _articulated_fixture()
    fixture["task_state_binding"]["moving_link_native_body_name"] = ""

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert (
        "native_task_runtime_moving_link_body_name_invalid" in excinfo.value.errors
    )


def test_task_asset_path_cannot_escape_the_provider_asset_directory() -> None:
    fixture = _articulated_fixture()
    fixture["assets"][2]["filename"] = "../task_object.usda"

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_asset_filename_invalid:task_object" in excinfo.value.errors


def test_robot_reset_joint_map_rejects_missing_and_extra_names() -> None:
    fixture = _articulated_fixture()
    fixture["robot_joint_reset_positions_rad"].pop("panda_joint7")
    fixture["robot_joint_reset_positions_rad"]["scene_specific_joint"] = 0.0

    with pytest.raises(NativeTaskRuntimeContractError) as excinfo:
        materialize_native_task_runtime_contract(**fixture)

    assert "native_task_runtime_robot_reset_joint_missing:panda_joint7" in excinfo.value.errors
    assert (
        "native_task_runtime_robot_reset_joint_unexpected:scene_specific_joint"
        in excinfo.value.errors
    )
