from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.articulation_graph_contract import validate_articulation_graph
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_rehearsal_contract import validate_task_freeze
from blueprint_pipeline.native_task_runtime_contract import (
    DROID_FRANKA_RESET_JOINT_NAMES,
)
from blueprint_pipeline.paired_target_native_arena_request import (
    materialize_paired_target_native_arena_requests,
)
from blueprint_pipeline.paired_target_native_construction_bindings import (
    materialize_paired_target_native_construction_bindings,
)
from blueprint_pipeline.paired_target_native_manipulation_preflight import (
    PairedTargetNativeManipulationPreflightError,
    materialize_paired_target_native_manipulation_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "docs/arm_decision_proof_v1/manifests"
TASKS = (
    ("task_a_washer_door_open", "a", "washer_asset"),
    ("task_b_notebook_relocation", "b", "notebook_asset"),
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict, digest_field: str) -> Path:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _record(path: Path, **extra: object) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
        **extra,
    }


def _camera(role: str) -> dict:
    return {
        "role": role,
        "policy_input": role != "overview",
        "scoring_input": False,
        "pose_frame": "robot_body" if role == "wrist" else "world",
        "parent_prim_path": "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        "intrinsics": {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 80.0,
            "cy": 60.0,
            "width": 160,
            "height": 120,
        },
    }


def _fixture(root: Path) -> dict:
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    collision = evidence / "scene_collision.usda"
    collision.write_text('#usda 1.0\ndef Xform "Scene" {}\n', encoding="utf-8")
    removed_collision = evidence / "collider_removal" / "scene_without_source_colliders.usda"
    removed_collision.parent.mkdir(parents=True)
    removed_collision.write_text(
        '#usda 1.0\ndef Xform "RetainedScene" {}\n', encoding="utf-8"
    )
    target_removals = []
    for index in range(len(TASKS)):
        removal_id = f"remove_source_{index}"
        target_prim_path = f"/Root/source_{index}"
        child = _write(
            removed_collision.parent / "independent" / f"{removal_id}.json",
            {
                "schema_version": "source_collider_subtree_removal.v1",
                "status": "exact_source_collider_subtree_removed",
                "removal_id": removal_id,
                "sage_collision_usd_sha256": _sha(collision),
                "removed_prim_path": target_prim_path,
                "removed_prim_count": 1,
                "source_bytes_unchanged": True,
                "unrelated_prim_inventory_unchanged": True,
                "remaining_target_collision_prim_count": 0,
                "replacement_inserted": False,
                "receipt_digest": "",
            },
            "receipt_digest",
        )
        target_removals.append(
            {
                "removal_id": removal_id,
                "target_prim_path": target_prim_path,
                "source_scene_sha256": _sha(collision),
                "removed_prim_count": 1,
                "receipt_digest": json.loads(child.read_text())["receipt_digest"],
                "receipt": {
                    "relative_path": child.relative_to(removed_collision.parent).as_posix(),
                    "size_bytes": child.stat().st_size,
                    "sha256": _sha(child),
                },
            }
        )
    collider_batch = _write(
        removed_collision.parent / "source_collider_batch_removal.v1.json",
        {
            "schema_version": "source_collider_batch_removal.v1",
            "status": "independent_and_shared_source_colliders_removed",
            "source_scene_usd": _record(collision),
            "shared_removed_scene_usd": {
                "relative_path": removed_collision.name,
                "size_bytes": removed_collision.stat().st_size,
                "sha256": _sha(removed_collision),
            },
            "target_count": len(TASKS),
            "target_removals": target_removals,
            "source_bytes_unchanged": True,
            "unrelated_prim_inventory_unchanged": True,
            "remaining_target_collision_prim_count": 0,
            "replacement_inserted": False,
            "independent_receipts_share_exact_source_digest": True,
            "independent_removed_scenes_are_distinct": True,
            "receipt_digest": "",
        },
        "receipt_digest",
    )
    appearance = evidence / "scene_appearance.usdz"
    # a usdz is a zip, and the chain now refuses a payload whose bytes are
    # not the format its filename declares, so the fixture carries the
    # real magic rather than free text that could never open
    appearance.write_bytes(b"PK\x03\x04development-only appearance fixture")

    support = _write(
        evidence / "support.json",
        {
            "schema_version": "interiorgs_sage_collision_identity.v1",
            "whole_object_collision_identity_passed": True,
            "whole_object_matches": [{"prim_path": "/Scene/desk"}],
            "receipt_digest": "",
        },
        "receipt_digest",
    )
    support_value = json.loads(support.read_text())

    paired_tasks: list[dict] = []
    import_rows: list[dict] = []
    manipulation_tasks: list[dict] = []
    arena_inputs: list[dict] = []
    for task_id, letter, asset_id in TASKS:
        task_root = evidence / task_id
        base_freeze = json.loads(
            (
                MANIFESTS / f"third_scene_840920_task_{letter}_freeze.v1.json"
            ).read_text()
        )
        placement = _write(
            task_root / "placement.json",
            {
                "schema_version": "registered_sage_franka_placement_packet.v1",
                "status": "placement_candidate_materialized",
                "placement": {
                    "robot_pose_xyzyaw_collision_stage": [0.0, -1.0, 0.0, 0.0]
                },
                "target_analysis": {
                    "selected_target": {
                        "target_label": base_freeze["source_object"]["semantic_label"],
                        "position_m": base_freeze["source_object"][
                            "observed_pose_world"
                        ]["position_world_m"],
                    }
                },
                "native_contact_reachability_qualified": False,
                "policy_execution_authorized": False,
                "packet_digest": "",
            },
            "packet_digest",
        )
        placement_value = json.loads(placement.read_text())
        base_freeze["source_object"]["franka_placement_packet_digest"] = (
            placement_value["packet_digest"]
        )
        if letter == "b":
            base_freeze["source_object"]["support_receipt_digest"] = support_value[
                "receipt_digest"
            ]
        freeze = _write(
            task_root / "freeze.json", base_freeze, "task_freeze_digest"
        )
        freeze_value = validate_task_freeze(json.loads(freeze.read_text()))

        scenario_value = json.loads(
            (
                MANIFESTS / f"third_scene_840920_task_{letter}_scenario_suite.v1.json"
            ).read_text()
        )
        scenario_value["task_freeze_digest"] = freeze_value["task_freeze_digest"]
        scenario = _write(task_root / "scenario.json", scenario_value, "suite_digest")
        scenario_value = json.loads(scenario.read_text())

        usd = task_root / "registered.usda"
        usd.parent.mkdir(parents=True, exist_ok=True)
        # newline-separated metadata: pxr's usda parser rejects the
        # space-separated form, and the request compiler now opens this stage
        # to ground articulated assets
        usd.write_text(
            '#usda 1.0\n(\n    defaultPrim = "Asset"\n'
            '    metersPerUnit = 1\n    upAxis = "Z"\n)\n'
            'def Xform "Asset" {}\n',
            encoding="utf-8",
        )
        usd_record = _record(usd)
        registered = _write(
            task_root / "registered.json",
            {
                "schema_version": "registered_replacement_asset.v1",
                "scene_id": "840920",
                "task_id": task_id,
                "asset_id": asset_id,
                "task_freeze_digest": freeze_value["task_freeze_digest"],
                "output_usd": usd_record,
                "receipt_digest": "",
            },
            "receipt_digest",
        )
        registered_value = json.loads(registered.read_text())

        contact_link = "door" if letter == "a" else "base"
        affordance = _write(
            task_root / "affordance.json",
            {
                "schema_version": "paired_target_interaction_affordance_candidate.v1",
                "status": "candidate_geometry_materialized_requires_native_contact",
                "scene_id": "840920",
                "task_id": task_id,
                "asset_id": asset_id,
                "registered_asset": {
                    "receipt_digest": registered_value["receipt_digest"]
                },
                "task_freeze": {
                    "task_freeze_digest": freeze_value["task_freeze_digest"]
                },
                "robot_base_position_world_m": [0.0, -1.0, 0.0],
                "native_contact_execution_authorized": False,
                "native_contact_executed": False,
                "selection_contract": {
                    "method": "graph_geometry_only",
                    "object_label_or_task_id_geometry_shortcut_used": False,
                    "candidate_geometry_authored_or_modified": False,
                },
                "candidate": {
                    "link_id": contact_link,
                    "link_prim_path": f"/Asset/links/{contact_link}",
                    "contact_body_prim_paths": [f"/Asset/links/{contact_link}"],
                    "contact_point_link_m": [0.0, 0.0, 0.0],
                    "contact_point_registered_stage_m": [0.0, 0.0, 0.0],
                    "approach_unit_registered_stage": [0.0, 1.0, 0.0],
                    "pinch_span_m": 0.04,
                    "pinch_span_within_stroke": True,
                },
                "receipt_digest": "",
            },
            "receipt_digest",
        )
        affordance_value = json.loads(affordance.read_text())

        cameras = [_camera(role) for role in ("external", "wrist", "overview")]
        reset = {
            name: float(index) / 100.0
            for index, name in enumerate(DROID_FRANKA_RESET_JOINT_NAMES)
        }
        camera = _write(
            task_root / "camera.json",
            {
                "schema_version": "paired_target_native_camera_rig_candidate.v1",
                "status": "native_camera_rig_requested_requires_readback_and_observability",
                "scene_id": "840920",
                "task_id": task_id,
                "interaction_affordance_candidate": {
                    "receipt_digest": affordance_value["receipt_digest"]
                },
                "franka_placement_packet": {
                    "packet_digest": placement_value["packet_digest"]
                },
                "robot_base_pose_world": {
                    "position_world_m": [0.0, -1.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "robot_joint_reset_positions_rad": reset,
                "cameras": cameras,
                "policy_input_roles": ["external", "wrist"],
                "native_camera_readback_qualified": False,
                "native_semantic_observability_qualified": False,
                "overview_review_only": True,
                "receipt_digest": "",
            },
            "receipt_digest",
        )

        probe = _write(
            evidence / "native" / "probes" / f"{letter}.json",
            {
                "schema_version": "simready_replacement_native_import_probe_result.v1",
                "status": "completed",
                "asset_id": asset_id,
                "native_simulator_import_qualified": True,
                "candidate_policy_queried": False,
                "result_digest": "",
            },
            "result_digest",
        )
        probe_value = json.loads(probe.read_text())
        import_rows.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "blockers": [],
                "native_simulator_import_qualified": True,
                "probe_result_path": f"probes/{letter}.json",
                "probe_result_sha256": _sha(probe),
                "probe_result_digest": probe_value["result_digest"],
            }
        )
        paired_tasks.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "scenario_suite": _record(
                    scenario, suite_digest=scenario_value["suite_digest"]
                ),
                "registered_replacement_asset_receipt": _record(
                    registered, receipt_digest=registered_value["receipt_digest"]
                ),
                "registered_replacement_usd": usd_record,
                "camera_index": {
                    "camera_ids": [f"review_{index}" for index in range(8)]
                },
            }
        )
        manipulation_tasks.append(
            {
                "task_id": task_id,
                "task_freeze_path": str(freeze),
                "franka_placement_packet_path": str(placement),
                "interaction_affordance_candidate_path": str(affordance),
                "native_camera_rig_candidate_path": str(camera),
            }
        )
        arena_input = {
            "task_freeze_path": str(freeze),
            "registered_asset_receipt_path": str(registered),
            "interaction_affordance_path": str(affordance),
            "camera_rig_path": str(camera),
            "scenario_suite_path": str(scenario),
            "appearance_path": str(appearance),
        }
        if letter == "a":
            graph = freeze_value["articulation_graph"]
            path_receipt = _write(
                task_root / "kinematic.json",
                {
                    "schema_version": "paired_target_articulated_kinematic_path.v1",
                    "task_id": task_id,
                    "interaction_affordance": {
                        "receipt_digest": affordance_value["receipt_digest"]
                    },
                    "articulation_graph_digest": canonical_digest(
                        validate_articulation_graph(graph)
                    ),
                    "joint_contact_path": [
                        {
                            "clearance_unit_asset_root": [0.0, 1.0, 0.0],
                            "joint_positions": {
                                row["joint_id"]: float(row["reset_position"])
                                for row in graph["joints"]
                            },
                        },
                        {
                            "clearance_unit_asset_root": [0.0, 1.0, 0.0],
                            "joint_positions": {
                                row["joint_id"]: (
                                    0.8
                                    if row["joint_id"] == "door_hinge"
                                    else float(row["reset_position"])
                                )
                                for row in graph["joints"]
                            },
                        },
                    ],
                    "receipt_digest": "",
                },
                "receipt_digest",
            )
            arena_input["kinematic_path_receipt_path"] = str(path_receipt)
        else:
            arena_input["support_receipt_path"] = str(support)
        arena_inputs.append(arena_input)

    paired = _write(
        evidence / "paired.json",
        {
            "schema_version": "paired_target_native_preflight.v1",
            "scene_id": "840920",
            "candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "replacement_object_count": 2,
            "collision_scene": _record(collision),
            "tasks": paired_tasks,
            "receipt_digest": "",
        },
        "receipt_digest",
    )
    native_import = _write(
        evidence / "native" / "result.json",
        {
            "schema_version": "paired_target_native_import_runtime_result.v1",
            "status": "completed",
            "scene_id": "840920",
            "replacement_count": 2,
            "native_isaac_executed": True,
            "all_replacements_import_qualified": True,
            "candidate_policy_queried": False,
            "replacements": import_rows,
            "result_digest": "",
        },
        "result_digest",
    )
    return {
        "evidence": evidence,
        "paired": paired,
        "native_import": native_import,
        "collider_batch": collider_batch,
        "manipulation_tasks": manipulation_tasks,
        "arena_inputs": arena_inputs,
    }


def test_two_task_chain_orders_pre_arena_construction_arena_then_full_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    pre_arena_path = tmp_path / "pre_arena.json"
    pre_arena = materialize_paired_target_native_manipulation_preflight(
        paired_target_preflight_path=fixture["paired"],
        native_import_result_path=fixture["native_import"],
        task_records=fixture["manipulation_tasks"],
        output_path=pre_arena_path,
        phase="pre_arena",
    )
    assert pre_arena["status"] == "ready_for_native_construction_bindings"
    assert pre_arena["blockers"] == []
    assert len(pre_arena["pending_requirements"]) == 2

    construction_path = tmp_path / "construction.json"
    construction = materialize_paired_target_native_construction_bindings(
        manipulation_preflight_path=pre_arena_path,
        source_collider_batch_removal_path=fixture["collider_batch"],
        output_path=construction_path,
    )
    assert construction["replacement_object_count"] == 2
    assert construction["native_reachability_qualified"] is False

    monkeypatch.setattr(
        "blueprint_pipeline.paired_target_native_arena_request._registered_root_pose",
        lambda _path: (
            {
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            np.eye(4, dtype=np.float64),
        ),
    )
    arena = materialize_paired_target_native_arena_requests(
        construction_bindings_path=construction_path,
        task_inputs=fixture["arena_inputs"],
        evidence_root=fixture["evidence"],
        output_root=tmp_path / "arena",
    )
    assert arena["replacement_object_count"] == 2
    assert arena["native_execution_performed"] is False

    request_by_task = {
        row["task_id"]: row["request"]["path"] for row in arena["tasks"]
    }
    final_tasks = [dict(row) for row in fixture["manipulation_tasks"]]
    for row in final_tasks:
        row["native_task_arena_request_path"] = request_by_task[row["task_id"]]
    final = materialize_paired_target_native_manipulation_preflight(
        paired_target_preflight_path=fixture["paired"],
        native_import_result_path=fixture["native_import"],
        task_records=final_tasks,
        output_path=tmp_path / "full.json",
        phase="arena_packet",
    )
    assert final["status"] == "ready_for_native_arena_packet_materialization"
    assert final["pending_requirements"] == []
    assert final["learned_policies_executed"] is False


def test_pre_arena_phase_rejects_missing_native_import_qualification(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    native_import = json.loads(fixture["native_import"].read_text())
    native_import["replacements"][0]["native_simulator_import_qualified"] = False
    _write(fixture["native_import"], native_import, "result_digest")

    with pytest.raises(
        PairedTargetNativeManipulationPreflightError,
        match="paired_target_manipulation_native_import_mismatch",
    ):
        materialize_paired_target_native_manipulation_preflight(
            paired_target_preflight_path=fixture["paired"],
            native_import_result_path=fixture["native_import"],
            task_records=fixture["manipulation_tasks"],
            output_path=tmp_path / "must_not_exist.json",
            phase="pre_arena",
        )


def test_pre_arena_phase_rejects_missing_registered_asset(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registered = Path(fixture["arena_inputs"][0]["registered_asset_receipt_path"])
    registered.unlink()

    with pytest.raises(
        PairedTargetNativeManipulationPreflightError,
        match="paired_target_manipulation_registered_asset_invalid",
    ):
        materialize_paired_target_native_manipulation_preflight(
            paired_target_preflight_path=fixture["paired"],
            native_import_result_path=fixture["native_import"],
            task_records=fixture["manipulation_tasks"],
            output_path=tmp_path / "must_not_exist.json",
            phase="pre_arena",
        )
