from __future__ import annotations

import gzip
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paired_target_native_preflight import (
    PairedTargetNativePreflightError,
    materialize_paired_target_native_preflight,
)
from blueprint_pipeline.paired_target_native_manipulation_preflight import (
    PairedTargetNativeManipulationPreflightError,
    materialize_paired_target_native_manipulation_preflight,
)
from blueprint_pipeline.replacement_asset_frame_registration import (
    seal_replacement_asset_frame_registration,
)


def _write(path: Path, value: dict, field: str) -> Path:
    value[field] = canonical_digest(value, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _task_freeze(task_id: str, *, placement_digest: str, index: int) -> dict:
    value = {
        "schema_version": "dual_task_task_freeze.v1",
        "scene_freeze_digest": _digest("a"),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "overview_camera_policy_input": False,
        "overview_camera_deterministic_scoring_input": False,
        "task_id": task_id,
        "prompt": "Relocate the observed rigid task object.",
        "task_kind": "rigid_object_manipulation",
        "source_object": {
            "instance_id": f"source_{index}",
            "semantic_label": f"object_{index}",
            "observed_bounds_world_m": {
                "minimum": [float(index), 0.0, 0.0],
                "maximum": [float(index) + 0.2, 0.2, 0.2],
            },
            "observed_pose_world": {
                "position_world_m": [float(index) + 0.1, 0.1, 0.1],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "support_or_attachment_id": f"support_{index}",
            "collision_identity_receipt_digest": _digest("b"),
            "support_receipt_digest": _digest("c"),
            "franka_placement_packet_digest": placement_digest,
            "visibility_receipt_digest": _digest("d"),
        },
        "removal_plan": {
            "removal_id": f"removal_{index}",
            "mask_set_id": f"mask_{index}",
            "source_collider_prim_path": f"/Root/source_{index}",
            "collider_deletion_id": f"collider_{index}",
            "replacement_asset_id": f"replacement_{index}",
            "replacement_qualification_id": f"qualification_{index}",
        },
        "cameras": {
            "external": f"external_{index}",
            "wrist": f"wrist_{index}",
            "overview": f"overview_{index}",
        },
        "execution_contract": {
            "control_frequency_hz": 20,
            "maximum_steps": 400,
            "settle_window_steps": 20,
            "seeds": [3101 + index],
            "canonical_scenario_cell_id": f"canonical_{index}",
            "reset_state": {"robot": "home", "object": "source"},
        },
        "deterministic_success_predicates": ["released", "settled", "retreated"],
        "failure_rungs": ["never_moved", "collision_failure"],
        "target_configuration": {
            "kind": "pose_volume",
            "position_bounds_world_m": {
                "minimum": [float(index) + 0.1, 0.2, 0.0],
                "maximum": [float(index) + 0.2, 0.3, 0.2],
            },
            "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
            "maximum_orientation_error_rad": 0.1,
            "support_id": f"support_{index}",
            "release_required": True,
        },
        "articulation_graph": None,
        "mechanism_provenance": "observed exterior; no hidden mechanism used",
        "task_freeze_digest": "",
    }
    value["task_freeze_digest"] = canonical_digest(
        value, digest_field="task_freeze_digest"
    )
    return value


def _placement(task_id: str, *, index: int) -> dict:
    value = {
        "schema_version": "registered_sage_franka_placement_packet.v1",
        "status": "placement_candidate_materialized",
        "conversion_receipt_digest": _digest("e"),
        "request": {"request_digest": _digest("f")},
        "target_analysis": {
            "selected_target": {
                "target_id": f"target_{index}",
                "target_label": f"object_{index}",
                "task_family": "rigid_object_relocation_with_locked_internal_joints",
                "position_m": [float(index) + 0.1, 0.1, 0.1],
            }
        },
        "placement": {
            "robot_pose_xyzyaw_collision_stage": [
                float(index),
                -0.5,
                0.0,
                0.0,
            ]
        },
        "render_options": {},
        "native_contact_reachability_qualified": False,
        "policy_execution_authorized": False,
        "blockers": [
            "franka_native_reset_contact_reachability_missing",
            "robot_base_and_camera_calibration_native_probe_missing",
        ],
        "packet_digest": "",
    }
    value["packet_digest"] = canonical_digest(value, digest_field="packet_digest")
    return value


def _fixture(root: Path, task_id: str) -> dict[str, str]:
    task = root / task_id
    task.mkdir(parents=True)
    if task_id == "task_a":
        index = 0
    elif task_id == "task_b":
        index = 1
    else:
        index = int(task_id.rsplit("_", 1)[-1])
    placement = _placement(task_id, index=index)
    placement_path = task / "placement.json"
    placement_path.write_text(json.dumps(placement), encoding="utf-8")
    freeze = _task_freeze(
        task_id, placement_digest=placement["packet_digest"], index=index
    )
    freeze_path = task / "task_freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    usdz = task / "scene.usdz"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(b"model")
    with usdz.open("wb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, body in (
                ("default.usda", b"default"),
                ("repaired_scene.nurec", buffer.getvalue()),
                ("gauss.usda", b"gauss"),
            ):
                info = zipfile.ZipInfo(name)
                padding = (-(raw.tell() + 30 + len(name.encode()))) % 64
                if padding:
                    if padding < 4:
                        padding += 64
                    info.extra = struct.pack("<HH", 0x1986, padding - 4) + b"\0" * (padding - 4)
                archive.writestr(info, body)
    from blueprint_pipeline.paired_target_native_preflight import _record

    members = []
    with usdz.open("rb") as raw, zipfile.ZipFile(usdz) as archive:
        for info in archive.infolist():
            raw.seek(info.header_offset)
            fields = struct.unpack("<IHHHHHIIIHH", raw.read(30))
            offset = info.header_offset + 30 + fields[-2] + fields[-1]
            members.append(
                {
                    "filename": info.filename,
                    "size_bytes": info.file_size,
                    "data_offset_bytes": offset,
                    "sha256": _record_bytes(archive.read(info)),
                }
            )
    appearance = {
        "schema_version": "public_scene_artifixer3d_native_appearance_export.v1",
        "native_import_qualified": False,
        "generated_output_is_capture_or_physical_evidence": False,
        "isaac_nurec_usdz": _record(usdz),
        "isaac_nurec_usdz_archive_contract": {
            "compression": "stored",
            "payload_alignment_bytes": 64,
            "all_payload_offsets_aligned": True,
            "nurec_gzip_mtime_normalized_to_zero": True,
            "members": members,
        },
    }
    appearance_path = _write(task / "appearance.json", appearance, "export_digest")
    trajectory = task / "review_transforms.json"
    trajectory.write_text("{}", encoding="utf-8")
    camera_index = task / "camera_index.json"
    camera_value = {
        "camera_index_digest": "sha256:" + "1" * 64,
        "frames": [{"camera_id": f"camera_{i}"} for i in range(8)],
    }
    camera_index.write_text(json.dumps(camera_value), encoding="utf-8")
    dual = {
        "schema_version": "public_scene_artifixer3d_dual_target_inputs.v1",
        "publisher_scene_id": "840920",
        "selected_task_ids": [task_id],
        "tasks": [
            {
                "task_id": task_id,
                "scene_directory": str(task),
                "camera_count": 8,
                "physical_camera_count": 8,
                "review_trajectory": {
                    "relative_path": trajectory.name,
                    "size_bytes": trajectory.stat().st_size,
                    "sha256": _record(trajectory)["sha256"],
                },
                "camera_index": {
                    "relative_path": camera_index.name,
                    "size_bytes": camera_index.stat().st_size,
                    "sha256": _record(camera_index)["sha256"],
                },
            }
        ],
    }
    dual_path = _write(task / "dual.json", dual, "receipt_digest")
    usd = task / "replacement.usda"
    usd.write_text("#usda 1.0", encoding="utf-8")
    cad = {
        "schema_version": "simready_graph_asset_receipt.v1",
        "status": "simready_candidate_authored",
        "task_id": task_id,
        "asset_id": f"asset_{task_id}",
        "claim_boundary": {"native_simulator_import_qualified": False},
        "output_usd": {**_record(usd)},
    }
    cad_path = _write(task / "cad.json", cad, "receipt_digest")
    static = {
        "schema_version": "simready_graph_asset_static_qualification.v1",
        "task_id": task_id,
        "asset_id": cad["asset_id"],
        "authored_structure_statically_qualified": True,
        "authoring_receipt": {"receipt_digest": cad["receipt_digest"]},
    }
    visual_usd = task / "replacement_visual.usda"
    visual_usd.write_text("#usda 1.0", encoding="utf-8")
    registered = {
        "schema_version": "registered_replacement_asset.v1",
        "status": "registered_replacement_materialized_pending_native_import",
        "scene_id": "840920",
        "task_id": task_id,
        "asset_id": cad["asset_id"],
        "task_freeze_digest": freeze["task_freeze_digest"],
        "output_usd": _record(visual_usd),
        "agent_authored_display_colors_preserved": True,
        "neutral_fallback_present": False,
        "deterministic_pose_composition_only": True,
        "geometry_generated_or_modified": False,
        "T_observed_world_axes_from_asset_local_axes": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    registered_path = _write(task / "registered.json", registered, "receipt_digest")
    static.update(
        {
            "replacement_usd": registered["output_usd"],
            "registered_replacement_asset": {
                "receipt_digest": registered["receipt_digest"]
            },
            "registered_visual_readback": {
                "asset_frame_registration_digest": "pending"
            },
        }
    )
    references = [task / "front.png", task / "oblique.png"]
    for index, reference in enumerate(references):
        reference.write_bytes(b"reference" + bytes([index]))
    registration_path = task / "registration.json"
    seal_replacement_asset_frame_registration(
        scene_id="840920",
        task_id=task_id,
        asset_id=cad["asset_id"],
        asset_local_forward_axis=[0, -1, 0],
        asset_local_up_axis=[0, 0, 1],
        observed_world_forward_axis=[0, -1, 0],
        observed_world_up_axis=[0, 0, 1],
        reference_image_paths=references,
        reviewed_by="fixture",
        output_path=registration_path,
    )
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registered["frame_registration"] = {
        "registration_digest": registration["registration_digest"]
    }
    registered["receipt_digest"] = canonical_digest(
        registered, digest_field="receipt_digest"
    )
    registered_path.write_text(json.dumps(registered), encoding="utf-8")
    static["registered_replacement_asset"] = {
        "receipt_digest": registered["receipt_digest"]
    }
    static["registered_visual_readback"]["asset_frame_registration_digest"] = registration[
        "registration_digest"
    ]
    static_path = _write(task / "static.json", static, "receipt_digest")
    scenario = {
        "schema_version": "third_scene_task_scenario_suite.v1",
        "scene_id": "840920",
        "task_id": task_id,
        "task_freeze_digest": freeze["task_freeze_digest"],
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "required_controls": ["zero_action_negative", "scripted_positive"],
        "initial_execution_order": [f"{task_id}_canonical", f"{task_id}_camera"],
    }
    scenario_path = _write(task / "scenario.json", scenario, "suite_digest")
    return {
        "task_id": task_id,
        "appearance_export_receipt_path": str(appearance_path),
        "dual_target_inputs_receipt_path": str(dual_path),
        "simready_asset_receipt_path": str(cad_path),
        "simready_static_qualification_path": str(static_path),
        "registered_replacement_asset_receipt_path": str(registered_path),
        "asset_frame_registration_path": str(registration_path),
        "scenario_suite_path": str(scenario_path),
        "task_freeze_path": str(freeze_path),
        "franka_placement_packet_path": str(placement_path),
    }


def _record_bytes(value: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value).hexdigest()


def test_preflight_binds_two_tasks_and_preserves_proof_boundary(tmp_path: Path) -> None:
    tasks = [_fixture(tmp_path, "task_a"), _fixture(tmp_path, "task_b")]
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0", encoding="utf-8")
    result = materialize_paired_target_native_preflight(
        scene_id="840920",
        task_records=tasks,
        collision_scene_path=collision,
        output_path=tmp_path / "result.json",
    )
    assert result["replacement_object_count"] == 2
    assert result["maximum_replacement_objects"] == 5
    assert result["native_isaac_import_executed"] is False
    assert result["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert all(len(task["camera_index"]["camera_ids"]) == 8 for task in result["tasks"])
    assert canonical_digest(result, digest_field="receipt_digest") == result["receipt_digest"]


def test_preflight_rejects_tampered_bytes_and_six_tasks(tmp_path: Path) -> None:
    task = _fixture(tmp_path, "task_a")
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0", encoding="utf-8")
    Path(task["simready_asset_receipt_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(PairedTargetNativePreflightError):
        materialize_paired_target_native_preflight(
            scene_id="840920",
            task_records=[task],
            collision_scene_path=collision,
            output_path=tmp_path / "result.json",
        )
    with pytest.raises(PairedTargetNativePreflightError, match="task_count"):
        materialize_paired_target_native_preflight(
            scene_id="840920",
            task_records=[task] * 6,
            collision_scene_path=collision,
            output_path=tmp_path / "other.json",
        )


def _native_import_result(path: Path, preflight: dict) -> Path:
    value = {
        "schema_version": "paired_target_native_import_runtime_result.v1",
        "status": "completed",
        "scene_id": preflight["scene_id"],
        "replacement_count": len(preflight["tasks"]),
        "replacements": [
            {
                "task_id": row["task_id"],
                "asset_id": row["asset_id"],
                "native_simulator_import_qualified": True,
                "blockers": [],
            }
            for row in preflight["tasks"]
        ],
        "native_isaac_executed": True,
        "all_replacements_import_qualified": True,
        "candidate_policy_queried": False,
        "physical_equivalence_claimed": False,
        "blockers": [],
        "result_digest": "",
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _interaction_candidate(path: Path, record: dict) -> Path:
    freeze = json.loads(Path(record["task_freeze_path"]).read_text())
    registered_path = Path(record["registered_replacement_asset_receipt_path"])
    registered = json.loads(registered_path.read_text())
    placement = json.loads(Path(record["franka_placement_packet_path"]).read_text())
    value = {
        "schema_version": "paired_target_interaction_affordance_candidate.v1",
        "status": "candidate_geometry_materialized_requires_native_contact",
        "scene_id": registered["scene_id"],
        "task_id": record["task_id"],
        "asset_id": registered["asset_id"],
        "task_freeze": {
            "path": record["task_freeze_path"],
            "task_freeze_digest": freeze["task_freeze_digest"],
        },
        "registered_asset": {
            "path": str(registered_path),
            "receipt_digest": registered["receipt_digest"],
        },
        "robot_base_position_world_m": placement["placement"][
            "robot_pose_xyzyaw_collision_stage"
        ][:3],
        "selection_contract": {
            "method": "rigid_root_thinnest_axis_pinch",
            "object_label_or_task_id_geometry_shortcut_used": False,
            "candidate_geometry_authored_or_modified": False,
        },
        "candidate": {
            "link_id": "base",
            "pinch_span_m": 0.02,
            "pinch_span_within_stroke": True,
        },
        "native_contact_execution_authorized": False,
        "native_contact_executed": False,
        "receipt_digest": "",
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_manipulation_preflight_binds_available_proof_and_types_missing_arena(
    tmp_path: Path,
) -> None:
    records = [_fixture(tmp_path, "task_a"), _fixture(tmp_path, "task_b")]
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0", encoding="utf-8")
    preflight_path = tmp_path / "preflight.json"
    preflight = materialize_paired_target_native_preflight(
        scene_id="840920",
        task_records=records,
        collision_scene_path=collision,
        output_path=preflight_path,
    )
    native_import = _native_import_result(tmp_path / "native_import.json", preflight)

    result = materialize_paired_target_native_manipulation_preflight(
        paired_target_preflight_path=preflight_path,
        native_import_result_path=native_import,
        task_records=records,
        output_path=tmp_path / "manipulation.json",
    )

    assert result["replacement_object_count"] == 2
    assert result["maximum_replacement_objects"] == 5
    assert result["native_import_qualified"] is True
    assert result["calibrated_review_camera_requests_bound"] is True
    assert result["status"] == "blocked_pending_native_manipulation_inputs"
    assert result["blockers"] == [
        "task_a:interaction_affordance_candidate_missing",
        "task_a:native_task_arena_packet_request_missing",
        "task_b:interaction_affordance_candidate_missing",
        "task_b:native_task_arena_packet_request_missing",
    ]
    assert all(row["review_camera_count"] == 8 for row in result["tasks"])
    assert all(
        row["native_reachability_execution_authorized"] is False
        for row in result["tasks"]
    )


def test_manipulation_preflight_scales_to_five_distinct_objects(tmp_path: Path) -> None:
    records = [_fixture(tmp_path, f"object_{index}") for index in range(5)]
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0", encoding="utf-8")
    preflight_path = tmp_path / "preflight.json"
    preflight = materialize_paired_target_native_preflight(
        scene_id="840920",
        task_records=records,
        collision_scene_path=collision,
        output_path=preflight_path,
    )
    native_import = _native_import_result(tmp_path / "native_import.json", preflight)

    result = materialize_paired_target_native_manipulation_preflight(
        paired_target_preflight_path=preflight_path,
        native_import_result_path=native_import,
        task_records=records,
        output_path=tmp_path / "manipulation.json",
    )

    assert result["replacement_object_count"] == 5
    assert len(result["blockers"]) == 10
    assert all(row["native_arena_packet_materialization_ready"] is False for row in result["tasks"])


def test_manipulation_preflight_binds_interaction_candidates_before_arena(
    tmp_path: Path,
) -> None:
    records = [_fixture(tmp_path, "task_a"), _fixture(tmp_path, "task_b")]
    for record in records:
        record["interaction_affordance_candidate_path"] = str(
            _interaction_candidate(
                tmp_path / f"{record['task_id']}_affordance.json", record
            )
        )
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0", encoding="utf-8")
    preflight_path = tmp_path / "preflight.json"
    preflight = materialize_paired_target_native_preflight(
        scene_id="840920",
        task_records=records,
        collision_scene_path=collision,
        output_path=preflight_path,
    )
    native_import = _native_import_result(tmp_path / "native_import.json", preflight)

    result = materialize_paired_target_native_manipulation_preflight(
        paired_target_preflight_path=preflight_path,
        native_import_result_path=native_import,
        task_records=records,
        output_path=tmp_path / "manipulation.json",
    )

    assert result["blockers"] == [
        "task_a:native_task_arena_packet_request_missing",
        "task_b:native_task_arena_packet_request_missing",
    ]
    assert all(
        row["interaction_affordance_candidate"]["pinch_span_m"] == 0.02
        for row in result["tasks"]
    )

    candidate_path = Path(records[0]["interaction_affordance_candidate_path"])
    candidate = json.loads(candidate_path.read_text())
    candidate["robot_base_position_world_m"][0] += 0.5
    candidate["receipt_digest"] = canonical_digest(
        candidate, digest_field="receipt_digest"
    )
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(
        PairedTargetNativeManipulationPreflightError,
        match="interaction_affordance_invalid:task_a",
    ):
        materialize_paired_target_native_manipulation_preflight(
            paired_target_preflight_path=preflight_path,
            native_import_result_path=native_import,
            task_records=records,
            output_path=tmp_path / "tampered.json",
        )


def test_manipulation_preflight_rejects_import_or_placement_mismatch(
    tmp_path: Path,
) -> None:
    record = _fixture(tmp_path, "task_a")
    collision = tmp_path / "collision.usda"
    collision.write_text("#usda 1.0", encoding="utf-8")
    preflight_path = tmp_path / "preflight.json"
    preflight = materialize_paired_target_native_preflight(
        scene_id="840920",
        task_records=[record],
        collision_scene_path=collision,
        output_path=preflight_path,
    )
    import_path = _native_import_result(tmp_path / "native_import.json", preflight)
    native_import = json.loads(import_path.read_text())
    native_import["replacements"][0]["native_simulator_import_qualified"] = False
    native_import["result_digest"] = canonical_digest(
        native_import, digest_field="result_digest"
    )
    import_path.write_text(json.dumps(native_import), encoding="utf-8")
    with pytest.raises(
        PairedTargetNativeManipulationPreflightError,
        match="native_import_mismatch",
    ):
        materialize_paired_target_native_manipulation_preflight(
            paired_target_preflight_path=preflight_path,
            native_import_result_path=import_path,
            task_records=[record],
            output_path=tmp_path / "bad_import.json",
        )

    _native_import_result(import_path, preflight)
    placement_path = Path(record["franka_placement_packet_path"])
    placement = json.loads(placement_path.read_text())
    placement["target_analysis"]["selected_target"]["position_m"][0] += 1.0
    placement["packet_digest"] = canonical_digest(
        placement, digest_field="packet_digest"
    )
    placement_path.write_text(json.dumps(placement), encoding="utf-8")
    with pytest.raises(
        PairedTargetNativeManipulationPreflightError,
        match="franka_placement_mismatch",
    ):
        materialize_paired_target_native_manipulation_preflight(
            paired_target_preflight_path=preflight_path,
            native_import_result_path=import_path,
            task_records=[record],
            output_path=tmp_path / "bad_placement.json",
        )
