from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
    OUTPUT_SCHEMA_VERSION,
    compile_native_arena_episode,
)
from tests.test_task_evaluation_configured_scene_revision import revision
from tests.test_task_evaluation_launch_preparation_contract import request


def _record(path: Path, contract_path: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "contract_path": contract_path,
        "uri": f"s3://blueprint-production-inputs/{path.name}",
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "materialized_path": str(path),
        "full_byte_service_account_readback_passed": True,
    }


def _write_json(root: Path, name: str, value: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _configured_bundle(path: Path) -> None:
    payloads = {
        "appearance.usda": b"#usda 1.0\n# appearance\n",
        "collision.usda": b"#usda 1.0\n# collision\n",
        "replacement.usda": b"#usda 1.0\n# replacement\n",
    }
    rows = []
    for role, name in (
        ("appearance", "appearance.usda"),
        ("collision", "collision.usda"),
        ("replacement", "replacement.usda"),
    ):
        payload = payloads[name]
        rows.append(
            {
                "role": role,
                "relative_path": name,
                "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest = {
        "schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
        "status": "assembled_pending_control_plane_publication",
        "robot_neutral": True,
        "robot_specific_base_registration_included": False,
        "assets": rows,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr(
            "configured_scene_bundle_candidate.v1.json",
            json.dumps(manifest, sort_keys=True) + "\n",
        )


def test_closed_compiler_joins_revision_and_robot_team_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    value = request()
    configured = revision()
    value["team_namespace"] = configured["team_namespace"]
    value["expected_production_commit"] = configured["source_commit"]
    value["scene"]["identity"] = configured["scene_identity"]
    value["task"]["identity"] = configured["task_template"]["identity"]
    value["task"]["subject"]["identity"] = configured["replacement"]["identity"]
    bundle_path = inputs / "configured-scene.zip"
    _configured_bundle(bundle_path)
    robot_identity = value["robot"]["identity"]
    controller_identity = value["controller"]["identity"]
    task_identity = value["task"]["identity"]
    docs = {
        "robot.configuration": {
            "schema_version": "task_evaluation_native_robot_configuration.v1",
            "identity": robot_identity,
            "joint_reset_positions_rad": {"panda_joint1": 0.0},
        },
        "robot.kinematics": {
            "schema_version": "task_evaluation_native_robot_kinematics.v1",
            "identity": robot_identity,
        },
        "robot.joint_bounds": {
            "schema_version": "task_evaluation_native_robot_joint_bounds.v1",
            "identity": robot_identity,
        },
        "robot.base_registration": {
            "schema_version": "task_evaluation_robot_to_scene_registration.v1",
            "robot_identity": robot_identity,
            "scene_identity": value["scene"]["identity"],
            "robot_mount_interface_digest": configured["registration"][
                "robot_mount_interface"
            ]["digest"],
            "pose_world": {
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "controller.configuration": {
            "schema_version": "task_evaluation_native_controller_configuration.v1",
            "identity": controller_identity,
            "kind": value["controller"]["kind"],
        },
        "sensors.configuration": {
            "schema_version": "task_evaluation_native_sensor_configuration.v1",
            "scene_camera_calibration_digest": configured["registration"][
                "camera_calibration"
            ]["digest"],
            "cameras": [{"role": "external"}],
        },
        "scene.configured_revision.task_template.definition": {
            "schema_version": "task_evaluation_rigid_relocation_template.v1",
            "status": "preregistered_candidate_pending_configured_scene_revision",
            "task_identity": task_identity,
            "object_identity": configured["replacement"]["identity"],
            "strategy": "planar_push",
            "start_center_xyz_m": [2.9742285, -6.7605156, 0.818319],
            "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319],
            "control_frequency_hz": 20,
            "maximum_episode_seconds": 12.0,
            "maximum_step_count": 240,
            "resolved_seed": 839873104,
            "controls_order": ["zero_action", "deterministic_scripted"],
            "failure_metrics": ["insufficient_displacement", "timeout"],
            "preregistration_rule": "Any task change creates a new version.",
            "success": {
                "authority": "deterministic_simulator_state",
                "forbidden_collision_allowed": False,
                "joint_limit_violation_allowed": False,
                "maximum_final_planar_target_error_m": 0.05,
                "minimum_planar_displacement_m": 0.1,
                "object_must_remain_on_registered_support": True,
            },
        },
        "scene.configured_revision.task_template.success_criteria": {
            "schema_version": "task_evaluation_rigid_relocation_success_criteria.v1",
            "status": "preregistered_before_any_episode",
            "authority": "deterministic_simulator_state",
            "forbidden_collision_allowed": False,
            "joint_limit_violation_allowed": False,
            "maximum_final_planar_target_error_m": 0.05,
            "minimum_planar_displacement_m": 0.1,
            "object_must_remain_on_registered_support": True,
            "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319],
        },
        "scene.configured_revision.task_template.execution": {
            "schema_version": "task_evaluation_rigid_relocation_execution_spec.v1",
            "status": "preregistered_before_any_episode",
            "strategy": "planar_push",
            "start_center_xyz_m": [2.9742285, -6.7605156, 0.818319],
            "target_center_xyz_m": [3.0942285, -6.7605156, 0.818319],
            "control_frequency_hz": 20,
            "maximum_episode_seconds": 12.0,
            "maximum_step_count": 240,
            "resolved_seed": 839873104,
            "action_bounds_m_per_step": {"minimum": -0.02, "maximum": 0.02},
            "collision_exclusions": [
                "robot_self_collision_pairs_declared_by_robot_configuration"
            ],
            "termination": ["success", "timeout"],
        },
    }
    document_paths = {
        contract_path: _write_json(inputs, f"input-{index}.json", document)
        for index, (contract_path, document) in enumerate(docs.items())
    }
    for contract_path, revision_field in (
        ("scene.configured_revision.task_template.definition", "definition"),
        (
            "scene.configured_revision.task_template.success_criteria",
            "success_criteria",
        ),
        ("scene.configured_revision.task_template.execution", "execution"),
    ):
        record = _record(document_paths[contract_path], contract_path)
        configured["task_template"][revision_field] = {
            key: record[key] for key in ("uri", "digest", "size_bytes")
        }
    configured["revision_digest"] = canonical_digest(
        configured, digest_field="revision_digest"
    )
    value["task"]["configured_scene_revision_digest"] = configured[
        "revision_digest"
    ]
    revision_path = _write_json(inputs, "revision.json", configured)
    references = {
        "scene.configured_revision": _record(
            revision_path, "scene.configured_revision"
        ),
        "scene.configured_revision.configured_scene_bundle": _record(
            bundle_path, "scene.configured_revision.configured_scene_bundle"
        ),
    }
    runtime_bundle = inputs / "runtime-source.zip"
    runtime_bundle.write_bytes(b"runtime-source")
    references["execution_adapter.runtime_source_bundle"] = _record(
        runtime_bundle, "execution_adapter.runtime_source_bundle"
    )
    for contract_path, path in document_paths.items():
        references[contract_path] = _record(path, contract_path)

    observed = {}

    def fake_materialize(*, request, evidence_root, output_dir):
        observed["packet_request"] = request
        assert Path(evidence_root) == (tmp_path / "output").resolve()
        Path(output_dir).mkdir()
        (Path(output_dir) / "packet.json").write_text("{}\n", encoding="utf-8")

    def fake_build(*, source_root, output_path, request, role):
        assert Path(source_root).is_dir()
        assert role == "construction_packet"
        Path(output_path).write_bytes(b"production-owned-adapter-bundle")

    def fake_adapter(**kwargs):
        assert kwargs["request"] == value
        assert Path(kwargs["compiled_episode_packet_path"]).is_file()
        output = Path(kwargs["output_root"])
        output.mkdir()
        result = {
            "result_digest": "sha256:" + "a" * 64,
            "packet_receipt_digest": "sha256:" + "b" * 64,
            "runtime_source_receipt_digest": "sha256:" + "c" * 64,
        }
        (output / "task_evaluation_native_arena_adapter_result.v1.json").write_text(
            json.dumps(result) + "\n", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_native_arena_episode_compiler."
        "materialize_native_task_arena_packet",
        fake_materialize,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_native_arena_episode_compiler."
        "build_task_evaluation_adapter_bundle",
        fake_build,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_native_arena_episode_compiler."
        "materialize_native_arena_adapter",
        fake_adapter,
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    envelope = {
        "request": value,
        "run_id": value["run_id"],
        "configured_scene_revision_digest": configured["revision_digest"],
    }
    result = compile_native_arena_episode(
        envelope=envelope,
        materialized_references=references,
        output_root=output_root,
    )

    assert result["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert result["compiled_by_production"] is True
    assert result["customer_supplied_prebuilt_episode_packet"] is False
    assert result["provider_mutation_performed"] is False
    assert result["adapter_result"]["packet_receipt_digest"] == (
        "sha256:" + "b" * 64
    )
    assert observed["packet_request"]["task_spec"]["subject_asset_id"] == (
        configured["replacement"]["identity"]["id"]
    )
    assert observed["packet_request"]["task_spec"]["manipulation_strategy"] == (
        "planar_push"
    )
    assert observed["packet_request"]["task_spec"]["task_kind"] == (
        "rigid_pick_place"
    )
    assert observed["packet_request"]["task_spec"]["start_pose_world"][:3] == [
        2.9742285,
        -6.7605156,
        0.818319,
    ]
    assert observed["packet_request"]["task_spec"][
        "configured_success_criteria"
    ]["maximum_final_planar_target_error_m"] == 0.05
    assert observed["packet_request"]["physics_frequency_hz"] == 120
    assert observed["packet_request"]["scenario"]["seed"] == 839873104
    assert observed["packet_request"]["scenario"]["cell_id"] == (
        "configured_scene_canonical.seed_839873104"
    )
    assert observed["packet_request"]["configured_task_template_adapter"][
        "manipulation_strategy"
    ] == "planar_push"
    assert result["configured_task_template_adapter"]["source_documents_digest"].startswith(
        "sha256:"
    )
    replacement = next(
        row
        for row in observed["packet_request"]["assets"]
        if row["semantic_role"] == "task_object"
    )
    assert replacement["asset_id"] == configured["replacement"]["identity"]["id"]


def test_closed_compiler_refuses_sensor_calibration_from_another_scene(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    value = request()
    configured = revision()
    value["team_namespace"] = configured["team_namespace"]
    value["expected_production_commit"] = configured["source_commit"]
    value["scene"]["identity"] = configured["scene_identity"]
    value["task"]["identity"] = configured["task_template"]["identity"]
    value["task"]["subject"]["identity"] = configured["replacement"]["identity"]
    value["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
    revision_path = _write_json(inputs, "revision.json", configured)
    bundle_path = inputs / "configured-scene.zip"
    _configured_bundle(bundle_path)
    robot_identity = value["robot"]["identity"]
    controller_identity = value["controller"]["identity"]
    task_identity = value["task"]["identity"]
    docs = {
        "robot.configuration": {
            "schema_version": "task_evaluation_native_robot_configuration.v1",
            "identity": robot_identity,
            "joint_reset_positions_rad": {"panda_joint1": 0.0},
        },
        "robot.kinematics": {
            "schema_version": "task_evaluation_native_robot_kinematics.v1",
            "identity": robot_identity,
        },
        "robot.joint_bounds": {
            "schema_version": "task_evaluation_native_robot_joint_bounds.v1",
            "identity": robot_identity,
        },
        "robot.base_registration": {
            "schema_version": "task_evaluation_robot_to_scene_registration.v1",
            "robot_identity": robot_identity,
            "scene_identity": value["scene"]["identity"],
            "robot_mount_interface_digest": configured["registration"]["robot_mount_interface"][
                "digest"
            ],
            "pose_world": {
                "position_world_m": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "controller.configuration": {
            "schema_version": "task_evaluation_native_controller_configuration.v1",
            "identity": controller_identity,
            "kind": value["controller"]["kind"],
        },
        "sensors.configuration": {
            "schema_version": "task_evaluation_native_sensor_configuration.v1",
            "scene_camera_calibration_digest": "sha256:" + "f" * 64,
            "cameras": [{"role": "external"}],
        },
        "scene.configured_revision.task_template.definition": {
            "schema_version": "task_evaluation_native_task_definition.v1",
            "identity": task_identity,
            "task_spec": {"task_kind": "rigid_pick_place"},
            "task_object_pose_world": {
                "position_world_m": [2.9, -6.7, 0.82],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "scene.configured_revision.task_template.success_criteria": {
            "schema_version": "task_evaluation_native_success_criteria.v1",
            "identity": task_identity,
            "criteria": {"minimum_displacement_m": 0.1},
        },
        "scene.configured_revision.task_template.execution": {
            "schema_version": "task_evaluation_native_episode_execution.v1",
            "identity": task_identity,
            "physics_frequency_hz": 120,
            "scenario": {"cell_id": "canonical.seed_17"},
        },
    }
    references = {
        "scene.configured_revision": _record(revision_path, "scene.configured_revision"),
        "scene.configured_revision.configured_scene_bundle": _record(
            bundle_path, "scene.configured_revision.configured_scene_bundle"
        ),
    }
    runtime_bundle = inputs / "runtime-source.zip"
    runtime_bundle.write_bytes(b"runtime-source")
    references["execution_adapter.runtime_source_bundle"] = _record(
        runtime_bundle, "execution_adapter.runtime_source_bundle"
    )
    for index, (contract_path, document) in enumerate(docs.items()):
        path = _write_json(inputs, f"input-{index}.json", document)
        references[contract_path] = _record(path, contract_path)

    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_native_arena_episode_compiler."
        "materialize_native_task_arena_packet",
        lambda **kwargs: None,
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    with pytest.raises(RuntimeError, match="episode_compiler_request_binding_mismatch"):
        compile_native_arena_episode(
            envelope={
                "request": value,
                "run_id": value["run_id"],
                "configured_scene_revision_digest": configured["revision_digest"],
            },
            materialized_references=references,
            output_root=output_root,
        )
