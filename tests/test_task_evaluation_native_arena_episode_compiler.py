from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
    OUTPUT_SCHEMA_VERSION,
    TaskEvaluationNativeArenaEpisodeCompilerError,
    _json_reference,
    _reference_path,
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


def test_policy_canary_keeps_registry_out_of_native_controller_slot() -> None:
    root = Path(__file__).resolve().parents[1]
    native = (
        root
        / "docs/arm_decision_proof_v1/manifests/scene839873_policy_canary_native_controller_configuration.v1.json"
    )
    registry = (
        root
        / "docs/arm_decision_proof_v1/manifests/scene839873_policy_canary_controller_configuration.v1.json"
    )
    native_record = _record(native, "controller.configuration")
    registry_record = _record(registry, "controller.configuration")

    controller = _json_reference(
        {"controller.configuration": native_record},
        "controller.configuration",
        "task_evaluation_native_controller_configuration.v1",
    )
    assert controller["identity"] == {"id": "paired-policy-canary", "version": "v1"}
    assert controller["kind"] == "policy_container"
    assert controller["document_digest"] == canonical_digest(
        controller, digest_field="document_digest"
    )

    with pytest.raises(
        TaskEvaluationNativeArenaEpisodeCompilerError,
        match="episode_compiler_reference_contract_invalid:controller.configuration",
    ):
        _json_reference(
            {"controller.configuration": registry_record},
            "controller.configuration",
            "task_evaluation_native_controller_configuration.v1",
        )


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
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr(
            "configured_scene_bundle_candidate.v1.json",
            json.dumps(manifest, sort_keys=True) + "\n",
        )


def _rigid_task_documents(
    task_identity: dict[str, object], object_identity: dict[str, object]
) -> dict[str, dict[str, object]]:
    success = {
        "authority": "deterministic_simulator_state",
        "forbidden_collision_allowed": False,
        "joint_limit_violation_allowed": False,
        "maximum_final_planar_target_error_m": 0.05,
        "minimum_planar_displacement_m": 0.1,
        "object_must_remain_on_registered_support": True,
    }
    return {
        "scene.configured_revision.task_template.definition": {
            "schema_version": "task_evaluation_rigid_relocation_template.v1",
            "status": "preregistered_candidate_pending_configured_scene_revision",
            "task_identity": task_identity,
            "object_identity": object_identity,
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
            "success": success,
        },
        "scene.configured_revision.task_template.success_criteria": {
            "schema_version": "task_evaluation_rigid_relocation_success_criteria.v1",
            "status": "preregistered_before_any_episode",
            **success,
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
            "collision_exclusions": ["robot_self_collision_pairs_declared_by_robot_configuration"],
            "termination": ["success", "timeout"],
        },
    }


def _configured_runtime_documents(configured: dict[str, object]) -> dict[str, dict]:
    scene_id = str(configured["scene_identity"]["id"]).rsplit("-", 1)[-1]
    source = {
        "schema_version": "task_evaluation_source_object_selection.v1",
        "status": "frozen_before_scene_configuration_run",
        "scene_id": scene_id,
        "center_xyz_m": [2.9742285, -6.7605156, 0.818319],
        "aabb_min_xyz_m": [2.9103536, -6.8264092, 0.7545],
        "aabb_max_xyz_m": [3.0381034, -6.6946220, 0.882138],
    }
    static = {
        "schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
        "status": "authored_structure_statically_qualified",
        "replacement_identity": configured["replacement"]["identity"],
        "observed_structure": {
            "center_of_mass_m": [0.0, 0.0, 0.063819],
            "rigid_body_paths": ["/Asset"],
        },
        "result_digest": "",
    }
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    native = {
        "schema_version": "task_evaluation_replacement_native_import_result.v1",
        "status": "qualified",
        "replacement_identity": configured["replacement"]["identity"],
        "native_simulator_import_qualified": True,
        "blockers": [],
        "result_digest": "",
    }
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    return {
        "scene.configured_revision.registration.support_plane": {
            "schema_version": "task_evaluation_support_plane_input.v1",
            "status": "frozen_candidate_pending_production_validation",
            "scene_id": scene_id,
            "sage_prim_path": "/Root/Support",
            "bounds_min_xyz_m": [2.5, -9.5, 0.0],
            "bounds_max_xyz_m": [4.5, -1.4, 0.7545],
            "top_z_m": 0.7545,
        },
        "scene.configured_revision.replacement.source_object": source,
        "scene.configured_revision.replacement.static_qualification": static,
        "scene.configured_revision.replacement.native_import_qualification": native,
    }


def test_closed_compiler_joins_revision_and_robot_team_inputs(tmp_path: Path, monkeypatch) -> None:
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
            "scene_camera_calibration_digest": configured["registration"]["camera_calibration"][
                "digest"
            ],
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
            "collision_exclusions": ["robot_self_collision_pairs_declared_by_robot_configuration"],
            "termination": ["success", "timeout"],
        },
        **_configured_runtime_documents(configured),
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
    for contract_path, section, field in (
        (
            "scene.configured_revision.registration.support_plane",
            "registration",
            "support_plane",
        ),
        (
            "scene.configured_revision.replacement.source_object",
            "replacement",
            "source_object",
        ),
        (
            "scene.configured_revision.replacement.static_qualification",
            "replacement",
            "static_qualification",
        ),
        (
            "scene.configured_revision.replacement.native_import_qualification",
            "replacement",
            "native_import_qualification",
        ),
    ):
        record = _record(document_paths[contract_path], contract_path)
        configured[section][field] = {key: record[key] for key in ("uri", "digest", "size_bytes")}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    value["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
    revision_path = _write_json(inputs, "revision.json", configured)
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
    for contract_path, path in document_paths.items():
        references[contract_path] = _record(path, contract_path)

    observed = {}

    def fake_materialize(*, request, evidence_root, output_dir, link_sources_within):
        observed["packet_request"] = request
        assert Path(evidence_root) == (tmp_path / "output").resolve()
        # The packet may hard-link only bytes inside the compiler's own
        # per-run root, which it extracted and retires together with the packet.
        assert Path(link_sources_within) == (tmp_path / "output").resolve()
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

    def fake_native_appearance(*, source_path, output_root):
        output_root.mkdir()
        output = output_root / "scene_appearance.usdc"
        output.write_bytes(Path(source_path).read_bytes())
        digest, size = (
            "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
            output.stat().st_size,
        )
        return {
            "status": "nurec_converted_to_particlefield",
            "path": str(output),
            "representation": "particlefield_3d_gaussian_splat",
            "source_configured_appearance_digest": digest,
            "source_configured_appearance_size_bytes": size,
            "particlefield_digest": digest,
            "particlefield_size_bytes": size,
            "representation_conversion_performed": True,
            "exact_learned_arrays_preserved": True,
            "gaussian_field_quality": {
                "schema_version": "gaussian_field_quality.v1",
                "status": "qualified",
                "blockers": [],
                "learned_tensors_mutated": False,
            },
        }

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
        native_appearance_materializer=fake_native_appearance,
    )

    assert result["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert result["compiled_by_production"] is True
    assert result["customer_supplied_prebuilt_episode_packet"] is False
    assert result["provider_mutation_performed"] is False
    assert result["adapter_result"]["packet_receipt_digest"] == ("sha256:" + "b" * 64)
    assert result["native_scene_appearance"]["representation"] == (
        "particlefield_3d_gaussian_splat"
    )
    assert result["native_scene_appearance"]["representation_conversion_performed"] is True
    appearance = next(
        row
        for row in observed["packet_request"]["assets"]
        if row["semantic_role"] == "scene_appearance"
    )
    assert appearance["filename"] == "scene_appearance.usdc"
    assert observed["packet_request"]["appearance_variant"] == {
        "representation": "particlefield_3d_gaussian_splat",
        "source_configured_appearance_digest": result["native_scene_appearance"][
            "source_configured_appearance_digest"
        ],
        "representation_conversion_performed": True,
        "exact_learned_arrays_preserved": True,
        "gaussian_field_quality": {
            "schema_version": "gaussian_field_quality.v1",
            "status": "qualified",
            "blockers": [],
            "learned_tensors_mutated": False,
        },
    }
    source_subject_id = configured["replacement"]["identity"]["id"]
    runtime_subject_id = source_subject_id.replace("-", "_")
    assert observed["packet_request"]["task_spec"]["subject_asset_id"] == (runtime_subject_id)
    assert observed["packet_request"]["task_spec"]["source_subject_identity"] == (source_subject_id)
    assert (
        observed["packet_request"]["task_spec"]["interaction_affordance"]["subject_asset_id"]
        == runtime_subject_id
    )
    assert observed["packet_request"]["assets"][2]["asset_id"] == (runtime_subject_id)
    assert observed["packet_request"]["assets"][2]["source_asset_id"] == (source_subject_id)
    assert observed["packet_request"]["task_spec"]["manipulation_strategy"] == ("planar_push")
    assert observed["packet_request"]["task_spec"]["task_kind"] == ("rigid_pick_place")
    assert observed["packet_request"]["task_spec"]["schema_version"] == ("adp_task_spec.v2")
    assert observed["packet_request"]["task_spec"]["start_pose_world"][:3] == [
        2.9742285,
        -6.7605156,
        0.818319,
    ]
    assert (
        observed["packet_request"]["task_spec"]["configured_success_criteria"][
            "maximum_final_planar_target_error_m"
        ]
        == 0.05
    )
    assert observed["packet_request"]["physics_frequency_hz"] == 120
    assert observed["packet_request"]["assets"][2]["pose_world"][
        "position_world_m"
    ] == pytest.approx([2.9742285, -6.7605156, 0.7545])
    assert observed["packet_request"]["scenario"]["seed"] == 839873104
    assert observed["packet_request"]["scenario"]["cell_id"] == (
        "configured_scene_canonical.seed_839873104"
    )
    assert (
        observed["packet_request"]["configured_task_template_adapter"]["manipulation_strategy"]
        == "planar_push"
    )
    assert result["configured_task_template_adapter"]["source_documents_digest"].startswith(
        "sha256:"
    )
    replacement = next(
        row for row in observed["packet_request"]["assets"] if row["semantic_role"] == "task_object"
    )
    assert replacement["asset_id"] == configured["replacement"]["identity"]["id"].replace("-", "_")
    assert replacement["source_asset_id"] == configured["replacement"]["identity"]["id"]


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
        **_rigid_task_documents(
            task_identity,
            configured["replacement"]["identity"],
        ),
        **_configured_runtime_documents(configured),
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
    for contract_path, section, field in (
        (
            "scene.configured_revision.registration.support_plane",
            "registration",
            "support_plane",
        ),
        (
            "scene.configured_revision.replacement.source_object",
            "replacement",
            "source_object",
        ),
        (
            "scene.configured_revision.replacement.static_qualification",
            "replacement",
            "static_qualification",
        ),
        (
            "scene.configured_revision.replacement.native_import_qualification",
            "replacement",
            "native_import_qualification",
        ),
    ):
        record = _record(document_paths[contract_path], contract_path)
        configured[section][field] = {key: record[key] for key in ("uri", "digest", "size_bytes")}
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    value["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
    revision_path = _write_json(inputs, "revision.json", configured)
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
    for contract_path, path in document_paths.items():
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


def test_reference_path_refuses_symlink_even_when_target_bytes_match(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"schema_version":"test.v1"}\n', encoding="utf-8")
    link = tmp_path / "reference.json"
    link.symlink_to(target)
    record = _record(target, "robot.configuration")
    record["materialized_path"] = str(link)

    with pytest.raises(
        TaskEvaluationNativeArenaEpisodeCompilerError,
        match="episode_compiler_reference_invalid:robot.configuration",
    ):
        _reference_path(
            {"robot.configuration": record},
            "robot.configuration",
        )
