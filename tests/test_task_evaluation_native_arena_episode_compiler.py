from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import blueprint_pipeline.task_evaluation_native_arena_episode_compiler as compiler

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
    OUTPUT_SCHEMA_VERSION,
    TaskEvaluationNativeArenaEpisodeCompilerError,
    _json_reference,
    _reference_path,
    _materialize_native_particlefield_appearance,
    _stage_destination_asset,
    _subject_bounds_in_scoring_frame,
    compile_native_arena_episode,
)
from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz
from blueprint_pipeline.nurec_volume_codec import build_state_dict
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
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


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _destination_case(
    tmp_path: Path,
    *,
    subject_identity: dict | None = None,
    subject_static_path: Path | None = None,
    subject_static: dict | None = None,
    subject_scoring_transform: dict | None = None,
    configured_scene_revision_digest: str | None = None,
    configured_scene_collision_path: Path | None = None,
    configured_scene_support_plane_path: Path | None = None,
) -> tuple[dict, dict[str, dict], dict]:
    subject_identity = subject_identity or {"id": "book", "version": "v1"}
    identity = {"id": "document-tray", "version": "v1"}
    pose = {
        "position_world_m": [3.2, -6.76, 0.82],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    asset = tmp_path / "document-tray.usda"
    asset.write_bytes(b"#usda 1.0\n")
    if subject_static is None:
        subject_static = {
            "schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
            "status": "authored_structure_statically_qualified",
            "replacement_identity": subject_identity,
            "observed_structure": {
                "center_of_mass_m": [0.0, 0.0, 0.0],
                "collision_bounds_body_frame_m": {
                    "minimum": [-0.04, -0.05, -0.01],
                    "maximum": [0.04, 0.05, 0.01],
                },
                "rigid_body_paths": ["/Book"],
            },
            "result_digest": "",
        }
        subject_static["result_digest"] = canonical_digest(
            subject_static, digest_field="result_digest"
        )
    if subject_static_path is None:
        subject_static_path = _write_json(
            tmp_path, "subject-static.json", subject_static
        )
    subject_scoring_transform = subject_scoring_transform or {
        "position_m": [0.0, 0.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    subject_lower, subject_upper = _subject_bounds_in_scoring_frame(
        bounds=subject_static["observed_structure"][
            "collision_bounds_body_frame_m"
        ],
        transform=subject_scoring_transform,
    )
    rights_value = {
        "schema_version": "task_evaluation_rigid_destination_rights_admission.v1",
        "status": "admitted",
        "destination_identity": identity,
        "private_provider_processing_allowed": True,
        "provider_training_allowed": False,
        "public_redistribution_allowed": False,
        "license_identifier": "Blueprint-generated-development-asset",
        "rights_admission_digest": "",
    }
    rights_value["rights_admission_digest"] = canonical_digest(
        rights_value, digest_field="rights_admission_digest"
    )
    rights = _write_json(tmp_path, "rights.json", rights_value)
    static = {
        "schema_version": "task_evaluation_rigid_replacement_static_qualification.v1",
        "status": "authored_structure_statically_qualified",
        "replacement_identity": identity,
        "replacement_usd": {
            "path": str(asset),
            "sha256": _sha(asset),
            "size_bytes": asset.stat().st_size,
        },
        "observed_structure": {
            "center_of_mass_m": [0.0, 0.0, 0.02],
            "collision_bounds_body_frame_m": {
                "minimum": [-0.12, -0.12, 0.0],
                "maximum": [0.12, 0.12, 0.06],
            },
            "rigid_body_paths": ["/Tray"],
        },
        "result_digest": "",
    }
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    static_path = _write_json(tmp_path, "destination-static.json", static)
    native = {
        "schema_version": "task_evaluation_replacement_native_import_result.v1",
        "status": "qualified",
        "replacement_identity": identity,
        "asset_digest": _sha(asset),
        "static_qualification_digest": _sha(static_path),
        "native_isaac_executed": True,
        "native_simulator_import_qualified": True,
        "support_contact_observed": True,
        "deterministic_reset_state_digest_repeat_count": 3,
        "blockers": [],
        "result_digest": "",
    }
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    native_path = _write_json(tmp_path, "destination-native.json", native)
    interior_lower = [-0.11, -0.11, -0.005]
    interior_upper = [0.11, 0.11, 0.25]
    center_lower = [
        interior_lower[axis] - subject_lower[axis] for axis in range(3)
    ]
    center_upper = [
        interior_upper[axis] - subject_upper[axis] for axis in range(3)
    ]
    geometry = {
        "schema_version": "task_evaluation_rigid_destination_geometry.v1",
        "status": "qualified",
        "subject_identity": subject_identity,
        "destination_identity": identity,
        "relation": "inside",
        "pose_world": pose,
        "subject_static_qualification_digest": _sha(subject_static_path),
        "destination_static_qualification_digest": _sha(static_path),
        "subject_collision_bounds_scoring_frame_m": {
            "minimum": subject_lower,
            "maximum": subject_upper,
        },
        "destination_interior_bounds_body_frame_m": {
            "minimum": interior_lower,
            "maximum": interior_upper,
        },
        "destination_position_bounds_destination_frame_m": {
            "minimum": center_lower,
            "maximum": center_upper,
        },
        "subject_orientation_destination_frame_xyzw": [0.0, 0.0, 0.0, 1.0],
        "support_height_interval_m": [0.80, 0.84],
        "intended_support_prim_paths": ["/Tray"],
        "insertion_withdrawal_unit_destination_frame": [0.0, 0.0, 1.0],
        "whole_subject_containment_encoded_by_shrunk_bounds": True,
        "geometry_digest": "",
    }
    geometry["geometry_digest"] = canonical_digest(
        geometry, digest_field="geometry_digest"
    )
    geometry_path = _write_json(tmp_path, "destination-geometry.json", geometry)
    if configured_scene_collision_path is None:
        configured_scene_collision_path = tmp_path / "configured-collision.usda"
        configured_scene_collision_path.write_bytes(b"#usda 1.0\n# collision\n")
    collision_digest = _sha(configured_scene_collision_path)
    support_plane_path = configured_scene_support_plane_path or _write_json(
        tmp_path,
        "destination-support-plane.json",
        {
            "schema_version": "task_evaluation_support_plane_input.v1",
            "scene_id": "841757",
            "sage_prim_path": "/Root/Support",
            "top_z_m": 0.275,
        },
    )
    scene_revision_digest = (
        configured_scene_revision_digest or "sha256:" + "e" * 64
    )
    placement = {
        "schema_version": "task_evaluation_rigid_destination_placement_qualification.v1",
        "status": "qualified",
        "producer": "task_evaluation_rigid_destination_placement_qualification",
        "native_observation_digest": "sha256:" + "f" * 64,
        "execution_commit": "a" * 40,
        "runtime_identity": {"id": "native-arena", "version": "v1"},
        "container_identity": {
            "image": NATIVE_TASK_ARENA_IMAGE,
            "digest": "sha256:"
            + NATIVE_TASK_ARENA_IMAGE.rsplit("@sha256:", 1)[-1],
        },
        "destination_identity": identity,
        "configured_scene_revision_digest": scene_revision_digest,
        "configured_scene_collision_digest": collision_digest,
        "configured_scene_support_plane_digest": _sha(support_plane_path),
        "destination_asset_digest": _sha(asset),
        "destination_static_qualification_digest": _sha(static_path),
        "destination_native_import_qualification_digest": _sha(native_path),
        "destination_geometry_digest": geometry["geometry_digest"],
        "pose_world": pose,
        "nonpenetration_passed": True,
        "support_stability_passed": True,
        "native_measurement_summary": {
            "maximum_penetration_m": 0.0001,
            "minimum_support_contact_force_n": 1.0,
            "maximum_forbidden_contact_force_n": 0.0,
            "raw_measurement_artifact_count": 5,
        },
        "no_policy_execution": {
            "policy_loaded": False,
            "candidate_policy_queried": False,
            "candidate_outcomes_accessed": False,
            "policy_actions_executed": 0,
        },
        "camera_visibility": {"external": True, "wrist": True, "overview": True},
        "repeated_reset_readback": {
            "repeat_count": 3,
            "maximum_translation_error_m": 0.0001,
            "maximum_rotation_error_rad": 0.0001,
            "translation_tolerance_m": 0.002,
            "rotation_tolerance_rad": 0.01,
        },
        "placement_qualification_digest": "",
    }
    placement["placement_qualification_digest"] = canonical_digest(
        placement, digest_field="placement_qualification_digest"
    )
    docs = {
        "task.destination.asset": asset,
        "task.destination.rights_admission": rights,
        "task.destination.static_qualification": static_path,
        "task.destination.native_import_qualification": native_path,
        "task.destination.geometry": geometry_path,
        "task.destination.placement_qualification": _write_json(
            tmp_path, "destination-placement.json", placement
        ),
        "scene.configured_revision.replacement.static_qualification": subject_static_path,
        "scene.configured_revision.registration.support_plane": support_plane_path,
    }
    references = {
        contract_path: _record(path, contract_path)
        for contract_path, path in docs.items()
    }
    request_value = {
        "expected_production_commit": "a" * 40,
        "runtime": {"identity": {"id": "native-arena", "version": "v1"}},
        "task": {
            "strategy": "pick_and_place",
            "configured_scene_revision_digest": scene_revision_digest,
            "subject": {"identity": subject_identity},
            "destination": {
                "schema_version": "task_evaluation_rigid_destination_asset.v1",
                "identity": identity,
                "relation": "inside",
                "visible_label": "blue document tray",
                "pose_world": pose,
                "provider_disclosure_allowed": True,
            },
        }
    }
    context = {
        "configured_collision_path": configured_scene_collision_path,
        "task_spec": {
            "interaction_affordance": {
                "asset_root_from_scoring_frame": subject_scoring_transform
            }
        },
    }
    return request_value, references, context


def test_destination_asset_is_qualified_staged_and_compiler_ready(
    tmp_path: Path,
) -> None:
    request_value, references, context = _destination_case(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    result = _stage_destination_asset(
        request=request_value,
        materialized_references=references,
        output_root=output,
        configured_collision_path=context["configured_collision_path"],
        task_spec=context["task_spec"],
    )

    assert result["asset_id"] == "document_tray"
    assert result["source_asset_id"] == "document-tray"
    assert result["relation"] == "inside"
    assert result["intended_support_prim_paths"] == ["/Tray"]
    assert result["target_position_world_m"] == pytest.approx(
        [3.2, -6.76, 0.9425]
    )
    assert Path(result["path"]).is_file()
    assert output in Path(result["path"]).parents


def test_large_nurec_requires_a_cached_official_particlefield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = {
        "positions": np.zeros((2, 3), dtype=np.float32),
        "rotations": np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        "scales": np.full((2, 3), -2.0, dtype=np.float32),
        "densities": np.zeros((2, 1), dtype=np.float32),
        "features_albedo": np.zeros((2, 3), dtype=np.float32),
        "features_specular": np.zeros((2, 45), dtype=np.float32),
    }
    document = {
        "version": "0.2.576",
        "model": "nre",
        "config": {
            "layers": {
                "gaussians": {
                    "precision": 32,
                    "density_activation": "sigmoid",
                    "scale_activation": "exp",
                    "rotation_activation": "normalize",
                    "particle": {"density_kernel_planar": False, "radiance_sph_degree": 3},
                }
            },
            "renderer": {"name": "3dgut-nrend"},
        },
        "state_dict": build_state_dict(arrays, precision=32),
    }
    source = tmp_path / "appearance.usdz"
    write_aura_nurec_usdz(document, source)
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_native_arena_episode_compiler."
        "materialize_cached_particlefield",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_native_arena_episode_compiler."
        "MAXIMUM_INLINE_NUREC_CONVERSION_BYTES",
        0,
    )

    with pytest.raises(
        TaskEvaluationNativeArenaEpisodeCompilerError,
        match="episode_compiler_official_particlefield_cache_required",
    ):
        _materialize_native_particlefield_appearance(
            source_path=source,
            output_root=tmp_path / "native-appearance",
        )


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
    assert controller["identity"] == {
        "id": "paired-droid-policy-canary",
        "version": "v2",
    }
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
            "collision_bounds_body_frame_m": {
                "minimum": [-0.0638749, -0.0658936, 0.0],
                "maximum": [0.0638749, 0.0658936, 0.127638],
            },
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


@pytest.mark.parametrize(
    ("policy_observation_override", "destination_support", "qualification_only"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_closed_compiler_joins_revision_and_robot_team_inputs(
    tmp_path: Path,
    monkeypatch,
    policy_observation_override: bool,
    destination_support: bool,
    qualification_only: bool,
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
            "scene_camera_calibration_digest": configured["registration"]["camera_calibration"][
                "digest"
            ],
            "cameras": (
                [
                    {
                        "role": role,
                        "intrinsics": {
                            "fx": 172.0,
                            "fy": 172.0,
                            "cx": 159.5,
                            "cy": 89.5,
                            "width": 320,
                            "height": 180,
                        },
                    }
                    for role in ("external", "wrist", "overview")
                ]
                if policy_observation_override
                else [{"role": "external"}]
            ),
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
    if destination_support:
        native_adapter = compiler.adapt_rigid_relocation_task_template

        def grounded_adapter(**kwargs):
            result = native_adapter(**kwargs)
            spec = result["native_task_definition"]["task_spec"]
            spec.update(
                instruction_subject_label="open book",
                prompt="Pick up the open book, place it fully inside the blue document tray, release it, and move the gripper clear.",
            )
            result["adapter_digest"] = canonical_digest(result, digest_field="adapter_digest")
            return result

        monkeypatch.setattr(compiler, "adapt_rigid_relocation_task_template", grounded_adapter)
    if destination_support:
        value["task"]["strategy"] = "pick_and_place"
        docs["scene.configured_revision.task_template.definition"][
            "strategy"
        ] = "pick_and_place"
        docs["scene.configured_revision.task_template.definition"][
            "interaction_affordance"
        ] = {
            "contact_point_scoring_frame_m": [-0.06, 0.0, 0.0],
            "approach_unit_scoring_frame": [-1.0, 0.0, 0.0],
            "jaw_unit_scoring_frame": [0.0, 1.0, 0.0],
            "lift_unit_world": [0.0, 0.0, 1.0],
            "pregrasp_clearance_m": 0.12,
            "minimum_lift_m": 0.08,
        }
        docs["scene.configured_revision.task_template.execution"][
            "strategy"
        ] = "pick_and_place"
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

    if destination_support:
        qualification_collision = inputs / "configured-collision-for-qualification.usda"
        qualification_collision.write_bytes(b"#usda 1.0\n# collision\n")
        subject_static_path = document_paths[
            "scene.configured_revision.replacement.static_qualification"
        ]
        subject_static = docs[
            "scene.configured_revision.replacement.static_qualification"
        ]
        destination_request, destination_references, _context = _destination_case(
            inputs,
            subject_identity=configured["replacement"]["identity"],
            subject_static_path=subject_static_path,
            subject_static=subject_static,
            subject_scoring_transform={
                "position_m": subject_static["observed_structure"][
                    "center_of_mass_m"
                ],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            configured_scene_revision_digest=configured["revision_digest"],
            configured_scene_collision_path=qualification_collision,
            configured_scene_support_plane_path=document_paths[
                "scene.configured_revision.registration.support_plane"
            ],
        )
        destination = destination_request["task"]["destination"]
        for field, contract_path in (
            ("asset", "task.destination.asset"),
            ("rights_admission", "task.destination.rights_admission"),
            ("static_qualification", "task.destination.static_qualification"),
            (
                "native_import_qualification",
                "task.destination.native_import_qualification",
            ),
            ("geometry", "task.destination.geometry"),
            (
                "placement_qualification",
                "task.destination.placement_qualification",
            ),
        ):
            record = destination_references[contract_path]
            destination[field] = {
                key: record[key] for key in ("uri", "digest", "size_bytes")
            }
        value["task"]["destination"] = destination
        for contract_path, record in destination_references.items():
            references.setdefault(contract_path, record)
        if qualification_only:
            value["run_mode"] = "destination_qualification"
            value["task"]["destination"].pop("placement_qualification")
            references.pop("task.destination.placement_qualification")
            value["task"]["destination"]["native_probe"] = {
                "schema_version": (
                    "task_evaluation_rigid_destination_native_probe_configuration.v1"
                ),
                "placement_support_scene_prim_paths": ["/Root/Support"],
                "qualification_limits": {
                    "maximum_penetration_m": 0.001,
                    "minimum_support_contact_force_n": 0.01,
                    "maximum_forbidden_contact_force_n": 0.1,
                    "settle_translation_tolerance_m": 0.002,
                    "settle_rotation_tolerance_rad": 0.01,
                    "reset_translation_tolerance_m": 0.002,
                    "reset_rotation_tolerance_rad": 0.01,
                    "minimum_camera_pixels": {
                        "external": 100,
                        "wrist": 100,
                        "overview": 100,
                    },
                },
                "settle_sample_count": 3,
                "settle_steps_per_sample": 60,
            }

    if policy_observation_override:
        appearance = inputs / "policy-observation.usdc"
        appearance.write_bytes(b"verified policy observation particlefield")
        receipt = {
            "schema_version": "nvidia_3dgrut_particlefield_transcode.v1",
            "status": "completed",
            "schema": "ParticleField3DGaussianSplat",
            "output": "/producer/path/not-present-on-control-plane.usdc",
            "output_bytes": appearance.stat().st_size,
            "output_sha256": "sha256:"
            + hashlib.sha256(appearance.read_bytes()).hexdigest(),
            "source_sha256": "sha256:" + "a" * 64,
            "splat_count": 1_000_000,
            "sh_degree": 3,
            "sh_primvar_element_size": 16,
            "sh_primvar_interpolation": "constant",
            "display_color_fallback_authored": False,
            "particlefield_emissive_material_binding_authored": False,
            "particlefield_emissive_material_inputs": None,
            "particlefield_custom_render_hints_authored": False,
            "particlefield_authoring_implementation": (
                "nvidia_3dgrut_direct_nurec_transcode"
            ),
            "upstream_converter": {
                "repository": "https://github.com/nv-tlabs/3dgrut.git",
                "source_revision": "a37ef721012dea0f29c0fcfff2d525023b4e854a",
                "module": "threedgrut.export.scripts.transcode",
                "module_sha256": "sha256:" + "b" * 64,
                "source_identity_verified": True,
            },
            "upstream_projection_mode_hint": "perspective",
            "upstream_sorting_mode_hint": "cameraDistance",
            "upstream_color_space": "srgb_rec709_display",
            "gaussian_field_quality": {
                "schema_version": "gaussian_field_quality.v1",
                "status": "qualified",
                "blockers": [],
                "learned_tensors_mutated": False,
            },
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        receipt_path = _write_json(inputs, "appearance-receipt.json", receipt)
        registry_path = (
            Path(__file__).resolve().parents[1]
            / "docs/arm_decision_proof_v1/manifests/"
            "franka_robotiq_policy_camera_mount_registry.v1.json"
        )
        observation_refs = {
            "appearance_asset": _record(
                appearance,
                "execution_adapter.policy_observation_setup.appearance_asset",
            ),
            "appearance_authoring_receipt": _record(
                receipt_path,
                "execution_adapter.policy_observation_setup.appearance_authoring_receipt",
            ),
            "wrist_camera_mount_registry": _record(
                registry_path,
                "execution_adapter.policy_observation_setup.wrist_camera_mount_registry",
            ),
        }
        value["execution_adapter"]["policy_observation_setup"] = {
            "schema_version": "task_evaluation_policy_observation_setup.v1",
            **{
                key: {
                    field: record[field]
                    for field in ("uri", "digest", "size_bytes")
                }
                for key, record in observation_refs.items()
            },
            "fresh_native_mount_sweep_required": True,
            "policy_master_resolution_wh": [640, 360],
            "overview_review_resolution_wh": [1280, 720],
        }
        references.update(
            {record["contract_path"]: record for record in observation_refs.values()}
        )

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
        observed["configured_appearance_materialized"] = True
        output_root.mkdir()
        output = output_root / "scene_appearance.usdc"
        output.write_bytes(Path(source_path).read_bytes())
        digest, size = (
            "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
            output.stat().st_size,
        )
        backend = {
            "schema_version": "task_evaluation_appearance_render_backend.v1",
            "kind": "nvidia_3dgrut_direct_nurec_transcode",
            "source_configured_appearance_digest": digest,
            "particlefield_digest": digest,
            "authoring_receipt_digest": "sha256:" + "9" * 64,
            "upstream_converter": {"source_revision": "test"},
            "projection_mode_hint": "perspective",
            "sorting_mode_hint": "cameraDistance",
            "color_space": "srgb_rec709_display",
            "backend_digest": "",
        }
        backend["backend_digest"] = canonical_digest(
            backend, digest_field="backend_digest"
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
            "appearance_render_backend": backend,
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
    assert ("destination_native_probe_request" in result) is qualification_only
    if qualification_only:
        probe = json.loads(
            Path(result["destination_native_probe_request"]["path"]).read_text()
        )
        assert probe["execution_commit"] == value["expected_production_commit"]
        assert probe["candidate_policy_queried"] is False
        assert observed["packet_request"]["task_spec"][
            "destination_qualification_probe"
        ] is True
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
    if policy_observation_override:
        assert "configured_appearance_materialized" not in observed
        assert result["native_scene_appearance"]["policy_observation_setup_bound"] is True
        assert observed["packet_request"]["appearance_variant"][
            "particlefield_authoring_implementation"
        ] == "nvidia_3dgrut_direct_nurec_transcode"
        assert observed["packet_request"]["appearance_variant"][
            "source_gaussian_sha256"
        ] == ("sha256:" + "a" * 64)
        assert observed["packet_request"]["wrist_camera_mount_registry"][
            "selection_authority"
        ] == "native_render_measurements"
        assert observed["packet_request"]["camera_resolution_contract"] == {
            "policy_master_resolution_wh": [640, 360],
            "overview_review_resolution_wh": [1280, 720],
            "policy_preprocessing": (
                "candidate_specific_aspect_preserving_resize_with_centred_black_pad"
            ),
            "source_request_digest": json.loads(
                (tmp_path / "output/policy_observation_appearance_packet_request.v1.json")
                .read_text(encoding="utf-8")
            )["request_digest"],
            "fresh_native_mount_sweep_required": True,
        }
    else:
        assert observed["packet_request"]["appearance_variant"] == {
            "representation": "particlefield_3d_gaussian_splat",
            "source_configured_appearance_digest": result[
                "native_scene_appearance"
            ]["source_configured_appearance_digest"],
            "representation_conversion_performed": True,
            "exact_learned_arrays_preserved": True,
            "gaussian_field_quality": {
                "schema_version": "gaussian_field_quality.v1",
                "status": "qualified",
                "blockers": [],
                "learned_tensors_mutated": False,
            },
            "render_backend": result["native_scene_appearance"][
                "appearance_render_backend"
            ],
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
    expected_strategy = "pick_and_place" if destination_support else "planar_push"
    assert observed["packet_request"]["task_spec"]["manipulation_strategy"] == (
        expected_strategy
    )
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
        == expected_strategy
    )
    assert result["configured_task_template_adapter"]["source_documents_digest"].startswith(
        "sha256:"
    )
    replacement = next(
        row for row in observed["packet_request"]["assets"] if row["semantic_role"] == "task_object"
    )
    assert replacement["asset_id"] == configured["replacement"]["identity"]["id"].replace("-", "_")
    assert replacement["source_asset_id"] == configured["replacement"]["identity"]["id"]
    if destination_support:
        support = next(
            row
            for row in observed["packet_request"]["assets"]
            if row["semantic_role"] == "task_support"
        )
        assert observed["packet_request"]["task_spec"]["prompt"] == (
            "Pick up the open book, place it fully inside the blue document tray, "
            "release it, and move the gripper clear."
        )
        assert support["asset_id"] == "document_tray"
        assert support["source_asset_id"] == "document-tray"
        assert observed["packet_request"]["task_spec"][
            "destination_relation"
        ] == "inside"
        assert observed["packet_request"]["task_spec"][
            "visible_target_label"
        ] == "blue document tray"


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
