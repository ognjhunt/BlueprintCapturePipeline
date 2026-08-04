from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_suite_admission import (
    build_public_scene_suite_admission_receipt,
)


SCHEMA_PATH = (
    Path(__file__).parents[1]
    / "docs"
    / "schemas"
    / "public_scene_suite_manifest.v1.schema.json"
)
IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
EVALUATED_ON = dt.date(2026, 8, 4)


def _artifact(
    artifact_id: str,
    role: str,
    *,
    digest_character: str,
    size_bytes: int,
    extension: str = "bin",
) -> dict:
    return {
        "artifact_id": artifact_id,
        "scene_id": "scene-001",
        "role": role,
        "relative_path": f"scene-001/{artifact_id}.{extension}",
        "sha256": "sha256:" + digest_character * 64,
        "size_bytes": size_bytes,
    }


def _rights(source_id: str, *, digest_character: str) -> dict:
    return {
        "source_id": source_id,
        "license_expression": "Fixture-Research-Only",
        "terms_url": f"https://example.invalid/{source_id}/terms",
        "terms_text_sha256": "sha256:" + digest_character * 64,
        "use_scope": "noncommercial_internal_research",
        "reviewer_status": "approved_for_declared_use",
        "reviewer_id": "fixture-rights-reviewer",
        "reviewed_on": "2026-08-04",
        "access_authority_reference": "fixture-human-acceptance-receipt",
        "expiration_policy": "no_expiration_declared",
        "valid_through": None,
        "agent_accepted_terms": False,
    }


def _frame(
    source_id: str, evidence_artifact_id: str, conversion_artifact_id: str
) -> dict:
    return {
        "source_id": source_id,
        "scene_id": "scene-001",
        "native_units": "meters",
        "native_unit_scale_to_meters": 1.0,
        "units": "meters",
        "handedness": "right_handed",
        "up_axis": "+Z",
        "world_frame": "blueprint_world_right_handed_z_up_meters",
        "normalization_history": {
            "status": "none",
            "reference": "fixture source declares no normalization",
            "inverse_transform": None,
        },
        "unit_conversion_artifact_ids": [conversion_artifact_id],
        "source_to_world": copy.deepcopy(IDENTITY),
        "world_to_source": copy.deepcopy(IDENTITY),
        "metric_scale_authority": {
            "kind": "authored_metric_environment",
            "authority_reference": "fixture authoring coordinates declared in meters",
            "evidence_artifact_ids": [evidence_artifact_id],
        },
    }


def _manifest() -> dict:
    value = {
        "schema_version": "public_scene_suite_manifest.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009A",
        "phase_label": "public_scene_qualification",
        "suite_id": "fixture-metric-hybrid-scene-v1",
        "admission_as_of": "2026-08-04",
        "suite_purpose": "public_scene_software_qualification",
        "component_scope": "hybrid_edit_replacement_case",
        "sources": [
            {
                "source_id": "fixture-appearance",
                "upstream_project_id": "InteriorGS",
                "source_kind": "synthetic_metric_scene",
                "scene_id": "scene-001",
                "revision": {"kind": "release_tag", "value": "v1.0.0"},
                "source_url": "https://example.invalid/appearance/v1.0.0",
                "artifacts": [
                    _artifact(
                        "scene-splat",
                        "appearance_3dgs",
                        digest_character="1",
                        size_bytes=101,
                    ),
                    _artifact(
                        "appearance-scale",
                        "metric_scale_evidence",
                        digest_character="2",
                        size_bytes=102,
                    ),
                    _artifact(
                        "calibration-view-001",
                        "calibration_observation",
                        digest_character="4",
                        size_bytes=104,
                    ),
                    _artifact(
                        "test-view-001",
                        "test_observation",
                        digest_character="5",
                        size_bytes=105,
                    ),
                    _artifact(
                        "method-rgb-001",
                        "method_input_rgb",
                        digest_character="d",
                        size_bytes=105,
                    ),
                    _artifact(
                        "method-mask-001",
                        "method_input_mask",
                        digest_character="e",
                        size_bytes=105,
                    ),
                    _artifact(
                        "method-splat-depth-001",
                        "method_input_splat_depth",
                        digest_character="f",
                        size_bytes=105,
                    ),
                    _artifact(
                        "render-camera-model",
                        "camera_model_bundle",
                        digest_character="c",
                        size_bytes=105,
                    ),
                    _artifact(
                        "appearance-conversion",
                        "unit_conversion_receipt",
                        digest_character="6",
                        size_bytes=106,
                    ),
                ],
            },
            {
                "source_id": "fixture-collision",
                "upstream_project_id": "SAGE-3D",
                "source_kind": "collision_companion",
                "scene_id": "scene-001",
                "revision": {"kind": "git_commit", "value": "4" * 40},
                "source_url": "https://example.invalid/collision/commit/4444",
                "artifacts": [
                    _artifact(
                        "static-collider",
                        "static_collision_geometry",
                        digest_character="5",
                        size_bytes=105,
                    ),
                    _artifact(
                        "collision-scale",
                        "metric_scale_evidence",
                        digest_character="6",
                        size_bytes=106,
                    ),
                    _artifact(
                        "collision-conversion",
                        "unit_conversion_receipt",
                        digest_character="a",
                        size_bytes=107,
                    ),
                ],
            },
            {
                "source_id": "fixture-task-object",
                "upstream_project_id": "Blueprint-controlled",
                "source_kind": "simready_task_object",
                "scene_id": "scene-001",
                "revision": {"kind": "content_digest", "value": "sha256:" + "7" * 64},
                "source_url": "https://example.invalid/task-object/apple-v1",
                "artifacts": [
                    _artifact(
                        "apple-usd",
                        "simready_usd_package",
                        digest_character="e",
                        size_bytes=107,
                        extension="usd",
                    ),
                    _artifact(
                        "apple-visual",
                        "task_object_visual_geometry",
                        digest_character="3",
                        size_bytes=108,
                    ),
                    _artifact(
                        "apple-collider",
                        "task_object_collision_geometry",
                        digest_character="7",
                        size_bytes=109,
                    ),
                    _artifact(
                        "apple-physics",
                        "task_object_physics_metadata",
                        digest_character="d",
                        size_bytes=110,
                    ),
                    _artifact(
                        "apple-scale",
                        "metric_scale_evidence",
                        digest_character="f",
                        size_bytes=111,
                    ),
                    _artifact(
                        "apple-conversion",
                        "unit_conversion_receipt",
                        digest_character="b",
                        size_bytes=112,
                    ),
                ],
            },
        ],
        "scene_pairings": [
            {
                "pairing_id": "fixture-appearance-collision-pair",
                "appearance_source_id": "fixture-appearance",
                "appearance_scene_id": "scene-001",
                "appearance_artifact_id": "scene-splat",
                "collision_source_id": "fixture-collision",
                "collision_scene_id": "scene-001",
                "collision_artifact_id": "static-collider",
                "exact_scene_match_required": True,
            }
        ],
        "rights_reviews": [
            _rights("fixture-appearance", digest_character="8"),
            _rights("fixture-collision", digest_character="9"),
            _rights("fixture-task-object", digest_character="0"),
        ],
        "code_dependencies": [
            {
                "dependency_id": "fixture-splat-renderer",
                "purpose": "Materialize frozen render-derived RGB observations",
                "capability_role": "scene_materialization",
                "upstream_project_id": "fixture-splat-renderer",
                "repository_url": "https://example.invalid/fixture-splat-renderer.git",
                "availability": "released",
                "revision": {"kind": "git_commit", "value": "9" * 40},
                "license": {
                    "license_expression": "Apache-2.0",
                    "terms_url": "https://example.invalid/fixture-splat-renderer/license",
                    "text_sha256": "sha256:" + "8" * 64,
                },
                "smoke_status": "passed",
                "smoke_receipt_digest": "sha256:" + "7" * 64,
                "runtime_lock_digest": "sha256:" + "6" * 64,
                "dependency_license_inventory_digest": "sha256:" + "5" * 64,
            },
            {
                "dependency_id": "infusion",
                "purpose": "Primary world-frame supplemental Gaussian adapter",
                "capability_role": "background_completion_primary_adapter",
                "upstream_project_id": "InFusion",
                "repository_url": "https://example.invalid/infusion.git",
                "availability": "released",
                "revision": {"kind": "git_commit", "value": "a" * 40},
                "license": {
                    "license_expression": "Apache-2.0",
                    "terms_url": "https://example.invalid/fixture-loader/license",
                    "text_sha256": "sha256:" + "b" * 64,
                },
                "smoke_status": "passed",
                "smoke_receipt_digest": "sha256:" + "c" * 64,
                "runtime_lock_digest": "sha256:" + "1" * 64,
                "dependency_license_inventory_digest": "sha256:" + "2" * 64,
            },
            {
                "dependency_id": "aurafusion360",
                "purpose": "Released multiview quality challenger",
                "capability_role": "background_completion_quality_challenger",
                "upstream_project_id": "AuraFusion360",
                "repository_url": "https://example.invalid/aurafusion360.git",
                "availability": "released",
                "revision": {"kind": "git_commit", "value": "b" * 40},
                "license": {
                    "license_expression": "Apache-2.0-with-dependency-review",
                    "terms_url": "https://example.invalid/aurafusion360/license",
                    "text_sha256": "sha256:" + "3" * 64,
                },
                "smoke_status": "passed",
                "smoke_receipt_digest": "sha256:" + "4" * 64,
                "runtime_lock_digest": "sha256:" + "5" * 64,
                "dependency_license_inventory_digest": "sha256:" + "6" * 64,
            },
        ],
        "coordinate_frames": [
            _frame("fixture-appearance", "appearance-scale", "appearance-conversion"),
            _frame("fixture-collision", "collision-scale", "collision-conversion"),
            _frame("fixture-task-object", "apple-scale", "apple-conversion"),
        ],
        "splits": {
            "calibration_trajectory_ids": ["calibration-view-001"],
            "test_trajectory_ids": ["test-view-001"],
        },
        "observation_bundle": {
            "bundle_id": "fixture-render-derived-observations-v1",
            "origin": "render_derived_synthetic",
            "camera_model": "COLMAP_PINHOLE",
            "camera_model_artifact_id": "render-camera-model",
            "appearance_artifact_id": "scene-splat",
            "materialization_dependency_id": "fixture-splat-renderer",
            "rgb_authority": "appearance_3dgs_render",
            "camera_preparation": "prebuilt_metric_colmap_without_mapper",
            "calibration_artifact_ids": ["calibration-view-001"],
            "test_artifact_ids": ["test-view-001"],
            "method_profiles": [
                {
                    "profile_id": "infusion-blueprint-adapter-v1",
                    "method_role": "primary_interface_adapter",
                    "dependency_id": "infusion",
                    "upstream_project_id": "InFusion",
                    "input_artifact_ids": [
                        "method-rgb-001",
                        "method-mask-001",
                        "method-splat-depth-001",
                        "render-camera-model",
                    ],
                    "input_modalities": [
                        "object_present_rgb",
                        "multiview_object_masks",
                        "splat_rendered_inverse_depth",
                        "camera_intrinsics",
                        "camera_to_world",
                    ],
                    "input_mount_policy": "allowlist_only",
                    "writes_delta_layer": True,
                    "preserves_source_world_frame": True,
                    "external_validation_oracle_access": False,
                },
                {
                    "profile_id": "aurafusion360-blueprint-adapter-v1",
                    "method_role": "multiview_quality_challenger",
                    "dependency_id": "aurafusion360",
                    "upstream_project_id": "AuraFusion360",
                    "input_artifact_ids": [
                        "method-rgb-001",
                        "method-mask-001",
                        "render-camera-model",
                        "scene-splat",
                    ],
                    "input_modalities": [
                        "object_present_rgb",
                        "multiview_object_masks",
                        "camera_intrinsics",
                        "camera_to_world",
                        "source_3dgs",
                    ],
                    "input_mount_policy": "allowlist_only",
                    "writes_delta_layer": True,
                    "preserves_source_world_frame": True,
                    "external_validation_oracle_access": False,
                },
            ],
            "validation_oracle": {
                "availability": "unavailable",
                "depth_artifact_id": None,
                "geometry_artifact_id": None,
                "authority": "none",
                "usage": "not_available",
                "method_access": False,
                "independent_of_method_inputs": True,
            },
            "truth_contract": {
                "availability": "unavailable",
                "clean_background_artifact_ids": [],
                "method_access": False,
                "edit_result_digest": None,
                "edit_seal_digest": None,
                "truth_release_join_digest": None,
            },
            "object_present_inputs": True,
            "unscaled_sfm_rerun": False,
            "partitions_disjoint": True,
            "independent_capture_evidence": False,
        },
        "representations": {
            "active_pairing_id": "fixture-appearance-collision-pair",
            "appearance": {
                "kind": "3dgs",
                "usage": "appearance_only",
                "artifact_ids": ["scene-splat"],
                "metric_measurement_authority": False,
                "collision_authority": False,
            },
            "metric_geometry": {
                "kind": "publisher_metric_frame_and_boxes",
                "usage": "metric_frame_reference_only",
                "artifact_ids": [],
                "measurement_authority": False,
            },
            "collision": {
                "kind": "openusd_collision",
                "usage": "collision_only",
                "artifact_ids": ["static-collider"],
                "separate_from_appearance": True,
            },
            "task_objects": [
                {
                    "task_object_id": "apple",
                    "source_object_id": "source-apple",
                    "replacement_mode": "remove_source_then_insert_exact_usd",
                    "asset_format": "simready_usd",
                    "asset_source_id": "fixture-task-object",
                    "usd_artifact_id": "apple-usd",
                    "visual_artifact_ids": ["apple-visual"],
                    "collision_artifact_ids": ["apple-collider"],
                    "physics_artifact_ids": ["apple-physics"],
                    "physics_properties": {
                        "dimensions_m": [0.079, 0.076, 0.083],
                        "mass_kg": 0.18,
                        "center_of_mass_m": [0.0, 0.0, 0.039],
                        "inertia_tensor_kg_m2": [
                            [0.0002, 0.0, 0.0],
                            [0.0, 0.0002, 0.0],
                            [0.0, 0.0, 0.0002],
                        ],
                        "static_friction": 0.55,
                        "dynamic_friction": 0.45,
                        "restitution": 0.1,
                        "authority": "preregistered_candidate",
                        "contact_material_id": "fruit-skin-candidate-v1",
                    },
                    "visual_and_collision_are_separate": True,
                    "source_object_pose_world": copy.deepcopy(IDENTITY),
                    "replacement_pose_world": copy.deepcopy(IDENTITY),
                    "reset_pose_world": copy.deepcopy(IDENTITY),
                    "pose_authority": "source_annotation",
                    "dimensions_uncertainty_m": [0.001, 0.001, 0.001],
                    "support_contact_point_world_m": [0.0, 0.0, 0.0],
                    "support_normal_world": [0.0, 0.0, 1.0],
                    "semantic_label": "apple",
                    "reset_state_id": "apple-reset-v1",
                }
            ],
        },
        "claim_ceiling": "development_only",
        "claim_boundaries": {
            "public_scene_manifest_admission": True,
            "public_scene_software_qualification": False,
            "artifact_bytes_verified": False,
            "metric_geometry_qualified": False,
            "task_physics_qualified": False,
            "partner_capture_qualified": False,
            "prospective_validation": False,
            "digital_twin": False,
            "deployment_readiness": False,
            "physical_safety": False,
            "customer_value": False,
            "general_sim_to_real_fidelity": False,
        },
    }
    value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")
    return value


def _redigest(value: dict) -> None:
    value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")


def _receipt(value: dict) -> dict:
    return build_public_scene_suite_admission_receipt(
        value, evaluated_on=EVALUATED_ON
    )


def test_happy_synthetic_manifest_is_schema_valid_and_admitted() -> None:
    value = _manifest()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)

    receipt = _receipt(value)

    assert receipt["status"] == "component_admitted"
    assert receipt["blockers"] == []
    assert receipt["manifest_digest"] == value["manifest_digest"]
    assert receipt["manifest_ready_for_materialization"] is True
    assert receipt["artifact_bytes_opened"] is False
    assert receipt["artifact_bytes_verified"] is False
    assert receipt["public_scene_software_qualified"] is False
    assert receipt["metric_geometry_qualified"] is False
    assert receipt["task_physics_qualified"] is False
    assert receipt["existing_adp_008_artifacts_modified"] is False
    assert receipt["adp009a_matrix_complete"] is False
    assert receipt["qualification_role"] == "component_admission_only"
    assert receipt["observation_bundle_id"] == (
        "fixture-render-derived-observations-v1"
    )
    assert receipt["observation_origin"] == "render_derived_synthetic"
    assert receipt["independent_capture_evidence"] is False
    assert receipt == _receipt(value)
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_released_simready_authoring_dependency_role_is_admissible() -> None:
    value = _manifest()
    value["code_dependencies"].append(
        {
            "dependency_id": "nvidia-usd-content-agents",
            "purpose": "Candidate geometry-preserving USD physicalization backend",
            "capability_role": "simready_authoring",
            "upstream_project_id": "nvidia-usd-content-agents",
            "repository_url": "https://example.invalid/usd-content-agents.git",
            "availability": "released",
            "revision": {"kind": "git_commit", "value": "3" * 40},
            "license": {
                "license_expression": "Apache-2.0",
                "terms_url": "https://example.invalid/usd-content-agents/license",
                "text_sha256": "sha256:" + "4" * 64,
            },
            "smoke_status": "passed",
            "smoke_receipt_digest": "sha256:" + "5" * 64,
            "runtime_lock_digest": "sha256:" + "6" * 64,
            "dependency_license_inventory_digest": "sha256:" + "7" * 64,
        }
    )
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "component_admitted"
    assert receipt["blockers"] == []


def test_consumer_capture_proxy_is_not_an_active_component_kind() -> None:
    value = _manifest()
    value["sources"][0]["source_kind"] = "consumer_capture_proxy"
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "sources[0].source_kind:invalid" in receipt["blockers"]


@pytest.mark.parametrize("project_id", ["ARKitScenes", "WildRGB-D"])
def test_removed_dataset_projects_cannot_enter_active_component(project_id: str) -> None:
    value = _manifest()
    value["sources"][0]["upstream_project_id"] = project_id
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "sources[0].upstream_project_id:not_active_program_source"
        in receipt["blockers"]
    )


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        (
            "unscaled_sfm_rerun",
            True,
            "observation_bundle.unscaled_sfm_rerun:must_be:false",
        ),
        (
            "independent_capture_evidence",
            True,
            "observation_bundle.independent_capture_evidence:must_be_false",
        ),
        (
            "camera_preparation",
            "source_calibration",
            "observation_bundle.camera_preparation:must_be:prebuilt_metric_colmap_without_mapper",
        ),
    ],
)
def test_render_derived_observation_bundle_fails_closed_on_truth_leakage(
    field: str, value: object, blocker: str
) -> None:
    manifest = _manifest()
    manifest["observation_bundle"][field] = value
    _redigest(manifest)

    receipt = _receipt(manifest)

    assert receipt["status"] == "blocked"
    assert blocker in receipt["blockers"]


def test_render_derived_observation_bundle_must_bind_the_frozen_splits() -> None:
    manifest = _manifest()
    manifest["observation_bundle"]["test_artifact_ids"] = [
        "calibration-view-001"
    ]
    _redigest(manifest)

    receipt = _receipt(manifest)

    assert receipt["status"] == "blocked"
    assert (
        "observation_bundle.test_artifact_ids:split_mismatch"
        in receipt["blockers"]
    )


def test_metric_geometry_cannot_come_from_an_unrelated_scene() -> None:
    manifest = _manifest()
    unrelated_metric = _artifact(
        "unrelated-metric-surface",
        "metric_geometry",
        digest_character="4",
        size_bytes=201,
    )
    unrelated_scale = _artifact(
        "unrelated-scale",
        "metric_scale_evidence",
        digest_character="5",
        size_bytes=202,
    )
    unrelated_conversion = _artifact(
        "unrelated-conversion",
        "unit_conversion_receipt",
        digest_character="6",
        size_bytes=203,
    )
    for artifact in (unrelated_metric, unrelated_scale, unrelated_conversion):
        artifact["scene_id"] = "unrelated-scene"
    manifest["sources"].append(
        {
            "source_id": "unrelated-source",
            "upstream_project_id": "ScanNet++",
            "source_kind": "real_metric_scene",
            "scene_id": "unrelated-scene",
            "revision": {"kind": "git_commit", "value": "8" * 40},
            "source_url": "https://example.invalid/unrelated-source.git",
            "artifacts": [
                unrelated_metric,
                unrelated_scale,
                unrelated_conversion,
            ],
        }
    )
    manifest["rights_reviews"].append(
        _rights("unrelated-source", digest_character="7")
    )
    unrelated_frame = _frame(
        "unrelated-source", "unrelated-scale", "unrelated-conversion"
    )
    unrelated_frame["scene_id"] = "unrelated-scene"
    manifest["coordinate_frames"].append(unrelated_frame)
    manifest["representations"]["metric_geometry"] = {
        "kind": "laser_mesh",
        "usage": "measurement_authority",
        "artifact_ids": ["unrelated-metric-surface"],
        "measurement_authority": True,
    }
    manifest["observation_bundle"]["validation_oracle"] = {
        "availability": "available",
        "depth_artifact_id": None,
        "geometry_artifact_id": "unrelated-metric-surface",
        "authority": "source_sensor_or_laser",
        "usage": "evaluation_only",
        "method_access": False,
        "independent_of_method_inputs": True,
    }
    _redigest(manifest)

    receipt = _receipt(manifest)

    assert receipt["status"] == "blocked"
    assert (
        "representations.metric_geometry.artifact_ids:not_active_scene:unrelated-metric-surface"
        in receipt["blockers"]
    )


def test_missing_rights_review_blocks_admission() -> None:
    value = _manifest()
    value["rights_reviews"] = []
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "rights_reviews:missing" in receipt["blockers"]
    assert "rights_reviews:missing_source:fixture-appearance" in receipt["blockers"]
    assert "rights_reviews:missing_source:fixture-collision" in receipt["blockers"]
    assert "rights_reviews:missing_source:fixture-task-object" in receipt["blockers"]


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    [
        (
            lambda value: value.__setitem__("unexpected", True),
            "unexpected:unknown_property",
        ),
        (
            lambda value: value["sources"][0].__setitem__("unexpected", True),
            "sources[0].unexpected:unknown_property",
        ),
    ],
)
def test_unknown_properties_fail_closed(mutation, expected_blocker: str) -> None:
    value = _manifest()
    mutation(value)
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert expected_blocker in receipt["blockers"]


def test_expired_rights_review_blocks_admission_deterministically() -> None:
    value = _manifest()
    review = value["rights_reviews"][0]
    review["expiration_policy"] = "expires_on_valid_through"
    review["valid_through"] = "2026-08-03"
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "rights_reviews[0].valid_through:expired" in receipt["blockers"]


def test_rights_review_cannot_postdate_external_evaluation_clock() -> None:
    value = _manifest()
    value["rights_reviews"][0]["reviewed_on"] = "2026-08-05"
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "rights_reviews[0].reviewed_on:after_evaluation_date" in receipt["blockers"]


def test_manifest_cannot_backdate_authoritative_evaluation_clock() -> None:
    value = _manifest()
    value["admission_as_of"] = "2026-08-03"
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "admission_as_of:does_not_match_authoritative_evaluation_date"
        in receipt["blockers"]
    )


def test_source_artifact_scene_mismatch_blocks_admission() -> None:
    value = _manifest()
    value["sources"][0]["artifacts"][0]["scene_id"] = "scene-999"
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "sources[0].artifacts[0].scene_id:source_scene_mismatch" in receipt["blockers"]


def test_exact_appearance_collision_pair_mismatch_blocks_admission() -> None:
    value = _manifest()
    value["scene_pairings"][0]["collision_scene_id"] = "scene-999"
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "scene_pairings[0].collision_scene_id:source_scene_mismatch" in receipt["blockers"]
    assert "scene_pairings[0]:exact_scene_pair_mismatch" in receipt["blockers"]


def test_active_representations_must_use_the_exact_active_pairing() -> None:
    value = _manifest()
    value["representations"]["appearance"]["artifact_ids"] = ["test-view-001"]
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "representations.appearance:does_not_match_active_pairing" in receipt["blockers"]


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_blocker"),
    [
        ("units", "centimeters", "coordinate_frames[0].units:must_be:meters"),
        ("up_axis", "unknown", "coordinate_frames[0].up_axis:unknown"),
    ],
)
def test_unknown_units_or_up_axis_blocks_admission(
    field: str,
    invalid_value: str,
    expected_blocker: str,
) -> None:
    value = _manifest()
    value["coordinate_frames"][0][field] = invalid_value
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert expected_blocker in receipt["blockers"]


def test_native_millimeter_source_is_preserved_before_metric_conversion() -> None:
    value = _manifest()
    frame = value["coordinate_frames"][0]
    frame["native_units"] = "millimeters"
    frame["native_unit_scale_to_meters"] = 0.001
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "component_admitted"


def test_normalized_source_requires_invertible_normalization_history() -> None:
    value = _manifest()
    frame = value["coordinate_frames"][0]
    frame["native_units"] = "unitless_normalized"
    frame["native_unit_scale_to_meters"] = 0.75
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "coordinate_frames[0].normalization_history:required_for_unitless_normalized"
        in receipt["blockers"]
    )


def test_manifest_tamper_is_bound_to_current_bytes_and_blocks() -> None:
    value = _manifest()
    supplied_digest = value["manifest_digest"]
    value["sources"][0]["artifacts"][0]["size_bytes"] += 1

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert receipt["supplied_manifest_digest"] == supplied_digest
    assert receipt["manifest_digest"] != supplied_digest
    assert "manifest_digest:mismatch" in receipt["blockers"]


@pytest.mark.parametrize(
    ("authority", "normalized_authority"),
    [("DA3", "da3"), ("SAM 3.1", "sam_3.1"), ("3DGS", "3dgs")],
)
def test_learned_appearance_or_segmentation_cannot_be_metric_scale_authority(
    authority: str,
    normalized_authority: str,
) -> None:
    value = _manifest()
    value["coordinate_frames"][0]["metric_scale_authority"]["kind"] = authority
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "coordinate_frames[0].metric_scale_authority.kind:forbidden:"
        + normalized_authority
        in receipt["blockers"]
    )


def test_calibration_and_test_trajectory_overlap_blocks_admission() -> None:
    value = _manifest()
    value["splits"]["test_trajectory_ids"] = ["calibration-view-001"]
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "splits:calibration_test_overlap:calibration-view-001" in receipt["blockers"]
    )


def test_split_ids_must_resolve_to_digest_bound_observation_artifacts() -> None:
    value = _manifest()
    value["splits"]["test_trajectory_ids"] = ["unbound-test-view"]
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "splits.test_trajectory_ids:unknown:unbound-test-view" in receipt["blockers"]


def test_metric_geometry_is_required_separately_from_splat_and_collision() -> None:
    value = _manifest()
    value["representations"].pop("metric_geometry")
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "representations.metric_geometry.kind:invalid" in receipt["blockers"]


@pytest.mark.parametrize("availability", ["paper_only", "proprietary_unverified"])
def test_unreleased_code_dependency_blocks_admission(availability: str) -> None:
    value = _manifest()
    value["code_dependencies"][0]["availability"] = availability
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        f"code_dependencies[0].availability:not_released:{availability}"
        in receipt["blockers"]
    )


@pytest.mark.parametrize(
    ("dependency_id", "expected_binding"),
    [
        (
            "infusion",
            "background_completion_primary_adapter:infusion",
        ),
        (
            "aurafusion360",
            "background_completion_quality_challenger:aurafusion360",
        ),
    ],
)
def test_suite_requires_primary_and_challenger_bindings(
    dependency_id: str, expected_binding: str
) -> None:
    value = _manifest()
    dependency = next(
        row
        for row in value["code_dependencies"]
        if row["dependency_id"] == dependency_id
    )
    dependency["upstream_project_id"] = "some-other-method"
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        f"code_dependencies:missing_required_released_binding:{expected_binding}"
        in receipt["blockers"]
    )


def test_claim_elevation_blocks_admission() -> None:
    value = _manifest()
    value["claim_boundaries"]["digital_twin"] = True
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert "claim_boundaries.digital_twin:must_be_false" in receipt["blockers"]


def test_source_world_inverse_must_round_trip() -> None:
    value = _manifest()
    value["coordinate_frames"][0]["source_to_world"][0][3] = 0.25
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "coordinate_frames[0]:source_world_inverse_round_trip_failed"
        in receipt["blockers"]
    )
    assert "coordinate_frames:round_trip_not_verified" in receipt["blockers"]


def test_declared_up_axis_must_map_to_world_positive_z() -> None:
    value = _manifest()
    value["coordinate_frames"][0]["up_axis"] = "+Y"
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "coordinate_frames[0].source_to_world:up_axis_mapping_mismatch"
        in receipt["blockers"]
    )


def test_task_object_requires_exact_simready_physics_metadata() -> None:
    value = _manifest()
    task_object = value["representations"]["task_objects"][0]
    task_object["asset_format"] = "visual_usd_only"
    task_object["physics_properties"].pop("inertia_tensor_kg_m2")
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "representations.task_objects[0].asset_format:must_be:simready_usd"
        in receipt["blockers"]
    )
    assert (
        "representations.task_objects[0].physics_properties.inertia_tensor_kg_m2:invalid"
        in receipt["blockers"]
    )


def test_task_object_must_bind_one_exact_usd_from_simready_source() -> None:
    value = _manifest()
    task_object = value["representations"]["task_objects"][0]
    task_object["usd_artifact_id"] = "scene-splat"
    task_object["visual_artifact_ids"] = ["scene-splat"]
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "representations.task_objects[0].usd_artifact_id:source_or_role_mismatch"
        in receipt["blockers"]
    )
    assert (
        "representations.task_objects[0].visual_artifact_ids:source_or_role_mismatch:scene-splat"
        in receipt["blockers"]
    )


def test_task_object_pose_contact_and_reset_authority_fail_closed() -> None:
    value = _manifest()
    task_object = value["representations"]["task_objects"][0]
    task_object["replacement_pose_world"][0][0] = 2.0
    task_object["support_normal_world"] = [0.0, 0.0, 2.0]
    task_object["reset_state_id"] = ""
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "representations.task_objects[0].replacement_pose_world:not_metric_isometry"
        in receipt["blockers"]
    )
    assert (
        "representations.task_objects[0].support_normal_world:not_unit_vector"
        in receipt["blockers"]
    )
    assert "representations.task_objects[0].reset_state_id:invalid" in receipt["blockers"]


def test_task_object_friction_and_inertia_fail_closed() -> None:
    value = _manifest()
    physics = value["representations"]["task_objects"][0]["physics_properties"]
    physics["dynamic_friction"] = 0.8
    physics["inertia_tensor_kg_m2"] = [
        [0.0002, 0.0003, 0.0],
        [0.0003, 0.0002, 0.0],
        [0.0, 0.0, 0.0002],
    ]
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "representations.task_objects[0].physics_properties.dynamic_friction:exceeds_static_friction"
        in receipt["blockers"]
    )
    assert (
        "representations.task_objects[0].physics_properties.inertia_tensor_kg_m2:not_physically_valid"
        in receipt["blockers"]
    )


def test_inertia_principal_moments_must_be_physically_realizable() -> None:
    value = _manifest()
    value["representations"]["task_objects"][0]["physics_properties"][
        "inertia_tensor_kg_m2"
    ] = [
        [0.001, 0.0, 0.0],
        [0.0, 0.001, 0.0],
        [0.0, 0.0, 0.003],
    ]
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "representations.task_objects[0].physics_properties.inertia_tensor_kg_m2:not_physically_valid"
        in receipt["blockers"]
    )


def _add_scene_artifact(manifest: dict, artifact: dict) -> None:
    manifest["sources"][0]["artifacts"].append(artifact)


def test_external_validation_depth_cannot_leak_into_method_inputs() -> None:
    value = _manifest()
    _add_scene_artifact(
        value,
        _artifact(
            "laser-depth-oracle",
            "validation_depth_oracle",
            digest_character="7",
            size_bytes=210,
        ),
    )
    value["observation_bundle"]["validation_oracle"] = {
        "availability": "available",
        "depth_artifact_id": "laser-depth-oracle",
        "geometry_artifact_id": None,
        "authority": "source_sensor_or_laser",
        "usage": "evaluation_only",
        "method_access": False,
        "independent_of_method_inputs": True,
    }
    value["observation_bundle"]["method_profiles"][0][
        "input_artifact_ids"
    ].append("laser-depth-oracle")
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "observation_bundle.validation_oracle:leaked_to_method_input:laser-depth-oracle"
        in receipt["blockers"]
    )


def test_method_profile_cannot_read_the_external_oracle_by_declaration() -> None:
    value = _manifest()
    value["observation_bundle"]["method_profiles"][0][
        "external_validation_oracle_access"
    ] = True
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "observation_bundle.method_profiles[0].external_validation_oracle_access:must_be:false"
        in receipt["blockers"]
    )


def test_method_runner_must_mount_only_the_declared_input_allowlist() -> None:
    value = _manifest()
    value["observation_bundle"]["method_profiles"][0]["input_mount_policy"] = (
        "whole_suite"
    )
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "observation_bundle.method_profiles[0].input_mount_policy:must_be:allowlist_only"
        in receipt["blockers"]
    )


def test_infusion_profile_requires_its_declared_splat_depth_input() -> None:
    value = _manifest()
    profile = value["observation_bundle"]["method_profiles"][0]
    profile["input_artifact_ids"].remove("method-splat-depth-001")
    profile["input_modalities"].remove("splat_rendered_inverse_depth")
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "observation_bundle.method_profiles[0].input_modalities:profile_mismatch"
        in receipt["blockers"]
    )
    assert (
        "observation_bundle.method_profiles[0].input_artifact_ids:missing_role:method_input_splat_depth"
        in receipt["blockers"]
    )


def test_controlled_truth_can_be_bound_but_withheld_before_edit() -> None:
    value = _manifest()
    _add_scene_artifact(
        value,
        _artifact(
            "clean-counter-rgb",
            "clean_background_rgb_truth",
            digest_character="8",
            size_bytes=211,
        ),
    )
    value["observation_bundle"]["truth_contract"] = {
        "availability": "available_withheld",
        "clean_background_artifact_ids": ["clean-counter-rgb"],
        "method_access": False,
        "edit_result_digest": None,
        "edit_seal_digest": None,
        "truth_release_join_digest": None,
    }
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "component_admitted"


def test_withheld_truth_cannot_claim_release_before_edit_seal() -> None:
    value = _manifest()
    _add_scene_artifact(
        value,
        _artifact(
            "clean-counter-rgb",
            "clean_background_rgb_truth",
            digest_character="8",
            size_bytes=211,
        ),
    )
    truth = value["observation_bundle"]["truth_contract"]
    truth["availability"] = "available_withheld"
    truth["clean_background_artifact_ids"] = ["clean-counter-rgb"]
    truth["edit_result_digest"] = "sha256:" + "1" * 64
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "observation_bundle.truth_contract:withheld_but_release_bound"
        in receipt["blockers"]
    )


def test_truth_release_requires_a_complete_distinct_digest_chain() -> None:
    value = _manifest()
    _add_scene_artifact(
        value,
        _artifact(
            "clean-counter-rgb",
            "clean_background_rgb_truth",
            digest_character="8",
            size_bytes=211,
        ),
    )
    truth = value["observation_bundle"]["truth_contract"]
    truth["availability"] = "released_after_edit_seal"
    truth["clean_background_artifact_ids"] = ["clean-counter-rgb"]
    truth["edit_result_digest"] = "sha256:" + "1" * 64
    truth["edit_seal_digest"] = "sha256:" + "2" * 64
    truth["truth_release_join_digest"] = None
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "observation_bundle.truth_contract:release_chain_incomplete"
        in receipt["blockers"]
    )

    truth["truth_release_join_digest"] = "sha256:" + "3" * 64
    _redigest(value)
    assert _receipt(value)["status"] == "component_admitted"


def test_interiorgs_metric_frame_cannot_be_promoted_to_surface_authority() -> None:
    value = _manifest()
    _add_scene_artifact(
        value,
        _artifact(
            "invented-local-surface",
            "metric_geometry",
            digest_character="9",
            size_bytes=212,
        ),
    )
    value["representations"]["metric_geometry"] = {
        "kind": "authored_metric_geometry",
        "usage": "measurement_authority",
        "artifact_ids": ["invented-local-surface"],
        "measurement_authority": True,
    }
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "blocked"
    assert (
        "representations.metric_geometry:interiorgs_cannot_claim_local_measurement_authority"
        in receipt["blockers"]
    )


def test_scannetpp_profile_can_bind_admitted_laser_surface_authority() -> None:
    value = _manifest()
    value["sources"][0]["upstream_project_id"] = "ScanNet++"
    value["sources"][0]["source_kind"] = "real_metric_scene"
    _add_scene_artifact(
        value,
        _artifact(
            "scannet-laser-surface",
            "metric_geometry",
            digest_character="a",
            size_bytes=213,
        ),
    )
    value["representations"]["metric_geometry"] = {
        "kind": "laser_mesh",
        "usage": "measurement_authority",
        "artifact_ids": ["scannet-laser-surface"],
        "measurement_authority": True,
    }
    value["observation_bundle"]["validation_oracle"] = {
        "availability": "available",
        "depth_artifact_id": None,
        "geometry_artifact_id": "scannet-laser-surface",
        "authority": "source_sensor_or_laser",
        "usage": "evaluation_only",
        "method_access": False,
        "independent_of_method_inputs": True,
    }
    _redigest(value)

    receipt = _receipt(value)

    assert receipt["status"] == "component_admitted"
