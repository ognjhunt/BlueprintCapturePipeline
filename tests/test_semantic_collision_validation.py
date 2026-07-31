"""Tests for independent semantic-OBB/collision consistency validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.scene_placement.semantic_collision_validation import (
    COLLISION_SCENE_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    VALIDATION_METHOD,
    validate_semantic_boxes_against_collision,
)
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import canonical_json_digest
from blueprint_pipeline.semantic_collision_validation_stage import (
    main as stage_main,
    run_semantic_collision_validation_stage,
)


_CAPTURE = "sha256:" + "a" * 64
_RECONSTRUCTION = "sha256:" + "b" * 64
_SPLAT = "sha256:" + "c" * 64
_RUNTIME = "sha256:" + "d" * 64
_IDENTITY = "sha256:" + "e" * 64


def _sign(payload: dict, field: str) -> None:
    payload.pop(field, None)
    payload[field] = canonical_json_digest(payload)


def _fixture() -> tuple[dict, dict, dict]:
    obb = {
        "schema_version": "semantic_oriented_box_result.v1",
        "status": "completed",
        "bindings": {
            "capture_digest": _CAPTURE,
            "reconstruction_digest": _RECONSTRUCTION,
            "analysis_splat_digest": _SPLAT,
        },
        "world": {"up_axis": "Z", "units": "meters", "scale_verified": True},
        "objects": [
            {
                "track_id": "track_tote_01",
                "label": "tote",
                "status": "qualified_metric_obb_candidate",
                "metric_obb_candidate_ready": True,
                "collision_ready": False,
                "physics_ready": False,
                "center_world_m": [0.0, 0.0, 1.0],
                "dimensions_m": [2.0, 1.0, 1.0],
                "yaw_rad": 0.0,
                "coordinate_frame": "analysis_splat_z_up_meters",
                "corners_world_m": [
                    [-1.0, -0.5, 0.5],
                    [1.0, -0.5, 0.5],
                    [1.0, 0.5, 0.5],
                    [-1.0, 0.5, 0.5],
                    [-1.0, -0.5, 1.5],
                    [1.0, -0.5, 1.5],
                    [1.0, 0.5, 1.5],
                    [-1.0, 0.5, 1.5],
                ],
            }
        ],
        "collision_ready": False,
        "physics_ready": False,
        "generated_regions_can_upgrade_claims": False,
    }
    _sign(obb, "result_digest")

    method_profile = {
        "method_id": "blueprint.hermetic_collision_scene_fixture",
        "method_version": "1.0.0",
        "runtime_digest": _RUNTIME,
        "producer_identity": "local:collision-scene-builder",
        "validator_identity": "local:independent-collision-validator",
        "deterministic": True,
        "source_capture_bound": True,
        "metric_transform_verified": True,
        "qualification_status": "qualified",
        "independent_from_semantic_geometry": True,
        "qualified_checks": [
            "coverage",
            "generated_region_intersection",
            "non_target_penetration",
            "support_contact",
            "target_volume_overlap",
            "verified_free_space_conflict",
        ],
    }
    _sign(method_profile, "method_profile_digest")
    scene = {
        "schema_version": COLLISION_SCENE_SCHEMA_VERSION,
        "source_capture_digest": _CAPTURE,
        "reconstruction_digest": _RECONSTRUCTION,
        "coordinate_frame": "analysis_splat_z_up_meters",
        "up_axis": "Z",
        "units": "meters",
        "scale_status": "metric_verified",
        "generated_geometry": False,
        "method_profile": method_profile,
        "validation": {
            "status": "qualified",
            "independent_validation": True,
            "validator_identity": "local:independent-collision-validator",
            "coverage": 0.95,
            "maximum_spatial_uncertainty_m": 0.005,
        },
        "target_bindings": [
            {
                "track_id": "track_tote_01",
                "object_id": "collision-tote-01",
                "primitive_id": "target-volume",
                "support_surface_id": "table-top",
                "identity_verified": True,
                "identity_evidence_digest": _IDENTITY,
            }
        ],
        "occupied_primitives": [
            {
                "primitive_id": "target-volume",
                "object_id": "collision-tote-01",
                "minimum_world_m": [-1.0, -0.5, 0.5],
                "maximum_world_m": [1.0, 0.5, 1.5],
                "source_class": "observed",
            },
            {
                "primitive_id": "far-obstacle",
                "object_id": "wall-01",
                "minimum_world_m": [5.0, 5.0, 0.0],
                "maximum_world_m": [6.0, 6.0, 2.0],
                "source_class": "verified_asset",
            },
        ],
        "verified_free_space_primitives": [],
        "generated_regions": [],
        "coverage_volumes": [
            {
                "region_id": "coverage-01",
                "minimum_world_m": [-2.0, -2.0, 0.0],
                "maximum_world_m": [2.0, 2.0, 2.0],
                "source_class": "observed",
            }
        ],
        "support_surfaces": [
            {
                "surface_id": "table-top",
                "z_world_m": 0.5,
                "polygon_xy_world_m": [
                    [-2.0, -2.0],
                    [2.0, -2.0],
                    [2.0, 2.0],
                    [-2.0, 2.0],
                ],
                "source_class": "observed",
            }
        ],
    }
    _sign(scene, "collision_scene_digest")
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "validation_method": VALIDATION_METHOD,
        "bindings": {
            "capture_digest": _CAPTURE,
            "reconstruction_digest": _RECONSTRUCTION,
            "analysis_splat_digest": _SPLAT,
            "semantic_oriented_box_result_digest": obb["result_digest"],
            "collision_scene_digest": scene["collision_scene_digest"],
            "collision_method_profile_digest": method_profile["method_profile_digest"],
        },
        "world": {
            "up_axis": "Z",
            "units": "meters",
            "scale_verified": True,
            "coordinate_frame": "analysis_splat_z_up_meters",
        },
        "qualification": {
            "min_scene_coverage": 0.8,
            "max_spatial_uncertainty_m": 0.02,
            "max_support_gap_m": 0.01,
            "max_support_penetration_m": 0.01,
            "min_support_overlap_fraction": 0.8,
            "min_target_iou": 0.8,
            "max_non_target_penetration_fraction": 0.01,
            "max_free_space_conflict_fraction": 0.01,
            "require_full_corner_coverage": True,
        },
    }
    return request, obb, scene


def _resign_obb(request: dict, obb: dict) -> None:
    _sign(obb, "result_digest")
    request["bindings"]["semantic_oriented_box_result_digest"] = obb["result_digest"]


def _resign_scene(request: dict, scene: dict) -> None:
    _sign(scene, "collision_scene_digest")
    request["bindings"]["collision_scene_digest"] = scene["collision_scene_digest"]


def _run(request: dict, obb: dict, scene: dict) -> dict:
    return validate_semantic_boxes_against_collision(
        request,
        oriented_box_result=obb,
        collision_scene=scene,
    )


def test_passes_independent_collision_consistency_without_upgrading_authority() -> None:
    request, obb, scene = _fixture()
    result = _run(request, obb, scene)

    assert result["status"] == "completed"
    assert result["collision_consistency_candidate_ready"] is True
    assert result["collision_ready"] is False
    assert result["physics_ready"] is False
    row = result["objects"][0]
    assert row["status"] == "independent_collision_consistency_candidate"
    assert row["metrics"]["target_collision_iou"] == pytest.approx(1.0)
    assert row["metrics"]["support_signed_gap_m"] == pytest.approx(0.0)
    assert row["metrics"]["support_horizontal_overlap_fraction"] == pytest.approx(1.0)
    assert row["metrics"]["covered_corner_fraction"] == pytest.approx(1.0)
    assert row["provenance"]["physical_robot_run_initiated"] is False


def test_provider_cannot_self_qualify_collision_scene() -> None:
    request, obb, scene = _fixture()
    profile = scene["method_profile"]
    profile["validator_identity"] = profile["producer_identity"]
    _sign(profile, "method_profile_digest")
    scene["validation"]["validator_identity"] = profile["validator_identity"]
    request["bindings"]["collision_method_profile_digest"] = profile["method_profile_digest"]
    _resign_scene(request, scene)
    result = _run(request, obb, scene)
    assert result["status"] == "blocked"
    assert "collision_provider_cannot_self_qualify" in result["blockers"]


def test_method_must_be_qualified_for_each_collision_consistency_check() -> None:
    request, obb, scene = _fixture()
    profile = scene["method_profile"]
    profile["qualified_checks"].remove("verified_free_space_conflict")
    _sign(profile, "method_profile_digest")
    request["bindings"]["collision_method_profile_digest"] = profile["method_profile_digest"]
    _resign_scene(request, scene)
    result = _run(request, obb, scene)
    assert "collision_method_check_not_qualified:verified_free_space_conflict" in result["blockers"]


def test_tampered_or_stale_inputs_are_blocked() -> None:
    request, obb, scene = _fixture()
    scene["validation"]["coverage"] = 1.0
    result = _run(request, obb, scene)
    assert "collision_scene_digest_invalid" in result["blockers"]

    request, obb, scene = _fixture()
    obb["objects"][0]["label"] = "cabinet"
    result = _run(request, obb, scene)
    assert "semantic_oriented_box_result_digest_invalid" in result["blockers"]

    request, obb, scene = _fixture()
    scene["source_capture_digest"] = "sha256:" + "f" * 64
    _resign_scene(request, scene)
    result = _run(request, obb, scene)
    assert "collision_scene_source_capture_mismatch" in result["blockers"]


def test_box_dimensions_corners_and_center_must_be_internally_consistent() -> None:
    request, obb, scene = _fixture()
    obb["objects"][0]["dimensions_m"][0] = 3.0
    _resign_obb(request, obb)
    result = _run(request, obb, scene)
    assert "semantic_object_box_geometry_invalid:track_tote_01" in result["blockers"]

    request, obb, scene = _fixture()
    obb["objects"][0]["center_world_m"][0] = 0.25
    _resign_obb(request, obb)
    result = _run(request, obb, scene)
    assert "semantic_object_box_geometry_invalid:track_tote_01" in result["blockers"]


def test_rotated_obb_uses_polygon_volume_intersection_not_axis_aligned_envelope() -> None:
    request, obb, scene = _fixture()
    root_two = 2.0**0.5
    obj = obb["objects"][0]
    obj["dimensions_m"] = [root_two, root_two, 1.0]
    obj["yaw_rad"] = math.pi / 4.0
    obj["corners_world_m"] = [
        [0.0, -1.0, 0.5],
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [-1.0, 0.0, 0.5],
        [0.0, -1.0, 1.5],
        [1.0, 0.0, 1.5],
        [0.0, 1.0, 1.5],
        [-1.0, 0.0, 1.5],
    ]
    _resign_obb(request, obb)
    scene["occupied_primitives"][0]["minimum_world_m"] = [-0.5, -0.5, 0.5]
    scene["occupied_primitives"][0]["maximum_world_m"] = [0.5, 0.5, 1.5]
    _resign_scene(request, scene)
    request["qualification"]["min_target_iou"] = 0.49
    result = _run(request, obb, scene)
    assert result["status"] == "completed"
    assert result["objects"][0]["metrics"]["target_collision_iou"] == pytest.approx(0.5)


def test_support_gap_and_penetration_return_precise_experiments() -> None:
    request, obb, scene = _fixture()
    scene["support_surfaces"][0]["z_world_m"] = 0.45
    _resign_scene(request, scene)
    gap = _run(request, obb, scene)
    assert gap["status"] == "abstained"
    assert "support_contact_gap_too_large" in gap["objects"][0]["abstention_reasons"]
    assert gap["objects"][0]["next_experiment"] == (
        "capture_the_object_support_contact_and_support_surface"
    )

    request, obb, scene = _fixture()
    scene["support_surfaces"][0]["z_world_m"] = 0.55
    _resign_scene(request, scene)
    penetration = _run(request, obb, scene)
    assert (
        "support_surface_penetration_too_large" in penetration["objects"][0]["abstention_reasons"]
    )
    assert penetration["objects"][0]["next_experiment"] == (
        "verify_the_site_to_object_transform_and_support_plane_height"
    )


def test_non_target_penetration_and_free_space_conflict_abstain() -> None:
    request, obb, scene = _fixture()
    scene["occupied_primitives"].append(
        {
            "primitive_id": "adjacent-box",
            "object_id": "adjacent-01",
            "minimum_world_m": [0.0, -0.5, 0.5],
            "maximum_world_m": [1.5, 0.5, 1.5],
            "source_class": "observed",
        }
    )
    _resign_scene(request, scene)
    result = _run(request, obb, scene)
    row = result["objects"][0]
    assert "non_target_penetration_exceeds_threshold" in row["abstention_reasons"]
    assert row["metrics"]["maximum_non_target_penetration_fraction"] == pytest.approx(0.5)

    request, obb, scene = _fixture()
    scene["verified_free_space_primitives"] = [
        {
            "primitive_id": "observed-empty-right-half",
            "minimum_world_m": [0.0, -0.5, 0.5],
            "maximum_world_m": [1.0, 0.5, 1.5],
            "source_class": "observed",
        }
    ]
    _resign_scene(request, scene)
    result = _run(request, obb, scene)
    row = result["objects"][0]
    assert "verified_free_space_conflict_exceeds_threshold" in row["abstention_reasons"]
    assert row["metrics"]["maximum_verified_free_space_conflict_fraction"] == pytest.approx(0.5)


def test_generated_region_and_incomplete_coverage_cannot_pass() -> None:
    request, obb, scene = _fixture()
    scene["generated_regions"] = [
        {
            "region_id": "hidden-rear",
            "minimum_world_m": [0.0, -0.5, 0.5],
            "maximum_world_m": [1.0, 0.5, 1.5],
            "source_class": "generated",
        }
    ]
    scene["coverage_volumes"][0]["maximum_world_m"] = [0.0, 2.0, 2.0]
    _resign_scene(request, scene)
    result = _run(request, obb, scene)
    row = result["objects"][0]
    assert "generated_region_intersection" in row["abstention_reasons"]
    assert "collision_coverage_incomplete" in row["abstention_reasons"]
    assert row["generated_region_intersection_ids"] == ["hidden-rear"]
    assert row["next_experiment"] == "recapture_the_generated_or_unobserved_object_region"


def test_target_identity_binding_and_collision_overlap_are_required() -> None:
    request, obb, scene = _fixture()
    scene["target_bindings"] = []
    _resign_scene(request, scene)
    missing = _run(request, obb, scene)
    assert "collision_target_binding_missing" in missing["objects"][0]["abstention_reasons"]

    request, obb, scene = _fixture()
    scene["occupied_primitives"][0]["minimum_world_m"] = [3.0, 3.0, 0.5]
    scene["occupied_primitives"][0]["maximum_world_m"] = [5.0, 4.0, 1.5]
    _resign_scene(request, scene)
    misaligned = _run(request, obb, scene)
    assert "target_collision_iou_below_threshold" in misaligned["objects"][0]["abstention_reasons"]


def test_upstream_box_abstention_is_preserved_as_abstention() -> None:
    request, obb, scene = _fixture()
    obj = obb["objects"][0]
    obj["status"] = "abstained"
    obj["metric_obb_candidate_ready"] = False
    for field in (
        "center_world_m",
        "dimensions_m",
        "yaw_rad",
        "coordinate_frame",
        "corners_world_m",
    ):
        obj.pop(field)
    _resign_obb(request, obb)
    result = _run(request, obb, scene)
    assert result["status"] == "abstained"
    assert result["blockers"] == []
    assert result["objects"][0]["abstention_reasons"] == ["semantic_oriented_box_not_qualified"]
    assert result["objects"][0]["next_experiment"] == (
        "complete_semantic_support_and_metric_box_qualification_first"
    )


def test_nonconvex_support_surface_and_generated_collision_source_fail_closed() -> None:
    request, obb, scene = _fixture()
    scene["support_surfaces"][0]["polygon_xy_world_m"] = [
        [-2.0, -2.0],
        [2.0, -2.0],
        [0.0, 0.0],
        [2.0, 2.0],
        [-2.0, 2.0],
    ]
    _resign_scene(request, scene)
    result = _run(request, obb, scene)
    assert "support_surface_geometry_invalid:table-top" in result["blockers"]

    request, obb, scene = _fixture()
    scene["occupied_primitives"][0]["source_class"] = "generated"
    _resign_scene(request, scene)
    result = _run(request, obb, scene)
    assert "occupied_primitive_source_class_invalid:target-volume" in result["blockers"]


def _write_stage_inputs(root: Path, request: dict, obb: dict, scene: dict) -> dict[str, Path]:
    payloads = {"oriented_box_result": obb, "collision_scene": scene}
    request["input_artifacts"] = {}
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = root / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths[name] = path
        request["input_artifacts"][name] = {
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    paths["request"] = root / "request.json"
    paths["request"].write_text(json.dumps(request, indent=2), encoding="utf-8")
    paths["output"] = root / "result.json"
    return paths


def test_file_stage_verifies_exact_inputs_and_cli(tmp_path: Path) -> None:
    request, obb, scene = _fixture()
    paths = _write_stage_inputs(tmp_path, request, obb, scene)
    result = run_semantic_collision_validation_stage(
        request_path=paths["request"],
        oriented_box_result_path=paths["oriented_box_result"],
        collision_scene_path=paths["collision_scene"],
        output_path=paths["output"],
    )
    assert result["status"] == "completed"
    assert result["transport_profile"] == "bounded_canonical_json_baseline.v1"
    assert set(result["stage_input_artifacts"]) == {
        "request",
        "oriented_box_result",
        "collision_scene",
    }
    assert (
        stage_main(
            [
                "--request",
                str(paths["request"]),
                "--oriented-box-result",
                str(paths["oriented_box_result"]),
                "--collision-scene",
                str(paths["collision_scene"]),
                "--output",
                str(tmp_path / "cli-result.json"),
            ]
        )
        == 0
    )


def test_file_stage_blocks_tampering_symlinks_and_input_overwrite(tmp_path: Path) -> None:
    request, obb, scene = _fixture()
    paths = _write_stage_inputs(tmp_path, request, obb, scene)
    paths["collision_scene"].write_text("{}", encoding="utf-8")
    result = run_semantic_collision_validation_stage(
        request_path=paths["request"],
        oriented_box_result_path=paths["oriented_box_result"],
        collision_scene_path=paths["collision_scene"],
        output_path=paths["output"],
    )
    assert result["status"] == "blocked"
    assert "input_artifact_sha256_mismatch:collision_scene" in result["blockers"]

    request, obb, scene = _fixture()
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    paths = _write_stage_inputs(linked_root, request, obb, scene)
    symlink = linked_root / "collision-link.json"
    symlink.symlink_to(paths["collision_scene"])
    result = run_semantic_collision_validation_stage(
        request_path=paths["request"],
        oriented_box_result_path=paths["oriented_box_result"],
        collision_scene_path=symlink,
        output_path=paths["output"],
    )
    assert "input_symlink_forbidden:collision_scene" in result["blockers"]

    with pytest.raises(ValueError, match="output_path_must_not_overwrite_an_input"):
        run_semantic_collision_validation_stage(
            request_path=paths["request"],
            oriented_box_result_path=paths["oriented_box_result"],
            collision_scene_path=paths["collision_scene"],
            output_path=paths["collision_scene"],
        )


def test_input_order_does_not_change_object_metrics() -> None:
    request, obb, scene = _fixture()
    first = _run(request, obb, scene)
    request_2, obb_2, scene_2 = _fixture()
    scene_2["occupied_primitives"] = list(reversed(copy.deepcopy(scene_2["occupied_primitives"])))
    _resign_scene(request_2, scene_2)
    second = _run(request_2, obb_2, scene_2)
    assert first["objects"][0]["metrics"] == second["objects"][0]["metrics"]
    assert first["objects"][0]["abstention_reasons"] == second["objects"][0]["abstention_reasons"]
