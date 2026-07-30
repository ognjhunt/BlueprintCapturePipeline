"""Tests for source-bound robust Z-up semantic object-box fitting."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import canonical_json_digest
from blueprint_pipeline.scene_placement.semantic_oriented_box import (
    FIT_METHOD,
    REQUEST_SCHEMA_VERSION,
    fit_semantic_oriented_boxes,
)
from blueprint_pipeline.semantic_oriented_box_stage import (
    main as stage_main,
    run_semantic_oriented_box_stage,
)


_CAPTURE = "sha256:" + "a" * 64
_RECONSTRUCTION = "sha256:" + "b" * 64
_SPLAT = "sha256:" + "c" * 64


def _rotate(local_x: float, local_y: float, *, yaw: float, center: tuple[float, float]) -> list[float]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        center[0] + local_x * cosine - local_y * sine,
        center[1] + local_x * sine + local_y * cosine,
    ]


def _fixture(
    *,
    point_source: str = "observed_depth",
    include_outlier: bool = True,
) -> tuple[dict, dict, list[dict], list[dict]]:
    yaw = math.radians(30.0)
    xy = [
        (-1.0, -0.5),
        (0.0, -0.5),
        (1.0, -0.5),
        (1.0, 0.0),
        (1.0, 0.5),
        (0.0, 0.5),
        (-1.0, 0.5),
        (-1.0, 0.0),
    ]
    support_points: list[dict] = []
    for z in (0.5, 1.5):
        for local_x, local_y in xy:
            gaussian_id = len(support_points)
            world_xy = _rotate(local_x, local_y, yaw=yaw, center=(3.0, 4.0))
            support_points.append(
                {
                    "point_id": f"point_{gaussian_id:02d}",
                    "gaussian_id": gaussian_id,
                    "point_source": point_source,
                    "point_world_m": [world_xy[0], world_xy[1], z],
                }
            )
    if include_outlier:
        support_points.append(
            {
                "point_id": "floater",
                "gaussian_id": len(support_points),
                "point_source": point_source,
                "point_world_m": [100.0, -100.0, 50.0],
            }
        )
    mapping = [
        {"gaussian_id": index, "source_index": 100 + index, "source_class": "observed"}
        for index in range(len(support_points))
    ]
    semantic = {
        "schema_version": "semantic_gaussian_lifting_result.v1",
        "status": "completed",
        "bindings": {
            "capture_digest": _CAPTURE,
            "reconstruction_digest": _RECONSTRUCTION,
            "analysis_splat_digest": _SPLAT,
            "gaussian_mapping_digest": canonical_json_digest(mapping),
        },
        "world": {"up_axis": "Z", "units": "meters", "scale_verified": True},
        "tracks": [
            {
                "track_id": "track_tote_01",
                "label": "tote",
                "status": "qualified_semantic_support_candidate",
                "selected_gaussian_ids": list(range(len(mapping))),
                "supporting_view_ids": ["view_a", "view_b", "view_c"],
                "angular_diversity_degrees": 75.0,
            }
        ],
        "generated_regions_can_upgrade_claims": False,
    }
    semantic["result_digest"] = canonical_json_digest(semantic)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "fit_method": FIT_METHOD,
        "bindings": {
            "capture_digest": _CAPTURE,
            "reconstruction_digest": _RECONSTRUCTION,
            "analysis_splat_digest": _SPLAT,
            "gaussian_mapping_digest": canonical_json_digest(mapping),
            "semantic_lifting_result_digest": semantic["result_digest"],
            "support_points_digest": canonical_json_digest(support_points),
        },
        "gaussian_count": len(mapping),
        "world": {"up_axis": "Z", "units": "meters", "scale_verified": True},
        "support_method_profile": {
            "method_id": "blueprint.hermetic_observed_support_fixture",
            "method_version": "1.0.0",
            "runtime_digest": "sha256:" + "e" * 64,
            "deterministic": True,
            "source_capture_bound": True,
            "metric_transform_verified": True,
        },
        "qualification": {
            "min_support_points": 8,
            "min_distinct_gaussians": 8,
            "outlier_mad_multiplier": 5.0,
            "vertical_trim_fraction": 0.0,
            "min_horizontal_extent_m": 0.05,
            "min_vertical_extent_m": 0.05,
            "max_dimension_m": 10.0,
            "min_inlier_fraction": 0.8,
        },
    }
    return request, semantic, mapping, support_points


def _fit(request: dict, semantic: dict, mapping: list[dict], points: list[dict]) -> dict:
    return fit_semantic_oriented_boxes(
        request,
        semantic_result=semantic,
        gaussian_mapping=mapping,
        support_points=points,
    )


def _resign_semantic(request: dict, semantic: dict) -> None:
    semantic.pop("result_digest", None)
    semantic["result_digest"] = canonical_json_digest(semantic)
    request["bindings"]["semantic_lifting_result_digest"] = semantic["result_digest"]


def _write_stage_inputs(
    root: Path,
    request: dict,
    semantic: dict,
    mapping: list[dict],
    points: list[dict],
) -> dict[str, Path]:
    payloads = {
        "semantic_result": semantic,
        "gaussian_mapping": mapping,
        "support_points": points,
    }
    paths: dict[str, Path] = {}
    request["input_artifacts"] = {}
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


def test_fits_metric_z_up_eight_corner_box_and_rejects_floater() -> None:
    request, semantic, mapping, points = _fixture()
    result = _fit(request, semantic, mapping, points)

    assert result["status"] == "completed"
    assert result["metric_obb_candidate_ready"] is True
    assert result["collision_ready"] is False
    assert result["physics_ready"] is False
    box = result["objects"][0]
    assert box["status"] == "qualified_metric_obb_candidate"
    assert box["claim_ceiling"] == "metric_obb_candidate_from_observed_surface_support"
    assert box["center_world_m"] == pytest.approx([3.0, 4.0, 1.0])
    assert box["dimensions_m"] == pytest.approx([2.0, 1.0, 1.0])
    assert box["yaw_rad"] == pytest.approx(math.radians(30.0))
    assert len(box["corners_world_m"]) == 8
    assert box["removed_outlier_point_ids"] == ["floater"]
    assert result["generated_regions_can_upgrade_claims"] is False


def test_gaussian_center_fallback_has_explicit_weaker_claim_ceiling() -> None:
    request, semantic, mapping, points = _fixture(point_source="gaussian_center")
    result = _fit(request, semantic, mapping, points)
    assert result["objects"][0]["claim_ceiling"] == (
        "approximate_metric_obb_candidate_from_gaussian_centers"
    )
    assert result["collision_ready"] is False


def test_input_order_does_not_change_fitted_object() -> None:
    request, semantic, mapping, points = _fixture()
    original = _fit(request, semantic, mapping, points)
    reversed_mapping = list(reversed(copy.deepcopy(mapping)))
    reversed_points = list(reversed(copy.deepcopy(points)))
    request["bindings"]["gaussian_mapping_digest"] = canonical_json_digest(reversed_mapping)
    semantic["bindings"]["gaussian_mapping_digest"] = request["bindings"][
        "gaussian_mapping_digest"
    ]
    _resign_semantic(request, semantic)
    request["bindings"]["support_points_digest"] = canonical_json_digest(reversed_points)
    replay = _fit(request, semantic, reversed_mapping, reversed_points)
    replay_box = dict(replay["objects"][0])
    original_box = dict(original["objects"][0])
    replay_box.pop("semantic_lifting_result_digest")
    original_box.pop("semantic_lifting_result_digest")
    assert replay_box == original_box


def test_tampered_semantic_result_is_blocked() -> None:
    request, semantic, mapping, points = _fixture()
    semantic["tracks"][0]["label"] = "cabinet"
    result = _fit(request, semantic, mapping, points)
    assert result["status"] == "blocked"
    assert "semantic_lifting_result_digest_invalid" in result["blockers"]


def test_unverified_scale_is_blocked_before_metric_fit() -> None:
    request, semantic, mapping, points = _fixture()
    request["world"]["scale_verified"] = False
    semantic["world"]["scale_verified"] = False
    _resign_semantic(request, semantic)
    result = _fit(request, semantic, mapping, points)
    assert result["status"] == "blocked"
    assert "verified_metric_scale_required" in result["blockers"]


def test_wrong_splat_binding_is_blocked() -> None:
    request, semantic, mapping, points = _fixture()
    semantic["bindings"]["analysis_splat_digest"] = "sha256:" + "d" * 64
    _resign_semantic(request, semantic)
    result = _fit(request, semantic, mapping, points)
    assert "semantic_binding_mismatch:analysis_splat_digest" in result["blockers"]


def test_generated_gaussian_cannot_enter_box_support() -> None:
    request, semantic, mapping, points = _fixture()
    mapping[0]["source_class"] = "generated"
    request["bindings"]["gaussian_mapping_digest"] = canonical_json_digest(mapping)
    semantic["bindings"]["gaussian_mapping_digest"] = request["bindings"][
        "gaussian_mapping_digest"
    ]
    _resign_semantic(request, semantic)
    result = _fit(request, semantic, mapping, points)
    box = result["objects"][0]
    assert result["status"] == "abstained"
    assert "generated_or_unknown_gaussian_in_support" in box["abstention_reasons"]
    assert box["next_experiment"] == "recapture_the_object_region_with_direct_observed_geometry"
    assert "corners_world_m" not in box


def test_semantic_abstention_cannot_be_upgraded_by_good_points() -> None:
    request, semantic, mapping, points = _fixture()
    semantic["status"] = "abstained"
    semantic["tracks"][0]["status"] = "abstained"
    _resign_semantic(request, semantic)
    result = _fit(request, semantic, mapping, points)
    box = result["objects"][0]
    assert box["status"] == "abstained"
    assert "semantic_track_not_qualified" in box["abstention_reasons"]
    assert box["next_experiment"] == (
        "capture_or_render_additional_track_views_before_box_fitting"
    )


def test_flat_support_requests_top_bottom_and_contact_capture() -> None:
    request, semantic, mapping, points = _fixture(include_outlier=False)
    for point in points:
        point["point_world_m"][2] = 0.8
    request["bindings"]["support_points_digest"] = canonical_json_digest(points)
    result = _fit(request, semantic, mapping, points)
    box = result["objects"][0]
    assert "insufficient_vertical_extent" in box["abstention_reasons"]
    assert box["next_experiment"] == "capture_the_object_top_bottom_and_support_contact"


def test_implausible_dimension_requests_scale_and_transform_verification() -> None:
    request, semantic, mapping, points = _fixture(include_outlier=False)
    request["qualification"]["max_dimension_m"] = 0.5
    result = _fit(request, semantic, mapping, points)
    box = result["objects"][0]
    assert "dimension_exceeds_configured_limit" in box["abstention_reasons"]
    assert box["next_experiment"] == "verify_metric_scale_and_the_site_to_object_transform"


def test_adjacent_same_label_tracks_remain_separate() -> None:
    request, semantic, mapping, points = _fixture(include_outlier=False)
    split = len(mapping) // 2
    semantic["tracks"] = [
        {
            "track_id": "track_box_01",
            "label": "box",
            "status": "qualified_semantic_support_candidate",
            "selected_gaussian_ids": list(range(split)),
        },
        {
            "track_id": "track_box_02",
            "label": "box",
            "status": "qualified_semantic_support_candidate",
            "selected_gaussian_ids": list(range(split, len(mapping))),
        },
    ]
    _resign_semantic(request, semantic)
    request["qualification"]["min_support_points"] = 4
    request["qualification"]["min_distinct_gaussians"] = 4
    request["qualification"]["min_vertical_extent_m"] = 0.0
    result = _fit(request, semantic, mapping, points)
    assert [row["track_id"] for row in result["objects"]] == ["track_box_01", "track_box_02"]
    assert result["objects"][0]["selected_gaussian_ids"] != result["objects"][1][
        "selected_gaussian_ids"
    ]


def test_support_point_digest_and_coordinate_validation_fail_closed() -> None:
    request, semantic, mapping, points = _fixture()
    points[0]["point_world_m"][0] = float("nan")
    result = _fit(request, semantic, mapping, points)
    assert result["status"] == "blocked"
    assert "support_points_not_canonical_json" in result["blockers"]


def test_file_stage_verifies_inputs_and_writes_result(tmp_path: Path) -> None:
    request, semantic, mapping, points = _fixture()
    paths = _write_stage_inputs(tmp_path, request, semantic, mapping, points)
    result = run_semantic_oriented_box_stage(
        request_path=paths["request"],
        semantic_result_path=paths["semantic_result"],
        gaussian_mapping_path=paths["gaussian_mapping"],
        support_points_path=paths["support_points"],
        output_path=paths["output"],
    )
    assert result["status"] == "completed"
    assert result["transport_profile"] == "bounded_canonical_json_baseline.v1"
    assert set(result["stage_input_artifacts"]) == {
        "request",
        "semantic_result",
        "gaussian_mapping",
        "support_points",
    }
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == result


def test_file_stage_detects_tampering_and_returns_blocked(tmp_path: Path) -> None:
    request, semantic, mapping, points = _fixture()
    paths = _write_stage_inputs(tmp_path, request, semantic, mapping, points)
    paths["support_points"].write_text("[]", encoding="utf-8")
    exit_code = stage_main(
        [
            "--request",
            str(paths["request"]),
            "--semantic-result",
            str(paths["semantic_result"]),
            "--gaussian-mapping",
            str(paths["gaussian_mapping"]),
            "--support-points",
            str(paths["support_points"]),
            "--output",
            str(paths["output"]),
        ]
    )
    result = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert exit_code == 2
    assert result["status"] == "blocked"
    assert "input_artifact_sha256_mismatch:support_points" in result["blockers"]
    assert "input_artifact_size_mismatch:support_points" in result["blockers"]


def test_file_stage_refuses_input_overwrite(tmp_path: Path) -> None:
    request, semantic, mapping, points = _fixture()
    paths = _write_stage_inputs(tmp_path, request, semantic, mapping, points)
    with pytest.raises(ValueError, match="output_path_must_not_overwrite_an_input"):
        run_semantic_oriented_box_stage(
            request_path=paths["request"],
            semantic_result_path=paths["semantic_result"],
            gaussian_mapping_path=paths["gaussian_mapping"],
            support_points_path=paths["support_points"],
            output_path=paths["support_points"],
        )
