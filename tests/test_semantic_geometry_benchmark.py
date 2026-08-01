"""Tests for independent semantic OBB benchmark diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import canonical_json_digest
from blueprint_pipeline.scene_placement.semantic_geometry_benchmark import (
    BENCHMARK_METHOD,
    REQUEST_SCHEMA_VERSION,
    benchmark_semantic_geometry,
)
from blueprint_pipeline.semantic_geometry_benchmark_stage import (
    main as stage_main,
    run_semantic_geometry_benchmark_stage,
)


_CAPTURE = "sha256:" + "a" * 64
_RECONSTRUCTION = "sha256:" + "b" * 64
_SPLAT = "sha256:" + "c" * 64


def _corners(center: list[float], dimensions: list[float], yaw: float) -> list[list[float]]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    half_x, half_y, half_z = (value * 0.5 for value in dimensions)
    return [
        [
            center[0] + x * cosine - y * sine,
            center[1] + x * sine + y * cosine,
            center[2] + z,
        ]
        for z in (-half_z, half_z)
        for x, y in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
    ]


def _object(
    identifier: str,
    *,
    reference: bool = False,
    label: str = "chair",
    center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    dimensions: tuple[float, float, float] = (1.0, 0.5, 1.0),
    yaw: float = 0.0,
    qualified: bool = True,
) -> dict:
    center_row = list(center)
    dimension_row = list(dimensions)
    row = {
        ("reference_object_id" if reference else "track_id"): identifier,
        "label": label,
        "center_world_m": center_row,
        "dimensions_m": dimension_row,
        "yaw_rad": yaw,
        "corners_world_m": _corners(center_row, dimension_row, yaw),
    }
    if not reference:
        row.update(
            {
                "status": "qualified_metric_obb_candidate" if qualified else "abstained",
                "metric_obb_candidate_ready": qualified,
                "collision_ready": False,
                "physics_ready": False,
            }
        )
    return row


def _prediction(objects: list[dict]) -> dict:
    result = {
        "schema_version": "semantic_oriented_box_result.v1",
        "status": "completed" if objects else "abstained",
        "bindings": {
            "capture_digest": _CAPTURE,
            "reconstruction_digest": _RECONSTRUCTION,
            "analysis_splat_digest": _SPLAT,
        },
        "world": {
            "up_axis": "Z",
            "units": "meters",
            "scale_verified": True,
            "coordinate_frame": "site_z_up",
        },
        "objects": objects,
        "collision_ready": False,
        "physics_ready": False,
        "generated_regions_can_upgrade_claims": False,
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def _ground_truth(objects: list[dict]) -> dict:
    result = {
        "schema_version": "semantic_geometry_ground_truth.v1",
        "bindings": {
            "capture_digest": _CAPTURE,
            "reconstruction_digest": _RECONSTRUCTION,
        },
        "world": {
            "up_axis": "Z",
            "units": "meters",
            "scale_verified": True,
            "coordinate_frame": "site_z_up",
        },
        "annotation_profile": {
            "source_type": "independently_registered_laser_scan",
            "source_artifact_digest": "sha256:" + "d" * 64,
            "alignment_digest": "sha256:" + "e" * 64,
            "producer_identity": "reference-team",
            "reviewer_identity": "independent-reviewer",
            "independent_from_prediction": True,
            "metric_authority_verified": True,
            "review_status": "accepted",
            "withheld_from_prediction": True,
            "rights_cleared_for_evaluation": True,
        },
        "objects": objects,
    }
    result["annotation_digest"] = canonical_json_digest(result)
    return result


def _fixture(
    *,
    predictions: list[dict] | None = None,
    references: list[dict] | None = None,
    ablations: list[dict] | None = None,
) -> tuple[dict, dict, dict, list[dict]]:
    prediction = _prediction(
        predictions
        if predictions is not None
        else [
            _object("track_a", center=(0.0, 0.0, 0.5), yaw=0.2),
            _object("track_b", center=(1.2, 0.0, 0.5), yaw=-0.1),
            _object("track_fp", label="table", center=(8.0, 8.0, 0.5)),
        ]
    )
    ground_truth = _ground_truth(
        references
        if references is not None
        else [
            _object("ref_a", reference=True, center=(0.0, 0.0, 0.5), yaw=0.2),
            _object("ref_b", reference=True, center=(1.2, 0.0, 0.5), yaw=-0.1),
        ]
    )
    ablation_runs = ablations or []
    manifest = [
        {
            "ablation_id": row["ablation_id"],
            "removed_view_ids": sorted(row["removed_view_ids"]),
            "prediction_result_digest": row["prediction_result"]["result_digest"],
        }
        for row in sorted(ablation_runs, key=lambda item: item["ablation_id"])
    ]
    profile = {
        "evaluator_id": "blueprint-semantic-geometry-benchmark",
        "evaluator_version": "1.0.0",
        "evaluator_identity": "blueprint-evaluator",
        "split_id": "held-out-room-01",
        "runtime_digest": "sha256:" + "f" * 64,
        "deterministic": True,
        "prediction_input_manifest_complete": True,
        "max_match_center_distance_m": 2.0,
        "min_match_3d_iou": 0.1,
        "adjacent_same_label_distance_m": 2.0,
        "square_yaw_ambiguity_ratio": 1.05,
    }
    profile["benchmark_profile_digest"] = canonical_json_digest(profile)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "benchmark_method": BENCHMARK_METHOD,
        "bindings": {
            "capture_digest": _CAPTURE,
            "reconstruction_digest": _RECONSTRUCTION,
            "analysis_splat_digest": _SPLAT,
            "prediction_result_digest": prediction["result_digest"],
            "ground_truth_digest": ground_truth["annotation_digest"],
            "benchmark_profile_digest": profile["benchmark_profile_digest"],
            "prediction_input_manifest_digest": canonical_json_digest(
                sorted([_CAPTURE, _RECONSTRUCTION, _SPLAT])
            ),
            "evaluation_view_registry_digest": canonical_json_digest(
                ["view_01", "view_08", "view_09"]
            ),
            "view_ablation_manifest_digest": canonical_json_digest(manifest),
        },
        "world": ground_truth["world"],
        "benchmark_profile": profile,
        "prediction_input_digests": [_SPLAT, _CAPTURE, _RECONSTRUCTION],
        "evaluation_view_ids": ["view_09", "view_01", "view_08"],
    }
    return request, prediction, ground_truth, ablation_runs


def _run(fixture: tuple[dict, dict, dict, list[dict]]) -> dict:
    request, prediction, ground_truth, ablations = fixture
    return benchmark_semantic_geometry(
        request,
        prediction_result=prediction,
        ground_truth=ground_truth,
        ablation_runs=ablations,
    )


def test_reports_detection_geometry_and_adjacent_instance_metrics() -> None:
    result = _run(_fixture())

    assert result["status"] == "completed"
    assert result["counts"] == {
        "reference_objects": 2,
        "predicted_objects": 3,
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 0,
    }
    assert result["metrics"]["object_recall"] == 1.0
    assert result["metrics"]["false_positive_fraction_of_predictions"] == pytest.approx(1 / 3)
    assert result["metrics"]["mean_center_error_cm"] == 0.0
    assert result["metrics"]["mean_dimension_abs_error_m"] == 0.0
    assert result["metrics"]["mean_yaw_error_deg"] == 0.0
    assert result["metrics"]["mean_obb_3d_iou"] == 1.0
    assert result["metrics"]["adjacent_same_label_pair_recall"] == 1.0
    assert result["unmatched_track_ids"] == ["track_fp"]
    assert result["collision_ready"] is False
    assert result["physics_ready"] is False


def test_dimension_swap_and_quarter_turn_are_same_cuboid() -> None:
    prediction = _object(
        "track_a", center=(0.0, 0.0, 0.5), dimensions=(0.5, 1.0, 1.0), yaw=math.pi / 2
    )
    reference = _object("ref_a", reference=True)
    result = _run(_fixture(predictions=[prediction], references=[reference]))

    assert result["metrics"]["mean_dimension_abs_error_m"] == 0.0
    assert result["metrics"]["mean_yaw_error_deg"] == 0.0
    assert result["metrics"]["mean_obb_3d_iou"] == 1.0


def test_square_reference_excludes_ambiguous_yaw() -> None:
    prediction = _object("track_a", dimensions=(1.0, 1.0, 1.0), yaw=0.7)
    reference = _object("ref_a", reference=True, dimensions=(1.0, 1.0, 1.0), yaw=0.0)
    result = _run(_fixture(predictions=[prediction], references=[reference]))

    assert result["metrics"]["mean_yaw_error_deg"] is None
    assert result["metrics"]["yaw_evaluable_match_count"] == 0


def test_view_removal_reports_track_survival_and_geometry_drift() -> None:
    ablated = _prediction(
        [_object("track_a", center=(0.02, 0.0, 0.5), yaw=0.2), _object("track_fp", label="table", center=(8.0, 8.0, 0.5))]
    )
    runs = [
        {
            "ablation_id": "remove_rear_views",
            "removed_view_ids": ["view_08", "view_09"],
            "prediction_result": ablated,
        }
    ]
    result = _run(_fixture(ablations=runs))

    row = result["view_ablation"][0]
    assert row["retained_track_count"] == 2
    assert row["retained_track_fraction"] == pytest.approx(2 / 3)
    assert row["unexpected_track_count"] == 0
    assert row["track_set_jaccard"] == pytest.approx(2 / 3)
    assert row["mean_center_drift_cm"] == pytest.approx(1.0)
    assert row["mean_obb_3d_iou_to_baseline"] < 1.0


def test_view_removal_penalizes_new_tracks_and_requires_registered_views() -> None:
    ablated = _prediction(
        [
            _object("track_a", yaw=0.2),
            _object("track_new", label="lamp", center=(4.0, 4.0, 0.5)),
        ]
    )
    runs = [
        {
            "ablation_id": "remove_side",
            "removed_view_ids": ["view_01"],
            "prediction_result": ablated,
        }
    ]
    result = _run(_fixture(ablations=runs))
    row = result["view_ablation"][0]
    assert row["unexpected_track_ids"] == ["track_new"]
    assert row["unexpected_track_count"] == 1
    assert row["track_set_jaccard"] == 0.25

    request, prediction, ground_truth, _ = _fixture(ablations=runs)
    runs[0]["removed_view_ids"] = ["unknown_view"]
    request["bindings"]["view_ablation_manifest_digest"] = canonical_json_digest(
        [
            {
                "ablation_id": "remove_side",
                "removed_view_ids": ["unknown_view"],
                "prediction_result_digest": ablated["result_digest"],
            }
        ]
    )
    result = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=runs
    )
    assert any("removed_view_id_not_in_evaluation_registry" in item for item in result["blockers"])


def test_optimizer_keeps_adjacent_same_label_instances_distinct() -> None:
    predictions = [
        _object("track_left", center=(-0.45, 0.0, 0.5)),
        _object("track_right", center=(0.45, 0.0, 0.5)),
    ]
    references = [
        _object("ref_left", reference=True, center=(-0.5, 0.0, 0.5)),
        _object("ref_right", reference=True, center=(0.5, 0.0, 0.5)),
    ]
    result = _run(_fixture(predictions=predictions, references=references))

    assert {(row["reference_object_id"], row["track_id"]) for row in result["matches"]} == {
        ("ref_left", "track_left"),
        ("ref_right", "track_right"),
    }
    assert result["metrics"]["adjacent_same_label_pair_recall"] == 1.0


def test_abstained_predictions_are_not_counted_as_objects() -> None:
    prediction = _object("track_a", qualified=False)
    result = _run(_fixture(predictions=[prediction]))

    assert result["counts"]["predicted_objects"] == 0
    assert result["metrics"]["object_recall"] == 0.0
    assert result["metrics"]["mean_center_error_cm"] is None


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda request, prediction, ground_truth: prediction["objects"][0].update(label="desk"), "prediction_result_digest_invalid"),
        (lambda request, prediction, ground_truth: request["bindings"].update(capture_digest="sha256:" + "9" * 64), "prediction_binding_mismatch:capture_digest"),
        (lambda request, prediction, ground_truth: ground_truth["annotation_profile"].update(rights_cleared_for_evaluation=False), "ground_truth_digest_invalid"),
        (lambda request, prediction, ground_truth: ground_truth["annotation_profile"].update(withheld_from_prediction=False), "ground_truth_digest_invalid"),
        (lambda request, prediction, ground_truth: request["benchmark_profile"].update(min_match_3d_iou=0.9), "benchmark_profile_digest_invalid"),
    ],
)
def test_tampering_and_stale_bindings_fail_closed(mutation, expected: str) -> None:
    request, prediction, ground_truth, runs = _fixture()
    mutation(request, prediction, ground_truth)
    result = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=runs
    )

    assert result["status"] == "blocked"
    assert expected in result["blockers"]


def test_recomputed_ground_truth_still_rejects_self_review_and_leakage() -> None:
    request, prediction, ground_truth, runs = _fixture()
    ground_truth["annotation_profile"]["reviewer_identity"] = "reference-team"
    ground_truth["annotation_profile"]["withheld_from_prediction"] = False
    ground_truth.pop("annotation_digest")
    ground_truth["annotation_digest"] = canonical_json_digest(ground_truth)
    request["bindings"]["ground_truth_digest"] = ground_truth["annotation_digest"]
    result = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=runs
    )

    assert result["status"] == "blocked"
    assert "ground_truth_independent_review_required" in result["blockers"]
    assert "ground_truth_prediction_leakage_forbidden" in result["blockers"]


def test_ground_truth_artifact_in_prediction_manifest_is_rejected() -> None:
    request, prediction, ground_truth, runs = _fixture()
    leaked_digest = ground_truth["annotation_profile"]["source_artifact_digest"]
    request["prediction_input_digests"].append(leaked_digest)
    request["bindings"]["prediction_input_manifest_digest"] = canonical_json_digest(
        sorted(str(item).lower() for item in request["prediction_input_digests"])
    )
    result = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=runs
    )

    assert result["status"] == "blocked"
    assert "ground_truth_leaked_into_prediction_inputs:source_artifact_digest" in result["blockers"]


def test_ablation_cannot_reuse_baseline_or_hide_manifest_change() -> None:
    request, prediction, ground_truth, _ = _fixture()
    runs = [
        {
            "ablation_id": "fake",
            "removed_view_ids": ["view_01"],
            "prediction_result": prediction,
        }
    ]
    request["bindings"]["view_ablation_manifest_digest"] = canonical_json_digest(
        [
            {
                "ablation_id": "fake",
                "removed_view_ids": ["view_01"],
                "prediction_result_digest": prediction["result_digest"],
            }
        ]
    )
    result = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=runs
    )
    assert result["status"] == "blocked"
    assert any("ablation_must_not_reuse_baseline_result" in item for item in result["blockers"])

    request["bindings"]["view_ablation_manifest_digest"] = "sha256:" + "0" * 64
    result = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=runs
    )
    assert "view_ablation_manifest_digest_mismatch" in result["blockers"]


def test_result_is_deterministic_under_input_order_changes() -> None:
    fixture = _fixture()
    first = _run(fixture)
    request, prediction, ground_truth, runs = copy.deepcopy(fixture)
    prediction["objects"].reverse()
    prediction.pop("result_digest")
    prediction["result_digest"] = canonical_json_digest(prediction)
    request["bindings"]["prediction_result_digest"] = prediction["result_digest"]
    ground_truth["objects"].reverse()
    ground_truth.pop("annotation_digest")
    ground_truth["annotation_digest"] = canonical_json_digest(ground_truth)
    request["bindings"]["ground_truth_digest"] = ground_truth["annotation_digest"]
    second = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=runs
    )

    comparable_first = copy.deepcopy(first)
    comparable_second = copy.deepcopy(second)
    for payload in (comparable_first, comparable_second):
        payload.pop("result_digest")
        payload["bindings"].pop("prediction_result_digest")
        payload["bindings"].pop("ground_truth_digest")
    assert comparable_first == comparable_second


def test_generated_region_authority_upgrade_is_rejected_even_when_digest_recomputed() -> None:
    request, prediction, ground_truth, runs = _fixture()
    prediction["generated_regions_can_upgrade_claims"] = True
    prediction.pop("result_digest")
    prediction["result_digest"] = canonical_json_digest(prediction)
    request["bindings"]["prediction_result_digest"] = prediction["result_digest"]
    result = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=runs
    )

    assert result["status"] == "blocked"
    assert "prediction_generated_region_boundary_invalid" in result["blockers"]


def test_malformed_ablation_row_and_registry_fail_closed_without_exception() -> None:
    request, prediction, ground_truth, _ = _fixture()
    result = benchmark_semantic_geometry(
        request,
        prediction_result=prediction,
        ground_truth=ground_truth,
        ablation_runs=["not-an-object"],  # type: ignore[list-item]
    )
    assert result["status"] == "blocked"
    assert "view_ablation_row_invalid" in result["blockers"]

    request["evaluation_view_ids"] = []
    result = benchmark_semantic_geometry(
        request, prediction_result=prediction, ground_truth=ground_truth, ablation_runs=[]
    )
    assert "evaluation_view_registry_invalid" in result["blockers"]


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _stage_fixture(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    request, prediction, ground_truth, runs = _fixture()
    paths = {
        "request": tmp_path / "request.json",
        "prediction_result": tmp_path / "prediction.json",
        "ground_truth": tmp_path / "ground_truth.json",
        "ablation_runs": tmp_path / "ablations.json",
        "output": tmp_path / "result.json",
    }
    for name, payload in (
        ("prediction_result", prediction),
        ("ground_truth", ground_truth),
        ("ablation_runs", runs),
    ):
        _write_json(paths[name], payload)
    request["input_artifacts"] = {
        name: {
            "sha256": "sha256:" + hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "size_bytes": paths[name].stat().st_size,
        }
        for name in ("prediction_result", "ground_truth", "ablation_runs")
    }
    _write_json(paths["request"], request)
    return paths, request


def test_file_stage_binds_exact_artifacts_and_cli(tmp_path: Path) -> None:
    paths, _ = _stage_fixture(tmp_path)
    result = run_semantic_geometry_benchmark_stage(
        request_path=paths["request"],
        prediction_result_path=paths["prediction_result"],
        ground_truth_path=paths["ground_truth"],
        ablation_runs_path=paths["ablation_runs"],
        output_path=paths["output"],
    )

    assert result["status"] == "completed"
    assert result["transport_profile"] == "bounded_canonical_json_baseline.v1"
    assert set(result["stage_input_artifacts"]) == {
        "request",
        "prediction_result",
        "ground_truth",
        "ablation_runs",
    }
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == result

    second_output = tmp_path / "result_cli.json"
    assert (
        stage_main(
            [
                "--request",
                str(paths["request"]),
                "--prediction-result",
                str(paths["prediction_result"]),
                "--ground-truth",
                str(paths["ground_truth"]),
                "--ablation-runs",
                str(paths["ablation_runs"]),
                "--output",
                str(second_output),
            ]
        )
        == 0
    )


def test_file_stage_rejects_tampering_symlinks_and_input_overwrite(tmp_path: Path) -> None:
    paths, _ = _stage_fixture(tmp_path)
    paths["prediction_result"].write_text("{}\n", encoding="utf-8")
    result = run_semantic_geometry_benchmark_stage(
        request_path=paths["request"],
        prediction_result_path=paths["prediction_result"],
        ground_truth_path=paths["ground_truth"],
        ablation_runs_path=paths["ablation_runs"],
        output_path=paths["output"],
    )
    assert result["status"] == "blocked"
    assert "input_artifact_sha256_mismatch:prediction_result" in result["blockers"]

    paths, _ = _stage_fixture(tmp_path / "symlink_case")
    link = tmp_path / "prediction_link.json"
    link.symlink_to(paths["prediction_result"])
    result = run_semantic_geometry_benchmark_stage(
        request_path=paths["request"],
        prediction_result_path=link,
        ground_truth_path=paths["ground_truth"],
        ablation_runs_path=paths["ablation_runs"],
        output_path=paths["output"],
    )
    assert "input_symlink_forbidden:prediction_result" in result["blockers"]

    with pytest.raises(ValueError, match="output_path_must_not_overwrite_an_input"):
        run_semantic_geometry_benchmark_stage(
            request_path=paths["request"],
            prediction_result_path=paths["prediction_result"],
            ground_truth_path=paths["ground_truth"],
            ablation_runs_path=paths["ablation_runs"],
            output_path=paths["ground_truth"],
        )
