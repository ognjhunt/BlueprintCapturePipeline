"""Contract tests for source-bound mask-to-Gaussian semantic lifting."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    CONTRIBUTION_SEMANTICS,
    REQUEST_SCHEMA_VERSION,
    canonical_json_digest,
    lift_semantic_masks_to_gaussians,
)
from blueprint_pipeline.semantic_gaussian_lifting_stage import (
    main as stage_main,
    run_semantic_gaussian_lifting_stage,
)


_EXTERNAL_DIGEST = "sha256:" + "a" * 64


def _fixture() -> tuple[dict, list[dict], list[dict], list[dict]]:
    mapping = [
        {"gaussian_id": 0, "source_index": 10, "source_class": "observed"},
        {"gaussian_id": 1, "source_index": 11, "source_class": "observed"},
        {"gaussian_id": 2, "source_index": 12, "source_class": "generated"},
    ]
    tracks = [
        {
            "track_id": "track_chair_01",
            "label": "chair",
            "mask_model_digest": _EXTERNAL_DIGEST,
            "track_evidence_digest": "sha256:" + "b" * 64,
        }
    ]

    def view(view_id: str, direction: list[float], suffix: str) -> dict:
        camera_record = {
            "intrinsics": [500.0, 500.0, 1.0, 0.5],
            "camera_to_world": [
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
            "coordinate_frame": "analysis_splat_z_up_meters",
        }
        pixels = [
            {
                "pixel_id": 0,
                "mask_probabilities": {"track_chair_01": 1.0},
                "contributions": [
                    {"gaussian_id": 0, "weight": 0.8},
                    {"gaussian_id": 2, "weight": 0.1},
                ],
            },
            {
                "pixel_id": 1,
                "mask_probabilities": {},
                "contributions": [{"gaussian_id": 1, "weight": 0.8}],
            },
        ]
        mask_payload = [
            {
                "pixel_id": pixel["pixel_id"],
                "mask_probabilities": pixel["mask_probabilities"],
            }
            for pixel in pixels
        ]
        contribution_payload = [
            {
                "pixel_id": pixel["pixel_id"],
                "contributions": pixel["contributions"],
            }
            for pixel in pixels
        ]
        return {
            "view_id": view_id,
            "source_frame_id": f"frame_{suffix}",
            "source_frame_digest": "sha256:" + suffix * 64,
            "decoded_pts_seconds": float(int(suffix)),
            "camera_record": camera_record,
            "camera_record_digest": canonical_json_digest(camera_record),
            "mask_artifact_digest": canonical_json_digest(mask_payload),
            "contribution_artifact_digest": canonical_json_digest(contribution_payload),
            "width": 2,
            "height": 1,
            "coverage_kind": "full_frame",
            "view_direction_world": direction,
            "pixels": pixels,
        }

    views = [view("view_a", [1.0, 0.0, 0.0], "1"), view("view_b", [0.0, 1.0, 0.0], "2")]
    frame_registry = [
        {
            "source_frame_id": view["source_frame_id"],
            "source_frame_digest": view["source_frame_digest"],
            "retained_video_digest": "sha256:" + "9" * 64,
            "decoded_pts_seconds": view["decoded_pts_seconds"],
            "sync_map_row_digest": "sha256:" + "8" * 64,
            "camera_record_digest": view["camera_record_digest"],
            "encoder_retained": True,
        }
        for view in views
    ]
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "bindings": {
            "capture_digest": _EXTERNAL_DIGEST,
            "retained_video_digest": "sha256:" + "9" * 64,
            "reconstruction_digest": "sha256:" + "1" * 64,
            "analysis_splat_digest": "sha256:" + "2" * 64,
            "camera_solution_digest": "sha256:" + "3" * 64,
            "gaussian_mapping_digest": canonical_json_digest(mapping),
            "frame_registry_digest": canonical_json_digest(frame_registry),
            "track_registry_digest": canonical_json_digest(tracks),
            "views_digest": canonical_json_digest(views),
        },
        "frame_registry": frame_registry,
        "gaussian_count": len(mapping),
        "renderer_profile": {
            "method_id": "blueprint.hermetic_contribution_fixture",
            "method_version": "1.0.0",
            "runtime_digest": "sha256:" + "4" * 64,
            "contribution_semantics": CONTRIBUTION_SEMANTICS,
            "exact_gaussian_ids": True,
            "deterministic": True,
        },
        "world": {"up_axis": "Z", "units": "meters", "scale_verified": True},
        "qualification": {
            "min_track_views": 2,
            "min_gaussian_views": 2,
            "min_view_foreground_contribution": 0.5,
            "min_gaussian_view_foreground_contribution": 0.05,
            "min_gaussian_total_contribution": 0.15,
            "foreground_probability_threshold": 0.7,
            "min_angular_diversity_degrees": 30.0,
        },
    }
    return request, mapping, tracks, views


def _lift(
    request: dict,
    mapping: list[dict],
    tracks: list[dict],
    views: list[dict],
) -> dict:
    return lift_semantic_masks_to_gaussians(
        request,
        gaussian_mapping=mapping,
        track_registry=tracks,
        views=views,
    )


def _rebind_views(request: dict, views: list[dict]) -> None:
    for view in views:
        pixels = view["pixels"]
        view["mask_artifact_digest"] = canonical_json_digest(
            [
                {
                    "pixel_id": pixel["pixel_id"],
                    "mask_probabilities": pixel["mask_probabilities"],
                }
                for pixel in pixels
            ]
        )
        view["contribution_artifact_digest"] = canonical_json_digest(
            [
                {
                    "pixel_id": pixel["pixel_id"],
                    "contributions": pixel["contributions"],
                }
                for pixel in pixels
            ]
        )
    request["bindings"]["views_digest"] = canonical_json_digest(views)


def _write_stage_inputs(
    root: Path,
    request: dict,
    mapping: list[dict],
    tracks: list[dict],
    views: list[dict],
) -> dict[str, Path]:
    paths = {
        "gaussian_mapping": root / "gaussian_mapping.json",
        "track_registry": root / "track_registry.json",
        "views": root / "views.json",
    }
    payloads = {
        "gaussian_mapping": mapping,
        "track_registry": tracks,
        "views": views,
    }
    request["input_artifacts"] = {}
    for name, path in paths.items():
        path.write_text(json.dumps(payloads[name], indent=2), encoding="utf-8")
        request["input_artifacts"][name] = {
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    request_path = root / "request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    paths["request"] = request_path
    paths["output"] = root / "result.json"
    return paths


def test_lifts_full_mask_contributions_and_excludes_generated_support() -> None:
    request, mapping, tracks, views = _fixture()
    result = _lift(request, mapping, tracks, views)

    assert result["status"] == "completed"
    assert result["claim_ceiling"] == "per_gaussian_semantic_support_candidate_metric_frame"
    assert result["canonical_object_geometry"] is False
    assert result["metric_box_ready"] is False
    assert result["physics_ready"] is False
    assert result["generated_regions_can_upgrade_claims"] is False
    track = result["tracks"][0]
    assert track["status"] == "qualified_semantic_support_candidate"
    assert track["supporting_view_ids"] == ["view_a", "view_b"]
    assert track["angular_diversity_degrees"] == pytest.approx(90.0)
    assert track["selected_gaussian_ids"] == [0]
    assert track["generated_candidate_gaussian_count"] == 1
    generated = next(row for row in track["gaussian_evidence"] if row["gaussian_id"] == 2)
    assert generated["foreground_probability"] == pytest.approx(1.0)
    assert generated["selected_for_semantic_support"] is False
    assert result["result_digest"].startswith("sha256:")


def test_tampered_embedded_payload_digest_blocks_before_lifting() -> None:
    request, mapping, tracks, views = _fixture()
    views[0]["pixels"][0]["contributions"][0]["weight"] = 0.7
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "blocked"
    assert "views_digest_mismatch" in result["blockers"]
    assert result["claim_ceiling"] == "none_invalid_or_unbound_input"


@pytest.mark.parametrize(
    ("mutator", "expected_blocker"),
    [
        (
            lambda views: views[0]["camera_record"]["intrinsics"].__setitem__(0, 600.0),
            "view_camera_record_digest_mismatch:view_a",
        ),
        (
            lambda views: views[0]["pixels"][0]["mask_probabilities"].update(
                {"track_chair_01": 0.8}
            ),
            "view_mask_artifact_digest_mismatch:view_a",
        ),
        (
            lambda views: views[0]["pixels"][0]["contributions"][0].update(
                {"weight": 0.7}
            ),
            "view_contribution_artifact_digest_mismatch:view_a",
        ),
    ],
)
def test_each_parsed_view_payload_is_independently_hash_verified(
    mutator, expected_blocker: str
) -> None:
    request, mapping, tracks, views = _fixture()
    mutator(views)
    request["bindings"]["views_digest"] = canonical_json_digest(views)
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "blocked"
    assert expected_blocker in result["blockers"]


def test_single_view_abstains_with_exact_next_experiment() -> None:
    request, mapping, tracks, views = _fixture()
    views = views[:1]
    _rebind_views(request, views)
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "abstained"
    track = result["tracks"][0]
    assert track["selected_gaussian_ids"] == []  # min_gaussian_views is still two
    assert "insufficient_distinct_views" in track["abstention_reasons"]
    assert track["next_experiment"] == "render_or_capture_additional_overlapping_views_of_this_track"


def test_view_must_match_retained_encoder_frame_and_decoded_pts() -> None:
    request, mapping, tracks, views = _fixture()
    views[0]["decoded_pts_seconds"] = 9.0
    request["bindings"]["views_digest"] = canonical_json_digest(views)
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "blocked"
    assert "view_decoded_pts_mismatch:view_a" in result["blockers"]

    request, mapping, tracks, views = _fixture()
    request["frame_registry"][0]["encoder_retained"] = False
    request["bindings"]["frame_registry_digest"] = canonical_json_digest(
        request["frame_registry"]
    )
    result = _lift(request, mapping, tracks, views)
    assert "frame_registry_encoder_retention_not_proven:frame_1" in result["blockers"]


def test_generated_only_semantic_evidence_abstains_and_requests_recapture() -> None:
    request, mapping, tracks, views = _fixture()
    mapping[0]["source_class"] = "generated"
    request["bindings"]["gaussian_mapping_digest"] = canonical_json_digest(mapping)
    result = _lift(request, mapping, tracks, views)
    track = result["tracks"][0]
    assert result["status"] == "abstained"
    assert track["selected_gaussian_ids"] == []
    assert track["generated_candidate_gaussian_count"] == 2
    assert track["next_experiment"] == "recapture_the_generated_only_or_unobserved_object_region"


def test_overfull_renderer_contributions_fail_closed() -> None:
    request, mapping, tracks, views = _fixture()
    views[0]["pixels"][0]["contributions"][0]["weight"] = 0.95
    _rebind_views(request, views)
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "blocked"
    assert "pixel_contribution_sum_exceeds_one:view_a:0" in result["blockers"]


def test_incomplete_or_crop_only_renderer_payload_fails_closed() -> None:
    request, mapping, tracks, views = _fixture()
    views[0]["coverage_kind"] = "masked_crop"
    views[0]["pixels"] = views[0]["pixels"][:1]
    _rebind_views(request, views)
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "blocked"
    assert "view_full_frame_contribution_coverage_required:view_a" in result["blockers"]
    assert "view_pixel_coverage_incomplete:view_a" in result["blockers"]


def test_persistent_track_ids_keep_adjacent_same_label_instances_separate() -> None:
    request, mapping, tracks, views = _fixture()
    mapping[2]["source_class"] = "observed"
    tracks.append(
        {
            "track_id": "track_chair_02",
            "label": "chair",
            "mask_model_digest": _EXTERNAL_DIGEST,
            "track_evidence_digest": "sha256:" + "f" * 64,
        }
    )
    for view in views:
        view["pixels"][0]["contributions"] = [{"gaussian_id": 0, "weight": 0.8}]
        view["pixels"][1]["mask_probabilities"] = {"track_chair_02": 1.0}
        view["pixels"][1]["contributions"] = [{"gaussian_id": 2, "weight": 0.8}]
    request["bindings"]["gaussian_mapping_digest"] = canonical_json_digest(mapping)
    request["bindings"]["track_registry_digest"] = canonical_json_digest(tracks)
    _rebind_views(request, views)
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "completed"
    by_track = {row["track_id"]: row for row in result["tracks"]}
    assert by_track["track_chair_01"]["selected_gaussian_ids"] == [0]
    assert by_track["track_chair_02"]["selected_gaussian_ids"] == [2]


def test_unverified_scale_preserves_semantics_but_lowers_claim_ceiling() -> None:
    request, mapping, tracks, views = _fixture()
    request["world"]["scale_verified"] = False
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "completed"
    assert result["claim_ceiling"] == "per_gaussian_semantic_support_candidate_unverified_scale"
    assert result["metric_box_ready"] is False


def test_contribution_order_does_not_change_semantic_evidence() -> None:
    request, mapping, tracks, views = _fixture()
    original = _lift(request, mapping, tracks, views)
    permuted_views = copy.deepcopy(views)
    permuted_views.reverse()
    for view in permuted_views:
        view["pixels"].reverse()
        for pixel in view["pixels"]:
            pixel["contributions"].reverse()
    _rebind_views(request, permuted_views)
    permuted = _lift(request, mapping, tracks, permuted_views)
    original_track = original["tracks"][0]
    permuted_track = permuted["tracks"][0]
    assert permuted_track["selected_gaussian_ids"] == original_track["selected_gaussian_ids"]
    assert permuted_track["gaussian_evidence"] == original_track["gaussian_evidence"]
    assert permuted_track["supporting_view_ids"] == original_track["supporting_view_ids"]


@pytest.mark.parametrize(
    ("mutator", "expected_blocker"),
    [
        (
            lambda request, mapping, tracks, views: request["renderer_profile"].update(
                {"contribution_semantics": "center_depth_proxy"}
            ),
            "renderer_contribution_semantics_unsupported",
        ),
        (
            lambda request, mapping, tracks, views: mapping[0].update(
                {"source_class": "physics_truth"}
            ),
            "gaussian_source_class_invalid",
        ),
        (
            lambda request, mapping, tracks, views: views[0]["pixels"][0][
                "contributions"
            ].append({"gaussian_id": 99, "weight": 0.01}),
            "pixel_gaussian_id_invalid_or_duplicate:view_a:0",
        ),
    ],
)
def test_invalid_authority_or_renderer_contracts_block(mutator, expected_blocker: str) -> None:
    request, mapping, tracks, views = _fixture()
    mutator(request, mapping, tracks, views)
    request["bindings"]["gaussian_mapping_digest"] = canonical_json_digest(mapping)
    request["bindings"]["track_registry_digest"] = canonical_json_digest(tracks)
    _rebind_views(request, views)
    result = _lift(request, mapping, tracks, views)
    assert result["status"] == "blocked"
    assert expected_blocker in result["blockers"]


def test_file_bound_stage_verifies_inputs_and_writes_terminal_result(tmp_path: Path) -> None:
    request, mapping, tracks, views = _fixture()
    paths = _write_stage_inputs(tmp_path, request, mapping, tracks, views)
    result = run_semantic_gaussian_lifting_stage(
        request_path=paths["request"],
        gaussian_mapping_path=paths["gaussian_mapping"],
        track_registry_path=paths["track_registry"],
        views_path=paths["views"],
        output_path=paths["output"],
    )
    assert result["status"] == "completed"
    assert result["transport_profile"] == "bounded_canonical_json_baseline.v1"
    assert set(result["stage_input_artifacts"]) == {
        "request",
        "gaussian_mapping",
        "track_registry",
        "views",
    }
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == result


def test_file_bound_stage_detects_post_manifest_artifact_tampering(tmp_path: Path) -> None:
    request, mapping, tracks, views = _fixture()
    paths = _write_stage_inputs(tmp_path, request, mapping, tracks, views)
    paths["views"].write_text("[]", encoding="utf-8")
    exit_code = stage_main(
        [
            "--request",
            str(paths["request"]),
            "--gaussian-mapping",
            str(paths["gaussian_mapping"]),
            "--track-registry",
            str(paths["track_registry"]),
            "--views",
            str(paths["views"]),
            "--output",
            str(paths["output"]),
        ]
    )
    result = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert exit_code == 2
    assert result["status"] == "blocked"
    assert "input_artifact_sha256_mismatch:views" in result["blockers"]
    assert "input_artifact_size_mismatch:views" in result["blockers"]


def test_file_bound_stage_refuses_to_overwrite_an_input(tmp_path: Path) -> None:
    request, mapping, tracks, views = _fixture()
    paths = _write_stage_inputs(tmp_path, request, mapping, tracks, views)
    with pytest.raises(ValueError, match="output_path_must_not_overwrite_an_input"):
        run_semantic_gaussian_lifting_stage(
            request_path=paths["request"],
            gaussian_mapping_path=paths["gaussian_mapping"],
            track_registry_path=paths["track_registry"],
            views_path=paths["views"],
            output_path=paths["views"],
        )
