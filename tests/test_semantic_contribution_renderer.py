"""Tests for executable source-mask to standard-3DGS contribution rendering."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.scene_placement.semantic_contribution_renderer import (
    METHOD_ID,
    METHOD_VERSION,
    PROJECTION_CONVENTION,
    REQUEST_SCHEMA_VERSION,
    render_semantic_contributions,
    renderer_runtime_digest,
)
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    CONTRIBUTION_SEMANTICS,
    canonical_json_digest,
    lift_semantic_masks_to_gaussians,
)
from blueprint_pipeline.semantic_contribution_renderer_stage import (
    run_semantic_contribution_renderer_stage,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _splat(count: int = 1) -> SplatData:
    return SplatData(
        count=count,
        xyz=np.asarray([[0.0, 0.0, 2.0 + index] for index in range(count)], dtype=np.float32),
        opacity=np.asarray([0.0] * count, dtype=np.float32),
        f_dc=np.zeros((count, 3), dtype=np.float32),
        scales=np.full((count, 3), math.log(0.05), dtype=np.float32),
        quats=np.asarray([[1.0, 0.0, 0.0, 0.0]] * count, dtype=np.float32),
        properties=(),
    )


def _camera(translation_x: float = 0.0, *, width: int = 4, height: int = 4) -> dict:
    camera_to_world = [
        1.0,
        0.0,
        0.0,
        translation_x,
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
    ]
    return {
        "intrinsics": [20.0, 20.0, width / 2.0, height / 2.0],
        "camera_to_world": camera_to_world,
        "coordinate_frame": "analysis_splat_z_up_meters",
        "projection_convention": PROJECTION_CONVENTION,
        "distortion_status": "rectified_none",
    }


def _fixture(
    *, count: int = 1, width: int = 4, height: int = 4
) -> tuple[dict, SplatData, list[dict], dict, list[dict]]:
    splat = _splat(count)
    mapping = [
        {"gaussian_id": index, "source_index": index, "source_class": "observed"}
        for index in range(count)
    ]
    cameras = []
    frames = []
    frame_masks = []
    for index, translation in enumerate((0.0, 0.05), start=1):
        frame_id = f"frame-{index}"
        camera = _camera(translation, width=width, height=height)
        camera_digest = canonical_json_digest(camera)
        source_digest = "sha256:" + str(index) * 64
        cameras.append(
            {
                "source_frame_id": frame_id,
                "camera_record": camera,
                "camera_record_digest": camera_digest,
            }
        )
        frames.append(
            {
                "source_frame_id": frame_id,
                "source_frame_digest": source_digest,
                "retained_video_digest": SHA_B,
                "decoded_pts_seconds": float(index),
                "sync_map_row_digest": SHA_C,
                "camera_record_digest": camera_digest,
                "encoder_retained": True,
            }
        )
        frame_masks.append(
            {
                "source_frame_id": frame_id,
                "source_frame_digest": source_digest,
                "decoded_pts_seconds": float(index),
                "camera_record_digest": camera_digest,
                "width": width,
                "height": height,
                "mask_encoding": "sparse_probability_rle.v1",
                "track_masks": [
                    {
                        "track_id": "track-box-1",
                        "runs": [
                            {
                                "start": 0,
                                "length": width * height,
                                "probability": 1.0,
                            }
                        ],
                    }
                ],
            }
        )
    track_registry = [
        {
            "track_id": "track-box-1",
            "label": "box",
            "label_source": "model_inferred",
            "mask_model_digest": SHA_D,
            "track_evidence_digest": SHA_C,
            "supporting_frame_ids": ["frame-1", "frame-2"],
            "observation_count": 2,
            "semantic_authority": "inferred_candidate",
        }
    ]
    source_bindings = {
        "capture_digest": SHA_A,
        "retained_video_digest": SHA_B,
        "camera_solution_digest": SHA_C,
        "frame_registry_digest": canonical_json_digest(frames),
        "track_registry_digest": canonical_json_digest(track_registry),
        "frame_masks_digest": canonical_json_digest(frame_masks),
    }
    source_tracks = {
        "schema_version": "semantic_source_track_import_result.v1",
        "status": "completed",
        "bindings": source_bindings,
        "provider_profile": {"method_id": "test"},
        "track_registry": track_registry,
        "frame_masks": frame_masks,
        "blockers": [],
        "warnings": [],
        "claim_ceiling": "source_bound_2d_mask_tracks_only",
        "directly_observed_object_fact": False,
        "canonical_object_geometry": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "generated_regions_can_upgrade_claims": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    source_tracks["result_digest"] = canonical_json_digest(source_tracks)
    profile = {
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "runtime_digest": renderer_runtime_digest(),
        "contribution_semantics": CONTRIBUTION_SEMANTICS,
        "projection_convention": PROJECTION_CONVENTION,
        "exact_gaussian_ids": True,
        "deterministic": True,
        "minimum_alpha": 1.0e-8,
        "minimum_emitted_weight": 1.0e-10,
        "sigma_extent": 3.0,
        "covariance_regularization_pixels_squared": 0.01,
        "near_plane_meters": 0.01,
        "max_projected_pixel_gaussian_pairs": 100_000,
        "max_contributions_per_pixel": 16,
    }
    bindings = {
        "capture_digest": SHA_A,
        "retained_video_digest": SHA_B,
        "reconstruction_digest": SHA_D,
        "analysis_splat_digest": SHA_A,
        "camera_solution_digest": SHA_C,
        "gaussian_mapping_digest": canonical_json_digest(mapping),
        "frame_registry_digest": canonical_json_digest(frames),
        "source_track_result_digest": source_tracks["result_digest"],
        "track_registry_digest": source_bindings["track_registry_digest"],
        "frame_masks_digest": source_bindings["frame_masks_digest"],
        "camera_records_digest": canonical_json_digest(cameras),
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "bindings": bindings,
        "frame_registry": frames,
        "renderer_profile": profile,
        "world": {"up_axis": "Z", "units": "meters", "scale_verified": True},
        "qualification": {
            "min_track_views": 2,
            "min_gaussian_views": 2,
            "min_view_foreground_contribution": 0.05,
            "min_gaussian_view_foreground_contribution": 0.05,
            "min_gaussian_total_contribution": 0.1,
            "foreground_probability_threshold": 0.7,
            "min_angular_diversity_degrees": 0.0,
        },
    }
    return request, splat, mapping, source_tracks, cameras


def test_renders_source_masks_to_exact_gaussian_weights_and_lifts_them() -> None:
    request, splat, mapping, source_tracks, cameras = _fixture()

    result = render_semantic_contributions(
        request,
        splat=splat,
        gaussian_mapping=mapping,
        source_tracks=source_tracks,
        camera_records=cameras,
    )
    repeated = render_semantic_contributions(
        request,
        splat=splat,
        gaussian_mapping=mapping,
        source_tracks=source_tracks,
        camera_records=cameras,
    )

    assert result["status"] == "completed"
    assert repeated == result
    assert result["reference_contribution_render_completed"] is True
    assert result["production_large_scene_ready"] is False
    assert result["comparative_policy_ranking_verdict"] == "thesis_not_supported"
    assert len(result["views"]) == 2
    assert any(row["contributions"] for row in result["views"][0]["pixels"])
    lifting = lift_semantic_masks_to_gaussians(
        result["lifting_request"],
        gaussian_mapping=result["gaussian_mapping"],
        track_registry=result["track_registry"],
        views=result["views"],
    )
    assert lifting["status"] == "completed"
    assert lifting["tracks"][0]["selected_gaussian_ids"] == [0]
    assert lifting["metric_box_ready"] is False
    assert lifting["physics_ready"] is False


def test_front_to_back_weights_are_transmittance_times_alpha() -> None:
    request, splat, mapping, source_tracks, cameras = _fixture(count=2, width=1, height=1)
    splat.xyz[1] = [0.0, 0.0, 3.0]

    result = render_semantic_contributions(
        request,
        splat=splat,
        gaussian_mapping=mapping,
        source_tracks=source_tracks,
        camera_records=cameras,
    )

    assert result["status"] == "completed"
    rows = result["views"][0]["pixels"][0]["contributions"]
    assert [row["gaussian_id"] for row in rows] == [0, 1]
    assert rows[0]["weight"] == pytest.approx(0.5, abs=1e-8)
    assert rows[1]["weight"] == pytest.approx(0.25, abs=1e-8)
    assert sum(row["weight"] for row in rows) <= 1.0


def test_rejects_unrectified_camera_and_runtime_profile_drift() -> None:
    request, splat, mapping, source_tracks, cameras = _fixture()
    cameras[0]["camera_record"]["distortion_status"] = "unknown"
    cameras[0]["camera_record_digest"] = canonical_json_digest(cameras[0]["camera_record"])
    request["bindings"]["camera_records_digest"] = canonical_json_digest(cameras)
    request["renderer_profile"]["runtime_digest"] = SHA_D

    result = render_semantic_contributions(
        request,
        splat=splat,
        gaussian_mapping=mapping,
        source_tracks=source_tracks,
        camera_records=cameras,
    )

    assert result["status"] == "blocked"
    assert "renderer_profile_mismatch:runtime_digest" in result["blockers"]
    assert "camera_distortion_must_be_rectified:frame-1" in result["blockers"]


def test_rejects_qualification_weakening_and_stale_frame_mask_binding() -> None:
    request, splat, mapping, source_tracks, cameras = _fixture()
    request["renderer_profile"]["minimum_emitted_weight"] = 0.25
    source_tracks["frame_masks"][0]["camera_record_digest"] = SHA_D
    source_tracks["bindings"]["frame_masks_digest"] = canonical_json_digest(
        source_tracks["frame_masks"]
    )
    request["bindings"]["frame_masks_digest"] = source_tracks["bindings"][
        "frame_masks_digest"
    ]
    source_tracks["result_digest"] = canonical_json_digest(
        {key: value for key, value in source_tracks.items() if key != "result_digest"}
    )
    request["bindings"]["source_track_result_digest"] = source_tracks["result_digest"]

    result = render_semantic_contributions(
        request,
        splat=splat,
        gaussian_mapping=mapping,
        source_tracks=source_tracks,
        camera_records=cameras,
    )

    assert result["status"] == "blocked"
    assert (
        "renderer_profile_range_invalid:minimum_emitted_weight" in result["blockers"]
    )
    assert (
        "frame_mask_binding_mismatch:frame-1:camera_record_digest" in result["blockers"]
    )


def test_abstains_before_unbounded_projected_work() -> None:
    request, splat, mapping, source_tracks, cameras = _fixture()
    request["renderer_profile"]["max_projected_pixel_gaussian_pairs"] = 1

    result = render_semantic_contributions(
        request,
        splat=splat,
        gaussian_mapping=mapping,
        source_tracks=source_tracks,
        camera_records=cameras,
    )

    assert result["status"] == "blocked"
    assert any(
        blocker.endswith("projected_pixel_gaussian_pair_limit_exceeded")
        for blocker in result["blockers"]
    )


def _write_stage_inputs(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    request, splat, mapping, source_tracks, cameras = _fixture()
    paths = {
        "analysis_splat": tmp_path / "analysis.ply",
        "gaussian_mapping": tmp_path / "mapping.json",
        "source_tracks": tmp_path / "source-tracks.json",
        "camera_records": tmp_path / "cameras.json",
        "request": tmp_path / "request.json",
        "output": tmp_path / "result.json",
    }
    write_standard_3dgs_ply(splat, paths["analysis_splat"])
    request["bindings"]["analysis_splat_digest"] = "sha256:" + hashlib.sha256(
        paths["analysis_splat"].read_bytes()
    ).hexdigest()
    payloads = {
        "gaussian_mapping": mapping,
        "source_tracks": source_tracks,
        "camera_records": cameras,
    }
    for name, payload in payloads.items():
        paths[name].write_text(json.dumps(payload), encoding="utf-8")
    request["input_artifacts"] = {}
    for name in ("analysis_splat", "gaussian_mapping", "source_tracks", "camera_records"):
        request["input_artifacts"][name] = {
            "sha256": "sha256:" + hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "size_bytes": paths[name].stat().st_size,
        }
    paths["request"].write_text(json.dumps(request), encoding="utf-8")
    return paths, request


def test_file_stage_verifies_standard_ply_and_emits_replayable_receipt(tmp_path: Path) -> None:
    paths, _request = _write_stage_inputs(tmp_path)

    result = run_semantic_contribution_renderer_stage(
        request_path=paths["request"],
        analysis_splat_path=paths["analysis_splat"],
        gaussian_mapping_path=paths["gaussian_mapping"],
        source_tracks_path=paths["source_tracks"],
        camera_records_path=paths["camera_records"],
        output_path=paths["output"],
    )

    assert result["status"] == "completed"
    assert result["stage_input_artifacts"]["analysis_splat"]["sha256"].startswith(
        "sha256:"
    )
    assert result["transport_profile"] == "bounded_canonical_json_reference.v1"
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == result


def test_file_stage_fails_closed_on_stale_splat_digest(tmp_path: Path) -> None:
    paths, request = _write_stage_inputs(tmp_path)
    request["bindings"]["analysis_splat_digest"] = SHA_D
    paths["request"].write_text(json.dumps(request), encoding="utf-8")

    result = run_semantic_contribution_renderer_stage(
        request_path=paths["request"],
        analysis_splat_path=paths["analysis_splat"],
        gaussian_mapping_path=paths["gaussian_mapping"],
        source_tracks_path=paths["source_tracks"],
        camera_records_path=paths["camera_records"],
        output_path=paths["output"],
    )

    assert result["status"] == "blocked"
    assert "analysis_splat_binding_digest_mismatch" in result["blockers"]


def test_file_stage_fails_closed_on_tampered_camera_artifact(tmp_path: Path) -> None:
    paths, _request = _write_stage_inputs(tmp_path)
    paths["camera_records"].write_text(
        paths["camera_records"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    result = run_semantic_contribution_renderer_stage(
        request_path=paths["request"],
        analysis_splat_path=paths["analysis_splat"],
        gaussian_mapping_path=paths["gaussian_mapping"],
        source_tracks_path=paths["source_tracks"],
        camera_records_path=paths["camera_records"],
        output_path=paths["output"],
    )

    assert result["status"] == "blocked"
    assert "input_artifact_size_mismatch:camera_records" in result["blockers"]
    assert "input_artifact_sha256_mismatch:camera_records" in result["blockers"]


def test_file_stage_rejects_output_overwriting_an_input(tmp_path: Path) -> None:
    paths, _request = _write_stage_inputs(tmp_path)
    with pytest.raises(ValueError, match="output_path_must_not_overwrite_an_input"):
        run_semantic_contribution_renderer_stage(
            request_path=paths["request"],
            analysis_splat_path=paths["analysis_splat"],
            gaussian_mapping_path=paths["gaussian_mapping"],
            source_tracks_path=paths["source_tracks"],
            camera_records_path=paths["camera_records"],
            output_path=paths["camera_records"],
        )
