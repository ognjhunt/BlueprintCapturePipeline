from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
    write_standard_3dgs_ply,
    write_standard_3dgs_ply_subset_exact,
)
from blueprint_pipeline.public_scene_replacement_occlusion import (
    CONTRIBUTION_SCHEMA,
    COVERAGE_SCHEMA,
    REQUEST_SCHEMA,
    ReplacementOcclusionError,
    build_replacement_occlusion_request,
    classify_gaussian_contributions,
    coverage_safe_ambiguous,
    materialize_bound_index_union_candidate,
    materialize_direct_evidence_expansion_candidate,
    materialize_replacement_occlusion_cutout,
    select_direct_calibration_evidence_expansion,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _source_splat(path: Path) -> Path:
    count = 6
    values = np.arange(count, dtype=np.float32)
    splat = SplatData(
        count=count,
        xyz=np.stack((values, values + 10, values + 20), axis=1),
        opacity=values + 30,
        f_dc=np.stack((values + 40, values + 50, values + 60), axis=1),
        scales=np.stack((values + 70, values + 80, values + 90), axis=1),
        quats=np.stack((values + 100, values + 110, values + 120, values + 130), axis=1),
        properties=(),
        sh_rest=None,
    )
    return write_standard_3dgs_ply(splat, path)


def test_exact_subset_writer_preserves_source_vertex_rows(tmp_path: Path) -> None:
    source = _source_splat(tmp_path / "source.ply")
    output = write_standard_3dgs_ply_subset_exact(
        source, tmp_path / "retained.ply", np.array([0, 2, 5], dtype=np.int64)
    )

    proof = verify_standard_3dgs_ply_subset_exact(source, output, [0, 2, 5])

    assert proof == {
        "source_vertex_count": 6,
        "retained_vertex_count": 3,
        "row_size_bytes": 56,
        "retained_rows_byte_exact": True,
        "retained_order_matches_source": True,
    }
    assert read_standard_3dgs_ply(output).xyz[:, 0].tolist() == [0.0, 2.0, 5.0]


def test_three_way_contribution_classification_and_coverage_conditioning() -> None:
    result = classify_gaussian_contributions(
        np.array([9, 1, 5, 6, 0, 8], dtype=float),
        np.array([1, 9, 5, 4, 0, 2], dtype=float),
        retained_max_foreground_fraction=0.2,
        owned_min_foreground_fraction=0.8,
        minimum_total_contribution=0.1,
    )
    ambiguous = np.flatnonzero(result["ambiguous"])
    safe = coverage_safe_ambiguous(
        ambiguous,
        np.array([2, 3], dtype=np.int64),
        np.ones((2, 2), dtype=float),
        np.array([[0.0, 0.2], [0.0, 0.0]], dtype=float),
        minimum_cell_visible_contribution=0.1,
        maximum_uncovered_fraction=0.0,
        maximum_uncovered_contribution=0.0,
    )

    assert np.flatnonzero(result["owned"]).tolist() == [0, 5]
    assert np.flatnonzero(result["retained"]).tolist() == [1, 4]
    assert ambiguous.tolist() == [2, 3]
    assert safe.tolist() == [True, False]


def _packet(
    tmp_path: Path,
    *,
    scene_id: str,
    instance_id: str,
    semantic_label: str,
    leave_residual: bool,
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    source = _source_splat(data / "inputs" / "scene_standard.ply")

    contribution_arrays = data / "evidence" / "contribution.npz"
    contribution_arrays.parent.mkdir(parents=True)
    np.savez(
        contribution_arrays,
        foreground_contribution=np.array([9, 1, 5, 6, 0, 8], dtype=np.float64),
        background_contribution=np.array([1, 9, 5, 4, 0, 2], dtype=np.float64),
    )
    scene = {
        "publisher_scene_id": scene_id,
        "target_instance_id": instance_id,
        "target_semantic_label": semantic_label,
    }
    contribution_manifest: dict[str, object] = {
        "schema_version": CONTRIBUTION_SCHEMA,
        "scene": scene,
        "source_standard_splat_sha256": _sha256(source),
        "method": {
            "name": "FlashSplat",
            "repository": "https://github.com/florinshen/FlashSplat",
            "commit": "a" * 40,
            "released_code_executed": True,
            "source_modified": False,
        },
        "arrays": _record(contribution_arrays, contribution_arrays.parent),
    }
    contribution_manifest["manifest_digest"] = canonical_digest(
        contribution_manifest, digest_field="manifest_digest"
    )
    contribution_manifest_path = data / "evidence" / "contribution_manifest.json"
    _write_json(contribution_manifest_path, contribution_manifest)

    camera_ids = ["external", "wrist"]
    angles = [0.0, 45.0]
    cells = [
        {
            "camera_id": camera,
            "commanded_door_angle_deg": angle,
            "readback_door_angle_deg": angle,
        }
        for camera in camera_ids
        for angle in angles
    ]
    removal_alpha = np.zeros((len(cells), 2, 2), dtype=np.float32)
    removal_alpha[:, 0, 0] = 1.0
    replacement_depth = np.full_like(removal_alpha, np.inf)
    replacement_depth[:, 0, 0] = 1.0
    if leave_residual:
        replacement_depth[-1, 0, 0] = np.inf
    coverage_arrays = data / "evidence" / "coverage.npz"
    np.savez(
        coverage_arrays,
        removal_alpha=removal_alpha,
        replacement_depth_m=replacement_depth,
        gaussian_indices=np.array([2, 3], dtype=np.int64),
        gaussian_visible_contribution=np.ones((len(cells), 2), dtype=np.float64),
        gaussian_uncovered_contribution=np.column_stack(
            (np.zeros(len(cells)), np.array([0.2, 0.0, 0.0, 0.0]))
        ),
    )
    coverage_manifest: dict[str, object] = {
        "schema_version": COVERAGE_SCHEMA,
        "scene": scene,
        "actual_mesh_depth_rasterized": True,
        "caller_supplied_coverage_mask": False,
        "renderer": {"name": "fixture_mesh_rasterizer", "version": "1"},
        "replacement_usd": {"sha256": "sha256:" + "b" * 64},
        "cells": cells,
        "arrays": _record(coverage_arrays, coverage_arrays.parent),
    }
    coverage_manifest["manifest_digest"] = canonical_digest(
        coverage_manifest, digest_field="manifest_digest"
    )
    coverage_manifest_path = data / "evidence" / "coverage_manifest.json"
    _write_json(coverage_manifest_path, coverage_manifest)

    request: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "frozen_before_cutout": True,
        "learned_policy_outcomes_observed": False,
        "scene": scene,
        "inputs": {
            "source_standard_splat": _record(source, data),
            "contribution_manifest": _record(contribution_manifest_path, data),
            "coverage_manifest": _record(coverage_manifest_path, data),
        },
        "policy": {
            "retained_max_foreground_fraction": 0.2,
            "owned_min_foreground_fraction": 0.8,
            "minimum_total_contribution": 0.1,
            "minimum_cell_visible_contribution": 0.1,
            "maximum_ambiguous_uncovered_fraction": 0.0,
            "maximum_ambiguous_uncovered_contribution": 0.0,
            "confident_removal_alpha_threshold": 0.5,
            "maximum_confident_uncovered_pixels_per_cell": 0,
            "maximum_residual_alpha_fraction_per_cell": 0.0,
            "door_angle_readback_tolerance_deg": 0.01,
            "required_camera_ids": camera_ids,
            "required_door_angles_deg": angles,
        },
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = repo / "requests" / "cutout.json"
    _write_json(request_path, request)
    return repo, data, request_path


@pytest.mark.parametrize(
    ("scene_id", "instance_id", "semantic_label", "leave_residual", "expected_status"),
    [
        ("840313", "160", "canned_beverage", False, "cutout_admitted_inpainting_not_required"),
        ("840796", "123", "refrigerator", True, "cutout_candidate_residual_measured"),
    ],
)
def test_materialize_scene_neutral_cutout_for_original_and_second_fixture(
    tmp_path: Path,
    scene_id: str,
    instance_id: str,
    semantic_label: str,
    leave_residual: bool,
    expected_status: str,
) -> None:
    repo, data, request_path = _packet(
        tmp_path,
        scene_id=scene_id,
        instance_id=instance_id,
        semantic_label=semantic_label,
        leave_residual=leave_residual,
    )

    receipt = materialize_replacement_occlusion_cutout(
        request_path=request_path,
        repo_root=repo,
        data_root=data,
        output_root=data / "output",
        receipt_output=repo / "receipts" / "cutout.json",
    )

    assert receipt["status"] == expected_status
    assert receipt["ownership"] == {
        **receipt["ownership"],
        "owned_count": 2,
        "retained_owned_evidence_count": 2,
        "ambiguous_count": 2,
        "coverage_safe_ambiguous_deleted_count": 1,
        "ambiguous_retained_count": 1,
        "total_deleted_count": 3,
        "total_retained_count": 3,
    }
    assert receipt["preservation"]["retained_rows_byte_exact"] is True
    assert np.load(data / "output" / "retained_source_indices.npy").tolist() == [1, 3, 4]
    assert read_standard_3dgs_ply(
        data / "output" / "scene_without_target_gaussians.ply"
    ).xyz[:, 0].tolist() == [1.0, 3.0, 4.0]
    assert receipt["source_collider_removed"] is False
    assert receipt["hybrid_render_qualified"] is False
    if leave_residual:
        assert receipt["inpainting_disposition"] == (
            "conditional_seam_ladder_required_for_measured_residual"
        )
        assert receipt["coverage"]["worst_confident_uncovered_pixel_count"] == 1
    else:
        assert receipt["inpainting_disposition"] == (
            "inpainting_not_required_by_replacement_occlusion"
        )
        assert receipt["coverage"]["worst_confident_uncovered_pixel_count"] == 0


def test_request_rejects_caller_asserted_inpainting_outcome() -> None:
    with pytest.raises(ReplacementOcclusionError) as exc:
        build_replacement_occlusion_request(
            {
                "schema_version": REQUEST_SCHEMA,
                "program_id": "arm-decision-proof-v1",
                "adp_item": "ADP-009B",
                "frozen_before_cutout": True,
                "learned_policy_outcomes_observed": False,
                "inpainting_not_required": True,
            }
        )
    assert "replacement_occlusion_caller_outcome_forbidden" in exc.value.codes


def test_direct_evidence_expansion_ignores_neighbor_score_and_outcomes() -> None:
    selected = select_direct_calibration_evidence_expansion(
        np.array([1, 2, 3, 4]),
        np.array([False, False, True, False, False]),
        np.array([0, 0, 0, 1, 0]),
        np.array([0, 2, 3, 4, 1]),
        np.array([0.0, 0.99, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.9, 1.0, 1.0, 1.0]),
        minimum_core_camera_count=2,
        minimum_core_fraction=0.9,
        minimum_geometry_score=0.5,
    )
    assert selected.tolist() == [1]


def test_direct_evidence_expansion_materializes_byte_exact_candidate(
    tmp_path: Path,
) -> None:
    source = _source_splat(tmp_path / "source.ply")
    arrays = {
        "owned": np.array([0], dtype=np.int64),
        "candidate": np.array([1, 2, 3, 4], dtype=np.int64),
        "protected": np.array([0, 0, 0, 1, 0, 0], dtype=np.int16),
        "core_count": np.array([0, 2, 1, 3, 2, 0], dtype=np.int16),
        "core_fraction": np.array([0.0, 0.99, 1.0, 1.0, 0.5, 0.0]),
        "geometry": np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    }
    paths = {}
    for name, values in arrays.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        paths[name] = path
    receipt = materialize_direct_evidence_expansion_candidate(
        source_standard_splat_path=source,
        owned_indices_path=paths["owned"],
        candidate_indices_path=paths["candidate"],
        protected_camera_count_path=paths["protected"],
        core_camera_count_path=paths["core_count"],
        core_fraction_path=paths["core_fraction"],
        geometry_score_path=paths["geometry"],
        output_root=tmp_path / "output",
        minimum_core_camera_count=2,
        minimum_core_fraction=0.9,
        minimum_geometry_score=0.5,
    )
    assert receipt["counts"] == {
        "source": 6,
        "owned": 1,
        "direct_evidence_expansion": 1,
        "deleted_total": 2,
        "retained_total": 4,
    }
    assert np.load(tmp_path / "output/deleted_source_indices.npy").tolist() == [0, 1]
    assert receipt["preservation"]["retained_rows_byte_exact"] is True


def test_bound_index_union_preserves_all_unselected_rows(tmp_path: Path) -> None:
    source = _source_splat(tmp_path / "source.ply")
    required = tmp_path / "required.npy"
    registered = tmp_path / "registered.npy"
    np.save(required, np.array([0, 4], dtype=np.int64), allow_pickle=False)
    np.save(registered, np.array([1, 4, 5], dtype=np.int64), allow_pickle=False)

    receipt = materialize_bound_index_union_candidate(
        source_standard_splat_path=source,
        required_deletion_indices_path=required,
        registered_volume_indices_path=registered,
        output_root=tmp_path / "output",
    )

    assert receipt["counts"] == {
        "source": 6,
        "required_deletion": 2,
        "registered_volume": 3,
        "registered_volume_only": 2,
        "deleted_total": 4,
        "retained_total": 2,
    }
    assert np.load(tmp_path / "output/deleted_source_indices.npy").tolist() == [0, 1, 4, 5]
    assert np.load(tmp_path / "output/retained_source_indices.npy").tolist() == [2, 3]
    assert receipt["preservation"]["retained_rows_byte_exact"] is True
