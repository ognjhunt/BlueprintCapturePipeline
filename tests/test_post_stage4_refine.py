"""Tests for post-Stage-4 gap analysis and view-repair helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_script_module(script_name: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_candidate_views_enforces_parallax_diversity() -> None:
    module = _load_script_module("post_stage4_gap_analyzer.py", "post_stage4_gap_analyzer_test")

    candidates = [
        {
            "id": "a",
            "score": 10.0,
            "hole_ratio": 0.2,
            "cluster_count": 10,
            "sharpness": 5.0,
            "parallax_to_nearest_captured_deg": 10.0,
            "view_dir": [0.0, 0.0, 1.0],
        },
        {
            "id": "b",
            "score": 9.5,
            "hole_ratio": 0.19,
            "cluster_count": 9,
            "sharpness": 5.0,
            "parallax_to_nearest_captured_deg": 10.0,
            "view_dir": [0.0, 0.0, 1.0],  # same direction as a -> should be filtered
        },
        {
            "id": "c",
            "score": 9.0,
            "hole_ratio": 0.18,
            "cluster_count": 8,
            "sharpness": 5.0,
            "parallax_to_nearest_captured_deg": 10.0,
            "view_dir": [1.0, 0.0, 0.0],
        },
    ]

    selected = module.rank_candidate_views(candidates, max_candidates=2, min_parallax_deg=7.0)
    selected_ids = [row["id"] for row in selected]
    assert selected_ids == ["a", "c"]


def test_build_repair_mask_marks_dark_or_alpha_holes() -> None:
    module = _load_script_module("post_stage4_view_repair.py", "post_stage4_view_repair_test")

    rgb = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[10, 10, 10], [200, 50, 50]],
        ],
        dtype=np.uint8,
    )
    alpha = np.array(
        [
            [255, 255],
            [255, 0],
        ],
        dtype=np.uint8,
    )

    mask = module.build_repair_mask(rgb, alpha=alpha)
    assert mask.shape == (2, 2)
    assert bool(mask[0, 0]) is True
    assert bool(mask[0, 1]) is False
    assert bool(mask[1, 1]) is True  # forced by alpha=0


def test_apply_acceptance_gate_filters_bad_views() -> None:
    module = _load_script_module("post_stage4_view_repair.py", "post_stage4_view_repair_gate_test")

    rows = [
        {
            "id": "ok",
            "cross_view_reprojection_error_px": 2.0,
            "photometric_drift_outside_mask": 0.05,
        },
        {
            "id": "bad_reproj",
            "cross_view_reprojection_error_px": 3.0,
            "photometric_drift_outside_mask": 0.05,
        },
        {
            "id": "bad_drift",
            "cross_view_reprojection_error_px": 2.0,
            "photometric_drift_outside_mask": 0.10,
        },
    ]

    accepted, rejected = module.apply_acceptance_gate(
        rows,
        max_reprojection_error_px=2.5,
        max_photometric_drift=0.08,
    )

    assert [row["id"] for row in accepted] == ["ok"]
    rejected_ids = {row["id"] for row in rejected}
    assert rejected_ids == {"bad_reproj", "bad_drift"}
