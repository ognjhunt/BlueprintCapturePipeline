"""The depth filter makes at most DEPTH_FILTER_MAX_FACE_VIEWS comparisons; past them, evenly spaced views vote.

2026-11-24 sweep: 0.38 s a view per million faces, so a 400-view room with a 13 M-face mesh
screened for 35 minutes in the sweep's container, past the 20 minutes the Atlas tick gives the
work after the mesh's purchase, and the step was ended and retried with the filter every time.
"""
from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline import atlas_depth_splat, atlas_mesh_registration as registration
from blueprint_pipeline.atlas_depth_splat import posed_frame
from blueprint_pipeline.atlas_scene_estimates import scene_estimates
from tests.test_atlas_mesh_registration_memory import _case


def _view(frame_id: str, *, x: float, depth: float):
    return posed_frame(frame_id=frame_id, rgb=np.zeros((64, 64, 3), np.uint8),
                       depth=np.full((64, 64), depth, np.float32), confidence=np.ones((64, 64), np.float32),
                       camera={"intrinsics": {"fx": 32.0, "fy": 32.0, "cx": 32.0, "cy": 32.0, "width": 64, "height": 64},
                               "extrinsics": {"position": [x, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0],
                                              "coordinateSystem": "rub"}})


def _triangles(count: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    centres = np.stack([rng.uniform(-0.3, 0.3, count), rng.uniform(-0.3, 0.3, count),
                        rng.uniform(-2.4, -0.8, count)], axis=1)
    corners = np.array([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0], [0.0, 0.02, 0.0]])
    return centres[:, None, :] + corners[None]


# Twelve views whose surfaces lie at different depths: which faces any two contradict depends on who votes.
VIEWS = [_view(f"view-{index:02d}", x=-0.25 + 0.05 * index, depth=(1.2, 1.6, 2.0, 2.3)[index % 4])
         for index in range(12)]


@pytest.fixture
def counted(monkeypatch):
    """Every face-view comparison the filter makes (four samples a face, each one view)."""
    made = [0]
    seen = atlas_depth_splat._seen_ratio

    def counting(view, points):
        made[0] += len(points) // 4
        return seen(view, points)

    monkeypatch.setattr(atlas_depth_splat, "_seen_ratio", counting)
    return made


def test_past_the_bound_evenly_spaced_views_vote_and_the_report_names_them(monkeypatch, counted):
    triangles = _triangles(500)
    monkeypatch.setattr(registration, "DEPTH_FILTER_MAX_FACE_VIEWS", 4 * len(triangles) + 3)
    keep, report = registration.mesh_depth_free_space_filter(triangles, frames=VIEWS, absolute_tolerance=0.03)
    assert counted[0] <= registration.DEPTH_FILTER_MAX_FACE_VIEWS
    bound = report["voting_view_bound"]
    assert bound["voting_frame_ids"] == ["view-00", "view-04", "view-07", "view-11"]
    assert (bound["distinct_views"], bound["voting_views"]) == (12, 4)
    assert report["distinct_reference_count"] == 12 and len(report["references"]) == 12
    # The same faces as a filter given only those views.
    monkeypatch.setattr(registration, "DEPTH_FILTER_MAX_FACE_VIEWS", 10**12)
    voters = [VIEWS[index] for index in (0, 4, 7, 11)]
    alone, _ = registration.mesh_depth_free_space_filter(triangles, frames=voters, absolute_tolerance=0.03)
    everyone, _ = registration.mesh_depth_free_space_filter(triangles, frames=VIEWS, absolute_tolerance=0.03)
    assert np.array_equal(keep, alone) and not np.array_equal(keep, everyone)


def test_a_removal_always_has_its_two_views(monkeypatch, counted):
    triangles = _triangles(64)
    monkeypatch.setattr(registration, "DEPTH_FILTER_MAX_FACE_VIEWS", 1)
    _, report = registration.mesh_depth_free_space_filter(triangles, frames=VIEWS, absolute_tolerance=0.03)
    assert report["voting_view_bound"]["voting_frame_ids"] == ["view-00", "view-11"]
    assert counted[0] == 2 * len(triangles)


def test_within_the_bound_every_view_votes_and_the_report_is_as_before(counted):
    triangles = _triangles(200)
    _, report = registration.mesh_depth_free_space_filter(triangles, frames=VIEWS, absolute_tolerance=0.03)
    assert "voting_view_bound" not in report
    assert counted[0] == len(VIEWS) * len(triangles)


def test_a_bounded_filter_flags_the_collider_as_an_estimate(tmp_path, monkeypatch):
    case = _case(tmp_path, "depth_filtered")
    unbounded = registration.register_atlas_mesh(**case, output_root=tmp_path / "unbounded")
    assert "depth_filter_estimate" not in unbounded
    monkeypatch.setattr(registration, "DEPTH_FILTER_MAX_FACE_VIEWS", 1)
    case["reference_frames"] = case["reference_frames"] + [_view("third", x=0.0, depth=2.0)]
    record = registration.register_atlas_mesh(**case, output_root=tmp_path / "bounded")
    estimate = record["depth_filter_estimate"]
    assert estimate["code"] == registration.COLLIDER_ESTIMATE_FLAG
    assert estimate["reason"] == registration.DEPTH_FILTER_VIEWS_BOUNDED
    assert (estimate["voting_views"], estimate["distinct_views"]) == (2, 3)
    rows = scene_estimates(levelling={}, registration=record)
    assert rows == [{"code": registration.COLLIDER_ESTIMATE_FLAG, "source": "mesh_registration", "evidence": estimate}]
