"""The look check's CPU Gaussian render: compositing order, coverage, and bounded memory."""
from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest

from blueprint_pipeline import atlas_splat_render as render_module
from blueprint_pipeline.gaussian_splat_decode import SplatData

C0 = render_module.SH_C0
K = {"fx": 40.0, "fy": 40.0, "cx": 16.5, "cy": 16.5, "width": 33, "height": 33}
CAMERA = {"position": np.zeros(3), "rotation": np.eye(3), "intrinsics": K}


def _splat(xyz, rgb, opacity, scale):
    count = len(xyz)
    opacity = np.clip(np.asarray(opacity, float), 1e-6, 1 - 1e-6)
    return SplatData(count=count, xyz=np.asarray(xyz, np.float32),
                     opacity=np.log(opacity / (1 - opacity)).astype(np.float32),
                     f_dc=((np.asarray(rgb, float) - .5) / C0).astype(np.float32),
                     scales=np.log(np.broadcast_to(np.reshape(scale, (-1, 1)), (count, 3))).astype(np.float32),
                     quats=np.tile([1.0, 0, 0, 0], (count, 1)).astype(np.float32), properties=())


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_nearer_gaussians_composite_first_whatever_their_order_in_the_splat(order):
    # Red at 1 unit, green at 2 units, both 0.8 opaque on the optical axis, wide enough to be flat at the centre.
    xyz, rgb = [[0, 0, -1.0], [0, 0, -2.0]], [[1, 0, 0], [0, 1, 0]]
    splat = _splat([xyz[i] for i in order], [rgb[i] for i in order], [.8, .8], [[.5, 1.0][i] for i in order])
    out = render_module.render(splat, CAMERA, 33, workers=1)
    centre = out["rgb"][16, 16]
    # Front to back: C = a1 c1 + (1 - a1) a2 c2; alpha = 1 - (1 - a1)(1 - a2).
    assert centre == pytest.approx([.8, .2 * .8, 0], abs=2e-3)
    assert out["alpha"][16, 16] == pytest.approx(1 - .2 * .2, abs=2e-3)
    assert out["depth"][16, 16] == pytest.approx((.8 * 1 + .16 * 2) / .96, rel=1e-3)


def test_a_gaussians_coverage_is_its_projected_footprint_and_nothing_lands_outside_it():
    sigma_world, depth = .1, 2.0
    splat = _splat([[0, 0, -depth]], [[.5, .5, .5]], [.6], [sigma_world])
    out = render_module.render(splat, {**CAMERA, "intrinsics": {**K, "width": 65, "height": 65, "cx": 32.5,
                                                                 "cy": 32.5}}, 65, workers=1)
    variance = (K["fx"] * sigma_world / depth) ** 2 + render_module.LOW_PASS
    assert out["alpha"].sum() == pytest.approx(.6 * 2 * math.pi * variance, rel=.04)
    ys, xs = np.nonzero(out["alpha"] > 0)
    reach = render_module.SIGMA_CUT * math.sqrt(variance) + 1
    assert np.abs(xs + .5 - 32.5).max() <= reach and np.abs(ys + .5 - 32.5).max() <= reach
    # Behind the camera nothing is drawn.
    behind = render_module.render(_splat([[0, 0, depth]], [[1, 1, 1]], [.9], [sigma_world]), CAMERA, 33, workers=1)
    assert behind["alpha"].max() == 0 and behind["visible_gaussians"] == 0


def test_tiles_threads_and_composite_chunks_do_not_change_the_image(monkeypatch):
    rng = np.random.default_rng(3)
    n = 3000
    splat = _splat(np.column_stack([rng.uniform(-.6, .6, n), rng.uniform(-.6, .6, n), -rng.uniform(1, 3, n)]),
                   rng.random((n, 3)), rng.uniform(.05, .9, n), .02)
    one = render_module.render(splat, CAMERA, 33, workers=1)
    monkeypatch.setattr(render_module, "COMPOSITE_CHUNK", 7)
    monkeypatch.setattr(render_module, "PROJECT_CHUNK", 500)
    other = render_module.render(splat, CAMERA, 33, workers=4)
    assert np.allclose(one["rgb"], other["rgb"], atol=1e-5) and np.allclose(one["alpha"], other["alpha"], atol=1e-5)


def test_view_dependent_colour_follows_the_spherical_harmonics_when_the_splat_has_them():
    splat = _splat([[0, 0, -1.0]], [[.5, .5, .5]], [.99], [.5])
    base = render_module.render(splat, CAMERA, 33, workers=1)["rgb"][16, 16]
    rest = np.zeros((1, 9), np.float32)
    rest[0, 1] = rest[0, 4] = rest[0, 7] = .5  # the z band of every channel; the view looks down -z
    splat.sh_rest = rest
    tinted = render_module.render(splat, CAMERA, 33, workers=1)
    assert tinted["colour_basis"] == "spherical_harmonics"
    assert tinted["rgb"][16, 16] == pytest.approx(base - .99 * .5 * 0.4886025119029199, abs=3e-3)


def test_the_render_working_memory_is_bounded_by_the_visible_rows_not_the_pixel_gaussian_pairs():
    rng = np.random.default_rng(0)
    n = 200_000
    splat = _splat(np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-1.7, 1.7, n), -rng.uniform(1, 7, n)]),
                   rng.random((n, 3)), rng.uniform(.01, .3, n), .03)
    camera = {**CAMERA, "intrinsics": {"fx": 607.45, "fy": 607.45, "cx": 640, "cy": 360, "width": 1280,
                                       "height": 720}}
    tracemalloc.start()
    try:
        out = render_module.render(splat, camera, 640)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert out["visible_gaussians"] > n // 2
    # About 0.6 KB a Gaussian at most (its kept row plus one projection chunk's temporaries amortised):
    # a 2M-Gaussian splat stays well under 2 GB.
    assert peak < 64 * 2**20 + 600 * n
