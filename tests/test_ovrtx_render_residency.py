from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_ovrtx_preflight_worker.py"
)
_spec = importlib.util.spec_from_file_location("_ovrtx_worker", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
depth_is_resident = _module.depth_is_resident


def test_all_infinite_depth_is_not_resident() -> None:
    """The exact shape of a frame rendered before the Gaussians streamed in.

    A live run rendered at 8-20 s while the 98 MB ParticleField only finished
    loading at 311 s.  Every ray missed, so DistanceToCameraSD came back
    uniformly infinite on both cameras while every other check passed.
    """

    depth = np.full((384, 512, 1), np.inf, dtype=np.float32)

    resident, count = depth_is_resident(depth, np)

    assert resident is False
    assert count == 0


def test_any_positive_finite_sample_counts_as_resident() -> None:
    depth = np.full((384, 512, 1), np.inf, dtype=np.float32)
    depth[10, 10, 0] = 1.25

    resident, count = depth_is_resident(depth, np)

    assert resident is True
    assert count == 1

    dense = np.full((8, 8, 1), 2.0, dtype=np.float32)
    resident, count = depth_is_resident(dense, np)
    assert resident is True
    assert count == 64


def test_degenerate_depth_never_reads_as_resident() -> None:
    """Zero, negative, NaN and empty must not be mistaken for traced geometry."""

    for depth in (
        np.zeros((4, 4, 1), dtype=np.float32),
        np.full((4, 4, 1), -1.0, dtype=np.float32),
        np.full((4, 4, 1), np.nan, dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    ):
        resident, count = depth_is_resident(depth, np)
        assert resident is False
        assert count == 0

    assert depth_is_resident(None, np) == (False, 0)


def test_worker_waits_on_residency_before_the_quality_steps() -> None:
    """Waiting must happen before the frames that get retained, not after."""

    source = _SCRIPT.read_text(encoding="utf-8")

    wait = source.index("_await_render_residency(")
    quality = source.index("for _ in range(quality_steps):")
    assert wait < quality, "residency must be established before the retained render"

    # The wait must be wall-clock bounded, not a fixed step count: the warmup
    # advances simulation time and completes in seconds, while streaming is a
    # wall-clock cost.
    body = source[source.index("def _await_render_residency(") :]
    body = body[: body.index("def _map_array(")]
    assert "time.monotonic()" in body
    assert "timeout_seconds" in body

    # The outcome must be reported so a timed-out wait cannot pass silently.
    assert '"render_content_resident"' in source


@pytest.mark.parametrize("shape", [(4, 4, 1), (4, 4), (16,)])
def test_rule_accepts_any_aov_shape(shape: tuple[int, ...]) -> None:
    depth = np.full(shape, np.inf, dtype=np.float32)
    assert depth_is_resident(depth, np) == (False, 0)
    depth.flat[0] = 3.0
    resident, count = depth_is_resident(depth, np)
    assert resident is True
    assert count == 1
