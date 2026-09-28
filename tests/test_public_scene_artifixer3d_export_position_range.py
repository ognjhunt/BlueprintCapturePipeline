"""Gaussian positions must be representable before the NuRec export runs.

A production run (scene 839873, 2026-08-29) spent ~61 minutes of paid GPU on the
Artifixer stage and then wrote a corrupt asset: the pinned NuRec template casts
``.gaussians_nodes.gaussians.positions`` to a narrow dtype, so diverged
far-field gaussians overflowed to ``inf``.  The exporter still produced a
``repaired_scene.usdz``; only the downstream frame-alignment gate caught it, as
``native_task_appearance_nurec_positions_invalid``, after the spend.

The export adapter now measures representability itself and refuses any
unrepresentable floater before either native exporter runs.  It also compares
the trained positions and scales with the immutable retained field so finite
but scene-destroying divergence cannot pass merely because float16 can store it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _load_real_provider_runner():
    path = Path(__file__).parents[1] / "scripts/public_scene_artifixer3d_runner.py"
    spec = importlib.util.spec_from_file_location("_exact_artifixer_export_position_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ValueTensor:
    """A checkpoint tensor carrying real values through the exporter seam."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    def detach(self) -> "_ValueTensor":
        return self

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return self._array.astype(dtype) if dtype is not None else self._array

    def __getitem__(self, key: Any) -> "_ValueTensor":
        return _ValueTensor(self._array[key])


def _checkpoint(positions: np.ndarray) -> dict[str, Any]:
    count = int(positions.shape[0])
    return {
        "positions": _ValueTensor(positions),
        "rotation": _ValueTensor(np.zeros((count, 4), dtype=np.float32)),
        "scale": _ValueTensor(np.full((count, 3), -3.0, dtype=np.float32)),
        "density": _ValueTensor(np.zeros((count, 1), dtype=np.float32)),
        "features_albedo": _ValueTensor(np.zeros((count, 3), dtype=np.float32)),
        "features_specular": _ValueTensor(np.zeros((count, 0), dtype=np.float32)),
        "max_n_features": 0,
        "n_active_features": 0,
    }


def _room(count: int) -> np.ndarray:
    """Metric room-scale gaussians: scene 839873 is sealed in meters."""

    return np.linspace(-3.0, 3.0, count * 3, dtype=np.float64).reshape(count, 3)


def _reference(positions: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        count=int(positions.shape[0]),
        xyz=np.asarray(positions, dtype=np.float32),
        scales=np.full_like(positions, -3.0, dtype=np.float32),
        quats=np.zeros((positions.shape[0], 4), dtype=np.float32),
    )


def _model(runner: Any, checkpoint: dict[str, Any], reference: SimpleNamespace):
    return runner._CheckpointExportModel(
        checkpoint,
        reference_splat=reference,
        geometry_policy=runner.RETAINED_GEOMETRY_POLICY,
    )


def test_representable_room_positions_export_every_gaussian() -> None:
    runner = _load_real_provider_runner()
    positions = _room(1000)
    model = _model(runner, _checkpoint(positions), _reference(positions))
    assert model.exported_gaussian_count == 1000
    assert model.unrepresentable_position_count == 0
    assert int(model.positions.shape[0]) == 1000


def test_one_unrepresentable_floater_refuses_the_export() -> None:
    """Learned tensors are evidence: never mutate one bad row and continue."""

    runner = _load_real_provider_runner()
    positions = _room(1000)
    positions[7] = [153532.48, -58422.664, 76406.766]
    with pytest.raises(ValueError, match="artifixer3d_native_export_positions_unrepresentable"):
        _model(runner, _checkpoint(positions), _reference(_room(1000)))


def test_non_finite_positions_are_treated_as_unrepresentable() -> None:
    # Sized like a real reconstruction (scene 839873 retained ~1.03M
    # gaussians), so two diverged floaters stay inside the prunable bound.
    runner = _load_real_provider_runner()
    positions = _room(4000)
    positions[3] = [np.inf, 0.0, 0.0]
    positions[4] = [0.0, np.nan, 0.0]
    with pytest.raises(ValueError, match="artifixer3d_native_export_positions_unrepresentable"):
        _model(runner, _checkpoint(positions), _reference(_room(4000)))


def test_widespread_unrepresentable_positions_fail_closed() -> None:
    """A scale defect is not a floater: refuse rather than gut the scene."""

    runner = _load_real_provider_runner()
    positions = _room(1000)
    positions[:200] = 1.0e6
    with pytest.raises(ValueError) as excinfo:
        _model(runner, _checkpoint(positions), _reference(_room(1000)))
    assert "artifixer3d_native_export_positions_unrepresentable" in str(excinfo.value)


def test_all_positions_unrepresentable_fail_closed() -> None:
    runner = _load_real_provider_runner()
    positions = np.full((16, 3), np.inf, dtype=np.float64)
    with pytest.raises(ValueError) as excinfo:
        _model(runner, _checkpoint(positions), _reference(_room(16)))
    assert "artifixer3d_native_export_positions_unrepresentable" in str(excinfo.value)


def test_limit_matches_the_pinned_nurec_cast_range() -> None:
    """The bound is the template's cast range, not an arbitrary number."""

    runner = _load_real_provider_runner()
    assert runner.NATIVE_EXPORT_POSITION_MAGNITUDE_LIMIT == float(np.finfo(np.float16).max)


def test_representable_but_far_field_position_drift_is_refused() -> None:
    runner = _load_real_provider_runner()
    reference = _room(4000)
    positions = reference.copy()
    positions[7] = [8_000.0, -7_000.0, 900.0]

    with pytest.raises(ValueError, match="retained_geometry_mismatch:positions"):
        _model(runner, _checkpoint(positions), _reference(reference))


def test_representable_but_massive_kernel_scale_is_refused() -> None:
    runner = _load_real_provider_runner()
    reference = _room(4000)
    checkpoint = _checkpoint(reference)
    checkpoint["scale"]._array[7] = [6.88, 0.0, 0.0]

    with pytest.raises(ValueError, match="retained_geometry_mismatch:scale"):
        _model(runner, checkpoint, _reference(reference))
