"""Real STEP round trips for the separately provisioned build123d/OCP runtime.

The hermetic readback-contract tests live in test_astra_cad_skill_runtime.py.
Selecting this integration lane requires the actual kernel; absence is a failure,
not an importorskip that silently weakens a full-lane result.
"""
import pytest

from blueprint_pipeline import astra_cad_skill_runtime as runtime

pytestmark = pytest.mark.external_runtime


def test_real_step_roundtrip_preserves_unrounded_dimensions(tmp_path):
    import build123d

    step = tmp_path / "box.step"
    build123d.export_step(build123d.Box(12.34567, 20, 30), str(step))
    assert runtime._read_step(step, (12.34567, 20, 30), 0.000001)["passed"]
    assert not runtime._read_step(step, (12.35, 20, 30), 0.000001)["passed"]


def test_real_step_roundtrip_counts_disconnected_solids(tmp_path):
    import build123d

    shape = build123d.Compound(children=[build123d.Box(1, 1, 1),
        build123d.Pos(2, 0, 0) * build123d.Box(1, 1, 1)])
    step = tmp_path / "disconnected.step"
    build123d.export_step(shape, str(step))
    result = runtime._read_step(step, (3, 1, 1), 0.01)
    assert result["solid_count"] == 2 and result["valid"] and not result["passed"]
