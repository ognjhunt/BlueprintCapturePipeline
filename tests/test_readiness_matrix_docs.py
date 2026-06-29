from __future__ import annotations

from pathlib import Path


def test_g1_render_seed_readiness_is_not_ready() -> None:
    text = Path("docs/READINESS_MATRIX.md").read_text(encoding="utf-8")
    row = next(
        line
        for line in text.splitlines()
        if line.startswith("| `Isaac/G1 kitchen-parity render seed` ")
    )

    assert "| `partial` |" in row
    assert "CPU/hermetic only" in row
    assert "no live GPU frame was produced on 2026-06-29" in row
    assert "`ready`" not in row
