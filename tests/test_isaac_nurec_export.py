"""Tests for blueprint_pipeline.isaac_nurec_export and the GPU runner CLI contract.

The transcode/render itself needs the GPU worker (3dgrut + Isaac RTX); here we pin the
fail-closed contract and that the runner is syntactically valid with a sane CLI.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.isaac_nurec_export import (
    convert_ply_to_isaac_usd,
    threedgrut_available,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_isaac_splat_nurec_render.py"


def test_threedgrut_unavailable_locally() -> None:
    # 3dgrut is a GPU/CUDA package; it is not installed in this environment.
    assert threedgrut_available(sys.executable) is False


def test_convert_blocked_unsupported_format(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply\n")
    out = convert_ply_to_isaac_usd(ply, tmp_path / "o.usdz", fmt="bogus", python=sys.executable)
    assert out["status"] == "blocked"
    assert "unsupported_isaac_splat_format" in out["blockers"]


def test_convert_blocked_missing_ply(tmp_path: Path) -> None:
    out = convert_ply_to_isaac_usd(tmp_path / "nope.ply", tmp_path / "o.usdz", python=sys.executable)
    assert out["status"] == "blocked"
    assert "standard_ply_missing" in out["blockers"]


def test_convert_blocked_threedgrut_unavailable(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    out = convert_ply_to_isaac_usd(ply, tmp_path / "o.usdz", python=sys.executable)
    assert out["status"] == "blocked"
    assert "threedgrut_unavailable" in out["blockers"]
    assert "remediation" in out  # tells the operator how to enable the GPU transcode


def test_runner_py_compiles() -> None:
    proc = subprocess.run([sys.executable, "-m", "py_compile", str(RUNNER)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_runner_help_exits_zero() -> None:
    proc = subprocess.run([sys.executable, str(RUNNER), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert "NuRec" in proc.stdout or "nurec" in proc.stdout.lower()


def test_runner_requires_ply_or_usdz(tmp_path: Path) -> None:
    cams = tmp_path / "cameras.json"
    cams.write_text("[]")
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--cameras", str(cams), "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2  # argparse error: one of --ply/--usdz required
    assert "ply" in proc.stderr.lower()
