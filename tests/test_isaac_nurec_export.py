"""Fail-closed contracts for the optional 3dgrut/Isaac export adapter."""

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
    assert threedgrut_available(sys.executable) is False


def test_convert_blocked_unsupported_format(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply\n")
    result = convert_ply_to_isaac_usd(
        ply,
        tmp_path / "o.usdz",
        fmt="bogus",
        python=sys.executable,
    )
    assert result["status"] == "blocked"
    assert "unsupported_isaac_splat_format" in result["blockers"]


def test_convert_blocked_missing_ply(tmp_path: Path) -> None:
    result = convert_ply_to_isaac_usd(
        tmp_path / "nope.ply",
        tmp_path / "o.usdz",
        python=sys.executable,
    )
    assert result["status"] == "blocked"
    assert "standard_ply_missing" in result["blockers"]


def test_convert_blocked_threedgrut_unavailable(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    result = convert_ply_to_isaac_usd(
        ply,
        tmp_path / "o.usdz",
        python=sys.executable,
    )
    assert result["status"] == "blocked"
    assert "threedgrut_unavailable" in result["blockers"]
    assert "remediation" in result


def test_runner_py_compiles() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "py_compile", str(RUNNER)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_runner_help_exits_zero() -> None:
    process = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0
    assert "nurec" in process.stdout.lower()


def test_runner_requires_ply_or_usdz(tmp_path: Path) -> None:
    cameras = tmp_path / "cameras.json"
    cameras.write_text("[]", encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--cameras",
            str(cameras),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 2
    assert "ply" in process.stderr.lower()
