"""Contract checks for ``scripts/run_isaac_splat_nurec_render.py``.

The in-package 3dgrut/Isaac export adapter these tests once covered was
retired; the runner script stays in scripts/ and is checked here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_isaac_splat_nurec_render.py"


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
