"""Executable smoke test for wrapper wiring with tiny fixtures."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


def _bash_major_version() -> int:
    proc = subprocess.run(
        ["bash", "-lc", "printf '%s' \"${BASH_VERSINFO[0]}\""],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(proc.stdout.strip() or "0")


@pytest.mark.skipif(_bash_major_version() < 4, reason="requires bash >=4 for wrapper lowercase expansion")
def test_run_pipeline_smoke_script_executes_with_tiny_fixture(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    smoke_root = tmp_path / "smoke_case"

    env = dict(os.environ)
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}:{existing}"

    subprocess.run(
        ["bash", str(repo_root / "scripts" / "run_pipeline_smoke.sh"), str(smoke_root)],
        check=True,
        cwd=repo_root,
        env=env,
    )

    run_summary_path = smoke_root / "full_pipeline" / "run_summary.json"
    log_summary_path = smoke_root / "full_pipeline" / "log_summary.json"
    assert run_summary_path.is_file()
    assert log_summary_path.is_file()

    summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert summary["runtime"]["status"] == "passed"
    assert summary["outputs"]["log_summary_json"]["exists"] is True
