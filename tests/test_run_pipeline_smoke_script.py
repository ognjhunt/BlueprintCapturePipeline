"""Regression checks for fast pipeline smoke harness."""

from pathlib import Path


def _script_text() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_pipeline_smoke.sh"
    return script_path.read_text(encoding="utf-8")


def test_smoke_script_exists_with_strict_mode() -> None:
    text = _script_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text


def test_smoke_script_uses_skip_nurec_best_effort_path() -> None:
    text = _script_text()
    assert "--completion-mode best_effort" in text
    assert "--skip-nurec" in text
    assert "--nurec-output-dir" in text
    assert "TEXT_ASSET_GENERATION_PROVIDER_CHAIN=\"sam3d\"" in text


def test_smoke_script_checks_summary_outputs() -> None:
    text = _script_text()
    assert "run_summary.json" in text
    assert "run_summary.md" in text
    assert "log_summary.json" in text
    assert "log_summary.md" in text
