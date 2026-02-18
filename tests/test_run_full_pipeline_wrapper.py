"""Static guards for strict defaults in run_full_pipeline.sh."""

from __future__ import annotations

from pathlib import Path


def _script_text() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "scripts" / "run_full_pipeline.sh").read_text(encoding="utf-8")


def test_wrapper_defaults_to_full_required_completion_mode() -> None:
    text = _script_text()
    assert 'COMPLETION_MODE="${COMPLETION_MODE:-full_required}"' in text
    assert "--completion-mode MODE" in text


def test_wrapper_enforces_strict_env_in_full_mode() -> None:
    text = _script_text()
    assert "export PIPELINE_STANDALONE_MODE=false" in text
    assert "export RUNTIME_PREFLIGHT_ENABLED=true" in text
    assert "export ADVANCED_QUALITY_GATES_ENABLED=true" in text
    assert "export SAM3_TRACKING_MODE" in text
    assert "export SWAP_INCLUDE_HEURISTIC_AS_EXPLICIT=false" in text


def test_wrapper_validates_full_runtime_and_non_stub_scene() -> None:
    text = _script_text()
    assert "validate_full_runtime" in text
    assert "usd-assembly-job/assemble_scene.py" in text
    assert "scene.usda is a standalone stub" in text
