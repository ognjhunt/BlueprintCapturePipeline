"""Tests for runtime preflight validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.runtime_preflight import enforce_preflight, validate_runtime_preflight


def _make_blueprintpipeline_stub(root: Path) -> None:
    (root / "interactive-job").mkdir(parents=True, exist_ok=True)
    (root / "simready-job").mkdir(parents=True, exist_ok=True)
    (root / "usd-assembly-job").mkdir(parents=True, exist_ok=True)
    (root / "tools/source_pipeline").mkdir(parents=True, exist_ok=True)
    (root / "tools/scene_manifest").mkdir(parents=True, exist_ok=True)

    (root / "interactive-job/run_interactive_assets.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "simready-job/prepare_simready_assets.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "usd-assembly-job/assemble_scene.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "tools/source_pipeline/adapter.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (root / "tools/scene_manifest/loader.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (root / "tools/__init__.py").write_text("", encoding="utf-8")
    (root / "tools/source_pipeline/__init__.py").write_text("", encoding="utf-8")
    (root / "tools/scene_manifest/__init__.py").write_text("", encoding="utf-8")


def test_runtime_preflight_passes_with_configured_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TEXT_SAM3D_API_HOST", "https://sam3d.example.internal")
    monkeypatch.setenv("TEXT_SAM3D_API_KEY", "sam3d-key")
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="sam3d",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    enforce_preflight(checks)
    assert all(item.passed for item in checks)


def test_runtime_preflight_fails_when_provider_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("TEXT_SAM3D_API_HOST", raising=False)
    monkeypatch.delenv("TEXT_SAM3D_API_KEY", raising=False)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="sam3d",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    with pytest.raises(Exception):
        enforce_preflight(checks)

    failed_names = {item.name for item in checks if not item.passed}
    assert "provider_sam3d" in failed_names
