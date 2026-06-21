from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import run_e2e
from blueprint_pipeline.common import PipelineError


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "storage" / "bucket" / "scenes" / "site-1" / "captures" / "cap-1"
    root.mkdir(parents=True)
    return root


def test_run_end_to_end_materializes_raw_and_threads_optional_lanes(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    raw_root = capture_root / "raw"
    raw_root.mkdir()
    (raw_root / "capture_upload_complete.json").write_text("{}", encoding="utf-8")

    calls: dict[str, object] = {}
    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda root: {"status": "ready", "root": str(root)})
    monkeypatch.setattr(run_e2e, "materialize_capture_bundle", lambda **kwargs: calls.setdefault("materialize", kwargs))
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", lambda **kwargs: {"status": "completed", "lanes": [kwargs["lane"]]})
    monkeypatch.setattr(
        run_e2e,
        "run_agent_review",
        lambda **kwargs: {
            "artifacts": {"readiness_report": "ready.md"},
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.zip",
            "provider": kwargs["provider_name"],
        },
    )
    monkeypatch.setattr(
        run_e2e,
        "run_evaluation_prep_stage",
        lambda **kwargs: {
            "manifest_path": "eval/manifest.json",
            "webapp_sync_result": {"status": "skipped"},
            "site_package_manifest": {"status": "blocked"},
            "hosted_review_readiness": {"ready": False},
            "proof_pack_manifest": {"proof": False},
            "proof_path_status": {"status": "blocked"},
        },
    )
    monkeypatch.setattr(run_e2e, "run_cosmos_zero_shot_validation_lane", lambda **kwargs: {"status": "completed"})

    result = run_e2e.run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        pipeline_lane="all",
        run_evaluation_prep=True,
        evaluation_prep_provider="manual",
        run_cosmos_validation=True,
    )

    assert result["schema_version"] == "v1"
    assert result["capture_root"] == str(capture_root)
    assert result["provider"] == "openai"
    assert result["preflight_status"] == "ready"
    assert result["pipeline_status"] == "completed"
    assert result["pipeline_lanes"] == ["all"]
    assert result["pipeline_summary"] == "ready.md"
    assert result["final_memo_path"] == "memo.md"
    assert result["webapp_sync_result"] == {"status": "skipped"}
    assert result["site_package_manifest"] == {"status": "blocked"}
    assert result["hosted_review_readiness"] == {"ready": False}
    assert result["proof_pack_manifest"] == {"proof": False}
    assert result["proof_path_status"] == {"status": "blocked"}
    assert result["cosmos_validation"] == {"status": "completed"}
    assert calls["materialize"]["raw_prefix_uri"] == "gs://bucket/scenes/site-1/captures/cap-1/raw"


def test_run_end_to_end_blocks_preflight_and_missing_descriptor(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _root: {"missing_required_inputs": ["raw/manifest.json", "raw/video.mov"]})
    with pytest.raises(PipelineError, match="raw/manifest.json,raw/video.mov"):
        run_e2e.run_end_to_end(capture_root=str(capture_root), provider="claude")

    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _root: {"status": "ready"})
    with pytest.raises(PipelineError, match="Descriptor is missing"):
        run_e2e.run_end_to_end(capture_root=str(capture_root), provider="claude")


def test_run_end_to_end_uses_existing_descriptor_without_optional_lanes(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _root: {"status": "ready"})
    monkeypatch.setattr(run_e2e, "materialize_capture_bundle", lambda **_kwargs: pytest.fail("materialize should not run"))
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", lambda **_kwargs: {"status": "completed", "lanes": ["current"]})
    monkeypatch.setattr(
        run_e2e,
        "run_agent_review",
        lambda **_kwargs: {"artifacts": {}, "final_memo_path": None, "final_bundle_path": None},
    )

    result = run_e2e.run_end_to_end(capture_root=str(capture_root), provider="claude")

    assert result["evaluation_prep"] is None
    assert result["webapp_sync_result"] is None
    assert result["site_package_manifest"] is None
    assert result["hosted_review_readiness"] is None
    assert result["proof_pack_manifest"] is None
    assert result["proof_path_status"] is None
    assert result["cosmos_validation"] is None


def test_run_e2e_main_success_and_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    capture_root = _capture_root(tmp_path)

    def fake_run_end_to_end(**kwargs):
        assert kwargs["provider"] == "openai"
        assert kwargs["openai_phase2_config"].mode == "codex_cli"
        assert kwargs["openai_phase2_config"].model == "gpt-test"
        assert kwargs["openai_phase2_config"].codex_bin == "codex-test"
        assert kwargs["openai_phase2_config"].timeout_seconds == 3
        assert kwargs["openai_phase2_config"].reasoning_effort == "low"
        assert kwargs["run_evaluation_prep"] is True
        assert kwargs["run_cosmos_validation"] is True
        return {
            "preflight_status": "ready",
            "pipeline_status": "completed",
            "pipeline_lanes": ["all"],
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.zip",
            "evaluation_prep": {"manifest_path": "eval.json"},
            "cosmos_validation": {"status": "completed"},
        }

    monkeypatch.setattr(run_e2e, "run_end_to_end", fake_run_end_to_end)
    assert run_e2e.main(
        [
            "--capture-root",
            str(capture_root),
            "--provider",
            "openai",
            "--pipeline-lane",
            "all",
            "--openai-phase2-mode",
            "codex_cli",
            "--openai-phase2-model",
            "gpt-test",
            "--openai-phase2-codex-bin",
            "codex-test",
            "--openai-phase2-timeout-seconds",
            "3",
            "--openai-phase2-reasoning-effort",
            "low",
            "--run-evaluation-prep",
            "--run-cosmos-validation",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "preflight_status=ready" in output
    assert "evaluation_prep=eval.json" in output
    assert "cosmos_validation=completed" in output

    monkeypatch.setattr(run_e2e, "run_end_to_end", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run_e2e.main(["--capture-root", str(capture_root), "--provider", "claude"]) == 1
    assert "[run-e2e] FAILED: boom" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["run-e2e", "--capture-root", str(capture_root), "--provider", "claude"],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("blueprint_pipeline.run_e2e", run_name="__main__")
    assert excinfo.value.code == 1
