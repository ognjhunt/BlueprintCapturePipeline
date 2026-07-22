from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.run_e2e import main, run_end_to_end


def test_run_e2e_supports_opt_in_agent_review_and_standalone_evaluation_prep(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")

    context = SimpleNamespace(
        capture_root=capture_root,
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        storage_root=tmp_path,
        raw_prefix_uri="gs://bucket/raw",
        descriptor_uri="gs://bucket/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        raw_complete_path=capture_root / "raw" / "capture_upload_complete.json",
        descriptor_path=capture_root / "capture_descriptor.json",
    )

    monkeypatch.setattr("blueprint_pipeline.run_e2e.resolve_local_capture_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr("blueprint_pipeline.run_e2e.build_capture_preflight_report", lambda *_args, **_kwargs: {"status": "passed", "missing_required_inputs": []})
    monkeypatch.setattr("blueprint_pipeline.run_e2e.materialize_capture_bundle", lambda **_kwargs: {"status": "ok"})
    monkeypatch.setattr("blueprint_pipeline.run_e2e.run_capture_pipeline", lambda **_kwargs: {"status": "completed"})
    monkeypatch.setattr(
        "blueprint_pipeline.run_e2e.run_agent_review",
        lambda **_kwargs: {"final_memo_path": "memo.md", "final_bundle_path": "bundle.json", "artifacts": {"readiness_report": "report.md"}},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.run_e2e.run_evaluation_prep_stage",
        lambda **_kwargs: {
            "manifest_path": "evaluation_prep_manifest.json",
            "webapp_sync_result": {"status": "succeeded", "syncs": {"evaluation_prep": {"attachment_payload": {"truth": True}}}},
            "hosted_review_readiness": {"status": "ready"},
            "proof_path_status": {"next_truthful_step": "operator_can_start_hosted_review"},
        },
    )

    result = run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        pipeline_lane="qualification",
        run_agent_review_stage=True,
        run_evaluation_prep=True,
    )
    assert result["final_memo_path"] == "memo.md"
    assert result["evaluation_prep"]["manifest_path"] == "evaluation_prep_manifest.json"
    assert result["webapp_sync_result"]["status"] == "succeeded"
    assert result["hosted_review_readiness"]["status"] == "ready"
    assert result["proof_path_status"]["next_truthful_step"] == "operator_can_start_hosted_review"
    assert "simready" not in result


def test_run_e2e_supports_full_lane_and_optional_cosmos_validation(monkeypatch, tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")

    context = SimpleNamespace(
        capture_root=capture_root,
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-2",
        storage_root=tmp_path,
        raw_prefix_uri="gs://bucket/raw",
        descriptor_uri="gs://bucket/scenes/scene-1/captures/capture-2/capture_descriptor.json",
        raw_complete_path=capture_root / "raw" / "capture_upload_complete.json",
        descriptor_path=capture_root / "capture_descriptor.json",
    )

    monkeypatch.setattr("blueprint_pipeline.run_e2e.resolve_local_capture_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr("blueprint_pipeline.run_e2e.build_capture_preflight_report", lambda *_args, **_kwargs: {"status": "passed", "missing_required_inputs": []})
    monkeypatch.setattr("blueprint_pipeline.run_e2e.materialize_capture_bundle", lambda **_kwargs: {"status": "ok"})
    monkeypatch.setattr(
        "blueprint_pipeline.run_e2e.run_capture_pipeline",
        lambda **kwargs: {"status": "completed", "lanes": [kwargs["lane"]]},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.run_e2e.run_agent_review",
        lambda **_kwargs: {"final_memo_path": "memo.md", "final_bundle_path": "bundle.json", "artifacts": {"readiness_report": "report.md"}},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.run_e2e._run_legacy_cosmos_predict2_5_validation",
        lambda **_kwargs: {"status": "completed", "synthesis_mode": "cosmos_i2w"},
    )

    result = run_end_to_end(
        capture_root=str(capture_root),
        provider="openai",
        pipeline_lane="all",
        allow_legacy_pipeline_lanes=True,
        run_cosmos_validation=True,
    )

    assert result["pipeline_lanes"] == ["all"]
    assert result["support_validation"]["backend"] == "cosmos_predict2_5_legacy"
    assert result["support_validation"]["result"]["synthesis_mode"] == "cosmos_i2w"


def test_legacy_cosmos_support_validation_requires_explicit_legacy_admission() -> None:
    with pytest.raises(
        PipelineError,
        match="legacy_cosmos_predict2_5_validation_requires_allow_legacy_pipeline_lanes",
    ):
        run_end_to_end(
            capture_root="unused-because-gate-is-preflight",
            provider="openai",
            run_cosmos_validation=True,
        )


def test_run_e2e_cli_runs_evaluation_prep_by_default(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_run_end_to_end(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "preflight_status": "passed",
            "pipeline_status": "completed",
            "pipeline_lanes": ["current"],
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.json",
            "evaluation_prep": {"manifest_path": "evaluation_prep_manifest.json"},
        }

    monkeypatch.setattr("blueprint_pipeline.run_e2e.run_end_to_end", fake_run_end_to_end)

    assert main(["--capture-root", str(tmp_path), "--provider", "openai"]) == 0
    assert seen["run_evaluation_prep"] is True
    assert seen["run_agent_review_stage"] is False


def test_run_e2e_cli_skip_evaluation_prep_is_explicit(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_run_end_to_end(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "preflight_status": "passed",
            "pipeline_status": "completed",
            "pipeline_lanes": ["current"],
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.json",
        }

    monkeypatch.setattr("blueprint_pipeline.run_e2e.run_end_to_end", fake_run_end_to_end)

    assert (
        main(
            [
                "--capture-root",
                str(tmp_path),
                "--provider",
                "openai",
                "--skip-evaluation-prep",
            ]
        )
        == 0
    )
    assert seen["run_evaluation_prep"] is False


def test_run_e2e_cli_accepts_local_no_llm_provider(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_run_end_to_end(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {
            "preflight_status": "passed",
            "pipeline_status": "completed",
            "pipeline_lanes": ["current"],
            "final_memo_path": "memo.md",
            "final_bundle_path": "bundle.json",
            "evaluation_prep": {"manifest_path": "evaluation_prep_manifest.json"},
        }

    monkeypatch.setattr("blueprint_pipeline.run_e2e.run_end_to_end", fake_run_end_to_end)

    assert main(["--capture-root", str(tmp_path), "--provider", "local"]) == 0
    assert seen["provider"] == "local"
    assert seen["run_evaluation_prep"] is True
    assert seen["run_agent_review_stage"] is False
