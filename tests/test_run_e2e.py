from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.run_e2e import run_end_to_end


def test_run_e2e_keeps_agent_review_and_evaluation_prep(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr("blueprint_pipeline.run_e2e.run_evaluation_prep_stage", lambda **_kwargs: {"manifest_path": "evaluation_prep_manifest.json"})

    result = run_end_to_end(capture_root=str(capture_root), provider="openai", run_evaluation_prep=True)
    assert result["final_memo_path"] == "memo.md"
    assert result["evaluation_prep"]["manifest_path"] == "evaluation_prep_manifest.json"
    assert "simready" not in result
