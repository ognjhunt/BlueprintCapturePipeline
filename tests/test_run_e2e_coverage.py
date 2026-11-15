from __future__ import annotations

import json
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


def _website_preparation(**extra) -> dict:
    """The capture pipeline's receipt for an admitted website preparation."""
    return {
        "status": "completed", "lanes": ["qualification"],
        "artifact_purpose": "evaluation_preparation", "robot_evaluation_performed": False,
        "results": [{"lane": "qualification", "status": "completed", "native_execution_complete": False,
                     "website_scene_preparation": {"status": "prepared", **extra}}],
    }


def test_run_end_to_end_materializes_raw_and_prepares_the_website_capture(
    monkeypatch, tmp_path: Path
) -> None:
    capture_root = _capture_root(tmp_path)
    raw_root = capture_root / "raw"
    raw_root.mkdir()
    (raw_root / "capture_upload_complete.json").write_text("{}", encoding="utf-8")

    calls: dict[str, object] = {}
    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda root: {"status": "ready", "root": str(root)},
    )

    def fake_materialize(**kwargs):
        (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
        return calls.setdefault("materialize", kwargs)

    def fake_pipeline(**kwargs):
        calls["pipeline"] = kwargs
        return _website_preparation()

    monkeypatch.setattr(run_e2e, "materialize_capture_bundle", fake_materialize)
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", fake_pipeline)

    result = run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai")

    assert result["schema_version"] == "v1"
    assert result["capture_root"] == str(capture_root)
    assert result["provider"] == "openai"
    assert result["preflight_status"] == "ready"
    assert result["pipeline_status"] == "completed"
    assert result["pipeline_lanes"] == ["qualification"]
    assert result["task_evaluation_supervisor"]["reason"] == "website_scene_preparation_uses_native_handoff"
    assert calls["pipeline"]["lane"] == "qualification"
    assert calls["materialize"]["raw_prefix_uri"] == "gs://bucket/scenes/site-1/captures/cap-1/raw"
    stage_ledger = json.loads(Path(result["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8"))
    assert stage_ledger["schema_version"] == "run_e2e_stage_ledger.v1"
    assert stage_ledger["status"] == "completed"
    assert stage_ledger["pipeline_lane"] == "qualification"
    assert stage_ledger["stages"]["preflight"]["status"] == "completed"
    assert stage_ledger["stages"]["materialization"]["status"] == "completed"
    assert stage_ledger["stages"]["capture_pipeline"]["detail"] == "completed"
    assert stage_ledger["stages"]["task_evaluation_supervisor"]["status"] == "skipped"
    run_summary = json.loads(Path(result["run_summary_path"]).read_text(encoding="utf-8"))
    assert run_summary["schema_version"] == "pipeline_run_summary.v1"
    assert run_summary["status"] == "completed"
    by_stage = {row["stage"]: row for row in run_summary["stage_timings"]}
    assert by_stage["capture_pipeline"]["duration_seconds"] is not None


def test_run_end_to_end_refuses_a_result_that_is_not_a_website_preparation(
    monkeypatch, tmp_path: Path
) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _root: {"status": "ready"})
    monkeypatch.setattr(run_e2e, "run_capture_pipeline",
                        lambda **_kwargs: {"status": "completed", "lanes": ["qualification"], "results": []})

    with pytest.raises(PipelineError, match="website_scene_preparation_missing"):
        run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai")
    stage_ledger = json.loads(
        (capture_root / "pipeline" / "run_e2e_stage_ledger.json").read_text(encoding="utf-8"))
    assert stage_ledger["failed_stage"] == "task_evaluation_supervisor"


def test_run_end_to_end_blocks_preflight_and_missing_descriptor(
    monkeypatch, tmp_path: Path
) -> None:
    capture_root = _capture_root(tmp_path)
    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda _root: {"missing_required_inputs": ["raw/manifest.json", "raw/video.mov"]},
    )
    with pytest.raises(PipelineError, match="raw/manifest.json,raw/video.mov"):
        run_e2e.run_end_to_end(capture_root=str(capture_root), provider="claude")
    stage_ledger = json.loads(
        (capture_root / "pipeline" / "run_e2e_stage_ledger.json").read_text(encoding="utf-8")
    )
    assert stage_ledger["status"] == "failed"
    assert stage_ledger["failed_stage"] == "preflight"
    assert stage_ledger["stages"]["preflight"]["error_type"] == "PipelineError"
    failed_summary = json.loads(
        (capture_root / "pipeline" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert failed_summary["status"] == "failed"
    assert failed_summary["failed_stage"] == "preflight"

    monkeypatch.setattr(
        run_e2e, "build_capture_preflight_report", lambda _root: {"status": "ready"}
    )
    with pytest.raises(PipelineError, match="Descriptor is missing"):
        run_e2e.run_end_to_end(capture_root=str(capture_root), provider="claude")
    stage_ledger = json.loads(
        (capture_root / "pipeline" / "run_e2e_stage_ledger.json").read_text(encoding="utf-8")
    )
    assert stage_ledger["status"] == "failed"
    assert stage_ledger["failed_stage"] == "materialization"
    assert stage_ledger["stages"]["preflight"]["status"] == "completed"
    assert stage_ledger["stages"]["materialization"]["error_type"] == "PipelineError"


def test_retry_rematerializes_restaged_descriptor_before_resuming_pipeline(
    monkeypatch, tmp_path: Path
) -> None:
    capture_root = _capture_root(tmp_path)
    raw_root = capture_root / "raw"
    raw_root.mkdir()
    (raw_root / "capture_upload_complete.json").write_text("{}")
    descriptor_path = capture_root / "capture_descriptor.json"
    qa_path = capture_root / "qa_report.json"
    cloud_descriptor = '{"capture_source": "unknown"}'
    descriptor_path.write_text(cloud_descriptor)
    descriptor = {"metadata": {"capture_entry_source": "browser_self_capture"}}
    qa = {"status": "verified"}
    calls = {"materialization": 0, "pipeline": 0}

    def materialize(**_kwargs):
        calls["materialization"] += 1
        descriptor_path.write_text(json.dumps(descriptor))
        qa_path.write_text(json.dumps(qa))
        return {"descriptor": descriptor, "qa_report": qa}

    def pipeline(**_kwargs):
        calls["pipeline"] += 1
        assert json.loads(descriptor_path.read_text()) == descriptor
        assert json.loads(qa_path.read_text()) == qa
        if calls["pipeline"] == 1:
            raise PipelineError("retryable preparation failure")
        return _website_preparation()

    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", lambda _root: {"status": "ready"})
    monkeypatch.setattr(run_e2e, "materialize_capture_bundle", materialize)
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", pipeline)
    with pytest.raises(PipelineError, match="retryable preparation failure"):
        run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai")

    # The listener downloads these original cloud records again on redelivery.
    descriptor_path.write_text(cloud_descriptor)
    qa_path.write_text('{}')
    result = run_e2e.run_end_to_end(
        capture_root=str(capture_root), provider="openai", resume_completed_stages=True
    )
    assert result["pipeline_status"] == "completed"
    assert calls == {"materialization": 2, "pipeline": 2}
    # Completed provider stages remain reusable on subsequent retries.
    descriptor_path.write_text(cloud_descriptor)
    qa_path.write_text('{}')
    run_e2e.run_end_to_end(
        capture_root=str(capture_root), provider="openai", resume_completed_stages=True
    )
    assert calls == {"materialization": 3, "pipeline": 2}


def test_run_end_to_end_resumes_completed_stage_snapshots(monkeypatch, tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text("{}", encoding="utf-8")
    calls = {"preflight": 0, "pipeline": 0}

    def fake_preflight(_root):
        calls["preflight"] += 1
        return {"status": "ready", "probe": calls["preflight"]}

    def fake_pipeline(**_kwargs):
        calls["pipeline"] += 1
        return _website_preparation(provider_token="must-not-be-written")

    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", fake_preflight)
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", fake_pipeline)

    first = run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai")
    assert first["preflight_status"] == "ready"
    assert calls == {"preflight": 1, "pipeline": 1}

    monkeypatch.setattr(
        run_e2e,
        "build_capture_preflight_report",
        lambda _root: pytest.fail("preflight should resume from stage snapshot"),
    )
    monkeypatch.setattr(
        run_e2e,
        "run_capture_pipeline",
        lambda **_kwargs: pytest.fail("pipeline should resume from stage snapshot"),
    )

    second = run_e2e.run_end_to_end(
        capture_root=str(capture_root), provider="openai", resume_completed_stages=True,
    )

    assert second["preflight_status"] == "ready"
    assert second["pipeline_status"] == "completed"
    assert second["pipeline_lanes"] == ["qualification"]
    stage_ledger = json.loads(Path(second["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8"))
    assert stage_ledger["resume_completed_stages_requested"] is True
    assert stage_ledger["resume_used_count"] == 2
    assert stage_ledger["stages"]["preflight"]["resume_used"] is True
    assert stage_ledger["stages"]["capture_pipeline"]["resume_used"] is True
    preparation = stage_ledger["stages"]["capture_pipeline"]["result_snapshot"]["results"][0]
    assert preparation["website_scene_preparation"]["provider_token"] == "<redacted>"

    # A historical completed result may have dispatched an evaluation. Rebuild its
    # projection while retaining the separately verified completed stages.
    stage_ledger["stages"]["capture_pipeline"]["result_snapshot"].pop("artifact_purpose")
    Path(second["run_e2e_stage_ledger_path"]).write_text(json.dumps(stage_ledger))
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", fake_pipeline)
    third = run_e2e.run_end_to_end(
        capture_root=str(capture_root), provider="openai", resume_completed_stages=True,
    )
    assert third["pipeline_status"] == "completed"
    assert calls == {"preflight": 1, "pipeline": 2}
    fresh_ledger = json.loads(Path(third["run_e2e_stage_ledger_path"]).read_text())
    assert fresh_ledger["stages"]["capture_pipeline"].get("resume_used", False) is False


def test_run_end_to_end_invalidates_resume_when_capture_inputs_change(
    monkeypatch, tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    (capture_root / "capture_descriptor.json").write_text(
        '{"version": 1}',
        encoding="utf-8",
    )
    raw_root = capture_root / "raw"
    raw_root.mkdir()
    (raw_root / "manifest.json").write_text('{"input": 1}', encoding="utf-8")
    calls = {"preflight": 0, "pipeline": 0}

    def fake_preflight(_root):
        calls["preflight"] += 1
        return {"status": "ready", "probe": calls["preflight"]}

    def fake_pipeline(**_kwargs):
        calls["pipeline"] += 1
        return _website_preparation(probe=calls["pipeline"])

    monkeypatch.setattr(run_e2e, "build_capture_preflight_report", fake_preflight)
    monkeypatch.setattr(run_e2e, "run_capture_pipeline", fake_pipeline)

    first = run_e2e.run_end_to_end(capture_root=str(capture_root), provider="openai")
    first_ledger = json.loads(Path(first["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8"))
    first_fingerprint = first_ledger["capture_input_fingerprint"]["fingerprint_sha256"]

    (raw_root / "manifest.json").write_text('{"input": 2}', encoding="utf-8")
    second = run_e2e.run_end_to_end(
        capture_root=str(capture_root), provider="openai", resume_completed_stages=True,
    )

    assert calls == {"preflight": 2, "pipeline": 2}
    assert second["preflight_status"] == "ready"
    assert second["pipeline_status"] == "completed"
    stage_ledger = json.loads(Path(second["run_e2e_stage_ledger_path"]).read_text(encoding="utf-8"))
    assert stage_ledger["resume_completed_stages_requested"] is True
    assert stage_ledger["resume_status"] == "no_compatible_completed_stage_ledger"
    assert stage_ledger.get("resume_used_count", 0) == 0
    assert stage_ledger["capture_input_fingerprint"]["fingerprint_sha256"] != first_fingerprint


def test_run_e2e_main_success_and_failure(monkeypatch, tmp_path: Path, capsys) -> None:
    capture_root = _capture_root(tmp_path)

    def fake_run_end_to_end(**kwargs):
        assert kwargs == {"capture_root": str(capture_root), "provider": "openai",
                          "resume_completed_stages": True}
        return {
            "preflight_status": "ready",
            "pipeline_status": "completed",
            "pipeline_lanes": ["qualification"],
            "task_evaluation_supervisor": {"status": "skipped"},
        }

    monkeypatch.setattr(run_e2e, "run_end_to_end", fake_run_end_to_end)
    assert (
        run_e2e.main(
            ["--capture-root", str(capture_root), "--provider", "openai", "--resume-completed-stages"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "preflight_status=ready" in output
    assert "task_evaluation_supervisor=skipped" in output

    monkeypatch.setattr(
        run_e2e, "run_end_to_end", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
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


def test_run_e2e_main_fails_before_work_on_invalid_environment(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    capture_root = _capture_root(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(run_e2e, "run_end_to_end", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("BLUEPRINT_ALLOW_GPU_PROVISIONING", "sometimes")
    assert run_e2e.main(["--capture-root", str(capture_root), "--provider", "local"]) == 1
    assert calls == []
    assert "invalid_boolean_environment_value" in capsys.readouterr().out
