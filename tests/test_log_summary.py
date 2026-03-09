"""Tests for pipeline log summarization utilities."""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.log_summary import main as log_summary_main
from blueprint_pipeline.log_summary import render_markdown, summarize_logs


def test_summarize_logs_extracts_timings_and_errors(tmp_path: Path) -> None:
    nurec_log = tmp_path / "nurec.log"
    nurec_log.write_text(
        "\n".join(
            [
                "[run-full-pipeline] PHASE 1: NuRec Shim (Stages 1-8)",
                "Stage 4 completed in 12.5s",
                "WARNING: fallback mesh path used",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    orchestrator_log = tmp_path / "orchestrator.log"
    orchestrator_log.write_text(
        "\n".join(
            [
                "[run-full-pipeline] PHASE 3: Swap Orchestrator (Stages C-I)",
                "assembly duration: 9.0s",
                "ERROR: missing texture atlas",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_logs({"nurec": nurec_log, "orchestrator": orchestrator_log})
    assert summary["stats"]["timing_count"] == 2
    assert summary["stats"]["error_count"] == 1
    assert summary["stats"]["warning_count"] == 1
    assert summary["stats"]["stage_count"] == 2

    timings = summary["stage_timings_seconds"]
    assert timings[0]["seconds"] == 12.5
    assert timings[1]["seconds"] == 9.0


def test_render_markdown_includes_sections(tmp_path: Path) -> None:
    summary = summarize_logs({"nurec": tmp_path / "missing_nurec.log", "orchestrator": tmp_path / "missing_orch.log"})
    markdown = render_markdown(summary)
    assert "# Pipeline Log Summary" in markdown
    assert "## Log Inputs" in markdown
    assert "## Timings" in markdown
    assert "No duration lines found." in markdown


def test_log_summary_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "full_pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "nurec.log").write_text("stage done in 1.0s\n", encoding="utf-8")

    rc = log_summary_main(["--pipeline-dir", str(pipeline_dir)])
    assert rc == 0

    out_json = pipeline_dir / "log_summary.json"
    out_md = pipeline_dir / "log_summary.md"
    assert out_json.is_file()
    assert out_md.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v1"
    assert payload["stats"]["timing_count"] >= 1
