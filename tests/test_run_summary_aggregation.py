from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.run_summary_aggregation import (
    RunSummaryAggregationError,
    aggregate_run_summaries,
    main,
)


def _write_summary(
    root: Path,
    capture_id: str,
    *,
    status: str,
    provider: str = "fixture_local",
    live: bool = False,
    gpu_seconds: float | None = None,
    budget_usd: float | None = None,
) -> Path:
    path = root / capture_id / "pipeline" / "run_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "pipeline_run_summary.v1",
                "status": status,
                "capture_root": str(root / capture_id),
                "provider": provider,
                "pipeline_lane": "current",
                "started_at": "2026-07-21T12:00:00+00:00",
                "completed_at": "2026-07-21T12:00:03+00:00",
                "failed_stage": "capture_pipeline" if status == "failed" else None,
                "stage_timings": [
                    {
                        "stage": "preflight",
                        "status": "completed",
                        "duration_seconds": 1.0,
                    },
                    {
                        "stage": "capture_pipeline",
                        "status": status,
                        "duration_seconds": 2.0 if status == "completed" else None,
                    },
                ],
                "spend": {
                    "requested_budget_usd": budget_usd,
                    "live_provider_calls_performed": live,
                    "actual_gpu_seconds": gpu_seconds,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_aggregate_run_summaries_preserves_unknown_usage_and_claim_boundaries(
    tmp_path: Path,
) -> None:
    _write_summary(
        tmp_path,
        "capture-a",
        status="completed",
        live=True,
        gpu_seconds=12.5,
        budget_usd=5.0,
    )
    _write_summary(
        tmp_path,
        "capture-b",
        status="failed",
        provider="claude",
        budget_usd=8.0,
    )

    result = aggregate_run_summaries(tmp_path)

    assert result["schema_version"] == "pipeline_fleet_run_summary.v1"
    assert result["run_count"] == 2
    assert result["status_counts"] == {"completed": 1, "failed": 1}
    assert result["provider_counts"] == {"claude": 1, "fixture_local": 1}
    assert result["spend"] == {
        "live_provider_run_count": 1,
        "no_live_provider_run_count": 1,
        "known_actual_gpu_seconds_run_count": 1,
        "unknown_actual_gpu_seconds_run_count": 1,
        "known_actual_gpu_seconds_total": 12.5,
        "declared_requested_budget_run_count": 2,
        "declared_requested_budget_total_usd": 13.0,
    }
    assert result["claim_boundary"]["missing_actual_gpu_seconds_are_not_zero"] is True
    stages = {row["stage"]: row for row in result["stage_aggregates"]}
    assert stages["preflight"]["known_duration_average_seconds"] == 1.0
    assert stages["capture_pipeline"]["unknown_duration_count"] == 1
    assert [row["summary_path"] for row in result["runs"]] == [
        "capture-a/pipeline/run_summary.json",
        "capture-b/pipeline/run_summary.json",
    ]


def test_aggregate_run_summaries_fails_closed_on_any_malformed_input(
    tmp_path: Path,
) -> None:
    _write_summary(tmp_path, "valid", status="completed")
    invalid = _write_summary(tmp_path, "invalid", status="completed")
    payload = json.loads(invalid.read_text(encoding="utf-8"))
    payload["spend"]["actual_gpu_seconds"] = "unknown"
    payload["stage_timings"].append(payload["stage_timings"][0])
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunSummaryAggregationError) as caught:
        aggregate_run_summaries(tmp_path)

    message = str(caught.value)
    assert "fleet_run_summary_inputs_invalid" in message
    assert "actual_gpu_seconds:must_be_nonnegative_number_or_null" in message
    assert "stage:duplicate:preflight" in message


def test_aggregate_run_summaries_empty_fleet_is_explicit(tmp_path: Path) -> None:
    result = aggregate_run_summaries(tmp_path)

    assert result["run_count"] == 0
    assert result["spend"]["known_actual_gpu_seconds_total"] is None
    assert result["spend"]["declared_requested_budget_total_usd"] is None
    assert result["runs"] == []


def test_run_summary_aggregation_cli_writes_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_summary(tmp_path, "capture-a", status="completed")
    output = tmp_path / "fleet.json"

    assert main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(output.resolve())
    assert json.loads(output.read_text(encoding="utf-8"))["run_count"] == 1
