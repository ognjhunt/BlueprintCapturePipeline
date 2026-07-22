"""Aggregate per-capture pipeline run summaries without upgrading evidence claims."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json


RUN_SUMMARY_SCHEMA_VERSION = "pipeline_run_summary.v1"
FLEET_RUN_SUMMARY_SCHEMA_VERSION = "pipeline_fleet_run_summary.v1"


class RunSummaryAggregationError(ValueError):
    """Raised when an invalid summary would make fleet totals untrustworthy."""


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _required_text(value: Any, *, field: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field}:must_be_nonempty_string")
        return "unknown"
    return value.strip()


def validate_run_summary(value: Mapping[str, Any], *, source: Path) -> None:
    """Validate the fields used for fleet totals before accepting a summary."""

    errors: list[str] = []
    if value.get("schema_version") != RUN_SUMMARY_SCHEMA_VERSION:
        errors.append(f"schema_version:must_be:{RUN_SUMMARY_SCHEMA_VERSION}")
    _required_text(value.get("status"), field="status", errors=errors)
    _required_text(value.get("capture_root"), field="capture_root", errors=errors)
    _required_text(value.get("provider"), field="provider", errors=errors)
    _required_text(value.get("pipeline_lane"), field="pipeline_lane", errors=errors)

    stages = value.get("stage_timings")
    if not isinstance(stages, list):
        errors.append("stage_timings:must_be_list")
    else:
        seen_stages: set[str] = set()
        for index, row in enumerate(stages):
            if not isinstance(row, Mapping):
                errors.append(f"stage_timings[{index}]:must_be_mapping")
                continue
            stage = _required_text(
                row.get("stage"),
                field=f"stage_timings[{index}].stage",
                errors=errors,
            )
            if stage in seen_stages:
                errors.append(f"stage_timings[{index}].stage:duplicate:{stage}")
            seen_stages.add(stage)
            _required_text(
                row.get("status"),
                field=f"stage_timings[{index}].status",
                errors=errors,
            )
            duration = row.get("duration_seconds")
            if duration is not None and _nonnegative_number(duration) is None:
                errors.append(
                    f"stage_timings[{index}].duration_seconds:must_be_nonnegative_number_or_null"
                )

    spend = value.get("spend")
    if not isinstance(spend, Mapping):
        errors.append("spend:must_be_mapping")
    else:
        if not isinstance(spend.get("live_provider_calls_performed"), bool):
            errors.append("spend.live_provider_calls_performed:must_be_boolean")
        for field in ("requested_budget_usd", "actual_gpu_seconds"):
            amount = spend.get(field)
            if amount is not None and _nonnegative_number(amount) is None:
                errors.append(f"spend.{field}:must_be_nonnegative_number_or_null")

    if errors:
        raise RunSummaryAggregationError(
            f"invalid_pipeline_run_summary:{source}:" + ",".join(errors)
        )


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunSummaryAggregationError(
            f"unreadable_pipeline_run_summary:{path}:{type(exc).__name__}"
        ) from exc
    if not isinstance(value, Mapping):
        raise RunSummaryAggregationError(
            f"invalid_pipeline_run_summary:{path}:root:must_be_mapping"
        )
    validate_run_summary(value, source=path)
    return dict(value)


def discover_run_summaries(root: Path) -> list[Path]:
    """Return canonical per-capture summaries below ``root`` in stable order."""

    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise RunSummaryAggregationError(f"fleet_root_not_directory:{resolved_root}")
    return sorted(
        path
        for path in resolved_root.rglob("run_summary.json")
        if path.parent.name == "pipeline" and path.is_file()
    )


def aggregate_run_summaries(root: Path) -> dict[str, Any]:
    """Build a fleet view, rejecting the entire view if any input is malformed."""

    resolved_root = root.expanduser().resolve()
    paths = discover_run_summaries(resolved_root)
    summaries: list[tuple[Path, dict[str, Any]]] = []
    failures: list[str] = []
    for path in paths:
        try:
            summaries.append((path, _load_summary(path)))
        except RunSummaryAggregationError as exc:
            failures.append(str(exc))
    if failures:
        raise RunSummaryAggregationError(
            "fleet_run_summary_inputs_invalid:" + "|".join(failures)
        )

    status_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    stages: dict[str, dict[str, Any]] = {}
    run_rows: list[dict[str, Any]] = []
    live_provider_run_count = 0
    known_gpu_seconds: list[float] = []
    requested_budgets: list[float] = []

    for path, summary in summaries:
        status = str(summary["status"])
        provider = str(summary["provider"])
        lane = str(summary["pipeline_lane"])
        status_counts[status] += 1
        provider_counts[provider] += 1
        lane_counts[lane] += 1

        spend = summary["spend"]
        if spend["live_provider_calls_performed"] is True:
            live_provider_run_count += 1
        actual_gpu_seconds = _nonnegative_number(spend.get("actual_gpu_seconds"))
        if actual_gpu_seconds is not None:
            known_gpu_seconds.append(actual_gpu_seconds)
        requested_budget = _nonnegative_number(spend.get("requested_budget_usd"))
        if requested_budget is not None:
            requested_budgets.append(requested_budget)

        for stage in summary["stage_timings"]:
            stage_name = str(stage["stage"])
            aggregate = stages.setdefault(
                stage_name,
                {
                    "run_count": 0,
                    "status_counts": Counter(),
                    "known_duration_count": 0,
                    "unknown_duration_count": 0,
                    "known_duration_total_seconds": 0.0,
                },
            )
            aggregate["run_count"] += 1
            aggregate["status_counts"][str(stage["status"])] += 1
            duration = _nonnegative_number(stage.get("duration_seconds"))
            if duration is None:
                aggregate["unknown_duration_count"] += 1
            else:
                aggregate["known_duration_count"] += 1
                aggregate["known_duration_total_seconds"] += duration

        run_rows.append(
            {
                "summary_path": str(path.relative_to(resolved_root)),
                "capture_root": summary["capture_root"],
                "status": status,
                "provider": provider,
                "pipeline_lane": lane,
                "started_at": summary.get("started_at"),
                "completed_at": summary.get("completed_at"),
                "failed_stage": summary.get("failed_stage"),
                "live_provider_calls_performed": spend[
                    "live_provider_calls_performed"
                ],
                "actual_gpu_seconds": spend.get("actual_gpu_seconds"),
            }
        )

    stage_rows: list[dict[str, Any]] = []
    for stage_name, aggregate in sorted(stages.items()):
        known_count = int(aggregate["known_duration_count"])
        total = round(float(aggregate["known_duration_total_seconds"]), 6)
        stage_rows.append(
            {
                "stage": stage_name,
                "run_count": aggregate["run_count"],
                "status_counts": dict(sorted(aggregate["status_counts"].items())),
                "known_duration_count": known_count,
                "unknown_duration_count": aggregate["unknown_duration_count"],
                "known_duration_total_seconds": total,
                "known_duration_average_seconds": (
                    round(total / known_count, 6) if known_count else None
                ),
            }
        )

    run_count = len(summaries)
    return {
        "schema_version": FLEET_RUN_SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "fleet_root": str(resolved_root),
        "run_count": run_count,
        "status_counts": dict(sorted(status_counts.items())),
        "provider_counts": dict(sorted(provider_counts.items())),
        "pipeline_lane_counts": dict(sorted(lane_counts.items())),
        "stage_aggregates": stage_rows,
        "spend": {
            "live_provider_run_count": live_provider_run_count,
            "no_live_provider_run_count": run_count - live_provider_run_count,
            "known_actual_gpu_seconds_run_count": len(known_gpu_seconds),
            "unknown_actual_gpu_seconds_run_count": run_count
            - len(known_gpu_seconds),
            "known_actual_gpu_seconds_total": (
                round(sum(known_gpu_seconds), 6) if known_gpu_seconds else None
            ),
            "declared_requested_budget_run_count": len(requested_budgets),
            "declared_requested_budget_total_usd": (
                round(sum(requested_budgets), 6) if requested_budgets else None
            ),
        },
        "claim_boundary": {
            "requested_budget_total_is_not_actual_spend": True,
            "missing_actual_gpu_seconds_are_not_zero": True,
            "fleet_summary_does_not_prove_provider_teardown": True,
            "fleet_summary_does_not_prove_semantic_or_ranking_success": True,
        },
        "runs": run_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = aggregate_run_summaries(args.root)
    if args.output:
        write_json(args.output.expanduser().resolve(), payload)
        print(args.output.expanduser().resolve())
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
