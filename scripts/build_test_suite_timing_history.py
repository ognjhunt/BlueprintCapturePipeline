#!/usr/bin/env python3
"""Append full-lane telemetry to bounded history and evaluate regressions."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


HISTORY_SCHEMA = "blueprint.test_suite_timing_history.v1"
REPORT_SCHEMA = "blueprint.test_suite_timing_regression.v1"
TELEMETRY_SCHEMA = "blueprint.test_suite_telemetry.v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_PATTERN = re.compile(r"^[1-9][0-9]*$")
MAX_OBSERVATIONS = 30
SLOW_LIMIT = 20
COUNT_FLOOR_RATIO = 0.98
WALL_WARNING_RATIO = 1.35
INSTABILITY_RATIO = 2.0
INSTABILITY_FLOOR_SECONDS = 1.0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"json_invalid:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _positive_number(value: object, *, field: str, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}_invalid")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ValueError(f"{field}_invalid")
    return result


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field}_invalid")
    return value


def _summary(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("summary_invalid")
    result = dict(value)
    for key in ("test_count", "test_file_count", "parametrized_case_count"):
        number = result.get(key)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"summary_{key}_invalid")
    for key in (
        "parametrized_case_fraction",
        "summed_case_duration_seconds",
        "reported_suite_wall_seconds",
        "maximum_test_file_duration_seconds",
    ):
        _positive_number(result.get(key), field=f"summary_{key}")
    if not isinstance(result.get("line_coverage_collected"), bool):
        raise ValueError("summary_line_coverage_collected_invalid")
    return result


def _slow_rows(value: object, *, identity: str, duration: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{identity}_rows_invalid")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in value[:SLOW_LIMIT]:
        if not isinstance(row, dict):
            raise ValueError(f"{identity}_row_invalid")
        name = row.get(identity)
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"{identity}_invalid")
        seen.add(name)
        result.append(
            {
                identity: name,
                duration: _positive_number(row.get(duration), field=duration),
            }
        )
    return result


def _observation_from_telemetry(
    telemetry: Mapping[str, Any],
    *,
    run_id: str,
    run_url: str,
    observed_at: str,
) -> dict[str, Any]:
    if telemetry.get("schema_version") != TELEMETRY_SCHEMA:
        raise ValueError("telemetry_schema_invalid")
    repository_sha = telemetry.get("repository_sha")
    if not isinstance(repository_sha, str) or SHA_PATTERN.fullmatch(repository_sha) is None:
        raise ValueError("repository_sha_invalid")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id_invalid")
    expected_suffix = f"/actions/runs/{run_id}"
    if not run_url.startswith("https://github.com/") or not run_url.endswith(expected_suffix):
        raise ValueError("run_url_invalid")
    try:
        parsed_time = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("observed_at_invalid") from exc
    if parsed_time.tzinfo is None:
        raise ValueError("observed_at_invalid")
    parallelization = telemetry.get("parallelization")
    if not isinstance(parallelization, dict):
        raise ValueError("parallelization_invalid")
    if parallelization.get("strategy") != "pytest_xdist_loadfile":
        raise ValueError("parallelization_strategy_invalid")
    return {
        "run_id": run_id,
        "run_url": run_url,
        "observed_at": observed_at,
        "repository_sha": repository_sha,
        "workers": _positive_int(parallelization.get("workers"), field="workers"),
        "summary": _summary(telemetry.get("summary")),
        "slow_testcases": _slow_rows(
            telemetry.get("top_testcases_by_duration"),
            identity="nodeid",
            duration="duration_seconds",
        ),
        "slow_files": _slow_rows(
            telemetry.get("top_test_files_by_duration"),
            identity="path",
            duration="case_duration_seconds",
        ),
    }


def _validate_observation(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("history_observation_invalid")
    telemetry = {
        "schema_version": TELEMETRY_SCHEMA,
        "repository_sha": value.get("repository_sha"),
        "summary": value.get("summary"),
        "parallelization": {
            "strategy": "pytest_xdist_loadfile",
            "workers": value.get("workers"),
        },
        "top_testcases_by_duration": value.get("slow_testcases"),
        "top_test_files_by_duration": value.get("slow_files"),
    }
    return _observation_from_telemetry(
        telemetry,
        run_id=str(value.get("run_id") or ""),
        run_url=str(value.get("run_url") or ""),
        observed_at=str(value.get("observed_at") or ""),
    )


def _load_history(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("schema_version") != HISTORY_SCHEMA:
        raise ValueError("history_schema_invalid")
    values = payload.get("observations")
    if not isinstance(values, list) or not values:
        raise ValueError("history_observations_invalid")
    observations = [_validate_observation(value) for value in values]
    run_ids = [row["run_id"] for row in observations]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("history_duplicate_run_id")
    return observations


def _median_summary(observations: Sequence[Mapping[str, Any]], key: str) -> float:
    return statistics.median(float(row["summary"][key]) for row in observations)


def _instability_rows(
    observations: Sequence[Mapping[str, Any]],
    *,
    collection: str,
    identity: str,
    duration: str,
) -> list[dict[str, Any]]:
    samples: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        for row in observation[collection]:
            samples[str(row[identity])].append(float(row[duration]))
    results: list[dict[str, Any]] = []
    for name, values in samples.items():
        if len(values) < 3 or max(values) < INSTABILITY_FLOOR_SECONDS:
            continue
        minimum = min(values)
        ratio = math.inf if minimum == 0 else max(values) / minimum
        if ratio < INSTABILITY_RATIO:
            continue
        results.append(
            {
                identity: name,
                "sample_count": len(values),
                "minimum_duration_seconds": round(minimum, 6),
                "maximum_duration_seconds": round(max(values), 6),
                "max_to_min_ratio": round(ratio, 6),
            }
        )
    results.sort(key=lambda row: (-float(row["max_to_min_ratio"]), str(row[identity])))
    return results[:SLOW_LIMIT]


def build_history_and_report(
    *,
    current_telemetry: Mapping[str, Any],
    baseline_observations: Sequence[Mapping[str, Any]],
    run_id: str,
    run_url: str,
    observed_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = _observation_from_telemetry(
        current_telemetry,
        run_id=run_id,
        run_url=run_url,
        observed_at=observed_at,
    )
    historical = [_validate_observation(row) for row in baseline_observations]
    historical = [row for row in historical if row["run_id"] != current["run_id"]]
    comparable = [row for row in historical if row["workers"] == current["workers"]]
    blockers: list[str] = []
    warnings: list[str] = []
    comparison: dict[str, Any] = {
        "same_worker_observation_count": len(comparable),
        "workers": current["workers"],
    }
    if comparable:
        reference_test_count = _median_summary(comparable[-5:], "test_count")
        reference_file_count = _median_summary(comparable[-5:], "test_file_count")
        reference_wall = _median_summary(
            comparable[-5:], "reported_suite_wall_seconds"
        )
        current_summary = current["summary"]
        test_ratio = float(current_summary["test_count"]) / reference_test_count
        file_ratio = float(current_summary["test_file_count"]) / reference_file_count
        wall_ratio = float(current_summary["reported_suite_wall_seconds"]) / reference_wall
        comparison.update(
            {
                "reference_test_count_median": reference_test_count,
                "reference_test_file_count_median": reference_file_count,
                "reference_suite_wall_seconds_median": round(reference_wall, 6),
                "test_count_ratio": round(test_ratio, 6),
                "test_file_count_ratio": round(file_ratio, 6),
                "suite_wall_ratio": round(wall_ratio, 6),
            }
        )
        if test_ratio < COUNT_FLOOR_RATIO:
            blockers.append("test_count_contraction")
        if file_ratio < COUNT_FLOOR_RATIO:
            blockers.append("test_file_count_contraction")
        if wall_ratio > WALL_WARNING_RATIO:
            warnings.append("suite_wall_regression")
    else:
        warnings.append("same_worker_baseline_unavailable")

    observations = [*historical, current][-MAX_OBSERVATIONS:]
    same_worker_observations = [
        row for row in observations if row["workers"] == current["workers"]
    ]
    unstable_testcases = _instability_rows(
        same_worker_observations,
        collection="slow_testcases",
        identity="nodeid",
        duration="duration_seconds",
    )
    unstable_files = _instability_rows(
        same_worker_observations,
        collection="slow_files",
        identity="path",
        duration="case_duration_seconds",
    )
    if unstable_testcases:
        warnings.append("unstable_slow_testcases")
    if unstable_files:
        warnings.append("unstable_slow_files")
    history = {
        "schema_version": HISTORY_SCHEMA,
        "max_observations": MAX_OBSERVATIONS,
        "observations": observations,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "blocked" if blockers else "passed",
        "repository_sha": current["repository_sha"],
        "run_id": current["run_id"],
        "blockers": blockers,
        "warnings": warnings,
        "policy": {
            "count_floor_ratio": COUNT_FLOOR_RATIO,
            "suite_wall_warning_ratio": WALL_WARNING_RATIO,
            "instability_ratio": INSTABILITY_RATIO,
            "instability_floor_seconds": INSTABILITY_FLOOR_SECONDS,
            "timing_warnings_are_blocking": False,
        },
        "comparison": comparison,
        "unstable_slow_testcases": unstable_testcases,
        "unstable_slow_files": unstable_files,
    }
    return history, report


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--history-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = args.history if args.history and args.history.is_file() else args.baseline
        observations = _load_history(source.resolve())
        history, report = build_history_and_report(
            current_telemetry=_load_json(args.current.resolve()),
            baseline_observations=observations,
            run_id=args.run_id,
            run_url=args.run_url,
            observed_at=args.observed_at,
        )
        _write_json_atomic(args.history_output.resolve(), history)
        _write_json_atomic(args.report_output.resolve(), report)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[test-suite-history] ERROR {exc}", file=sys.stderr)
        return 1
    print(
        "[test-suite-history] "
        f"status={report['status']} warnings={len(report['warnings'])} "
        f"observations={len(history['observations'])}"
    )
    return 1 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
