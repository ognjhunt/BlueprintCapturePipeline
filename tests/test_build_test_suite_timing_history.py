from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_test_suite_timing_history",
    ROOT / "scripts" / "build_test_suite_timing_history.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _telemetry(
    *,
    sha: str,
    workers: int = 4,
    tests: int = 100,
    files: int = 20,
    wall: float = 10.0,
    slow_duration: float = 2.0,
) -> dict[str, object]:
    return {
        "schema_version": "blueprint.test_suite_telemetry.v1",
        "repository_sha": sha,
        "summary": {
            "test_count": tests,
            "test_file_count": files,
            "parametrized_case_count": 10,
            "parametrized_case_fraction": 0.1,
            "summed_case_duration_seconds": 30.0,
            "reported_suite_wall_seconds": wall,
            "maximum_test_file_duration_seconds": 4.0,
            "line_coverage_collected": False,
        },
        "parallelization": {
            "strategy": "pytest_xdist_loadfile",
            "workers": workers,
        },
        "top_testcases_by_duration": [
            {
                "nodeid": "tests/test_slow.py::test_slow",
                "duration_seconds": slow_duration,
            }
        ],
        "top_test_files_by_duration": [
            {
                "path": "tests/test_slow.py",
                "case_duration_seconds": slow_duration,
            }
        ],
    }


def _observation(index: int, **kwargs: object) -> dict[str, object]:
    run_id = str(100 + index)
    return MODULE._observation_from_telemetry(
        _telemetry(sha=f"{index:x}" * 40, **kwargs),
        run_id=run_id,
        run_url=f"https://github.com/example/repo/actions/runs/{run_id}",
        observed_at=f"2026-08-{index + 1:02d}T00:00:00Z",
    )


def test_history_compares_only_same_worker_runs_and_warns_on_timing() -> None:
    baselines = [
        _observation(1, workers=1, wall=40.0),
        _observation(2, workers=4, wall=10.0),
    ]

    history, report = MODULE.build_history_and_report(
        current_telemetry=_telemetry(sha="a" * 40, workers=4, wall=14.0),
        baseline_observations=baselines,
        run_id="999",
        run_url="https://github.com/example/repo/actions/runs/999",
        observed_at="2026-08-14T00:00:00Z",
    )

    assert report["status"] == "passed"
    assert report["warnings"] == ["suite_wall_regression"]
    assert report["comparison"]["same_worker_observation_count"] == 1
    assert report["comparison"]["suite_wall_ratio"] == 1.4
    assert [row["run_id"] for row in history["observations"]] == ["101", "102", "999"]


def test_history_blocks_test_or_file_count_contraction() -> None:
    baselines = [_observation(1), _observation(2)]

    _, report = MODULE.build_history_and_report(
        current_telemetry=_telemetry(sha="b" * 40, tests=95, files=18),
        baseline_observations=baselines,
        run_id="999",
        run_url="https://github.com/example/repo/actions/runs/999",
        observed_at="2026-08-14T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        "test_count_contraction",
        "test_file_count_contraction",
    ]


def test_history_flags_repeated_same_worker_instability() -> None:
    baselines = [
        _observation(1, slow_duration=1.0),
        _observation(2, slow_duration=1.5),
    ]

    _, report = MODULE.build_history_and_report(
        current_telemetry=_telemetry(sha="c" * 40, slow_duration=3.0),
        baseline_observations=baselines,
        run_id="999",
        run_url="https://github.com/example/repo/actions/runs/999",
        observed_at="2026-08-14T00:00:00Z",
    )

    assert report["status"] == "passed"
    assert report["warnings"] == [
        "unstable_slow_testcases",
        "unstable_slow_files",
    ]
    assert report["unstable_slow_testcases"] == [
        {
            "nodeid": "tests/test_slow.py::test_slow",
            "sample_count": 3,
            "minimum_duration_seconds": 1.0,
            "maximum_duration_seconds": 3.0,
            "max_to_min_ratio": 3.0,
        }
    ]
