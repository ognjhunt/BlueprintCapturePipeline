from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_test_suite_telemetry",
    ROOT / "scripts" / "build_test_suite_telemetry.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_junit(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4" time="8.0">
  <testcase classname="tests.test_a" name="test_matrix[one]" time="1.0">
    <properties><property name="blueprint_nodeid" value="tests/test_a.py::test_matrix[one]" /></properties>
  </testcase>
  <testcase classname="tests.test_a" name="test_matrix[two]" time="2.0">
    <properties><property name="blueprint_nodeid" value="tests/test_a.py::test_matrix[two]" /></properties>
  </testcase>
  <testcase classname="tests.test_b" name="test_b" time="4.0">
    <properties><property name="blueprint_nodeid" value="tests/test_b.py::test_b" /></properties>
  </testcase>
  <testcase classname="tests.test_c" name="test_c" time="1.0">
    <properties><property name="blueprint_nodeid" value="tests/test_c.py::test_c" /></properties>
  </testcase>
</testsuite></testsuites>
""",
        encoding="utf-8",
    )


def test_telemetry_reports_duration_parametrization_and_file_shards(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    _write_junit(junit)

    result = MODULE.build_telemetry(
        junit=junit,
        repository_sha="a" * 40,
        workers=2,
    )

    assert result["summary"] == {
        "test_count": 4,
        "test_file_count": 3,
        "parametrized_case_count": 2,
        "parametrized_case_fraction": 0.5,
        "summed_case_duration_seconds": 8.0,
        "reported_suite_wall_seconds": 8.0,
        "maximum_test_file_duration_seconds": 4.0,
        "line_coverage_collected": False,
    }
    assert result["top_parametrized_families_by_case_count"] == [
        {
            "family": "tests/test_a.py::test_matrix",
            "case_count": 2,
            "case_duration_seconds": 3.0,
        }
    ]
    assert result["parallelization"]["strategy"] == "pytest_xdist_loadfile"
    shards = result["parallelization"]["shards"]
    assert [shard["estimated_case_duration_seconds"] for shard in shards] == [4.0, 4.0]
    assert sorted(path for shard in shards for path in shard["files"]) == [
        "tests/test_a.py",
        "tests/test_b.py",
        "tests/test_c.py",
    ]


def test_telemetry_labels_one_worker_execution_as_serial(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    _write_junit(junit)

    result = MODULE.build_telemetry(
        junit=junit,
        repository_sha="a" * 40,
        workers=1,
    )

    assert result["parallelization"]["strategy"] == "serial"
    assert result["parallelization"]["workers"] == 1
    assert len(result["parallelization"]["shards"]) == 1


def test_telemetry_rejects_duplicate_or_unbound_nodeids(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    _write_junit(junit)
    text = junit.read_text(encoding="utf-8")
    junit.write_text(text.replace("tests/test_c.py::test_c", "tests/test_b.py::test_b"), encoding="utf-8")

    try:
        MODULE.build_telemetry(junit=junit, repository_sha="a" * 40, workers=2)
    except ValueError as exc:
        assert str(exc) == "junit_duplicate_nodeid:tests/test_b.py::test_b"
    else:  # pragma: no cover - regression guard
        raise AssertionError("duplicate node ID was accepted")
