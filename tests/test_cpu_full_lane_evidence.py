from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_cpu_full_lane_evidence import (
    build_cpu_full_lane_evidence,
    validate_cpu_full_lane_evidence,
)


SHA = "a" * 40


def _write_collection(path: Path, *, phase: str, nodeids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "blueprint_full_lane_collection.v1",
                "phase": phase,
                "test_count": len(nodeids),
                "nodeids_sha256": hashlib.sha256("\n".join(nodeids).encode("utf-8")).hexdigest(),
                "nodeids": nodeids,
            }
        ),
        encoding="utf-8",
    )


def _seed(tmp_path: Path, *, skipped: bool = False) -> tuple[Path, Path, Path]:
    nodeids = ["tests/test_one.py::test_a", "tests/test_two.py::test_b"]
    planned = tmp_path / "full-test-lane-planned.json"
    executed = tmp_path / "full-test-lane-executed.json"
    junit = tmp_path / "full-test-lane-junit.xml"
    _write_collection(planned, phase="planned", nodeids=nodeids)
    _write_collection(executed, phase="executed", nodeids=nodeids)
    skipped_xml = '<skipped message="native dependency unavailable"/>' if skipped else ""
    junit.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" '
        f'skipped="{1 if skipped else 0}">'
        '<testcase classname="tests.test_one" name="test_a"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_one.py::test_a"/>'
        f"</properties>{skipped_xml}</testcase>"
        '<testcase classname="tests.test_two" name="test_b"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_two.py::test_b"/>'
        "</properties></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return planned, executed, junit


def test_cpu_full_evidence_binds_collection_junit_sha_and_artifacts(
    tmp_path: Path,
) -> None:
    planned, executed, junit = _seed(tmp_path)

    evidence = build_cpu_full_lane_evidence(
        planned=planned,
        executed=executed,
        junit=junit,
        repository_sha=SHA,
    )

    assert evidence["status"] == "passed"
    assert evidence["executed"] is True
    assert evidence["test_count"] == 2
    assert evidence["passed_count"] == 2
    assert evidence["skipped_count"] == 0
    assert evidence["failure_count"] == 0
    assert evidence["error_count"] == 0
    assert evidence["repository_sha"] == SHA
    assert set(evidence["artifact_digests"]) == {
        "full-test-lane-planned.json",
        "full-test-lane-executed.json",
        "full-test-lane-junit.xml",
    }
    assert (
        validate_cpu_full_lane_evidence(
            evidence,
            planned=planned,
            executed=executed,
            junit=junit,
            repository_sha=SHA,
        )
        == []
    )


def test_cpu_full_evidence_accepts_parallel_completion_order(tmp_path: Path) -> None:
    planned, executed, junit = _seed(tmp_path)
    junit.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.test_two" name="test_b"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_two.py::test_b"/>'
        "</properties></testcase>"
        '<testcase classname="tests.test_one" name="test_a"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_one.py::test_a"/>'
        "</properties></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )

    evidence = build_cpu_full_lane_evidence(
        planned=planned,
        executed=executed,
        junit=junit,
        repository_sha=SHA,
    )

    assert evidence["status"] == "passed"


def test_cpu_full_evidence_blocks_and_records_every_skip(tmp_path: Path) -> None:
    planned, executed, junit = _seed(tmp_path, skipped=True)

    evidence = build_cpu_full_lane_evidence(
        planned=planned,
        executed=executed,
        junit=junit,
        repository_sha=SHA,
    )

    assert evidence["status"] == "blocked"
    assert evidence["skipped_count"] == 1
    assert evidence["skipped_testcases"] == [
        {
            "testcase": "tests/test_one.py::test_a",
            "reason": "native dependency unavailable",
        }
    ]
    assert "cpu_full_junit_skipped:1" in evidence["blockers"]


def test_cpu_full_evidence_recomputation_rejects_tampering(tmp_path: Path) -> None:
    planned, executed, junit = _seed(tmp_path)
    evidence = build_cpu_full_lane_evidence(
        planned=planned,
        executed=executed,
        junit=junit,
        repository_sha=SHA,
    )
    evidence["nodeids_sha256"] = "0" * 64

    blockers = validate_cpu_full_lane_evidence(
        evidence,
        planned=planned,
        executed=executed,
        junit=junit,
        repository_sha=SHA,
    )

    assert "cpu_full_evidence_field_mismatch:nodeids_sha256" in blockers


def test_cpu_full_evidence_rejects_same_count_substituted_and_duplicate_nodeids(
    tmp_path: Path,
) -> None:
    planned, executed, junit = _seed(tmp_path)
    junit.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="tests.test_other" name="test_x"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_other.py::test_x"/>'
        "</properties></testcase>"
        '<testcase classname="tests.test_other" name="test_x"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_other.py::test_x"/>'
        "</properties></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )

    evidence = build_cpu_full_lane_evidence(
        planned=planned,
        executed=executed,
        junit=junit,
        repository_sha=SHA,
    )

    assert evidence["status"] == "blocked"
    assert "cpu_full_junit_duplicate_nodeids" in evidence["blockers"]
    assert "cpu_full_junit_nodeids_mismatch" in evidence["blockers"]
