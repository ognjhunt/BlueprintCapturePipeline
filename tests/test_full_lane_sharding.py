from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.full_lane_sharding import (
    FullLaneShardError,
    _validate_baseline,
    _plan_expected_nodeids,
    aggregate_shards,
    build_duration_baseline,
    build_shard_plan,
    main as sharding_main,
    validate_sharded_artifact,
    verify_shard,
)


SHA = "a" * 40
SOURCE_RUN_ID = 123456
ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collection(nodeids: list[str], *, phase: str) -> dict[str, object]:
    return {
        "schema_version": "blueprint_full_lane_collection.v1",
        "phase": phase,
        "test_count": len(nodeids),
        "nodeids_sha256": hashlib.sha256("\n".join(nodeids).encode()).hexdigest(),
        "nodeids": nodeids,
    }


def _junit(path: Path, nodeids: list[str], *, failure: str | None = None) -> None:
    testcases = []
    for index, nodeid in enumerate(nodeids):
        failed = failure == nodeid
        testcases.append(
            f'<testcase classname="case{index}" name="test{index}" time="{index + 1}">'
            "<properties>"
            f'<property name="blueprint_nodeid" value="{nodeid}"/>'
            "</properties>"
            + ('<failure message="failed"/>' if failed else "")
            + "</testcase>"
        )
    path.write_text(
        '<testsuites><testsuite name="seed" '
        f'tests="{len(nodeids)}" failures="{1 if failure else 0}" '
        'errors="0" skipped="0" time="1">'
        + "".join(testcases)
        + "</testsuite></testsuites>",
        encoding="utf-8",
    )


def _nodeids() -> list[str]:
    return [
        f"tests/test_file_{file_index}.py::test_case_{case_index}"
        for file_index in range(8)
        for case_index in range(file_index % 3 + 1)
    ]


def _seed_plan(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    nodeids = _nodeids()
    planned_path = tmp_path / "full-test-lane-planned.json"
    baseline_junit = tmp_path / "baseline-junit.xml"
    baseline_path = tmp_path / "full-test-lane-duration-baseline.json"
    plan_path = tmp_path / "full-test-lane-shard-plan.json"
    _write_json(planned_path, _collection(nodeids, phase="planned"))
    _junit(baseline_junit, nodeids)
    baseline = build_duration_baseline(
        junit=baseline_junit,
        source_sha=SHA,
        source_run_id=SOURCE_RUN_ID,
    )
    _write_json(baseline_path, baseline)
    plan = build_shard_plan(
        planned=_collection(nodeids, phase="planned"),
        duration_baseline=baseline,
        repository_sha=SHA,
    )
    _write_json(plan_path, plan)
    return planned_path, baseline_path, plan_path, plan


def test_plan_is_file_preserving_disjoint_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    planned_path, baseline_path, _plan_path, plan = _seed_plan(tmp_path)
    nodeids = _nodeids()
    planned = json.loads(planned_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert plan == build_shard_plan(
        planned=planned,
        duration_baseline=baseline,
        repository_sha=SHA,
    )
    assert plan["strategy"] == "lpt_file_preserving_serial_shards"
    assert plan["shard_count"] == 4
    assignments: dict[str, int] = {}
    union: list[str] = []
    for shard in plan["shards"]:  # type: ignore[index]
        index = int(shard["index"])
        expected = _plan_expected_nodeids(
            plan=plan, planned_nodeids=nodeids, shard_index=index
        )
        assert len(expected) == shard["expected_test_count"]
        assert hashlib.sha256("\n".join(expected).encode()).hexdigest() == shard[
            "expected_nodeids_sha256"
        ]
        union.extend(expected)
        for path in shard["files"]:
            assert path not in assignments
            assignments[str(path)] = index

    assert set(union) == set(nodeids)
    assert len(union) == len(nodeids)
    assert len(assignments) == 8
    assert set(assignments) == {nodeid.split("::", 1)[0] for nodeid in nodeids}


def test_committed_duration_baseline_is_exact_green_run_evidence() -> None:
    baseline_path = ROOT / ".github" / "full-test-lane-duration-baseline.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    rows = _validate_baseline(payload)

    assert payload["source_repository_sha"] == (
        "809d0304ccd616a0a85da8055afa02e9784b622a"
    )
    assert payload["source_run_id"] == 31897024458
    assert payload["test_count"] == 12866
    assert payload["file_count"] == len(rows) == 1124


def _seed_shard_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    planned_path, baseline_path, plan_path, plan = _seed_plan(tmp_path)
    nodeids = _nodeids()
    shard_root = tmp_path / "shard-artifacts"
    for index in range(4):
        shard_dir = shard_root / f"artifact-{index}"
        shard_dir.mkdir(parents=True)
        expected = _plan_expected_nodeids(
            plan=plan, planned_nodeids=nodeids, shard_index=index
        )
        local_planned = shard_dir / "full-test-lane-planned.json"
        local_baseline = shard_dir / "full-test-lane-duration-baseline.json"
        local_plan = shard_dir / "full-test-lane-shard-plan.json"
        local_executed = shard_dir / "full-test-lane-shard-executed.json"
        local_junit = shard_dir / "full-test-lane-shard-junit.xml"
        local_planned.write_bytes(planned_path.read_bytes())
        local_baseline.write_bytes(baseline_path.read_bytes())
        local_plan.write_bytes(plan_path.read_bytes())
        _write_json(local_executed, _collection(expected, phase="executed"))
        _junit(local_junit, expected)
        receipt = verify_shard(
            planned_path=local_planned,
            duration_baseline_path=local_baseline,
            plan_path=local_plan,
            executed_path=local_executed,
            junit_path=local_junit,
            repository_sha=SHA,
            shard_index=index,
        )
        assert receipt["status"] == "passed"
        _write_json(shard_dir / "full-test-lane-shard-verification.json", receipt)
    return shard_root, tmp_path / "aggregate"


def test_aggregate_reconstructs_canonical_artifacts_and_revalidates_every_shard(
    tmp_path: Path,
) -> None:
    shard_root, output = _seed_shard_artifacts(tmp_path)

    receipt = aggregate_shards(
        shard_root=shard_root, output_dir=output, repository_sha=SHA
    )

    assert receipt["status"] == "passed"
    assert receipt["test_count"] == len(_nodeids())
    assert receipt["zero_duplicates"] is True
    assert receipt["zero_omissions"] is True
    assert receipt["zero_failures_errors_and_skips"] is True
    assert len(receipt["shards"]) == 4
    assert (output / "full-test-lane-planned.json").is_file()
    assert (output / "full-test-lane-executed.json").is_file()
    assert (output / "full-test-lane-junit.xml").is_file()
    assert (output / "cpu_full.json").exists() is False
    assert validate_sharded_artifact(output, repository_sha=SHA) == receipt

    executed = output / "shards/shard-2/full-test-lane-shard-executed.json"
    payload = json.loads(executed.read_text(encoding="utf-8"))
    payload["nodeids"] = payload["nodeids"][:-1]
    _write_json(executed, payload)
    with pytest.raises(FullLaneShardError):
        validate_sharded_artifact(output, repository_sha=SHA)


def test_shard_verification_fails_closed_on_failure_skip_or_substitution(
    tmp_path: Path,
) -> None:
    planned_path, baseline_path, plan_path, plan = _seed_plan(tmp_path)
    expected = _plan_expected_nodeids(
        plan=plan, planned_nodeids=_nodeids(), shard_index=0
    )
    executed = tmp_path / "full-test-lane-shard-executed.json"
    junit = tmp_path / "full-test-lane-shard-junit.xml"
    _write_json(executed, _collection(expected, phase="executed"))
    _junit(junit, expected, failure=expected[0])

    receipt = verify_shard(
        planned_path=planned_path,
        duration_baseline_path=baseline_path,
        plan_path=plan_path,
        executed_path=executed,
        junit_path=junit,
        repository_sha=SHA,
        shard_index=0,
    )

    assert receipt["status"] == "blocked"
    assert "shard_junit_failures:1" in receipt["blockers"]

    substituted = list(expected)
    substituted[-1] = "tests/test_other.py::test_substituted"
    _write_json(executed, _collection(substituted, phase="executed"))
    _junit(junit, substituted)
    receipt = verify_shard(
        planned_path=planned_path,
        duration_baseline_path=baseline_path,
        plan_path=plan_path,
        executed_path=executed,
        junit_path=junit,
        repository_sha=SHA,
        shard_index=0,
    )
    assert receipt["status"] == "blocked"
    assert "shard_executed_nodeids_mismatch" in receipt["blockers"]
    assert "shard_junit_nodeids_mismatch" in receipt["blockers"]


def test_verify_command_retains_blocked_receipt_when_inputs_are_missing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "full-test-lane-shard-verification.json"

    assert (
        sharding_main(
            [
                "verify-shard",
                "--planned",
                str(tmp_path / "missing-planned.json"),
                "--duration-baseline",
                str(tmp_path / "missing-baseline.json"),
                "--plan",
                str(tmp_path / "missing-plan.json"),
                "--executed",
                str(tmp_path / "missing-executed.json"),
                "--junit",
                str(tmp_path / "missing-junit.xml"),
                "--repository-sha",
                SHA,
                "--shard-index",
                "0",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["blockers"][0].startswith("shard_verification_error:")


def test_aggregate_command_retains_blocked_receipt_when_shards_are_missing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "aggregate"

    assert (
        sharding_main(
            [
                "aggregate",
                "--shard-root",
                str(tmp_path / "missing-shards"),
                "--repository-sha",
                SHA,
                "--output-dir",
                str(output),
            ]
        )
        == 1
    )
    payload = json.loads(
        (output / "full-test-lane-shard-aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["aggregate_shard_receipt_count_invalid"]
