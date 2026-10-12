from __future__ import annotations

import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from blueprint_pipeline import pytest_full_lane_evidence as EVIDENCE_PLUGIN
from scripts.full_lane_sharding import (
    BASELINE_SCHEMA,
    HEAVY_TEST_SECONDS,
    PLAN_SCHEMA,
    SHARD_COUNT,
    FullLaneShardError,
    _plan_expected_nodeids,
    _validate_baseline,
    aggregate_shards,
    build_duration_baseline,
    build_shard_plan,
    validate_sharded_artifact,
    verify_shard,
)
from scripts.full_lane_sharding import (
    main as sharding_main,
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


def _junit(
    path: Path,
    nodeids: list[str],
    *,
    failure: str | None = None,
    durations: dict[str, float] | None = None,
) -> None:
    testcases = []
    for index, nodeid in enumerate(nodeids):
        failed = failure == nodeid
        seconds = (durations or {}).get(nodeid, index + 1)
        testcases.append(
            f'<testcase classname="case{index}" name="test{index}" time="{seconds}">'
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
    return sorted(
        f"tests/test_file_{file_index:02d}.py::test_case_{case_index}"
        for file_index in range(30)
        for case_index in range(file_index % 3 + 1)
    )


def _plan_for(
    nodeids: list[str], durations: dict[str, float], tmp_path: Path
) -> dict[str, object]:
    """Plan the collection against a baseline measured with these durations."""
    junit = tmp_path / "measured-junit.xml"
    _junit(junit, nodeids, durations=durations)
    baseline = build_duration_baseline(
        junit=junit, source_sha=SHA, source_run_id=SOURCE_RUN_ID
    )
    return build_shard_plan(
        planned=_collection(nodeids, phase="planned"),
        duration_baseline=baseline,
        repository_sha=SHA,
    )


def _owner_of(plan: dict[str, object], nodeid: str) -> int:
    owners = [
        int(shard["index"])
        for shard in plan["shards"]  # type: ignore[union-attr]
        if nodeid in shard["nodeids"] or nodeid.split("::", 1)[0] in shard["files"]
    ]
    assert len(owners) == 1, (nodeid, owners)
    return owners[0]


def test_plan_splits_only_files_heavier_than_half_a_shard(tmp_path: Path) -> None:
    heavy = [f"tests/test_heavy.py::test_case[{index:02d}]" for index in range(24)]
    light = [f"tests/test_light_{index:02d}.py::test_case" for index in range(40)]
    nodeids = sorted(heavy + light)
    durations = {nodeid: 50.0 for nodeid in heavy} | {nodeid: 10.0 for nodeid in light}

    plan = _plan_for(nodeids, durations, tmp_path)

    assert plan["schema_version"] == PLAN_SCHEMA
    assert plan["shard_count"] == SHARD_COUNT
    assert plan["split_files"] == ["tests/test_heavy.py"]
    owners = {nodeid: _owner_of(plan, nodeid) for nodeid in nodeids}
    # The heavy file is spread by test, every light file stays whole.
    assert len({owners[nodeid] for nodeid in heavy}) == SHARD_COUNT
    for shard in plan["shards"]:  # type: ignore[union-attr]
        assert "tests/test_heavy.py" not in shard["files"]
        assert all(nodeid in heavy for nodeid in shard["nodeids"])
        expected_paths = sorted(
            set(shard["files"]) | {nodeid.split("::", 1)[0] for nodeid in shard["nodeids"]}
        )
        assert shard["run_paths"] == expected_paths
    loads = [float(shard["estimated_duration_seconds"]) for shard in plan["shards"]]  # type: ignore[union-attr]
    assert max(loads) <= 1.2 * (sum(loads) / len(loads))


def test_recorded_heavy_test_durations_steer_split_assignment(tmp_path: Path) -> None:
    giant = "tests/test_scale.py::test_case[giant]"
    small = [f"tests/test_scale.py::test_case[small{index}]" for index in range(9)]
    light = [f"tests/test_light_{index:02d}.py::test_case" for index in range(30)]
    nodeids = sorted([giant, *small, *light])
    durations = {giant: 600.0} | {nodeid: 10.0 for nodeid in small} | {
        nodeid: 30.0 for nodeid in light
    }

    plan = _plan_for(nodeids, durations, tmp_path)

    giant_shard = plan["shards"][_owner_of(plan, giant)]  # type: ignore[index]
    # A uniform per-test estimate (69 s) would pack other work beside the
    # 600 s test; the recorded duration isolates it on its own shard.
    assert giant_shard["nodeids"] == [giant]
    assert giant_shard["files"] == []
    assert giant_shard["expected_test_count"] == 1
    assert giant_shard["estimated_duration_seconds"] == 600.0


def test_baseline_records_heavy_tests_and_rejects_inconsistent_rows(
    tmp_path: Path,
) -> None:
    nodeids = [
        "tests/test_a.py::test_fast",
        "tests/test_a.py::test_slow",
        "tests/test_b.py::test_edge",
        "tests/test_b.py::test_under",
    ]
    junit = tmp_path / "junit.xml"
    _junit(
        junit,
        nodeids,
        durations={
            nodeids[0]: 0.5,
            nodeids[1]: 12.0,
            nodeids[2]: HEAVY_TEST_SECONDS,
            nodeids[3]: HEAVY_TEST_SECONDS - 0.1,
        },
    )

    baseline = build_duration_baseline(
        junit=junit, source_sha=SHA, source_run_id=SOURCE_RUN_ID
    )

    assert baseline["schema_version"] == BASELINE_SCHEMA
    assert baseline["heavy_test_threshold_seconds"] == HEAVY_TEST_SECONDS
    assert baseline["heavy_tests"] == [
        {"nodeid": nodeids[1], "duration_seconds": 12.0},
        {"nodeid": nodeids[2], "duration_seconds": HEAVY_TEST_SECONDS},
    ]
    _validate_baseline(baseline)

    tampered = [
        {"nodeid": "tests/test_missing.py::test_x", "duration_seconds": 12.0},
        {"nodeid": nodeids[1], "duration_seconds": 13.0},
        {"nodeid": nodeids[3], "duration_seconds": HEAVY_TEST_SECONDS - 0.1},
    ]
    for row in tampered:
        payload = json.loads(json.dumps(baseline))
        payload["heavy_tests"] = [row]
        with pytest.raises(FullLaneShardError, match="duration_baseline_heavy_test_invalid"):
            _validate_baseline(payload)
    duplicated = json.loads(json.dumps(baseline))
    duplicated["heavy_tests"].append(dict(duplicated["heavy_tests"][0]))
    with pytest.raises(FullLaneShardError, match="duration_baseline_heavy_test_invalid"):
        _validate_baseline(duplicated)


def test_plan_refuses_a_collection_too_small_to_fill_every_shard(
    tmp_path: Path,
) -> None:
    nodeids = [f"tests/test_{index:02d}.py::test_case" for index in range(SHARD_COUNT - 1)]

    with pytest.raises(FullLaneShardError, match="shard_plan_empty_shard"):
        _plan_for(nodeids, {}, tmp_path)


def test_verify_shard_rejects_a_split_test_owned_by_another_shard(
    tmp_path: Path,
) -> None:
    heavy = [f"tests/test_heavy.py::test_case[{index:02d}]" for index in range(24)]
    light = [f"tests/test_light_{index:02d}.py::test_case" for index in range(40)]
    nodeids = sorted(heavy + light)
    durations = {nodeid: 50.0 for nodeid in heavy} | {nodeid: 10.0 for nodeid in light}
    junit = tmp_path / "measured-junit.xml"
    _junit(junit, nodeids, durations=durations)
    baseline_path = tmp_path / "full-test-lane-duration-baseline.json"
    _write_json(
        baseline_path,
        build_duration_baseline(junit=junit, source_sha=SHA, source_run_id=SOURCE_RUN_ID),
    )
    planned_path = tmp_path / "full-test-lane-planned.json"
    _write_json(planned_path, _collection(nodeids, phase="planned"))
    plan = build_shard_plan(
        planned=_collection(nodeids, phase="planned"),
        duration_baseline=json.loads(baseline_path.read_text(encoding="utf-8")),
        repository_sha=SHA,
    )
    plan_path = tmp_path / "full-test-lane-shard-plan.json"
    _write_json(plan_path, plan)
    expected = _plan_expected_nodeids(plan=plan, planned_nodeids=nodeids, shard_index=0)
    foreign = next(nodeid for nodeid in heavy if nodeid not in expected)
    executed = tmp_path / "full-test-lane-shard-executed.json"
    shard_junit = tmp_path / "full-test-lane-shard-junit.xml"
    widened = sorted([*expected, foreign])
    _write_json(executed, _collection(widened, phase="executed"))
    _junit(shard_junit, widened)

    receipt = verify_shard(
        planned_path=planned_path,
        duration_baseline_path=baseline_path,
        plan_path=plan_path,
        executed_path=executed,
        junit_path=shard_junit,
        repository_sha=SHA,
        shard_index=0,
    )

    assert receipt["status"] == "blocked"
    assert "shard_executed_nodeids_mismatch" in receipt["blockers"]
    assert "shard_junit_nodeids_mismatch" in receipt["blockers"]


class _Hooks:
    def __init__(self) -> None:
        self.deselected: list[object] = []

    def pytest_deselected(self, *, items: list[object]) -> None:
        self.deselected.extend(items)


def test_plugin_deselects_only_split_tests_planned_for_other_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heavy = [f"tests/test_heavy.py::test_case[{index:02d}]" for index in range(24)]
    light = [f"tests/test_light_{index:02d}.py::test_case" for index in range(40)]
    nodeids = sorted(heavy + light)
    durations = {nodeid: 50.0 for nodeid in heavy} | {nodeid: 10.0 for nodeid in light}
    plan = _plan_for(nodeids, durations, tmp_path)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    shard = plan["shards"][3]  # type: ignore[index]
    own_paths = set(shard["run_paths"])
    # Everything pytest collects for this shard's paths, plus one test the
    # plan never saw: an unplanned test must run and fail verification,
    # never vanish through selection.
    unplanned = "tests/test_heavy.py::test_case[late]"
    collected = [
        SimpleNamespace(nodeid=nodeid, user_properties=[])
        for nodeid in [*nodeids, unplanned]
        if nodeid.split("::", 1)[0] in own_paths
    ]
    hooks = _Hooks()
    monkeypatch.setenv(EVIDENCE_PLUGIN.SHARD_PLAN_ENV, str(plan_path))
    monkeypatch.setenv(EVIDENCE_PLUGIN.SHARD_INDEX_ENV, "3")

    items = list(collected)
    EVIDENCE_PLUGIN.pytest_collection_modifyitems(
        config=SimpleNamespace(hook=hooks), items=items
    )

    expected = _plan_expected_nodeids(plan=plan, planned_nodeids=nodeids, shard_index=3)
    assert sorted(item.nodeid for item in items) == sorted([*expected, unplanned])
    assert {item.nodeid for item in hooks.deselected}.isdisjoint(expected)
    assert all(
        item.nodeid in heavy and item.nodeid not in shard["nodeids"]
        for item in hooks.deselected
    )
    assert all(
        (EVIDENCE_PLUGIN.NODEID_PROPERTY, item.nodeid) in item.user_properties
        for item in items
    )


def test_plugin_refuses_an_unusable_shard_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EVIDENCE_PLUGIN.SHARD_PLAN_ENV, str(tmp_path / "missing.json"))
    monkeypatch.setenv(EVIDENCE_PLUGIN.SHARD_INDEX_ENV, "0")
    with pytest.raises(pytest.UsageError, match="full-lane shard selection"):
        EVIDENCE_PLUGIN.pytest_collection_modifyitems(
            config=SimpleNamespace(hook=_Hooks()), items=[]
        )

    _write_json(tmp_path / "plan.json", {"schema_version": PLAN_SCHEMA, "shards": []})
    monkeypatch.setenv(EVIDENCE_PLUGIN.SHARD_PLAN_ENV, str(tmp_path / "plan.json"))
    with pytest.raises(pytest.UsageError, match="full-lane shard selection"):
        EVIDENCE_PLUGIN.pytest_collection_modifyitems(
            config=SimpleNamespace(hook=_Hooks()), items=[]
        )


def test_plugin_and_planner_share_one_plan_schema() -> None:
    assert EVIDENCE_PLUGIN.SHARD_PLAN_SCHEMA == PLAN_SCHEMA


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


def test_plan_is_disjoint_complete_and_deterministic(
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
    assert plan["strategy"] == "lpt_heavy_file_split_serial_shards"
    assert plan["shard_count"] == SHARD_COUNT
    owners: dict[str, int] = {}
    union: list[str] = []
    for shard in plan["shards"]:  # type: ignore[index]
        index = int(shard["index"])
        expected = _plan_expected_nodeids(
            plan=plan, planned_nodeids=nodeids, shard_index=index
        )
        assert expected
        assert len(expected) == shard["expected_test_count"]
        assert hashlib.sha256("\n".join(expected).encode()).hexdigest() == shard[
            "expected_nodeids_sha256"
        ]
        union.extend(expected)
        for nodeid in expected:
            assert nodeid not in owners
            owners[nodeid] = index
        # A split file is never also scheduled whole.
        split_paths = {nodeid.split("::", 1)[0] for nodeid in shard["nodeids"]}
        assert split_paths <= set(plan["split_files"])  # type: ignore[arg-type]
        assert not split_paths & set(shard["files"])

    assert set(union) == set(nodeids)
    assert len(union) == len(nodeids)
    assert set(owners) == set(nodeids)
    for shard in plan["shards"]:  # type: ignore[index]
        assert not set(shard["files"]) & set(plan["split_files"])  # type: ignore[arg-type]


def test_committed_duration_baseline_is_exact_green_run_evidence() -> None:
    baseline_path = ROOT / ".github" / "full-test-lane-duration-baseline.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    rows = _validate_baseline(payload)

    # Run 38193701899 is the 2026-10-12 production promotion whose 68/51/35/28
    # minute shards exposed the stale 2026-08 baseline (12866 tests, 1124 files).
    assert payload["schema_version"] == BASELINE_SCHEMA
    assert payload["source_repository_sha"] == (
        "e9af5888afc5efa25dfcfc8742fcfed1e30e7ffd"
    )
    assert payload["source_run_id"] == 38193701899
    assert payload["test_count"] == 34524
    assert payload["file_count"] == len(rows) == 2276
    assert payload["heavy_tests"]


def test_committed_baseline_balances_the_measured_lane() -> None:
    """Pin the reason the planner exists: shards finish together.

    Replays the committed measured baseline as a planned collection. Every
    shard must land within 10% of the mean, which the 2026-10-12 lane (shards
    of 41.7/58.4/18.2/26.0 test-minutes against a 16.3 minute estimate) failed.
    """
    payload = json.loads(
        (ROOT / ".github" / "full-test-lane-duration-baseline.json").read_text(
            encoding="utf-8"
        )
    )
    heavy_by_path: dict[str, list[str]] = {}
    for row in payload["heavy_tests"]:
        heavy_by_path.setdefault(row["nodeid"].split("::", 1)[0], []).append(row["nodeid"])
    nodeids: list[str] = []
    for row in payload["files"]:
        recorded = heavy_by_path.get(row["path"], [])
        nodeids.extend(recorded)
        nodeids.extend(
            f"{row['path']}::replayed_{index}"
            for index in range(int(row["test_count"]) - len(recorded))
        )
    nodeids.sort()

    plan = build_shard_plan(
        planned=_collection(nodeids, phase="planned"),
        duration_baseline=payload,
        repository_sha=SHA,
    )

    loads = [float(shard["estimated_duration_seconds"]) for shard in plan["shards"]]
    assert sum(loads) == pytest.approx(payload["total_duration_seconds"], rel=1e-6)
    assert max(loads) <= 1.10 * (sum(loads) / len(loads))
    assert sum(int(shard["expected_test_count"]) for shard in plan["shards"]) == len(nodeids)


def test_workflow_retains_committed_baseline_when_collection_fails(tmp_path: Path) -> None:
    """An early collection error must not erase the shard's source telemetry."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/full-test-lane.yml").read_text())
    steps = workflow["jobs"]["full-pytest-shard"]["steps"]
    collection_index = next(
        index for index, step in enumerate(steps) if step["name"] == "Collect full lane"
    )
    workspace = tmp_path / "workspace"
    (workspace / ".github").mkdir(parents=True)
    baseline = (ROOT / ".github/full-test-lane-duration-baseline.json").read_bytes()
    (workspace / ".github/full-test-lane-duration-baseline.json").write_bytes(baseline)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # No pytest collection, install or provider call: preserve its actual exit 2.
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 2\n")
    uv.chmod(0o755)
    env = {"PATH": str(bin_dir) + os.pathsep + os.defpath}
    for step in steps[:collection_index]:
        script = step.get("run", "")
        if "cp .github/full-test-lane-duration-baseline.json" in script:
            result = subprocess.run(
                ["bash", "-e", "-c", script.replace("${{ runner.temp }}", str(runner_temp))],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [
            "bash",
            "-e",
            "-c",
            steps[collection_index]["run"].replace("${{ runner.temp }}", str(runner_temp)),
        ],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    staged = runner_temp / "blueprint-ci/full-test-lane-duration-baseline.json"
    assert staged.read_bytes() == baseline
    assert not (staged.parent / "full-test-lane-shard-plan.json").exists()
    assert not (staged.parent / "full-test-lane-shard-executed.json").exists()


def _seed_shard_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    planned_path, baseline_path, plan_path, plan = _seed_plan(tmp_path)
    nodeids = _nodeids()
    shard_root = tmp_path / "shard-artifacts"
    for index in range(SHARD_COUNT):
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
    assert len(receipt["shards"]) == SHARD_COUNT
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

    _junit(junit, expected)
    tree = ET.parse(junit)
    tree.getroot().find("testsuite").set("skipped", "1")
    ET.SubElement(next(tree.getroot().iter("testcase")), "skipped", {
        "message": "mandatory disposable hosted Linux proof unavailable",
    })
    tree.write(junit)
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
    assert "shard_junit_skipped:1" in receipt["blockers"]

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
