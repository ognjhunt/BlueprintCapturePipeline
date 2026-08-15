from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_deploy_release_provenance import (
    ProvenanceError,
    parse_run_url,
    repository_from_remote,
    validate_downloaded_artifact,
    validate_run_metadata,
)
from scripts.build_cpu_full_lane_evidence import build_cpu_full_lane_evidence
from scripts.full_lane_sharding import (
    _plan_expected_nodeids,
    aggregate_shards,
    build_duration_baseline,
    build_shard_plan,
    verify_shard,
)


SHA = "a" * 40
REPOSITORY = "ognjhunt/BlueprintCapturePipeline"
RUN_ID = 123456


def _metadata() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    run: dict[str, object] = {
        "id": RUN_ID,
        "head_sha": SHA,
        "status": "completed",
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "name": "Full Test Lane / production_deployment_promotion",
        "display_title": "Full Test Lane / production_deployment_promotion",
        "path": ".github/workflows/full-test-lane.yml@refs/heads/main",
        "workflow_id": 42,
        "repository": {"full_name": REPOSITORY},
        "head_repository": {"full_name": REPOSITORY},
    }
    workflow: dict[str, object] = {
        "id": 42,
        "name": "Full Test Lane",
        "path": ".github/workflows/full-test-lane.yml",
        "state": "active",
    }
    jobs: dict[str, object] = {
        "jobs": [
            {
                "name": "Full pytest lane on CPU runner",
                "status": "completed",
                "conclusion": "success",
                "steps": [
                    {
                        "name": "Aggregate exact full-lane shards",
                        "conclusion": "success",
                    },
                    {
                        "name": "Build fail-closed CPU full-lane evidence",
                        "conclusion": "success",
                    },
                    {"name": "Upload full lane report", "conclusion": "success"},
                ],
            },
            *[
                {
                    "name": f"Full pytest shard {index} of 4",
                    "status": "completed",
                    "conclusion": "success",
                    "steps": [
                        {"name": "Collect full lane", "conclusion": "success"},
                        {
                            "name": "Build deterministic full-lane shard plan",
                            "conclusion": "success",
                        },
                        {"name": "Run full lane shard", "conclusion": "success"},
                        {
                            "name": "Verify exact full-lane shard",
                            "conclusion": "success",
                        },
                        {
                            "name": "Upload full-lane shard evidence",
                            "conclusion": "success",
                        },
                    ],
                }
                for index in range(4)
            ],
        ]
    }
    artifacts: dict[str, object] = {
        "artifacts": [
            {
                "id": 99,
                "name": f"full-test-lane-{RUN_ID}",
                "expired": False,
                "size_in_bytes": 2048,
            }
        ]
    }
    return run, workflow, jobs, artifacts


def test_canonical_run_url_and_remote_parsing_reject_lookalikes() -> None:
    assert parse_run_url(f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}") == (
        REPOSITORY,
        RUN_ID,
    )
    assert (
        repository_from_remote("https://github.com/ognjhunt/BlueprintCapturePipeline.git")
        == REPOSITORY
    )
    assert (
        repository_from_remote("git@github.com:ognjhunt/BlueprintCapturePipeline.git") == REPOSITORY
    )

    for invalid in (
        f"http://github.com/{REPOSITORY}/actions/runs/{RUN_ID}",
        f"https://github.example/{REPOSITORY}/actions/runs/{RUN_ID}",
        f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}?trusted=true",
        f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}/job/7",
    ):
        with pytest.raises(ProvenanceError):
            parse_run_url(invalid)


def test_run_metadata_requires_exact_sha_workflow_job_steps_and_artifact() -> None:
    run, workflow, jobs, artifacts = _metadata()
    artifact = validate_run_metadata(
        run=run,
        workflow=workflow,
        jobs=jobs,
        artifacts=artifacts,
        expected_repository=REPOSITORY,
        expected_sha=SHA,
        expected_run_id=RUN_ID,
    )
    assert artifact["id"] == 99

    mutations = (
        (run, "head_sha", "b" * 40),
        (run, "conclusion", "failure"),
        (run, "path", ".github/workflows/ci.yml"),
        (run, "event", "push"),
        (run, "name", "Full Test Lane"),
        (run, "display_title", "Full Test Lane / cross_cutting_diagnostic"),
        (workflow, "name", "CI"),
        (workflow, "path", ".github/workflows/ci.yml"),
        (artifacts["artifacts"][0], "expired", True),  # type: ignore[index]
        (jobs["jobs"][0]["steps"][1], "conclusion", "failure"),  # type: ignore[index]
        (jobs["jobs"][2]["steps"][2], "conclusion", "failure"),  # type: ignore[index]
    )
    for target, key, invalid_value in mutations:
        previous = target[key]  # type: ignore[index]
        target[key] = invalid_value  # type: ignore[index]
        with pytest.raises(ProvenanceError):
            validate_run_metadata(
                run=run,
                workflow=workflow,
                jobs=jobs,
                artifacts=artifacts,
                expected_repository=REPOSITORY,
                expected_sha=SHA,
                expected_run_id=RUN_ID,
            )
        target[key] = previous  # type: ignore[index]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _collection(nodeids: list[str], *, phase: str) -> dict[str, object]:
    return {
        "schema_version": "blueprint_full_lane_collection.v1",
        "phase": phase,
        "test_count": len(nodeids),
        "nodeids_sha256": hashlib.sha256("\n".join(nodeids).encode()).hexdigest(),
        "nodeids": nodeids,
    }


def _write_junit(path: Path, nodeids: list[str]) -> None:
    testcases = "".join(
        '<testcase classname="seed" name="case" time="1"><properties>'
        f'<property name="blueprint_nodeid" value="{nodeid}"/>'
        "</properties></testcase>"
        for nodeid in nodeids
    )
    path.write_text(
        '<testsuites><testsuite name="seed" '
        f'tests="{len(nodeids)}" failures="0" errors="0" skipped="0" time="1">'
        f"{testcases}</testsuite></testsuites>",
        encoding="utf-8",
    )


def _seed_sharded_artifact(tmp_path: Path) -> list[str]:
    nodeids = [f"tests/test_{index}.py::test_case" for index in range(4)]
    source = tmp_path / "source"
    planned = source / "full-test-lane-planned.json"
    baseline_junit = source / "baseline-junit.xml"
    baseline = source / "full-test-lane-duration-baseline.json"
    plan_path = source / "full-test-lane-shard-plan.json"
    planned_payload = _collection(nodeids, phase="planned")
    _write_json(planned, planned_payload)
    _write_junit(baseline_junit, nodeids)
    baseline_payload = build_duration_baseline(
        junit=baseline_junit,
        source_sha=SHA,
        source_run_id=RUN_ID,
    )
    _write_json(baseline, baseline_payload)
    plan = build_shard_plan(
        planned=planned_payload,
        duration_baseline=baseline_payload,
        repository_sha=SHA,
    )
    _write_json(plan_path, plan)
    shard_root = source / "shard-artifacts"
    for index in range(4):
        shard_dir = shard_root / f"shard-{index}"
        shard_dir.mkdir(parents=True)
        expected = _plan_expected_nodeids(
            plan=plan, planned_nodeids=nodeids, shard_index=index
        )
        for source_path in (planned, baseline, plan_path):
            (shard_dir / source_path.name).write_bytes(source_path.read_bytes())
        executed = shard_dir / "full-test-lane-shard-executed.json"
        junit = shard_dir / "full-test-lane-shard-junit.xml"
        _write_json(executed, _collection(expected, phase="executed"))
        _write_junit(junit, expected)
        receipt = verify_shard(
            planned_path=shard_dir / planned.name,
            duration_baseline_path=shard_dir / baseline.name,
            plan_path=shard_dir / plan_path.name,
            executed_path=executed,
            junit_path=junit,
            repository_sha=SHA,
            shard_index=index,
        )
        _write_json(
            shard_dir / "full-test-lane-shard-verification.json", receipt
        )
    aggregate_shards(
        shard_root=shard_root, output_dir=tmp_path, repository_sha=SHA
    )
    junit = tmp_path / "full-test-lane-junit.xml"
    (tmp_path / "cpu_full.json").write_text(
        json.dumps(
            build_cpu_full_lane_evidence(
                planned=tmp_path / "full-test-lane-planned.json",
                executed=tmp_path / "full-test-lane-executed.json",
                junit=junit,
                repository_sha=SHA,
            )
        ),
        encoding="utf-8",
    )
    return nodeids


def test_downloaded_artifact_binds_every_shard_to_green_aggregate(
    tmp_path: Path,
) -> None:
    nodeids = _seed_sharded_artifact(tmp_path)
    digest = hashlib.sha256("\n".join(nodeids).encode()).hexdigest()

    result = validate_downloaded_artifact(tmp_path, expected_sha=SHA)
    assert result["test_count"] == 4
    assert result["skipped_count"] == 0
    assert result["nodeids_sha256"] == digest
    assert result["shard_count"] == 4

    cpu_path = tmp_path / "cpu_full.json"
    cpu_bytes = cpu_path.read_bytes()
    cpu_path.unlink()
    with pytest.raises(ProvenanceError, match="cpu_full.json"):
        validate_downloaded_artifact(tmp_path, expected_sha=SHA)
    cpu_path.write_bytes(cpu_bytes)

    junit = tmp_path / "full-test-lane-junit.xml"
    junit_bytes = junit.read_bytes()
    junit.write_bytes(junit_bytes + b"\n<!-- tampered -->\n")
    with pytest.raises(ProvenanceError, match="full_lane_shard_evidence_invalid"):
        validate_downloaded_artifact(tmp_path, expected_sha=SHA)
    junit.write_bytes(junit_bytes)

    cpu_payload = build_cpu_full_lane_evidence(
        planned=tmp_path / "full-test-lane-planned.json",
        executed=tmp_path / "full-test-lane-executed.json",
        junit=junit,
        repository_sha=SHA,
    )
    cpu_payload["skipped_count"] = 7
    (tmp_path / "cpu_full.json").write_text(json.dumps(cpu_payload), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="cpu_full_evidence_mismatch"):
        validate_downloaded_artifact(tmp_path, expected_sha=SHA)
