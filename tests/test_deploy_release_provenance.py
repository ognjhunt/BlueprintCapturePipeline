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


SHA = "a" * 40
REPOSITORY = "ognjhunt/BlueprintCapturePipeline"
RUN_ID = 123456


def _metadata() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    run: dict[str, object] = {
        "id": RUN_ID,
        "head_sha": SHA,
        "status": "completed",
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "name": "Full Test Lane",
        "display_title": "Full Test Lane / production_deployment_promotion",
        "path": ".github/workflows/full-test-lane.yml@refs/heads/main",
        "repository": {"full_name": REPOSITORY},
        "head_repository": {"full_name": REPOSITORY},
    }
    jobs: dict[str, object] = {
        "jobs": [
            {
                "name": "Full pytest lane on CPU runner",
                "status": "completed",
                "conclusion": "success",
                "steps": [
                    {"name": "Collect full lane", "conclusion": "success"},
                    {"name": "Run full lane", "conclusion": "success"},
                    {
                        "name": "Verify exact full-lane collection",
                        "conclusion": "success",
                    },
                ],
            }
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
    return run, jobs, artifacts


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
    run, jobs, artifacts = _metadata()
    artifact = validate_run_metadata(
        run=run,
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
        (run, "display_title", "Full Test Lane / cross_cutting_diagnostic"),
        (artifacts["artifacts"][0], "expired", True),  # type: ignore[index]
        (jobs["jobs"][0]["steps"][1], "conclusion", "failure"),  # type: ignore[index]
    )
    for target, key, invalid_value in mutations:
        previous = target[key]  # type: ignore[index]
        target[key] = invalid_value  # type: ignore[index]
        with pytest.raises(ProvenanceError):
            validate_run_metadata(
                run=run,
                jobs=jobs,
                artifacts=artifacts,
                expected_repository=REPOSITORY,
                expected_sha=SHA,
                expected_run_id=RUN_ID,
            )
        target[key] = previous  # type: ignore[index]


def test_downloaded_artifact_binds_exact_collection_to_green_junit(
    tmp_path: Path,
) -> None:
    nodeids = ["tests/test_one.py::test_a", "tests/test_two.py::test_b"]
    digest = hashlib.sha256("\n".join(nodeids).encode()).hexdigest()
    for phase in ("planned", "executed"):
        (tmp_path / f"full-test-lane-{phase}.json").write_text(
            json.dumps(
                {
                    "schema_version": "blueprint_full_lane_collection.v1",
                    "phase": phase,
                    "test_count": len(nodeids),
                    "nodeids_sha256": digest,
                    "nodeids": nodeids,
                }
            ),
            encoding="utf-8",
        )
    junit = tmp_path / "full-test-lane-junit.xml"
    junit.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="test_one" name="test_a"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_one.py::test_a"/>'
        "</properties></testcase>"
        '<testcase classname="test_two" name="test_b"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_two.py::test_b"/>'
        "</properties></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
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

    result = validate_downloaded_artifact(tmp_path, expected_sha=SHA)
    assert result["test_count"] == 2
    assert result["skipped_count"] == 0
    assert result["nodeids_sha256"] == digest

    cpu_path = tmp_path / "cpu_full.json"
    cpu_bytes = cpu_path.read_bytes()
    cpu_path.unlink()
    with pytest.raises(ProvenanceError, match="cpu_full.json"):
        validate_downloaded_artifact(tmp_path, expected_sha=SHA)
    cpu_path.write_bytes(cpu_bytes)

    junit.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1">'
        '<testcase classname="test_one" name="test_a"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_one.py::test_a"/>'
        '</properties><skipped message="torch missing"/></testcase>'
        '<testcase classname="test_two" name="test_b"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_two.py::test_b"/>'
        "</properties></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="full_lane_junit_has_skips:1"):
        validate_downloaded_artifact(tmp_path, expected_sha=SHA)

    junit.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="test_one" name="test_a"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_one.py::test_a"/>'
        "</properties></testcase>"
        '<testcase classname="test_two" name="test_b"><properties>'
        '<property name="blueprint_nodeid" value="tests/test_two.py::test_b"/>'
        "</properties></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
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

    junit.write_text(
        '<testsuites><testsuite tests="2" failures="1" errors="0"/></testsuites>',
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceError, match="full_lane_junit_not_green"):
        validate_downloaded_artifact(tmp_path, expected_sha=SHA)
