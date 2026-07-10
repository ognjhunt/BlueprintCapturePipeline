from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts import collect_github_actions_evidence as collector


def test_collect_github_actions_evidence_includes_junit_counts_and_job_summaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """
<testsuites>
  <testsuite name="unit" tests="2" failures="0" errors="0" skipped="1" time="1.25" />
  <testsuite name="integration" tests="3" failures="0" errors="0" skipped="0" time="2.5" />
</testsuites>
""".strip(),
        encoding="utf-8",
    )

    def fake_run_gh(repo: str, run_id: str) -> dict[str, Any]:
        assert repo == "ognjhunt/BlueprintCapturePipeline"
        assert run_id == "123"
        return {
            "workflowName": "Full Test Lane",
            "status": "completed",
            "conclusion": "success",
            "url": "https://github.test/actions/runs/123",
            "headSha": "abc123",
            "event": "workflow_dispatch",
            "createdAt": "2026-07-07T00:00:00Z",
            "updatedAt": "2026-07-07T00:10:00Z",
            "jobs": [
                {
                    "name": "full-test-lane",
                    "status": "completed",
                    "conclusion": "success",
                    "startedAt": "2026-07-07T00:00:00Z",
                    "completedAt": "2026-07-07T00:10:00Z",
                    "url": "https://github.test/actions/runs/123/job/1",
                }
            ],
        }

    monkeypatch.setattr(collector, "_run_gh", fake_run_gh)

    evidence = collector.collect_evidence(
        repo="ognjhunt/BlueprintCapturePipeline",
        run_id="123",
        evidence_id="pipeline_full_test_lane_ci_evidence",
        junit=junit,
    )

    assert evidence["schema_version"] == collector.SCHEMA_VERSION
    assert evidence["evidence_id"] == "pipeline_full_test_lane_ci_evidence"
    assert evidence["workflow_name"] == "Full Test Lane"
    assert evidence["conclusion"] == "success"
    assert evidence["head_sha"] == "abc123"
    assert evidence["junit_artifact_name"] == "junit.xml"
    assert str(tmp_path) not in str(evidence)
    assert evidence["test_counts"] == {
        "tests": 5,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
        "time_seconds": 3.75,
    }
    assert evidence["jobs"] == [
        {
            "name": "full-test-lane",
            "status": "completed",
            "conclusion": "success",
            "started_at": "2026-07-07T00:00:00Z",
            "completed_at": "2026-07-07T00:10:00Z",
            "url": "https://github.test/actions/runs/123/job/1",
        }
    ]
