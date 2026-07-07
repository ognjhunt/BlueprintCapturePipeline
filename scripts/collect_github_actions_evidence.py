#!/usr/bin/env python3
"""Collect a bounded GitHub Actions run summary for launch-readiness packets."""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "blueprint.github_actions_evidence.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_gh(repo: str, run_id: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "gh",
            "run",
            "view",
            run_id,
            "--repo",
            repo,
            "--json",
            "status,conclusion,url,headSha,workflowName,event,createdAt,updatedAt,jobs",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _junit_counts(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    if root.tag == "testsuites":
        suites = list(root)
        counts = {
            key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        counts["time_seconds"] = round(
            sum(float(suite.attrib.get("time", "0")) for suite in suites),
            2,
        )
        return counts
    counts = {
        key: int(float(root.attrib.get(key, "0")))
        for key in ("tests", "failures", "errors", "skipped")
    }
    counts["time_seconds"] = round(float(root.attrib.get("time", "0")), 2)
    return counts


def _job_summaries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return []
    summaries: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        summaries.append(
            {
                "name": job.get("name"),
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "started_at": job.get("startedAt"),
                "completed_at": job.get("completedAt"),
                "url": job.get("url"),
            }
        )
    return summaries


def collect_evidence(
    *,
    repo: str,
    run_id: str,
    evidence_id: str,
    junit: Path | None = None,
) -> dict[str, Any]:
    payload = _run_gh(repo, run_id)
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "evidence_id": evidence_id,
        "repo": repo,
        "run_id": run_id,
        "workflow_name": payload.get("workflowName"),
        "status": payload.get("status"),
        "conclusion": payload.get("conclusion"),
        "url": payload.get("url"),
        "head_sha": payload.get("headSha"),
        "event": payload.get("event"),
        "created_at": payload.get("createdAt"),
        "updated_at": payload.get("updatedAt"),
        "jobs": _job_summaries(payload),
    }
    if junit is not None:
        evidence["junit_path"] = str(junit)
        evidence["test_counts"] = _junit_counts(junit)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repo, e.g. ognjhunt/BlueprintCapturePipeline")
    parser.add_argument("--run-id", required=True, help="GitHub Actions run id")
    parser.add_argument("--evidence-id", required=True, help="Stable packet evidence id")
    parser.add_argument("--junit", type=Path, help="Optional downloaded JUnit XML path")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    evidence = collect_evidence(
        repo=args.repo,
        run_id=args.run_id,
        evidence_id=args.evidence_id,
        junit=args.junit.resolve() if args.junit else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[github-actions-evidence] id={args.evidence_id}")
    print(f"[github-actions-evidence] conclusion={evidence.get('conclusion')}")
    print(f"[github-actions-evidence] output={args.output}")


if __name__ == "__main__":
    main()
