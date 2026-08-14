#!/usr/bin/env python3
"""Verify deploy provenance against the canonical successful GitHub full lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from defusedxml import ElementTree as ET

if __package__ in {None, ""}:  # Direct ``python scripts/...`` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_full_lane_collection import verify as verify_full_lane_collection
from scripts.build_cpu_full_lane_evidence import validate_cpu_full_lane_evidence


RUN_URL_PATTERN = re.compile(r"^/([^/]+)/([^/]+)/actions/runs/([1-9][0-9]*)/?$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
CANONICAL_WORKFLOW_PATH = ".github/workflows/full-test-lane.yml"
CANONICAL_WORKFLOW_NAME = "Full Test Lane"
CANONICAL_PRODUCTION_DISPLAY_TITLE = (
    "Full Test Lane / production_deployment_promotion"
)
CANONICAL_JOB_NAME = "Full pytest lane on CPU runner"
REQUIRED_SUCCESSFUL_STEPS = {
    "Collect full lane",
    "Run full lane",
    "Verify exact full-lane collection",
}


class ProvenanceError(RuntimeError):
    """Raised when release provenance is absent, ambiguous, or untrusted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run_url(value: str) -> tuple[str, int]:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.query
        or parsed.fragment
    ):
        raise ProvenanceError("full_lane_evidence_uri_not_canonical_github_run")
    match = RUN_URL_PATTERN.fullmatch(parsed.path)
    if not match:
        raise ProvenanceError("full_lane_evidence_uri_not_canonical_github_run")
    owner, repository, run_id = match.groups()
    return f"{owner}/{repository}", int(run_id)


def repository_from_remote(remote: str) -> str:
    value = remote.strip()
    patterns = (
        r"^https://github\.com/([^/]+/[^/]+?)(?:\.git)?$",
        r"^git@github\.com:([^/]+/[^/]+?)(?:\.git)?$",
        r"^ssh://git@github\.com/([^/]+/[^/]+?)(?:\.git)?$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value)
        if match:
            return match.group(1)
    raise ProvenanceError("origin_remote_is_not_canonical_github_repository")


def validate_run_metadata(
    *,
    run: Mapping[str, Any],
    jobs: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    expected_repository: str,
    expected_sha: str,
    expected_run_id: int,
) -> Mapping[str, Any]:
    blockers: list[str] = []
    if not SHA_PATTERN.fullmatch(expected_sha):
        blockers.append("expected_sha_invalid")
    if run.get("id") != expected_run_id:
        blockers.append("run_id_mismatch")
    if str(run.get("head_sha") or "") != expected_sha:
        blockers.append("run_head_sha_mismatch")
    if str(run.get("status") or "") != "completed":
        blockers.append("run_not_completed")
    if str(run.get("conclusion") or "") != "success":
        blockers.append("run_conclusion_not_success")
    if str(run.get("event") or "") != "workflow_dispatch":
        blockers.append("run_event_not_workflow_dispatch")
    if str(run.get("head_branch") or "") != "main":
        blockers.append("run_branch_not_main")
    if str(run.get("name") or "") != CANONICAL_WORKFLOW_NAME:
        blockers.append("workflow_name_mismatch")
    if str(run.get("display_title") or "") != CANONICAL_PRODUCTION_DISPLAY_TITLE:
        blockers.append("production_promotion_reason_mismatch")
    workflow_path = str(run.get("path") or "").split("@", 1)[0]
    if workflow_path != CANONICAL_WORKFLOW_PATH:
        blockers.append("workflow_path_mismatch")
    repository = run.get("repository")
    repository_name = (
        str(repository.get("full_name") or "") if isinstance(repository, Mapping) else ""
    )
    head_repository = run.get("head_repository")
    head_repository_name = (
        str(head_repository.get("full_name") or "") if isinstance(head_repository, Mapping) else ""
    )
    if repository_name.lower() != expected_repository.lower():
        blockers.append("run_repository_mismatch")
    if head_repository_name.lower() != expected_repository.lower():
        blockers.append("run_head_repository_mismatch")

    raw_jobs = jobs.get("jobs")
    job_rows = raw_jobs if isinstance(raw_jobs, list) else []
    canonical_jobs = [
        job
        for job in job_rows
        if isinstance(job, Mapping) and job.get("name") == CANONICAL_JOB_NAME
    ]
    if len(canonical_jobs) != 1:
        blockers.append("canonical_full_lane_job_count_invalid")
    else:
        canonical_job = canonical_jobs[0]
        if canonical_job.get("status") != "completed":
            blockers.append("canonical_full_lane_job_not_completed")
        if canonical_job.get("conclusion") != "success":
            blockers.append("canonical_full_lane_job_not_success")
        raw_steps = canonical_job.get("steps")
        steps = raw_steps if isinstance(raw_steps, list) else []
        step_results = {
            str(step.get("name") or ""): str(step.get("conclusion") or "")
            for step in steps
            if isinstance(step, Mapping)
        }
        for step_name in sorted(REQUIRED_SUCCESSFUL_STEPS):
            if step_results.get(step_name) != "success":
                blockers.append(f"required_step_not_success:{step_name}")

    artifact_name = f"full-test-lane-{expected_run_id}"
    raw_artifacts = artifacts.get("artifacts")
    artifact_rows = raw_artifacts if isinstance(raw_artifacts, list) else []
    matching_artifacts = [
        artifact
        for artifact in artifact_rows
        if isinstance(artifact, Mapping) and artifact.get("name") == artifact_name
    ]
    if len(matching_artifacts) != 1:
        blockers.append("canonical_full_lane_artifact_count_invalid")
        artifact: Mapping[str, Any] = {}
    else:
        artifact = matching_artifacts[0]
        if artifact.get("expired") is not False:
            blockers.append("canonical_full_lane_artifact_expired")
        if (
            not isinstance(artifact.get("size_in_bytes"), int)
            or int(artifact.get("size_in_bytes") or 0) <= 0
        ):
            blockers.append("canonical_full_lane_artifact_empty")

    if blockers:
        raise ProvenanceError(",".join(blockers))
    return artifact


def validate_downloaded_artifact(artifact_dir: Path, *, expected_sha: str) -> dict[str, Any]:
    planned = artifact_dir / "full-test-lane-planned.json"
    executed = artifact_dir / "full-test-lane-executed.json"
    junit = artifact_dir / "full-test-lane-junit.xml"
    cpu_evidence = artifact_dir / "cpu_full.json"
    required = (planned, executed, junit, cpu_evidence)
    symlinks = [path.name for path in required if path.is_symlink()]
    if symlinks:
        raise ProvenanceError("full_lane_artifact_symlinks_forbidden:" + ",".join(symlinks))
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ProvenanceError("full_lane_artifact_files_missing:" + ",".join(missing))
    collection_blockers = verify_full_lane_collection(planned, executed)
    if collection_blockers:
        raise ProvenanceError("full_lane_collection_invalid:" + ",".join(collection_blockers))

    planned_payload = json.loads(planned.read_text(encoding="utf-8"))
    try:
        junit_root = ET.parse(junit).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ProvenanceError("full_lane_junit_invalid") from exc
    suites = (
        [junit_root] if junit_root.tag == "testsuite" else list(junit_root.findall("testsuite"))
    )
    if not suites:
        raise ProvenanceError("full_lane_junit_has_no_suites")
    test_count = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    if failures or errors:
        raise ProvenanceError("full_lane_junit_not_green")
    if skipped:
        raise ProvenanceError(f"full_lane_junit_has_skips:{skipped}")
    if test_count != int(planned_payload["test_count"]):
        raise ProvenanceError("full_lane_junit_test_count_mismatch")
    try:
        cpu_payload = json.loads(cpu_evidence.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError("cpu_full_evidence_invalid") from exc
    if not isinstance(cpu_payload, Mapping):
        raise ProvenanceError("cpu_full_evidence_invalid")
    cpu_blockers = validate_cpu_full_lane_evidence(
        cpu_payload,
        planned=planned,
        executed=executed,
        junit=junit,
        repository_sha=expected_sha,
    )
    if cpu_blockers:
        raise ProvenanceError("cpu_full_evidence_mismatch:" + ",".join(cpu_blockers))
    return {
        "test_count": test_count,
        "skipped_count": skipped,
        "nodeids_sha256": planned_payload["nodeids_sha256"],
        "files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in required
        },
    }


def _run_json(argv: Sequence[str], *, cwd: Path) -> Mapping[str, Any]:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ProvenanceError(f"command_failed:{argv[0]}:{completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"command_returned_invalid_json:{argv[0]}") from exc
    if not isinstance(payload, Mapping):
        raise ProvenanceError(f"command_returned_non_object:{argv[0]}")
    return payload


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
        temporary_path = Path(handle.name)
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_live(
    *, root: Path, expected_sha: str, run_url: str, output_path: Path
) -> dict[str, Any]:
    run_repository, run_id = parse_run_url(run_url)
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if remote.returncode != 0:
        raise ProvenanceError("origin_remote_unavailable")
    expected_repository = repository_from_remote(remote.stdout)
    if run_repository.lower() != expected_repository.lower():
        raise ProvenanceError("run_url_repository_mismatch")

    api_base = f"repos/{expected_repository}/actions"
    run = _run_json(["gh", "api", f"{api_base}/runs/{run_id}"], cwd=root)
    jobs = _run_json(
        ["gh", "api", f"{api_base}/runs/{run_id}/jobs?per_page=100"],
        cwd=root,
    )
    artifacts = _run_json(
        ["gh", "api", f"{api_base}/runs/{run_id}/artifacts?per_page=100"],
        cwd=root,
    )
    artifact = validate_run_metadata(
        run=run,
        jobs=jobs,
        artifacts=artifacts,
        expected_repository=expected_repository,
        expected_sha=expected_sha,
        expected_run_id=run_id,
    )
    artifact_name = str(artifact["name"])
    with tempfile.TemporaryDirectory(prefix="blueprint-full-lane-") as temporary_dir:
        artifact_dir = Path(temporary_dir)
        completed = subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run_id),
                "--repo",
                expected_repository,
                "--name",
                artifact_name,
                "--dir",
                str(artifact_dir),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ProvenanceError("full_lane_artifact_download_failed:" + completed.stderr.strip())
        collection = validate_downloaded_artifact(artifact_dir, expected_sha=expected_sha)

    result = {
        "schema_version": "blueprint.deploy_release_provenance.v1",
        "status": "verified",
        "repository": expected_repository,
        "git_sha": expected_sha,
        "run_id": run_id,
        "run_url": run_url,
        "workflow_name": CANONICAL_WORKFLOW_NAME,
        "workflow_path": CANONICAL_WORKFLOW_PATH,
        "job_name": CANONICAL_JOB_NAME,
        "artifact_id": artifact["id"],
        "artifact_name": artifact_name,
        "collection": collection,
        "claim_boundary": {
            "canonical_full_lane_verified": True,
            "live_deployment_health_proven": False,
        },
    }
    _write_json_atomic(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_live(
            root=args.root.resolve(),
            expected_sha=args.expected_sha.strip(),
            run_url=args.run_url.strip(),
            output_path=args.output.resolve(),
        )
    except (OSError, UnicodeError, ValueError, ProvenanceError) as exc:
        print(f"[deploy-provenance] ERROR {exc}", file=sys.stderr)
        return 1
    print(
        "[deploy-provenance] verified "
        f"run={result['run_id']} tests={result['collection']['test_count']} "
        f"sha={result['git_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
