from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.run_sim_only_beta_deployment_parity_proof import (
    _load_env_file_values,
    _pipeline_intake_token,
    build_deployment_parity_proof,
)


def _fetcher(url: str, headers: Mapping[str, str] | None, timeout: int) -> dict[str, Any]:
    del headers, timeout
    if url.endswith("/health/ready"):
        return {
            "ok": True,
            "http_status": 200,
            "json": {"status": "ready", "blockers": []},
            "error": None,
        }
    if url.endswith("/health"):
        return {
            "ok": True,
            "http_status": 200,
            "json": {"ok": True, "token_configured": True},
            "error": None,
        }
    if url.endswith("/api/live-pipeline/intake-audit"):
        return {
            "ok": True,
            "http_status": 200,
            "json": {"status": "staged_for_control_plane", "input_blockers": []},
            "error": None,
        }
    raise AssertionError(f"unexpected URL: {url}")


def _git_probe(repo: Path) -> dict[str, Any]:
    head = "abc123"
    return {
        "path": str(repo),
        "head": head,
        "origin_main": head,
        "head_matches_origin_main": True,
        "worktree_clean": True,
        "dirty_entries_count": 0,
        "dirty_entries": [],
        "dirty_entries_truncated": False,
        "errors": [],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_deployment_parity_token_can_load_from_forwarding_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", raising=False)
    monkeypatch.delenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN", raising=False)
    env_file = tmp_path / "forwarding.env"
    env_file.write_text(
        "export BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN='file-token'\n",
        encoding="utf-8",
    )

    values = _load_env_file_values([env_file])

    assert (
        _pipeline_intake_token("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", values)
        == "file-token"
    )


def test_deployment_parity_process_env_wins_over_forwarding_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", "process-token")
    env_file = tmp_path / "forwarding.env"
    env_file.write_text(
        "export ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN='file-token'\n",
        encoding="utf-8",
    )

    values = _load_env_file_values([env_file])

    assert (
        _pipeline_intake_token("ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN", values)
        == "process-token"
    )


def test_deployment_parity_proof_passes_with_health_and_commit_parity() -> None:
    seen_headers: list[Mapping[str, str] | None] = []

    def capturing_fetcher(
        url: str,
        headers: Mapping[str, str] | None,
        timeout: int,
    ) -> dict[str, Any]:
        seen_headers.append(headers)
        return _fetcher(url, headers, timeout)

    report = build_deployment_parity_proof(
        webapp_url="https://paperclip.tryblueprint.io",
        pipeline_intake_url="https://pipeline.tryblueprint.io/api/live-pipeline/job-requests",
        pipeline_intake_token="secret-token",
        webapp_repo=Path("/repos/webapp"),
        pipeline_repo=Path("/repos/pipeline"),
        capture_repo=Path("/repos/capture"),
        webapp_deployed_commit="abc123",
        pipeline_deployed_commit="abc123",
        now_iso="2026-06-17T00:00:00+00:00",
        fetcher=capturing_fetcher,
        git_probe=_git_probe,
    )

    assert report["status"] == "verified"
    assert report["production_deployment_proven"] is True
    assert report["webapp_health_ready"] is True
    assert report["pipeline_intake_health_ready"] is True
    assert report["git_parity_proven"] is True
    assert report["simulator_execution_proven"] is False
    assert report["public_claim_upgrade_allowed"] is False
    assert report["proof_boundary"]["simulator_execution_proven"] is False
    assert report["proof_boundary"]["public_claim_upgrade_allowed"] is False
    readiness_key = "physical_robot_" "readiness_proven"
    assert readiness_key not in report
    assert readiness_key not in report["proof_boundary"]
    assert report["blockers"] == []
    assert any(
        headers and headers.get("Authorization") == "Bearer secret-token"
        for headers in seen_headers
    )
    assert "secret-token" not in str(report)


def test_deployment_parity_proof_infers_urls_from_route_forwarding_proof(
    tmp_path: Path,
) -> None:
    route_proof = tmp_path / "route-proof.json"
    _write_json(
        route_proof,
        {
            "webapp_route": {
                "remote_webapp_url": "https://paperclip.tryblueprint.io",
            },
            "forwarding_endpoint": {
                "endpoint_url": "https://paperclip.tryblueprint.io/api/live-pipeline/job-requests",
                "endpoint_url_source": "local_script_forward_url",
            },
        },
    )

    report = build_deployment_parity_proof(
        webapp_url="",
        pipeline_intake_url="",
        pipeline_intake_token="secret-token",
        webapp_repo=Path("/repos/webapp"),
        pipeline_repo=Path("/repos/pipeline"),
        webapp_deployed_commit="abc123",
        pipeline_deployed_commit="abc123",
        route_forwarding_proof_path=route_proof,
        fetcher=_fetcher,
        git_probe=_git_probe,
    )

    assert report["status"] == "verified"
    assert report["webapp_url"] == "https://paperclip.tryblueprint.io"
    assert report["pipeline_intake_url"] == "https://paperclip.tryblueprint.io"
    assert report["checks"]["route_forwarding_proof"] == {
        "path": str(route_proof),
        "load_error": None,
        "webapp_url": "https://paperclip.tryblueprint.io",
        "pipeline_intake_url": "https://paperclip.tryblueprint.io/api/live-pipeline/job-requests",
        "pipeline_intake_url_source": "local_script_forward_url",
    }


def test_deployment_parity_proof_accepts_authenticated_intake_audit_health() -> None:
    def same_host_fetcher(
        url: str,
        headers: Mapping[str, str] | None,
        timeout: int,
    ) -> dict[str, Any]:
        del timeout
        if url.endswith("/health/ready"):
            return {
                "ok": True,
                "http_status": 200,
                "json": {"status": "ready", "blockers": []},
                "error": None,
            }
        if url.endswith("/health"):
            return {
                "ok": True,
                "http_status": 200,
                "json": {"status": "healthy"},
                "error": None,
            }
        if url.endswith("/api/live-pipeline/intake-audit"):
            assert headers and headers.get("Authorization") == "Bearer secret-token"
            return {
                "ok": True,
                "http_status": 200,
                "json": {"status": "staged_for_control_plane", "input_blockers": []},
                "error": None,
            }
        raise AssertionError(f"unexpected URL: {url}")

    report = build_deployment_parity_proof(
        webapp_url="https://paperclip.tryblueprint.io",
        pipeline_intake_url="https://paperclip.tryblueprint.io/api/live-pipeline/job-requests",
        pipeline_intake_token="secret-token",
        webapp_repo=Path("/repos/webapp"),
        pipeline_repo=Path("/repos/pipeline"),
        webapp_deployed_commit="abc123",
        pipeline_deployed_commit="abc123",
        fetcher=same_host_fetcher,
        git_probe=_git_probe,
    )

    assert report["status"] == "verified"
    assert report["pipeline_intake_health_ready"] is True
    assert report["checks"]["pipeline_intake_health"]["token_configured"] is False
    assert report["checks"]["pipeline_intake_audit"]["status"] == "staged_for_control_plane"


def test_deployment_parity_proof_blocks_without_deployed_commit_and_clean_tree() -> None:
    def dirty_git_probe(repo: Path) -> dict[str, Any]:
        payload = _git_probe(repo)
        if repo.name == "webapp":
            payload["worktree_clean"] = False
            payload["dirty_entries_count"] = 2
            payload["dirty_entries"] = [" M client/src/App.tsx", "?? output/pipeline/"]
        return payload

    report = build_deployment_parity_proof(
        webapp_url="https://paperclip.tryblueprint.io",
        pipeline_intake_url="https://pipeline.tryblueprint.io",
        pipeline_intake_token="secret-token",
        webapp_repo=Path("/repos/webapp"),
        pipeline_repo=Path("/repos/pipeline"),
        webapp_deployed_commit="",
        pipeline_deployed_commit="",
        fetcher=_fetcher,
        git_probe=dirty_git_probe,
    )

    assert report["status"] == "blocked"
    assert report["production_deployment_proven"] is False
    assert report["git_parity_proven"] is False
    assert "webapp_worktree_dirty" in report["blockers"]
    assert "webapp_deployed_commit_missing" in report["blockers"]
    assert "pipeline_deployed_commit_missing" in report["blockers"]
    assert report["checks"]["git"]["repos"]["webapp"]["dirty_entries"] == [
        " M client/src/App.tsx",
        "?? output/pipeline/",
    ]


def test_deployment_parity_proof_blocks_unready_health() -> None:
    def blocked_fetcher(
        url: str,
        headers: Mapping[str, str] | None,
        timeout: int,
    ) -> dict[str, Any]:
        del headers, timeout
        if url.endswith("/health/ready"):
            return {
                "ok": False,
                "http_status": 503,
                "json": {"status": "not_ready", "blockers": ["firebase_missing"]},
                "error": "http_503",
            }
        if url.endswith("/health"):
            return {
                "ok": True,
                "http_status": 200,
                "json": {"ok": True, "token_configured": False},
                "error": None,
            }
        return {
            "ok": False,
            "http_status": 401,
            "json": {"detail": "invalid intake token"},
            "error": "http_401",
        }

    report = build_deployment_parity_proof(
        webapp_url="https://paperclip.tryblueprint.io",
        pipeline_intake_url="https://pipeline.tryblueprint.io",
        pipeline_intake_token="secret-token",
        webapp_repo=Path("/repos/webapp"),
        pipeline_repo=Path("/repos/pipeline"),
        webapp_deployed_commit="abc123",
        pipeline_deployed_commit="abc123",
        fetcher=blocked_fetcher,
        git_probe=_git_probe,
    )

    assert report["status"] == "blocked"
    assert report["webapp_health_ready"] is False
    assert report["pipeline_intake_health_ready"] is False
    assert "webapp_health_not_ready" in report["blockers"]
    assert "pipeline_intake_health_not_ready" in report["blockers"]
    assert "pipeline_intake_audit_not_reachable" in report["blockers"]
