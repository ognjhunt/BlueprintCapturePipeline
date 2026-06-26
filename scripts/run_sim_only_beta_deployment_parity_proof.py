#!/usr/bin/env python3
"""Build production deployment/parity proof for the sim-only beta release gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.common import ensure_dir, read_json_any, utc_now_iso, write_json  # noqa: E402

SCHEMA_VERSION = "blueprint.sim_only_beta_deployment_parity_proof.v1"


JsonFetcher = Callable[[str, Mapping[str, str] | None, int], dict[str, Any]]
GitProbe = Callable[[Path], dict[str, Any]]


def _string(value: Any) -> str:
    return str(value or "").strip()


def _normalize_base_url(value: str) -> str:
    url = _string(value).rstrip("/")
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    if parsed.path.startswith("/api/live-pipeline/"):
        return f"{parsed.scheme}://{parsed.netloc}"
    return url


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_route_forwarding_proof(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {}, None
    if not path.is_file():
        return {}, "missing"
    payload = read_json_any(path)
    if not isinstance(payload, Mapping):
        return {}, "not_json_object"
    return dict(payload), None


def _route_proof_url_hints(path: Path | None) -> dict[str, Any]:
    route_proof, load_error = _load_route_forwarding_proof(path)
    webapp_route = _mapping(route_proof.get("webapp_route"))
    forwarding_endpoint = _mapping(route_proof.get("forwarding_endpoint"))
    return {
        "path": str(path) if path else None,
        "load_error": load_error,
        "webapp_url": _string(webapp_route.get("remote_webapp_url")),
        "pipeline_intake_url": _string(forwarding_endpoint.get("endpoint_url")),
        "pipeline_intake_url_source": _string(forwarding_endpoint.get("endpoint_url_source")),
    }


def _fetch_json_url(
    url: str,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    request = Request(url, headers=dict(headers or {}))
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            text = response.read().decode("utf-8", errors="replace")
            try:
                payload: Any = json.loads(text) if text else None
            except json.JSONDecodeError:
                payload = None
            return {
                "ok": 200 <= response.status < 300,
                "http_status": response.status,
                "json": payload if isinstance(payload, Mapping) else None,
                "error": None,
            }
    except HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(text) if text else None
        except json.JSONDecodeError:
            payload = None
        return {
            "ok": False,
            "http_status": error.code,
            "json": payload if isinstance(payload, Mapping) else None,
            "error": f"http_{error.code}",
        }
    except (URLError, TimeoutError, OSError) as error:
        return {
            "ok": False,
            "http_status": None,
            "json": None,
            "error": error.__class__.__name__,
        }


def _git_output(repo: Path, *args: str) -> tuple[str, str | None]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return "", error.__class__.__name__
    return completed.stdout.strip(), None


def probe_git_repo(repo: Path) -> dict[str, Any]:
    head, head_error = _git_output(repo, "rev-parse", "HEAD")
    origin_main, origin_error = _git_output(repo, "rev-parse", "origin/main")
    status, status_error = _git_output(repo, "status", "--porcelain")
    dirty_entries = [line for line in status.splitlines() if line.strip()]
    max_dirty_entries = 200
    return {
        "path": str(repo),
        "head": head or None,
        "origin_main": origin_main or None,
        "head_matches_origin_main": bool(head and origin_main and head == origin_main),
        "worktree_clean": status == "" and status_error is None,
        "dirty_entries_count": len(dirty_entries),
        "dirty_entries": dirty_entries[:max_dirty_entries],
        "dirty_entries_truncated": len(dirty_entries) > max_dirty_entries,
        "errors": [error for error in (head_error, origin_error, status_error) if error],
    }


def _repo_blockers(label: str, status: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if status.get("errors"):
        blockers.append(f"{label}_git_probe_failed")
    if status.get("head_matches_origin_main") is not True:
        blockers.append(f"{label}_head_not_origin_main")
    if status.get("worktree_clean") is not True:
        blockers.append(f"{label}_worktree_dirty")
    return blockers


def _commit_blockers(
    *,
    label: str,
    repo_status: Mapping[str, Any],
    deployed_commit: str,
    require_deployed_commit: bool,
) -> list[str]:
    head = _string(repo_status.get("head"))
    deployed = _string(deployed_commit)
    if not require_deployed_commit:
        return []
    if not deployed:
        return [f"{label}_deployed_commit_missing"]
    if head and deployed != head:
        return [f"{label}_deployed_commit_mismatch"]
    return []


def _json_status(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    return _string(payload.get("status") or payload.get("ok"))


def build_deployment_parity_proof(
    *,
    webapp_url: str,
    pipeline_intake_url: str,
    pipeline_intake_token: str,
    webapp_repo: Path,
    pipeline_repo: Path,
    capture_repo: Path | None = None,
    webapp_deployed_commit: str = "",
    pipeline_deployed_commit: str = "",
    route_forwarding_proof_path: Path | None = None,
    require_deployed_commit: bool = True,
    timeout_seconds: int = 10,
    now_iso: str | None = None,
    fetcher: JsonFetcher = _fetch_json_url,
    git_probe: GitProbe = probe_git_repo,
) -> dict[str, Any]:
    blockers: list[str] = []
    route_proof_hints = _route_proof_url_hints(route_forwarding_proof_path)
    resolved_webapp_url = _string(webapp_url) or _string(route_proof_hints.get("webapp_url"))
    resolved_pipeline_intake_url = _string(pipeline_intake_url) or _string(
        route_proof_hints.get("pipeline_intake_url")
    )
    webapp_base_url = _normalize_base_url(resolved_webapp_url)
    pipeline_base_url = _normalize_base_url(resolved_pipeline_intake_url)

    if not webapp_base_url:
        blockers.append("webapp_url_missing_or_invalid")
    if not pipeline_base_url:
        blockers.append("pipeline_intake_url_missing_or_invalid")

    webapp_health: dict[str, Any] = {
        "url": urljoin(f"{webapp_base_url}/", "health/ready") if webapp_base_url else None,
        "ok": False,
        "http_status": None,
        "status": None,
        "error": "not_attempted",
    }
    if webapp_base_url:
        result = fetcher(str(webapp_health["url"]), None, timeout_seconds)
        payload = result.get("json") if isinstance(result.get("json"), Mapping) else {}
        webapp_health.update(
            {
                "ok": result.get("ok") is True,
                "http_status": result.get("http_status"),
                "status": _json_status(payload),
                "error": result.get("error"),
                "blockers": list(payload.get("blockers") or []) if isinstance(payload, Mapping) else [],
            }
        )

    pipeline_health: dict[str, Any] = {
        "url": urljoin(f"{pipeline_base_url}/", "health") if pipeline_base_url else None,
        "ok": False,
        "http_status": None,
        "status": None,
        "token_configured": False,
        "error": "not_attempted",
    }
    if pipeline_base_url:
        result = fetcher(str(pipeline_health["url"]), None, timeout_seconds)
        payload = result.get("json") if isinstance(result.get("json"), Mapping) else {}
        pipeline_health.update(
            {
                "ok": result.get("ok") is True,
                "http_status": result.get("http_status"),
                "status": _json_status(payload),
                "token_configured": payload.get("token_configured") is True
                if isinstance(payload, Mapping)
                else False,
                "error": result.get("error"),
            }
        )

    intake_audit: dict[str, Any] = {
        "url": urljoin(f"{pipeline_base_url}/", "api/live-pipeline/intake-audit")
        if pipeline_base_url
        else None,
        "attempted": False,
        "ok": False,
        "http_status": None,
        "status": None,
        "error": "not_attempted",
    }
    if pipeline_base_url and pipeline_intake_token:
        result = fetcher(
            str(intake_audit["url"]),
            {"Authorization": f"Bearer {pipeline_intake_token}"},
            timeout_seconds,
        )
        payload = result.get("json") if isinstance(result.get("json"), Mapping) else {}
        intake_audit.update(
            {
                "attempted": True,
                "ok": result.get("ok") is True,
                "http_status": result.get("http_status"),
                "status": _json_status(payload),
                "input_blockers_count": len(list(payload.get("input_blockers") or []))
                if isinstance(payload, Mapping)
                else None,
                "error": result.get("error"),
            }
        )
    elif pipeline_base_url:
        blockers.append("pipeline_intake_token_missing")

    webapp_health_ready = (
        webapp_health.get("ok") is True
        and webapp_health.get("http_status") == 200
        and webapp_health.get("status") == "ready"
        and not webapp_health.get("blockers")
    )
    pipeline_health_ready = (
        pipeline_health.get("ok") is True
        and pipeline_health.get("http_status") == 200
        and pipeline_health.get("token_configured") is True
    )
    intake_audit_health_ready = (
        intake_audit.get("attempted") is True
        and intake_audit.get("ok") is True
        and intake_audit.get("http_status") == 200
        and intake_audit.get("status") == "staged_for_control_plane"
        and intake_audit.get("input_blockers_count") in (0, None)
    )
    pipeline_intake_health_ready = pipeline_health_ready or intake_audit_health_ready

    if not webapp_health_ready:
        blockers.append("webapp_health_not_ready")
    if not pipeline_intake_health_ready:
        blockers.append("pipeline_intake_health_not_ready")
    if intake_audit.get("attempted") and intake_audit.get("ok") is not True:
        blockers.append("pipeline_intake_audit_not_reachable")

    repos: dict[str, dict[str, Any]] = {
        "webapp": git_probe(webapp_repo),
        "pipeline": git_probe(pipeline_repo),
    }
    if capture_repo is not None:
        repos["capture"] = git_probe(capture_repo)

    git_blockers: list[str] = []
    for label, status in repos.items():
        git_blockers.extend(_repo_blockers(label, status))
    git_blockers.extend(
        _commit_blockers(
            label="webapp",
            repo_status=repos["webapp"],
            deployed_commit=webapp_deployed_commit,
            require_deployed_commit=require_deployed_commit,
        )
    )
    git_blockers.extend(
        _commit_blockers(
            label="pipeline",
            repo_status=repos["pipeline"],
            deployed_commit=pipeline_deployed_commit,
            require_deployed_commit=require_deployed_commit,
        )
    )
    blockers.extend(git_blockers)
    git_parity_proven = not git_blockers

    production_deployment_proven = (
        webapp_health_ready and pipeline_intake_health_ready and git_parity_proven
    )
    status = "verified" if production_deployment_proven and not blockers else "blocked"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso or utc_now_iso(),
        "status": status,
        "production_deployment_proven": production_deployment_proven,
        "webapp_health_ready": webapp_health_ready,
        "pipeline_intake_health_ready": pipeline_intake_health_ready,
        "git_parity_proven": git_parity_proven,
        "webapp_url": webapp_base_url or None,
        "pipeline_intake_url": pipeline_base_url or None,
        "blockers": blockers,
        "checks": {
            "route_forwarding_proof": route_proof_hints,
            "webapp_health": webapp_health,
            "pipeline_intake_health": pipeline_health,
            "pipeline_intake_audit": intake_audit,
            "git": {
                "repos": repos,
                "require_deployed_commit": require_deployed_commit,
                "webapp_deployed_commit_configured": bool(_string(webapp_deployed_commit)),
                "pipeline_deployed_commit_configured": bool(_string(pipeline_deployed_commit)),
            },
        },
        "proof_boundary": {
            "production_deployment_proven": production_deployment_proven,
            "webapp_health_ready": webapp_health_ready,
            "pipeline_intake_health_ready": pipeline_intake_health_ready,
            "git_parity_proven": git_parity_proven,
            "simulator_execution_proven": False,
            "generated_world_rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }


def _default_output_path(capture_root: Path) -> Path:
    return (
        capture_root
        / "pipeline"
        / "live_pipeline_control_plane"
        / "sim_only_beta_production_deployment_proof.json"
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write sim-only beta production deployment/parity proof JSON."
    )
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--webapp-url", default=os.getenv("BLUEPRINT_WEBAPP_PRODUCTION_URL") or os.getenv("ALPHA_BASE_URL") or os.getenv("BASE_URL") or "")
    parser.add_argument("--pipeline-intake-url", default=os.getenv("ROBOT_EVAL_JOB_REQUEST_FORWARD_URL") or os.getenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_URL") or "")
    parser.add_argument("--route-forwarding-proof", type=Path)
    parser.add_argument("--pipeline-intake-token-env", default="ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN")
    parser.add_argument("--webapp-repo", type=Path, default=ROOT.parent / "Blueprint-WebApp")
    parser.add_argument("--pipeline-repo", type=Path, default=ROOT)
    parser.add_argument("--capture-repo", type=Path, default=ROOT.parent / "BlueprintCapture")
    parser.add_argument("--webapp-deployed-commit", default=os.getenv("BLUEPRINT_WEBAPP_DEPLOYED_COMMIT", ""))
    parser.add_argument("--pipeline-deployed-commit", default=os.getenv("BLUEPRINT_PIPELINE_DEPLOYED_COMMIT", ""))
    parser.add_argument(
        "--allow-local-git-parity-only",
        action="store_true",
        help="Do not require explicit deployed commit values. Health checks still must pass.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    capture_root = args.capture_root.resolve()
    output_path = (args.output or _default_output_path(capture_root)).resolve()
    token = _string(os.getenv(args.pipeline_intake_token_env))
    if not token and args.pipeline_intake_token_env != "BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN":
        token = _string(os.getenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN"))

    report = build_deployment_parity_proof(
        webapp_url=args.webapp_url,
        pipeline_intake_url=args.pipeline_intake_url,
        pipeline_intake_token=token,
        webapp_repo=args.webapp_repo.resolve(),
        pipeline_repo=args.pipeline_repo.resolve(),
        capture_repo=args.capture_repo.resolve() if args.capture_repo else None,
        webapp_deployed_commit=args.webapp_deployed_commit,
        pipeline_deployed_commit=args.pipeline_deployed_commit,
        route_forwarding_proof_path=args.route_forwarding_proof.resolve()
        if args.route_forwarding_proof
        else None,
        require_deployed_commit=not args.allow_local_git_parity_only,
        timeout_seconds=max(1, args.timeout_seconds),
    )
    ensure_dir(output_path.parent)
    write_json(output_path, report)
    print(f"[sim-only-beta-deployment-proof] report={output_path}")
    print(f"[sim-only-beta-deployment-proof] status={report['status']}")
    print(f"[sim-only-beta-deployment-proof] blockers={len(report['blockers'])}")
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
