#!/usr/bin/env python3
"""Build a checksum-backed launch readiness evidence packet.

The packet is intentionally conservative: it links current local artifacts and
summarizes remaining live/manual blockers, but it does not upgrade local or
sim-only evidence into production readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from blueprint_pipeline.alpha_readiness import validate_operator_launch_evidence


SCHEMA_VERSION = "blueprint.launch_readiness_packet.v1"
CI_EVIDENCE_SCHEMA_VERSION = "blueprint.github_actions_evidence.v1"
PIPELINE_SOURCE_REQUIRED_ARTIFACT_IDS = {
    "paid_marketplace_launch_gate_json",
    "external_alpha_launch_gate_json",
    "sim_only_beta_local_gate_report",
    "live_pipeline_setup_audit",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo: Path, *args: str) -> str | None:
    if not (repo / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip()


def _repo_info(repo: Path) -> dict[str, Any]:
    status = _run_git(repo, "status", "--short") or ""
    dirty_entries = [
        line[3:].split(" -> ")[-1]
        for line in status.splitlines()
        if line.strip() and len(line) > 3
    ]
    head = _run_git(repo, "rev-parse", "HEAD")
    origin_main = _run_git(repo, "rev-parse", "--verify", "origin/main")
    git_metadata_present = (repo / ".git").exists()
    return {
        "path": str(repo),
        "exists": repo.exists(),
        "git_metadata_present": git_metadata_present,
        "branch": _run_git(repo, "branch", "--show-current"),
        "head": head,
        "origin_main_head": origin_main,
        "head_matches_origin_main": bool(head and origin_main and head == origin_main),
        "dirty_entry_count": len(dirty_entries),
        "dirty_entries": dirty_entries,
    }


def _artifact(id_: str, path: Path, *, source_repo: str) -> dict[str, Any]:
    payload = _read_json(path) if path.suffix == ".json" else {}
    status = payload.get("overall_status") or payload.get("status")
    pipeline_source = (
        payload.get("pipeline_source")
        if isinstance(payload.get("pipeline_source"), Mapping)
        else {}
    )
    return {
        "id": id_,
        "source_repo": source_repo,
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha256(path),
        "json_status": status if isinstance(status, str) else None,
        "schema_version": payload.get("schema_version") if isinstance(payload.get("schema_version"), str) else None,
        "pipeline_source_head": pipeline_source.get("head")
        if isinstance(pipeline_source.get("head"), str)
        else None,
        "pipeline_source_repo_name": pipeline_source.get("repo_name")
        if isinstance(pipeline_source.get("repo_name"), str)
        else None,
    }


def _ci_artifact(id_: str, path: Path, *, repo_name: str, workflow: str) -> dict[str, Any]:
    artifact = _artifact(id_, path, source_repo="GitHubActions")
    payload = _read_json(path)
    artifact["repo_name"] = repo_name
    artifact["workflow"] = workflow
    artifact["conclusion"] = payload.get("conclusion") if isinstance(payload.get("conclusion"), str) else None
    artifact["head_sha"] = payload.get("head_sha") if isinstance(payload.get("head_sha"), str) else None
    artifact["run_url"] = payload.get("url") if isinstance(payload.get("url"), str) else None
    test_counts = payload.get("test_counts")
    if isinstance(test_counts, Mapping):
        artifact["test_counts"] = {
            key: test_counts.get(key)
            for key in ("tests", "failures", "errors", "skipped")
            if key in test_counts
        }
    return artifact


def _latest_sim_only_report(pipeline_repo: Path) -> Path:
    candidates = sorted(
        pipeline_repo.glob("output/sim_only_beta_local_gate_fixture/**/sim_only_beta_local_gate_report.json"),
        key=lambda candidate: candidate.stat().st_mtime if candidate.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return pipeline_repo / "output" / "sim_only_beta_local_gate_report.json"


def _operator_evidence_path(pipeline_repo: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    for candidate in (
        pipeline_repo / "output" / "operator_launch_evidence.json",
        pipeline_repo / "output" / "pipeline" / "operator_launch_evidence.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _manual_ids_from_paid_gate(payload: Mapping[str, Any]) -> list[str]:
    closeout = payload.get("closeout_summary")
    if isinstance(closeout, Mapping):
        ids = closeout.get("remaining_manual_evidence_ids")
        if isinstance(ids, list):
            return [str(item) for item in ids if str(item).strip()]
    checks = payload.get("manual_checks")
    if isinstance(checks, list):
        return [
            str(check.get("id"))
            for check in checks
            if isinstance(check, Mapping) and str(check.get("id") or "").strip()
        ]
    return []


def _external_manual_items(payload: Mapping[str, Any]) -> list[str]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return []
    items: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        status = str(check.get("status") or "")
        check_id = str(check.get("id") or "")
        if check_id and status and status != "passed":
            items.append(f"{check_id}:{status}")
    return items


def _forwarding_packet_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers = [
        str(item)
        for item in payload.get("blockers", [])
        if str(item).strip()
    ]
    if blockers:
        return blockers

    status = str(payload.get("status") or "").strip()
    probe = payload.get("probe") if isinstance(payload.get("probe"), Mapping) else {}
    if payload.get("forwarding_required") is not True:
        blockers.append("webapp_forwarding_not_required")
    if payload.get("endpoint_configured") is not True:
        blockers.append("webapp_forwarding_endpoint_not_configured")
    if status != "ready_for_required_forwarding_with_probe":
        blockers.append(f"webapp_forwarding_status_not_required_probe_ready:{status or 'missing'}")
    if probe.get("attempted") is not True:
        blockers.append("webapp_forwarding_probe_not_attempted")
    probe_status = str(probe.get("status") or "").strip()
    if probe_status != "reachable":
        blockers.append(f"webapp_forwarding_probe_not_reachable:{probe_status or 'missing'}")
    return blockers


def _artifact_blockers(artifacts: list[Mapping[str, Any]]) -> list[str]:
    return [f"missing_artifact:{artifact['id']}" for artifact in artifacts if not artifact.get("exists")]


def _artifact_trust_blockers(
    artifacts: list[Mapping[str, Any]],
    repository_blockers: list[str],
) -> list[str]:
    blocked_repo_names = {
        blocker.split(":", 2)[1]
        for blocker in repository_blockers
        if blocker.startswith(("repo_dirty:", "repo_not_at_origin_main:", "repo_git_metadata_missing:"))
        and len(blocker.split(":", 2)) >= 2
    }
    blockers: list[str] = []
    for artifact in artifacts:
        source_repo = str(artifact.get("source_repo") or "")
        artifact_id = str(artifact.get("id") or "")
        if artifact.get("exists") and source_repo in blocked_repo_names and artifact_id:
            blockers.append(f"untrusted_artifact_repo:{artifact_id}:{source_repo}")
    return blockers


def _artifact_source_blockers(
    artifacts: list[Mapping[str, Any]],
    repos: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    pipeline_head = str(repos.get("BlueprintCapturePipeline", {}).get("head") or "")
    blockers: list[str] = []
    for artifact in artifacts:
        artifact_id = str(artifact.get("id") or "")
        if artifact_id not in PIPELINE_SOURCE_REQUIRED_ARTIFACT_IDS:
            continue
        if not artifact.get("exists"):
            continue
        source_head = str(artifact.get("pipeline_source_head") or "")
        source_repo = str(artifact.get("pipeline_source_repo_name") or "")
        if not source_head:
            blockers.append(f"artifact_source_head_missing:{artifact_id}")
            continue
        if source_repo and source_repo != "BlueprintCapturePipeline":
            blockers.append(f"artifact_source_repo_mismatch:{artifact_id}:{source_repo}")
        if not pipeline_head:
            blockers.append(f"artifact_source_repo_head_unavailable:{artifact_id}")
        elif source_head != pipeline_head:
            blockers.append(f"artifact_source_head_mismatch:{artifact_id}:{source_head}")
    return blockers


def _is_allowed_dirty_entry(entry: str, prefixes: tuple[str, ...]) -> bool:
    normalized = entry.lstrip("/")
    return any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in prefixes)


def _repository_blockers(
    repos: Mapping[str, Mapping[str, Any]],
    *,
    allowed_dirty_prefixes: Mapping[str, tuple[str, ...]],
) -> list[str]:
    blockers: list[str] = []
    for name, info in repos.items():
        if not info.get("exists"):
            blockers.append(f"repo_missing:{name}")
            continue
        if not info.get("git_metadata_present"):
            blockers.append(f"repo_git_metadata_missing:{name}")
            continue
        dirty_entries = [
            str(entry)
            for entry in info.get("dirty_entries", [])
            if str(entry).strip()
        ]
        allowed_prefixes = allowed_dirty_prefixes.get(name, ())
        blocking_dirty_entries = [
            entry
            for entry in dirty_entries
            if not _is_allowed_dirty_entry(entry, allowed_prefixes)
        ]
        if blocking_dirty_entries:
            blockers.append(f"repo_dirty:{name}:{len(blocking_dirty_entries)}")
        if not info.get("head_matches_origin_main"):
            blockers.append(f"repo_not_at_origin_main:{name}")
    return blockers


def _ci_evidence_blockers(
    ci_artifacts: list[Mapping[str, Any]],
    repos: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for artifact in ci_artifacts:
        artifact_id = str(artifact.get("id") or "")
        repo_name = str(artifact.get("repo_name") or "")
        if not artifact.get("exists"):
            blockers.append(f"missing_ci_evidence:{artifact_id}")
            continue
        if artifact.get("schema_version") != CI_EVIDENCE_SCHEMA_VERSION:
            blockers.append(f"ci_evidence_schema_mismatch:{artifact_id}")
        conclusion = str(artifact.get("conclusion") or "")
        if conclusion != "success":
            blockers.append(f"ci_evidence_not_success:{artifact_id}:{conclusion or 'unknown'}")
        repo = repos.get(repo_name)
        repo_head = str((repo or {}).get("head") or "")
        head_sha = str(artifact.get("head_sha") or "")
        if not repo_head:
            blockers.append(f"ci_evidence_repo_head_unavailable:{artifact_id}:{repo_name}")
        elif head_sha != repo_head:
            blockers.append(f"ci_evidence_head_mismatch:{artifact_id}:{head_sha or 'missing'}")
        test_counts = artifact.get("test_counts")
        if isinstance(test_counts, Mapping):
            failures = int(test_counts.get("failures") or 0)
            errors = int(test_counts.get("errors") or 0)
            tests = int(test_counts.get("tests") or 0)
            if failures or errors:
                blockers.append(f"ci_evidence_test_failures:{artifact_id}")
            if tests <= 0:
                blockers.append(f"ci_evidence_empty_test_count:{artifact_id}")
    return blockers


def build_launch_readiness_packet(
    *,
    pipeline_repo: Path,
    webapp_repo: Path,
    contracts_repo: Path,
    capture_repo: Path,
    operator_evidence_path: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    paid_gate = pipeline_repo / "output" / "paid_marketplace_launch_gate.json"
    paid_gate_md = pipeline_repo / "output" / "paid_marketplace_launch_gate.md"
    external_gate = pipeline_repo / "output" / "external_alpha_launch_gate.json"
    external_gate_md = pipeline_repo / "output" / "external_alpha_launch_gate.md"
    live_setup = pipeline_repo / "output" / "launch_audit_live_pipeline_setup_20260707.json"
    sim_only_report = _latest_sim_only_report(pipeline_repo)
    forwarding_preflight = (
        webapp_repo / "output" / "pipeline" / "robot_eval_job_requests" / "forwarding_preflight.json"
    )
    pipeline_ci_evidence = pipeline_repo / "output" / "pipeline_main_ci_evidence.json"
    pipeline_full_lane_evidence = pipeline_repo / "output" / "pipeline_full_test_lane_ci_evidence.json"
    pipeline_sim_only_evidence = pipeline_repo / "output" / "pipeline_sim_only_local_gate_ci_evidence.json"
    webapp_ci_evidence = pipeline_repo / "output" / "webapp_main_ci_evidence.json"
    resolved_operator_evidence_path = _operator_evidence_path(pipeline_repo, operator_evidence_path)

    artifacts = [
        _artifact("paid_marketplace_launch_gate_json", paid_gate, source_repo="BlueprintCapturePipeline"),
        _artifact("paid_marketplace_launch_gate_markdown", paid_gate_md, source_repo="BlueprintCapturePipeline"),
        _artifact("external_alpha_launch_gate_json", external_gate, source_repo="BlueprintCapturePipeline"),
        _artifact("external_alpha_launch_gate_markdown", external_gate_md, source_repo="BlueprintCapturePipeline"),
        _artifact("sim_only_beta_local_gate_report", sim_only_report, source_repo="BlueprintCapturePipeline"),
        _artifact("live_pipeline_setup_audit", live_setup, source_repo="BlueprintCapturePipeline"),
        _artifact("webapp_forwarding_preflight", forwarding_preflight, source_repo="Blueprint-WebApp"),
    ]
    if resolved_operator_evidence_path is not None:
        artifacts.append(
            _artifact(
                "operator_launch_evidence",
                resolved_operator_evidence_path,
                source_repo="OperatorEvidence",
            )
        )
    ci_artifacts = [
        _ci_artifact(
            "pipeline_main_ci_evidence",
            pipeline_ci_evidence,
            repo_name="BlueprintCapturePipeline",
            workflow="CI",
        ),
        _ci_artifact(
            "pipeline_full_test_lane_ci_evidence",
            pipeline_full_lane_evidence,
            repo_name="BlueprintCapturePipeline",
            workflow="Full Test Lane",
        ),
        _ci_artifact(
            "pipeline_sim_only_local_gate_ci_evidence",
            pipeline_sim_only_evidence,
            repo_name="BlueprintCapturePipeline",
            workflow="Sim-Only Local Gate",
        ),
        _ci_artifact(
            "webapp_main_ci_evidence",
            webapp_ci_evidence,
            repo_name="Blueprint-WebApp",
            workflow="CI",
        ),
    ]
    all_artifacts = [*artifacts, *ci_artifacts]

    paid_payload = _read_json(paid_gate)
    external_payload = _read_json(external_gate)
    live_payload = _read_json(live_setup)
    forwarding_payload = _read_json(forwarding_preflight)
    sim_payload = _read_json(sim_only_report)
    operator_evidence_payload = (
        _read_json(resolved_operator_evidence_path)
        if resolved_operator_evidence_path is not None
        else {}
    )

    paid_gate_manual_evidence_ids = _manual_ids_from_paid_gate(paid_payload)
    operator_evidence_status = validate_operator_launch_evidence(
        operator_evidence_payload,
        paid_gate_manual_evidence_ids,
    )
    manual_evidence_ids = [
        str(item)
        for item in operator_evidence_status.get("remaining_ids", paid_gate_manual_evidence_ids)
        if str(item).strip()
    ]
    live_setup_blockers = [
        str(item)
        for item in live_payload.get("blockers", [])
        if str(item).strip()
    ]
    forwarding_blockers = _forwarding_packet_blockers(forwarding_payload)
    external_manual_items = _external_manual_items(external_payload)
    missing_artifacts = _artifact_blockers(all_artifacts)
    repos = {
        "BlueprintCapturePipeline": _repo_info(pipeline_repo),
        "Blueprint-WebApp": _repo_info(webapp_repo),
        "BlueprintContracts": _repo_info(contracts_repo),
        "BlueprintCapture": _repo_info(capture_repo),
    }
    repository_blockers = _repository_blockers(
        repos,
        allowed_dirty_prefixes={
            "BlueprintCapturePipeline": ("output/",),
            "Blueprint-WebApp": ("output/",),
        },
    )
    artifact_trust_blockers = _artifact_trust_blockers(all_artifacts, repository_blockers)
    artifact_source_blockers = _artifact_source_blockers(all_artifacts, repos)
    ci_evidence_blockers = _ci_evidence_blockers(ci_artifacts, repos)

    status = "ready"
    if (
        missing_artifacts
        or repository_blockers
        or artifact_trust_blockers
        or artifact_source_blockers
        or ci_evidence_blockers
    ):
        status = "incomplete_packet"
    elif manual_evidence_ids or live_setup_blockers or forwarding_blockers or external_manual_items:
        status = "local_ready_live_external_blocked"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "status": status,
        "repos": repos,
        "artifacts": all_artifacts,
        "artifact_blockers": missing_artifacts,
        "artifact_trust_blockers": artifact_trust_blockers,
        "artifact_source_blockers": artifact_source_blockers,
        "repository_blockers": repository_blockers,
        "ci_evidence_blockers": ci_evidence_blockers,
        "readiness_summary": {
            "paid_marketplace_launch_gate": paid_payload.get("overall_status"),
            "external_alpha_launch_gate": external_payload.get("overall_status"),
            "sim_only_beta_local_gate": sim_payload.get("status"),
            "live_pipeline_setup": live_payload.get("status"),
            "webapp_forwarding_preflight": forwarding_payload.get("status"),
            "pipeline_main_ci": ci_artifacts[0].get("conclusion"),
            "pipeline_full_test_lane_ci": ci_artifacts[1].get("conclusion"),
            "pipeline_sim_only_local_gate_ci": ci_artifacts[2].get("conclusion"),
            "webapp_main_ci": ci_artifacts[3].get("conclusion"),
            "operator_evidence": operator_evidence_status.get("status"),
        },
        "remaining_blockers": {
            "manual_live_evidence_ids": manual_evidence_ids,
            "external_alpha_manual_items": external_manual_items,
            "live_pipeline_setup_blockers": live_setup_blockers,
            "webapp_forwarding_blockers": forwarding_blockers,
            "artifact_source_blockers": artifact_source_blockers,
            "ci_evidence_blockers": ci_evidence_blockers,
        },
        "operator_evidence_status": {
            **operator_evidence_status,
            "evidence_file": str(resolved_operator_evidence_path) if resolved_operator_evidence_path else None,
            "paid_gate_manual_evidence_ids": paid_gate_manual_evidence_ids,
        },
        "commands_to_refresh": [
            "python scripts/run_paid_marketplace_launch_gate.py",
            "python scripts/run_external_alpha_launch_gate.py",
            "python scripts/run_sim_only_beta_local_gate.py",
            "python scripts/collect_github_actions_evidence.py --repo ognjhunt/BlueprintCapturePipeline --run-id <pipeline-ci-run-id> --evidence-id pipeline_main_ci_evidence --output output/pipeline_main_ci_evidence.json",
            "python scripts/collect_github_actions_evidence.py --repo ognjhunt/BlueprintCapturePipeline --run-id <full-lane-run-id> --evidence-id pipeline_full_test_lane_ci_evidence --junit <downloaded-junit.xml> --output output/pipeline_full_test_lane_ci_evidence.json",
            "python scripts/collect_github_actions_evidence.py --repo ognjhunt/BlueprintCapturePipeline --run-id <sim-only-local-gate-run-id> --evidence-id pipeline_sim_only_local_gate_ci_evidence --output output/pipeline_sim_only_local_gate_ci_evidence.json",
            "python scripts/collect_github_actions_evidence.py --repo ognjhunt/Blueprint-WebApp --run-id <webapp-ci-run-id> --evidence-id webapp_main_ci_evidence --output output/webapp_main_ci_evidence.json",
            "python -m blueprint_pipeline.live_pipeline_setup --no-load-env-files --output-path output/launch_audit_live_pipeline_setup_20260707.json",
            "cd ../Blueprint-WebApp && npm run pipeline:forwarding:preflight -- --require-forwarding",
        ],
        "claim_boundary": {
            "packet_links_current_local_artifacts": True,
            "github_actions_evidence_is_source_control_ci_only": True,
            "sim_only_local_gate_is_not_production_forwarding_proof": True,
            "automated_contracts_do_not_prove_live_payments_or_payouts": True,
            "automated_contracts_do_not_prove_real_pubsub_delivery": True,
            "automated_contracts_do_not_prove_live_provider_execution": True,
            "operator_evidence_required_for_paid_beta": True,
        },
    }


def _markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Blueprint Launch Readiness Packet",
        "",
        f"Generated: `{packet.get('generated_at')}`",
        f"Status: `{packet.get('status')}`",
        "",
        "## Repository Evidence",
    ]
    repos = packet.get("repos")
    if isinstance(repos, Mapping):
        for name, info in repos.items():
            if not isinstance(info, Mapping):
                continue
            local = info.get("head") or "unavailable"
            origin_main = info.get("origin_main_head") or "unavailable"
            branch = info.get("branch") or "detached_or_unavailable"
            dirty_count = info.get("dirty_entry_count")
            lines.append(
                f"- `{name}`: branch `{branch}`, local `{local}`, origin/main `{origin_main}`, dirty entries `{dirty_count}`"
            )

    lines.extend(
        [
            "",
            "## Artifact Checksums",
        ]
    )
    for artifact in packet.get("artifacts", []):
        if not isinstance(artifact, Mapping):
            continue
        status = artifact.get("json_status") or ("present" if artifact.get("exists") else "missing")
        sha = artifact.get("sha256") or "missing"
        lines.append(f"- `{artifact.get('id')}`: `{status}` `{sha}`")

    lines.extend(["", "## Remaining Blockers"])
    blockers = packet.get("remaining_blockers")
    if isinstance(blockers, Mapping):
        for key, values in blockers.items():
            if isinstance(values, list) and values:
                lines.append(f"- `{key}`: {len(values)}")
                for value in values:
                    lines.append(f"  - `{value}`")
    artifact_blockers = packet.get("artifact_blockers")
    if isinstance(artifact_blockers, list) and artifact_blockers:
        lines.append(f"- `artifact_blockers`: {len(artifact_blockers)}")
        for value in artifact_blockers:
            lines.append(f"  - `{value}`")
    artifact_trust_blockers = packet.get("artifact_trust_blockers")
    if isinstance(artifact_trust_blockers, list) and artifact_trust_blockers:
        lines.append(f"- `artifact_trust_blockers`: {len(artifact_trust_blockers)}")
        for value in artifact_trust_blockers:
            lines.append(f"  - `{value}`")
    artifact_source_blockers = packet.get("artifact_source_blockers")
    if isinstance(artifact_source_blockers, list) and artifact_source_blockers:
        lines.append(f"- `artifact_source_blockers`: {len(artifact_source_blockers)}")
        for value in artifact_source_blockers:
            lines.append(f"  - `{value}`")
    repository_blockers = packet.get("repository_blockers")
    if isinstance(repository_blockers, list) and repository_blockers:
        lines.append(f"- `repository_blockers`: {len(repository_blockers)}")
        for value in repository_blockers:
            lines.append(f"  - `{value}`")
    if len(lines) > 0 and lines[-1] == "## Remaining Blockers":
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "This packet links local artifacts and checksums. It is not live Pub/Sub, production forwarding, live payment, live payout, real-device upload, or live provider proof.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-repo", type=Path, default=_repo_root())
    parser.add_argument("--webapp-repo", type=Path, default=_repo_root().parent / "Blueprint-WebApp")
    parser.add_argument("--contracts-repo", type=Path, default=_repo_root().parent / "BlueprintContracts")
    parser.add_argument("--capture-repo", type=Path, default=_repo_root().parent / "BlueprintCapture")
    parser.add_argument(
        "--operator-evidence",
        type=Path,
        help=(
            "Optional operator_launch_evidence.v1 file. When omitted, "
            "output/operator_launch_evidence.json is used if present."
        ),
    )
    parser.add_argument("--output", type=Path, default=_repo_root() / "output" / "launch_readiness_packet.json")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=_repo_root() / "output" / "launch_readiness_packet.md",
    )
    args = parser.parse_args()

    packet = build_launch_readiness_packet(
        pipeline_repo=args.pipeline_repo.resolve(),
        webapp_repo=args.webapp_repo.resolve(),
        contracts_repo=args.contracts_repo.resolve(),
        capture_repo=args.capture_repo.resolve(),
        operator_evidence_path=args.operator_evidence.resolve() if args.operator_evidence else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(packet), encoding="utf-8")
    print(f"[launch-readiness-packet] status={packet['status']}")
    print(f"[launch-readiness-packet] json={args.output}")
    print(f"[launch-readiness-packet] markdown={args.markdown_output}")


if __name__ == "__main__":
    main()
