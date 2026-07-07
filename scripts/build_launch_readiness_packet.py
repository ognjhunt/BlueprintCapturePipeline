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


SCHEMA_VERSION = "blueprint.launch_readiness_packet.v1"


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
    return completed.stdout.strip()


def _repo_info(repo: Path) -> dict[str, Any]:
    status = _run_git(repo, "status", "--short") or ""
    return {
        "path": str(repo),
        "exists": repo.exists(),
        "branch": _run_git(repo, "branch", "--show-current"),
        "head": _run_git(repo, "rev-parse", "HEAD"),
        "dirty_entry_count": len([line for line in status.splitlines() if line.strip()]),
    }


def _artifact(id_: str, path: Path) -> dict[str, Any]:
    payload = _read_json(path) if path.suffix == ".json" else {}
    status = payload.get("overall_status") or payload.get("status")
    return {
        "id": id_,
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha256(path),
        "json_status": status if isinstance(status, str) else None,
        "schema_version": payload.get("schema_version") if isinstance(payload.get("schema_version"), str) else None,
    }


def _latest_sim_only_report(pipeline_repo: Path) -> Path:
    candidates = sorted(
        pipeline_repo.glob("output/sim_only_beta_local_gate_fixture/**/sim_only_beta_local_gate_report.json"),
        key=lambda candidate: candidate.stat().st_mtime if candidate.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return pipeline_repo / "output" / "sim_only_beta_local_gate_report.json"


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


def _artifact_blockers(artifacts: list[Mapping[str, Any]]) -> list[str]:
    return [f"missing_artifact:{artifact['id']}" for artifact in artifacts if not artifact.get("exists")]


def build_launch_readiness_packet(
    *,
    pipeline_repo: Path,
    webapp_repo: Path,
    contracts_repo: Path,
    capture_repo: Path,
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

    artifacts = [
        _artifact("paid_marketplace_launch_gate_json", paid_gate),
        _artifact("paid_marketplace_launch_gate_markdown", paid_gate_md),
        _artifact("external_alpha_launch_gate_json", external_gate),
        _artifact("external_alpha_launch_gate_markdown", external_gate_md),
        _artifact("sim_only_beta_local_gate_report", sim_only_report),
        _artifact("live_pipeline_setup_audit", live_setup),
        _artifact("webapp_forwarding_preflight", forwarding_preflight),
    ]

    paid_payload = _read_json(paid_gate)
    external_payload = _read_json(external_gate)
    live_payload = _read_json(live_setup)
    forwarding_payload = _read_json(forwarding_preflight)
    sim_payload = _read_json(sim_only_report)

    manual_evidence_ids = _manual_ids_from_paid_gate(paid_payload)
    live_setup_blockers = [
        str(item)
        for item in live_payload.get("blockers", [])
        if str(item).strip()
    ]
    forwarding_blockers = [
        str(item)
        for item in forwarding_payload.get("blockers", [])
        if str(item).strip()
    ]
    external_manual_items = _external_manual_items(external_payload)
    missing_artifacts = _artifact_blockers(artifacts)

    status = "ready"
    if missing_artifacts:
        status = "incomplete_packet"
    elif manual_evidence_ids or live_setup_blockers or forwarding_blockers or external_manual_items:
        status = "local_ready_live_external_blocked"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "status": status,
        "repos": {
            "BlueprintCapturePipeline": _repo_info(pipeline_repo),
            "Blueprint-WebApp": _repo_info(webapp_repo),
            "BlueprintContracts": _repo_info(contracts_repo),
            "BlueprintCapture": _repo_info(capture_repo),
        },
        "artifacts": artifacts,
        "artifact_blockers": missing_artifacts,
        "readiness_summary": {
            "paid_marketplace_launch_gate": paid_payload.get("overall_status"),
            "external_alpha_launch_gate": external_payload.get("overall_status"),
            "sim_only_beta_local_gate": sim_payload.get("status"),
            "live_pipeline_setup": live_payload.get("status"),
            "webapp_forwarding_preflight": forwarding_payload.get("status"),
        },
        "remaining_blockers": {
            "manual_live_evidence_ids": manual_evidence_ids,
            "external_alpha_manual_items": external_manual_items,
            "live_pipeline_setup_blockers": live_setup_blockers,
            "webapp_forwarding_blockers": forwarding_blockers,
        },
        "commands_to_refresh": [
            "python scripts/run_paid_marketplace_launch_gate.py",
            "python scripts/run_external_alpha_launch_gate.py",
            "python scripts/run_sim_only_beta_local_gate.py",
            "python -m blueprint_pipeline.live_pipeline_setup --no-load-env-files --output-path output/launch_audit_live_pipeline_setup_20260707.json",
            "cd ../Blueprint-WebApp && npm run pipeline:forwarding:preflight -- --require-forwarding",
        ],
        "claim_boundary": {
            "packet_links_current_local_artifacts": True,
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
        "## Artifact Checksums",
    ]
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
