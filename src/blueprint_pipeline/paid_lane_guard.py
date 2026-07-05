"""Unified paid-lane guard: one pre-spend chokepoint + crash-safe orphan reaping.

Every paid GPU lane (splat render, RunPod WAM async, robot-eval provider launcher)
must route through :func:`require_pre_spend_preflight` before any billable provider
call, and must open a ``pending_teardown.v1`` record around the launch so that a
process crash between launch and collect never leaves a silently billing
allocation: the standalone :func:`reap_orphans` entrypoint (also a CLI:
``python -m blueprint_pipeline.paid_lane_guard reap-orphans``) reads those records,
queries live provider state, force-terminates anything past its max age, and only
closes a record on an API-confirmed terminal state (see
``build_teardown_proof(status_source="provider_api")``).

Fail-closed rules:
- a failing preflight raises :class:`PreSpendPreflightBlocked` — identically for
  every lane; there is no lane-specific soft path around it;
- a pending-teardown record can only be closed by a PASSING teardown proof;
- a reap that cannot API-confirm the terminal state leaves the record open and
  reports ``open_billing_risk`` instead of assuming the money stopped.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

from .common import ensure_dir, utc_now_iso, write_json
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_pre_spend_preflight,
    build_teardown_proof,
)

PENDING_TEARDOWN_SCHEMA_VERSION = "pending_teardown.v1"
ORPHAN_REAP_REPORT_SCHEMA_VERSION = "orphan_reap_report.v1"
PRE_SPEND_PREFLIGHT_RECORD_NAME = "pre_spend_preflight.json"
PENDING_TEARDOWN_DIR_ENV = "BLUEPRINT_PENDING_TEARDOWN_DIR"
DEFAULT_PENDING_TEARDOWN_DIR = Path.home() / ".blueprint-pending-teardowns"
# An open allocation with no collect for this long is an orphan, not patience.
DEFAULT_PENDING_TEARDOWN_MAX_AGE_SECONDS = 7200


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _registry_dir(registry_dir: str | Path | None) -> Path:
    if registry_dir:
        return Path(registry_dir).expanduser()
    env_dir = str(os.getenv(PENDING_TEARDOWN_DIR_ENV) or "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return DEFAULT_PENDING_TEARDOWN_DIR


# ---------------------------------------------------------------------------
# Pre-spend chokepoint.
# ---------------------------------------------------------------------------


class PreSpendPreflightBlocked(RuntimeError):
    """Raised by the chokepoint when a paid launch must not spend."""

    def __init__(self, preflight: Mapping[str, Any]):
        self.preflight = dict(preflight)
        blockers = ",".join(str(b) for b in self.preflight.get("blockers") or [])
        super().__init__(f"pre_spend_preflight_blocked:{blockers}")


def require_pre_spend_preflight(
    *,
    lane: str,
    provider: str,
    credential_present: bool,
    capacity_evidence: Mapping[str, Any] | None,
    image_contract: Mapping[str, Any] | None,
    runtime_contract: Mapping[str, Any] | None,
    spend_gate_open: Any = None,
    record_dir: str | Path | None = None,
) -> dict[str, Any]:
    """The single fail-closed gate every paid launch must pass through.

    Builds the shared ``pre_spend_preflight.v1`` contract from the lane's
    evidence, persists it when ``record_dir`` is given (PASS or FAIL — the
    artifact always exists), and raises :class:`PreSpendPreflightBlocked` unless
    the preflight passed. Lanes must not call provider launch APIs on the
    exception path.
    """
    preflight = build_pre_spend_preflight(
        provider=provider,
        credential_present=credential_present,
        capacity_evidence=capacity_evidence,
        image_contract=image_contract,
        runtime_contract=runtime_contract,
        spend_gate_open=spend_gate_open,
    )
    lane_name = str(lane or "").strip()
    preflight["lane"] = lane_name or None
    if not lane_name:
        preflight["blockers"] = sorted(
            {*preflight.get("blockers", []), "pre_spend_chokepoint_lane_missing"}
        )
        preflight["status"] = "FAIL"
        preflight["spend_allowed"] = False
    if record_dir:
        record_path = Path(record_dir)
        ensure_dir(record_path)
        write_json(record_path / PRE_SPEND_PREFLIGHT_RECORD_NAME, preflight)
    if preflight.get("status") != "PASS":
        raise PreSpendPreflightBlocked(preflight)
    return preflight


def image_contract_from_ref(image_ref: str) -> dict[str, Any]:
    """Shared image-contract evidence: a digest or non-``latest`` tag counts as pinned."""
    image = str(image_ref or "").strip()
    image_leaf = image.rsplit("/", 1)[-1]
    tag = image_leaf.rsplit(":", 1)[-1] if ":" in image_leaf else ""
    digest_pinned = "@sha256:" in image
    tag_pinned = bool(tag and tag != "latest")
    return {
        "image_ref": image or None,
        "pinned": bool(digest_pinned or tag_pinned),
        "digest": image.split("@", 1)[1] if digest_pinned else None,
    }


# ---------------------------------------------------------------------------
# pending_teardown.v1 registry.
# ---------------------------------------------------------------------------


def _record_filename(provider: str, run_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{provider}-{run_id}").strip("-")
    return f"{slug or 'pending-teardown'}.json"


def _read_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def open_pending_teardown(
    *,
    provider: str,
    lane: str,
    run_id: str,
    instance_id: str = "",
    job_dir: str | Path | None = None,
    max_age_seconds: int = DEFAULT_PENDING_TEARDOWN_MAX_AGE_SECONDS,
    registry_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the teardown obligation BEFORE the billable launch call.

    ``instance_id`` is usually unknown at open time (the provider has not
    allocated yet); bind it with :func:`bind_pending_teardown_instance` as soon
    as the create/start response returns an id. A crash in between still leaves
    this record for the reaper.
    """
    registry = _registry_dir(registry_dir)
    ensure_dir(registry)
    path = registry / _record_filename(provider, run_id)
    record = {
        "schema_version": PENDING_TEARDOWN_SCHEMA_VERSION,
        "provider": str(provider or "").strip().lower(),
        "lane": str(lane or "").strip(),
        "run_id": str(run_id or "").strip(),
        "instance_id": str(instance_id or "").strip() or None,
        "job_dir": str(job_dir) if job_dir else None,
        "started_at": utc_now_iso(),
        "started_at_epoch": time.time(),
        "max_age_seconds": max(1, int(max_age_seconds)),
        "status": "open",
        "teardown_proof": None,
        "path": str(path),
    }
    write_json(path, record)
    return record


def bind_pending_teardown_instance(
    record_path: str | Path, instance_id: str
) -> dict[str, Any]:
    path = Path(record_path)
    record = _read_record(path)
    record["instance_id"] = str(instance_id or "").strip() or None
    write_json(path, record)
    return record


def close_pending_teardown(
    record_path: str | Path, teardown_proof: Mapping[str, Any]
) -> dict[str, Any]:
    """Close only on a PASSING teardown proof; anything else stays open."""
    path = Path(record_path)
    record = _read_record(path)
    proof = _mapping(teardown_proof)
    if str(proof.get("status") or "").strip().upper() != "PASS":
        record["close_refused_reason"] = "teardown_proof_not_passed"
        record["last_refused_teardown_proof"] = proof or None
        write_json(path, record)
        return record
    record["status"] = "closed"
    record["closed_at"] = utc_now_iso()
    record["teardown_proof"] = proof
    record.pop("close_refused_reason", None)
    write_json(path, record)
    return record


def cancel_pending_teardown(
    record_path: str | Path, *, reason: str, evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Cancel a record for a launch that never produced an allocation id.

    Only for the no-allocation case (provider returned no instance id): keeping
    such records open would flood every sweep with unresolvable alarms. The
    launch response is kept as evidence. A record that HAS an instance id must be
    closed by a passing teardown proof, never cancelled.
    """
    path = Path(record_path)
    record = _read_record(path)
    if str(record.get("instance_id") or "").strip():
        record["cancel_refused_reason"] = "instance_id_bound_requires_teardown_proof"
        write_json(path, record)
        return record
    record["status"] = "cancelled_no_allocation"
    record["cancelled_at"] = utc_now_iso()
    record["cancel_reason"] = str(reason or "").strip() or "no_allocation"
    record["cancel_evidence"] = _mapping(evidence) or None
    write_json(path, record)
    return record


def load_pending_teardowns(
    *, registry_dir: str | Path | None = None, include_closed: bool = False
) -> list[dict[str, Any]]:
    registry = _registry_dir(registry_dir)
    if not registry.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(registry.glob("*.json")):
        record = _read_record(path)
        if record.get("schema_version") != PENDING_TEARDOWN_SCHEMA_VERSION:
            continue
        record["path"] = str(path)
        if include_closed or record.get("status") == "open":
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Orphan reaper.
# ---------------------------------------------------------------------------


def provider_state_from_inspect(inspect_result: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a provider ``inspect`` result into API-confirmed state evidence.

    ``api_confirmed`` is True only when the provider actually answered the state
    query (200 with a status, or 404/410 meaning the allocation is gone). Probe
    errors and missing credentials are NOT confirmation of anything.
    """
    result = _mapping(inspect_result)
    http = result.get("http")
    try:
        http_code = int(http)
    except (TypeError, ValueError):
        http_code = 0
    if http_code in (404, 410):
        return {"provider_status": "not_found", "api_confirmed": True, "http": http_code}
    if str(result.get("status") or "") == "observed" and http_code == 200:
        desired = str(result.get("desiredStatus") or "").strip().lower()
        return {
            "provider_status": desired or "present",
            "api_confirmed": True,
            "http": http_code,
        }
    return {"provider_status": "", "api_confirmed": False, "http": http_code}


def _default_provider_client(provider: str):
    from .gpu_render_providers import get_render_provider

    return get_render_provider(provider)


def _reap_one(
    record: Mapping[str, Any],
    *,
    client: Any,
    now_epoch: float,
    dry_run: bool,
) -> dict[str, Any]:
    provider = str(record.get("provider") or "").strip().lower()
    instance_id = str(record.get("instance_id") or "").strip()
    entry: dict[str, Any] = {
        "run_id": record.get("run_id"),
        "lane": record.get("lane"),
        "provider": provider,
        "instance_id": instance_id or None,
        "record_path": record.get("path"),
        "age_seconds": round(now_epoch - float(record.get("started_at_epoch") or 0), 1),
        "open_billing_risk": False,
        "teardown_proof": None,
    }
    if not instance_id:
        entry["outcome"] = "unresolvable_instance_id_missing"
        entry["open_billing_risk"] = True
        return entry
    if client is None:
        entry["outcome"] = "provider_client_unavailable"
        entry["open_billing_risk"] = True
        return entry
    entry["pre_terminate_state"] = provider_state_from_inspect(
        client.inspect(instance_id)
    )
    if dry_run:
        entry["outcome"] = "would_terminate"
        return entry
    terminate_result = client.terminate(instance_id)
    entry["terminate_result"] = _mapping(terminate_result) or {
        "status": str(terminate_result)
    }
    verify = provider_state_from_inspect(client.inspect(instance_id))
    entry["post_terminate_state"] = verify
    proof = build_teardown_proof(
        provider=provider,
        allocation_id=instance_id,
        terminate_requested=True,
        provider_terminal_status=verify["provider_status"] or None,
        verified_at=utc_now_iso() if verify["api_confirmed"] else None,
        status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API
        if verify["api_confirmed"]
        else None,
    )
    entry["teardown_proof"] = proof
    if proof.get("status") == "PASS":
        close_pending_teardown(str(record.get("path")), proof)
        entry["outcome"] = "reaped_terminal_proven"
    else:
        entry["outcome"] = "terminate_not_proven"
        entry["open_billing_risk"] = True
    return entry


def reap_orphans(
    *,
    registry_dir: str | Path | None = None,
    provider_clients: Mapping[str, Any] | None = None,
    now_epoch: float | None = None,
    max_age_override_seconds: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sweep open pending_teardown records independent of any launching process.

    For every open record past its max age: query provider state, force-terminate,
    re-query, and close the record only on an API-confirmed terminal state. A
    record the sweep cannot prove terminal stays open and is reported as an open
    billing risk — never silently dropped.
    """
    now = time.time() if now_epoch is None else float(now_epoch)
    clients = dict(provider_clients or {})
    records = load_pending_teardowns(registry_dir=registry_dir)
    entries: list[dict[str, Any]] = []
    for record in records:
        max_age = (
            int(max_age_override_seconds)
            if max_age_override_seconds is not None
            else int(record.get("max_age_seconds") or DEFAULT_PENDING_TEARDOWN_MAX_AGE_SECONDS)
        )
        age = now - float(record.get("started_at_epoch") or 0)
        if age < max_age:
            entries.append(
                {
                    "run_id": record.get("run_id"),
                    "lane": record.get("lane"),
                    "provider": record.get("provider"),
                    "instance_id": record.get("instance_id"),
                    "record_path": record.get("path"),
                    "age_seconds": round(age, 1),
                    "outcome": "not_due",
                    "open_billing_risk": False,
                    "teardown_proof": None,
                }
            )
            continue
        provider = str(record.get("provider") or "").strip().lower()
        client = clients.get(provider)
        if client is None:
            try:
                client = _default_provider_client(provider)
            except ValueError:
                client = None
        entries.append(_reap_one(record, client=client, now_epoch=now, dry_run=dry_run))
    reaped = [e for e in entries if e.get("outcome") == "reaped_terminal_proven"]
    risks = [e for e in entries if e.get("open_billing_risk")]
    return {
        "schema_version": ORPHAN_REAP_REPORT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "registry_dir": str(_registry_dir(registry_dir)),
        "dry_run": bool(dry_run),
        "records": entries,
        "open_record_count": len(records),
        "due_count": len([e for e in entries if e.get("outcome") != "not_due"]),
        "reaped_count": len(reaped),
        "open_billing_risk_count": len(risks),
        "open_billing_risks": [
            {"run_id": e.get("run_id"), "instance_id": e.get("instance_id"), "outcome": e.get("outcome")}
            for e in risks
        ],
        "claim_boundary": (
            "The reap report proves only which pending allocations reached an "
            "API-confirmed terminal state. It says nothing about run success, and an "
            "empty registry does not prove no allocation is billing — only that none "
            "was recorded."
        ),
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Paid-lane guard utilities (pending-teardown orphan reaper)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    reap = sub.add_parser(
        "reap-orphans",
        help="Force-terminate and prove terminal any pending teardown past max age.",
    )
    reap.add_argument("--registry-dir", default=None)
    reap.add_argument("--max-age-seconds", type=int, default=None)
    reap.add_argument("--dry-run", action="store_true")
    reap.add_argument("--json-report", default=None, help="Also write the report here.")
    args = parser.parse_args(argv)

    report = reap_orphans(
        registry_dir=args.registry_dir,
        max_age_override_seconds=args.max_age_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, default=str))
    if args.json_report:
        write_json(Path(args.json_report), report)
    return 0 if report["open_billing_risk_count"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
