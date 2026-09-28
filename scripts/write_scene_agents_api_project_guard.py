"""Record an operator-observed OpenAI project hard limit for one scene lane.

This command cannot query or change dashboard hard-limit enforcement. Run it
only after an authorized operator has inspected that project's limit and the
dedicated key identity in the OpenAI dashboard. It records that observation in
an owner-only file; the signed website policy must carry the printed digest.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def _cents(raw: str) -> int:
    try:
        cents = Decimal(raw) * 100
    except InvalidOperation as exc:
        raise ValueError("guard_amount_invalid") from exc
    if cents <= 0 or cents != cents.to_integral_value():
        raise ValueError("guard_amount_invalid")
    return int(cents)


def write_guard(*, output: Path, project_id: str, credential_id: str,
                hard_limit_usd: str, stage_cap_usd: str,
                expires_in_seconds: int, observed_at: float) -> str:
    """Write once, then return only the canonical receipt digest."""
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or output.parent.is_symlink() or not output.parent.is_dir()
            or stat.S_IMODE(output.parent.stat().st_mode) & 0o077
            or output.parent.stat().st_uid != os.geteuid()
            or not project_id.startswith("proj_") or not credential_id
            or not 1800 <= expires_in_seconds <= 86_400):
        raise ValueError("guard_output_or_identity_invalid")
    hard_limit_cents = _cents(hard_limit_usd)
    if hard_limit_cents > _cents(stage_cap_usd):
        raise ValueError("guard_hard_limit_exceeds_stage_cap")
    receipt = {"schema_version": "blueprint_agent_project_admission_observation.v1",
        "project_id": project_id, "credential_id": credential_id,
        "dashboard_hard_limit_enabled": True,
        "disclosure_scope": "task_asset_source_frames_and_metric_envelope",
        "budget_policy": "project_guard_accepted_uncertainty",
        "session_retention": "until_deleted", "trace_retention": "provider_default",
        "provider_api_region": "us", "observed_at": observed_at,
        "expires_at": observed_at + expires_in_seconds,
        "spend_limit": {"object": "project.spend_limit", "currency": "USD",
                        "interval": "month", "threshold_amount": hard_limit_cents}}
    body = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result = "sha256:" + hashlib.sha256(body).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(output, flags, 0o600), "wb") as stream:
        stream.write(body + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--credential-id", required=True)
    parser.add_argument("--dashboard-hard-limit-usd", required=True)
    parser.add_argument("--signed-stage-cap-usd", required=True)
    parser.add_argument("--expires-in-seconds", type=int, default=3600)
    parser.add_argument("--ack-dashboard-hard-limit-enabled", action="store_true", required=True)
    parser.add_argument("--ack-dedicated-project-and-key", action="store_true", required=True)
    args = parser.parse_args()
    print(write_guard(output=args.output, project_id=args.project_id,
        credential_id=args.credential_id,
        hard_limit_usd=args.dashboard_hard_limit_usd,
        stage_cap_usd=args.signed_stage_cap_usd,
        expires_in_seconds=args.expires_in_seconds,
        observed_at=time.time()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
