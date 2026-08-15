#!/usr/bin/env python3
"""Materialize one bounded authorization before a website launch is minted.

The production dispatcher can admit a paid request either through an exact
``--execute-launch-id`` or through a standing authorization bound to one exact
launch profile.  A website launch ID does not exist until after submission, so
the former cannot be installed without racing the active dispatcher.  This
command makes the existing second path executable before submission.

It validates the exact profile and its immutable inputs, requires explicit
expiry, launch-count, and aggregate-spend bounds, and writes only the canonical
``<profile_id>.json`` target.  Existing different bytes are never overwritten.
The resulting file is read-only to its owner and the dispatcher service group,
and its digest is read back as the service account before success is reported.

This command writes authorization bytes only.  It does not submit a website
request, consume the authorization, invoke the allocator, or mutate a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import stat
import subprocess  # nosec B404 - fixed runuser/sha256sum argv only
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.host_resident_launch_inputs import PRODUCTION_LAUNCH_INPUT_ROOTS
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)
from blueprint_pipeline.task_evaluation_standing_launch_authorization import (
    SCHEMA_VERSION,
    validate_standing_authorization,
)

DEFAULT_SERVICE_ACCOUNT = "blueprint"
RUNUSER_PATH = "/usr/sbin/runuser"
SHA256SUM_PATH = "/usr/bin/sha256sum"


def _parse_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TaskEvaluationLaunchError(f"standing_authorization_{field}_invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_profile(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TaskEvaluationLaunchError("standing_authorization_profile_source_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchError(
            "standing_authorization_profile_source_invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError("standing_authorization_profile_object_required")
    return dict(value)


def _under_production_root(path: Path) -> bool:
    value = str(path)
    return any(
        value == root or value.startswith(root.rstrip("/") + "/")
        for root in PRODUCTION_LAUNCH_INPUT_ROOTS
    )


def _service_identity(target_root: Path, service_account: str | None) -> tuple[str, int, int]:
    account = service_account
    if account is None:
        account = (
            DEFAULT_SERVICE_ACCOUNT
            if _under_production_root(target_root)
            else pwd.getpwuid(os.geteuid()).pw_name
        )
    try:
        entry = pwd.getpwnam(account)
    except KeyError as exc:
        raise TaskEvaluationLaunchError(
            f"standing_authorization_service_account_missing:{account}"
        ) from exc
    return account, entry.pw_uid, entry.pw_gid


def _digest_as_account(path: Path, *, account: str, uid: int) -> str:
    if os.geteuid() == uid:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if os.geteuid() != 0:
        return ""
    try:
        completed = subprocess.run(  # nosec B603 - absolute fixed executable and argv
            [RUNUSER_PATH, "-u", account, "--", SHA256SUM_PATH, str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0 or not completed.stdout.strip():
        return ""
    return "sha256:" + completed.stdout.split()[0]


def _verify_installed(
    path: Path, *, payload: bytes, account: str, uid: int, gid: int
) -> None:
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()
    observed = _digest_as_account(path, account=account, uid=uid)
    if (
        observed != expected
        or path.stat().st_gid != gid
        or stat.S_IMODE(path.stat().st_mode) != stat.S_IRUSR | stat.S_IRGRP
    ):
        raise TaskEvaluationLaunchError(
            "standing_authorization_consumer_readback_failed"
        )


def _install_exact(
    path: Path, *, payload: bytes, account: str, uid: int, gid: int
) -> bool:
    """Verify hidden bytes first, then expose the canonical name atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TaskEvaluationLaunchError("standing_authorization_target_invalid")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise TaskEvaluationLaunchError(
                f"standing_authorization_immutable_conflict:{path.name}"
            )
        _verify_installed(path, payload=payload, account=account, uid=uid, gid=gid)
        return False

    staging = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with staging.open("xb") as stream:
            stream.write(payload)
        if staging.stat().st_gid != gid:
            os.chown(staging, -1, gid)
        staging.chmod(stat.S_IRUSR | stat.S_IRGRP)
        _verify_installed(staging, payload=payload, account=account, uid=uid, gid=gid)
        try:
            os.link(staging, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise TaskEvaluationLaunchError(
                f"standing_authorization_immutable_conflict:{path.name}"
            ) from exc
    except OSError as exc:
        raise TaskEvaluationLaunchError(
            "standing_authorization_permission_install_failed"
        ) from exc
    finally:
        staging.unlink(missing_ok=True)
    return True


def materialize_standing_launch_authorization(
    *,
    profile_path: str | Path,
    output_dir: str | Path,
    authorized_by: str,
    authorization_reference: str,
    issued_at: str,
    expires_at: str,
    max_launches: int,
    max_total_spend_usd: float,
    service_account: str | None = None,
) -> dict[str, Any]:
    """Validate and immutably install one exact-profile authorization."""

    source_input = Path(profile_path).expanduser()
    profile = _read_profile(source_input)
    blockers = validate_launch_profile(profile)
    blockers.extend(verify_profile_immutable_inputs(profile))
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))

    authorizer = str(authorized_by).strip()
    reference = str(authorization_reference).strip()
    if not authorizer:
        raise TaskEvaluationLaunchError("standing_authorization_authorized_by_missing")
    if not reference:
        raise TaskEvaluationLaunchError("standing_authorization_reference_missing")
    issued = _parse_timestamp(issued_at, field="issued_at")
    expiry = _parse_timestamp(expires_at, field="expires_at")
    if expiry <= issued:
        raise TaskEvaluationLaunchError("standing_authorization_interval_invalid")

    authorization: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile["profile_id"],
        "profile_digest": profile["profile_digest"],
        "authorized_by": authorizer,
        "authorization_reference": reference,
        "issued_at": issued.isoformat(),
        "expires_at": expiry.isoformat(),
        "max_launches": max_launches,
        "max_total_spend_usd": max_total_spend_usd,
        "provider_mutation_performed": False,
    }
    blockers = validate_standing_authorization(
        authorization,
        profile=profile,
        launches_consumed=0,
        spend_consumed_usd=0.0,
    )
    if blockers:
        raise TaskEvaluationLaunchError(",".join(blockers))

    target_input = Path(output_dir).expanduser()
    if target_input.is_symlink():
        raise TaskEvaluationLaunchError("standing_authorization_output_dir_invalid")
    target_root = target_input.resolve()
    account, uid, gid = _service_identity(target_root, service_account)
    payload = (json.dumps(authorization, sort_keys=True, separators=(",", ":")) + "\n").encode()
    target = target_root / f"{profile['profile_id']}.json"
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()
    created = _install_exact(
        target, payload=payload, account=account, uid=uid, gid=gid
    )
    return {
        "schema_version": "task_evaluation_standing_launch_authorization_publication.v1",
        "status": "published",
        "profile_id": profile["profile_id"],
        "profile_digest": profile["profile_digest"],
        "authorization_path": str(target),
        "authorization_file_digest": expected,
        "created": created,
        "service_account": account,
        "max_launches": max_launches,
        "max_total_spend_usd": max_total_spend_usd,
        "expires_at": expiry.isoformat(),
        "provider_mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--issued-at", required=True)
    parser.add_argument("--expires-at", required=True)
    parser.add_argument("--max-launches", required=True, type=int)
    parser.add_argument("--max-total-spend-usd", required=True, type=float)
    parser.add_argument("--service-account")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize_standing_launch_authorization(
            profile_path=args.profile,
            output_dir=args.output_dir,
            authorized_by=args.authorized_by,
            authorization_reference=args.authorization_reference,
            issued_at=args.issued_at,
            expires_at=args.expires_at,
            max_launches=args.max_launches,
            max_total_spend_usd=args.max_total_spend_usd,
            service_account=args.service_account,
        )
    except (OSError, TaskEvaluationLaunchError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
