"""Authorize paid launches per profile instead of per launch id.

A live paid run is admitted only when the dispatcher is given
``--execute-launch-id`` equal to that launch's id. The id is minted by the
website per launch, so every run requires a human to copy it into the host's
environment file and restart the unit. That is a hand-patched env var per run:
it cannot be tested, it does not survive a host rebuild, and the operator doing
the copying is the only thing standing between a typo and a launch that will not
start.

A standing authorization moves the same decision to the profile, where it can be
written down, digest-bound, bounded, and expired. The properties the per-run
handshake provided are kept rather than traded away:

* it names one exact ``profile_id`` *and* its ``profile_digest``, so republishing
  a profile under a new digest does not inherit the old authority;
* it expires, so an authorization left behind stops admitting launches;
* it bounds both the number of launches and total spend across all of them, so a
  loop cannot spend without limit under one approval;
* each admission is counted, so the bounds hold across restarts rather than
  resetting whenever the unit is restarted.

It is strictly additive: a launch carrying a matching ``--execute-launch-id`` is
admitted exactly as before, and a launch carrying neither is refused.

Reads and writes retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "task_evaluation_standing_launch_authorization.v1"

#: Directory holding standing authorizations, one JSON file per profile.
STANDING_AUTHORIZATION_DIR_ENV = "BLUEPRINT_TASK_EVALUATION_STANDING_AUTHORIZATION_DIR"

_CONSUMED_DIRECTORY = "consumed"

_REQUIRED_FIELDS = (
    "schema_version",
    "profile_id",
    "profile_digest",
    "max_launches",
    "max_total_spend_usd",
    "expires_at",
)


class StandingAuthorizationError(ValueError):
    """The standing authorization cannot admit a launch."""


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def validate_standing_authorization(
    authorization: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    launches_consumed: int,
    spend_consumed_usd: float,
    now: datetime | None = None,
) -> list[str]:
    """Return every reason this authorization cannot admit a launch.

    All reasons are collected rather than short-circuited: an operator fixing a
    stale authorization should not have to rediscover the next problem on the
    next paid attempt.
    """
    blockers: list[str] = []
    moment = now or datetime.now(timezone.utc)

    for field in _REQUIRED_FIELDS:
        if authorization.get(field) in (None, ""):
            blockers.append(f"standing_authorization_missing_{field}")
    if authorization.get("schema_version") not in (None, "", SCHEMA_VERSION):
        blockers.append("standing_authorization_schema_version_mismatch")

    profile_id = str(profile.get("profile_id") or "")
    if str(authorization.get("profile_id") or "") != profile_id:
        blockers.append("standing_authorization_profile_mismatch")
    # Digest as well as id: a republished profile is a different artifact even
    # under the same name, and must not inherit an approval granted to the old
    # bytes.
    if str(authorization.get("profile_digest") or "") != str(
        profile.get("profile_digest") or ""
    ):
        blockers.append("standing_authorization_profile_digest_mismatch")

    expires_at = _parse_timestamp(authorization.get("expires_at"))
    if authorization.get("expires_at") not in (None, "") and expires_at is None:
        blockers.append("standing_authorization_expires_at_invalid")
    elif expires_at is not None and expires_at <= moment:
        blockers.append("standing_authorization_expired")

    max_launches = authorization.get("max_launches")
    if isinstance(max_launches, bool) or not isinstance(max_launches, int):
        if "standing_authorization_missing_max_launches" not in blockers:
            blockers.append("standing_authorization_max_launches_invalid")
    elif max_launches <= 0:
        blockers.append("standing_authorization_max_launches_invalid")
    elif launches_consumed >= max_launches:
        blockers.append("standing_authorization_launches_exhausted")

    max_spend = _positive_number(authorization.get("max_total_spend_usd"))
    if max_spend is None:
        if "standing_authorization_missing_max_total_spend_usd" not in blockers:
            blockers.append("standing_authorization_max_total_spend_usd_invalid")
    else:
        profile_spend = _positive_number(
            (profile.get("allocator") or {}).get("max_spend_usd")
        ) or 0.0
        # The next launch must fit under the ceiling *before* it runs; checking
        # afterwards would authorize the overspend it was meant to prevent.
        if spend_consumed_usd + profile_spend > max_spend:
            blockers.append("standing_authorization_spend_ceiling_reached")

    return sorted(set(blockers))


def load_standing_authorization(
    *, profile_id: str, directory: str | Path
) -> dict[str, Any] | None:
    """Return the authorization for one profile, or ``None`` when absent.

    Absence is not an error: a host with no standing authorizations simply has
    none, and the per-launch handshake remains the way in.
    """
    root = Path(directory).expanduser()
    path = root / f"{profile_id}.json"
    if not path.is_file():
        return None
    if path.is_symlink():
        # A symlink lets the bytes behind an approval change without the
        # approval changing.
        raise StandingAuthorizationError(f"standing_authorization_source_invalid:{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StandingAuthorizationError(
            f"standing_authorization_unreadable:{path.name}"
        ) from exc
    if not isinstance(value, Mapping):
        raise StandingAuthorizationError(
            f"standing_authorization_not_an_object:{path.name}"
        )
    return dict(value)


def consumption_totals(*, directory: str | Path, profile_id: str) -> tuple[int, float]:
    """Count launches and spend already admitted under this profile's approval.

    Read from disk on every call rather than held in memory: the bounds have to
    survive a unit restart, and an in-process counter resets exactly when an
    operator restarts the service to change something.
    """
    root = Path(directory).expanduser() / _CONSUMED_DIRECTORY / profile_id
    if not root.is_dir():
        return (0, 0.0)
    launches = 0
    spend = 0.0
    for path in sorted(root.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # An unreadable record is spend we cannot account for, so refusing
            # beats resuming from a total we know is too low.
            raise StandingAuthorizationError(
                f"standing_authorization_consumption_unreadable:{path.name}"
            ) from exc
        launches += 1
        amount = record.get("max_spend_usd")
        if isinstance(amount, (int, float)) and not isinstance(amount, bool):
            spend += float(amount)
    return (launches, spend)


def record_launch(
    *, directory: str | Path, profile_id: str, launch_id: str, max_spend_usd: float
) -> Path:
    """Record one admitted launch. Exclusive create, so a replay cannot re-spend."""
    root = Path(directory).expanduser() / _CONSUMED_DIRECTORY / profile_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{launch_id}.json"
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "profile_id": profile_id,
            "launch_id": launch_id,
            "max_spend_usd": float(max_spend_usd),
        },
        sort_keys=True,
    )
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise StandingAuthorizationError(
            f"standing_authorization_launch_already_recorded:{launch_id}"
        ) from exc
    return path


def consume_standing_authorization_once(
    *,
    profile: Mapping[str, Any],
    directory: str | Path,
    launch_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Atomically validate and consume one profile-bound launch allowance.

    Validation and recording must share one lock.  Otherwise two distinct
    website launch ids can both observe ``launches_consumed == 0`` under a
    one-launch authority and then create two different consumption records.
    The per-profile lock serializes that check-and-create boundary across
    dispatcher processes; the durable exclusive record remains the replay
    guard for one exact launch id.
    """

    profile_id = str(profile.get("profile_id") or "")
    if not profile_id or not launch_id:
        raise StandingAuthorizationError("standing_authorization_consumption_identity_invalid")
    consumed_root = (
        Path(directory).expanduser() / _CONSUMED_DIRECTORY / profile_id
    )
    consumed_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = consumed_root / ".consume.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "r+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            launches, spend = consumption_totals(
                directory=directory, profile_id=profile_id
            )
            decision = standing_authorization_admits(
                profile=profile,
                directory=directory,
                launches_consumed=launches,
                spend_consumed_usd=spend,
                now=now,
            )
            if not decision.get("admitted"):
                return {**decision, "consumed": False}
            max_spend = _positive_number(
                (profile.get("allocator") or {}).get("max_spend_usd")
            )
            if max_spend is None:
                raise StandingAuthorizationError(
                    "standing_authorization_profile_spend_invalid"
                )
            record = record_launch(
                directory=directory,
                profile_id=profile_id,
                launch_id=launch_id,
                max_spend_usd=max_spend,
            )
            return {
                **decision,
                "consumed": True,
                "launches_consumed": launches + 1,
                "spend_consumed_usd": spend + max_spend,
                "consumption_record": record.name,
            }
    except OSError as exc:
        raise StandingAuthorizationError(
            "standing_authorization_consumption_lock_failed"
        ) from exc


def standing_authorization_admits(
    *,
    profile: Mapping[str, Any],
    directory: str | Path | None,
    launches_consumed: int,
    spend_consumed_usd: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Decide whether a standing authorization admits this profile's launch."""
    profile_id = str(profile.get("profile_id") or "")
    if not directory or not profile_id:
        return {
            "admitted": False,
            "reason": "standing_authorization_not_configured",
            "blockers": [],
        }
    try:
        authorization = load_standing_authorization(
            profile_id=profile_id, directory=directory
        )
    except StandingAuthorizationError as exc:
        return {"admitted": False, "reason": str(exc), "blockers": [str(exc)]}
    if authorization is None:
        return {
            "admitted": False,
            "reason": "standing_authorization_absent",
            "blockers": [],
        }
    blockers = validate_standing_authorization(
        authorization,
        profile=profile,
        launches_consumed=launches_consumed,
        spend_consumed_usd=spend_consumed_usd,
        now=now,
    )
    return {
        "admitted": not blockers,
        "reason": "standing_authorization_admitted" if not blockers else "blocked",
        "blockers": blockers,
        "profile_id": profile_id,
        "max_launches": authorization.get("max_launches"),
        "max_total_spend_usd": authorization.get("max_total_spend_usd"),
        "expires_at": authorization.get("expires_at"),
        "launches_consumed": launches_consumed,
        "spend_consumed_usd": spend_consumed_usd,
    }


__all__ = [
    "SCHEMA_VERSION",
    "STANDING_AUTHORIZATION_DIR_ENV",
    "StandingAuthorizationError",
    "consume_standing_authorization_once",
    "consumption_totals",
    "load_standing_authorization",
    "record_launch",
    "standing_authorization_admits",
    "validate_standing_authorization",
]
