"""Release Pub/Sub handoff leases that an older release deferred.

A retryable handoff is deferred by holding its Pub/Sub lease for
``RETRY_DEFER_SECONDS`` (``_AckDeadlineHeartbeat.defer_retry``); nothing local
recorded that it was waiting, so after a deploy that fixed the cause it still
waited out the lease plus Pub/Sub's retry backoff (11-20 minutes). Each
deferral now leaves a private record of its lease, and the first listener tick
under a newer release ends those leases so Pub/Sub redelivers the handoffs now.

Pub/Sub stays the delivery authority: a released lease is an ordinary nack, so
redelivery still follows the subscription's retry policy, counts a delivery
attempt, and dead-letters at the same cap. A handoff on its last attempt is left
alone, since releasing it could only dead-letter it sooner. Records are a cache
of lease credentials, never evidence; any failure here leaves the message on its
existing lease, exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

from .task_evaluation_release_identity import running_release_commit

SCHEMA_VERSION = "pubsub_retry_deferral.v1"
DEFERRALS_DIRNAME = ".pubsub_retry_deferrals"
_COMMIT = re.compile(r"[0-9a-f]{40}")

logger = logging.getLogger(__name__)


def _record_path(storage_root: Path, received: Any) -> Path | None:
    message = getattr(received, "message", None)
    identity = str(getattr(message, "message_id", "") or "")
    if not identity:
        return None
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return Path(storage_root) / DEFERRALS_DIRNAME / f"{token}.json"


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_deferral(
    *,
    storage_root: Path,
    subscription: str,
    received: Any,
    payload_digest: str,
    lease_seconds: int,
    now: float | None = None,
) -> None:
    """Remember a deferred lease so a later release can end it early."""
    try:
        path = _record_path(storage_root, received)
        ack_id = str(getattr(received, "ack_id", "") or "")
        if path is None or not ack_id:
            return
        attempt = getattr(received, "delivery_attempt", None)
        deferred_at = time.time() if now is None else float(now)
        _write_private(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "subscription": subscription,
                "ack_id": ack_id,
                "message_id": str(received.message.message_id),
                "payload_sha256": payload_digest,
                "delivery_attempt": (
                    attempt if isinstance(attempt, int) and not isinstance(attempt, bool) else None
                ),
                "release_commit": running_release_commit(),
                "deferred_at_epoch": deferred_at,
                "lease_expires_at_epoch": deferred_at + int(lease_seconds),
            },
        )
    except Exception:  # noqa: BLE001 - the message still retries on its lease
        logger.exception("pubsub_handoff.retry_deferral_record_failed")


def clear_deferral(*, storage_root: Path, received: Any) -> None:
    """Forget a handoff's deferral once Pub/Sub no longer holds it for us."""
    try:
        path = _record_path(storage_root, received)
        if path is not None:
            path.unlink(missing_ok=True)
    except OSError:
        logger.exception("pubsub_handoff.retry_deferral_clear_failed")


def _read(path: Path) -> dict[str, Any] | None:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("ack_id"), str)
        or not value["ack_id"]
        or not isinstance(value.get("subscription"), str)
        or not isinstance(value.get("release_commit"), str)
        or not isinstance(value.get("lease_expires_at_epoch"), (int, float))
    ):
        return None
    return value


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # not a record this module wrote (e.g. a directory): leave it
        logger.warning("pubsub_handoff.retry_deferral_entry_unremovable", extra={"path": str(path)})


def wake_older_release_deferrals(
    *,
    storage_root: Path,
    subscriber: Any,
    subscription: str,
    max_delivery_attempts: int,
    now: float | None = None,
) -> dict[str, int]:
    """End the leases of handoffs deferred under any release but the running one.

    Never raises: a wake that cannot run leaves every message on its lease.
    """
    counts = {"woken": 0, "expired": 0, "failed": 0, "kept": 0}
    try:
        _wake(storage_root=storage_root, subscriber=subscriber, subscription=subscription,
              max_delivery_attempts=max_delivery_attempts, now=now, counts=counts)
    except Exception:  # noqa: BLE001 - the pull below must still run
        logger.exception("pubsub_handoff.retry_deferral_wake_unavailable")
    return counts


def _wake(
    *,
    storage_root: Path,
    subscriber: Any,
    subscription: str,
    max_delivery_attempts: int,
    now: float | None,
    counts: dict[str, int],
) -> None:
    release = running_release_commit()
    directory = Path(storage_root) / DEFERRALS_DIRNAME
    if _COMMIT.fullmatch(release or "") is None or not directory.is_dir():
        return
    current = time.time() if now is None else float(now)
    for path in sorted(directory.glob("*.json")):
        record = _read(path)
        if record is None:
            _discard(path)
            continue
        if record["subscription"] != subscription:
            counts["kept"] += 1
            continue
        if current >= float(record["lease_expires_at_epoch"]):
            _discard(path)  # the lease already ended; Pub/Sub holds it
            counts["expired"] += 1
            continue
        attempt = record.get("delivery_attempt")
        if record["release_commit"] == release or (
            isinstance(attempt, int) and attempt >= max_delivery_attempts
        ):
            counts["kept"] += 1
            continue
        try:
            subscriber.modify_ack_deadline(
                request={
                    "subscription": subscription,
                    "ack_ids": [record["ack_id"]],
                    "ack_deadline_seconds": 0,
                }
            )
            counts["woken"] += 1
        except Exception:  # noqa: BLE001 - the message still retries on its lease
            logger.warning(
                "pubsub_handoff.retry_deferral_wake_failed",
                extra={"message_id": record.get("message_id")},
            )
            counts["failed"] += 1
        _discard(path)
    if counts["woken"] or counts["failed"]:
        logger.info(
            "pubsub_handoff.retry_deferrals_woken",
            extra={"release_commit": release, **counts},
        )
