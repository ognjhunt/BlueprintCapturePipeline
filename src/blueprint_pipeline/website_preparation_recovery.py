"""Recovery that keeps failing, counted per capture where the stall alarm and the door can read it.

The listener tick's recovery passes (the host's preparation retry, the Atlas-wait
replay and the other catch-up drains) used to swallow their errors at debug
level. Each failure now logs at warning with its capture id and counts here, in
``RECOVERY_FILE`` under the listener's storage root; a pass that succeeds clears
its capture's count. Kept free of pipeline imports: the listener's light lane
imports it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json

RECOVERY_FILE = ".website_preparation_recovery_failures.json"
RECOVERY_SCHEMA = "website_preparation_recovery_failures.v1"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def recovery_failures(storage_root: Path) -> dict[str, Any]:
    """The per-capture count of recovery passes that failed in a row (``RECOVERY_FILE``)."""
    path = Path(storage_root) / RECOVERY_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
    except (OSError, ValueError):
        value = {}
    if not isinstance(value, dict) or value.get("schema_version") != RECOVERY_SCHEMA:
        value = {}
    captures = value.get("captures")
    return {"schema_version": RECOVERY_SCHEMA, "captures": dict(captures) if isinstance(captures, dict) else {}}


def note_recovery(listener: Any, storage_root: Path, *, key: str, recovery: str,
                  error: BaseException | str | None = None, capture_id: str | None = None) -> None:
    """Count a failed recovery pass at warning level (``error``), or clear the count. Never raises."""
    try:
        state = recovery_failures(storage_root)
        previous = state["captures"].get(key)
        if error is None:
            if previous is None:
                return
            state["captures"].pop(key)
        else:
            text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
            count = (previous.get("consecutive_failures", 0) if isinstance(previous, dict) else 0) + 1
            state["captures"][key] = {"capture_id": capture_id, "recovery": recovery,
                                      "consecutive_failures": count, "last_error": text[:500],
                                      "last_failed_at": _iso_now()}
            listener.logger.warning("pubsub_handoff.preparation_recovery_failed", extra={
                "capture_id": capture_id, "recovery": recovery, "consecutive_failures": count,
                "error": text[:500]})
        write_json(Path(storage_root) / RECOVERY_FILE, {**state, "updated_at": _iso_now()})
    except Exception:  # noqa: BLE001 - counting never stops the tick
        listener.logger.warning("pubsub_handoff.preparation_recovery_count_unavailable", extra={"key": key})
