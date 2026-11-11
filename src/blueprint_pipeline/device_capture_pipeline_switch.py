"""The capture app's processing path, closed unless one switch turns it on.

The swap dispatch worker (``functions/storage_trigger.py``) and the
``blueprint-pipeline`` Cloud Run job (``capture_orchestrator.main``) exist for
captures from the BlueprintCapture app. The app is not in use (owner,
2026-11-11): every capture comes from a site filming itself through the WebApp,
and the host listener processes those. The device path still holds six paid
call sites that cannot see the account's monthly limit
(``scripts/paid_api_call_sites.json``, class ``unadmitted``), so a message that
still reaches it must not be able to spend.

So the path does nothing unless ``BLUEPRINT_DEVICE_CAPTURE_PIPELINE=enabled``.
When it is off, the entry writes one record saying why (a structured log line,
``device_capture_pipeline_closed.v1``) and exits successfully: a delivery is
acknowledged, never retried, and nothing is left for an operator to clear.
Turning the switch on requires admitting those six call sites first
(``docs/CAPTURE_BRIDGE_CONTRACT.md``).

A website capture is refused by the dispatch whatever the switch says
(``website_capture_entry.is_website_capture_identity``).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping

SWITCH_ENV = "BLUEPRINT_DEVICE_CAPTURE_PIPELINE"
CLOSED_RECORD_SCHEMA = "device_capture_pipeline_closed.v1"
CLOSED_REASON = "device_capture_pipeline_closed"


def device_capture_pipeline_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(SWITCH_ENV) or "").strip().lower() == "enabled"


def closed_device_capture_record(*, entry: str, scene_id: Any = None, capture_id: Any = None,
                                 environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """None when the path is switched on; otherwise the one record saying why nothing ran."""

    if device_capture_pipeline_enabled(environ):
        return None
    return {
        "schema_version": CLOSED_RECORD_SCHEMA,
        "status": "skipped",
        "reason": CLOSED_REASON,
        "entry": entry,
        "scene_id": str(scene_id) if scene_id else None,
        "capture_id": str(capture_id) if capture_id else None,
        "switch": SWITCH_ENV,
        "detail": "The capture app's processing path is closed; nothing ran and nothing was bought.",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["CLOSED_REASON", "CLOSED_RECORD_SCHEMA", "SWITCH_ENV", "closed_device_capture_record",
           "device_capture_pipeline_enabled"]
