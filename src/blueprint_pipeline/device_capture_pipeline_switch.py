"""The capture app's processing path was removed on 2026-11-11; its cloud entries do nothing.

The swap dispatch worker (``functions/storage_trigger.py``) and the
``blueprint-pipeline`` Cloud Run job (``capture_orchestrator.main``) existed for
captures from the BlueprintCapture app. The app is not in use (owner,
2026-11-11): every capture comes from a site filming itself and uploading it
through the WebApp, and the host listener processes those. The code those
entries ran was deleted the same day, so there is no switch to turn it back on.

The entries stay deployed as they are (how production forwards the upload
function's message is not yet verified, ``docs/CAPTURE_BRIDGE_CONTRACT.md``).
Each one writes this record (a structured log line,
``device_capture_pipeline_closed.v1``) and returns successfully: a delivery is
acknowledged, never retried, and nothing is left for an operator to clear.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

CLOSED_RECORD_SCHEMA = "device_capture_pipeline_closed.v1"
CLOSED_REASON = "device_capture_pipeline_removed"
REMOVED_ON = "2026-11-11"


def closed_device_capture_record(*, entry: str, scene_id: Any = None, capture_id: Any = None) -> dict[str, Any]:
    """The one record saying why nothing ran."""

    return {
        "schema_version": CLOSED_RECORD_SCHEMA,
        "status": "skipped",
        "reason": CLOSED_REASON,
        "entry": entry,
        "scene_id": str(scene_id) if scene_id else None,
        "capture_id": str(capture_id) if capture_id else None,
        "removed_on": REMOVED_ON,
        "detail": ("The capture app's processing path was removed on 2026-11-11; "
                   "nothing ran and nothing was bought."),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["CLOSED_REASON", "CLOSED_RECORD_SCHEMA", "REMOVED_ON", "closed_device_capture_record"]
