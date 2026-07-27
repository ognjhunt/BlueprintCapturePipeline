"""Cycle-free owner-to-watchdog cancellation artifact contract."""

from __future__ import annotations

import os
from pathlib import Path

from .common import utc_now_iso, write_json


WATCHDOG_EVIDENCE_NAME = "groot_oscar_runpod_canary_watchdog.json"
OWNER_TEARDOWN_CANCEL_NAME = "groot_oscar_runpod_canary_watchdog_cancel.json"
OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION = (
    "groot_oscar_runpod_canary_watchdog_cancel.v1"
)


def write_owner_teardown_cancel_request(
    *, root: Path, pod_name_prefix: str, provider_name: str, instance_id: str
) -> dict[str, object]:
    """Persist the private request that asks a watchdog to verify zero early."""

    payload: dict[str, object] = {
        "schema_version": OWNER_TEARDOWN_CANCEL_SCHEMA_VERSION,
        "requested_at": utc_now_iso(),
        "requested_by": "qualification_owner_teardown",
        "provider": provider_name,
        "instance_id": instance_id,
        "pod_name_prefix": pod_name_prefix,
        "provider_absence_confirmed": True,
        "provider_absence_evidence": (
            "provider_api_exact_id_prefix_and_global_inventory"
        ),
        "raw_secret_values_recorded": False,
    }
    path = root / OWNER_TEARDOWN_CANCEL_NAME
    write_json(path, payload)
    os.chmod(path, 0o600)
    return payload
