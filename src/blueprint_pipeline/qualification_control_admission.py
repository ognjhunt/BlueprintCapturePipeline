"""Fresh paid-admission gate for live qualification-session controls."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


MUTATING_CONTROL_ACTIONS = frozenset(
    {"install-checkpoint", "refresh", "restart", "run", "stop"}
)


def admit_qualification_control_mutation(
    admission_out: str | Path | None,
    manifest: Mapping[str, Any],
    inspected: Mapping[str, Any],
    instance_id: str,
    action: str,
    component: str,
    *,
    clock: Callable[[], float] = time.time,
) -> PaidResourceAdmissionGrant:
    """Write and require a fresh admission immediately before SSH mutation."""

    if admission_out is None or admission_out == "":
        raise ValueError("qualification_control_admission_out_missing")
    blockers: list[str] = []
    now = float(clock())
    if action not in MUTATING_CONTROL_ACTIONS:
        blockers.append("qualification_control_mutation_action_invalid")
    if not instance_id:
        blockers.append("qualification_control_instance_id_missing")
    if manifest.get("continuing_spend") is not True:
        blockers.append("qualification_control_session_not_live")
    if now >= float(manifest.get("watchdog_deadline_epoch") or 0):
        blockers.append("qualification_control_session_ttl_expired")
    if (
        inspected.get("status") != "observed"
        or str(inspected.get("instance_id") or "") != instance_id
        or inspected.get("name") != manifest.get("resource_name")
    ):
        blockers.append("qualification_control_provider_binding_unverified")
    admission = build_paid_lane_admission(
        resource_class="gpu_render",
        blockers=blockers,
    )
    admission.update(
        {
            "scope": "persistent_single_g1_kitchen_qualification_control",
            "provider": "vast",
            "instance_id": instance_id,
            "launch_session_id": str(manifest.get("launch_session_id") or ""),
            "action": action,
            "component": component,
            "generated_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "fresh_control_admission": True,
        }
    )
    write_json(Path(admission_out).expanduser().resolve(), admission)
    return require_paid_resource_admission(
        admission,
        resource_class="gpu_render",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
