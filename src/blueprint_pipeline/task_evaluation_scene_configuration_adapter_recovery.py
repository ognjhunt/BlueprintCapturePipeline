"""Preserve post-create identity when a scene-configuration adapter's finalization escapes.

Moved unchanged from ``task_evaluation_scene_configuration_vast`` (2026-11-15) so that module stays
inside its line budget; it imports the function back under its old private name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import write_json


def recover_escaped_adapter_failure(
    *,
    provider_run: Path,
    started_instance_id_path: Path,
    failure_detail: str,
) -> tuple[dict[str, Any], bool]:
    """Preserve post-create identity when adapter finalization escapes.

    The adapter writes the independent watchdog handoff immediately after Vast
    returns an instance id.  Its own ``finally`` normally destroys that
    instance and seals the adapter and teardown receipts, but an error while
    sealing those receipts can escape to this caller.  In that case the
    handoff is the only durable proof that allocation occurred; calling the
    unallocated sealer would contradict it.
    """

    started_path_present = (
        started_instance_id_path.exists() or started_instance_id_path.is_symlink()
    )
    instance_ids: list[int] = []
    blockers = [f"vast_adapter_failed:{failure_detail}"]
    if started_path_present:
        try:
            if started_instance_id_path.is_symlink():
                raise ValueError("started instance id path is a symlink")
            candidate = started_instance_id_path.read_text(encoding="utf-8").strip()
            instance_id = int(candidate)
            if instance_id <= 0 or candidate != str(instance_id):
                raise ValueError("started instance id is not canonical")
            instance_ids.append(instance_id)
        except (OSError, ValueError):
            blockers.append("vast_started_instance_id_evidence_invalid")

    adapter = {
        "schema_version": "vast_provider_adapter_result.v1",
        "status": "blocked",
        "reason": "vast_adapter_finalization_failed",
        "blockers": blockers,
        "provider_create_attempted": started_path_present,
        "vast_side_effects_may_have_occurred": started_path_present,
        "vast_instance_ids": instance_ids,
        "retained_owned": False,
        "continuing_spend_from_this_run": started_path_present,
        "raw_secret_values_recorded": False,
    }
    if started_path_present:
        try:
            write_json(
                provider_run / "vast_provider_adapter_result.json",
                adapter,
            )
        except OSError:
            adapter["blockers"].append(
                "vast_adapter_failure_receipt_write_failed"
            )
    return adapter, started_path_present


__all__ = ["recover_escaped_adapter_failure"]
