"""Existing recorded output-admission checks and read-only disk projection."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .control_plane_disk_budget import disk_headroom
from .task_evaluation_scene_configuration_provider_contracts import (
    PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES,
)

MEASURED_MODE = "measured"


OUTPUT_ROLE = "scene_configuration_output"


ADMISSION_SCHEMA_VERSION = "scene_configuration_provider_output_disk_admission.v1"


BUDGET_EXCEEDED_BLOCKER = "scene_configuration_provider_output_disk_budget_exceeded"


PREALLOCATION_PHASE = "before_allocation_and_staging"


DEFERRED_HOLD_PHASE = "after_cpu_prefix"


API_PRETRAINING_CONSUMED = "api_pretraining_consumed"


PREFIX_SPEND_UNPROVEN = "prefix_spend_unproven"


PREFIX_SPEND_NONE = "no_external_spend_recorded"


def output_volume_requirement(hold_bytes: int, phases: list[Mapping[str, Any]]) -> int:
    """Bytes the output's volume must have available above its floor.

    ``phases`` are the pre-GPU CPU phases in the order they run. Each leaves
    archives in the job directory after releasing its own reservation, so a
    phase that shares the volume needs its peak beside what earlier phases
    left, and the output hold then needs to fit beside everything they left.
    """

    required, residue = hold_bytes, 0
    for phase in phases:
        if phase["shares_output_volume"]:
            required = max(required, int(phase["peak_bytes"]) + residue)
        residue += int(phase["residue_bytes"])
    return max(required, hold_bytes + residue)


def _output_projection(
    target: Path, *, reservation_root: str | Path, disk_usage: Callable[[Path], Any]
) -> dict[str, int]:
    """Free, floor, live reservations and available bytes as this role's admission sees them."""

    headroom = disk_headroom(
        target_root=target, reservation_root=reservation_root, disk_usage=disk_usage
    )
    row = next(row for row in headroom["targets"] if row["role"] == OUTPUT_ROLE)
    return {
        key: int(row[key])
        for key in ("free_bytes", "floor_bytes", "reserved_bytes", "available_bytes")
    }


def recovery_withheld_reason(result: Mapping[str, Any]) -> str | None:
    """Why capacity recovery must not retry this measured refusal, or None.

    It keeps its typed blocker, but a retry would repeat paid external work:
    API pretraining that already ran, as for a credit refusal, or a CPU prefix
    unless the sealed record carries its proof of zero external spend. An
    older or incomplete record counts as spent.
    """

    record = measured_admission_record(result)
    if record is None:
        return None
    if record.get("recovery_withheld"):
        return str(record["recovery_withheld"])
    if record.get("hold_phase") != DEFERRED_HOLD_PHASE:
        return None
    if result.get("api_pretraining") is not None:
        return API_PRETRAINING_CONSUMED
    proof = record.get("prefix_spend")
    if result.get("cpu_prestage") is not None and not (
        isinstance(proof, Mapping) and proof.get("status") == PREFIX_SPEND_NONE
    ):
        return PREFIX_SPEND_UNPROVEN
    return None


def recovery_withheld(result: Mapping[str, Any]) -> bool:
    return recovery_withheld_reason(result) is not None


def recorded_preallocation_refusal(
    result: Mapping[str, Any], *, maximum_archive_bytes: int, job_dir: Path
) -> dict[str, Any] | None:
    """The measured refusal this job sealed before allocation, or None if it is not one.

    Refused before staging, or after a CPU prefix sharing the volume when no
    paid API pretraining ran; the ledger's own numbers must show the refusal.
    """

    record = result.get("provider_output_disk_capacity")
    if not isinstance(record, Mapping):
        return None
    required = recorded_output_requirement(record, maximum_archive_bytes=maximum_archive_bytes)
    available = record.get("available_bytes")
    valid = bool(
        required is not None
        and result.get("blockers") == [BUDGET_EXCEEDED_BLOCKER]
        and result.get("expected_provider_upload_bytes") == maximum_archive_bytes
        and record.get("schema_version") == ADMISSION_SCHEMA_VERSION
        and record.get("mode") == MEASURED_MODE
        and record.get("role") == OUTPUT_ROLE
        and record.get("phase") == PREALLOCATION_PHASE
        and record.get("hold_phase") in (PREALLOCATION_PHASE, DEFERRED_HOLD_PHASE)
        and not recovery_withheld(result)
        and record.get("status") == "blocked"
        and record.get("blockers") == [BUDGET_EXCEEDED_BLOCKER]
        and record.get("measurement_path") == str(job_dir)
        and record.get("maximum_archive_bytes") == maximum_archive_bytes
        and type(available) is int
        and 0 <= available < required
    )
    return dict(record) if valid else None


def recorded_output_requirement(
    record: Mapping[str, Any], *, maximum_archive_bytes: int
) -> int | None:
    """The requirement a measured record states, when its own hold and phases give it.

    Capacity recovery re-checks with the admission formula re-derived from the
    record, never with a number it cannot re-derive.
    """

    hold = maximum_archive_bytes + PROVIDER_OUTPUT_OPERATIONAL_RESERVE_BYTES
    phases = record.get("sequential_phases")
    if (
        record.get("hold_bytes") != hold
        or not isinstance(phases, list)
        or not all(
            isinstance(row, Mapping)
            and all(type(row.get(key)) is int and row[key] >= 0
                    for key in ("peak_bytes", "residue_bytes"))
            and type(row.get("shares_output_volume")) is bool
            for row in phases
        )
    ):
        return None
    required = output_volume_requirement(hold, phases)
    return required if record.get("required_available_bytes") == required else None


def measured_admission_record(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The measured pre-allocation record a sealed lane result carries, or None.

    A refusal carries it as ``provider_output_disk_capacity``; a run that got
    past admission nests it under ``before_allocation_and_staging``. None
    means the run was admitted by the ceiling formula.
    """

    capacity = result.get("provider_output_disk_capacity")
    if not isinstance(capacity, Mapping):
        return None
    record = (
        capacity if "schema_version" in capacity
        else capacity.get("before_allocation_and_staging")
    )
    if (
        isinstance(record, Mapping)
        and record.get("schema_version") == ADMISSION_SCHEMA_VERSION
        and record.get("mode") == MEASURED_MODE
    ):
        return record
    return None


def output_role_projection(
    *,
    output_path: Path,
    reservation_root: str | Path,
    required_available_bytes: int,
    disk_usage: Callable[[Path], Any],
) -> dict[str, Any]:
    """Re-measure the output volume exactly as this role's admission will: free
    bytes must cover the bulk floor, every live reservation and the need."""

    projection = _output_projection(
        output_path, reservation_root=reservation_root, disk_usage=disk_usage
    )
    return {
        "role": OUTPUT_ROLE,
        **projection,
        "required_available_bytes": required_available_bytes,
        "required_free_bytes": (
            projection["floor_bytes"] + projection["reserved_bytes"] + required_available_bytes
        ),
    }
