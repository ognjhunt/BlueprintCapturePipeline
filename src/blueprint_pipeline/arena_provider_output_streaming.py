"""Stream a Quick-10's provider output instead of downloading it (plan 15, 15.C3).

The arena lane's ``stream`` path, which only the Quick-10 session takes
(``BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY=stream``). Download mode never
reaches this module.

1. Before the session authority is consumed, ``reserve_forecast_hold`` takes
   a ``policy_canary_output`` hold of the role's declared footprint (review
   I3), so a host without room refuses before any spend. The session releases
   it with the lane's outcome.
2. Inside the paid window the adapter observes the staged object by range
   (``collector``): no ZIP and no MP4 copy on the host.
3. After teardown the lane promotes the staged object to B2 with a full
   readback, indexes it, and runs the gated cleanup
   (``provider_output_promotion.promote_then_cleanup``). An SSH-recovered ZIP
   is published and indexed the same way, then removed behind its pointer.
4. ``ingest_needed_members`` selects the member contract's needed set from the
   sealed index. A needed set over the contract's budget, or a hold that would
   have to grow, blocks with the archive durable; otherwise the hold shrinks in
   place to the need and the members are fetched from B2 by range into
   ``immutable_execution/`` (0440), each checked against the index. Only once
   ingestion is materialized is the view descriptor written and the native
   result path exposed (review I8).
5. ``stream_manifest_roles`` and ``stream_result_fields`` add the streamed
   records to the artifact manifest and the lane result.

A failure after the paid run seals ``blocked`` like a failed download, but
names whether the archive is durable (``archive_durable``); the door's
``provider-output-resume`` promotes or ingests later. When the provider's
output did arrive -- observed in the paid window, promoted, or recovered over
SSH -- such a run is not "before first observation": ``not_ingested_gap``
records a ``provider_output_not_ingested`` media gap carrying the stream
blocker and claims nothing about the policy (review important 2). ``resume_ingestion`` is
that ingestion, and it short-circuits on a materialized receipt (review I8):
readers such as partial recovery and interpretation write into the evidence
root afterwards, which a second ingestion pass would refuse.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .control_plane_disk_budget import (
    DEFAULT_RESERVATION_ROOT,
    ControlPlaneDiskBudgetError,
    DiskReservation,
    reserve_control_plane_disk,
)
from .control_plane_disk_ledger import footprint_bytes
from .control_plane_disk_usage import tree_usage
from .policy_canary_output_members import NOT_INGESTED_GAP, POLICY_CANARY_OUTPUT_CONTRACT, PolicyCanaryOutputContract

OUTPUT_ROLE = "policy_canary_output"
WORKLOAD = "quick10_needed_members"
RESERVATION_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"
EVIDENCE_DIRNAME = "immutable_execution"
INGESTION_DIRNAME = ".provider_output_ingestion"
INGESTION_RECEIPT_NAME = "receipt.json"
STAGING_DIRNAME = "object_store_staging"
DESCRIPTOR_NAME = EVIDENCE_DIRNAME + ".member_view.v1.json"
# The Quick-10 session's output upload bound (native_task_arena_vast: 8 GB plus
# the paired witness's own capacity); the output alone never exceeds it.
OUTPUT_ARCHIVE_MAXIMUM_BYTES = 8_000_000_000
PRESIGN_EXPIRATION_SECONDS = 3600
# Replaced by tests (Linux CI runs with fake disk usage).
disk_usage_provider: Callable[[Any], Any] = shutil.disk_usage


def reserve_forecast_hold(*, job_dir: str | Path, environ: Mapping[str, str] | None = None) -> DiskReservation:
    """The ``policy_canary_output`` forecast hold, taken before the session authority is consumed.

    It holds the role's declared footprint (1 GiB, or the operator's
    ``BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_POLICY_CANARY_OUTPUT_BYTES``) and
    is only ever shrunk afterwards. Raises ``ControlPlaneDiskBudgetError``.
    """
    values = os.environ if environ is None else environ
    return reserve_control_plane_disk(
        OUTPUT_ROLE, target_root=Path(job_dir), expected_bytes=footprint_bytes(OUTPUT_ROLE),
        reservation_root=values.get(RESERVATION_ROOT_ENV) or DEFAULT_RESERVATION_ROOT,
        disk_usage=lambda path: disk_usage_provider(path), workload=WORKLOAD)


def collector():
    """The adapter's range collector for a Quick-10 output (which expects no inspection videos)."""
    from .provider_output_remote_collection import RemoteProviderOutputCollector

    return RemoteProviderOutputCollector(maximum_archive_bytes=OUTPUT_ARCHIVE_MAXIMUM_BYTES, expected_video_count=0)


def promote(*, staging_dir: Path, attempt_root: Path, observation: Mapping[str, Any] | None,
            local_archive: Path, cleanup: Callable[[], Mapping[str, Any]]) -> tuple[dict, dict]:
    """Promote the staged (or SSH-recovered) output, then run the gated cleanup; never raises."""
    from .provider_output_promotion import promote_then_cleanup

    local = local_archive if local_archive.is_file() and not local_archive.is_symlink() else None
    return promote_then_cleanup(cleanup=cleanup, staging_dir=staging_dir, attempt_root=attempt_root,
                                observation=observation, local_archive=local,
                                maximum_archive_bytes=OUTPUT_ARCHIVE_MAXIMUM_BYTES)


def _sealed_index(attempt_root: Path, receipt: Mapping[str, Any]) -> dict | None:
    """The member index the promotion receipt names, re-checked against its file record."""
    import hashlib

    from .provider_output_member_index import ProviderOutputMemberIndexError, validate_member_index
    from .provider_output_promotion import INDEX_FILENAME

    record = receipt.get("member_index")
    path = attempt_root / INDEX_FILENAME
    if not isinstance(record, Mapping) or path.is_symlink() or not path.is_file():
        return None
    data = path.read_bytes()
    try:
        index = json.loads(data)
        validate_member_index(index)
    except (UnicodeError, ValueError, ProviderOutputMemberIndexError):
        return None
    if ("sha256:" + hashlib.sha256(data).hexdigest() != record.get("sha256")
            or index["index_digest"] != record.get("index_digest")
            or index["archive"]["durable_reference"] is None
            or index["archive"]["sha256"] != receipt.get("archive_sha256")):
        return None
    return index


def _ingestion_summary(receipt: Mapping[str, Any] | None, path: Path) -> dict | None:
    if not isinstance(receipt, Mapping):
        return None
    return {"status": receipt.get("status"), "receipt_path": str(path),
            "receipt_digest": receipt.get("receipt_digest"),
            **{key: receipt.get(key) for key in ("selection_version", "materialized_member_count",
                                                 "remote_member_count", "materialized_bytes", "remote_bytes",
                                                 "transferred_bytes", "http_request_count", "blockers")}}


def _materialized_receipt(attempt_root: Path, index: Mapping[str, Any]) -> dict | None:
    """An earlier pass's materialized ingestion receipt for this index, with its descriptor."""
    from .decision_evidence_contracts import canonical_digest
    from .provider_output_member_view import ProviderOutputMemberViewError, open_member_view

    path = attempt_root / INGESTION_DIRNAME / INGESTION_RECEIPT_NAME
    try:
        receipt = json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else None
    except (OSError, UnicodeError, ValueError):
        return None
    if (not isinstance(receipt, dict) or receipt.get("status") != "materialized"
            or receipt.get("member_index_digest") != index["index_digest"]
            or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")):
        return None
    try:
        view = open_member_view(attempt_root / EVIDENCE_DIRNAME)
    except ProviderOutputMemberViewError:
        return None
    return receipt if view is not None and view.index["index_digest"] == index["index_digest"] else None


def ingest_needed_members(
    *,
    attempt_root: Path,
    promotion: Mapping[str, Any],
    contract: PolicyCanaryOutputContract = POLICY_CANARY_OUTPUT_CONTRACT,
    reservation: DiskReservation | None,
    blocker_prefix: str,
    result_name: str | None,
    read_json: Callable[[Path], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Materialize the contract's needed set from the durable archive; the lane's extraction shape.

    Returns ``{status, result_path, execution, blockers}`` as the lane's
    ``_extract_provider_output`` does, plus ``archive_durable``,
    ``needed_set``, ``ingestion`` and ``member_view_path``. ``result_path`` is
    set only once ingestion is materialized and its view descriptor written;
    without ``result_name`` (a resume) no result is read.
    """
    from .provider_output_member_view import ProviderOutputMemberViewError, write_member_view_descriptor
    from .provider_output_promotion import INDEX_FILENAME
    from .provider_output_range_ingestion import (
        CasArchiveSource,
        ProviderOutputIngestionError,
        ingest_selected_members,
    )
    from .task_evaluation_configured_scene_object_store import presign_configured_scene_artifact

    outcome: dict[str, Any] = {"status": "blocked", "result_path": None, "execution": {}, "blockers": [],
                               "archive_durable": promotion.get("status") == "promoted", "needed_set": None,
                               "ingestion": None, "member_view_path": None}

    def blocked(*codes: str) -> dict[str, Any]:
        outcome["blockers"] = sorted({code for code in codes if code})
        outcome["reason"] = codes[0]  # the stream blocker, before the codes it carries
        return outcome

    if promotion.get("status") == "absent_confirmed":
        return blocked(f"{blocker_prefix}_provider_output_zip_missing")
    if promotion.get("status") != "promoted":
        return blocked(f"{blocker_prefix}_provider_output_promotion_failed", *promotion.get("blockers") or [])
    index = _sealed_index(attempt_root, promotion) if promotion.get("member_index") else None
    if index is None:
        return blocked(f"{blocker_prefix}_provider_output_not_indexed", *promotion.get("blockers") or [])
    selection = contract.selection(index)
    index_path = attempt_root / INDEX_FILENAME
    needed = contract.needed_bytes(index)
    hold = contract.hold_bytes(needed_bytes=needed, member_count=len(index["members"]),
                               index_file_bytes=index_path.stat().st_size)
    outcome["needed_set"] = {"contract": contract.version, "member_count": len(selection["members"]),
                             "bytes": needed, "budget_bytes": contract.needed_set_budget_bytes,
                             "hold_bytes": hold,
                             "forecast_bytes": reservation.expected_bytes if reservation is not None else None}
    if needed > contract.needed_set_budget_bytes:
        return blocked(f"{blocker_prefix}_provider_output_needed_set_over_budget")
    receipt_path = attempt_root / INGESTION_DIRNAME / INGESTION_RECEIPT_NAME
    receipt = _materialized_receipt(attempt_root, index)
    if receipt is None:
        if reservation is None:
            return blocked(f"{blocker_prefix}_provider_output_disk_reservation_missing")
        try:
            # Shrink only: growth would be admitted against space the live dispatch hold reduced.
            reservation.resize(hold)
        except ControlPlaneDiskBudgetError as exc:
            return blocked(f"{blocker_prefix}_provider_output_disk_budget_exceeded_after_run", str(exc))

        def reserve(outstanding: int) -> None:
            if outstanding > reservation.expected_bytes:
                raise ControlPlaneDiskBudgetError("control_plane_disk_budget_resize_growth_refused")

        reference = index["archive"]["durable_reference"]
        source = CasArchiveSource(reference, presign=lambda: presign_configured_scene_artifact(
            reference=reference, expiration_seconds=PRESIGN_EXPIRATION_SECONDS))
        try:
            receipt = ingest_selected_members(
                source=source, index=index, selection=selection, members_root=attempt_root / EVIDENCE_DIRNAME,
                metadata_root=attempt_root / INGESTION_DIRNAME, reserve=reserve,
                disk_usage_provider=lambda path: disk_usage_provider(path))
        except ProviderOutputIngestionError as exc:
            return blocked(f"{blocker_prefix}_provider_output_ingestion_blocked", str(exc))
    outcome["ingestion"] = _ingestion_summary(receipt, receipt_path)
    if receipt.get("status") != "materialized":
        return blocked(f"{blocker_prefix}_provider_output_ingestion_blocked", *receipt.get("blockers") or [])
    if reservation is not None:
        reservation.observe(int(receipt["materialized_bytes"]))
    try:
        write_member_view_descriptor(evidence_root=attempt_root / EVIDENCE_DIRNAME, index_path=index_path,
                                     ingestion_receipt_path=receipt_path)
    except ProviderOutputMemberViewError as exc:
        return blocked(f"{blocker_prefix}_provider_output_member_view_unbound", str(exc))
    outcome["member_view_path"] = str(attempt_root / DESCRIPTOR_NAME)
    if result_name is None or read_json is None:
        outcome.update(status="completed", blockers=[])
        return outcome
    result_path = attempt_root / EVIDENCE_DIRNAME / result_name
    execution = read_json(result_path)
    outcome.update(result_path=str(result_path), execution=execution)
    if not execution:
        return blocked(f"{blocker_prefix}_runtime_result_missing")
    outcome.update(status="completed", blockers=[])
    return outcome


def not_ingested_gap(outcome: Mapping[str, Any], *, observation: Mapping[str, Any] | None,
                     promotion: Mapping[str, Any], local_archive: Path) -> dict[str, Any] | None:
    """The lane's policy evidence for an output that arrived but was never ingested, else None.

    With no execution receipt on the host the lane would seal "before first
    observation". Once the provider's output was observed in the paid window,
    promoted, or recovered over SSH, that is false; whether the candidate
    policy was queried is unknown until the door ingests the members, so no
    policy claim is made. An ingestion that materialized is download mode's
    case (the members are there, the result is not) and is left to it.
    """
    arrived = (observation is not None or promotion.get("status") == "promoted"
               or (local_archive.is_file() and not local_archive.is_symlink()))
    if not arrived or outcome.get("member_view_path"):
        return None
    return {"visual_evidence": {"status": NOT_INGESTED_GAP, "media_gap": {
        "type": NOT_INGESTED_GAP, "reason": outcome.get("reason") or next(iter(outcome.get("blockers") or []), None)}}}


def resume_ingestion(attempt_root: str | Path, *, contract: PolicyCanaryOutputContract = POLICY_CANARY_OUTPUT_CONTRACT,
                     environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Ingest a promoted, indexed attempt's needed set after the fact; never twice.

    The door's ``provider-output-resume --ingest``. An ingestion already
    materialized for this index, with its view descriptor, short-circuits
    before any hold, request or check of the evidence root (review I8): readers
    write there afterwards (partial recovery, interpretation receipts), which a
    second pass would refuse as a changed inventory. Otherwise the exact hold
    is admitted now -- the paid run and its dispatch hold are long over -- and
    released with the outcome. The sealed lane result is never rewritten.
    """
    from .provider_output_promotion_records import load_promotion_receipt, staging_manifest_sha256

    attempt = Path(attempt_root).expanduser().resolve()
    staging = attempt / STAGING_DIRNAME
    receipt = load_promotion_receipt(staging, staging_manifest_sha256=staging_manifest_sha256(staging))
    result: dict[str, Any] = {"status": "blocked", "short_circuited": False, "ingestion": None, "needed_set": None,
                              "blockers": []}
    if receipt is None or receipt.get("status") != "promoted":
        return {**result, "blockers": ["provider_output_resume_ingestion_output_not_promoted"]}
    index = _sealed_index(attempt, receipt)
    if index is None:
        return {**result, "blockers": ["provider_output_resume_ingestion_output_not_indexed"]}
    materialized = _materialized_receipt(attempt, index)
    receipt_path = attempt / INGESTION_DIRNAME / INGESTION_RECEIPT_NAME
    if materialized is not None:
        return {**result, "status": "materialized", "short_circuited": True,
                "ingestion": _ingestion_summary(materialized, receipt_path)}
    values = os.environ if environ is None else environ
    from .provider_output_promotion import INDEX_FILENAME

    need = contract.hold_bytes(needed_bytes=contract.needed_bytes(index), member_count=len(index["members"]),
                               index_file_bytes=(attempt / INDEX_FILENAME).stat().st_size)
    try:
        hold = reserve_control_plane_disk(
            OUTPUT_ROLE, target_root=attempt, expected_bytes=need,
            reservation_root=values.get(RESERVATION_ROOT_ENV) or DEFAULT_RESERVATION_ROOT,
            disk_usage=lambda path: disk_usage_provider(path), workload=WORKLOAD)
    except ControlPlaneDiskBudgetError as exc:
        return {**result, "blockers": ["provider_output_resume_ingestion_disk_admission_refused", str(exc)]}
    outcome_label = "failed"
    try:
        outcome = ingest_needed_members(
            attempt_root=attempt, promotion=receipt, contract=contract, reservation=hold,
            blocker_prefix="provider_output_resume", result_name=None, read_json=None)
        done = bool(outcome.get("member_view_path"))
        outcome_label = "completed" if done else "blocked"
    finally:
        hold.release(outcome=outcome_label)
    return {**result, "status": "materialized" if done else "blocked", "ingestion": outcome.get("ingestion"),
            "needed_set": outcome.get("needed_set"), "blockers": [] if done else outcome["blockers"]}


def stream_manifest_roles(attempt_root: Path, outcome: Mapping[str, Any]) -> tuple[dict, list, dict | None]:
    """Artifact-manifest roles, required roles and archive members for a streamed attempt."""
    from .provider_output_promotion import INDEX_FILENAME
    from .provider_output_promotion_records import RECEIPT_FILENAME

    roles: dict[str, Path] = {}
    receipt = attempt_root / STAGING_DIRNAME / RECEIPT_FILENAME
    if receipt.is_file():
        roles["provider_output_promotion"] = receipt
    archive_members = None
    if outcome.get("member_view_path"):
        roles.update({
            "provider_output_member_index": attempt_root / INDEX_FILENAME,
            "provider_output_ingestion_receipt": attempt_root / INGESTION_DIRNAME / INGESTION_RECEIPT_NAME,
            "provider_output_member_view": attempt_root / DESCRIPTOR_NAME,
        })
        index = json.loads((attempt_root / INDEX_FILENAME).read_text(encoding="utf-8"))
        archive_members = {"provider_runtime_evidence": (index, EVIDENCE_DIRNAME)}
    return roles, sorted(roles), archive_members


def stream_result_fields(*, attempt_root: Path, promotion: Mapping[str, Any], outcome: Mapping[str, Any],
                         reservation: DiskReservation | None,
                         contract: PolicyCanaryOutputContract = POLICY_CANARY_OUTPUT_CONTRACT) -> dict[str, Any]:
    """The lane-result fields only a streamed attempt carries (measured when the lane seals)."""
    from .provider_output_promotion_records import RECEIPT_FILENAME

    usage, evidence = tree_usage(attempt_root), tree_usage(attempt_root / EVIDENCE_DIRNAME)
    witness = promotion.get("witness") if isinstance(promotion.get("witness"), Mapping) else {}
    receipt = attempt_root / STAGING_DIRNAME / RECEIPT_FILENAME
    return {
        "provider_output_delivery": "stream",
        "provider_output_member_contract": contract.version,
        "archive_durable": outcome.get("archive_durable") is True,
        "provider_output_promotion": {
            "receipt_path": str(receipt) if receipt.is_file() else None,
            **{key: promotion.get(key) for key in ("status", "source", "receipt_digest", "archive_sha256",
                                                   "size_bytes", "durable_reference", "member_index", "blockers")},
            "witness_disposition": witness.get("disposition"),
        },
        "provider_output_needed_set": outcome.get("needed_set"),
        "provider_output_ingestion": outcome.get("ingestion"),
        "provider_output_member_view_path": outcome.get("member_view_path"),
        "provider_output_disk_reservation": {
            "role": OUTPUT_ROLE, "workload": WORKLOAD,
            "held_bytes": reservation.expected_bytes if reservation is not None else None,
        },
        # M1 (plan 15, acceptance): the attempt's host bytes, by the ledger's own walker.
        "provider_output_host_bytes": {
            "attempt_root_allocated_bytes": usage.allocated_bytes,
            "attempt_root_apparent_bytes": usage.apparent_bytes,
            "evidence_root_allocated_bytes": evidence.allocated_bytes,
            "measured_by": "control_plane_disk_usage.tree_usage",
        },
    }


__all__ = [
    "NOT_INGESTED_GAP",
    "OUTPUT_ARCHIVE_MAXIMUM_BYTES",
    "OUTPUT_ROLE",
    "WORKLOAD",
    "collector",
    "ingest_needed_members",
    "not_ingested_gap",
    "promote",
    "reserve_forecast_hold",
    "resume_ingestion",
    "stream_manifest_roles",
    "stream_result_fields",
]
