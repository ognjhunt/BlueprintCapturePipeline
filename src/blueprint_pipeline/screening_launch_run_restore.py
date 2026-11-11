"""Bring one stored job's offloaded launch run back to the host, so the job can be tested again.

Plan 1.3 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``). A finished job's configured
scene launch run goes to cold storage two days after it is sealed
(``control_plane_evidence_offload``): a digest-bound pointer stands where the directory was. A new
screening campaign re-validates that run from disk, so a cold job cannot be tested. This is the door's
operation (``restore-launch-run``) over the existing ``restore_offloaded_evidence``:

1. find the job's newest completed scene request (``beta_policy_screening_on_job.job_source``);
2. refuse, before any byte is fetched, a site whose consent is withdrawn (or cannot be read) or whose
   current source the WebApp no longer allows (``beta_policy_screening_on_job.current_consent``, the same
   privacy stop a job test passes);
3. nothing to do when the run is already on disk;
4. reserve the restore's peak footprint (the archive plus its extraction) with the existing disk ledger
   (``reserve_control_plane_disk``), refusing by name when there is no room;
5. restore, verifying every member digest, and record a receipt. Every failure is a typed reason with
   its receipt: a bad pointer, a download error, a digest or member mismatch. The restore stages in a
   private directory beside the run and moves the verified tree into place in one step, so a failure
   leaves nothing half written.

It starts no compute and spends nothing on a provider; the object-store read is the only transfer (B2
egress is not priced in this repository).

A restored run is an ordinary live tree again. The disk ledger counts free bytes minus held
reservations, so the restored bytes count from the moment they land, and the reservation (held only for
the restore's peak) is released after. The run's directory time is set to the restore's, so the
offloader's hot window starts again from now; the pointer stays beside it (it is the only record of the
archive, which withdrawal deletion also reads), and once that window has passed the offloader removes
the restored copy again, without uploading it a second time, only if it still matches the archive byte
for byte and a metadata read of the store confirms the sealed archive is still there
(``control_plane_evidence_offload``).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from .decision_evidence_contracts import canonical_digest

RECEIPT_SCHEMA = "blueprint.screening_launch_run_restore_receipt.v1"
RECEIPT_DIRNAME = "launch_run_restores"
DISK_ROLE = "evidence_offload"
NO_ROOM = "launch_run_restore_disk_headroom_insufficient"
POINTER_MISSING = "launch_run_offload_pointer_missing"
POINTER_INVALID = "launch_run_offload_pointer_invalid"
DOWNLOAD_FAILED = "launch_run_restore_download_failed"
#: A restore refusal whose own message is not a typed code (the offload module's codes are kept as they are).
RESTORE_REFUSED = "launch_run_restore_refused"
_TYPED = re.compile(r"[a-z][a-z0-9_]{0,127}")
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def restore_launch_run(*, site_id: str, request_id: str, packet_queue_root: str | Path,
                       queue_root: str | Path | None = None, reservation_root: str | Path | None = None,
                       materializer: Any = None, reserve: Any = None) -> dict[str, Any]:
    """Restore the job's launch run, or refuse by name; returns the receipt it records."""
    from .beta_policy_screening_on_job import JobTestRefused, current_consent, job_source
    from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT, reserve_control_plane_disk
    from .control_plane_disk_ledger import ControlPlaneDiskBudgetError
    from .control_plane_evidence_offload import (
        POINTER_SUFFIX, ControlPlaneEvidenceOffloadError, restore_offloaded_evidence,
    )

    if not _REQUEST_ID.fullmatch(str(request_id)):
        raise ValueError("launch_run_restore_request_invalid")
    outcome: dict[str, Any] = {"site_id": site_id}
    try:
        request, capture = job_source(Path(packet_queue_root), site_id)
        launch = Path(request["source_launch_root"])
        pointer_path = launch.parent / f"{launch.name}{POINTER_SUFFIX}"
        outcome["launch_run"] = str(launch)
        # The privacy stop comes first: a withdrawn site's run is never brought back.
        current_consent(site_id, request, capture)
        if launch.is_dir() and not launch.is_symlink():
            outcome["status"] = "already_on_disk"
        elif not pointer_path.is_file() or pointer_path.is_symlink():
            outcome.update(status="refused", reason=POINTER_MISSING)
        else:
            outcome.update(_restore(launch, pointer_path, reserve=reserve or reserve_control_plane_disk,
                                    reservation_root=reservation_root or DEFAULT_RESERVATION_ROOT,
                                    materializer=materializer, restore=restore_offloaded_evidence,
                                    no_room=ControlPlaneDiskBudgetError, typed=ControlPlaneEvidenceOffloadError))
    except JobTestRefused as exc:
        outcome.update(status="refused", reason=exc.code, **exc.details)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        # A refusal the WebApp read or the job lookup raised by its own typed code; anything else by kind.
        code = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else ""
        outcome.update(status="refused", reason=code if _TYPED.fullmatch(code) else "launch_run_restore_failed",
                       **({} if _TYPED.fullmatch(code) else {"error_type": type(exc).__name__}))
    receipt = {"schema_version": RECEIPT_SCHEMA, "request_id": request_id,
               "recorded_at": datetime.now(timezone.utc).isoformat(), **outcome,
               "provider_allocations_performed": 0, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if queue_root is not None:
        from .beta_policy_screening_coordinator import _root, _write

        root = _root(queue_root)
        directory = root / RECEIPT_DIRNAME
        directory.mkdir(exist_ok=True, mode=0o700)
        _write(directory / f"{request_id}.json", receipt)
        owner = root.stat()
        if os.geteuid() == 0:
            for path in (directory, directory / f"{request_id}.json"):
                os.chown(path, owner.st_uid, owner.st_gid)
    return receipt


def _restore(launch: Path, pointer_path: Path, *, reserve: Any, reservation_root: Any, materializer: Any,
             restore: Any, no_room: type[Exception], typed: type[Exception]) -> dict[str, Any]:
    """Reserve the peak, restore, release; every failure as a typed outcome (nothing half written)."""
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        members = pointer.get("members") if isinstance(pointer, dict) else None
        if (not isinstance(members, list) or type(pointer.get("size_bytes")) is not int
                or any(not isinstance(row, dict) or type(row.get("size_bytes")) is not int for row in members)):
            raise ValueError(POINTER_INVALID)
    except (OSError, ValueError):
        return {"status": "refused", "reason": POINTER_INVALID}
    # Peak: the downloaded archive and its extracted tree side by side.
    need = pointer["size_bytes"] + sum(row["size_bytes"] for row in members)
    outcome: dict[str, Any] = {"reserved_bytes": need}
    try:
        reservation = reserve(DISK_ROLE, target_root=launch.parent, expected_bytes=max(need, 1),
                              reservation_root=reservation_root)
    except no_room as exc:
        return {**outcome, "status": "refused", "reason": NO_ROOM, "disk_refusal": str(exc).split(" ")[0][:200]}
    try:
        kwargs = {"materializer": materializer} if materializer is not None else {}
        restored = restore(pointer_path=pointer_path, destination=launch, **kwargs)
    except typed as exc:  # the offload module's own codes: pointer, digest, member or destination
        code = str(exc)
        return {**outcome, "status": "refused", "reason": code if _TYPED.fullmatch(code) else RESTORE_REFUSED}
    except Exception as exc:  # noqa: BLE001 - the object-store read: one typed reason, never its message
        return {**outcome, "status": "refused", "reason": DOWNLOAD_FAILED, "error_type": type(exc).__name__[:80]}
    finally:
        reservation.release()
    outcome.update(status="restored", member_count=restored["member_count"], digest=restored["digest"])
    try:
        os.utime(launch)  # the hot window starts again now: the offloader sees a tree last changed at its restore
    except OSError:
        outcome["hot_window_restarted"] = False  # flagged; the tree is restored and the window runs from its files
    return outcome


def main(argv: list[str] | None = None) -> int:
    from .beta_policy_screening_coordinator import QUEUE_ROOT_ENV
    from .team_scene_packet_worker import QUEUE_ROOT_ENV as PACKET_QUEUE_ENV

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--door-request-id", required=True)
    parser.add_argument("--packet-queue-root", default=os.environ.get(PACKET_QUEUE_ENV))
    parser.add_argument("--queue-root", default=os.environ.get(QUEUE_ROOT_ENV))
    parser.add_argument("--result-out", required=True)
    args = parser.parse_args(argv)
    if not args.packet_queue_root:
        receipt = {"status": "refused", "reason": "launch_run_restore_packet_queue_not_configured"}
    else:
        receipt = restore_launch_run(site_id=args.site_id, request_id=args.door_request_id,
                                     packet_queue_root=args.packet_queue_root, queue_root=args.queue_root)
    Path(args.result_out).write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "reason": receipt.get("reason")}, sort_keys=True))
    return 0 if receipt["status"] in {"restored", "already_on_disk"} else 3


if __name__ == "__main__":
    sys.exit(main())
