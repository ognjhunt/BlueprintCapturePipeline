"""Bring one stored job's offloaded launch run back to the host, so the job can be tested again.

Plan 1.3 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``). A finished job's configured
scene launch run goes to cold storage two days after it is sealed
(``control_plane_evidence_offload``): a digest-bound pointer stands where the directory was. A new
screening campaign re-validates that run from disk, so a cold job cannot be tested. This is the door's
operation (``restore-launch-run``) over the existing ``restore_offloaded_evidence``:

1. find the job's newest completed scene request (``beta_policy_screening_on_job.job_source``);
2. nothing to do when the run is already on disk;
3. reserve the restore's peak footprint (the archive plus its extraction) with the existing disk ledger
   (``reserve_control_plane_disk``), refusing by name when there is no room;
4. restore, verifying every member digest, and record a receipt.

It starts no compute and spends nothing on a provider; the object-store read is the only transfer (B2
egress is not priced in this repository). The pointer stays in place, so the offloader never archives the
run a second time.
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
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def restore_launch_run(*, site_id: str, request_id: str, packet_queue_root: str | Path,
                       queue_root: str | Path | None = None, reservation_root: str | Path | None = None,
                       materializer: Any = None, reserve: Any = None) -> dict[str, Any]:
    """Restore the job's launch run, or refuse by name; returns the receipt it records."""
    from .beta_policy_screening_on_job import JobTestRefused, job_source
    from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT, reserve_control_plane_disk
    from .control_plane_disk_ledger import ControlPlaneDiskBudgetError
    from .control_plane_evidence_offload import POINTER_SUFFIX, restore_offloaded_evidence

    if not _REQUEST_ID.fullmatch(str(request_id)):
        raise ValueError("launch_run_restore_request_invalid")
    outcome: dict[str, Any]
    try:
        request, _capture = job_source(Path(packet_queue_root), site_id)
        launch = Path(request["source_launch_root"])
        pointer_path = launch.parent / f"{launch.name}{POINTER_SUFFIX}"
        outcome = {"site_id": site_id, "launch_run": str(launch)}
        if launch.is_dir() and not launch.is_symlink():
            outcome["status"] = "already_on_disk"
        elif not pointer_path.is_file() or pointer_path.is_symlink():
            outcome.update(status="refused", reason=POINTER_MISSING)
        else:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            members = pointer.get("members") if isinstance(pointer, dict) else None
            if not isinstance(members, list) or type(pointer.get("size_bytes")) is not int:
                raise ValueError(POINTER_INVALID)
            # Peak: the downloaded archive and its extracted tree side by side.
            need = pointer["size_bytes"] + sum(int(row.get("size_bytes", 0)) for row in members)
            outcome["reserved_bytes"] = need
            try:
                reservation = (reserve or reserve_control_plane_disk)(
                    DISK_ROLE, target_root=launch.parent, expected_bytes=max(need, 1),
                    reservation_root=reservation_root or DEFAULT_RESERVATION_ROOT)
            except ControlPlaneDiskBudgetError as exc:
                outcome.update(status="refused", reason=NO_ROOM, disk_refusal=str(exc).split(" ")[0][:200])
            else:
                try:
                    kwargs = {"materializer": materializer} if materializer is not None else {}
                    restored = restore_offloaded_evidence(pointer_path=pointer_path, destination=launch, **kwargs)
                finally:
                    reservation.release()
                outcome.update(status="restored", member_count=restored["member_count"], digest=restored["digest"])
    except JobTestRefused as exc:
        outcome = {"site_id": site_id, "status": "refused", "reason": exc.code, **exc.details}
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
