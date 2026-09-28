"""Terminalize one interrupted launch-preparation claim whose worker is gone.

The launch-preparation worker claims a queued envelope by moving it from
``pending/`` to ``processing/`` and moves it to a terminal state only at the end
of the same process (``process_launch_preparation_queue``). A worker killed
mid-claim -- a deploy restart, SIGTERM, OOM -- leaves the envelope in
``processing/`` forever: later workers only scan ``pending/`` and the resumable
waiting state, and the scene progression refuses to mint a release successor
until the previous attempt's preparation is terminal
(``previous_release_attempt_not_terminal``). Nothing in the tree reclaimed such
a claim before 2026-09-12 (scene 840938: the controller tick that held the
claim was SIGTERMed by the post-#1875 restart, and every later tick reported
``processed_count: 0`` against a queue whose only work sat in ``processing/``).

This operation records a truthful ``blocked`` result for exactly one such claim
and moves the envelope, byte for byte, to ``blocked/``. It proves ownership
before it mutates anything, and refuses otherwise:

* no live worker process started at or before the claim was taken -- a worker
  started later cannot own a ``processing/`` entry it never moved;
* the previous owner pid, when named, is gone;
* no non-terminal child job in the GPU child queue references the request;
* the envelope is digest-valid, present only in ``processing/``, and has no
  result yet.

It never re-executes the request, never touches child jobs, reservations,
budgets, or billing, and runs only as the service account so no root-owned
file enters the queue. The result's ``source_commit`` is the commit the claim
was made under (the worker release that owned it), which is what
``task_evaluation_scene_progression._queue`` binds a result to; the recovering
release is recorded separately under ``claim_recovery``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_contract import launch_preparation_request_digest
from .task_evaluation_launch_preparation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    QUEUE_STATES,
    write_launch_preparation_record_exclusive,
)

RECOVERY_SCHEMA_VERSION = "task_evaluation_preparation_claim_recovery.v1"
RECEIPT_SCHEMA_VERSION = "task_evaluation_preparation_claim_recovery_receipt.v1"
# Same document schema the worker seals; pinned against the worker constant in tests.
RESULT_SCHEMA_VERSION = "task_evaluation_launch_preparation_result.v1"
INTERRUPTED_CLAIM_BLOCKER = "launch_preparation_claim_interrupted:worker_process_terminated"
TERMINAL_STATE = "blocked"
# Every entrypoint that can hold a claim on an owned preparation queue.
WORKER_MODULES = (
    "blueprint_pipeline.task_evaluation_scene_progression",
    "blueprint_pipeline.task_evaluation_scene_preparation_service",
    "blueprint_pipeline.task_evaluation_launch_preparation_worker",
)
# Child (GPU) queue states in which a job may still be executing or about to.
CHILD_LIVE_STATES = ("pending", "processing", "started", "waiting_external", "wake-pending")
_ENVELOPE_NAME = re.compile(r"^(?P<preparation_id>[A-Za-z0-9][A-Za-z0-9._-]*)-(?P<digest>[0-9a-f]{64})\.json$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class PreparationClaimRecoveryError(ValueError):
    """Refusal; the message is the blocker code."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise PreparationClaimRecoveryError(code)


def live_worker_processes(proc_root: str | Path = "/proc") -> list[dict[str, Any]]:
    """Every live process running a worker entrypoint, with its start epoch.

    Reads the process table directly so the answer is the kernel's, not a
    remembered pid. Fails closed when the table cannot be read.
    """
    root = Path(proc_root)
    try:
        stat_text = (root / "stat").read_text(encoding="utf-8")
    except OSError as exc:
        raise PreparationClaimRecoveryError("process_table_unavailable") from exc
    boot_epoch = None
    for line in stat_text.splitlines():
        if line.startswith("btime "):
            boot_epoch = int(line.split()[1])
    _require(boot_epoch is not None, "process_table_boot_time_missing")
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    rows: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
            stat_line = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue  # exited between listing and read; it holds nothing now
        text = [argument.decode("utf-8", "replace") for argument in arguments]
        module = next((name for name in WORKER_MODULES if any(name in argument for argument in text)), None)
        if module is None:
            continue
        # Field 22 (starttime, clock ticks since boot) counted after the ')' that
        # closes comm, which may itself contain spaces.
        fields = stat_line.rsplit(")", 1)[1].split()
        rows.append({"pid": int(entry.name), "module": module,
                     "started_at_epoch": boot_epoch + int(fields[19]) / ticks_per_second})
    return rows


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_interrupted_preparation_claim(
    *,
    queue_root: str | Path,
    envelope_name: str,
    recovering_source_commit: str,
    child_queue_root: str | Path | None = None,
    service_account: str | None = None,
    previous_owner_pid: int | None = None,
    apply: bool = False,
    live_workers: Callable[[], list[dict[str, Any]]] | None = None,
    proc_root: str | Path = "/proc",
    now: float | None = None,
) -> dict[str, Any]:
    """Prove the claim is orphaned, then (with ``apply``) seal it ``blocked``."""
    moment = time.time() if now is None else float(now)
    queue = Path(queue_root)
    _require(queue.is_absolute() and queue.is_dir() and not queue.is_symlink(), "queue_root_invalid")
    match = _ENVELOPE_NAME.match(envelope_name)
    _require(match is not None, "envelope_name_invalid")
    _require(bool(_COMMIT.fullmatch(recovering_source_commit or "")), "recovering_source_commit_invalid")
    if service_account is not None:
        try:
            account = pwd.getpwnam(service_account)
        except KeyError as exc:
            raise PreparationClaimRecoveryError("service_account_missing") from exc
        _require(os.geteuid() == account.pw_uid, "service_account_identity_mismatch")
    if previous_owner_pid is not None:
        _require(type(previous_owner_pid) is int and previous_owner_pid > 1, "previous_owner_pid_invalid")

    claimed = queue / "processing" / envelope_name
    _require(claimed.is_file() and not claimed.is_symlink(), "claim_not_in_processing")
    elsewhere = [state for state in QUEUE_STATES
                 if state != "processing" and (queue / state / envelope_name).exists()]
    _require(not elsewhere, "claim_identity_ambiguous:" + ",".join(elsewhere))
    result_path = queue / "results" / envelope_name
    _require(not result_path.exists(), "claim_result_already_exists")

    raw = claimed.read_bytes()
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreparationClaimRecoveryError("claim_envelope_invalid") from exc
    _require(isinstance(envelope, dict) and envelope.get("schema_version") == ENVELOPE_SCHEMA_VERSION
             and envelope.get("envelope_digest") == canonical_digest(envelope, digest_field="envelope_digest"),
             "claim_envelope_invalid")
    request = envelope.get("request")
    _require(isinstance(request, dict)
             and envelope.get("request_digest") == launch_preparation_request_digest(request),
             "claim_request_digest_mismatch")
    assert match is not None and isinstance(request, dict)
    _require(request.get("preparation_id") == match["preparation_id"]
             and envelope["request_digest"] == "sha256:" + match["digest"], "claim_name_mismatch")
    claim_source_commit = str(request.get("expected_production_commit") or "")
    _require(bool(_COMMIT.fullmatch(claim_source_commit)), "claim_source_commit_invalid")
    claim_stat = claimed.stat()
    claimed_at = claim_stat.st_ctime  # the rename into processing/ is the claim

    # Ownership: a worker that started after the claim never moved this file.
    workers = list((live_workers or (lambda: live_worker_processes(proc_root)))())
    for row in workers:
        _require({"pid", "module", "started_at_epoch"} <= set(row), "live_worker_row_invalid")
    earlier = sorted(row["pid"] for row in workers if row["started_at_epoch"] <= claimed_at)
    _require(not earlier, "claim_owner_may_be_alive:" + ",".join(str(pid) for pid in earlier))
    if previous_owner_pid is not None:
        _require(not _pid_alive(previous_owner_pid), f"previous_owner_alive:{previous_owner_pid}")

    # Paid children: a live child job bound to this request means the claim is
    # not idle even though its parent process is gone.
    child_references: list[dict[str, str]] = []
    if child_queue_root is not None:
        child_root = Path(child_queue_root)
        _require(child_root.is_absolute() and child_root.is_dir(), "child_queue_root_missing")
        needle = envelope["request_digest"].removeprefix("sha256:")
        for state in CHILD_LIVE_STATES:
            for path in sorted((child_root / state).glob("*.json")):
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError as exc:
                    raise PreparationClaimRecoveryError(f"child_queue_unreadable:{state}") from exc
                if needle in text:
                    child_references.append({"state": state, "name": path.name})
        _require(not child_references, "claim_child_live:" + ";".join(
            f"{row['state']}/{row['name']}" for row in child_references))

    # Read-only: what the interrupted worker had recorded about the source chain.
    last_progress = None
    progress_dir = queue / "source-progress" / envelope_name[: -len(".json")]
    records = sorted(progress_dir.glob("*.json")) if progress_dir.is_dir() else []
    if records:
        try:
            value = json.loads(records[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PreparationClaimRecoveryError("claim_source_progress_unreadable") from exc
        last_progress = {"record": records[-1].name, "sequence": value.get("sequence"),
                         "status": (value.get("advancement") or {}).get("status")}

    recovery = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "envelope_name": envelope_name,
        "request_digest": envelope["request_digest"],
        "envelope_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "envelope_size_bytes": len(raw),
        "claimed_at_epoch": claimed_at,
        "recovered_at_epoch": moment,
        "claim_source_commit": claim_source_commit,
        "recovered_by_source_commit": recovering_source_commit,
        "previous_owner_pid": previous_owner_pid,
        "previous_owner_alive": False,
        "live_worker_processes": workers,
        "child_queue_root": None if child_queue_root is None else str(child_queue_root),
        "child_queue_live_references": child_references,
        "last_source_progress": last_progress,
        "request_re_executed": False,
        "envelope_bytes_modified": False,
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": TERMINAL_STATE,
        "preparation_id": request["preparation_id"],
        "blockers": [INTERRUPTED_CLAIM_BLOCKER],
        "source_commit": claim_source_commit,
        "claim_recovery": recovery,
        "provider_mutation_performed": False,
        "catalog_mutation_performed": False,
        "paid_execution_requested": False,
        "observed_at_iso": datetime.fromtimestamp(moment, timezone.utc).isoformat(),
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    terminal_path = queue / TERMINAL_STATE / envelope_name
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "recoverable",
        "applied": False,
        "queue_root": str(queue),
        "envelope_name": envelope_name,
        "result_path": str(result_path),
        "terminal_state": TERMINAL_STATE,
        "terminal_path": str(terminal_path),
        "result": result,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }
    if apply:
        current = claimed.stat()
        _require((current.st_ino, current.st_ctime_ns, current.st_size)
                 == (claim_stat.st_ino, claim_stat.st_ctime_ns, claim_stat.st_size),
                 "claim_changed_during_recovery")
        result_path.parent.mkdir(mode=0o750, exist_ok=True)
        try:
            write_launch_preparation_record_exclusive(result_path, result)
        except FileExistsError as exc:
            raise PreparationClaimRecoveryError("claim_result_already_exists") from exc
        try:
            os.replace(claimed, terminal_path)
        except FileNotFoundError as exc:
            raise PreparationClaimRecoveryError("claim_moved_during_recovery") from exc
        receipt.update(status="recovered", applied=True)
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--envelope-name", required=True)
    parser.add_argument("--recovering-source-commit", default="",
                        help="commit of the code performing the recovery; defaults to the running release")
    parser.add_argument("--child-queue-root")
    parser.add_argument("--service-account", default="blueprint")
    parser.add_argument("--previous-owner-pid", type=int)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--receipt-out")
    parser.add_argument("--apply", action="store_true", help="seal the result and move the envelope")
    args = parser.parse_args(argv)
    commit = args.recovering_source_commit
    if not commit:
        from .task_evaluation_release_identity import running_release_commit
        commit = running_release_commit()
    try:
        receipt = recover_interrupted_preparation_claim(
            queue_root=args.queue_root, envelope_name=args.envelope_name, recovering_source_commit=commit,
            child_queue_root=args.child_queue_root, service_account=args.service_account,
            previous_owner_pid=args.previous_owner_pid, apply=args.apply, proc_root=args.proc_root)
    except PreparationClaimRecoveryError as exc:
        print(json.dumps({"schema_version": RECEIPT_SCHEMA_VERSION, "status": "refused", "applied": False,
                          "blockers": [str(exc)], "provider_mutation_performed": False,
                          "paid_execution_requested": False}, sort_keys=True))
        return 2
    if args.receipt_out:
        Path(args.receipt_out).write_text(json.dumps(receipt, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
