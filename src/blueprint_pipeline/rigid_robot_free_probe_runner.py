"""Queue and run the capped GPU lane that records each prepared rigid scene's robot-free probe.

Beta journey step: L5 team policy evaluation. Owner approval 2026-10-26: under
each capture's website grant, build the scene and run the no-robot check
automatically; the first real run is a capped test.

Hook, for the worker that compiles a scene's team packet, called once the
scene's derived team profile is written::

    enqueue_rigid_robot_free_probe(capture_root) -> {"status": ...}

It reads the capture's profile, plans the probe request (bound to the task
specification teams are scored with), and writes one pending entry. It launches
nothing, never raises, and is idempotent.

Runner, ``python -m blueprint_pipeline.rigid_robot_free_probe_runner run-queue``
(``blueprint-rigid-robot-free-probe-runner.service``, woken when the pending
directory changes). With ``BLUEPRINT_RIGID_ROBOT_FREE_PROBE_ENABLED`` unset it
touches nothing. Enabled, for each pending capture it:

1. settles the latest attempt. An allocator still running, or one whose fate
   is unknown, holds that capture (never a second allocation); an allocated run
   whose provider zero is unproven holds it for an operator;
2. stops when the receipt is decided (passed or failed, never replaced), the
   task cannot be probed, or the attempt budget is spent (two funded attempts,
   four refusals before funding). An attempt refused while staging (the packet
   or its plan cannot carry the probe, or spawns its object off the task's
   start, so no episode could be scored) launched nothing and counts toward
   neither; the refusal seals the job ``staging_refused``. The WebApp's own
   refusal of the grant (an exhausted upstream cap or job ceiling, its provider
   terms, a refused or expired grant) is a money stop the owner can lift: it
   counts toward neither, never seals, and no attempt is staged again for
   ``GRANT_REFUSAL_RETRY_SECONDS``;
3. otherwise rechecks every precondition it can without spending (enable
   setting, receipt, the capture's retained website grant, driver ceiling,
   launch gate and watchdog scope), stages one attempt (probe input, sealed
   bundle, single-use authority) and runs the canonical allocator for it
   (``paid_resource_allocator gpu-canary --probe-kind rigid-robot-free-probe``).

Every terminal records the release that sealed it (``release_commit``, the
commit this runner stages bundles at). A ``staging_refused`` terminal is a
deterministic refusal a new release may have fixed, so each newly deployed
release sets it aside once, before the pass: the retired entry goes back to
pending, the terminal moves under ``superseded_terminals/`` beside a marker
naming the release that retried it (history is kept, never deleted), and the
next terminal lists the releases already tried, the ones that sealed it
included (``release_requeued``), so a refusal never loops. A terminal sealed by the running release stays final, as
does every other terminal: a decided receipt, a trace conflict, or a spent
attempt budget. Without a known running release nothing is retried.

Ingest happens only once the allocator result is ``completed``, which proves
teardown, provider zero and staged-object absence. It checks the result
wrapper's digest against the allocator's binding, the trace's digest, and the
trace's binding and plan against the request this attempt sealed. A recorded
trace is written once, immutably, to
``<robot_free_probe_root>/rigid_robot_free_probe_trace.v1.json``; then
``team_robot_free_probe.retain_recorded_probe`` scores it with the scored
specification, retains the receipt (a decided receipt is never replaced) and
republishes the job's offer. An undecided receipt leaves the capture queued for
its one further funded attempt.

Bulk bytes: with ``BLUEPRINT_RIGID_ROBOT_FREE_PROBE_WORK_ROOT`` set (the unit sets the
work volume), a new attempt writes its sealed bundle and allocator job directory (provider
staging, output, extraction and its disk hold) under
``<work root>/jobs/<job>/attempts/<NN>`` and records that directory; every other file stays in
the queue. An attempt staged without it keeps everything in the queue, as before.

For the capped canary: ``preflight --capture-root X --scratch-dir Y`` stages
one attempt outside the queue and runs the lane dry (no grant, no GPU), and
``run-queue --capture-root X --max-attempts 1`` starts at most one GPU attempt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest
from .rigid_robot_free_probe_bundle import (
    BUNDLE_RECEIPT_FILENAME,
    PROBE_KIND,
    RigidRobotFreeProbeBundleError,
    current_probe_state,
)
from .rigid_robot_free_probe_lane import (
    FUNDING_FILENAME,
    HARD_CAP_USD,
    HARD_TTL_SECONDS,
    MAX_HOURLY_RATE_USD,
    lane_enabled,
)
from .rigid_robot_free_probe_worker import INPUTS_FILENAME, RESULT_SCHEMA_VERSION, TRACE_FILENAME

QUEUE_ENV = "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE"
DEFAULT_QUEUE = "/var/lib/blueprint/pipeline-control-plane/rigid-robot-free-probes"
# The bulk volume a new attempt stages its sealed bundle and allocator job directory under (the
# unit sets /mnt/blueprint-work/rigid-robot-free-probe-work). The queue, its ledgers, the attempt
# records and the allocator receipts stay in the queue root. Unset keeps everything there, as
# before; an attempt records where its bulk went, so a later setting never moves an existing one.
WORK_ENV = "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_WORK_ROOT"
WORK_DIR_FIELD = "work_dir"
ENTRY_SCHEMA = "blueprint.rigid_robot_free_probe_queue_entry.v1"
TERMINAL_SCHEMA = "blueprint.rigid_robot_free_probe_terminal.v1"
INGEST_SCHEMA = "blueprint.rigid_robot_free_probe_ingest.v1"
TERMINAL_FILENAME = "rigid_robot_free_probe_terminal.json"
INGEST_FILENAME = "rigid_robot_free_probe_ingest.json"
AUTHORITY_FILENAME = "rigid_robot_free_probe_authority.v1.json"
ATTEMPT_FILENAME = "rigid_robot_free_probe_attempt.json"
INTENT_FILENAME = "allocator_launch_intent.json"
PROCESS_FILENAME = "allocator_process.json"
SPAWN_FAILED_FILENAME = "allocator_spawn_failed.json"
RESULT_FILENAME = "allocator_result.json"
MAX_FUNDED_ATTEMPTS = 2
MAX_UNFUNDED_ATTEMPTS = 4
# Nothing will be wanted of these captures under their current profile. A missing profile is
# not final: the hook runs only once a profile exists, so its absence here is configuration.
FINAL_STATES = frozenset({"receipt_decided", "unavailable_for_task", "scene_revision_unbound"})
WAITING_STATES = frozenset({"running", "unknown", "provider_zero_unproven", "retention_pending"})
# Refused before any grant was reserved: bounded, and never retried within the same pass.
UNFUNDED_STATES = frozenset({"unfunded_blocked", "not_launched"})
# The WebApp itself refused the grant (its upstream cap or the job's ceiling exhausted, its provider terms, a
# refused, expired or missing grant): a money or rights stop the owner can lift, so it counts toward no limit
# and never seals the job (capture-447ae4f4: four such refusals sealed ``attempts_exhausted``). Nothing was
# reserved and no paid run is staged for it again until ``GRANT_REFUSAL_RETRY_SECONDS`` pass: the WebApp offers
# no read of a ledger's headroom, so only a reservation can tell, and each staging re-reads the room collider.
GRANT_REFUSED_STATE = "website_grant_refused"
GRANT_REFUSAL_BLOCKERS = ("rigid_robot_free_probe_website_grant_exhausted",
                          "rigid_robot_free_probe_website_grant_refused:",
                          "rigid_robot_free_probe_website_grant_expired",
                          "rigid_robot_free_probe_website_grant_missing")
GRANT_REFUSAL_RETRY_SECONDS = 3 * 3600
# Refused while staging, before any argv existed: nothing was launched, funded or allocated, so the
# attempt counts toward neither limit. Its refusal seals the job, so it cannot loop.
NOT_STAGED_STATE = "not_staged"
TERMINAL_STATES = frozenset({"decided", "trace_conflict", "staging_refused"})
# A deterministic refusal a new release may have fixed: each terminal names the release that sealed it,
# each newly deployed release sets one aside once, and the next terminal keeps the releases already tried
# (the shape of ``team_scene_packet_worker``'s once-per-release retry).
RELEASE_COMMIT_FIELD = "release_commit"
RELEASE_REQUEUE_FIELD = "release_requeued"
RELEASE_RETRIED_STATES = frozenset({"staging_refused"})
SUPERSEDED_DIRNAME = "superseded_terminals"
SUPERSESSION_SCHEMA = "blueprint.rigid_robot_free_probe_terminal_supersession.v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
REPO_ROOT = Path(__file__).resolve().parents[2]


class RigidRobotFreeProbeIngestError(ValueError):
    """The returned run cannot be ingested; the reason names why."""


class RigidRobotFreeProbeWorkRootError(RuntimeError):
    """The configured work root cannot hold a new attempt; nothing is staged (not a staging refusal)."""


def _queue_root(queue_root: str | Path | None) -> Path:
    return Path(queue_root or os.environ.get(QUEUE_ENV) or DEFAULT_QUEUE)


def new_attempt_work_dir(job: Path, index: int) -> Path | None:
    """Where a new attempt's bulk goes: ``<work root>/jobs/<job>/attempts/<NN>``, or None for the queue."""

    raw = str(os.environ.get(WORK_ENV) or "").strip()
    if not raw:
        return None
    root = Path(raw)
    if (not root.is_absolute() or ".." in root.parts or not root.is_dir()
            or any(path.is_symlink() for path in (root, *root.parents))):
        raise RigidRobotFreeProbeWorkRootError("rigid_robot_free_probe_work_root_unavailable")
    return root / "jobs" / job.name / "attempts" / f"{index:02d}"


def attempt_work_dir(attempt: Path) -> Path:
    """Where this attempt's bundle and allocator job directory live: the one it recorded, else itself."""

    recorded = (_read_optional(attempt / ATTEMPT_FILENAME) or {}).get(WORK_DIR_FIELD)
    if recorded is None:
        return attempt
    path = Path(recorded) if isinstance(recorded, str) else Path()
    if not path.is_absolute() or ".." in path.parts or path.name != attempt.name:
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_work_dir_invalid")
    return path


def _funded(attempt: Path) -> bool:
    return (attempt_work_dir(attempt) / "allocator" / FUNDING_FILENAME).is_file()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_record_invalid")
    return value


def _read_optional(path: Path) -> dict[str, Any] | None:
    return _read(path) if path.is_file() and not path.is_symlink() else None


def _text(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_exclusive(path: Path, value: Mapping[str, Any] | str, *, mode: int = 0o640) -> Path:
    """Write once: an existing file raises ``FileExistsError`` and is never replaced."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    text = value if isinstance(value, str) else _text(value)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False,
                                     encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.chmod(stream.name, mode)
        os.link(stream.name, path)
    finally:
        os.unlink(stream.name)
    return path


def _replace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False,
                                     encoding="utf-8") as stream:
        stream.write(_text(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(stream.name, 0o640)
    os.replace(stream.name, path)


def _capture_key(capture_root: str) -> str:
    return hashlib.sha256(str(capture_root).encode("utf-8")).hexdigest()[:32]


def _job_path(root: Path, capture_root: str, request_digest: str) -> Path:
    return root / "jobs" / f"{_capture_key(capture_root)}-{str(request_digest).removeprefix('sha256:')[:16]}"


def _job_dir(root: Path, state: Mapping[str, Any]) -> Path:
    return _job_path(root, state["capture_root"], state["request"]["request_digest"])


def enqueue_rigid_robot_free_probe(capture_root: str | Path, *, queue_root: str | Path | None = None
                                   ) -> dict[str, Any]:
    """Queue the capture's robot-free probe when its current profile wants one; launch nothing, never raise."""

    try:
        state = current_probe_state(capture_root)
        if state["status"] != "probe_required":
            return {"status": state["status"], "queued": False, "capture_root": state.get("capture_root")}
        root = _queue_root(queue_root)
        path = root / "pending" / f"{_capture_key(state['capture_root'])}.json"
        digest = state["request"]["request_digest"]
        existing = _read_optional(path)
        common = {"queued": True, "lane_enabled": lane_enabled(), "path": str(path), "probe_request_digest": digest}
        if existing is not None and existing.get("probe_request_digest") == digest:
            return {"status": "already_queued", **common}
        _replace(path, {"schema_version": ENTRY_SCHEMA, "capture_root": state["capture_root"],
                        "probe_request_digest": digest, "queued_at_iso": utc_now_iso()})
        return {"status": "queued", **common}
    except Exception as exc:  # noqa: BLE001 - the hook never breaks the packet worker that calls it
        return {"status": "refused", "queued": False, "reason": f"{type(exc).__name__}:{str(exc)[:200]}"}


def allocator_argv(*, attempt: Path, authority: Mapping[str, Any], authority_path: Path,
                   bundle_receipt_path: Path, machine_avoidlist_path: str | None = None,
                   work: Path | None = None) -> list[str]:
    """The one canonical allocator invocation for one staged attempt.

    Its job directory (provider staging, output and extraction) is in ``work``, the attempt's
    bulk directory, when it has one; the admission and result receipts stay in the attempt.
    """

    argv = ["gpu-canary", "--provider", "vast", "--probe-kind", PROBE_KIND,
            "--native-task-arena-bundle-receipt", str(bundle_receipt_path),
            "--native-task-arena-attempt-authority", str(authority_path),
            "--adp-job-dir", str((attempt if work is None else work) / "allocator"),
            "--adp-max-hourly-rate-usd", str(authority["max_hourly_rate_usd"]),
            "--adp-max-spend-usd", str(authority["hard_cap_usd"]),
            "--adp-hard-ttl-seconds", str(authority["hard_ttl_seconds"]),
            "--admission-out", str(attempt / "paid_admission.json"),
            "--adapter-output", str(attempt / RESULT_FILENAME),
            "--pod-name", str(authority["resource_name"]), "--execute"]
    if machine_avoidlist_path:
        argv.extend(["--adp-machine-avoidlist", str(machine_avoidlist_path)])
    return argv


def _process_start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def _allocator_alive(attempt: Path) -> bool:
    record = _read_optional(attempt / PROCESS_FILENAME)
    if not record:
        return False
    try:
        os.kill(int(record["pid"]), 0)
    except PermissionError:
        return True  # existence without authority to inspect is not proof of exit
    except (OSError, KeyError, TypeError, ValueError):
        return False
    current = _process_start_ticks(int(record["pid"]))
    return current is not None and current == record.get("start_ticks")


def run_allocator(argv: list[str], *, attempt: Path) -> int:
    """Run the canonical allocator for one attempt; its own watchdog survives this process."""

    log_path = attempt / "allocator.log"
    with log_path.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(  # nosec B603 - fixed module, runner-staged argv
                [sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator", *argv],
                cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            # No process exists, so nothing can have been allocated.
            _write_exclusive(attempt / SPAWN_FAILED_FILENAME, {"reason": f"{type(exc).__name__}:{exc}"[:300],
                                                              "observed_at_iso": utc_now_iso()})
            return 127
        _write_exclusive(attempt / PROCESS_FILENAME, {"pid": process.pid, "start_ticks": _process_start_ticks(
            process.pid), "started_at_iso": utc_now_iso()})
        return process.wait()


def source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _attempts(job: Path) -> list[Path]:
    return sorted(path for path in (job / "attempts").glob("[0-9][0-9]") if path.is_dir())


def _record_ingest(attempt: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = {"schema_version": INGEST_SCHEMA, **dict(value), "settled_at_iso": utc_now_iso(), "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    try:
        _write_exclusive(attempt / INGEST_FILENAME, record)
    except FileExistsError:
        return _read(attempt / INGEST_FILENAME)
    return record


def verified_trace(result: Mapping[str, Any], *, attempt: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    """The trace the GPU host recorded, once its wrapper and binding prove it is this attempt's request."""

    # The attempt's own directories: itself, and the bulk directory it recorded at staging.
    roots = {attempt.resolve(), attempt_work_dir(attempt).resolve()}
    staged = _read(attempt / ATTEMPT_FILENAME)
    if result.get("bundle_sha256") != staged.get("bundle_sha256"):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_result_bundle_mismatch")
    native_path = Path(str(result.get("native_control_result_path") or "")).resolve()
    if (not any(root in native_path.parents for root in roots) or native_path.is_symlink()
            or not native_path.is_file()):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_result_missing")
    native = _read(native_path)
    if (native.get("result_digest") != result.get("native_control_result_digest")
            or native.get("result_digest") != canonical_digest(native, digest_field="result_digest")):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_result_digest_invalid")
    if (native.get("schema_version") != RESULT_SCHEMA_VERSION or native.get("status") != "completed"
            or native.get("candidate_policy_queried") is not False or native.get("policy_loaded") is not False):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_result_contract_invalid")
    if native.get("probe_request_digest") != request.get("request_digest"):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_result_request_mismatch")
    trace = native.get("trace")
    if (not isinstance(trace, Mapping) or trace.get("trace_digest") != native.get("trace_digest")
            or trace.get("trace_digest") != canonical_digest(dict(trace), digest_field="trace_digest")):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_trace_digest_invalid")
    if trace.get("binding") != request.get("binding"):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_trace_binding_mismatch")
    if trace.get("plan") != request.get("plan"):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_trace_plan_mismatch")
    copy = native_path.parent / TRACE_FILENAME
    if copy.is_file() and json.loads(copy.read_text(encoding="utf-8")) != trace:
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_trace_copy_mismatch")
    return dict(trace)


def retain_trace_file(run_root: Path, trace: Mapping[str, Any]) -> str:
    """Write the recorded trace beside the source launch once; a different trace there is a conflict."""

    path = Path(run_root) / TRACE_FILENAME
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_trace_path_unsafe")
    text = json.dumps(dict(trace), sort_keys=True) + "\n"
    try:
        _write_exclusive(path, text, mode=0o440)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != text:
            raise RigidRobotFreeProbeIngestError("rigid_robot_free_probe_trace_conflict") from None
        return "unchanged"
    return "written"


def settle_attempt(attempt: Path, *, state: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one attempt's outcome; ingest a completed run once teardown and provider zero are proven."""

    from .robot_free_readiness_retention import DECIDED_STATUSES
    from .task_evaluation_scene_execution_authority import proves_no_provider_allocation
    from .team_robot_free_probe import retain_recorded_probe

    recorded = _read_optional(attempt / INGEST_FILENAME)
    if recorded is not None:
        return recorded
    result = _read_optional(attempt / RESULT_FILENAME)
    if result is None:
        if (attempt / SPAWN_FAILED_FILENAME).is_file() and not (attempt / PROCESS_FILENAME).exists():
            return _record_ingest(attempt, {"state": "not_launched", "funded": False, "allocated": False,
                                            "reason": "allocator_spawn_failed"})
        if (attempt / INTENT_FILENAME).is_file():
            # A launched allocator with no result is never followed by a second allocation.
            return {"state": "running" if _allocator_alive(attempt) else "unknown"}
        return _record_ingest(attempt, {"state": "not_launched", "funded": False, "allocated": False})
    work = attempt_work_dir(attempt)
    if work != attempt and not work.is_dir():
        # Its bulk volume is not there: whether it was funded cannot be read. Hold the capture
        # (never a second allocation) until the volume is back.
        return {"state": "unknown", "reason": "rigid_robot_free_probe_work_dir_unavailable"}
    funded = _funded(attempt)
    allocated = not proves_no_provider_allocation(result)
    zero = result.get("continuing_spend_from_this_run") is False and result.get("all_staged_objects_absent") is True
    if allocated and not zero:
        return {"state": "provider_zero_unproven", "funded": funded, "allocated": True}
    base = {"funded": funded, "allocated": allocated, "allocator_status": result.get("status"),
            "allocator_blockers": [str(item) for item in result.get("blockers") or []][:20],
            "estimated_cost_usd": result.get("estimated_cost_usd")}
    if result.get("status") != "completed":
        name = "funded_unallocated" if funded and not allocated else "allocated_blocked" if allocated else (
            GRANT_REFUSED_STATE if any(str(item).startswith(GRANT_REFUSAL_BLOCKERS) for item in base[
                "allocator_blockers"]) else "unfunded_blocked")
        return _record_ingest(attempt, {**base, "state": name})
    request = _read(attempt / INPUTS_FILENAME)["probe_request"]
    try:
        trace = verified_trace(result, attempt=attempt, request=request)
    except (OSError, ValueError) as exc:
        return _record_ingest(attempt, {**base, "state": "undecided", "ingest_refusal": str(exc)[:200]})
    if request.get("request_digest") != state["request"]["request_digest"]:
        return _record_ingest(attempt, {**base, "state": "superseded", "trace_digest": trace.get("trace_digest")})
    try:
        trace_write = (retain_trace_file(Path(state["robot_free_probe_root"]), trace)
                       if trace.get("status") == "recorded" else "not_recorded")
    except RigidRobotFreeProbeIngestError as exc:
        return _record_ingest(attempt, {**base, "state": "trace_conflict", "ingest_refusal": str(exc)})
    try:
        retention = retain_recorded_probe(profile=state["profile"], facts=state["facts"], trace=trace)
    except Exception as exc:  # noqa: BLE001 - local retention is retried by the next pass; nothing is respent
        return {"state": "retention_pending", "reason": f"{type(exc).__name__}:{str(exc)[:200]}"}
    decided = retention.get("receipt_status") in DECIDED_STATUSES
    return _record_ingest(attempt, {**base, "state": "decided" if decided else "undecided",
                                    "trace_digest": trace.get("trace_digest"), "trace_status": trace.get("status"),
                                    "trace_write": trace_write, "retention": retention})


def _never_staged(attempt: Path) -> bool:
    """No argv, intent, process or result was ever written: no allocator can have run from it."""

    return not any((attempt / name).exists() for name in (
        INGEST_FILENAME, ATTEMPT_FILENAME, INTENT_FILENAME, PROCESS_FILENAME, SPAWN_FAILED_FILENAME, RESULT_FILENAME))


def _record_not_staged(attempt: Path, reason: str, **extra: Any) -> None:
    if attempt.is_dir() and _never_staged(attempt):
        _record_ingest(attempt, {"state": NOT_STAGED_STATE, "funded": False, "allocated": False,
                                 "reason": reason[:300], **extra})


def _release(value: Any) -> str | None:
    return value if isinstance(value, str) and _COMMIT.fullmatch(value) else None


def _running_release(hooks: Mapping[str, Callable[..., Any]]) -> str | None:
    """The release this pass stages bundles at, or None when it cannot be told (then nothing is retried)."""

    try:
        return _release(hooks["source_commit"]())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _supersessions(job: Path) -> list[dict[str, Any]]:
    rows = [_read(path) for path in (job / SUPERSEDED_DIRNAME).glob("*.supersession.json")
            if path.is_file() and not path.is_symlink()]
    return sorted(rows, key=lambda row: str(row.get("superseded_at_iso") or ""))


def _releases_tried(job: Path) -> list[str]:
    """Every release that already sealed or retried one of this job's superseded refusals, in order."""

    tried: list[str] = []
    for row in _supersessions(job):
        for release in (row.get("superseded_release_commit"), row.get(RELEASE_COMMIT_FIELD)):
            if _release(release) and release not in tried:
                tried.append(release)
    return tried


def _new_release_untried(terminal: Mapping[str, Any], release: str | None) -> bool:
    """A staging refusal sealed by another release, or by none recorded, that ``release`` has not yet retried."""

    tried = terminal.get(RELEASE_REQUEUE_FIELD)
    return (release is not None and terminal.get("status") in RELEASE_RETRIED_STATES
            and terminal.get(RELEASE_COMMIT_FIELD) != release
            and release not in (tried if isinstance(tried, list) else []))


def _supersede_for_release(job: Path, release: str | None) -> dict[str, Any] | None:
    """Set the job's staging refusal aside for a newly deployed release, once; return the marker.

    The marker naming the release is written before the terminal moves beside it, so each release
    sets a terminal aside at most once and a pass interrupted between the two finishes the move.
    The refused attempt is recorded ``not_staged`` (it launched nothing), so it counts toward no
    limit and the next attempt takes the next index. Nothing is deleted. The job lock is held.
    """

    terminal = _read_optional(job / TERMINAL_FILENAME)
    if terminal is None:
        return None
    digest = str(terminal.get("terminal_digest") or "")
    name = digest.removeprefix("sha256:")[:16]
    if re.fullmatch(r"[0-9a-f]{16}", name) is None:
        return None
    history = job / SUPERSEDED_DIRNAME
    marker_path = history / f"{name}.supersession.json"
    marker = _read_optional(marker_path)
    if marker is None:
        if not _new_release_untried(terminal, release):
            return None
        marker = {"schema_version": SUPERSESSION_SCHEMA, RELEASE_COMMIT_FIELD: release,
                  "superseded_terminal_digest": digest, "superseded_status": terminal.get("status"),
                  "superseded_release_commit": terminal.get(RELEASE_COMMIT_FIELD),
                  "superseded_detail": terminal.get("detail"), "superseded_at_iso": utc_now_iso(), "marker_digest": ""}
        marker["marker_digest"] = canonical_digest(marker, digest_field="marker_digest")
        try:
            _write_exclusive(marker_path, marker)
        except FileExistsError:
            marker = _read(marker_path)
    attempts = _attempts(job)
    if attempts:
        _record_not_staged(attempts[-1], str((terminal.get("detail") or {}).get("reason") or "staging_refused"),
                           recorded_at_supersession_of=digest)
    os.replace(job / TERMINAL_FILENAME, history / f"{name}.terminal.json")
    return marker


def requeue_new_release(queue_root: str | Path | None, release: str | None, *,
                        names: set[str] | None = None) -> list[dict[str, Any]]:
    """Return each retired entry whose staging refusal ``release`` has not yet retried to pending.

    The pass that follows sets the terminal aside under the job lock (``_supersede_for_release``).
    An entry already pending is left as it is; it is processed the same way.
    """

    root = _queue_root(queue_root)
    requeued: list[dict[str, Any]] = []
    if _release(release) is None:
        return requeued
    for path in sorted((root / "done").glob("*.json")):
        target = root / "pending" / path.name
        if (names is not None and path.name not in names) or target.exists():
            continue
        try:
            entry = _read(path)
            job = _job_path(root, str(entry["capture_root"]), str(entry["probe_request_digest"]))
            terminal = _read_optional(job / TERMINAL_FILENAME)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if terminal is None or not _new_release_untried(terminal, release):
            continue
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        os.replace(path, target)
        requeued.append({"entry": path.name, "reason": "new_release", RELEASE_COMMIT_FIELD: release,
                         "terminal_status": terminal.get("status"),
                         "terminal_release_commit": terminal.get(RELEASE_COMMIT_FIELD)})
    return requeued


def final_terminal(state: Mapping[str, Any], *, queue_root: str | Path | None = None,
                   release: str | None = None) -> dict[str, Any] | None:
    """The sealed terminal of the capture's current probe request that no pass under ``release`` retries, or None.

    Read only. A staging refusal a newly deployed release has not yet tried is not final: that release's pass
    sets it aside and stages again (``requeue_new_release``). Every other terminal is final, as it is without a
    known release. The screening pass names a final one to the website job (``beta_policy_screening_holds``).
    """

    terminal = _read_optional(_job_dir(_queue_root(queue_root), state) / TERMINAL_FILENAME)
    if terminal is None or _new_release_untried(terminal, _release(release)):
        return None
    return terminal


def _grant_refusal_wait(settled: Mapping[str, Any], *, now: float) -> int | None:
    """Seconds until a WebApp grant refusal may be tried again, or None when nothing waits."""

    if settled.get("state") != GRANT_REFUSED_STATE:
        return None
    try:
        refused = datetime.fromisoformat(str(settled["settled_at_iso"])).timestamp()
    except (KeyError, TypeError, ValueError):
        return GRANT_REFUSAL_RETRY_SECONDS  # an unreadable time waits the whole interval, never less
    remaining = refused + GRANT_REFUSAL_RETRY_SECONDS - float(now)
    return min(GRANT_REFUSAL_RETRY_SECONDS, max(1, int(remaining))) if remaining > 0 else None


def _counts(job: Path) -> tuple[int, int]:
    funded = unfunded = 0
    for attempt in _attempts(job):
        record = _read_optional(attempt / INGEST_FILENAME) or {}
        # The ingest record (beside the queue) counts too, so a funded attempt whose bulk volume
        # cannot be read still counts as funded.
        if record.get("funded") is True or record.get("allocated") is True or _funded(attempt):
            funded += 1
        elif record.get("state") in UNFUNDED_STATES:
            unfunded += 1
    return funded, unfunded


def precheck_blockers(state: Mapping[str, Any], *, job: Path) -> list[str]:
    """Everything a run needs that can be checked without spending; no attempt is staged otherwise."""

    from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
    from .rigid_robot_free_probe_lane import (
        driver_ceiling_blockers,
        launch_environment_blockers,
        resource_name_for,
        retained_website_grant,
    )

    blockers = [] if lane_enabled() else ["rigid_robot_free_probe_lane_disabled"]
    blockers.extend(driver_ceiling_blockers())
    blockers.extend(retained_website_grant(state["capture_root"])[1])
    if not blockers:
        blockers.extend(launch_environment_blockers(
            job_dir=job / "precheck", resource_name=resource_name_for("precheck-" + job.name),
            container_image=NATIVE_TASK_ARENA_IMAGE, max_hourly_rate_usd=MAX_HOURLY_RATE_USD,
            hard_cap_usd=HARD_CAP_USD, hard_ttl_seconds=HARD_TTL_SECONDS))
    return blockers


def stage_attempt(job: Path, index: int, *, state: Mapping[str, Any], commit: str,
                  work_dir: Path | None = None) -> dict[str, Any]:
    """Write one attempt's probe input, sealed bundle, single-use authority and allocator argv.

    With ``work_dir`` (``new_attempt_work_dir``) the sealed bundle and the allocator job
    directory are written there, a fresh directory, and the attempt record names it.
    """

    from .rigid_robot_free_probe_bundle import build_rigid_robot_free_probe_bundle, probe_inputs
    from .rigid_robot_free_probe_lane import build_paid_authority

    attempt = job / "attempts" / f"{index:02d}"
    attempt.mkdir(parents=True, mode=0o750)
    work = attempt if work_dir is None else Path(work_dir)
    if work != attempt:
        work.mkdir(parents=True, mode=0o750)
    facts = state["facts"]
    inputs_path = _write_exclusive(attempt / INPUTS_FILENAME, probe_inputs(state=state))
    bundle = build_rigid_robot_free_probe_bundle(
        job_dir=work / "bundle", packet_dir=facts["packet_dir"],
        runtime_source_packet_receipt=facts["runtime_source_packet_receipt"], inputs_path=inputs_path,
        implementation_commit=commit)
    receipt_path = work / "bundle" / BUNDLE_RECEIPT_FILENAME
    authority = build_paid_authority(attempt_id=f"{job.name}-a{index:02d}", inputs_path=inputs_path,
                                     bundle_receipt_path=receipt_path, bundle=bundle, source_commit=commit)
    authority_path = _write_exclusive(attempt / AUTHORITY_FILENAME, authority)
    argv = allocator_argv(attempt=attempt, authority=authority, authority_path=authority_path,
                          bundle_receipt_path=receipt_path,
                          machine_avoidlist_path=state["profile"].get("machine_avoidlist_path"),
                          work=None if work == attempt else work)
    _write_exclusive(attempt / ATTEMPT_FILENAME, {"attempt_id": authority["attempt_id"], "argv": argv,
                                                  "authority_digest": authority["authority_digest"],
                                                  "bundle_sha256": bundle["bundle_sha256"],
                                                  **({WORK_DIR_FIELD: str(work)} if work != attempt else {}),
                                                  "staged_at_iso": utc_now_iso()})
    return {"root": attempt, "argv": argv}


def _finish(root: Path, item: Path, entry: Mapping[str, Any], outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Retire the pending entry unless a newer request replaced it meanwhile."""

    current = _read_optional(item)
    if current is not None and current.get("probe_request_digest") == entry.get("probe_request_digest"):
        (root / "done").mkdir(parents=True, exist_ok=True, mode=0o750)
        os.replace(item, root / "done" / item.name)
    return dict(outcome)


def _seal_terminal(root: Path, item: Path, entry: Mapping[str, Any], job: Path, status: str,
                   detail: Mapping[str, Any], release: str | None = None) -> dict[str, Any]:
    tried = _releases_tried(job)
    value = {"schema_version": TERMINAL_SCHEMA, "status": status, "capture_root": entry.get("capture_root"),
             "probe_request_digest": entry.get("probe_request_digest"), "detail": dict(detail),
             # The release that sealed it, and those that already retried it: a later release retries a
             # staging refusal once.
             **({RELEASE_COMMIT_FIELD: release} if _release(release) else {}),
             **({RELEASE_REQUEUE_FIELD: tried} if tried else {}),
             "sealed_at_iso": utc_now_iso(), "terminal_digest": ""}
    value["terminal_digest"] = canonical_digest(value, digest_field="terminal_digest")
    try:
        _write_exclusive(job / TERMINAL_FILENAME, value)
    except FileExistsError:
        value = _read(job / TERMINAL_FILENAME)
    return _finish(root, item, entry, {"capture_root": entry.get("capture_root"), **value})


def _advance(root: Path, item: Path, entry: Mapping[str, Any], job: Path, state: Mapping[str, Any],
             hooks: Mapping[str, Callable[..., Any]], max_attempts: int | None,
             release: str | None = None) -> dict[str, Any]:
    staged_now: set[Path] = set()
    while True:
        terminal = _read_optional(job / TERMINAL_FILENAME)
        if terminal is not None:
            return _finish(root, item, entry, terminal)
        attempts = _attempts(job)
        settled: dict[str, Any] = {}
        if attempts:
            settled = settle_attempt(attempts[-1], state=state)
            if settled["state"] in WAITING_STATES:
                return {"status": "waiting", "capture_root": state["capture_root"], "attempt": attempts[-1].name,
                        "reason": settled["state"]}
            if settled["state"] in TERMINAL_STATES:
                return _seal_terminal(root, item, entry, job, settled["state"], settled, release)
        funded, unfunded = _counts(job)
        if funded >= MAX_FUNDED_ATTEMPTS or unfunded >= MAX_UNFUNDED_ATTEMPTS:
            return _seal_terminal(root, item, entry, job, "attempts_exhausted",
                                  {"funded_attempts": funded, "unfunded_attempts": unfunded, "last": settled},
                                  release)
        before_funding = settled.get("state") in {*UNFUNDED_STATES, GRANT_REFUSED_STATE}
        if attempts and attempts[-1] in staged_now and before_funding:
            # Refused before funding; the next wake-up retries, never this pass.
            return {"status": "deferred", "capture_root": state["capture_root"],
                    "blockers": settled.get("allocator_blockers") or [settled.get("reason") or settled["state"]]}
        state = current_probe_state(state["capture_root"])
        if state["status"] == "receipt_decided":
            return _seal_terminal(root, item, entry, job, "decided", {"receipt_status": state["receipt_status"]},
                                  release)
        if state["status"] != "probe_required" or _job_dir(root, state) != job:
            return {"status": "deferred", "capture_root": entry.get("capture_root"),
                    "reason": "request_changed" if state["status"] == "probe_required" else state["status"]}
        waiting = _grant_refusal_wait(settled, now=hooks.get("now", time.time)())
        if waiting is not None:
            # The WebApp refused the grant on the last attempt: nothing is staged until the wait passes.
            return {"status": "deferred", "capture_root": state["capture_root"], "reason": GRANT_REFUSED_STATE,
                    "retry_after_seconds": waiting, "blockers": settled.get("allocator_blockers") or []}
        if max_attempts is not None and len(staged_now) >= max_attempts:
            # A capped canary pass: its attempts are settled, and nothing more is sealed or spent.
            return {"status": "deferred", "capture_root": state["capture_root"],
                    "reason": "attempt_limit_reached_for_this_pass", "last": settled}
        blockers = precheck_blockers(state, job=job)
        if blockers:
            return {"status": "deferred", "capture_root": state["capture_root"], "blockers": sorted(set(blockers))}
        index = len(attempts)
        # Outside the refusal below: an unusable work root is configuration, so the entry stays
        # pending (the pass reports the error) instead of sealing a staging refusal.
        work_dir = new_attempt_work_dir(job, index)
        try:
            staged = stage_attempt(job, index, state=state, commit=hooks["source_commit"](), work_dir=work_dir)
        except (RigidRobotFreeProbeBundleError, ValueError) as exc:
            # The packet or its plan cannot carry the probe: deterministic, so never retried under this
            # release. The refused attempt launched nothing and counts toward no limit.
            refused = job / "attempts" / f"{index:02d}"
            _record_not_staged(refused, str(exc))
            return _seal_terminal(root, item, entry, job, "staging_refused",
                                  {"reason": str(exc)[:300], "attempt": refused.name}, release)
        staged_now.add(staged["root"])
        _write_exclusive(staged["root"] / INTENT_FILENAME, {"observed_at_iso": utc_now_iso()})
        hooks["allocator_runner"](staged["argv"], attempt=staged["root"])


def _process_entry(root: Path, item: Path, hooks: Mapping[str, Callable[..., Any]],
                   max_attempts: int | None = None, release: str | None = None) -> dict[str, Any]:
    entry = _read(item)
    state = current_probe_state(str(entry.get("capture_root") or ""))
    if state["status"] in FINAL_STATES:
        return _finish(root, item, entry, {"status": state["status"], "capture_root": state.get("capture_root")})
    if state["status"] != "probe_required":
        return {"status": "deferred", "capture_root": state.get("capture_root"), "reason": state["status"]}
    job = _job_dir(root, state)
    job.mkdir(parents=True, exist_ok=True, mode=0o750)
    with (job / "runner.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "locked", "capture_root": state["capture_root"]}
        superseded = _supersede_for_release(job, release)
        row = _advance(root, item, entry, job, state, hooks, max_attempts, release)
        return {**row, "superseded_terminal": superseded} if superseded is not None else row


def run_queue(queue_root: str | Path | None = None, *, capture_root: str | Path | None = None,
              hooks: Mapping[str, Callable[..., Any]] | None = None,
              max_attempts: int | None = None) -> list[dict[str, Any]]:
    """Advance every pending capture (or just ``capture_root``); with the lane switched off, touch nothing.

    ``max_attempts`` caps the GPU attempts this pass may start per capture (the
    capped canary runs one); the scene's own budget is unchanged.
    """

    if max_attempts is not None and (type(max_attempts) is not int or max_attempts < 1):
        raise ValueError("rigid_robot_free_probe_max_attempts_invalid")
    root = _queue_root(queue_root)
    names = None if capture_root is None else {f"{_capture_key(str(Path(capture_root).resolve()))}.json"}
    if not lane_enabled():
        pending = [item for item in (root / "pending").glob("*.json") if names is None or item.name in names]
        return [{"status": "lane_disabled", "pending": len(pending)}]
    bound = {"allocator_runner": run_allocator, "source_commit": source_commit, **dict(hooks or {})}
    release = _running_release(bound)
    # A staging refusal this release has not yet retried returns to pending before the pass reads it.
    requeue_new_release(root, release, names=names)
    pending = [item for item in sorted((root / "pending").glob("*.json")) if names is None or item.name in names]
    rows = []
    for item in pending:
        try:
            rows.append(_process_entry(root, item, bound, max_attempts, release))
        except Exception as exc:  # noqa: BLE001 - one capture's failure never stops the others
            rows.append({"status": "error", "entry": item.name, "reason": f"{type(exc).__name__}:{str(exc)[:200]}"})
    return rows


def preflight_capture(capture_root: str | Path, *, scratch_dir: str | Path,
                      hooks: Mapping[str, Callable[..., Any]] | None = None) -> dict[str, Any]:
    """Stage one attempt in ``scratch_dir`` and run the allocator lane dry: every check, no grant, no GPU.

    The queue is never touched, so a preflight can never become a paid attempt.
    """

    bound = {"allocator_runner": run_allocator, "source_commit": source_commit, **dict(hooks or {})}
    state = current_probe_state(capture_root)
    if state["status"] != "probe_required":
        return {"status": state["status"], "capture_root": state.get("capture_root")}
    scratch = Path(scratch_dir)
    if scratch.exists() and any(scratch.iterdir()):
        return {"status": "refused", "reason": "rigid_robot_free_probe_preflight_scratch_not_empty"}
    blockers = precheck_blockers(state, job=scratch)
    staged = stage_attempt(scratch, 0, state=state, commit=bound["source_commit"]())
    bound["allocator_runner"]([value for value in staged["argv"] if value != "--execute"], attempt=staged["root"])
    result = _read_optional(staged["root"] / RESULT_FILENAME) or {}
    ready = not blockers and result.get("status") == "dry_run_ready"
    return {"status": "ready" if ready else "blocked", "capture_root": state["capture_root"],
            "probe_request_digest": state["request"]["request_digest"], "precheck_blockers": blockers,
            "allocator_status": result.get("status"), "allocator_blockers": result.get("blockers") or [],
            "attempt_root": str(staged["root"]), "execute_argv": staged["argv"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Queue and run the rigid robot-free probe lane.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-queue", help="advance pending captures (the path-activated runner)")
    run.add_argument("--queue-root", type=Path)
    run.add_argument("--capture-root", help="advance only this capture's entry (a capped canary)")
    run.add_argument("--max-attempts", type=int, help="start at most this many GPU attempts per capture this pass")
    enqueue = commands.add_parser("enqueue", help="queue one capture's probe; launches nothing")
    enqueue.add_argument("--capture-root", required=True)
    enqueue.add_argument("--queue-root", type=Path)
    preflight = commands.add_parser("preflight", help="stage one attempt in a scratch dir and run the lane dry")
    preflight.add_argument("--capture-root", required=True)
    preflight.add_argument("--scratch-dir", required=True, type=Path)
    commands.add_parser("condition", help="exit 0 only when the enable setting is on (systemd ExecCondition)")
    args = parser.parse_args(argv)
    if args.command == "condition":
        return 0 if lane_enabled() else 1
    if args.command == "enqueue":
        outcome = enqueue_rigid_robot_free_probe(args.capture_root, queue_root=args.queue_root)
        print(json.dumps(outcome, sort_keys=True))
        return 0 if outcome["status"] in {"queued", "already_queued"} else 1
    if args.command == "preflight":
        outcome = preflight_capture(args.capture_root, scratch_dir=args.scratch_dir)
        print(json.dumps(outcome, sort_keys=True, default=str))
        return 0 if outcome["status"] == "ready" else 1
    rows = run_queue(args.queue_root, capture_root=args.capture_root, max_attempts=args.max_attempts)
    print(json.dumps(rows, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_QUEUE",
    "MAX_FUNDED_ATTEMPTS",
    "MAX_UNFUNDED_ATTEMPTS",
    "QUEUE_ENV",
    "RigidRobotFreeProbeIngestError",
    "allocator_argv",
    "enqueue_rigid_robot_free_probe",
    "final_terminal",
    "main",
    "precheck_blockers",
    "preflight_capture",
    "requeue_new_release",
    "retain_trace_file",
    "run_allocator",
    "run_queue",
    "settle_attempt",
    "stage_attempt",
    "verified_trace",
]
