"""Hold a deploy while a paid stage of a website job is live on the running release.

Every deploy makes scene progression retire the in-flight attempt and rebuild it
(about 15 min of preparation, then about an hour of activation), because each
later stage refuses work bound to an older release. So a deploy that lands while
a paid stage runs throws away work that was already paid for. This holds the
deploy, never the job: the stage keeps going, and the deploy goes out as soon as
it ends, when the hold reaches its cap, or at once with an override. It only
reads; it never stops, retries or changes a job, queue entry, lock or ledger.

Live means bound to the running release, and one of:

- a scene-configuration activation (lane ``task_evaluation_scene_configuration``):
  - pending in the activation queue, while the queue's path unit can claim it (a
    door hold on that unit means nothing will, so it holds nothing), or
  - claimed in processing/ with the worker's flock still held (a claim whose
    worker died holds no lock and does not count), or
  - finished with its profile authority materialized, but its launch not yet
    submitted, and finished recently (``SUBMIT_WINDOW_SECONDS``);
- a scene-configuration launch whose progression state says it was submitted for
  this release (``launch_progression.json``) and whose request is pending in the
  launch queue (while the dispatcher's path unit can claim it) or claimed in
  processing/ while a live dispatcher holds the queue's ``.dispatcher.lock``;
- a paid-launch slot lock (``provider-locks/*.lock``, not the deploy gate) whose
  flock is held by a process running from this release's tree (``/proc/<pid>/cwd``);
- a beta-screening candidate ``launching`` or ``running`` in the screening ledger
  whose plan froze this release's commit (``implementation_commit``).

Anything unreadable or unexpected counts as not live. One cap (``CAP_SECONDS``)
bounds the hold, whichever stages are live and however one hands over to the next.

Run by door-deploy.sh with ``python3 -I -S``, so standard library only. The
runner cannot do this itself: it has no capabilities and these files are the
``blueprint`` user's own.

Exit codes:
    0  deploy now (nothing live, override given, cap reached, or unknown)
    5  a paid stage is live; stay parked (the hold file says until when to recheck).
       A second stdout line then says when to retry and when the cap lets it go:
       ``<retry ISO> <retry epoch> <cap ISO> <cap epoch>``, for a deploy that cannot park.

``activation_hold.py park`` turns a direct deploy into a parked one, writing the
record the runner would have written (exit 0), or exits 6 when it cannot.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import stat
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EXIT_DEPLOY = 0
EXIT_HOLD = 5
EXIT_CANNOT_PARK = 6
PARKED_SCHEMA = "blueprint_operator_door_parked_deploy.v1"
PARKED_MAX = 16
_DEPLOY_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-deploy-[0-9a-f]{8}")
LANE = "task_evaluation_scene_configuration"
MATERIALIZED_STATUS = "profile_authority_materialized_no_execution"
LAUNCH_QUEUED_STATUS = "scene_configuration_launch_queued"
# The activation unit's TimeoutStartSec: no worker run outlives it.
ACTIVATION_TIMEOUT_SECONDS = 255 * 60
# The progression timer submits a materialized launch within minutes. One that has
# not been submitted by then is blocked on something a deploy will not change.
SUBMIT_WINDOW_SECONDS = 30 * 60
CAP_SECONDS = ACTIVATION_TIMEOUT_SECONDS + SUBMIT_WINDOW_SECONDS
RECHECK_SECONDS = 5 * 60
HOLD_FILE = ".activation-hold"
HOLD_SCHEMA = "blueprint_operator_door_activation_hold.v1"
# Any short reason, written to the log; "fixes_live_activation" is the common one,
# since holding the fix the live activation needs protects nothing.
OVERRIDE_REASON = re.compile(r"[a-z0-9][a-z0-9_:.-]{0,63}")
ACTIVATION_PATH_UNIT = "blueprint-task-evaluation-launch-activation.path"
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_SCAN = 512
_COMMIT = re.compile(r"[0-9a-f]{40}")
# Stages a hold record names, in its ``stages`` field and each ``live`` row.
STAGE_ACTIVATION = "scene_configuration_activation"
STAGE_LAUNCH = "scene_configuration_launch"
STAGE_PAID_LOCK = "paid_launch_lock"
STAGE_SCREENING = "beta_screening"
# Under the control-plane state root, as door status reads them.
LAUNCH_QUEUE = "task-evaluation-launches"
LAUNCH_PATH_UNIT = "blueprint-task-evaluation-launch-dispatcher.path"
# task_evaluation_launch_dispatcher_ownership.DISPATCHER_LOCK_NAME: shared by every live dispatcher.
DISPATCHER_LOCK = ".dispatcher.lock"
PROVIDER_LOCKS = "provider-locks"
# vast_launch_slots.vast_launch_gate_path: a deploy takes it, a launch only while taking a slot.
GATE_SUFFIX = ".gate.lock"
SCREENING_ROOT = "beta-policy-screening"
SCREENING_LEDGER = "screening_ledger.v1.json"
SCREENING_LIVE = ("launching", "running")
LAUNCH_STATE = "launch_progression.json"
# Checks this close together are one hold, so a stage handing over to the next
# never restarts the clock. A parked deploy rechecks every RECHECK_SECONDS.
CONTINUITY_SECONDS = 3 * RECHECK_SECONDS
_IDENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_LAUNCH_FILE = re.compile(r"(.+)-[0-9a-f]{16}\.json")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECORD_BYTES:
                return None
            document = json.loads(stream.read(MAX_RECORD_BYTES + 1))
    except (OSError, ValueError, RecursionError):
        return None
    return document if isinstance(document, dict) else None


def lock_holders(proc_locks: str = "/proc/locks") -> dict[int, set[int]]:
    """Inodes with a held flock, and the pids holding them. Waiters (``->``) hold nothing.

    Matched on the inode alone: /proc/locks names the device as the kernel sees it,
    which differs from ``st_dev`` on some filesystems. A collision with another
    filesystem's lock can only hold a deploy, and only up to the cap.
    """

    holders: dict[int, set[int]] = {}
    try:
        lines = Path(proc_locks).read_text(encoding="ascii", errors="replace").splitlines()
    except OSError:
        return holders
    for line in lines:
        fields = line.split()
        if len(fields) < 6 or fields[1] == "->" or fields[1] != "FLOCK":
            continue
        parts = fields[5].split(":")
        if len(parts) == 3 and parts[2].isdigit():
            pids = holders.setdefault(int(parts[2]), set())
            if fields[4].isdigit():
                pids.add(int(fields[4]))
    return holders


def locked_inodes(proc_locks: str = "/proc/locks") -> set[int]:
    """Inodes with a held flock (``lock_holders`` without the pids)."""

    return set(lock_holders(proc_locks))


def activation_path_held(holds_dir: Path | None, unit: str = ACTIVATION_PATH_UNIT) -> bool:
    """True while a door hold keeps a queue's path unit from claiming work.

    The unit's ConditionPathExists=! names this record, so an active record is what
    stops it. Unknown counts as not held: pending work then holds the deploy.
    """

    if holds_dir is None:
        return False
    record = _read_json(holds_dir / f"{unit}.json")
    return bool(record) and record.get("status") == "active"


def _bound_request(envelope: dict[str, Any] | None, active: str) -> dict[str, Any] | None:
    request = (envelope or {}).get("request")
    if (isinstance(request, dict) and request.get("lane") == LANE
            and request.get("expected_production_commit") == active):
        return request
    return None


def _json_files(directory: Path, *, modified_since: float | None = None) -> Iterable[Path]:
    """Regular ``*.json`` files, newest first, at most ``MAX_SCAN`` of them.

    The bound applies after the cheap filters (a regular file, modified since
    ``modified_since``), so old records cannot crowd a recent one out of the scan.
    """

    found: list[tuple[float, str, Path]] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                if modified_since is not None and info.st_mtime < modified_since:
                    continue
                found.append((info.st_mtime, entry.name, Path(entry.path)))
    except OSError:
        return []
    found.sort(key=lambda row: (-row[0], row[1]))
    return [path for _, _, path in found[:MAX_SCAN]]


def live_activations(
    *, queue_root: Path, progression_root: Path, active: str, now: float, held: set[int],
    pending_claimable: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The live activations, and notes on the ones that do not count."""

    live: list[dict[str, Any]] = []
    notes: list[str] = []
    for path in _json_files(queue_root / "pending"):
        request = _bound_request(_read_json(path), active)
        if request is None:
            continue
        if pending_claimable:
            live.append(_row(STAGE_ACTIVATION, "pending", request.get("activation_id")))
        else:
            notes.append(f"pending_while_activation_path_held:{path.name}")
    for path in _json_files(queue_root / "processing"):
        request = _bound_request(_read_json(path), active)
        if request is None:
            continue
        try:
            inode = path.lstat().st_ino
        except OSError:
            continue
        if inode in held:
            live.append(_row(STAGE_ACTIVATION, "processing", request.get("activation_id")))
        else:
            notes.append(f"stuck_claim_without_lock:{path.name}")
    for path in _json_files(queue_root / "results", modified_since=now - SUBMIT_WINDOW_SECONDS):
        result = _read_json(path)
        if (not result or result.get("status") != MATERIALIZED_STATUS or result.get("lane") != LANE
                or result.get("source_commit") != active):
            continue
        preparation = str(result.get("preparation_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}", preparation) or preparation in {".", ".."}:
            continue
        launch = _read_json(progression_root / "scene-configuration-activations" / preparation
                            / "launch_progression.json")
        if launch and launch.get("status") == LAUNCH_QUEUED_STATUS:
            continue
        live.append(_row(STAGE_ACTIVATION, "awaiting_launch_submission", result.get("activation_id")))
    return live, notes


def _row(stage: str, state: str, identifier: Any, activation_id: Any = None) -> dict[str, Any]:
    """One live row. An activation's (and a launch's) keeps ``activation_id``, as before."""

    row: dict[str, Any] = {"stage": stage, "state": state, "id": str(identifier)}
    if stage == STAGE_ACTIVATION:
        row["activation_id"] = identifier
    elif activation_id is not None:
        row["activation_id"] = activation_id
    return row


def _launch_states(directory: Path) -> list[Path]:
    """``<preparation>/launch_progression.json`` files, newest first, at most ``MAX_SCAN``."""

    found: list[tuple[float, str, Path]] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    path = Path(entry.path) / LAUNCH_STATE
                    info = path.lstat()
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    found.append((info.st_mtime, entry.name, path))
    except OSError:
        return []
    found.sort(key=lambda row: (-row[0], row[1]))
    return [path for _, _, path in found[:MAX_SCAN]]


def live_launches(
    *, launch_queue_root: Path, progression_root: Path, active: str, held: set[int],
    pending_claimable: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Scene-configuration launches submitted for this release and not yet finished.

    A request leaves pending/ and processing/ for completed/ or blocked/ when its
    dispatch ends, so only those two count. A claim counts while some dispatcher
    holds the queue's lock; a claim nobody owns is the reconciler's, not live work.
    """

    queued = {state: _json_files(launch_queue_root / state) for state in ("pending", "processing")}
    if not any(queued.values()):
        return [], []
    submitted: dict[str, Any] = {}
    for path in _launch_states(progression_root / "scene-configuration-activations"):
        state = _read_json(path)
        if (state and state.get("status") == LAUNCH_QUEUED_STATUS
                and state.get("expected_production_commit") == active
                and isinstance(state.get("launch_id"), str) and _IDENT.fullmatch(state["launch_id"])):
            submitted[state["launch_id"]] = state.get("activation_id")
    live: list[dict[str, Any]] = []
    notes: list[str] = []
    owned: bool | None = None
    for state_name, paths in queued.items():
        for path in paths:
            match = _LAUNCH_FILE.fullmatch(path.name)
            request = _read_json(path)
            launch_id = (request or {}).get("launch_id")
            if match is None or launch_id != match.group(1) or launch_id not in submitted:
                continue
            if state_name == "pending" and not pending_claimable:
                notes.append(f"launch_pending_while_dispatcher_path_held:{path.name}")
                continue
            if state_name == "processing":
                if owned is None:
                    try:
                        owned = (launch_queue_root / DISPATCHER_LOCK).lstat().st_ino in held
                    except OSError:
                        owned = False
                if not owned:
                    notes.append(f"launch_claim_without_dispatcher:{path.name}")
                    continue
            live.append(_row(STAGE_LAUNCH, f"launch_{state_name}", launch_id, submitted[launch_id]))
    return live, notes


def process_release(pid: int, proc_root: Path) -> str:
    """The release a process runs from: the last 40-hex component of its working directory.

    Units ``cd -P`` into the active release link, so this is the release tree they
    started in. Unreadable or unnamed is "" (unknown), which holds nothing.
    """

    try:
        cwd = os.readlink(proc_root / str(pid) / "cwd")
    except OSError:
        return ""
    names = [part for part in Path(cwd).parts if _COMMIT.fullmatch(part)]
    return names[-1] if names else ""


def live_paid_locks(
    *, lock_dir: Path, active: str, holders: dict[int, set[int]], proc_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Paid-launch slot locks held by a process running this release.

    The slot is a paid run's for its whole life (vast_launch_slots). The gate beside
    the slots is skipped: a deploy holds it, and a launch only while taking a slot.
    """

    live: list[dict[str, Any]] = []
    notes: list[str] = []
    try:
        with os.scandir(lock_dir) as entries:
            locks = sorted(entry.name for entry in entries
                           if entry.name.endswith(".lock") and not entry.name.endswith(GATE_SUFFIX))
    except OSError:
        return [], []
    for name in locks[:MAX_SCAN]:
        try:
            info = (lock_dir / name).lstat()
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_ino not in holders:
            continue
        releases = {process_release(pid, proc_root) for pid in sorted(holders[info.st_ino])[:16]}
        if active in releases:
            live.append(_row(STAGE_PAID_LOCK, "paid_launch_lock_held", name))
        else:
            notes.append(f"paid_launch_lock_held_by_other_or_unknown_release:{name}")
    return live, notes


def live_screening(*, screening_root: Path, active: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Beta-screening candidates launching or running under a plan frozen at this release."""

    ledger = _read_json(screening_root / SCREENING_LEDGER)
    campaigns = (ledger or {}).get("campaigns")
    if not isinstance(campaigns, dict):
        return [], []
    live: list[dict[str, Any]] = []
    notes: list[str] = []
    scanned = 0
    for campaign_id in sorted(campaigns, key=str):
        campaign = campaigns[campaign_id]
        rows = campaign.get("candidates") if isinstance(campaign, dict) else None
        if not isinstance(rows, dict):
            continue
        busy = sorted(str(candidate) for candidate, row in rows.items()
                      if isinstance(row, dict) and row.get("status") in SCREENING_LIVE)
        if not busy or not isinstance(campaign_id, str) or not _IDENT.fullmatch(campaign_id):
            continue
        scanned += 1
        if scanned > MAX_SCAN:
            notes.append("screening_scan_limit_reached")
            break
        plan = _read_json(screening_root / "plans" / f"{campaign_id}.json")
        frozen: dict[str, Any] = {}
        for candidate in (plan or {}).get("candidates") or []:
            if isinstance(candidate, dict) and isinstance(candidate.get("execution_spec"), dict):
                frozen[str(candidate.get("candidate_id"))] = candidate["execution_spec"].get("implementation_commit")
        for candidate_id in busy:
            if frozen.get(candidate_id) == active:
                live.append(_row(STAGE_SCREENING, f"screening_{rows[candidate_id]['status']}",
                                 f"{campaign_id}/{candidate_id}"))
            else:
                notes.append(f"screening_candidate_of_other_or_unknown_release:{campaign_id}/{candidate_id}")
    return live, notes


def _collect(name: str, reader: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """One source's rows; a source that fails holds nothing and hides no other source."""

    try:
        return reader()
    except Exception as error:  # noqa: BLE001 - unknown means deploy now
        return [], [f"{name}_unreadable:{type(error).__name__}"]


def active_release(link: str) -> str:
    try:
        name = Path(os.readlink(link)).name
    except OSError:
        return ""
    return name if _COMMIT.fullmatch(name) else ""


def _write_hold(parked_dir: Path, document: dict[str, Any]) -> None:
    temporary = parked_dir / f".activation-hold.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
        os.replace(temporary, parked_dir / HOLD_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def _clear_hold(parked_dir: Path | None) -> None:
    if parked_dir is not None:
        try:
            (parked_dir / HOLD_FILE).unlink(missing_ok=True)
        except OSError:
            pass


def decide(
    *, queue_root: Path, progression_root: Path, active: str, parked_dir: Path | None,
    override: str, now: float, held: set[int], pending_claimable: bool = True,
    times: dict[str, float] | None = None,
    launch_queue_root: Path | None = None, launch_pending_claimable: bool = True,
    lock_dir: Path | None = None, holders: dict[int, set[int]] | None = None,
    proc_root: Path = Path("/proc"), screening_root: Path | None = None, request_id: str = "",
) -> tuple[int, str]:
    """Whether to deploy now, and the reason. Writes or clears the hold file.

    On a hold, ``times`` (when given) gets ``retry_after_epoch``, the next recheck,
    and ``cap_at_epoch``, after which the same release's hold always lets it go.
    A source left as None is not read.
    """

    if override:
        # The record stays: if this deploy fails before switching, a later one keeps the same cap.
        return EXIT_DEPLOY, f"activation_hold_overridden:{override}"
    if not active:
        # Unknown stays unknown: no release to protect is proven, so this holds nothing.
        return EXIT_DEPLOY, "activation_hold_active_release_unknown"
    live, notes = _collect("activations", lambda: live_activations(
        queue_root=queue_root, progression_root=progression_root, active=active, now=now, held=held,
        pending_claimable=pending_claimable))
    sources = []
    if launch_queue_root is not None:
        sources.append(("launches", lambda: live_launches(
            launch_queue_root=launch_queue_root, progression_root=progression_root, active=active,
            held=held, pending_claimable=launch_pending_claimable)))
    if lock_dir is not None:
        sources.append(("paid_launch_locks", lambda: live_paid_locks(
            lock_dir=lock_dir, active=active, holders=holders or {}, proc_root=proc_root)))
    if screening_root is not None:
        sources.append(("beta_screening", lambda: live_screening(screening_root=screening_root, active=active)))
    for name, reader in sources:
        rows, more = _collect(name, reader)
        live.extend(rows)
        notes.extend(more)
    for note in notes:
        print(f"activation hold: not counted: {note}", file=sys.stderr)
    if not live:
        _clear_hold(parked_dir)
        return EXIT_DEPLOY, "no_live_scene_configuration_activation"
    previous = _read_json(parked_dir / HOLD_FILE) if parked_dir is not None else None
    request_id = request_id if _DEPLOY_ID.fullmatch(request_id or "") else ""
    first = _episode_start(previous, active=active, live=live, now=now, request_id=request_id)
    names = ",".join(sorted(f"{row['state']}:{row['id']}" for row in live))
    stages = sorted({row["stage"] for row in live})
    capped = now - first >= CAP_SECONDS
    if parked_dir is not None:
        # Kept past the cap, so a retry, or a deploy after a capped one failed before
        # switching, still finds the cap spent instead of starting a new hold.
        _write_hold(parked_dir, {
            "schema": HOLD_SCHEMA, "active_release": active, "first_held_at_epoch": first,
            "cap_at_epoch": first + CAP_SECONDS, "last_checked_at_epoch": now,
            "recheck_after_epoch": now if capped else now + RECHECK_SECONDS,
            "live": live, "activation_ids": sorted({str(row["activation_id"]) for row in live
                                                    if "activation_id" in row}),
            # Added fields: which stages hold it, every live row's key, and when each
            # deploy request was first held.
            "stages": stages, "live_ids": sorted(_keys(live)),
            "requests": _requests(previous, active=active, request_id=request_id, first=first),
        })
    if capped:
        return EXIT_DEPLOY, f"activation_hold_cap_reached:{names}"
    if times is not None:
        times.update(retry_after_epoch=now + RECHECK_SECONDS, cap_at_epoch=first + CAP_SECONDS)
    if stages == [STAGE_ACTIVATION]:
        return EXIT_HOLD, f"scene_configuration_activation_live:{names}"
    return EXIT_HOLD, f"paid_stage_live:{names}"


def _keys(live: list[dict[str, Any]]) -> set[str]:
    """``<stage>:<id>`` for each row, and the bare activation id a row carries.

    A launch carries its activation's id, so an activation handing over to its
    launch shares a key with it.
    """

    keys = {f"{row['stage']}:{row['id']}" for row in live}
    return keys | {str(row["activation_id"]) for row in live if "activation_id" in row}


def _epoch(value: Any, now: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value > now:
        return None
    return float(value)


def _requests(previous: dict[str, Any] | None, *, active: str, request_id: str, first: float) -> dict[str, float]:
    """When each deploy request was first held, for this release; the newest ``PARKED_MAX``."""

    kept: dict[str, float] = {}
    if previous and previous.get("schema") == HOLD_SCHEMA and previous.get("active_release") == active:
        rows = previous.get("requests")
        for key, value in (rows.items() if isinstance(rows, dict) else ()):
            if isinstance(key, str) and _DEPLOY_ID.fullmatch(key) and _epoch(value, math.inf) is not None:
                kept[key] = float(value)
    if request_id:
        kept[request_id] = min(kept.get(request_id, first), first)
    return dict(sorted(kept.items())[-PARKED_MAX:])


def _episode_start(previous: dict[str, Any] | None, *, active: str, live: list[dict[str, Any]],
                   now: float, request_id: str = "") -> float:
    """When the hold on this work began.

    One hold lasts while the same release runs and the work it waits for continues:
    some row it saw is still live (a pending activation becoming the running one,
    or its launch, counts), or the last check was within ``CONTINUITY_SECONDS``
    (any stage handing over to another). A deploy request already held keeps its
    own first hold too. Its age alone never restarts it, so a late retry cannot
    buy a second cap. Only work that shares nothing with it after a gap, or a new
    release, starts a new hold; a check that finds nothing live ends it.
    """

    if not previous or previous.get("schema") != HOLD_SCHEMA or previous.get("active_release") != active:
        return now
    starts: list[float] = []
    rows = previous.get("requests")
    if request_id and isinstance(rows, dict) and _epoch(rows.get(request_id), now) is not None:
        starts.append(float(rows[request_id]))
    first, seen = _epoch(previous.get("first_held_at_epoch"), now), previous.get("activation_ids")
    if first is not None and isinstance(seen, list):
        ids = previous.get("live_ids")
        seen_keys = {str(item) for item in seen} | (
            {str(item) for item in ids} if isinstance(ids, list) else set())
        checked = _epoch(previous.get("last_checked_at_epoch"), now)
        if _keys(live) & seen_keys or checked is not None and now - checked <= CONTINUITY_SECONDS:
            starts.append(first)
    return min(starts) if starts else now


def _iso(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat(timespec="seconds")


def retry_line(times: dict[str, float]) -> str:
    """``<retry ISO> <retry epoch> <cap ISO> <cap epoch>``, read by door-deploy.sh."""

    retry, cap = math.ceil(times["retry_after_epoch"]), math.ceil(times["cap_at_epoch"])
    return f"{_iso(retry)} {retry} {_iso(cap)} {cap}"


def park(*, parked_dir: Path, request_id: str, commit: str, wait_for_idle: bool, active: str,
         now: float) -> int:
    """Park this direct deploy as the runner parks one asked for during another deploy."""

    if _DEPLOY_ID.fullmatch(request_id) is None or _COMMIT.fullmatch(commit) is None:
        return EXIT_CANNOT_PARK
    if not parked_dir.is_dir() or len(list(parked_dir.glob("*.json"))) >= PARKED_MAX:
        return EXIT_CANNOT_PARK
    record = {"schema": PARKED_SCHEMA, "id": request_id, "commit": commit, "wait_for_idle": wait_for_idle,
              "parked_at_epoch": now, "active_release_at_park": active, "parked_by": "activation_hold"}
    temporary = parked_dir / f".{request_id}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
        os.replace(temporary, parked_dir / f"{request_id}.json")
    finally:
        temporary.unlink(missing_ok=True)
    return EXIT_DEPLOY


def main(argv: list[str]) -> int:
    if argv[1:] == ["park"]:
        return park(parked_dir=Path(os.environ["DOOR_ACTIVATION_HOLD_PARKED_DIR"]), request_id=os.environ["DOOR_REQUEST_ID"],
                    commit=os.environ["DOOR_COMMIT"], wait_for_idle=os.environ.get("DOOR_WAIT_FOR_IDLE", "1") == "1",
                    active=active_release(os.environ["DOOR_ACTIVE_RELEASE_LINK"]), now=time.time())
    state = Path(os.environ["DOOR_STATE_ROOT"])
    override = os.environ.get("DOOR_ACTIVATION_HOLD_OVERRIDE", "")
    if override and OVERRIDE_REASON.fullmatch(override) is None:
        override = ""
    parked = os.environ.get("DOOR_ACTIVATION_HOLD_PARKED_DIR", "")
    # The door's spool keeps parked/ and holds/ side by side.
    holds_dir = Path(parked).parent / "holds" if parked else None
    holders = lock_holders()
    times: dict[str, float] = {}
    code, reason = decide(
        queue_root=state / "task-evaluation-launch-activations",
        progression_root=state / "task-evaluation-configured-controls",
        active=active_release(os.environ["DOOR_ACTIVE_RELEASE_LINK"]),
        parked_dir=Path(parked) if parked else None,
        override=override, now=time.time(), held=set(holders),
        pending_claimable=not activation_path_held(holds_dir),
        times=times,
        launch_queue_root=state / LAUNCH_QUEUE,
        launch_pending_claimable=not activation_path_held(holds_dir, LAUNCH_PATH_UNIT),
        lock_dir=state / PROVIDER_LOCKS, holders=holders,
        screening_root=state / SCREENING_ROOT,
        request_id=os.environ.get("DOOR_REQUEST_ID", ""),
    )
    print(reason)
    if code == EXIT_HOLD and times:
        print(retry_line(times))
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except (OSError, KeyError, ValueError) as error:
        if sys.argv[1:] == ["park"]:
            print(f"activation_hold_park_failed:{type(error).__name__}")
            raise SystemExit(EXIT_CANNOT_PARK) from None
        # This hold protects paid work in flight, not privacy, rights or money limits: a
        # failure to read it never blocks a deploy.
        print(f"activation_hold_check_failed:{type(error).__name__}")
        raise SystemExit(EXIT_DEPLOY) from None
