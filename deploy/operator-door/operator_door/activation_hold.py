"""Hold a deploy while a scene-configuration activation is live on the running release.

Every deploy makes scene progression retire the in-flight attempt and rebuild it
(about 15 min of preparation, then about an hour of activation), because each
later stage refuses work bound to an older release. So a deploy that lands while
an activation runs throws away work that was already paid for. This holds the
deploy, never the job: the activation keeps going, and the deploy goes out as
soon as it ends, when the hold reaches its cap, or at once with an override.

Live means bound to the running release and in the scene-configuration lane, and:

- pending in the activation queue, while the queue's path unit can claim it (a
  door hold on that unit means nothing will, so it holds nothing), or
- claimed in processing/ with the worker's flock still held (a claim whose
  worker died holds no lock and does not count), or
- finished with its profile authority materialized, but its launch not yet
  submitted, and finished recently (``SUBMIT_WINDOW_SECONDS``).

Run by door-deploy.sh with ``python3 -I -S``, so standard library only. The
runner cannot do this itself: it has no capabilities and these files are the
``blueprint`` user's own.

Exit codes:
    0  deploy now (nothing live, override given, cap reached, or unknown)
    5  an activation is live; stay parked (the hold file says until when to recheck).
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


def locked_inodes(proc_locks: str = "/proc/locks") -> set[int]:
    """Inodes with a held flock. Waiters (``->``) hold nothing.

    Matched on the inode alone: /proc/locks names the device as the kernel sees it,
    which differs from ``st_dev`` on some filesystems. A collision with another
    filesystem's lock can only hold a deploy, and only up to the cap.
    """

    inodes: set[int] = set()
    try:
        lines = Path(proc_locks).read_text(encoding="ascii", errors="replace").splitlines()
    except OSError:
        return inodes
    for line in lines:
        fields = line.split()
        if len(fields) < 6 or fields[1] == "->" or fields[1] != "FLOCK":
            continue
        parts = fields[5].split(":")
        if len(parts) == 3 and parts[2].isdigit():
            inodes.add(int(parts[2]))
    return inodes


def activation_path_held(holds_dir: Path | None) -> bool:
    """True while a door hold keeps the activation queue's path unit from claiming work.

    The unit's ConditionPathExists=! names this record, so an active record is what
    stops it. Unknown counts as not held: pending work then holds the deploy.
    """

    if holds_dir is None:
        return False
    record = _read_json(holds_dir / f"{ACTIVATION_PATH_UNIT}.json")
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
            live.append({"activation_id": request.get("activation_id"), "state": "pending"})
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
            live.append({"activation_id": request.get("activation_id"), "state": "processing"})
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
        live.append({"activation_id": result.get("activation_id"), "state": "awaiting_launch_submission"})
    return live, notes


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
) -> tuple[int, str]:
    """Whether to deploy now, and the reason. Writes or clears the hold file.

    On a hold, ``times`` (when given) gets ``retry_after_epoch``, the next recheck,
    and ``cap_at_epoch``, after which the same release's hold always lets it go.
    """

    if override:
        # The record stays: if this deploy fails before switching, a later one keeps the same cap.
        return EXIT_DEPLOY, f"activation_hold_overridden:{override}"
    if not active:
        # Unknown stays unknown: no release to protect is proven, so this holds nothing.
        return EXIT_DEPLOY, "activation_hold_active_release_unknown"
    live, notes = live_activations(queue_root=queue_root, progression_root=progression_root,
                                   active=active, now=now, held=held, pending_claimable=pending_claimable)
    for note in notes:
        print(f"activation hold: not counted: {note}", file=sys.stderr)
    if not live:
        _clear_hold(parked_dir)
        return EXIT_DEPLOY, "no_live_scene_configuration_activation"
    previous = _read_json(parked_dir / HOLD_FILE) if parked_dir is not None else None
    first = _episode_start(previous, active=active, live=live, now=now)
    names = ",".join(sorted(f"{row['state']}:{row['activation_id']}" for row in live))
    capped = now - first >= CAP_SECONDS
    if parked_dir is not None:
        # Kept past the cap, so a retry, or a deploy after a capped one failed before
        # switching, still finds the cap spent instead of starting a new hold.
        _write_hold(parked_dir, {
            "schema": HOLD_SCHEMA, "active_release": active, "first_held_at_epoch": first,
            "cap_at_epoch": first + CAP_SECONDS, "last_checked_at_epoch": now,
            "recheck_after_epoch": now if capped else now + RECHECK_SECONDS,
            "live": live, "activation_ids": sorted({str(row["activation_id"]) for row in live}),
        })
    if capped:
        return EXIT_DEPLOY, f"activation_hold_cap_reached:{names}"
    if times is not None:
        times.update(retry_after_epoch=now + RECHECK_SECONDS, cap_at_epoch=first + CAP_SECONDS)
    return EXIT_HOLD, f"scene_configuration_activation_live:{names}"


def _episode_start(previous: dict[str, Any] | None, *, active: str, live: list[dict[str, Any]],
                   now: float) -> float:
    """When the hold on this work began.

    One hold lasts while the same release runs and the work it waits for continues:
    some activation it saw is still live (a pending one becoming the running one
    counts). Its age alone never restarts it, so a late retry cannot buy a second
    cap. Only work that shares nothing with it, or a new release, starts a new hold;
    a check that finds nothing live ends it.
    """

    if not previous or previous.get("schema") != HOLD_SCHEMA or previous.get("active_release") != active:
        return now
    first, seen = previous.get("first_held_at_epoch"), previous.get("activation_ids")
    if (isinstance(first, bool) or not isinstance(first, (int, float)) or not math.isfinite(first)
            or first > now or not isinstance(seen, list)):
        return now
    if not {str(row["activation_id"]) for row in live} & {str(item) for item in seen}:
        return now
    return float(first)


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
    times: dict[str, float] = {}
    code, reason = decide(
        queue_root=state / "task-evaluation-launch-activations",
        progression_root=state / "task-evaluation-configured-controls",
        active=active_release(os.environ["DOOR_ACTIVE_RELEASE_LINK"]),
        parked_dir=Path(parked) if parked else None,
        override=override, now=time.time(), held=locked_inodes(),
        pending_claimable=not activation_path_held(holds_dir),
        times=times,
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
        # This hold protects pre-paid work, not privacy, rights or money limits: a
        # failure to read it never blocks a deploy.
        print(f"activation_hold_check_failed:{type(error).__name__}")
        raise SystemExit(EXIT_DEPLOY) from None
