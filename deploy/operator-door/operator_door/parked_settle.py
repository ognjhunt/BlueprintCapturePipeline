"""Decide what a parked deploy does, after door-deploy.sh has fetched main.

The runner parks requests and starts the newest one, and knows nothing about
commits: it has no network and no git, and a freshly merged commit may be in no
local clone. So this decides, from the same fetch the deploy will use.

A request is reported satisfied only once a receipt proves that what runs contains
its commit. So a deploy that also covers other parked requests leaves them parked,
and the next wake-up finds them inside the release it proved.

A deploy runs for the request that named its commit, whichever request was started.
If that commit fails, only that request fails: a red newest commit cannot take an
older good one with it, and each of the others gets its own try.

Run by door-deploy.sh with ``python3 -I -S``, so standard library only.

Exit codes:
    0  deploy the commit printed on stdout, for the request id printed after it
    2  could not decide; this request's outcome is the caller's to write
    3  nothing to deploy for this request; its outcome is already written
    4  cannot tell now, or another deploy holds the deploy lock; every record stays parked
"""

from __future__ import annotations

import dataclasses
import datetime
import fcntl
import json
import os
import re
import stat
import subprocess  # nosec B404 - fixed git argv over validated commits
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

EXIT_DEPLOY = 0
EXIT_CANNOT_DECIDE = 2
EXIT_NOTHING_TO_DEPLOY = 3
EXIT_TRY_AGAIN = 4
PARKED_SCHEMA = "blueprint_operator_door_parked_deploy.v1"
OUTCOME_SCHEMA = "blueprint_operator_door_outcome.v1"
DEPLOY_LOCK_NAME = "control-plane-deploy.lock"
MAX_RECORD_BYTES = 16 * 1024
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DEPLOY_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-deploy-[0-9a-f]{8}")


@dataclasses.dataclass(frozen=True)
class Parked:
    id: str
    commit: str
    active_at_park: str


@dataclasses.dataclass(frozen=True)
class Plan:
    #: The commit to deploy, or None when nothing deploys for this request.
    deploy: str | None
    #: Outcome fields for each request settled now. Each is already true.
    settled: dict[str, dict[str, Any]]
    #: The request that named ``deploy``. This run's outcome is that request's.
    deploy_for: str | None = None


def _superseded(*, commit: str) -> dict[str, Any]:
    return {"status": "superseded", "superseded_by_commit": commit}


def _failed(code: str, **fields: Any) -> dict[str, Any]:
    return {"status": "failed", "code": code, **fields}


def decide(
    records: Sequence[Parked],
    *,
    carrier: str,
    active: str,
    active_proven: bool,
    contains: Callable[[str, str], bool | None],
    on_main: Callable[[str], bool | None],
) -> Plan:
    """What happens to each parked request. Pure, apart from the two git questions.

    ``records`` is oldest first and ``carrier`` is the one running this. A record
    whose commit is not in the clone yet arrived after this deploy's fetch: it is
    left parked for the next deploy, unless it is the carrier's own.
    """

    settled: dict[str, dict[str, Any]] = {}
    forward: list[Parked] = []
    # Requests a forward deploy may still satisfy, with what to say if none comes.
    waiting: dict[str, dict[str, Any]] = {}
    for record in records:
        reachable = on_main(record.commit)
        if reachable is None and record.id != carrier:
            continue
        if not reachable:
            settled[record.id] = _failed("commit_not_on_main")
        elif (record.commit != record.active_at_park and record.active_at_park
              and contains(record.commit, record.active_at_park) is True):
            # Already inside what was running when it was asked for, so it asked to
            # go backwards. Timing must not decide that: refuse, as a direct request
            # during a deploy always was, instead of quietly reporting success.
            settled[record.id] = _failed("parked_rollback_refused", active_release_at_park=record.active_at_park)
        elif (record.active_at_park and record.active_at_park != active
              and contains(active, record.active_at_park) is True):
            # What runs is older than what ran when this was asked for. Someone rolled
            # back while it waited, and a request from before that must not undo it.
            settled[record.id] = _failed("parked_release_rolled_back", active_release=active,
                                         active_release_at_park=record.active_at_park)
        elif record.commit == active or contains(record.commit, active) is True:
            if active_proven:
                settled[record.id] = _superseded(commit=active)
            elif record.commit == active:
                forward.append(record)  # the link moved, but no receipt says that deploy finished
            else:
                waiting[record.id] = _failed("parked_active_release_not_proven", active_release=active)
        elif contains(active, record.commit) is True:
            forward.append(record)
        else:
            # Neither contains the other, or git could not say: deploying it could drop
            # what runs. A later main commit in the queue may still contain it.
            waiting[record.id] = _failed("parked_commit_not_contained", active_release=active)

    if carrier in settled:
        return Plan(None, settled)
    if not forward:
        return Plan(None, {**settled, **waiting})

    def covers(container: Parked, record: Parked) -> bool:
        return record.commit == container.commit or contains(record.commit, container.commit) is True

    # This request's own commit goes out if it can, inside the commit that also covers
    # the most others. Failing that, whichever forward commit covers the most. What it
    # leaves out stays parked for the next deploy.
    own = next((row for row in forward if row.id == carrier), None)
    pool = [row for row in forward if own is None or covers(row, own)]
    chosen = max(enumerate(pool), key=lambda item: (sum(covers(item[1], row) for row in forward), item[0]))[1]
    return Plan(chosen.commit, settled, carrier if own is not None and own.commit == chosen.commit else chosen.id)


def _git(clone: str, *args: str) -> int:
    return subprocess.run(  # nosec B603 B607 - fixed git argv over validated commits
        ["git", "-C", clone, *args], capture_output=True, check=False, timeout=120,
    ).returncode


def _contains(clone: str, commit: str, container: str) -> bool | None:
    """True or False from git; None when git cannot say, such as a missing object."""
    code = _git(clone, "merge-base", "--is-ancestor", commit, container)
    return True if code == 0 else False if code == 1 else None


def _read_record(path: Path) -> Parked | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o022 or not 0 < info.st_size <= MAX_RECORD_BYTES):
                return None
            document = json.loads(stream.read(MAX_RECORD_BYTES + 1))
    except (OSError, ValueError, RecursionError):
        return None
    if (not isinstance(document, dict) or document.get("schema") != PARKED_SCHEMA
            or document.get("id") != path.stem or _DEPLOY_ID.fullmatch(path.stem) is None):
        return None
    commit, active = document.get("commit"), document.get("active_release_at_park")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        return None
    if active != "" and (not isinstance(active, str) or _COMMIT.fullmatch(active) is None):
        return None
    return Parked(path.stem, commit, active)


def _active_commit(link: str, clone: str) -> str | None:
    """The commit the release link names, fetched if needed, or None if it cannot be known."""
    try:
        name = Path(os.readlink(link)).name
    except OSError:
        return None
    if _COMMIT.fullmatch(name) is None:
        return None
    if _git(clone, "cat-file", "-e", f"{name}^{{commit}}") != 0:
        _git(clone, "fetch", "--quiet", "--no-tags", "origin", name)
        if _git(clone, "cat-file", "-e", f"{name}^{{commit}}") != 0:
            return None
    return name


def _active_is_proven(state_root: str, active: str, active_link: str) -> bool:
    """Only a deploy's own receipt says it finished. The link moves before the last checks.

    An older receipt for the same commit proves nothing about a later deploy of it
    that moved the link and then died, so the receipt must be newer than the link.
    """
    try:
        linked = os.lstat(active_link).st_mtime_ns
    except OSError:
        return False
    for mode in ("iteration", "production"):
        path = Path(state_root) / "deploy-receipts" / f"{mode}_{active[:12]}_door.json"
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECEIPT_BYTES
                        or info.st_mtime_ns < linked):
                    continue
                receipt = json.loads(stream.read(MAX_RECEIPT_BYTES + 1))
        except (OSError, ValueError, RecursionError):
            continue
        if isinstance(receipt, dict) and receipt.get("status") == "deployed" and receipt.get("source_commit") == active:
            return True
    return False


def _another_deploy_running(state_root: str) -> bool:
    """A deploy outside the door's own units, such as the iteration wrapper, holds this lock."""
    try:
        descriptor = os.open(Path(state_root) / DEPLOY_LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False


def _write_outcome(results: Path, request_id: str, fields: dict[str, Any]) -> None:
    status = fields["status"]
    document = {
        "schema": OUTCOME_SCHEMA, "status": status, "code": None,
        "exit_code": 0 if status == "superseded" else 1,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **fields,
    }
    temporary = results / f".{request_id}.outcome.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
        os.replace(temporary, results / f"{request_id}.outcome.json")
    finally:
        temporary.unlink(missing_ok=True)


def _mark_deploying(parked: Path, carrier: str, commit: str) -> None:
    """A record marked this way that outlives its unit is a deploy that was killed."""
    path = parked / f"{carrier}.json"
    document = json.loads(path.read_bytes())
    document["deploying_commit"] = commit
    temporary = parked / f".{carrier}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def settle(
    *, carrier: str, parked_dir: str, results_dir: str, clone: str, active_link: str, state_root: str,
) -> tuple[int, str, str]:
    """Returns the exit code, and the commit to deploy and the request it runs for, if any."""
    if _another_deploy_running(state_root):
        return EXIT_TRY_AGAIN, "", ""
    parked, results = Path(parked_dir), Path(results_dir)
    records = [row for row in (_read_record(path) for path in sorted(parked.glob("*.json"))) if row]
    if carrier not in {row.id for row in records}:
        # A record is the only thing that authorizes a parked deploy.
        _write_outcome(results, carrier, _failed("deploy_parked_record_missing"))
        return EXIT_NOTHING_TO_DEPLOY, "", ""
    active = _active_commit(active_link, clone)
    if active is None:
        return EXIT_TRY_AGAIN, "", ""
    plan = decide(
        records, carrier=carrier, active=active,
        active_proven=_active_is_proven(state_root, active, active_link),
        contains=lambda commit, container: _contains(clone, commit, container),
        on_main=lambda commit: (None if _git(clone, "cat-file", "-e", f"{commit}^{{commit}}") != 0
                                else _contains(clone, commit, "origin/main")),
    )
    carrier_written = False
    for request_id, fields in plan.settled.items():
        try:
            _write_outcome(results, request_id, fields)
            (parked / f"{request_id}.json").unlink(missing_ok=True)
            carrier_written = carrier_written or request_id == carrier
        except OSError as error:
            # Keep this record for the next deploy, and go on to settle the rest.
            print(f"parked request {request_id} not settled: {type(error).__name__}", file=sys.stderr)
    if plan.deploy is None or plan.deploy_for is None:
        return (EXIT_NOTHING_TO_DEPLOY if carrier_written else EXIT_CANNOT_DECIDE), "", ""
    _mark_deploying(parked, plan.deploy_for, plan.deploy)
    return EXIT_DEPLOY, plan.deploy, plan.deploy_for


def main() -> int:
    code, commit, request_id = settle(
        carrier=os.environ["DOOR_REQUEST_ID"], parked_dir=os.environ["DOOR_PARKED_DIR"],
        results_dir=os.environ["DOOR_RESULTS_DIR"], clone=os.environ["DOOR_SOURCE_CLONE"],
        active_link=os.environ["DOOR_ACTIVE_RELEASE_LINK"], state_root=os.environ["DOOR_STATE_ROOT"],
    )
    if commit:
        print(commit, request_id)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.SubprocessError) as error:
        # A full disk or a hung git says nothing about any request.
        print(f"parked settle failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(EXIT_TRY_AGAIN) from None
    except (KeyError, ValueError) as error:
        print(f"parked settle failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(EXIT_CANNOT_DECIDE) from None
