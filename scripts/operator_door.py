#!/usr/bin/env python3
"""Client for the control-plane operator door (deploy/operator-door).

Cloud agent sessions cannot SSH to the host; this is how they read run state
and ask for the few privileged operations the door allows. It works from a
laptop too. Standard library only.

    python3 scripts/operator_door.py status
    python3 scripts/operator_door.py usage     # what uses the disk, from the capacity survey
    python3 scripts/operator_door.py ls  /var/lib/blueprint/pipeline-control-plane/deploy-receipts --sort mtime
    python3 scripts/operator_door.py cat /var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/latest.json
    python3 scripts/operator_door.py pull <host path> <local path>    # file, or directory as an archive
    python3 scripts/operator_door.py journal blueprint-task-evaluation-scene-progression.service -n 200
    python3 scripts/operator_door.py deploy <sha on main> --wait
    python3 scripts/operator_door.py unit start blueprint-pubsub-handoff-listener.timer
    python3 scripts/operator_door.py hold blueprint-scene-progression.timer --owner alice --reason "inspection" --for 2h
    python3 scripts/operator_door.py release-hold blueprint-scene-progression.timer
    python3 scripts/operator_door.py retire-scene-workspace <scene_id> [--bucket B] [--apply] --wait

Authentication: in a cloud session the egress proxy adds the bearer token, so
nothing is configured in the VM and no header is sent. Elsewhere the token
comes from BLUEPRINT_OPERATOR_DOOR_TOKEN, BLUEPRINT_OPERATOR_DOOR_TOKEN_FILE,
or ~/.blueprint-secrets/operator_door_token. The token is never printed.

Exit codes: 0 success; 1 a waited-for request did not succeed, or `usage` found
no usage survey; 2 the door refused (the rule is printed on stderr); 3
unauthorized or missing scope; 4 network error; 5 server error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "https://paperclip.tryblueprint.io/api/live-pipeline/operator/v1"
DEFAULT_TOKEN_FILE = "~/.blueprint-secrets/operator_door_token"
MAX_CHECKED_PULL_BYTES = 16 * 1024 * 1024
CHECKED_PULL_CHUNK_BYTES = 1024 * 1024
MAX_CHECKED_PULL_REQUESTS = 4096
MAX_CHECKED_HTTP_ERROR_BYTES = 4096
# A retirement that planned or retired succeeded; "retained" (the scene did not qualify) exits 1
# and the printed outcome carries the first reason.
_TERMINAL_OK = {"deployed", "upgraded", "planned", "retired", "restored", "listed", "renewed", "released"}


class DoorError(Exception):
    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def base_url() -> str:
    return os.environ.get("BLUEPRINT_OPERATOR_DOOR_URL", DEFAULT_URL).rstrip("/")


def auth_headers() -> dict[str, str]:
    token = os.environ.get("BLUEPRINT_OPERATOR_DOOR_TOKEN", "").strip()
    if not token:
        path = Path(os.environ.get("BLUEPRINT_OPERATOR_DOOR_TOKEN_FILE", DEFAULT_TOKEN_FILE)).expanduser()
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(method: str, route: str, query: dict[str, Any] | None = None, body: Any = None,
             *, checked_error_max_bytes: int | None = None):
    url = base_url() + route
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=auth_headers())
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        return urllib.request.urlopen(request, timeout=120)  # nosec B310 - operator-configured URL
    except urllib.error.HTTPError as error:
        if checked_error_max_bytes is not None:
            code = ""
            try:
                payload = _bounded_body(error, checked_error_max_bytes + 1)
                if len(payload) <= checked_error_max_bytes:
                    document = json.loads(payload or b"{}", parse_constant=_invalid_json_constant)
                    code = document.get("error", "") if isinstance(document, dict) else ""
            except (DoorError, ValueError, RecursionError, OSError, http.client.HTTPException):
                pass
            finally:
                try:
                    error.close()
                except OSError:
                    pass
            exit_code = (3 if error.code == 401 or (error.code == 403 and isinstance(code, str)
                         and code.startswith("scope_missing")) else 5 if error.code >= 500 else 2)
            raise _checked_remote_error(exit_code) from error
        try:
            code = json.loads(error.read() or b"{}").get("error", "")
        except (ValueError, AttributeError):
            code = ""
        if error.code == 401 or (error.code == 403 and str(code).startswith("scope_missing")):
            raise DoorError(3, f"door refused authorization: {code or error.code}") from error
        if error.code >= 500:
            raise DoorError(5, f"door error {error.code}: {code}") from error
        raise DoorError(2, f"refused: {code or error.code}") from error
    except (urllib.error.URLError, OSError) as error:
        if checked_error_max_bytes is not None:
            raise _checked_remote_error(4) from error
        raise DoorError(4, f"cannot reach {base_url()}: {getattr(error, 'reason', error)}") from error


def _json(method: str, route: str, query: dict[str, Any] | None = None, body: Any = None) -> Any:
    with _request(method, route, query, body) as response:
        return json.loads(response.read())


def _print(document: Any) -> None:
    print(json.dumps(document, indent=2, sort_keys=True))


def _pull_file(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    partial = local.with_name(local.name + ".partial")
    offset = 0
    with partial.open("wb") as stream:
        while True:
            with _request("GET", "/fs/read", {"path": remote, "offset": offset}) as response:
                data = response.read()
                eof = response.headers.get("X-Door-Eof") == "true"
            stream.write(data)
            offset += len(data)
            if eof or not data:
                break
    partial.replace(local)



def _invalid_json_constant(_value):
    raise ValueError("nonfinite JSON")


def _checked_remote_error(exit_code: int) -> DoorError:
    messages = {2: "checked_pull_remote_refused", 3: "checked_pull_unauthorized",
                4: "checked_pull_network_failed", 5: "checked_pull_server_failed"}
    return DoorError(exit_code, messages.get(exit_code, "checked_pull_remote_refused"))


def _bounded_body(response, limit: int) -> bytes:
    pieces = []
    remaining = limit
    while remaining:
        part = response.read(remaining)
        if not isinstance(part, bytes) or len(part) > remaining:
            raise DoorError(2, "checked_pull_response_invalid")
        if not part:
            break
        pieces.append(part)
        remaining -= len(part)
    return b"".join(pieces)


def _checked_pull_options(digest: Any, size: Any) -> tuple[str, int]:
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise DoorError(2, "checked_pull_options_invalid")
    if isinstance(size, str):
        if (len(size) > len(str(MAX_CHECKED_PULL_BYTES))
                or re.fullmatch(r"0|[1-9][0-9]*", size) is None):
            raise DoorError(2, "checked_pull_options_invalid")
        size = int(size)
    if type(size) is not int or not 0 <= size <= MAX_CHECKED_PULL_BYTES:
        raise DoorError(2, "checked_pull_options_invalid")
    return digest, size


def _checked_header(headers, name: str) -> int:
    value = headers.get(name)
    if (not isinstance(value, str) or len(value) > len(str(MAX_CHECKED_PULL_BYTES))
            or re.fullmatch(r"0|[1-9][0-9]*", value) is None):
        raise DoorError(2, "checked_pull_response_invalid")
    number = int(value)
    if number > MAX_CHECKED_PULL_BYTES:
        raise DoorError(2, "checked_pull_response_invalid")
    return number


def _checked_chunk(remote: str, offset: int, length: int, expected_size: int) -> bytes:
    try:
        with _request("GET", "/fs/read", {"path": remote, "offset": offset, "length": length},
                      checked_error_max_bytes=MAX_CHECKED_HTTP_ERROR_BYTES) as response:
            size = _checked_header(response.headers, "X-Door-Size")
            start = _checked_header(response.headers, "X-Door-Offset")
            declared = _checked_header(response.headers, "X-Door-Length")
            eof = response.headers.get("X-Door-Eof")
            if size != expected_size or start != offset or declared > length:
                raise DoorError(2, "checked_pull_response_invalid")
            data = _bounded_body(response, length + 1)
            if (len(data) != declared or eof not in ("true", "false")
                    or (eof == "true") != (offset + len(data) == expected_size)
                    or (length > 0 and not data)):
                raise DoorError(2, "checked_pull_response_invalid")
            return data
    except DoorError as error:
        if str(error) == "checked_pull_response_invalid":
            raise
        raise _checked_remote_error(error.exit_code) from error
    except (urllib.error.URLError, OSError) as error:
        raise _checked_remote_error(4) from error
    except (http.client.HTTPException, AttributeError, TypeError, ValueError) as error:
        raise DoorError(2, "checked_pull_response_invalid") from error


def _pull_checked_file(remote: str, local: Path, *, expected_sha256: Any, expected_size: Any) -> dict[str, Any]:
    expected_sha256, expected_size = _checked_pull_options(expected_sha256, expected_size)
    temporary = None
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{local.name}.", suffix=".partial", dir=local.parent)
        temporary = Path(name)
        digest = hashlib.sha256()
        offset = 0
        requests = 0
        try:
            stream = os.fdopen(descriptor, "wb")
        except OSError:
            os.close(descriptor)
            raise
        with stream:
            while requests == 0 or offset < expected_size:
                if requests >= MAX_CHECKED_PULL_REQUESTS:
                    raise DoorError(2, "checked_pull_request_limit")
                length = min(CHECKED_PULL_CHUNK_BYTES, expected_size - offset)
                data = _checked_chunk(remote, offset, length, expected_size)
                stream.write(data)
                digest.update(data)
                offset += len(data)
                requests += 1
            if "sha256:" + digest.hexdigest() != expected_sha256:
                raise DoorError(2, "checked_pull_identity_mismatch")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, local)
        temporary = None
        return {"path": remote, "saved": str(local), "bytes": offset,
                "verified_digest": expected_sha256, "verified_bytes": offset}
    except OSError as error:
        raise DoorError(2, "checked_pull_local_write_failed") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                raise DoorError(2, "checked_pull_local_write_failed") from error


def _safe_members(archive: tarfile.TarFile, destination: Path):
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not (member.isfile() or member.isdir()) or (target != root and root not in target.parents):
            continue  # the door only sends regular files; anything else is ignored
        yield member


def _pull_directory(remote: str, local: Path) -> dict[str, Any]:
    local.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as buffer:
        with _request("GET", "/fs/archive", {"path": remote}) as response:
            shutil.copyfileobj(response, buffer)
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as archive:
            if ".operator-door-manifest.json" not in archive.getnames():
                raise DoorError(5, "archive incomplete: the door's manifest is missing (stream cut short)")
            archive.extractall(local, members=list(_safe_members(archive, local)))  # nosec B202 - filtered
    manifest_path = local / ".operator-door-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.unlink()
    return manifest


def _terminal(state: dict[str, Any]) -> bool | None:
    """True/False once the request finished well/badly; None while it runs."""

    result = state.get("result") or {}
    outcome = state.get("outcome") or {}
    if outcome:
        return outcome.get("status") in _TERMINAL_OK
    status = result.get("status")
    if status == "done":
        return True
    if status in {"refused", "failed"}:
        return False
    return None


def _wait(request_id: str, *, timeout: float, poll: float) -> int:
    deadline = time.monotonic() + timeout
    last = None
    while True:
        state = _json("GET", f"/requests/{request_id}")
        summary = (state.get("state"), (state.get("result") or {}).get("status"),
                   (state.get("outcome") or {}).get("status"))
        if summary != last:
            print(f"{request_id}: state={summary[0]} result={summary[1]} outcome={summary[2]}", file=sys.stderr)
            last = summary
        verdict = _terminal(state)
        if verdict is not None:
            _print(state)
            return 0 if verdict else 1
        if time.monotonic() >= deadline:
            _print(state)
            print(f"gave up waiting after {timeout:.0f}s; the request keeps running on the host", file=sys.stderr)
            return 1
        time.sleep(poll)


def _submit(body: dict[str, Any], args: argparse.Namespace) -> int:
    accepted = _json("POST", "/requests", body=body)
    if not getattr(args, "wait", False):
        _print(accepted)
        return 0
    print(f"spooled {accepted['id']}", file=sys.stderr)
    return _wait(accepted["id"], timeout=args.timeout, poll=args.poll)


def _size(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(amount) < 1024:
            return f"{amount:.0f} B" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{value * 100:.1f}%"


def _utc(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    try:
        return dt.datetime.fromtimestamp(value, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return "-"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows)]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
            for row in (headers, *rows)]


_USAGE_TABLES = (
    ("mounts", ["mount", "used", "surveyed", "classified", "attributed"],
     lambda row: [str(row.get("mount")), _size(row.get("used_bytes")), _size(row.get("surveyed_bytes")),
                  _size(row.get("classified_bytes")), _percent(row.get("attributed_fraction"))]),
    ("by_class", ["storage class", "allocated", "apparent", "files"],
     lambda row: [str(row.get("storage_class")), _size(row.get("allocated_bytes")),
                  _size(row.get("apparent_bytes")), str(row.get("files", "-"))]),
    ("top_roots", ["root", "class", "allocated"],
     lambda row: [str(row.get("root")), str(row.get("storage_class")), _size(row.get("allocated_bytes"))]),
    ("top_owners", ["owner", "class", "allocated", "root"],
     lambda row: [str(row.get("owner")), str(row.get("storage_class")), _size(row.get("allocated_bytes")),
                  str(row.get("root"))]),
    ("unclassified_roots", ["unclassified root", "allocated"],
     lambda row: [str(row.get("root")), _size(row.get("allocated_bytes"))]),
    ("orphan_scratch_roots", ["unowned scratch", "allocated", "newest change"],
     lambda row: [str(row.get("root")), _size(row.get("allocated_bytes")),
                  _utc(row.get("newest_mtime_epoch"))]),
)


def _print_usage(status: dict[str, Any]) -> int:
    """Print ``capacity.usage`` from a status document as tables."""

    capacity = status.get("capacity")
    usage = capacity.get("usage") if isinstance(capacity, dict) else None
    if not isinstance(usage, dict) or usage.get("status") in (None, "unavailable"):
        reason = (capacity or {}).get("error") if isinstance(capacity, dict) else None
        reason = reason or (usage or {}).get("error") or "usage_survey_unavailable"
        print(f"no usage survey: {reason}", file=sys.stderr)
        return 1
    header = f"usage survey: {usage.get('status')}"
    age = usage.get("age_seconds")
    if isinstance(age, (int, float)) and not isinstance(age, bool):
        header += f", {age:.0f} s old"
    lines = [header]
    if usage.get("error"):
        lines.append(f"last survey attempt: {usage['error']}")
    if isinstance(usage.get("orphan_scratch_bytes"), int) and isinstance(usage.get("orphan_scratch_count"), int):
        lines.append(f"unowned scratch: {_size(usage['orphan_scratch_bytes'])} "
                     f"in {usage['orphan_scratch_count']} folders")
    for key, headers, cells in _USAGE_TABLES:
        rows = [cells(row) for row in usage.get(key) or [] if isinstance(row, dict)]
        if rows:
            lines += ["", *_table(headers, rows)]
    print("\n".join(lines))
    return 0


def _add_wait(parser: argparse.ArgumentParser, timeout: float) -> None:
    parser.add_argument("--wait", action="store_true", help="poll until the request finishes")
    parser.add_argument("--timeout", type=float, default=timeout)
    parser.add_argument("--poll", type=float, default=15.0)


def _hold_duration(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([hms])", value)
    if match is None:
        raise argparse.ArgumentTypeError("hold duration must be 2h, 90m, or 3600s")
    seconds = int(match.group(1)) * {"h": 3600, "m": 60, "s": 1}[match.group(2)]
    if not 60 <= seconds <= 86400:
        raise argparse.ArgumentTypeError("hold duration must be between 60s and 24h")
    return seconds


def _scratch_duration(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([dhms])", value)
    if match is None:
        raise argparse.ArgumentTypeError("lease duration must be 2d, 12h, 90m, or 3600s")
    seconds = int(match.group(1)) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[match.group(2)]
    if not 0 < seconds <= 14 * 86400:
        raise argparse.ArgumentTypeError("lease duration must be at most 14 days")
    return seconds


def build_parser(*, checked_mode: bool = False) -> argparse.ArgumentParser:
    class Parser(argparse.ArgumentParser):
        def error(self, message: str) -> None:
            if checked_mode:
                raise DoorError(2, "checked_pull_options_invalid")
            super().error(message)

    parser = Parser(prog="operator_door.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("whoami")
    commands.add_parser("status")
    commands.add_parser("usage", help="what uses the disk: the capacity survey from status, as tables")
    ls = commands.add_parser("ls")
    ls.add_argument("path")
    ls.add_argument("--sort", choices=("name", "mtime"), default="name")
    ls.add_argument("--match")
    cat = commands.add_parser("cat")
    cat.add_argument("path")
    cat.add_argument("--offset", type=int, default=0)
    cat.add_argument("--length", type=int)
    pull = commands.add_parser("pull")
    pull.add_argument("path")
    pull.add_argument("destination")
    pull.add_argument("--expected-sha256", help="trusted publication sha256: digest; requires expected size")
    pull.add_argument("--expected-size", help="trusted canonical byte count, at most 16 MiB; file-only")
    journal = commands.add_parser("journal")
    journal.add_argument("unit")
    journal.add_argument("-n", "--lines", type=int, default=200)
    journal.add_argument("--since")
    units = commands.add_parser("units")
    units.add_argument("--pattern", default="blueprint-*")
    units.add_argument("--state")
    show = commands.add_parser("show")
    show.add_argument("units", nargs="+")
    unit = commands.add_parser("unit")
    unit.add_argument("action", choices=("start", "reset-failed", "stop", "restart"))
    unit.add_argument("unit")
    hold = commands.add_parser("hold", help="pause a timer or path with an owner and automatic expiry")
    hold.add_argument("unit")
    hold.add_argument("--owner", required=True)
    hold.add_argument("--reason", required=True)
    hold.add_argument("--for", dest="expires_in_seconds", type=_hold_duration, required=True)
    _add_wait(hold, 120)
    release = commands.add_parser("release-hold", help="release an owned timer or path hold early")
    release.add_argument("unit")
    _add_wait(release, 120)
    deploy = commands.add_parser("deploy", help="deploy a commit that is on origin/main")
    deploy.add_argument("commit")
    deploy.add_argument("--no-wait-for-idle", action="store_true")
    _add_wait(deploy, 3 * 3600)
    upgrade = commands.add_parser("upgrade-door")
    upgrade.add_argument("commit")
    _add_wait(upgrade, 1800)
    retire = commands.add_parser(
        "retire-scene-workspace",
        help="plan a finished website scene workspace's retirement; --apply retires it behind a receipt",
    )
    retire.add_argument("scene_id")
    retire.add_argument("--bucket")
    retire.add_argument("--apply", action="store_true")
    _add_wait(retire, 2 * 3600 + 600)
    restore = commands.add_parser("restore-scene-workspace", help="restore a retired scene at its canonical path")
    restore.add_argument("scene_id")
    restore.add_argument("--bucket", required=True)
    _add_wait(restore, 2 * 3600 + 600)
    scratch = commands.add_parser("lane-scratch", help="inspect or end an owned lane scratch lease")
    scratch_actions = scratch.add_subparsers(dest="scratch_action", required=True)
    for action in ("ls", "renew", "release"):
        sub = scratch_actions.add_parser(action)
        sub.add_argument("lane")
        if action != "ls":
            sub.add_argument("name")
            sub.add_argument("--owner", required=True)
            sub.add_argument("--digest", required=True)
        sub.add_argument("--root", choices=("work", "inputs"), required=True)
        if action == "ls":
            sub.add_argument("--limit", type=int, default=50)
            sub.add_argument("--offset", type=int, default=0)
        elif action == "renew":
            sub.add_argument("--for", dest="ttl_seconds", type=_scratch_duration, required=True)
        _add_wait(sub, 120)
    request = commands.add_parser("request")
    request.add_argument("id")
    _add_wait(request, 3 * 3600)
    commands.add_parser("requests")
    return parser


def run(args: argparse.Namespace) -> int:
    command = args.command
    if command in ("whoami", "status", "requests"):
        _print(_json("GET", "/" + command))
    elif command == "usage":
        return _print_usage(_json("GET", "/status"))
    elif command == "ls":
        _print(_json("GET", "/fs/list", {"path": args.path, "sort": args.sort, "match": args.match}))
    elif command == "cat":
        with _request("GET", "/fs/read", {"path": args.path, "offset": args.offset, "length": args.length}) as r:
            data = r.read()
        sys.stdout.write(data.decode("utf-8", "replace"))
        sys.stdout.flush()
    elif command == "pull":
        if args.expected_sha256 is not None or args.expected_size is not None:
            _print(_pull_checked_file(args.path, Path(args.destination),
                                      expected_sha256=args.expected_sha256,
                                      expected_size=args.expected_size))
            return 0
        listing = _json("GET", "/fs/list", {"path": args.path})
        destination = Path(args.destination)
        if listing.get("type") == "dir":
            _print(_pull_directory(args.path, destination))
        else:
            _pull_file(args.path, destination)
            _print({"path": args.path, "saved": str(destination), "bytes": destination.stat().st_size})
    elif command == "journal":
        with _request("GET", "/journal", {"unit": args.unit, "lines": args.lines, "since": args.since}) as r:
            sys.stdout.write(r.read().decode("utf-8", "replace"))
    elif command == "units":
        _print(_json("GET", "/units", {"pattern": args.pattern, "state": args.state}))
    elif command == "show":
        _print(_json("GET", "/units/show", {"unit": ",".join(args.units)}))
    elif command == "unit":
        return _submit({"kind": "unit", "unit": args.unit, "action": args.action}, args)
    elif command == "hold":
        return _submit({"kind": "hold", "unit": args.unit, "owner": args.owner, "reason": args.reason,
                        "expires_in_seconds": args.expires_in_seconds}, args)
    elif command == "release-hold":
        return _submit({"kind": "release-hold", "unit": args.unit}, args)
    elif command == "deploy":
        return _submit({"kind": "deploy", "commit": args.commit,
                        "wait_for_idle": not args.no_wait_for_idle}, args)
    elif command == "upgrade-door":
        return _submit({"kind": "door-upgrade", "commit": args.commit}, args)
    elif command == "retire-scene-workspace":
        body = {"kind": "retire-scene-workspace", "scene_id": args.scene_id, "apply": args.apply}
        if args.bucket:
            body["bucket"] = args.bucket
        return _submit(body, args)
    elif command == "restore-scene-workspace":
        return _submit({"kind": "restore-scene-workspace", "scene_id": args.scene_id,
                        "bucket": args.bucket}, args)
    elif command == "lane-scratch":
        body = {"kind": "lane-scratch", "action": args.scratch_action, "root": args.root, "lane": args.lane}
        if args.scratch_action == "ls":
            body.update(limit=args.limit, offset=args.offset)
        else:
            body.update(name=args.name, owner=args.owner, expected_digest=args.digest)
            if args.scratch_action == "renew":
                body["ttl_seconds"] = args.ttl_seconds
        return _submit(body, args)
    elif command == "request":
        if args.wait:
            return _wait(args.id, timeout=args.timeout, poll=args.poll)
        _print(_json("GET", f"/requests/{args.id}"))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    checked_intent = any(option.startswith(token.split("=", 1)[0])
                         for token in arguments if token.startswith("--") and len(token) > 2
                         for option in ("--expected-sha256", "--expected-size"))
    try:
        args = build_parser(checked_mode=checked_intent).parse_args(arguments)
        return run(args)
    except DoorError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
