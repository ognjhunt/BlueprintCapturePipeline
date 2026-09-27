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
import json
import os
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
# A retirement that planned or retired succeeded; "retained" (the scene did not qualify) exits 1
# and the printed outcome carries the first reason.
_TERMINAL_OK = {"deployed", "upgraded", "planned", "retired", "restored"}


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


def _request(method: str, route: str, query: dict[str, Any] | None = None, body: Any = None):
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="operator_door.py", description=__doc__,
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
    elif command == "request":
        if args.wait:
            return _wait(args.id, timeout=args.timeout, poll=args.poll)
        _print(_json("GET", f"/requests/{args.id}"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except DoorError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
