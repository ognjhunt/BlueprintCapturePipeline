#!/usr/bin/env python3
"""Client for the control-plane operator door (deploy/operator-door).

Cloud agent sessions cannot SSH to the host; this is how they read run state
and ask for the few privileged operations the door allows. It works from a
laptop too. Standard library only.

    python3 scripts/operator_door.py status
    python3 scripts/operator_door.py ls  /var/lib/blueprint/pipeline-control-plane/deploy-receipts --sort mtime
    python3 scripts/operator_door.py cat /var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/latest.json
    python3 scripts/operator_door.py pull <host path> <local path>    # file, or directory as an archive
    python3 scripts/operator_door.py journal blueprint-task-evaluation-scene-progression.service -n 200
    python3 scripts/operator_door.py deploy <sha> [--mode canary] --wait
    python3 scripts/operator_door.py replay --child sam31-<digest> --commit <sha> --wait
    python3 scripts/operator_door.py unit start blueprint-pubsub-handoff-listener.timer

Authentication: in a cloud session the egress proxy adds the bearer token, so
nothing is configured in the VM and no header is sent. Elsewhere the token
comes from BLUEPRINT_OPERATOR_DOOR_TOKEN, BLUEPRINT_OPERATOR_DOOR_TOKEN_FILE,
or ~/.blueprint-secrets/operator_door_token. The token is never printed.

Exit codes: 0 success; 1 a waited-for request did not succeed; 2 the door
refused (the rule is printed on stderr); 3 unauthorized or missing scope;
4 network error; 5 server error.
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
_TERMINAL_OK = {"deployed", "replayed", "upgraded"}


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
    deploy = commands.add_parser("deploy")
    deploy.add_argument("commit")
    deploy.add_argument("--mode", choices=("main", "canary"), default="main")
    deploy.add_argument("--no-wait-for-idle", action="store_true")
    _add_wait(deploy, 3 * 3600)
    replay = commands.add_parser("replay")
    target = replay.add_mutually_exclusive_group(required=True)
    target.add_argument("--child")
    target.add_argument("--parent")
    replay.add_argument("--commit", required=True)
    _add_wait(replay, 3600)
    upgrade = commands.add_parser("upgrade-door")
    upgrade.add_argument("commit")
    _add_wait(upgrade, 1800)
    request = commands.add_parser("request")
    request.add_argument("id")
    _add_wait(request, 3 * 3600)
    commands.add_parser("requests")
    return parser


def run(args: argparse.Namespace) -> int:
    command = args.command
    if command in ("whoami", "status", "requests"):
        _print(_json("GET", "/" + command))
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
        return _submit({"kind": "deploy", "commit": args.commit, "mode": args.mode,
                        "wait_for_idle": not args.no_wait_for_idle}, args)
    elif command == "replay":
        body = {"kind": "stage-replay", "commit": args.commit}
        body.update({"child": args.child} if args.child else {"parent": args.parent})
        return _submit(body, args)
    elif command == "upgrade-door":
        return _submit({"kind": "door-upgrade", "commit": args.commit}, args)
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
