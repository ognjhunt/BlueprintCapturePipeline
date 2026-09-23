"""Command line: ``python3 -m operator_door <command>``.

``serve``      run the HTTP door (the unprivileged systemd service)
``run-spool``  process pending requests (the root oneshot)
``self-test``  check config, tokens and state directories; exit 1 on problems
``token``      add, list or revoke token hashes (``add`` takes a hash, never a token)
``hash-token`` read a token on stdin and print its hash, so plaintext never hits argv
``caddy-patch`` write a copy of a Caddyfile with the operator route added (exit 3 if present)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .auth import SCOPES, add_token, hash_token, list_tokens, revoke_token
from .config import DoorConfigError, load_config

DEFAULT_CONFIG = "/etc/blueprint-operator-door/door.json"


def _self_test(config_path: str) -> int:
    report: dict[str, object] = {}
    try:
        config = load_config(config_path)
        report["config"] = "ok"
    except DoorConfigError as error:
        print(json.dumps({"config": str(error)}))
        return 1
    try:
        report["tokens"] = len(list_tokens(config.token_file))
    except Exception as error:  # noqa: BLE001
        report["tokens"] = f"error:{type(error).__name__}"
    report["read_roots"] = {root: os.path.isdir(root) for root in config.read_roots}
    state = Path(config.state_root)
    report["state_root_writable"] = state.is_dir() and os.access(state, os.W_OK)
    spool = Path(config.spool_root)
    report["spool_dirs"] = {
        name: (spool / name).is_dir() for name in ("pending", "processing", "completed", "results")
    }
    print(json.dumps(report, sort_keys=True))
    healthy = (
        isinstance(report["tokens"], int) and report["tokens"] > 0
        and report["state_root_writable"] and all(report["spool_dirs"].values())  # type: ignore[union-attr]
    )
    return 0 if healthy else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="operator_door")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "run-spool", "self-test", "hash-token"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default=argparse.SUPPRESS)
    caddy = commands.add_parser("caddy-patch")
    caddy.add_argument("source")
    caddy.add_argument("target")
    token = commands.add_parser("token")
    token.add_argument("--config", default=argparse.SUPPRESS)
    token_commands = token.add_subparsers(dest="token_command", required=True)
    add = token_commands.add_parser("add")
    add.add_argument("--name", required=True)
    add.add_argument("--sha256", required=True, help="sha256:<hex> of the token; never the token")
    add.add_argument("--scopes", required=True, help=f"comma-separated subset of {sorted(SCOPES)}")
    token_commands.add_parser("list")
    revoke = token_commands.add_parser("revoke")
    revoke.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    if args.command == "hash-token":
        value = sys.stdin.read().strip()
        if not value:
            print("empty token on stdin", file=sys.stderr)
            return 2
        print(hash_token(value))
        return 0
    if args.command == "self-test":
        return _self_test(args.config)
    if args.command == "caddy-patch":
        from .caddy import patch_caddyfile

        patched = patch_caddyfile(Path(args.source).read_text(encoding="utf-8"))
        if patched is None:
            return 3
        Path(args.target).write_text(patched, encoding="utf-8")
        return 0
    config = load_config(args.config)
    if args.command == "serve":
        from .hostinfo import HostInfo
        from .server import make_server

        server = make_server(config, host=HostInfo(config))
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return 0
    if args.command == "run-spool":
        from .runner import process_spool

        return process_spool(config)
    if args.token_command == "add":
        add_token(config.token_file, name=args.name, sha256=args.sha256,
                  scopes=[scope.strip() for scope in args.scopes.split(",") if scope.strip()])
    elif args.token_command == "revoke":
        revoke_token(config.token_file, name=args.name)
    print(json.dumps(list_tokens(config.token_file), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
