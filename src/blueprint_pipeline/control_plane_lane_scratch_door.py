"""Bounded active-release entry point for operator-door lane scratch requests."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any

from .control_plane_lane_scratch import (
    LaneScratchError,
    list_lane_scratch,
    release_lane_scratch,
    renew_lane_scratch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("ls", "renew", "release"):
        sub = commands.add_parser(action)
        sub.add_argument("--root", type=Path, required=True)
        sub.add_argument("--lane", required=True)
        sub.add_argument("--result-out", type=Path, required=True)
        if action == "ls":
            sub.add_argument("--limit", type=int, required=True)
            sub.add_argument("--offset", type=int, required=True)
        else:
            sub.add_argument("--name", required=True)
            sub.add_argument("--owner", required=True)
            sub.add_argument("--expected-digest", required=True)
            if action == "renew":
                sub.add_argument("--ttl-seconds", type=int, required=True)
    return parser


def _write_result(path: Path, document: dict[str, Any]) -> None:
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if len(payload) > 64 * 1024:
        raise LaneScratchError("lane_scratch_result_too_large")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    file_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(file_fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "ls":
            document = {"status": "listed", **list_lane_scratch(
                root=args.root, lane=args.lane, limit=args.limit, offset=args.offset)}
        elif args.action == "renew":
            lease = renew_lane_scratch(root=args.root, lane=args.lane, name=args.name,
                                       owner=args.owner, expected_digest=args.expected_digest,
                                       ttl_seconds=args.ttl_seconds)
            document = {"status": "renewed", "lease": lease}
        else:
            lease = release_lane_scratch(root=args.root, lane=args.lane, name=args.name,
                                         owner=args.owner, expected_digest=args.expected_digest)
            document = {"status": "released", "lease": lease}
        _write_result(args.result_out, document)
        return 0
    except (LaneScratchError, OSError) as exc:
        code = str(exc) if isinstance(exc, LaneScratchError) else "lane_scratch_io_failed"
        try:
            _write_result(args.result_out, {"status": "failed", "code": code})
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
