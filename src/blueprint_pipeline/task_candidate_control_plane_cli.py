"""Operate Pipeline-owned task discovery publication and approval state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json_any, write_json
from .task_candidate_control_plane import (
    TaskCandidateControlPlaneError,
    load_task_candidate_control_plane_state,
    publish_and_sync_task_candidate_discovery,
    publish_task_candidate_discovery,
)


def _json_object(path: str | Path) -> dict[str, Any]:
    value = read_json_any(Path(path).expanduser().resolve())
    if not isinstance(value, Mapping):
        raise TaskCandidateControlPlaneError("input_json:not_object")
    return dict(value)


def _emit(result: Mapping[str, Any], output: str | None) -> None:
    if output:
        write_json(Path(output).expanduser().resolve(), dict(result))
        return
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


def _publish(args: argparse.Namespace) -> dict[str, Any]:
    discovery = _json_object(args.discovery_json)
    if args.sync_mode == "off":
        return publish_task_candidate_discovery(
            state_root=args.state_root,
            capture_session_id=args.capture_session_id,
            intake_id=args.intake_id,
            discovery=discovery,
        )
    return publish_and_sync_task_candidate_discovery(
        state_root=args.state_root,
        capture_session_id=args.capture_session_id,
        intake_id=args.intake_id,
        discovery=discovery,
        endpoint_url=args.webapp_url,
        sync_required=args.sync_mode == "required",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser(
        "publish-discovery",
        help="persist one immutable discovery and optionally sync its safe projection",
    )
    publish.add_argument("--state-root", required=True)
    publish.add_argument("--capture-session-id", required=True)
    publish.add_argument("--intake-id", required=True)
    publish.add_argument("--discovery-json", required=True)
    publish.add_argument("--webapp-url")
    publish.add_argument(
        "--sync-mode",
        choices=("off", "optional", "required"),
        default="optional",
        help=(
            "off persists only in Pipeline; optional records a skipped/failed sync; "
            "required fails closed unless the signed WebApp publication succeeds"
        ),
    )
    publish.add_argument("--output")

    inspect = subparsers.add_parser(
        "inspect-state", help="print the durable state for one capture session"
    )
    inspect.add_argument("--state-root", required=True)
    inspect.add_argument("--capture-session-id", required=True)
    inspect.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "publish-discovery":
            result = _publish(args)
        else:
            result = load_task_candidate_control_plane_state(
                state_root=args.state_root,
                capture_session_id=args.capture_session_id,
            )
            if not result:
                raise TaskCandidateControlPlaneError(
                    "task_candidate_control_plane_state_not_found", status_code=404
                )
        _emit(result, args.output)
    except (OSError, ValueError, TaskCandidateControlPlaneError) as exc:
        parser = _parser()
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
