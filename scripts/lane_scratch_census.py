#!/usr/bin/env python3
"""Read-only lane scratch census; print an owner annotation table and optional JSON.

Run only after reviewing root and reference arguments. This command never
creates a lease, deletes a folder, offloads data, or contacts a provider.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blueprint_pipeline.control_plane_lane_scratch_census import (  # noqa: E402
    DEFAULT_INPUTS_ROOT,
    DEFAULT_WORK_ROOT,
    build_census,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS_ROOT)
    parser.add_argument("--process-root", type=Path, default=Path("/proc"))
    parser.add_argument("--pins-root", type=Path,
                        default=Path("/var/lib/blueprint/pipeline-control-plane/storage-pins"))
    parser.add_argument("--release-link", type=Path,
                        default=Path("/opt/blueprint/task-evaluation-control-plane"))
    queue = parser.add_mutually_exclusive_group()
    queue.add_argument("--queue-root", type=Path, action="append")
    queue.add_argument("--queue-inventory-empty", action="store_true")
    active = parser.add_mutually_exclusive_group()
    active.add_argument("--active-run-root", type=Path, action="append")
    active.add_argument("--active-run-inventory-empty", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=240.0)
    parser.add_argument("--json-out", type=Path)
    return parser


def _table(report: dict) -> str:
    def cell(value: object) -> str:
        return (str(value).replace("\\", "\\\\").replace("\t", "\\t")
                .replace("\r", "\\r").replace("\n", "\\n"))

    columns = ("family", "owner guess", "allocated bytes", "newest mtime", "age seconds",
               "references", "owner decision", "approved expiry", "path")
    lines = ["\t".join(columns)]
    for row in report["rows"]:
        lines.append("\t".join(cell(value) for value in (
            str(row["family"]), str(row["owner_guess"]), str(row["allocated_bytes"]),
            str(row["newest_mtime_epoch"] or ""), str(row["age_seconds"] or ""),
            ",".join(row["references"]), "", "", str(row["path"]),
        )))
    lines.append(f"status={report['status']} candidates={report['candidate_count']} "
                 f"listed={len(report['rows'])} unique_allocated_bytes={report['unique_allocated_bytes']}")
    if report["scan_errors"]:
        lines.append("scan_errors=" + ",".join(report["scan_errors"]))
    return "\n".join(lines)


def _write_json(path: Path, report: dict) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 0 < args.max_seconds <= 240:
        parser.error("--max-seconds must be between 0 and 240")
    queue_roots = () if args.queue_inventory_empty else args.queue_root
    active_roots = () if args.active_run_inventory_empty else args.active_run_root
    report = build_census(work_root=args.work_root, inputs_root=args.inputs_root,
                          process_root=args.process_root, pins_root=args.pins_root,
                          queue_roots=queue_roots, release_link=args.release_link,
                          active_run_roots=active_roots, max_seconds=args.max_seconds)
    print(_table(report))
    if args.json_out is not None:
        _write_json(args.json_out, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
