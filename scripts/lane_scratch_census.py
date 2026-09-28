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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blueprint_pipeline.control_plane_lane_scratch_census import (  # noqa: E402
    DEFAULT_INPUTS_ROOT,
    DEFAULT_WORK_ROOT,
    build_census,
)

from blueprint_pipeline.control_plane_lane_scratch_decisions import (  # noqa: E402
    CensusDecisionError, VALIDATION_SCHEMA, encode_validation_report,
    read_census_input_record, validate_census_annotations, write_census_validation_report,
)


class _CensusParser(argparse.ArgumentParser):
    validation_mode = False

    def error(self, message: str) -> None:
        if self.validation_mode:
            raise CensusDecisionError("census_annotations_invalid")
        super().error(message)


def _parser(*, validation_mode: bool = False) -> argparse.ArgumentParser:
    parser = _CensusParser(description=__doc__)
    parser.validation_mode = validation_mode
    parser.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    parser.add_argument("--inputs-root", default=str(DEFAULT_INPUTS_ROOT))
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
    parser.add_argument("--validate-census", type=Path)
    parser.add_argument("--annotations", type=Path)
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


def _validation_refusal(code: str) -> int:
    report = {"schema_version": VALIDATION_SCHEMA, "status": "refused", "blockers": [code],
              "mutations": 0, "execution_authorized": False,
              "requires_fresh_reference_check": True}
    print(encode_validation_report(report).decode("utf-8"), end="")
    return 1


def _validation_mode(args, argv: list[str]) -> int:
    scan_options = {"--process-root", "--pins-root", "--release-link", "--queue-root",
                    "--queue-inventory-empty", "--active-run-root", "--active-run-inventory-empty",
                    "--max-seconds"}
    try:
        if (args.validate_census is None or args.annotations is None
                or any(option.startswith(token.split("=", 1)[0])
                       for token in argv if token.startswith("--") for option in scan_options)):
            raise CensusDecisionError("census_annotations_invalid")
        census, census_identity = read_census_input_record(args.validate_census)
        annotations, annotation_identity = read_census_input_record(args.annotations)
        report = validate_census_annotations(census, annotations, now=time.time(),
                                            allowed_roots=(args.work_root, args.inputs_root))
        payload = encode_validation_report(report)
        if args.json_out is not None:
            write_census_validation_report(args.json_out, payload,
                input_paths=(args.validate_census, args.annotations),
                input_identities=(census_identity, annotation_identity))
    except CensusDecisionError as exc:
        return _validation_refusal(exc.code)
    print(payload.decode("utf-8"), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    validation_intent = any(option.startswith(token.split("=", 1)[0])
                            for token in arguments if token.startswith("--") and len(token) > 2
                            for option in ("--validate-census", "--annotations"))
    parser = _parser(validation_mode=validation_intent)
    try:
        args = parser.parse_args(arguments)
    except CensusDecisionError as exc:
        return _validation_refusal(exc.code)
    if args.validate_census is not None or args.annotations is not None:
        return _validation_mode(args, arguments)
    if not 0 < args.max_seconds <= 240:
        parser.error("--max-seconds must be between 0 and 240")
    queue_roots = () if args.queue_inventory_empty else args.queue_root
    active_roots = () if args.active_run_inventory_empty else args.active_run_root
    report = build_census(work_root=Path(args.work_root), inputs_root=Path(args.inputs_root),
                          process_root=args.process_root, pins_root=args.pins_root,
                          queue_roots=queue_roots, release_link=args.release_link,
                          active_run_roots=active_roots, max_seconds=args.max_seconds)
    print(_table(report))
    if args.json_out is not None:
        _write_json(args.json_out, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
