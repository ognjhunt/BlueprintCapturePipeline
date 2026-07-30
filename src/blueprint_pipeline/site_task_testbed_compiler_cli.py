"""Compile and immutably persist one maintained Site-Task Testbed version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json_any, write_json
from .site_task_testbed_compiler import (
    SiteTaskTestbedCompilerError,
    compile_site_task_testbed,
    write_testbed_version,
)


def _object(path: Path, *, label: str) -> dict[str, Any]:
    value = read_json_any(path.expanduser().resolve())
    if not isinstance(value, Mapping):
        raise SiteTaskTestbedCompilerError([f"{label}:not_object"])
    return dict(value)


def _rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    value = read_json_any(path.expanduser().resolve())
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise SiteTaskTestbedCompilerError([f"{label}:not_object_list"])
    return [dict(row) for row in value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testbed-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--capture-intake-envelope", type=Path, required=True)
    parser.add_argument("--capture-qa-report", type=Path, required=True)
    parser.add_argument("--approved-task-definition", type=Path, required=True)
    parser.add_argument("--reconstruction-plan", type=Path, required=True)
    parser.add_argument("--reconstruction-results", type=Path, required=True)
    parser.add_argument("--simready-decision", type=Path, required=True)
    parser.add_argument("--robot-placement-result", type=Path, required=True)
    parser.add_argument("--artifact-references", type=Path, required=True)
    parser.add_argument("--supported-condition-ranges", type=Path, required=True)
    parser.add_argument("--previous-testbed", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        testbed = compile_site_task_testbed(
            testbed_id=args.testbed_id,
            version=args.version,
            capture_intake_envelope=_object(
                args.capture_intake_envelope, label="capture_intake_envelope"
            ),
            capture_qa_report=_object(args.capture_qa_report, label="capture_qa_report"),
            approved_task_definition=_object(
                args.approved_task_definition, label="approved_task_definition"
            ),
            reconstruction_plan=_object(
                args.reconstruction_plan, label="reconstruction_plan"
            ),
            reconstruction_results=_rows(
                args.reconstruction_results, label="reconstruction_results"
            ),
            simready_decision=_object(args.simready_decision, label="simready_decision"),
            robot_placement_result=_object(
                args.robot_placement_result, label="robot_placement_result"
            ),
            artifact_references=_object(
                args.artifact_references, label="artifact_references"
            ),
            supported_condition_ranges=_object(
                args.supported_condition_ranges, label="supported_condition_ranges"
            ),
            previous_testbed=(
                _object(args.previous_testbed, label="previous_testbed")
                if args.previous_testbed
                else None
            ),
        )
        result = write_testbed_version(output_root=args.output_root, testbed=testbed)
        if args.result_output:
            write_json(args.result_output.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
