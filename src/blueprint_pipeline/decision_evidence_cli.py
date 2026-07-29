"""Plan, execute, aggregate, and ingest outcomes for one Task Evaluation Run.

The CLI is fail-closed. ``plan`` performs no execution. ``execute`` accepts only
an explicitly enabled hermetic fixture-adapter registry in v1; live providers,
paid compute, and physical robot operation remain outside this command.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json, write_json
from .decision_evidence_execution import (
    EvidenceMethodAdapterRegistry,
    build_decision_envelope,
    execute_evidence_plan,
)
from .decision_evidence_router import route_decision_evidence
from .physical_outcome_learning import join_physical_outcome


def _read_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [dict(read_json(path)) for path in paths]


@dataclass(frozen=True)
class _FixtureResultAdapter:
    adapter_reference: str
    result: Mapping[str, Any]

    def execute(self, **_: Any) -> Mapping[str, Any]:
        return dict(self.result)


def _fixture_registry(path: Path) -> EvidenceMethodAdapterRegistry:
    value = read_json(path)
    if value.get("schema_version") != "evidence_fixture_adapter_registry.v1":
        raise ValueError("fixture_adapter_registry_schema_mismatch")
    rows = value.get("adapters")
    if not isinstance(rows, list):
        raise ValueError("fixture_adapter_registry_rows_missing")
    adapters: list[_FixtureResultAdapter] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"fixture_adapter_registry_row_invalid:{index}")
        reference = str(row.get("adapter_reference") or "").strip()
        result = row.get("result")
        if not reference or not isinstance(result, Mapping):
            raise ValueError(f"fixture_adapter_registry_binding_invalid:{index}")
        adapters.append(_FixtureResultAdapter(reference, dict(result)))
    return EvidenceMethodAdapterRegistry(adapters)


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = route_decision_evidence(
        read_json(args.request),
        read_json(args.testbed),
        _read_many(args.method_profile),
        _read_many(args.qualification),
    ).to_mapping()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "evidence_plan.json", plan)
    return {
        "schema_version": "decision_evidence_cli_result.v1",
        "operation": "plan",
        "status": "prepared",
        "plan_digest": plan["plan_digest"],
        "output": str(args.output_dir / "evidence_plan.json"),
        "execution_started": False,
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_fixture_adapters:
        raise ValueError("fixture_adapter_execution_not_authorized")
    registry = _fixture_registry(args.fixture_adapter_registry)
    execution = execute_evidence_plan(
        read_json(args.plan),
        read_json(args.request),
        read_json(args.testbed),
        _read_many(args.method_profile),
        _read_many(args.qualification),
        registry=registry,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[str] = []
    for result in execution.results:
        mapping = result.to_mapping()
        path = args.output_dir / f"{mapping['result_id']}.json"
        write_json(path, mapping)
        result_paths.append(str(path))
    write_json(args.output_dir / "evidence_plan_execution.json", execution.execution_manifest)
    return {
        "schema_version": "decision_evidence_cli_result.v1",
        "operation": "execute",
        "status": execution.execution_manifest["status"],
        "execution_manifest": str(args.output_dir / "evidence_plan_execution.json"),
        "results": sorted(result_paths),
        "fixture_adapters_explicitly_authorized": True,
        "live_provider_execution": False,
        "physical_robot_run_initiated": False,
    }


def _aggregate(args: argparse.Namespace) -> dict[str, Any]:
    envelope = build_decision_envelope(
        read_json(args.request),
        read_json(args.testbed),
        read_json(args.plan),
        _read_many(args.result),
    ).to_mapping()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "decision_envelope.json", envelope)
    return {
        "schema_version": "decision_evidence_cli_result.v1",
        "operation": "aggregate",
        "status": envelope["overall_outcome"],
        "decision_envelope_digest": envelope["decision_envelope_digest"],
        "output": str(args.output_dir / "decision_envelope.json"),
    }


def _ingest_outcome(args: argparse.Namespace) -> dict[str, Any]:
    update = join_physical_outcome(
        testbed_value=read_json(args.testbed),
        decision_value=read_json(args.decision),
        outcome_value=read_json(args.outcome),
        method_profile_value=read_json(args.method_profile),
        existing_outcome_values=_read_many(args.existing_outcome),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcome = update.physical_outcome.to_mapping()
    testbed = update.new_testbed.to_mapping()
    qualification = update.calibration_record.to_mapping()
    write_json(args.output_dir / "physical_outcome_join.json", outcome)
    write_json(args.output_dir / "maintained_site_task_testbed.json", testbed)
    write_json(args.output_dir / "evidence_method_qualification.json", qualification)
    manifest = {
        "schema_version": "physical_outcome_learning_update.v1",
        "status": "updated_append_only",
        "physical_outcome_digest": outcome["physical_outcome_digest"],
        "predecessor_testbed_digest": testbed["predecessor_testbed_digest"],
        "new_testbed_digest": testbed["testbed_digest"],
        "qualification_digest": qualification["qualification_digest"],
        "historical_decision_mutated": False,
        "historical_testbed_mutated": False,
        "cross_domain_transfer_enabled": False,
    }
    write_json(args.output_dir / "physical_outcome_learning_update.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    plan = subparsers.add_parser("plan", help="Build a deterministic claim-level evidence plan.")
    plan.add_argument("--request", required=True, type=Path)
    plan.add_argument("--testbed", required=True, type=Path)
    plan.add_argument("--method-profile", required=True, action="append", type=Path)
    plan.add_argument("--qualification", required=True, action="append", type=Path)
    plan.add_argument("--output-dir", required=True, type=Path)
    plan.set_defaults(handler=_plan)

    execute = subparsers.add_parser(
        "execute", help="Execute a plan through explicitly authorized hermetic fixture adapters."
    )
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--request", required=True, type=Path)
    execute.add_argument("--testbed", required=True, type=Path)
    execute.add_argument("--method-profile", required=True, action="append", type=Path)
    execute.add_argument("--qualification", required=True, action="append", type=Path)
    execute.add_argument("--fixture-adapter-registry", required=True, type=Path)
    execute.add_argument("--allow-fixture-adapters", action="store_true")
    execute.add_argument("--output-dir", required=True, type=Path)
    execute.set_defaults(handler=_execute)

    aggregate = subparsers.add_parser("aggregate", help="Build the final Decision Envelope.")
    aggregate.add_argument("--request", required=True, type=Path)
    aggregate.add_argument("--testbed", required=True, type=Path)
    aggregate.add_argument("--plan", required=True, type=Path)
    aggregate.add_argument("--result", required=True, action="append", type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    aggregate.set_defaults(handler=_aggregate)

    ingest = subparsers.add_parser(
        "ingest-outcome", help="Join physical evidence append-only and emit a new testbed version."
    )
    ingest.add_argument("--testbed", required=True, type=Path)
    ingest.add_argument("--decision", required=True, type=Path)
    ingest.add_argument("--outcome", required=True, type=Path)
    ingest.add_argument("--method-profile", required=True, type=Path)
    ingest.add_argument("--existing-outcome", action="append", type=Path, default=[])
    ingest.add_argument("--output-dir", required=True, type=Path)
    ingest.set_defaults(handler=_ingest_outcome)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except Exception as exc:  # noqa: BLE001 - CLI must emit bounded terminal evidence
        result = {
            "schema_version": "decision_evidence_cli_result.v1",
            "operation": args.operation,
            "status": "blocked",
            "blockers": [str(exc)],
            "error_type": type(exc).__name__,
            "execution_started": False,
            "live_provider_execution": False,
            "physical_robot_run_initiated": False,
        }
        print(json.dumps(result, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
