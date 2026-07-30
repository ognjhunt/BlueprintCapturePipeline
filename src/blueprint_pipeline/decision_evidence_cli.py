"""Plan, supervise, execute, aggregate, and ingest one Task Evaluation Run.

The CLI is fail-closed. ``plan`` performs no execution. ``execute`` accepts only
an explicitly enabled hermetic fixture-adapter registry in v1; live providers,
paid compute, and physical robot operation remain outside this command.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .common import read_json, write_json
from .decision_evidence_execution import (
    EvidenceMethodAdapter,
    EvidenceMethodAdapterRegistry,
    build_decision_envelope,
    execute_evidence_plan,
)
from .decision_evidence_router import route_decision_evidence
from .physical_outcome_learning import join_physical_outcome
from .task_evaluation_supervisor import (
    AutonomyMode,
    DEFAULT_SUPERVISOR_AGENT_MODEL,
    OpenAIOrganizationCostsClient,
    OpenAIProjectCandidateCostAuthority,
    SupervisorContext,
    TaskEvaluationSupervisor,
    evaluate_recorded_supervisor_corpus,
    freeze_supervisor_evaluation_configuration,
    load_capture_build_ingress,
    load_sealed_supervisor_evaluation_corpus,
    reconcile_neutral_candidate_policy_costs,
)


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
    return EvidenceMethodAdapterRegistry(cast(Sequence[EvidenceMethodAdapter], adapters))


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


def _supervise(args: argparse.Namespace) -> dict[str, Any]:
    capture_build = load_capture_build_ingress(args.capture_build) if args.capture_build else None
    request = read_json(args.request) if args.request else None
    testbed = read_json(args.testbed) if args.testbed else None
    if capture_build is None and request is None and testbed is None:
        raise ValueError("supervisor_requires_capture_build_request_or_testbed")
    plan = read_json(args.plan) if args.plan else None
    decision = read_json(args.decision) if args.decision else None
    results = _read_many(args.result)
    identity = (
        (request or {}).get("request_id")
        or (testbed or {}).get("testbed_id")
        or (capture_build or {}).get("capture_build_digest", "capture-build")[7:23]
    )
    run_id = str(args.run_id or f"supervisor-{identity}").strip()
    question = str((request or {}).get("decision_question") or "").strip()
    if not question:
        question = (
            "What task evaluations can this capture build currently support, "
            "and what customer, robot, task, success, or evidence details are still missing?"
        )
    execution = TaskEvaluationSupervisor(
        agent_model=args.agent_model,
        allow_live_agents_sdk=args.allow_live_agent_sdk,
        agent_inference_budget_usd=args.agent_inference_budget_usd,
    ).run(
        SupervisorContext(
            run_id=run_id,
            customer_question=question,
            capture_build=capture_build,
            decision_request=request,
            testbed=testbed,
            method_profiles=_read_many(args.method_profile),
            qualifications=_read_many(args.qualification),
            evidence_plan=plan,
            evidence_results=results,
            decision_envelope=decision,
        ),
        output_dir=args.output_dir,
        mode=AutonomyMode(args.mode),
    )
    report = execution.report.to_mapping()
    live_invocations = [
        row.to_mapping()
        for row in execution.invocation_manifests
        if row.to_mapping().get("provider") == "openai"
        and row.to_mapping().get("validation_status") == "accepted_as_proposal"
    ]
    actions_executed = report.get("actions_executed") is True
    inference_spend = dict(report.get("inference_spend") or {})
    agent_inference_started = (
        bool(live_invocations) or int(inference_spend.get("reservation_count") or 0) > 0
    )
    tool_execution_started = actions_executed or any(
        int(report.get(key) or 0) > 0
        for key in (
            "registered_tool_reads_executed",
            "registered_non_spend_actions_executed",
            "registered_preauthorized_actions_executed",
        )
    )
    return {
        "schema_version": "decision_evidence_cli_result.v1",
        "operation": "supervise",
        "status": report["status"],
        "mode": args.mode,
        "supervisor_run_digest": execution.run.digest,
        "terminal_report_digest": report["terminal_report_digest"],
        "terminal_report": str(args.output_dir / "terminal_supervisor_report.json"),
        "event_ledger": str(args.output_dir / "supervisor_events.jsonl"),
        "capability_count": len(execution.capability_results),
        "triggered_capability_count": len(execution.capability_results),
        "registered_capability_count": len(execution.run.to_mapping().get("capabilities") or []),
        "agent_harness": "openai_agents_sdk",
        "agent_model": args.agent_model,
        "agent_inference_budget_usd": args.agent_inference_budget_usd,
        "capture_build_ingested": capture_build is not None,
        "execution_started": tool_execution_started,
        "actions_executed": actions_executed,
        "agent_inference_started": agent_inference_started,
        "live_agent_inference": agent_inference_started,
        # Model inference is not an evidence-provider or robot-policy execution.
        "live_provider_execution": False,
        "physical_robot_run_initiated": False,
        "proof_state_mutated_by_agent": False,
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


def _reconcile_candidate_costs(args: argparse.Namespace) -> dict[str, Any]:
    attestation = read_json(args.scope_attestation)
    client = OpenAIOrganizationCostsClient(
        project_id=args.openai_project_id,
        api_key_id=args.openai_api_key_id,
        admin_api_key_file=args.openai_admin_key_file,
        timeout_seconds=args.provider_read_timeout_seconds,
    )
    authority = OpenAIProjectCandidateCostAuthority(
        client=client,
        scope_attestation=attestation,
        provider_id=args.provider_id,
        paid_resource_class="openai_api_candidate",
    )
    report = reconcile_neutral_candidate_policy_costs(
        args.execution_dir,
        candidate_cost_authorities=[authority],
    )
    return {
        "schema_version": "decision_evidence_cli_result.v1",
        "operation": "reconcile-candidate-costs",
        "status": report["status"],
        "candidate_cost_reconciliation_digest": report["candidate_cost_reconciliation_digest"],
        "reported_cost_usd": report.get("reported_cost_usd"),
        "reported_cost_is_final": report["reported_cost_is_final"],
        "candidate_execution_repeated": False,
        "candidate_evaluation_repeated": False,
        "provider_cost_reconciliation_requested": True,
        "provider_mutation_performed": False,
        "candidate_reported_cost_accepted": False,
        "physical_robot_run_initiated": False,
        "proof_effect": "none",
    }


def _recorded_run_mapping(rows: Sequence[str]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for row in rows:
        case_id, separator, raw_path = row.partition("=")
        if not separator or not case_id or not raw_path or case_id in mapped:
            raise ValueError("recorded_run_mapping_invalid")
        mapped[case_id] = Path(raw_path).expanduser().resolve()
    return mapped


def _validate_supervisor_corpus(args: argparse.Namespace) -> dict[str, Any]:
    corpus, cases = load_sealed_supervisor_evaluation_corpus(args.corpus)
    result = {
        "schema_version": "task_evaluation_supervisor_corpus_validation.v1",
        "operation": "validate-supervisor-corpus",
        "status": "passed",
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": corpus["corpus_digest"],
        "heldout_case_count": sum(case.split == "heldout" for case in cases),
        "development_case_count": sum(case.split == "development" for case in cases),
        "hidden_case_properties_emitted": False,
        "proof_effect": "none",
    }
    write_json(args.output.expanduser().resolve(), result)
    return result


def _freeze_supervisor_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    corpus, _cases = load_sealed_supervisor_evaluation_corpus(args.corpus)
    result = freeze_supervisor_evaluation_configuration(
        read_json(args.spec),
        corpus_digest=str(corpus["corpus_digest"]),
    )
    write_json(args.output.expanduser().resolve(), result)
    return result


def _evaluate_recorded_supervisor(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_recorded_supervisor_corpus(
        corpus_path=args.corpus,
        configuration=read_json(args.configuration),
        recorded_runs=_recorded_run_mapping(args.run),
        output_dir=args.output_dir,
    )


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

    supervise = subparsers.add_parser(
        "supervise",
        help="Run the proof-safe Task Evaluation Supervisor in an explicit autonomy mode.",
    )
    supervise.add_argument(
        "--capture-build",
        type=Path,
        help="Completed capture root or capture-build manifest; sufficient to start supervision.",
    )
    supervise.add_argument("--request", type=Path)
    supervise.add_argument("--testbed", type=Path)
    supervise.add_argument("--method-profile", action="append", type=Path, default=[])
    supervise.add_argument("--qualification", action="append", type=Path, default=[])
    supervise.add_argument("--plan", type=Path)
    supervise.add_argument("--result", action="append", type=Path, default=[])
    supervise.add_argument("--decision", type=Path)
    supervise.add_argument("--run-id")
    supervise.add_argument("--agent-model", default=DEFAULT_SUPERVISOR_AGENT_MODEL)
    supervise.add_argument("--agent-inference-budget-usd", type=float, default=0.0)
    supervise.add_argument(
        "--allow-live-agent-sdk",
        action="store_true",
        help=(
            "Authorize live OpenAI Agents SDK inference for this run. The shared "
            "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS gate is also required."
        ),
    )
    supervise.add_argument(
        "--mode",
        choices=[mode.value for mode in AutonomyMode],
        default=AutonomyMode.SHADOW.value,
    )
    supervise.add_argument("--output-dir", required=True, type=Path)
    supervise.set_defaults(handler=_supervise)

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

    validate_corpus = subparsers.add_parser(
        "validate-supervisor-corpus",
        help="Validate a sealed supervisor corpus without exposing hidden properties.",
    )
    validate_corpus.add_argument("--corpus", required=True, type=Path)
    validate_corpus.add_argument("--output", required=True, type=Path)
    validate_corpus.set_defaults(handler=_validate_supervisor_corpus)

    freeze_evaluation = subparsers.add_parser(
        "freeze-supervisor-evaluation",
        help="Freeze the manager and six specialist identities before held-out execution.",
    )
    freeze_evaluation.add_argument("--corpus", required=True, type=Path)
    freeze_evaluation.add_argument("--spec", required=True, type=Path)
    freeze_evaluation.add_argument("--output", required=True, type=Path)
    freeze_evaluation.set_defaults(handler=_freeze_supervisor_evaluation)

    recorded_evaluation = subparsers.add_parser(
        "evaluate-recorded-supervisor",
        help="Replay and independently score a complete recorded held-out matrix.",
    )
    recorded_evaluation.add_argument("--corpus", required=True, type=Path)
    recorded_evaluation.add_argument("--configuration", required=True, type=Path)
    recorded_evaluation.add_argument(
        "--run",
        action="append",
        default=[],
        required=True,
        metavar="CASE_ID=RUN_DIR",
    )
    recorded_evaluation.add_argument("--output-dir", required=True, type=Path)
    recorded_evaluation.set_defaults(handler=_evaluate_recorded_supervisor)

    reconcile_costs = subparsers.add_parser(
        "reconcile-candidate-costs",
        help=(
            "Reconcile delayed paid-candidate cost from OpenAI's read-only "
            "organization Costs endpoint without rerunning candidates."
        ),
    )
    reconcile_costs.add_argument("--execution-dir", required=True, type=Path)
    reconcile_costs.add_argument(
        "--provider-id",
        default="pigey_external_candidate",
    )
    reconcile_costs.add_argument("--openai-project-id", required=True)
    reconcile_costs.add_argument("--openai-api-key-id", required=True)
    reconcile_costs.add_argument("--openai-admin-key-file", required=True, type=Path)
    reconcile_costs.add_argument("--scope-attestation", required=True, type=Path)
    reconcile_costs.add_argument(
        "--provider-read-timeout-seconds",
        type=float,
        default=30.0,
    )
    reconcile_costs.set_defaults(handler=_reconcile_candidate_costs)
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
        if args.operation in {
            "validate-supervisor-corpus",
            "freeze-supervisor-evaluation",
            "evaluate-recorded-supervisor",
        }:
            result.update(
                {
                    "model_invoked": False,
                    "recorded_runs_mutated": False,
                    "proof_effect": "none",
                }
            )
        print(json.dumps(result, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    if args.operation == "supervise" and result.get("status") == "blocked":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
