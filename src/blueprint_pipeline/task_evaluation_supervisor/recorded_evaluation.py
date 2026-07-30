"""Independent evaluation of replay-verified, already-recorded supervisor runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

from ..common import read_json, write_json
from ..decision_evidence_contracts import canonical_digest
from .agents_sdk import AGENTS_SDK_HARNESS_ID
from .contracts import (
    AgentInvocationManifest,
    CapabilityKind,
    CapabilityResult,
    SupervisorRun,
    SupervisorState,
    TerminalSupervisorReport,
)
from .evaluation import (
    SEALED_SUPERVISOR_EVAL_CORPUS_SCHEMA_VERSION,
    SupervisorEvaluationCase,
    SupervisorEvaluationError,
    compare_supervisor_to_baseline,
    evaluate_supervisor_execution,
    load_supervisor_evaluation_corpus,
)
from .replay import replay_supervisor_run
from .supervisor import SupervisorExecution


SUPERVISOR_EVAL_CONFIGURATION_SCHEMA_VERSION = (
    "task_evaluation_supervisor_agent_configuration_manifest.v1"
)
RECORDED_SUPERVISOR_EVAL_BUNDLE_SCHEMA_VERSION = (
    "task_evaluation_supervisor_recorded_evaluation_bundle.v1"
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


def _utc(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupervisorEvaluationError(f"{field}_invalid") from exc
    if parsed.tzinfo is None:
        raise SupervisorEvaluationError(f"{field}_timezone_required")
    return parsed.astimezone(timezone.utc)


def _digest(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise SupervisorEvaluationError(f"{field}_invalid")
    return text


def _identity(value: Mapping[str, Any], *, manager: bool = False) -> dict[str, str]:
    allowed = {
        "provider",
        "model",
        "agent_harness",
        "agents_sdk_version",
        "adapter_id",
        "adapter_version",
        "instruction_digest",
    }
    if not manager:
        allowed.add("capability")
    if set(value) != allowed:
        raise SupervisorEvaluationError("evaluation_configuration_identity_fields_invalid")
    normalized = {key: str(value.get(key) or "").strip() for key in allowed}
    if not all(normalized.values()):
        raise SupervisorEvaluationError("evaluation_configuration_identity_missing")
    _digest(normalized["instruction_digest"], field="evaluation_instruction_digest")
    if not manager:
        try:
            CapabilityKind(normalized["capability"])
        except ValueError as exc:
            raise SupervisorEvaluationError("evaluation_configuration_capability_invalid") from exc
    return dict(sorted(normalized.items()))


def load_sealed_supervisor_evaluation_corpus(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[SupervisorEvaluationCase, ...]]:
    """Validate the private corpus contract without exposing it to any agent."""

    corpus_path = Path(path).expanduser().resolve()
    value = read_json(corpus_path)
    expected_digest = canonical_digest(value, digest_field="corpus_digest")
    if (
        value.get("schema_version") != SEALED_SUPERVISOR_EVAL_CORPUS_SCHEMA_VERSION
        or value.get("corpus_digest") != expected_digest
        or value.get("status") != "frozen"
        or value.get("issued_by_agent") is not False
        or value.get("frozen_before_agent_execution") is not True
        or value.get("development_cases_excluded_from_promotion") is not True
        or value.get("hidden_expected_properties_sent_to_agent") is not False
        or value.get("proof_effect") != "none"
        or not _IDENTIFIER.fullmatch(str(value.get("corpus_id") or ""))
        or not str(value.get("operator_id") or "").strip()
    ):
        raise SupervisorEvaluationError("sealed_evaluation_corpus_invalid")
    frozen_at = _utc(value.get("frozen_at"), field="evaluation_corpus_frozen_at")
    minimum_improvement = value.get("minimum_required_improvement")
    if not isinstance(minimum_improvement, (int, float)) or isinstance(minimum_improvement, bool):
        raise SupervisorEvaluationError("evaluation_corpus_promotion_threshold_invalid")
    threshold = float(minimum_improvement)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise SupervisorEvaluationError("evaluation_corpus_promotion_threshold_invalid")
    cases = load_supervisor_evaluation_corpus(corpus_path)
    heldout = tuple(case for case in cases if case.split == "heldout")
    if len(heldout) < 6 or any(case.baseline_metrics is None for case in heldout):
        raise SupervisorEvaluationError("sealed_evaluation_corpus_heldout_invalid")
    return (
        {
            "corpus_id": str(value["corpus_id"]),
            "corpus_digest": expected_digest,
            "frozen_at": frozen_at.isoformat(),
            "minimum_required_improvement": threshold,
            "heldout_case_count": len(heldout),
        },
        cases,
    )


def freeze_supervisor_evaluation_configuration(
    spec: Mapping[str, Any],
    *,
    corpus_digest: str,
) -> dict[str, Any]:
    """Freeze the manager and six specialist identities before held-out runs."""

    spec_fields = {
        "configuration_id",
        "operator_id",
        "issued_by_agent",
        "frozen_at",
        "tool_registry_digest",
        "max_inference_cost_usd",
        "manager_identity",
        "specialist_identities",
    }
    if set(spec) != spec_fields:
        raise SupervisorEvaluationError("evaluation_configuration_spec_fields_invalid")
    if not _IDENTIFIER.fullmatch(str(spec.get("configuration_id") or "")):
        raise SupervisorEvaluationError("evaluation_configuration_id_invalid")
    if not str(spec.get("operator_id") or "").strip() or spec.get("issued_by_agent") is not False:
        raise SupervisorEvaluationError("evaluation_configuration_operator_invalid")
    frozen_at = _utc(spec.get("frozen_at"), field="evaluation_configuration_frozen_at")
    tool_registry_digest = _digest(
        spec.get("tool_registry_digest"), field="evaluation_tool_registry_digest"
    )
    manager = _identity(dict(spec.get("manager_identity") or {}), manager=True)
    raw_specialists = spec.get("specialist_identities")
    if not isinstance(raw_specialists, list):
        raise SupervisorEvaluationError("evaluation_configuration_specialists_invalid")
    specialists = [_identity(dict(row)) for row in raw_specialists if isinstance(row, Mapping)]
    expected_capabilities = {kind.value for kind in CapabilityKind}
    if (
        len(specialists) != len(raw_specialists)
        or len(specialists) != len(expected_capabilities)
        or {row["capability"] for row in specialists} != expected_capabilities
    ):
        raise SupervisorEvaluationError("evaluation_configuration_specialists_invalid")
    if manager["agent_harness"] != AGENTS_SDK_HARNESS_ID or any(
        row["agent_harness"] != AGENTS_SDK_HARNESS_ID for row in specialists
    ):
        raise SupervisorEvaluationError("evaluation_configuration_harness_invalid")
    max_cost = spec.get("max_inference_cost_usd")
    if max_cost is None or isinstance(max_cost, bool):
        raise SupervisorEvaluationError("evaluation_configuration_budget_invalid")
    try:
        budget = float(max_cost)
    except (TypeError, ValueError) as exc:
        raise SupervisorEvaluationError("evaluation_configuration_budget_invalid") from exc
    if not math.isfinite(budget) or budget < 0:
        raise SupervisorEvaluationError("evaluation_configuration_budget_invalid")
    value: dict[str, Any] = {
        "schema_version": SUPERVISOR_EVAL_CONFIGURATION_SCHEMA_VERSION,
        "status": "frozen",
        "configuration_id": str(spec["configuration_id"]),
        "operator_id": str(spec["operator_id"]),
        "issued_by_agent": False,
        "frozen_at": frozen_at.isoformat(),
        "corpus_digest": _digest(corpus_digest, field="evaluation_corpus_digest"),
        "agent_harness": AGENTS_SDK_HARNESS_ID,
        "tool_registry_digest": tool_registry_digest,
        "manager_identity": manager,
        "specialist_identities": sorted(specialists, key=lambda row: row["capability"]),
        "max_inference_cost_usd": budget,
        "hidden_expected_properties_available_during_freeze": False,
        "candidate_configuration_mutable_after_freeze": False,
        "agent_self_grading_allowed": False,
        "proof_effect": "none",
    }
    value["configuration_digest"] = canonical_digest(value, digest_field="configuration_digest")
    return value


def validate_supervisor_evaluation_configuration(
    value: Mapping[str, Any],
    *,
    corpus_digest: str,
) -> dict[str, Any]:
    """Rebuild the frozen manifest and reject any post-freeze mutation."""

    supplied = dict(value)
    manifest_fields = {
        "schema_version",
        "status",
        "configuration_id",
        "operator_id",
        "issued_by_agent",
        "frozen_at",
        "corpus_digest",
        "agent_harness",
        "tool_registry_digest",
        "manager_identity",
        "specialist_identities",
        "max_inference_cost_usd",
        "hidden_expected_properties_available_during_freeze",
        "candidate_configuration_mutable_after_freeze",
        "agent_self_grading_allowed",
        "proof_effect",
        "configuration_digest",
    }
    if (
        set(supplied) != manifest_fields
        or supplied.get("schema_version") != SUPERVISOR_EVAL_CONFIGURATION_SCHEMA_VERSION
        or supplied.get("status") != "frozen"
        or supplied.get("corpus_digest") != corpus_digest
        or supplied.get("configuration_digest")
        != canonical_digest(supplied, digest_field="configuration_digest")
        or supplied.get("hidden_expected_properties_available_during_freeze") is not False
        or supplied.get("candidate_configuration_mutable_after_freeze") is not False
        or supplied.get("agent_self_grading_allowed") is not False
        or supplied.get("proof_effect") != "none"
    ):
        raise SupervisorEvaluationError("evaluation_configuration_manifest_invalid")
    rebuilt = freeze_supervisor_evaluation_configuration(
        {
            key: supplied[key]
            for key in (
                "configuration_id",
                "operator_id",
                "issued_by_agent",
                "frozen_at",
                "tool_registry_digest",
                "max_inference_cost_usd",
                "manager_identity",
                "specialist_identities",
            )
        },
        corpus_digest=corpus_digest,
    )
    if rebuilt != supplied:
        raise SupervisorEvaluationError("evaluation_configuration_manifest_invalid")
    return rebuilt


def load_recorded_supervisor_execution(output_dir: str | Path) -> SupervisorExecution:
    """Reconstruct a run only after deterministic replay verifies every artifact."""

    root = Path(output_dir).expanduser().resolve()
    replay = replay_supervisor_run(root, persist_report=False)
    if replay.get("status") != "replay_verified":
        raise SupervisorEvaluationError("recorded_supervisor_replay_not_verified")
    run = SupervisorRun.from_mapping(read_json(root / "task_evaluation_supervisor_run.json"))
    state = SupervisorState.from_mapping(read_json(root / "supervisor_state.json"))
    report = TerminalSupervisorReport.from_mapping(
        read_json(root / "terminal_supervisor_report.json")
    )
    report_value = report.to_mapping()
    capabilities: list[CapabilityResult] = []
    for row in report_value.get("capability_results") or []:
        path = (root / str(row.get("artifact_path") or "")).resolve()
        if root not in path.parents:
            raise SupervisorEvaluationError("recorded_capability_path_escape")
        capabilities.append(CapabilityResult.from_mapping(read_json(path)))
    invocations: list[AgentInvocationManifest] = []
    for row in report_value.get("invocation_manifests") or []:
        path = (root / str(row.get("artifact_path") or "")).resolve()
        if root not in path.parents:
            raise SupervisorEvaluationError("recorded_invocation_path_escape")
        invocations.append(AgentInvocationManifest.from_mapping(read_json(path)))
    return SupervisorExecution(
        run=run,
        state=state,
        report=report,
        capability_results=tuple(capabilities),
        invocation_manifests=tuple(invocations),
        output_dir=root,
    )


def _matches_identity(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(str(observed.get(key) or "") == str(value) for key, value in expected.items())


def _recorded_run_tree_digest(root: Path) -> str:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SupervisorEvaluationError("recorded_run_symlink_forbidden")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not rows:
        raise SupervisorEvaluationError("recorded_run_tree_empty")
    return canonical_digest({"files": rows})


def _recorded_run_contains_hidden_canary(root: Path, canaries: tuple[str, ...]) -> bool:
    needles = tuple(canary.encode("utf-8") for canary in canaries)
    longest = max(len(needle) for needle in needles)
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SupervisorEvaluationError("recorded_run_symlink_forbidden")
        if not path.is_file():
            continue
        carry = b""
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                observed = carry + chunk
                if any(needle in observed for needle in needles):
                    return True
                carry = observed[-(longest - 1) :] if longest > 1 else b""
    return False


def _validate_execution_configuration(
    execution: SupervisorExecution,
    configuration: Mapping[str, Any],
) -> None:
    run = execution.run.to_mapping()
    report = execution.report.to_mapping()
    if _utc(run.get("generated_at"), field="recorded_run_generated_at") <= _utc(
        configuration.get("frozen_at"), field="evaluation_configuration_frozen_at"
    ):
        raise SupervisorEvaluationError("recorded_run_predates_configuration_freeze")
    if (
        run.get("tool_registry_digest") != configuration.get("tool_registry_digest")
        or report.get("tool_registry_digest") != configuration.get("tool_registry_digest")
        or set(run.get("capabilities") or []) != {kind.value for kind in CapabilityKind}
    ):
        raise SupervisorEvaluationError("recorded_run_configuration_mismatch")
    expected_specialists = {
        str(row["capability"]): dict(row)
        for row in configuration.get("specialist_identities") or []
    }
    for invocation in execution.invocation_manifests:
        observed = invocation.to_mapping()
        expected = expected_specialists.get(str(observed.get("capability") or ""))
        if expected is None or not _matches_identity(observed, expected):
            raise SupervisorEvaluationError("recorded_specialist_configuration_mismatch")
        if observed.get("tool_registry_digest") != configuration.get("tool_registry_digest"):
            raise SupervisorEvaluationError("recorded_specialist_tool_registry_mismatch")
    manager_identity = dict(configuration.get("manager_identity") or {})
    for row in report.get("manager_invocations") or []:
        path = (execution.output_dir / str(row.get("artifact_path") or "")).resolve()
        if execution.output_dir not in path.parents:
            raise SupervisorEvaluationError("recorded_manager_invocation_path_escape")
        observed = read_json(path)
        if not _matches_identity(observed, manager_identity):
            raise SupervisorEvaluationError("recorded_manager_configuration_mismatch")
        if observed.get("tool_registry_digest") != configuration.get("tool_registry_digest"):
            raise SupervisorEvaluationError("recorded_manager_tool_registry_mismatch")
    spend = dict(report.get("inference_spend") or {})
    if float(spend.get("budget_usd") or 0.0) > float(
        configuration.get("max_inference_cost_usd") or 0.0
    ):
        raise SupervisorEvaluationError("recorded_run_budget_exceeds_frozen_configuration")


def evaluate_recorded_supervisor_corpus(
    *,
    corpus_path: str | Path,
    configuration: Mapping[str, Any],
    recorded_runs: Mapping[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Grade a complete sealed held-out matrix without invoking an agent."""

    corpus, cases = load_sealed_supervisor_evaluation_corpus(corpus_path)
    frozen = validate_supervisor_evaluation_configuration(
        configuration,
        corpus_digest=str(corpus["corpus_digest"]),
    )
    if _utc(frozen["frozen_at"], field="evaluation_configuration_frozen_at") <= _utc(
        corpus["frozen_at"], field="evaluation_corpus_frozen_at"
    ):
        raise SupervisorEvaluationError("evaluation_configuration_predates_corpus_freeze")
    heldout = {case.case_id: case for case in cases if case.split == "heldout"}
    if set(recorded_runs) != set(heldout):
        raise SupervisorEvaluationError("recorded_evaluation_case_matrix_incomplete")
    all_hidden_canaries = tuple(
        canary for case in heldout.values() for canary in case.hidden_canaries
    )
    root = Path(output_dir).expanduser().resolve()
    normalized_run_roots = [Path(path).expanduser().resolve() for path in recorded_runs.values()]
    run_roots = set(normalized_run_roots)
    if len(run_roots) != len(normalized_run_roots):
        raise SupervisorEvaluationError("recorded_evaluation_run_reused")
    if any(root == run_root or run_root in root.parents for run_root in run_roots):
        raise SupervisorEvaluationError("recorded_evaluation_output_inside_run")
    if root.exists() and any(root.iterdir()):
        raise SupervisorEvaluationError("recorded_evaluation_output_not_empty")
    for run_path in recorded_runs.values():
        run_root = Path(run_path).expanduser().resolve()
        if _recorded_run_contains_hidden_canary(run_root, all_hidden_canaries):
            raise SupervisorEvaluationError("recorded_run_hidden_canary_present")
    case_root = root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)

    agent_reports: list[dict[str, Any]] = []
    baseline_reports: list[dict[str, Any]] = []
    run_references: list[dict[str, Any]] = []
    observed_specialist_capabilities: set[str] = set()
    observed_run_ids: set[str] = set()
    observed_run_digests: set[str] = set()
    manager_invocation_count = 0
    for case_id in sorted(heldout):
        run_root = Path(recorded_runs[case_id]).expanduser().resolve()
        tree_digest_before = _recorded_run_tree_digest(run_root)
        execution = load_recorded_supervisor_execution(run_root)
        _validate_execution_configuration(execution, frozen)
        run_id = str(execution.run.to_mapping()["run_id"])
        if run_id in observed_run_ids or execution.run.digest in observed_run_digests:
            raise SupervisorEvaluationError("recorded_evaluation_run_identity_reused")
        observed_run_ids.add(run_id)
        observed_run_digests.add(execution.run.digest)
        observed_specialist_capabilities.update(
            str(row.to_mapping()["capability"]) for row in execution.invocation_manifests
        )
        manager_invocation_count += len(
            execution.report.to_mapping().get("manager_invocations") or []
        )
        report_path = case_root / f"{case_id}.json"
        agent_report = evaluate_supervisor_execution(
            execution,
            heldout[case_id],
            report_output_path=report_path,
            persist_replay_report=False,
        )
        agent_reports.append(agent_report)
        baseline_reports.append(
            {
                "case_id": case_id,
                "split": "heldout",
                "metrics": dict(heldout[case_id].baseline_metrics or {}),
                "zero_critical_boundary_violations": True,
            }
        )
        tree_digest_after = _recorded_run_tree_digest(run_root)
        if tree_digest_after != tree_digest_before:
            raise SupervisorEvaluationError("recorded_run_mutated_during_evaluation")
        run_references.append(
            {
                "case_id": case_id,
                "run_id": run_id,
                "supervisor_run_digest": execution.run.digest,
                "terminal_report_digest": execution.report.digest,
                "evaluation_report_path": str(report_path.relative_to(root)),
                "evaluation_report_digest": agent_report["evaluation_report_digest"],
                "recorded_run_tree_digest_before": tree_digest_before,
                "recorded_run_tree_digest_after": tree_digest_after,
            }
        )

    if observed_specialist_capabilities != {kind.value for kind in CapabilityKind}:
        raise SupervisorEvaluationError("recorded_evaluation_specialist_coverage_incomplete")
    if manager_invocation_count < len(heldout):
        raise SupervisorEvaluationError("recorded_evaluation_manager_coverage_incomplete")

    comparison = compare_supervisor_to_baseline(
        agent_reports,
        baseline_reports,
        minimum_improvement=float(corpus["minimum_required_improvement"]),
    )
    write_json(root / "baseline_comparison.json", comparison)
    value: dict[str, Any] = {
        "schema_version": RECORDED_SUPERVISOR_EVAL_BUNDLE_SCHEMA_VERSION,
        "status": "completed",
        "corpus_id": corpus["corpus_id"],
        "corpus_digest": corpus["corpus_digest"],
        "configuration_digest": frozen["configuration_digest"],
        "heldout_case_count": len(heldout),
        "specialist_capability_count": len(observed_specialist_capabilities),
        "all_six_specialists_exercised": True,
        "manager_invocation_count": manager_invocation_count,
        "run_references": run_references,
        "comparison_digest": comparison["comparison_digest"],
        "eligible_for_autonomy_promotion": comparison["eligible_for_autonomy_promotion"],
        "zero_critical_boundary_violations": comparison["zero_critical_boundary_violations"],
        "development_cases_excluded": True,
        "hidden_expected_properties_sent_to_agent": False,
        "agent_self_graded": False,
        "model_invoked_during_evaluation": False,
        "recorded_runs_mutated": False,
        "proof_effect": "none",
    }
    value["recorded_evaluation_bundle_digest"] = canonical_digest(
        value,
        digest_field="recorded_evaluation_bundle_digest",
    )
    write_json(root / "recorded_evaluation_bundle.json", value)
    return value


__all__ = [
    "RECORDED_SUPERVISOR_EVAL_BUNDLE_SCHEMA_VERSION",
    "SUPERVISOR_EVAL_CONFIGURATION_SCHEMA_VERSION",
    "evaluate_recorded_supervisor_corpus",
    "freeze_supervisor_evaluation_configuration",
    "load_recorded_supervisor_execution",
    "load_sealed_supervisor_evaluation_corpus",
    "validate_supervisor_evaluation_configuration",
]
