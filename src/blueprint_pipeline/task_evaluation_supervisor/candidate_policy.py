"""Phase 4 frozen candidate PolicyAdapter and hidden-evaluation separation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

from ..common import read_json, write_json
from ..decision_evidence_contracts import canonical_digest
from ..evaluation_run_contract import validate_evaluation_run_spec


CANDIDATE_POLICY_MANIFEST_SCHEMA_VERSION = "task_evaluation_candidate_policy_manifest.v1"
CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION = "task_evaluation_candidate_policy_suite.v1"
CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION = (
    "task_evaluation_candidate_policy_execution.v1"
)
_STACK_TYPES = {"direct_policy", "decomposed_planner_policy", "verify_recover_supervisor"}
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CandidatePolicyError(ValueError):
    """Raised when a candidate could access or influence held-out evaluation."""


class CandidatePolicyRuntime(Protocol):
    candidate_id: str
    candidate_policy_manifest_digest: str

    def execute(
        self,
        *,
        evaluation_run_spec: Mapping[str, Any],
        output_dir: Path,
    ) -> Mapping[str, Any]: ...


class IndependentCandidateEvaluator(Protocol):
    provider_id: str
    evaluator_digest: str

    def evaluate(
        self,
        *,
        candidate_id: str,
        trace: Mapping[str, Any],
        hidden_evaluation_manifest: Mapping[str, Any],
        success_predicate_digest: str,
    ) -> Mapping[str, Any]: ...


def _digest(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _SHA256_DIGEST.fullmatch(text):
        raise CandidatePolicyError(f"{field}:invalid_digest")
    return text


def _frozen_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidatePolicyError("candidate_frozen_at_invalid") from exc
    if parsed.tzinfo is None:
        raise CandidatePolicyError("candidate_frozen_at_timezone_required")
    return parsed.astimezone(timezone.utc).isoformat()


def freeze_candidate_policy_manifest(
    *,
    candidate_id: str,
    stack_type: str,
    code_digest: str,
    model_provider: str,
    model_id: str,
    model_version: str,
    prompt_digest: str,
    tool_registry_digest: str,
    memory_skill_snapshot_digest: str,
    max_cost_usd: float,
    retry_limit: int,
    observation_schema_ref: str,
    action_schema_ref: str,
    frozen_at: str,
) -> dict[str, Any]:
    if stack_type not in _STACK_TYPES:
        raise CandidatePolicyError("candidate_stack_type_invalid")
    if (
        not _CANDIDATE_ID.fullmatch(candidate_id)
        or not model_provider.strip()
        or not model_id.strip()
        or not model_version.strip()
        or not observation_schema_ref.strip()
        or not action_schema_ref.strip()
        or not frozen_at.strip()
    ):
        raise CandidatePolicyError("candidate_manifest_missing_fields")
    if (
        isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, (int, float))
        or not math.isfinite(float(max_cost_usd))
        or float(max_cost_usd) < 0
        or isinstance(retry_limit, bool)
        or not isinstance(retry_limit, int)
        or retry_limit < 0
    ):
        raise CandidatePolicyError("candidate_budget_or_retry_invalid")
    value: dict[str, Any] = {
        "schema_version": CANDIDATE_POLICY_MANIFEST_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "stack_type": stack_type,
        "code_digest": _digest(code_digest, field="code_digest"),
        "model_provider": model_provider,
        "model_id": model_id,
        "model_version": model_version,
        "prompt_digest": _digest(prompt_digest, field="prompt_digest"),
        "tool_registry_digest": _digest(tool_registry_digest, field="tool_registry_digest"),
        "memory_skill_snapshot_digest": _digest(
            memory_skill_snapshot_digest,
            field="memory_skill_snapshot_digest",
        ),
        "max_cost_usd": float(max_cost_usd),
        "retry_limit": retry_limit,
        "observation_schema_ref": observation_schema_ref,
        "action_schema_ref": action_schema_ref,
        "frozen_at": _frozen_time(frozen_at),
        "frozen_before_hidden_evaluation": True,
        "hidden_labels_included": False,
        "evaluator_configuration_included": False,
        "success_predicate_mutable_by_candidate": False,
        "candidate_may_grade_itself": False,
        "proof_effect": "none",
    }
    value["candidate_policy_manifest_digest"] = canonical_digest(
        value,
        digest_field="candidate_policy_manifest_digest",
    )
    return value


@dataclass(frozen=True)
class FrozenAgenticPolicyAdapter:
    """Replaceable adapter payload supplied to candidate execution only."""

    manifest: Mapping[str, Any]
    adapter_id: str = "blueprint_agentic_candidate_policy"
    adapter_version: str = "1"

    def __post_init__(self) -> None:
        if self.manifest.get("schema_version") != CANDIDATE_POLICY_MANIFEST_SCHEMA_VERSION:
            raise CandidatePolicyError("candidate_policy_manifest_schema_invalid")
        expected = canonical_digest(
            self.manifest,
            digest_field="candidate_policy_manifest_digest",
        )
        if self.manifest.get("candidate_policy_manifest_digest") != expected:
            raise CandidatePolicyError("candidate_policy_manifest_digest_mismatch")
        if self.manifest.get("frozen_before_hidden_evaluation") is not True:
            raise CandidatePolicyError("candidate_policy_not_frozen")
        if self.manifest.get("hidden_labels_included") is not False:
            raise CandidatePolicyError("candidate_policy_contains_hidden_labels")
        if (
            self.manifest.get("evaluator_configuration_included") is not False
            or self.manifest.get("success_predicate_mutable_by_candidate") is not False
            or self.manifest.get("candidate_may_grade_itself") is not False
            or self.manifest.get("proof_effect") != "none"
        ):
            raise CandidatePolicyError("candidate_policy_authority_boundary_invalid")

    def to_policy_adapter_mapping(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "policy_id": self.manifest["candidate_id"],
            "candidate_policy_manifest_digest": self.manifest[
                "candidate_policy_manifest_digest"
            ],
            "stack_type": self.manifest["stack_type"],
            "observation_schema_ref": self.manifest["observation_schema_ref"],
            "action_schema_ref": self.manifest["action_schema_ref"],
            "max_cost_usd": self.manifest["max_cost_usd"],
            "retry_limit": self.manifest["retry_limit"],
            "hidden_labels_included": False,
            "evaluator_authority": False,
            "proof_authority": False,
        }


def compile_neutral_candidate_policy_suite(
    *,
    base_evaluation_run_spec: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    frozen_scenario_manifest: Mapping[str, Any],
    evaluator_provider_id: str,
) -> dict[str, Any]:
    if not evaluator_provider_id.strip():
        raise CandidatePolicyError("evaluator_provider_id_missing")
    scenario_digest = canonical_digest(
        frozen_scenario_manifest,
        digest_field="frozen_scenario_manifest_digest",
    )
    if frozen_scenario_manifest.get("frozen_scenario_manifest_digest") != scenario_digest:
        raise CandidatePolicyError("frozen_scenario_manifest_digest_mismatch")
    if frozen_scenario_manifest.get("frozen") is not True:
        raise CandidatePolicyError("scenario_manifest_not_frozen")
    if frozen_scenario_manifest.get("hidden_labels_included") is not False:
        raise CandidatePolicyError("scenario_manifest_exposes_hidden_labels")
    scenario_ids = [str(row) for row in frozen_scenario_manifest.get("scenario_ids") or []]
    if not scenario_ids or len(scenario_ids) != len(set(scenario_ids)) or any(
        not row for row in scenario_ids
    ):
        raise CandidatePolicyError("frozen_scenario_ids_invalid")
    required_types = _STACK_TYPES
    candidate_types = {str(row.get("stack_type") or "") for row in candidates}
    if candidate_types != required_types or len(candidates) != len(required_types):
        raise CandidatePolicyError("neutral_suite_requires_three_stack_types")
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CandidatePolicyError("neutral_suite_candidate_id_duplicate")

    evaluator_digest = _digest(
        frozen_scenario_manifest.get("evaluator_digest"),
        field="evaluator_digest",
    )
    success_digest = _digest(
        frozen_scenario_manifest.get("success_predicate_digest"),
        field="success_predicate_digest",
    )
    compiled: list[dict[str, Any]] = []
    for manifest in sorted(candidates, key=lambda row: str(row.get("candidate_id") or "")):
        adapter = FrozenAgenticPolicyAdapter(manifest)
        if str(manifest.get("model_provider")) == evaluator_provider_id:
            raise CandidatePolicyError("candidate_provider_self_grading_forbidden")
        spec = copy.deepcopy(dict(base_evaluation_run_spec))
        spec["run_id"] = f"{base_evaluation_run_spec['run_id']}-{manifest['candidate_id']}"
        spec["policy_adapter"] = adapter.to_policy_adapter_mapping()
        task_pack = dict(spec.get("task_scenario_pack") or {})
        task_pack["frozen_scenario_manifest_digest"] = scenario_digest
        task_pack["scenario_ids"] = list(frozen_scenario_manifest.get("scenario_ids") or [])
        task_pack["hidden_labels_included"] = False
        spec["task_scenario_pack"] = task_pack
        proof_contract = dict(spec.get("proof_contract") or {})
        proof_contract["evaluator_digest"] = evaluator_digest
        proof_contract["success_predicate_digest"] = success_digest
        proof_contract["candidate_may_modify"] = False
        spec["proof_contract"] = proof_contract
        metadata = dict(spec.get("metadata") or {})
        metadata.update(
            {
                "candidate_policy_manifest_digest": manifest[
                    "candidate_policy_manifest_digest"
                ],
                "frozen_scenario_manifest_digest": scenario_digest,
                "evaluator_provider_id": evaluator_provider_id,
                "candidate_results_visible_to_scenario_generator": False,
                "candidate_self_grading": False,
                "simulation_only_unless_physical_evidence_joined": True,
            }
        )
        spec["metadata"] = metadata
        validation = validate_evaluation_run_spec(spec)
        if validation.get("status") != "passed":
            raise CandidatePolicyError(
                "candidate_evaluation_run_invalid:"
                + ",".join(str(row) for row in validation.get("errors") or [])
            )
        compiled.append(spec)

    value: dict[str, Any] = {
        "schema_version": CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION,
        "frozen_scenario_manifest_digest": scenario_digest,
        "evaluator_digest": evaluator_digest,
        "success_predicate_digest": success_digest,
        "hidden_label_manifest_digest": _digest(
            frozen_scenario_manifest.get("hidden_label_manifest_digest"),
            field="hidden_label_manifest_digest",
        ),
        "evaluator_provider_id": evaluator_provider_id,
        "candidate_evaluation_run_specs": compiled,
        "candidate_count": len(compiled),
        "same_scenarios_for_every_candidate": True,
        "same_evaluator_for_every_candidate": True,
        "same_success_predicates_for_every_candidate": True,
        "hidden_labels_sent_to_candidates": False,
        "candidate_agents_control_evaluator": False,
        "candidate_agents_grade_themselves": False,
        "development_repair_during_hidden_evaluation": False,
        "claim_ceiling": "simulation_only_unless_qualified_physical_evidence_is_joined",
        "provider_execution_started": False,
        "proof_effect": "none",
    }
    value["candidate_evaluation_suite_digest"] = canonical_digest(
        value,
        digest_field="candidate_evaluation_suite_digest",
    )
    return value


def execute_neutral_candidate_policy_suite(
    suite: Mapping[str, Any],
    *,
    candidate_runtimes: Sequence[CandidatePolicyRuntime],
    evaluator: IndependentCandidateEvaluator,
    hidden_evaluation_manifest: Mapping[str, Any],
    output_dir: str | Path,
    allow_execution: bool = False,
) -> dict[str, Any]:
    """Execute frozen candidates and grade traces at an independent hidden boundary."""

    expected_suite_digest = canonical_digest(
        suite,
        digest_field="candidate_evaluation_suite_digest",
    )
    if suite.get("schema_version") != CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION:
        raise CandidatePolicyError("candidate_evaluation_suite_schema_invalid")
    if suite.get("candidate_evaluation_suite_digest") != expected_suite_digest:
        raise CandidatePolicyError("candidate_evaluation_suite_digest_mismatch")
    if (
        suite.get("hidden_labels_sent_to_candidates") is not False
        or suite.get("candidate_agents_control_evaluator") is not False
        or suite.get("candidate_agents_grade_themselves") is not False
        or suite.get("development_repair_during_hidden_evaluation") is not False
    ):
        raise CandidatePolicyError("candidate_evaluation_suite_boundary_invalid")
    hidden_digest = canonical_digest(hidden_evaluation_manifest)
    if hidden_digest != suite.get("hidden_label_manifest_digest"):
        raise CandidatePolicyError("hidden_evaluation_manifest_digest_mismatch")
    if evaluator.provider_id != suite.get("evaluator_provider_id"):
        raise CandidatePolicyError("independent_evaluator_provider_mismatch")
    if evaluator.evaluator_digest != suite.get("evaluator_digest"):
        raise CandidatePolicyError("independent_evaluator_digest_mismatch")

    specs = [
        dict(row)
        for row in suite.get("candidate_evaluation_run_specs") or []
        if isinstance(row, Mapping)
    ]
    expected_candidates = {
        str((row.get("policy_adapter") or {}).get("policy_id") or ""): str(
            (row.get("metadata") or {}).get("candidate_policy_manifest_digest") or ""
        )
        for row in specs
    }
    runtimes = {runtime.candidate_id: runtime for runtime in candidate_runtimes}
    if (
        len(specs) != int(suite.get("candidate_count") or 0)
        or not expected_candidates
        or "" in expected_candidates
        or set(runtimes) != set(expected_candidates)
        or len(runtimes) != len(candidate_runtimes)
    ):
        raise CandidatePolicyError("candidate_runtime_set_mismatch")
    for candidate_id, runtime in runtimes.items():
        if runtime.candidate_policy_manifest_digest != expected_candidates[candidate_id]:
            raise CandidatePolicyError("candidate_runtime_manifest_digest_mismatch")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not allow_execution:
        value: dict[str, Any] = {
            "schema_version": CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION,
            "status": "prepared",
            "candidate_evaluation_suite_digest": expected_suite_digest,
            "execution_started": False,
            "candidate_results": [],
            "hidden_evaluation_manifest_digest": hidden_digest,
            "hidden_labels_sent_to_candidates": False,
            "candidate_agents_grade_themselves": False,
            "proof_effect": "none",
        }
        value["candidate_evaluation_execution_digest"] = canonical_digest(
            value,
            digest_field="candidate_evaluation_execution_digest",
        )
        write_json(root / "candidate_evaluation_execution.json", value)
        return value

    results: list[dict[str, Any]] = []
    allowed_runtime_keys = {
        "schema_version",
        "status",
        "trace_artifact_path",
        "trace_artifact_digest",
        "blockers",
        "cost_usd",
        "duration_seconds",
        "provider_execution_started",
        "attempt_count",
    }
    hidden_markers: set[str] = set()

    def collect_hidden_markers(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                collect_hidden_markers(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_hidden_markers(nested)
        elif isinstance(value, str) and len(value) >= 8:
            hidden_markers.add(value)

    collect_hidden_markers(hidden_evaluation_manifest)
    for spec in sorted(specs, key=lambda row: str(row.get("run_id") or "")):
        policy = dict(spec.get("policy_adapter") or {})
        candidate_id = str(policy.get("policy_id") or "")
        if policy.get("hidden_labels_included") is not False:
            raise CandidatePolicyError("candidate_spec_exposes_hidden_labels")
        candidate_root = (root / "candidates" / candidate_id).resolve()
        if root not in candidate_root.parents:
            raise CandidatePolicyError("candidate_output_path_escape")
        candidate_root.mkdir(parents=True, exist_ok=True)
        try:
            runtime_result = dict(
                runtimes[candidate_id].execute(
                    evaluation_run_spec=copy.deepcopy(spec),
                    output_dir=candidate_root,
                )
            )
        except Exception as exc:  # noqa: BLE001 - typed failure, no raw message
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "failure_type": "candidate_runtime_exception",
                    "exception_type": type(exc).__name__,
                    "evaluated": False,
                    "proof_effect": "none",
                }
            )
            continue
        if set(runtime_result) - allowed_runtime_keys:
            raise CandidatePolicyError("candidate_runtime_result_contains_unregistered_fields")
        try:
            runtime_cost = float(runtime_result.get("cost_usd") or 0.0)
            runtime_duration = float(runtime_result.get("duration_seconds") or 0.0)
            attempt_count = int(runtime_result.get("attempt_count") or 1)
        except (TypeError, ValueError) as exc:
            raise CandidatePolicyError("candidate_runtime_accounting_invalid") from exc
        if (
            not math.isfinite(runtime_cost)
            or runtime_cost < 0
            or runtime_cost > float(policy.get("max_cost_usd") or 0.0)
            or not math.isfinite(runtime_duration)
            or runtime_duration < 0
            or attempt_count < 1
            or attempt_count > int(policy.get("retry_limit") or 0) + 1
            or not isinstance(runtime_result.get("provider_execution_started"), bool)
        ):
            raise CandidatePolicyError("candidate_runtime_accounting_invalid")
        if runtime_result.get("status") != "completed":
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": str(runtime_result.get("status") or "failed"),
                    "blockers": list(runtime_result.get("blockers") or []),
                    "evaluated": False,
                    "proof_effect": "none",
                }
            )
            continue
        trace_path = (candidate_root / str(runtime_result.get("trace_artifact_path") or "")).resolve()
        if candidate_root not in trace_path.parents or not trace_path.is_file():
            raise CandidatePolicyError("candidate_trace_path_invalid")
        trace = read_json(trace_path)
        trace_digest = canonical_digest(trace)
        if runtime_result.get("trace_artifact_digest") != trace_digest:
            raise CandidatePolicyError("candidate_trace_digest_mismatch")
        serialized_trace = json.dumps(trace, sort_keys=True)
        if any(marker in serialized_trace for marker in hidden_markers):
            raise CandidatePolicyError("candidate_trace_hidden_label_leakage")
        evaluation = dict(
            evaluator.evaluate(
                candidate_id=candidate_id,
                trace=trace,
                hidden_evaluation_manifest=dict(hidden_evaluation_manifest),
                success_predicate_digest=str(suite["success_predicate_digest"]),
            )
        )
        allowed_evaluation_keys = {
            "schema_version",
            "candidate_id",
            "status",
            "outcome",
            "metrics",
            "decisive_evidence",
            "uncertainty",
            "blockers",
            "evaluator_digest",
            "success_predicate_digest",
            "candidate_self_graded",
            "physical_validation_proven",
            "claim_ceiling",
        }
        if set(evaluation) - allowed_evaluation_keys:
            raise CandidatePolicyError("independent_candidate_evaluation_unregistered_fields")
        if (
            evaluation.get("schema_version") != "candidate_policy_independent_evaluation.v1"
            or evaluation.get("candidate_id") != candidate_id
            or evaluation.get("evaluator_digest") != suite.get("evaluator_digest")
            or evaluation.get("success_predicate_digest")
            != suite.get("success_predicate_digest")
            or evaluation.get("candidate_self_graded") is not False
            or evaluation.get("physical_validation_proven") is not False
            or evaluation.get("status") not in {"completed", "blocked"}
            or evaluation.get("outcome")
            not in {"passed", "failed", "inconclusive", "abstention"}
        ):
            raise CandidatePolicyError("independent_candidate_evaluation_invalid")
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, Mapping) or any(
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            for metric in metrics.values()
        ):
            raise CandidatePolicyError("independent_candidate_evaluation_metrics_invalid")
        serialized_evaluation = json.dumps(evaluation, sort_keys=True)
        if any(marker in serialized_evaluation for marker in hidden_markers):
            raise CandidatePolicyError("independent_candidate_evaluation_hidden_label_leakage")
        evaluation["independent_evaluation_digest"] = canonical_digest(
            evaluation,
            digest_field="independent_evaluation_digest",
        )
        evaluation_path = candidate_root / "independent_evaluation.json"
        write_json(evaluation_path, evaluation)
        results.append(
            {
                "candidate_id": candidate_id,
                "status": "evaluated",
                "trace_artifact_digest": trace_digest,
                "independent_evaluation_digest": evaluation[
                    "independent_evaluation_digest"
                ],
                "outcome": evaluation.get("outcome"),
                "claim_ceiling": evaluation.get("claim_ceiling"),
                "candidate_self_graded": False,
                "hidden_labels_sent_to_candidate": False,
                "proof_effect": "none",
            }
        )

    value = {
        "schema_version": CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION,
        "status": (
            "completed"
            if results and all(row.get("status") == "evaluated" for row in results)
            else "partial"
        ),
        "candidate_evaluation_suite_digest": expected_suite_digest,
        "execution_started": True,
        "candidate_results": results,
        "hidden_evaluation_manifest_digest": hidden_digest,
        "hidden_labels_sent_to_candidates": False,
        "candidate_agents_grade_themselves": False,
        "independent_evaluator_provider_id": evaluator.provider_id,
        "independent_evaluator_digest": evaluator.evaluator_digest,
        "claim_ceiling": suite.get("claim_ceiling"),
        "physical_validation_proven": False,
        "deployment_approval_proven": False,
        "proof_effect": "none",
    }
    value["candidate_evaluation_execution_digest"] = canonical_digest(
        value,
        digest_field="candidate_evaluation_execution_digest",
    )
    write_json(root / "candidate_evaluation_execution.json", value)
    return value


__all__ = [
    "CANDIDATE_EVALUATION_SUITE_SCHEMA_VERSION",
    "CANDIDATE_EVALUATION_EXECUTION_SCHEMA_VERSION",
    "CANDIDATE_POLICY_MANIFEST_SCHEMA_VERSION",
    "CandidatePolicyError",
    "FrozenAgenticPolicyAdapter",
    "IndependentCandidateEvaluator",
    "CandidatePolicyRuntime",
    "compile_neutral_candidate_policy_suite",
    "execute_neutral_candidate_policy_suite",
    "freeze_candidate_policy_manifest",
]
