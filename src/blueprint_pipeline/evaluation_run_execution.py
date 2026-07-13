"""Execution authority for compiled, six-part Evaluation Runs.

Compilation and execution remain separate.  This module performs no runtime
work unless an explicit execution gate is supplied, resolves exactly one
registered execution adapter, and binds every result to the compiled spec
digest.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .claim_contract_keys import PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY
from .common import read_json, utc_now_iso, write_json
from .evaluation_run_contract import (
    EvaluationRunAdapterRegistry,
    EvaluationRunSpec,
    compile_evaluation_run,
)


EVALUATION_RUN_EXECUTION_SCHEMA_VERSION = "evaluation_run_execution.v1"


class EvaluationRunExecutionAdapter(Protocol):
    """Runtime port implemented by concrete provider/orchestrator adapters."""

    adapter_id: str

    def execute(
        self,
        *,
        spec: EvaluationRunSpec,
        output_dir: Path,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class EvaluationRunExecutionRegistry:
    """Resolve execution implementations independently of component adapters."""

    def __init__(self, adapters: Sequence[EvaluationRunExecutionAdapter] = ()) -> None:
        self._adapters: dict[str, EvaluationRunExecutionAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: EvaluationRunExecutionAdapter) -> None:
        adapter_id = str(getattr(adapter, "adapter_id", "") or "").strip()
        if not adapter_id:
            raise ValueError("evaluation_run_execution_adapter_id_missing")
        if adapter_id in self._adapters:
            raise ValueError(f"duplicate_evaluation_run_execution_adapter:{adapter_id}")
        self._adapters[adapter_id] = adapter

    def resolve(self, adapter_id: str) -> EvaluationRunExecutionAdapter | None:
        return self._adapters.get(str(adapter_id or "").strip())

    def manifest(self) -> list[str]:
        return sorted(self._adapters)


@dataclass(frozen=True)
class EvaluationRunExecutionResult:
    manifest: Mapping[str, Any]
    adapter_result: Mapping[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            **dict(self.manifest),
            "adapter_result": dict(self.adapter_result),
        }


def default_evaluation_run_execution_registry() -> EvaluationRunExecutionRegistry:
    # Resolve concrete adapters only when execution is requested.  Keeping
    # their module names as data prevents CPU-only control-plane imports from
    # acquiring the kitchen/provider hot lane transitively.
    package = __package__ or "blueprint_pipeline"
    robot_module = importlib.import_module(
        f"{package}.robot_eval_evaluation_run_adapter"
    )
    kitchen_module = importlib.import_module(
        f"{package}.g1_kitchen_evaluation_run_adapter"
    )

    return EvaluationRunExecutionRegistry(
        [
            robot_module.RobotEvalEvaluationRunExecutor(),
            kitchen_module.G1KitchenEvaluationRunExecutor(),
        ]
    )


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_string(item) for item in value if _string(item)]
    return []


def _adapter_result_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
        "blockers": _string_list(value.get("blockers")),
        "manifest_path": value.get("manifest_path"),
        "job_dir": value.get("job_dir"),
        "raw_result_persisted": False,
    }


def _proof_contract_evaluation(
    spec: EvaluationRunSpec,
    adapter_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate declared evidence IDs without persisting adapter evidence values."""

    required = _string_list(spec.proof_contract.get("required_evidence"))
    raw_evidence = adapter_result.get("evaluation_run_proof_evidence")
    evidence = dict(raw_evidence) if isinstance(raw_evidence, Mapping) else {}
    satisfied: list[str] = []
    for evidence_id in required:
        value = evidence.get(evidence_id)
        if value is True or (
            isinstance(value, Mapping) and value.get("satisfied") is True
        ):
            satisfied.append(evidence_id)
    missing = [value for value in required if value not in satisfied]
    return {
        "contract_id": spec.proof_contract.get("contract_id"),
        "status": "passed" if not missing else "evidence_incomplete",
        "required_evidence": required,
        "satisfied_evidence": satisfied,
        "missing_evidence": missing,
        "adapter_evidence_values_persisted": False,
        "claim_ceiling": dict(spec.proof_contract.get("claim_ceiling") or {}),
        "prohibited_claims": _string_list(
            spec.proof_contract.get("prohibited_claims")
        ),
        PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY: bool(required) and not missing,
    }


def execute_evaluation_run(
    value: Mapping[str, Any],
    *,
    output_dir: str | Path,
    allow_execution: bool = False,
    context: Mapping[str, Any] | None = None,
    component_registry: EvaluationRunAdapterRegistry | None = None,
    execution_registry: EvaluationRunExecutionRegistry | None = None,
    generated_at: str | None = None,
) -> EvaluationRunExecutionResult:
    """Compile, resolve, and optionally execute one Evaluation Run.

    ``context`` carries local materialization paths, ephemeral transport
    credentials, and explicit runtime gates.  Its values are never persisted.
    """

    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    generated = generated_at or utc_now_iso()
    plan = compile_evaluation_run(
        value,
        output_dir=root,
        generated_at=generated,
        adapter_registry=component_registry,
    )
    spec = EvaluationRunSpec.from_mapping(value)
    adapter_id = _string(spec.runtime_provider_profile.get("execution_adapter_id"))
    registry = execution_registry or default_evaluation_run_execution_registry()
    blockers: list[str] = []
    if plan["status"] != "prepared":
        blockers.append("evaluation_run_plan_blocked")
        blockers.extend(plan["validation"]["errors"])
    if not adapter_id:
        blockers.append("evaluation_run_execution_adapter_missing")
    adapter = registry.resolve(adapter_id) if adapter_id else None
    if adapter_id and adapter is None:
        blockers.append(f"evaluation_run_execution_adapter_unavailable:{adapter_id}")
    if not allow_execution:
        blockers.append("evaluation_run_execution_not_authorized")

    adapter_result: Mapping[str, Any] = {}
    execution_started = False
    if not blockers and adapter is not None:
        execution_started = True
        adapter_context = {
            **dict(context or {}),
            "_evaluation_run_binding": {
                "run_id": spec.run_id,
                "spec_digest": plan["spec_digest"],
                "plan_path": str(root / "evaluation_run_plan.json"),
            },
        }
        try:
            adapter_result = adapter.execute(
                spec=spec,
                output_dir=root,
                context=adapter_context,
            )
        except Exception as exc:  # noqa: BLE001 - terminal evidence must still be written
            adapter_result = {
                "schema_version": "evaluation_run_execution_adapter_error.v1",
                "status": "failed",
                "blockers": ["evaluation_run_execution_adapter_raised"],
                "error_type": type(exc).__name__,
                "raw_error_message_recorded": False,
            }
        adapter_status = _string(adapter_result.get("status"))
        if adapter_status in {"blocked", "failed", "error"}:
            blockers.append("evaluation_run_execution_adapter_blocked")
            blockers.extend(_string_list(adapter_result.get("blockers")))

    resolution_blocked = (
        plan["status"] != "prepared" or not adapter_id or adapter is None
    )
    if resolution_blocked:
        status = "blocked"
    elif blockers:
        status = "blocked" if allow_execution else "prepared"
    else:
        status = _string(adapter_result.get("status")) or "completed"
    proof_evaluation = _proof_contract_evaluation(spec, adapter_result)
    manifest = {
        "schema_version": EVALUATION_RUN_EXECUTION_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "run_id": spec.run_id or None,
        "mode": spec.mode,
        "spec_digest": plan["spec_digest"],
        "plan_path": "evaluation_run_plan.json",
        "spec_path": "evaluation_run_spec.json",
        "execution_adapter_id": adapter_id or None,
        "registered_execution_adapters": registry.manifest(),
        "allow_execution": bool(allow_execution),
        "execution_started": execution_started,
        "context_keys_supplied": sorted(str(key) for key in (context or {})),
        "context_values_persisted": False,
        "blockers": sorted(set(blockers)),
        "adapter_result_summary": _adapter_result_summary(adapter_result),
        "proof_contract_evaluation": proof_evaluation,
        "claim_boundary": {
            "compiled_spec_digest_binds_execution": True,
            "execution_manifest_alone_is_not_task_success_proof": True,
            "adapter_result_must_satisfy_proof_contract": True,
            PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY: proof_evaluation[
                PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY
            ],
            "physical_robot_readiness_proven": False,
            "deployment_approval_proven": False,
            "raw_context_values_recorded": False,
        },
    }
    write_json(root / "evaluation_run_execution.json", manifest)
    return EvaluationRunExecutionResult(manifest=manifest, adapter_result=adapter_result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--allow-execution", action="store_true")
    args = parser.parse_args(argv)
    context = read_json(args.context) if args.context else {}
    result = execute_evaluation_run(
        read_json(args.spec),
        output_dir=args.output_dir,
        allow_execution=args.allow_execution,
        context=context,
    )
    # The adapter result can contain provider-specific details.  The durable,
    # redacted execution manifest is the public CLI response; callers that
    # need the in-process result use ``execute_evaluation_run`` directly.
    payload = dict(result.manifest)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] not in {"blocked", "failed", "error"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
