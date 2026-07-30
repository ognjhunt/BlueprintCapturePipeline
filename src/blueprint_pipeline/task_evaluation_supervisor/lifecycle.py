"""Required capture-build entrypoint for the Task Evaluation Supervisor."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

from ..agent_operator_runtime import LIVE_AGENTS_SDK_ENV, env_truthy
from ..decision_evidence_contracts import canonical_digest
from .agents_sdk import DEFAULT_SUPERVISOR_AGENT_MODEL
from .capabilities import SupervisorContext
from .capture_ingress import load_capture_build_ingress
from .contracts import AutonomyMode
from .supervisor import TaskEvaluationSupervisor


CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION = "task_evaluation_capture_supervisor_lifecycle.v3"
CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK_ENV = (
    "BLUEPRINT_CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK"
)
CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD_ENV = (
    "BLUEPRINT_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD"
)
CAPTURE_SUPERVISOR_AGENT_MODEL_ENV = "BLUEPRINT_CAPTURE_SUPERVISOR_AGENT_MODEL"
MAX_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD = 100.0


def capture_supervisor_execution_options_from_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load one strict service-side inference envelope for capture ingress."""

    source = os.environ if environ is None else environ
    raw_allow = str(source.get(CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK_ENV, "")).strip().lower()
    if raw_allow in {"", "0", "false", "no", "off"}:
        allow_live = False
    elif raw_allow in {"1", "true", "yes", "on"}:
        allow_live = True
    else:
        raise ValueError("capture_supervisor_live_agents_sdk_env_invalid")

    raw_budget = str(source.get(CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD_ENV, "")).strip()
    try:
        budget = 0.0 if not raw_budget else float(raw_budget)
    except ValueError as exc:
        raise ValueError("capture_supervisor_inference_budget_invalid") from exc
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("capture_supervisor_inference_budget_invalid")
    if budget > MAX_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD:
        raise ValueError("capture_supervisor_inference_budget_exceeds_service_ceiling")
    if allow_live and budget <= 0:
        raise ValueError("capture_supervisor_live_inference_requires_positive_budget")
    if not allow_live and budget != 0:
        raise ValueError("capture_supervisor_disabled_inference_budget_must_be_zero")

    model = str(source.get(CAPTURE_SUPERVISOR_AGENT_MODEL_ENV, "")).strip()
    if not model:
        model = DEFAULT_SUPERVISOR_AGENT_MODEL
    if len(model) > 256 or any(ord(character) < 32 for character in model):
        raise ValueError("capture_supervisor_agent_model_invalid")
    return {
        "agent_model": model,
        "allow_live_agents_sdk": allow_live,
        "agent_inference_budget_usd": budget,
    }


def capture_supervisor_execution_profile(
    *,
    agent_model: str = DEFAULT_SUPERVISOR_AGENT_MODEL,
    allow_live_agents_sdk: bool = False,
    agent_inference_budget_usd: float = 0.0,
) -> dict[str, Any]:
    """Bind the exact execution authority that determines lifecycle idempotency."""

    value = {
        "schema_version": "task_evaluation_capture_supervisor_execution_profile.v1",
        "agent_harness": "openai_agents_sdk",
        "agent_model": agent_model,
        "allow_live_agents_sdk": allow_live_agents_sdk,
        "live_operator_gate_enabled": (
            env_truthy(LIVE_AGENTS_SDK_ENV) if allow_live_agents_sdk else False
        ),
        "agent_inference_budget_usd": float(agent_inference_budget_usd),
        "autonomy_mode": AutonomyMode.EXECUTE_NON_SPEND.value,
    }
    value["execution_profile_digest"] = canonical_digest(
        value,
        digest_field="execution_profile_digest",
    )
    return value


def capture_supervisor_health_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a non-secret, fail-closed view of capture-supervisor readiness."""

    status = {
        "agent_harness": "openai_agents_sdk",
        "configuration_status": "invalid",
        "zero_spend_lifecycle_ready": False,
        "live_inference_configured": False,
        "live_operator_gate_enabled": False,
        "live_inference_ready": False,
        "execution_profile_digest": None,
        "proof_or_recovery_authority_granted": False,
    }
    try:
        options = capture_supervisor_execution_options_from_env(environ)
        profile = capture_supervisor_execution_profile(**options)
    except ValueError:
        return status
    live_configured = options["allow_live_agents_sdk"] is True
    live_gate = profile["live_operator_gate_enabled"] is True
    return status | {
        "configuration_status": "valid",
        "zero_spend_lifecycle_ready": True,
        "live_inference_configured": live_configured,
        "live_operator_gate_enabled": live_gate,
        "live_inference_ready": live_configured and live_gate,
        "execution_profile_digest": profile["execution_profile_digest"],
    }


def run_capture_build_supervisor(
    *,
    capture_root: str | Path,
    agent_model: str = DEFAULT_SUPERVISOR_AGENT_MODEL,
    allow_live_agents_sdk: bool = False,
    agent_inference_budget_usd: float = 0.0,
) -> dict[str, Any]:
    """Enter every completed capture build into the required supervisor.

    A capture is enough to create the run. Missing customer, task, embodiment,
    success, testbed, or rights details remain explicit blockers for the agents
    to clarify; they are never synthesized into authoritative facts.
    """

    source = Path(capture_root).expanduser().resolve()
    capture_build = load_capture_build_ingress(source)
    root = source if source.is_dir() else source.parent
    digest_suffix = str(capture_build["capture_build_digest"]).removeprefix("sha256:")[:24]
    execution_profile = capture_supervisor_execution_profile(
        agent_model=agent_model,
        allow_live_agents_sdk=allow_live_agents_sdk,
        agent_inference_budget_usd=agent_inference_budget_usd,
    )
    profile_suffix = str(execution_profile["execution_profile_digest"]).removeprefix("sha256:")[:16]
    run_id = f"capture-supervisor-v3-{digest_suffix}-{profile_suffix}"
    output_dir = root / "pipeline" / "task_evaluation_supervisor" / "runs" / run_id
    execution = TaskEvaluationSupervisor(
        agent_model=agent_model,
        allow_live_agents_sdk=allow_live_agents_sdk,
        agent_inference_budget_usd=agent_inference_budget_usd,
    ).run(
        SupervisorContext(
            run_id=run_id,
            customer_question=(
                "What task evaluations can this completed capture build support, and what "
                "customer decision, robot embodiment, task, success, rights, testbed, or "
                "evidence details must be clarified before Blueprint can decide?"
            ),
            capture_build=capture_build,
        ),
        output_dir=output_dir,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        resume=True,
    )
    report = execution.report.to_mapping()
    registered_capabilities = list(execution.run.to_mapping().get("capabilities") or [])
    return {
        "schema_version": CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION,
        "status": report["status"],
        "run_id": run_id,
        "capture_build_digest": capture_build["capture_build_digest"],
        "supervisor_run_digest": execution.run.digest,
        "terminal_report_digest": execution.report.digest,
        "output_dir": str(output_dir),
        "terminal_report_path": str(output_dir / "terminal_supervisor_report.json"),
        "customer_report_path": str(output_dir / "customer_decision_report.json"),
        "event_ledger_path": str(output_dir / "supervisor_events.jsonl"),
        "agent_harness": "openai_agents_sdk",
        "agent_model": agent_model,
        "execution_profile": execution_profile,
        "execution_profile_digest": execution_profile["execution_profile_digest"],
        "autonomy_mode": AutonomyMode.EXECUTE_NON_SPEND.value,
        "capability_count": len(execution.capability_results),
        "triggered_capability_count": len(execution.capability_results),
        "registered_capability_count": len(registered_capabilities),
        "all_six_capabilities_present": len(registered_capabilities) == 6,
        "all_six_capabilities_registered": len(registered_capabilities) == 6,
        "manager_invocation_count": int(
            (report.get("inference_spend") or {}).get("manager_invocation_count") or 0
        ),
        "agent_inference_started": (
            int((report.get("inference_spend") or {}).get("live_invocation_count") or 0) > 0
            or int((report.get("inference_spend") or {}).get("reservation_count") or 0) > 0
        ),
        "actions_executed": bool(report.get("actions_executed")),
        "registered_tool_reads_executed": int(report.get("registered_tool_reads_executed") or 0),
        "registered_non_spend_actions_executed": int(
            report.get("registered_non_spend_actions_executed") or 0
        ),
        "proof_state_mutated_by_agent": False,
        "capture_build_alone_can_start_run": True,
    }


__all__ = [
    "CAPTURE_SUPERVISOR_AGENT_MODEL_ENV",
    "CAPTURE_SUPERVISOR_ALLOW_LIVE_AGENTS_SDK_ENV",
    "CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD_ENV",
    "CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION",
    "MAX_CAPTURE_SUPERVISOR_INFERENCE_BUDGET_USD",
    "capture_supervisor_execution_options_from_env",
    "capture_supervisor_health_status",
    "capture_supervisor_execution_profile",
    "run_capture_build_supervisor",
]
