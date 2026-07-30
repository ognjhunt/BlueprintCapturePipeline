"""Required capture-build entrypoint for the Task Evaluation Supervisor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agent_operator_runtime import LIVE_AGENTS_SDK_ENV, env_truthy
from ..decision_evidence_contracts import canonical_digest
from .agents_sdk import DEFAULT_SUPERVISOR_AGENT_MODEL
from .capabilities import SupervisorContext
from .capture_ingress import load_capture_build_ingress
from .contracts import AutonomyMode
from .supervisor import TaskEvaluationSupervisor


CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION = "task_evaluation_capture_supervisor_lifecycle.v3"


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

    root = Path(capture_root).expanduser().resolve()
    capture_build = load_capture_build_ingress(root)
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
    "CAPTURE_SUPERVISOR_LIFECYCLE_SCHEMA_VERSION",
    "capture_supervisor_execution_profile",
    "run_capture_build_supervisor",
]
