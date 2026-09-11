"""Prepare an immutable operational task using the configured server scope.

This trusted control-plane entrypoint does not call a model or launch a paid
worker. It binds retained inputs, the current code, tools and inference budget.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from ..common import write_json
from ..task_evaluation_stage_replay import (
    DEFAULT_APPROVED_ROOTS, DEFAULT_INPUT_ROOT, DEFAULT_PARENT_QUEUE_ROOT, DEFAULT_QUEUE_ROOT, locate_child,
)
from ..task_evaluation_supervisor.capabilities import SupervisorContext
from ..task_evaluation_supervisor.contracts import AuthorityEnvelope, AutonomyMode
from ..task_evaluation_supervisor.supervisor import default_authority_envelope
from .contracts import AgentAdmission, AgentExecutionError, AgentTask, RUNTIME_API, RUNTIME_SDK, digest
from .production import OperationalDiagnosis, ProductionAgentService, TaskRecord, configured_service
from .stage_recovery import StageReplayBinding, StageReplayTools, summarize_report
from .supervisor_bridge import context_revision
from .controller_recovery import ControllerRecoveryBinding, ControllerRecoveryTools, validate_controller_binding


def prepare_retained_failure(
    service: ProductionAgentService, *, task_id: str, run_id: str, child_id: str, owner_client_id: str,
    inference_budget_usd: float, runtime: str = RUNTIME_SDK, model: str | None = None,
    queue_root: Path = DEFAULT_QUEUE_ROOT, parent_queue_root: Path = DEFAULT_PARENT_QUEUE_ROOT,
    input_root: Path = DEFAULT_INPUT_ROOT, approved_roots: tuple[Path, ...] = DEFAULT_APPROVED_ROOTS,
    ttl_seconds: int = 600, autostart: bool = True,
    controller_recovery: ControllerRecoveryBinding | None = None,
    persist: bool = True,
) -> TaskRecord:
    if (not math.isfinite(inference_budget_usd) or not 0 < inference_budget_usd <= service.config.max_task_budget_usd
            or not 1 <= ttl_seconds <= 1800 or (model and model not in service.config.allowed_models)
            or runtime not in {RUNTIME_SDK, RUNTIME_API}
            or (runtime == RUNTIME_API and not service.config.managed_api_enabled)):
        raise AgentExecutionError("agent_preparation_outside_server_scope")
    located = locate_child(queue_root, child_id)
    if located.state not in {"failed", "completed"}:
        raise AgentExecutionError("agent_recovery_requires_terminal_saved_child")
    raw_job = located.job_path.read_bytes()
    job_digest = "sha256:" + hashlib.sha256(raw_job).hexdigest()
    job = json.loads(raw_job)
    saved = json.loads(located.result_path.read_text()) if located.result_path.is_file() else {}
    binding = StageReplayBinding(
        replay_id="failed_boundary", child_id=child_id, job_sha256=job_digest,
        queue_root=str(queue_root.resolve()), parent_queue_root=str(parent_queue_root.resolve()),
        input_root=str(input_root.resolve()), approved_roots=tuple(str(path.resolve()) for path in approved_roots),
        timeout_seconds=min(300, ttl_seconds),
    )
    tools = StageReplayTools(journal=service.journal, bindings=(binding,), source_commit=service.config.source_commit).tools()
    recovery = (controller_recovery,) if controller_recovery is not None else ()
    recovery_authority = {}
    extra_digests = []
    if controller_recovery is not None:
        _, intent = validate_controller_binding(controller_recovery)
        if (controller_recovery.required_replay_id != binding.replay_id
                or controller_recovery.parent_request_digest != job["parent_request_digest"]):
            raise AgentExecutionError("agent_recovery_failed_child_lineage_mismatch")
        execution = intent["request"]["execution"]
        recovery_authority = {"action_max_cost_usd": execution["max_total_spend_usd"],
            "action_max_retries": execution["max_retries"], "preauthorization_receipt_digest": intent["intent_digest"],
            "preauthorization_expires_at": datetime.fromtimestamp(execution["expires_at_epoch"], timezone.utc).isoformat()}
        extra_digests = [controller_recovery.intent_digest, controller_recovery.controller_config_sha256,
                         controller_recovery.preparation_link_sha256]
        tools = (*tools, *ControllerRecoveryTools(service.journal, recovery, service.config.source_commit).tools())
    summary = summarize_report({**saved, "phase": job.get("phase")}, replay_id=binding.replay_id,
                               job_sha256=job_digest, source_commit=service.config.source_commit)
    summary["record_kind"] = "retained_failure_summary_before_replay"
    source_commit = str(job.get("expected_source_commit", ""))
    if len(source_commit) == 40 and all(letter in "0123456789abcdef" for letter in source_commit):
        summary["saved_job_source_commit"] = source_commit
    summary_digest = digest(summary)
    authority = default_authority_envelope(
        run_id=run_id, mode=AutonomyMode.EXECUTE_PREAUTHORIZED if recovery else AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=service.registry, immutable_input_digests=[summary_digest, job_digest, *extra_digests],
        agent_inference_budget_usd=inference_budget_usd, allow_agent_inference=True, **recovery_authority,
    ).to_mapping()
    authority.pop("authority_digest")
    authority["allowed_tool_ids"] = [tool.tool_id for tool in tools]
    authority = AuthorityEnvelope.from_mapping(authority).to_mapping()
    question = "Investigate the saved failure using its isolated retained-stage replay and identify the supported next action."
    context = SupervisorContext(run_id=run_id, customer_question=question, authority_envelope=authority,
                                supervisor_output_dir=str(service.journal.root / "supervisor" / task_id))
    payload = [{"role": "user", "content": json.dumps({
        "question": question, "run_id": run_id, "source_commit": service.config.source_commit,
        "context_revision": context_revision(context), "saved_failure": summary,
        "saved_failure_digest": summary_digest, "replay_id": binding.replay_id,
        "required_action": "Call replay_retained_stage before final diagnosis. Cite the returned report_digest.",
        "permitted_recovery_ids": [row.recovery_id for row in recovery],
    }, sort_keys=True)}]
    deadline = time.time() + ttl_seconds
    managed = runtime == RUNTIME_API
    admission = AgentAdmission(
        authority_digest=authority["authority_digest"], authority_reference="server-admitted-task:" + task_id,
        project_id=service.config.project_id, runtime=runtime, disclosure_scope="blueprint_sanitized_operational_records",
        allowed_input_digests=tuple(dict.fromkeys([summary_digest, job_digest, digest(payload), *extra_digests])),
        allowed_tool_ids=tuple(tool.tool_id for tool in tools),
        budget_policy="project_guard_accepted_uncertainty" if managed else "strict_per_call",
        session_retention="until_deleted" if managed else "not_admitted",
        trace_retention="provider_default" if managed else "not_admitted", region="us" if managed else "default",
        inference_budget_usd=inference_budget_usd, expires_at=deadline,
        project_guard_receipt_digest=service.managed_guard_digest("blueprint_sanitized_operational_records") if managed else None,
    )
    task = AgentTask(
        task_id=task_id, run_id=run_id, capability="runtime_failure_recovery",
        source_commit=service.config.source_commit, context_revision=context_revision(context),
        instructions="You investigate a Blueprint Task Evaluation Run under a fixed server authority. "
        "Use the admitted replay tool to inspect the failed boundary. A successful replay is local diagnostic evidence. "
        "Separate observed facts from hypotheses. Cite the exact replay report_digest in evidence_references. "
        "Never claim deployment, paid execution, scientific success, resource release or customer delivery from diagnosis. "
        "If the boundary requires unavailable evidence or authority, report that explicitly. Preserve prior failures.",
        model=model or service.config.allowed_models[0], input=payload, input_digests=(summary_digest, job_digest),
        output_schema=OperationalDiagnosis.model_json_schema(), tool_ids=tuple(tool.tool_id for tool in tools),
        tool_digests={tool.tool_id: tool.tool_digest for tool in tools}, admission=admission, deadline=deadline,
        # Covers two bounded tool replies plus the final answer under the SDK's
        # conservative context-growth reservation, not only the initial prompt.
        max_model_turns=3, max_input_tokens=120_000, max_output_tokens=2_000,
        max_tool_calls=4, max_tool_output_bytes=24_000,
    )
    record = TaskRecord(schema_version="blueprint_agent_admitted_task.v1", enabled=True, autostart=autostart,
                        cleanup_when_terminal=True,
                        owner_client_ids=(owner_client_id,), task=task, context=asdict(context), stage_replays=(binding,),
                        controller_recoveries=recovery)
    if not persist:
        return record
    destination = Path(service.config.task_store_root) / (task_id + ".json")
    document = record.model_dump(mode="json")
    with service.journal.own_task(task_id):
        if destination.exists():
            raise AgentExecutionError("agent_admitted_task_already_exists")
        # Controller-owned directory; web/API agents have no file-write tool here.
        write_json(destination, document)
        destination.chmod(0o640)
        service.validate_admission(task)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--child-id", required=True)
    parser.add_argument("--owner-client-id", required=True)
    parser.add_argument("--inference-budget-usd", type=float, required=True)
    parser.add_argument("--runtime", choices=(RUNTIME_SDK, RUNTIME_API), default=RUNTIME_SDK)
    parser.add_argument("--model")
    parser.add_argument("--ttl-seconds", type=int, default=600)
    parser.add_argument("--no-autostart", action="store_true")
    parser.add_argument("--controller-recovery", type=Path, help="Optional private, preauthorized controller recovery binding.")
    args = parser.parse_args(argv)
    service = configured_service()
    record = prepare_retained_failure(service, task_id=args.task_id, run_id=args.run_id, child_id=args.child_id,
                                     owner_client_id=args.owner_client_id, inference_budget_usd=args.inference_budget_usd,
                                     runtime=args.runtime, model=args.model, ttl_seconds=args.ttl_seconds,
                                     autostart=not args.no_autostart,
                                     controller_recovery=ControllerRecoveryBinding.model_validate_json(args.controller_recovery.read_bytes())
                                     if args.controller_recovery else None)
    print(json.dumps({"task_id": record.task.task_id, "task_digest": record.task.task_digest,
                      "state": "admitted", "provider_execution_started": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
