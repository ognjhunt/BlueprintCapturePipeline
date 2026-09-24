"""Bounded CPU authoring on an explicitly selected managed Agents API session."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping

from .agent_execution.contracts import AgentAdmission, AgentExecutionError, AgentTask, digest
from .agent_execution.journal import AgentJournal
from .agent_execution.openai_agents_api import OpenAIAgentsRuntime
from .agent_execution.openai_transport import OpenAIAgentsHTTP
from .agent_execution.operations import AgentOperations
from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_object_agents_api import (
    AgentsAPIAssetTools, _write_once, asset_input, prepare_asset_task, prepare_repair_task,
)


DISCLOSURE_SCOPE = "task_asset_source_frames_and_metric_envelope"
GUARD_SCHEMA = "blueprint_agent_project_admission_observation.v1"


def _read_private(path: Path) -> dict:
    if not path.is_absolute() or path.is_symlink():
        raise AgentExecutionError("asset_api_private_file_invalid")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise AgentExecutionError("asset_api_private_file_invalid") from exc
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077
                or not 0 < metadata.st_size <= 65_536):
            raise AgentExecutionError("asset_api_private_file_invalid")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            value = json.loads(stream.read(65_537))
    except (OSError, ValueError) as exc:
        raise AgentExecutionError("asset_api_private_file_invalid") from exc
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise AgentExecutionError("asset_api_private_file_invalid")
    return value


def validate_managed_asset_guard(*, policy: Mapping[str, Any], guard_file: Path,
                                 project_id: str, credential_id: str,
                                 maximum_cost_usd: float, deadline: float,
                                 now: float) -> str:
    """Require a current observed project hard limit no larger than the stage cap."""
    guard = _read_private(Path(guard_file))
    cost = guard.get("spend_limit") or {}
    expected_digest = policy.get("project_guard_receipt_digest")
    if (not isinstance(cost, dict) or guard.get("schema_version") != GUARD_SCHEMA
            or digest(guard) != expected_digest
            or guard.get("project_id") != project_id
            or guard.get("credential_id") != credential_id
            or guard.get("dashboard_hard_limit_enabled") is not True
            or guard.get("disclosure_scope") != DISCLOSURE_SCOPE
            or guard.get("budget_policy") != "project_guard_accepted_uncertainty"
            or guard.get("session_retention") != "until_deleted"
            or guard.get("trace_retention") != "provider_default"
            or guard.get("provider_api_region") != "us"
            or type(guard.get("observed_at")) not in (int, float)
            or type(guard.get("expires_at")) not in (int, float)
            or not 0 < guard["observed_at"] <= now < deadline <= guard["expires_at"]
            or guard["expires_at"] - guard["observed_at"] > 86_401
            or cost.get("object") != "project.spend_limit"
            or cost.get("currency") != "USD" or cost.get("interval") != "month"
            or type(cost.get("threshold_amount")) is not int
            or not 0 < cost["threshold_amount"] <= round(maximum_cost_usd * 100)):
        raise AgentExecutionError("asset_api_project_guard_not_admitted")
    return expected_digest


def run_managed_asset_authoring(*, request_value: dict, output_root: Path, budget_root: Path,
                                cad_executor, blender_runner, blender_executable: str,
                                review_invoker, policy: Mapping[str, Any],
                                authority_digest: str, source_commit: str,
                                project_id: str, credential_id: str, guard_file: Path,
                                maximum_cost_usd: float, key_file: Path | None = None,
                                authoring_instructions: str = "", transport=None,
                                clock=time.time, sleep=time.sleep) -> dict:
    """Run at most three reviewed turns, then delete the managed session.

    A saved task, provider session or tool operation is reused on restart. An
    uncertain request or local side effect blocks; it cannot launch another
    managed session for the same object merely because an HTTP response was lost.
    """
    if (policy.get("schema_version") != "scene_configuration_agents_api_policy.v1"
            or policy.get("disclosure_scope") != DISCLOSURE_SCOPE
            or policy.get("session_retention") != "until_deleted"
            or policy.get("trace_retention") != "provider_default"
            or policy.get("region") != "us"
            or policy.get("budget_policy") != "project_guard_accepted_uncertainty"
            or type(policy.get("ttl_seconds")) is not int
            or not 60 <= policy["ttl_seconds"] <= 1800
            or type(policy.get("maximum_review_cycles")) is not int
            or not 1 <= policy["maximum_review_cycles"] <= 3
            or not isinstance(maximum_cost_usd, (float, int))
            or isinstance(maximum_cost_usd, bool)
            or not math.isfinite(maximum_cost_usd) or maximum_cost_usd <= 0):
        raise AgentExecutionError("asset_api_policy_invalid")
    request = request_value
    if not isinstance(request, dict):
        raise AgentExecutionError("asset_api_request_invalid")
    output_root, budget_root = Path(output_root), Path(budget_root)
    budget_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    budget_root.chmod(0o700)
    binding_path = budget_root / "stage_binding.json"
    binding = {"request_digest": request.get("request_digest"),
        "authority_digest": authority_digest, "source_commit": source_commit,
        "policy_digest": digest(dict(policy)), "project_id": project_id,
        "credential_id": credential_id, "maximum_cost_usd": maximum_cost_usd}
    if binding_path.exists() or binding_path.is_symlink():
        retained = _read_private(binding_path)
        if {key: retained.get(key) for key in binding} != binding:
            raise AgentExecutionError("asset_api_stage_binding_changed")
        deadline = retained.get("deadline")
        if type(deadline) not in (float, int):
            raise AgentExecutionError("asset_api_stage_binding_invalid")
    else:
        deadline = clock() + policy["ttl_seconds"]
        _write_once(binding_path, {**binding, "deadline": deadline})
    guard_digest = validate_managed_asset_guard(policy=policy, guard_file=guard_file,
        project_id=project_id, credential_id=credential_id,
        maximum_cost_usd=maximum_cost_usd, deadline=deadline, now=clock())
    if transport is None:
        if key_file is None:
            raise AgentExecutionError("asset_api_key_missing")
        secret = Path(key_file)
        if secret.is_symlink() or not secret.is_absolute() or secret.stat().st_mode & 0o077:
            raise AgentExecutionError("asset_api_key_invalid")
        key = secret.read_text().strip()
        if not key:
            raise AgentExecutionError("asset_api_key_invalid")
        transport = OpenAIAgentsHTTP(api_key=key, project_id=project_id)
    if transport.project_id != project_id:
        raise AgentExecutionError("asset_api_transport_project_changed")
    tools = AgentsAPIAssetTools(request_value=request, output_root=output_root,
        journal_root=budget_root / "tools", cad_executor=cad_executor,
        blender_runner=blender_runner, blender_executable=blender_executable)
    journal = AgentJournal(budget_root / "journal")
    selected_tools = tools.tools()

    def validate(task: AgentTask) -> None:
        if (task.run_id != tools.request.run_id or task.model != "gpt-6-sol"
                or task.source_commit != source_commit
                or task.admission.authority_digest != authority_digest
                or task.admission.inference_budget_usd > maximum_cost_usd
                or task.admission.project_id != project_id):
            raise AgentExecutionError("asset_api_stage_authority_changed")
        validate_managed_asset_guard(policy=policy, guard_file=guard_file,
            project_id=project_id, credential_id=credential_id,
            maximum_cost_usd=maximum_cost_usd, deadline=task.deadline, now=clock())

    operations = AgentOperations(journal, selected_tools,
        authorize=lambda task, _tool, _args: validate(task), clock=clock)
    runtime = OpenAIAgentsRuntime(transport=transport, project_id=project_id,
        journal=journal, operations=operations, validate_admission=validate, clock=clock)
    initial = asset_input(request)
    admitted_digests = [digest(initial)]
    def admission() -> AgentAdmission:
        return AgentAdmission(authority_digest=authority_digest,
            authority_reference="signed-scene-configuration:" + tools.request.run_id,
            project_id=project_id, runtime="openai_agents_api",
            disclosure_scope=DISCLOSURE_SCOPE, allowed_input_digests=tuple(admitted_digests),
            allowed_tool_ids=tuple(tool.tool_id for tool in selected_tools),
            session_retention="until_deleted", trace_retention="provider_default", region="us",
            budget_policy="project_guard_accepted_uncertainty",
            inference_budget_usd=maximum_cost_usd,
            project_guard_receipt_digest=guard_digest, expires_at=deadline)
    prefix = "asset_" + digest({"run": tools.request.run_id, "object": tools.request.object_id})[7:23]
    task = prepare_asset_task(request_value=request, tools=tools, admission=admission(),
        task_id=prefix + "_0", source_commit=source_commit, deadline=deadline,
        instructions=authoring_instructions)
    task_ids = []
    for review_index in range(policy["maximum_review_cycles"]):
        task_ids.append(task.task_id)
        if review_index:
            runtime.continue_task(task)
        else:
            runtime.start(task)
        while True:
            state = runtime.step(task.task_id)
            if state["state"] in {"completed", "failed", "cancelled"}:
                break
            if state["state"] in {"creation_unresolved", "reconciling"}:
                raise AgentExecutionError("asset_api_session_or_tool_outcome_unresolved")
            if clock() >= deadline:
                runtime.cancel(task.task_id)
                raise AgentExecutionError("asset_api_authoring_deadline")
            sleep(2)
        if state["state"] != "completed":
            raise AgentExecutionError("asset_api_author_turn_failed")
        reviewed = tools.review(task_state=state, invoker=review_invoker)
        if reviewed["accepted"]:
            for _ in range(10):
                cleaned = runtime.cleanup(task.task_id)
                if cleaned["cleanup_state"] == "deleted":
                    break
                sleep(2)
            else:
                raise AgentExecutionError("asset_api_session_cleanup_unverified")
            result = reviewed["result"]
            receipt = {"schema_version": "task_asset_agents_api_stage_receipt.v1",
                "run_id": tools.request.run_id, "object_id": tools.request.object_id,
                "provider": "openai", "model": "gpt-6-sol", "runtime": "openai_agents_api",
                "authority_digest": authority_digest, "project_guard_digest": guard_digest,
                "session_id": state["session_id"], "task_ids": task_ids,
                "review_cycles": review_index + 1, "review_digest": digest(reviewed),
                "result_digest": result["result_digest"], "session_cleanup": "deleted",
                "claim_ceiling": "development_only"}
            receipt["receipt_digest"] = canonical_digest(receipt)
            path = budget_root / "agents_api_stage_receipt.json"
            if path.exists() or path.is_symlink():
                if _read_private(path) != receipt:
                    raise AgentExecutionError("asset_api_stage_receipt_changed")
            else:
                _write_once(path, receipt)
            return result
        if review_index + 1 == policy["maximum_review_cycles"]:
            runtime.cleanup(task.task_id)
            raise AgentExecutionError("asset_api_independent_review_limit_reached")
        feedback = [{"role": "user", "content": [{"type": "input_text", "text":
            "Independent review rejected the candidate. Repair within the existing tool limits: "
            + canonical_json(reviewed)}]}]
        admitted_digests.append(digest(feedback))
        task = prepare_repair_task(previous=task, reviewed=reviewed, tools=tools,
            admission=admission(), task_id=prefix + f"_{review_index + 1}", deadline=deadline)
    raise AgentExecutionError("asset_api_review_loop_invalid")
