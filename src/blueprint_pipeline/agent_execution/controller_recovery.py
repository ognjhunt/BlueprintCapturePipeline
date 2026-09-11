"""Ask the existing scene controller to resume one preauthorized intent.

The model selects a frozen recovery id after a successful saved-stage replay.
The existing controller reopens owner rights, retry/spend limits, provider-zero,
preflight and completed-prefix adoption. No model-supplied command is executed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from pydantic import BaseModel, ConfigDict, Field

from ..common import write_json
from ..decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .contracts import AgentExecutionError, AgentTool, DIGEST, IDENTIFIER, ToolReconciliation, digest
from .operations import OperationPending, ToolRefused


class ControllerRecoveryBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recovery_id: str = Field(pattern=IDENTIFIER)
    intent_id: str = Field(pattern=r"^scene-[A-Za-z0-9._-]+$")
    intent_digest: str = Field(pattern=DIGEST)
    controller_config_path: str
    controller_config_sha256: str = Field(pattern=DIGEST)
    required_replay_id: str = Field(pattern=IDENTIFIER)
    parent_request_digest: str = Field(pattern=DIGEST)
    preparation_link_path: str
    preparation_link_sha256: str = Field(pattern=DIGEST)
    allow_controller_successors: bool = False


def validate_controller_binding(binding):
    from .production import _read_private
    path = Path(binding.controller_config_path)
    if not path.is_absolute() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise AgentExecutionError("agent_recovery_controller_path_invalid")
    raw = _read_private(path)
    if "sha256:" + hashlib.sha256(raw).hexdigest() != binding.controller_config_sha256:
        raise AgentExecutionError("agent_recovery_controller_configuration_changed")
    config = json.loads(raw)
    if (config.get("schema_version") != "task_evaluation_scene_progression_config.v1"
            or config.get("config_digest") != canonical_digest(config, digest_field="config_digest")):
        raise AgentExecutionError("agent_recovery_controller_configuration_invalid")
    intent_path = Path(config["intent_root"]) / binding.intent_id / "intent.json"
    if not intent_path.is_absolute() or any(parent.is_symlink() for parent in (intent_path, *intent_path.parents)):
        raise AgentExecutionError("agent_recovery_intent_path_invalid")
    intent = json.loads(_read_private(intent_path))
    if (intent.get("intent_digest") != binding.intent_digest
            or intent.get("intent_digest") != cross_runtime_canonical_digest(intent, digest_field="intent_digest")):
        raise AgentExecutionError("agent_recovery_intent_changed")
    # This is only the entry check. The controller still runs its complete
    # current authority, failure-class, retry, spend and provider-zero checks.
    consent = intent.get("request", {}).get("consent", {})
    if consent.get("spend_authorized") is not True or consent.get("task_confirmed") is not True:
        raise AgentExecutionError("agent_recovery_owner_authority_missing")
    link_raw = _read_private(Path(binding.preparation_link_path))
    link = json.loads(link_raw)
    if ("sha256:" + hashlib.sha256(link_raw).hexdigest() != binding.preparation_link_sha256
            or link.get("link_digest") != canonical_digest(link, digest_field="link_digest")
            or link.get("intent_digest") != binding.intent_digest or link.get("intent_id") != binding.intent_id
            or link.get("request_digest") != binding.parent_request_digest):
        raise AgentExecutionError("agent_recovery_preparation_lineage_invalid")
    return config, intent


class ControllerRecoveryTools:
    def __init__(self, journal, bindings, source_commit):
        self.journal, self.source_commit = journal, source_commit
        self.bindings = {binding.recovery_id: binding for binding in bindings}
        if len(self.bindings) != len(bindings):
            raise AgentExecutionError("agent_recovery_duplicate_binding")
        self.root = journal.root / "controller-requests"

    def paths(self, operation_id):
        return self.root / "requests" / (operation_id[7:] + ".json"), self.root / "results" / (operation_id[7:] + ".json")

    def reconcile(self, args, context):
        request_path, result_path = self.paths(context.operation_id)
        if result_path.exists():
            value = json.loads(result_path.read_text())
            request = json.loads(request_path.read_text())
            if value.get("request_digest") != digest(request) or value.get("operation_id") != context.operation_id:
                raise AgentExecutionError("agent_recovery_result_binding_invalid")
            return ToolReconciliation("completed", value)
        return ToolReconciliation("pending" if request_path.exists() else "not_started")

    def invoke(self, args, context):
        binding = self.bindings[args["recovery_id"]]
        validate_controller_binding(binding)
        replay = next((operation for operation in self.journal.task_operations(context.task_id)
            if operation["request"]["tool_id"] == "replay_retained_stage"
            and operation["request"]["arguments"].get("replay_id") == binding.required_replay_id
            and (operation["outcome"] or {}).get("success") is True
            and (operation["outcome"] or {}).get("output", {}).get("status") == "completed"
            and (operation["outcome"] or {}).get("output", {}).get("report_digest") == args["replay_report_digest"]), None)
        if replay is None:
            raise ToolRefused("agent_recovery_successful_replay_required")
        value = {"schema_version": "blueprint_agent_controller_request.v1", "operation_id": context.operation_id,
            "task_id": context.task_id, "run_id": context.run_id, "source_commit": self.source_commit,
            "binding": binding.model_dump(mode="json"), "replay_report_digest": args["replay_report_digest"],
            "deadline": context.deadline}
        request_path, _ = self.paths(context.operation_id)
        if request_path.exists():
            if json.loads(request_path.read_text()) != value:
                raise AgentExecutionError("agent_recovery_request_conflict")
        else:
            write_json(request_path, value)
        result = self.reconcile(args, context)
        if result.status == "completed":
            return result.output
        raise OperationPending("agent_existing_controller_pending")

    def tools(self):
        if not self.bindings:
            return ()
        return (AgentTool("request_preauthorized_scene_progression", "1",
            "After a completed retained-stage replay, ask the existing controller to resume one frozen owner intent. "
            "It independently checks all rights, retry limits, spend, provider-zero and completed-stage adoption. Bindings: "
            + digest({key: value.model_dump(mode="json") for key, value in self.bindings.items()}),
            {"type": "object", "properties": {"recovery_id": {"type": "string", "enum": sorted(self.bindings)},
                "replay_report_digest": {"type": "string", "pattern": DIGEST}},
             "required": ["recovery_id", "replay_report_digest"], "additionalProperties": False},
            "external_side_effect", self.invoke, self.reconcile),)


def consume_controller_requests(*, controller_config_path, agent_config_path="/etc/blueprint/agent-execution.json",
                                only_intent_id=None):
    """Run inside the existing scene progression service, not the model worker."""
    from .production import ProductionAgentService, ProductionConfig, _read_private
    from ..task_evaluation_scene_progression import process_scene_intents
    from ..task_evaluation_release_identity import running_release_commit
    path = Path(agent_config_path)
    if not path.exists():
        return []
    config = ProductionConfig.model_validate_json(_read_private(path))
    service = ProductionAgentService(path, source_commit=running_release_commit())
    root = service.journal.root / "controller-requests"
    rows = []
    pending = [path for path in sorted((root / "requests").glob("*.json")) if not (root / "results" / path.name).exists()]
    for request_path in pending[:64]:
        result_path = root / "results" / request_path.name
        if result_path.exists():
            continue
        request = json.loads(_read_private(request_path))
        binding = ControllerRecoveryBinding.model_validate(request["binding"])
        if only_intent_id is not None and binding.intent_id != only_intent_id:
            continue
        if Path(binding.controller_config_path).resolve() != Path(controller_config_path).resolve():
            continue
        with service.journal.own_task("controller-request:" + request_path.stem):
            if result_path.exists():
                continue
            state = service.journal.task(request["task_id"])
            started_id = "controller_started_" + request["operation_id"][7:]
            started = service.journal.event(started_id)
            operation = service.journal.operation(request["operation_id"])
            if (operation["request"]["task_id"] != request["task_id"]
                    or operation["request"]["tool_id"] != "request_preauthorized_scene_progression"
                    or operation["request"]["arguments"] != {"recovery_id": binding.recovery_id,
                        "replay_report_digest": request["replay_report_digest"]}
                    or request["deadline"] != state["task"]["deadline"]):
                raise AgentExecutionError("agent_recovery_operation_binding_invalid")
            result = None
            try:
                if started is None and (state["cancel_requested"] or time.time() >= request["deadline"]):
                    result = {"status": "cancelled_before_controller"}
                else:
                    record = service.record(request["task_id"])
                    if started is None:
                        service.validate_admission(record.task)
                    if (binding not in record.controller_recoveries
                            or request["source_commit"] != config.source_commit or request["run_id"] != record.task.run_id):
                        raise AgentExecutionError("agent_recovery_request_scope_changed")
                    controller_config, intent = validate_controller_binding(binding)
                    from ..task_evaluation_scene_progression_state import load_progression
                    directory = Path(controller_config["intent_root"]) / binding.intent_id
                    before = load_progression(directory, intent)
                    report_path = root / "controller-reports" / request_path.name
                    if report_path.exists():
                        report = json.loads(_read_private(report_path))
                    elif started is not None:
                        # An interrupted handoff is never blindly dispatched.
                        # Read the original controller's retained progression.
                        if not before or before["progression_digest"] == started["before_progression_digest"]:
                            continue
                        report = {"status": "reconciled_from_original_controller", "results": [{
                            "intent_id": binding.intent_id, "status": before["status"], "blockers": before["blockers"]}],
                            "progression_digest": before["progression_digest"], "causal_execution_claimed": False}
                    else:
                        claimed = service.journal.claim_controller_request(record.task, request["operation_id"], {
                            "task_digest": record.task.task_digest, "request_digest": digest(request),
                            "before_progression_digest": before["progression_digest"] if before else None})
                        if not claimed:
                            continue
                        report = process_scene_intents(config_path=controller_config_path, only_intent_id=binding.intent_id)
                        write_json(report_path, report)
                    result = {"status": "controller_pass_observed", "controller_report_digest": canonical_digest(report),
                        "controller_states": [{key: row.get(key) for key in ("intent_id", "status", "blockers")}
                                              for row in report.get("results", [])]}
            except (ValueError, AgentExecutionError, OSError) as exc:
                if service.journal.event(started_id) is not None:
                    # Handoff may have happened. Preserve uncertainty until the
                    # original controller supplies a readable later revision.
                    continue
                result = {"status": "controller_refused", "error_type": type(exc).__name__}
            value = {"schema_version": "blueprint_agent_controller_result.v1", "operation_id": request["operation_id"],
                "request_digest": digest(request), "intent_id": binding.intent_id, **result,
                "paid_authority_expanded": False, "model_granted_acceptance": False, "proof_effect": "none"}
            write_json(result_path, value)
            rows.append(value)
    return rows
