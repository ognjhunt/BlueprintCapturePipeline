"""Retained-stage replay requested by a reasoning agent, executed offline.

The agent worker writes a bounded request. A separate network/device-isolated
service invokes the existing replay command against server-bound saved inputs.
Raw jobs and replay reports remain local; only a typed summary is disclosed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from pydantic import BaseModel, ConfigDict, Field

from ..common import write_json
from .contracts import AgentExecutionError, AgentTool, DIGEST, IDENTIFIER, ToolReconciliation, digest
from .journal import AgentJournal
from .operations import OperationPending, ToolRefused


class StageReplayBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    replay_id: str = Field(pattern=IDENTIFIER)
    child_id: str = Field(pattern=r"^sam31-[a-zA-Z0-9_-]{1,160}$")
    job_sha256: str = Field(pattern=DIGEST)
    queue_root: str
    parent_queue_root: str
    input_root: str
    approved_roots: tuple[str, ...]
    timeout_seconds: int = Field(default=300, ge=1, le=1800)


def _job(binding: StageReplayBinding):
    from ..task_evaluation_stage_replay import locate_child

    located = locate_child(binding.queue_root, binding.child_id)
    path = located.job_path
    if path.is_symlink() or not path.is_file():
        raise AgentExecutionError("agent_replay_job_path_invalid")
    raw = path.read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != binding.job_sha256:
        raise AgentExecutionError("agent_replay_saved_job_changed")
    return located, json.loads(raw)


def _code(value):
    # Errors can contain paths, URIs, provider bodies and source names. Only
    # an opaque predicate identifier is eligible for the operational prompt.
    return value if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,159}", value) else None


def _predicate(value):
    if not isinstance(value, str) or len(value) > 500:
        return None
    try:
        parsed = ast.parse(value, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(parsed):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and not re.fullmatch(
            r"[A-Za-z0-9_]{0,80}", node.value
        ):
            return None
    return value


def summarize_report(report: dict, *, replay_id: str, job_sha256: str, source_commit: str) -> dict:
    predicates = report.get("fired_predicates", [])
    if not isinstance(predicates, list):
        predicates = []
    return {
        "schema_version": "blueprint_agent_stage_replay_summary.v1", "replay_id": replay_id,
        "source_commit": source_commit, "job_sha256": job_sha256,
        "report_digest": digest(report), "status": _code(report.get("status")) or "unclassified",
        "phase": _code(report.get("phase")), "blocker_code": _code(report.get("blocker")),
        "fired_predicates": [value for value in predicates if _predicate(value)],
        "redacted_predicate_count": sum(_predicate(value) is None for value in predicates),
        "external_boundary_reached": report.get("boundary_hint") == "external_boundary_unreachable_by_design",
        "raw_detail_retained_locally": True, "provider_mutation_performed": False,
        "scientific_acceptance_granted": False, "proof_effect": "none",
    }


class StageReplayTools:
    def __init__(self, *, journal: AgentJournal, bindings: tuple[StageReplayBinding, ...], source_commit: str):
        self.journal, self.source_commit = journal, source_commit
        self.bindings = {binding.replay_id: binding for binding in bindings}
        if len(self.bindings) != len(bindings):
            raise ValueError("agent_duplicate_replay_binding")
        self.root = journal.root / "stage-replays"

    def _paths(self, context):
        stem = digest({"task_id": context.task_id, "operation_id": context.operation_id})[7:]
        return self.root / "requests" / (stem + ".json"), self.root / "results" / (stem + ".json")

    def _request(self, arguments, context):
        binding = self.bindings.get(arguments["replay_id"])
        if binding is None:
            raise ToolRefused("agent_replay_target_not_admitted")
        return {"schema_version": "blueprint_agent_stage_replay_request.v1",
                "task_id": context.task_id, "run_id": context.run_id,
                "operation_id": context.operation_id, "context_revision": context.context_revision,
                "source_commit": self.source_commit, "binding": binding.model_dump(mode="json"),
                "deadline": context.deadline}

    def _reconcile(self, arguments, context):
        request_path, result_path = self._paths(context)
        if result_path.exists():
            value = json.loads(result_path.read_text())
            saved = json.loads(request_path.read_text())
            expected = self._request(arguments, context)
            # Cancellation uses a short observation deadline, not a new grant.
            expected["deadline"] = saved["deadline"]
            if (saved != expected or value.get("request_digest") != digest(saved)
                    or value.get("source_commit") != self.source_commit):
                raise AgentExecutionError("agent_replay_result_binding_invalid")
            return ToolReconciliation("completed", value["summary"])
        return ToolReconciliation("pending" if request_path.exists() else "not_started")

    def _invoke(self, arguments, context):
        request = self._request(arguments, context)
        binding = self.bindings[arguments["replay_id"]]
        try:
            _job(binding)
        except (AgentExecutionError, OSError, ValueError):
            raise ToolRefused("agent_replay_saved_input_unavailable") from None
        request_path, _ = self._paths(context)
        if request_path.exists():
            if json.loads(request_path.read_text()) != request:
                raise AgentExecutionError("agent_replay_request_identity_conflict")
        else:
            write_json(request_path, request)
        reconciled = self._reconcile(arguments, context)
        if reconciled.status == "completed":
            return reconciled.output
        raise OperationPending("agent_isolated_replay_pending")

    def tools(self):
        if not self.bindings:
            return ()
        return (AgentTool(
            "replay_retained_stage", "1", "Replay one admitted failed child offline against this release. "
            "Wait for a typed refusing-predicate summary; completed paid stages are never dispatched. "
            "Bound inputs: " + digest({key: item.model_dump(mode="json") for key, item in self.bindings.items()}),
            {"type": "object", "properties": {"replay_id": {"type": "string", "enum": sorted(self.bindings)}},
             "required": ["replay_id"], "additionalProperties": False},
            "idempotent_write", self._invoke, self._reconcile,
        ),)


def require_secret_isolation(config):
    for value in (config.credential_file, config.webhook_secret_file):
        if value and os.access(value, os.R_OK):
            raise AgentExecutionError("agent_replay_configured_secret_accessible")


def require_offline_isolation(config):
    """Read kernel namespace identity instead of trusting an environment flag."""
    if sys.platform != "linux" or os.geteuid() == 0:
        raise AgentExecutionError("agent_replay_requires_unprivileged_linux_service")
    unit = "blueprint-agent-stage-replay.service"
    observed = subprocess.run(
        ["systemctl", "show", unit, "-p", "MainPID", "-p", "PrivateNetwork", "-p", "PrivateDevices",
         "-p", "ProtectSystem", "-p", "FragmentPath"], check=True, capture_output=True, text=True, timeout=10,
    )
    properties = dict(line.split("=", 1) for line in observed.stdout.splitlines() if "=" in line)
    expected_unit = Path(__file__).resolve().parents[3] / "deploy" / "systemd" / unit
    fragment = Path(properties.get("FragmentPath", ""))
    if (properties.get("MainPID") != str(os.getpid()) or properties.get("PrivateNetwork") != "yes"
            or properties.get("PrivateDevices") != "yes" or properties.get("ProtectSystem") != "strict"
            or not fragment.is_file() or fragment.read_bytes() != expected_unit.read_bytes()):
        raise AgentExecutionError("agent_replay_service_isolation_unproven")
    if list(Path("/dev").glob("nvidia*")) or Path("/dev/dri").exists():
        raise AgentExecutionError("agent_replay_network_or_device_isolation_missing")
    if Path("/etc/blueprint/provider-secrets").exists() and os.access("/etc/blueprint/provider-secrets", os.R_OK):
        raise AgentExecutionError("agent_replay_provider_secrets_accessible")
    require_secret_isolation(config)


def replay_command(request: dict, *, output_root: Path) -> list[str]:
    binding = StageReplayBinding.model_validate(request["binding"])
    return [sys.executable, "-m", "blueprint_pipeline.task_evaluation_stage_replay",
            "--child", binding.child_id, "--queue-root", binding.queue_root,
            "--parent-queue-root", binding.parent_queue_root, "--input-root", binding.input_root,
            "--replay-root", str(output_root / "scratch"), "--json-out", str(output_root / "report.json"),
            *[part for root in binding.approved_roots for part in ("--approved-root", root)]]


def process_one(service) -> dict | None:
    require_offline_isolation(service.config)
    root = service.journal.root / "stage-replays"
    for path in sorted((root / "requests").glob("*.json")):
        result_path = root / "results" / path.name
        if result_path.exists():
            continue
        with service.journal.own_task("offline-replay-" + path.stem):
            if result_path.exists():
                continue
            request = json.loads(path.read_text())
            task = _task_from_service(service, request)
            binding = StageReplayBinding.model_validate(request["binding"])
            output_root = root / "runs" / path.stem
            intent_path = output_root / "intent.json"
            output_root.mkdir(parents=True, exist_ok=True)
            report_path = output_root / "report.json"
            state = service.journal.task(task.task_id)
            if report_path.exists() and intent_path.exists():
                if json.loads(intent_path.read_text()) != {"request_digest": digest(request)}:
                    raise AgentExecutionError("agent_replay_intent_changed")
                report = json.loads(report_path.read_text())
            elif intent_path.exists():
                # The previous service stopped before a report. Preserve that
                # failure and scratch; never rerun an uncertain operation.
                report = {"status": "interrupted", "blocker": "prior_replay_outcome_unresolved"}
            elif state["cancel_requested"] or time.time() >= request["deadline"]:
                report = {"status": "cancelled", "blocker": "agent_replay_cancelled_before_start"}
            else:
                report = _run_replay(service, task, binding, request, output_root)
            summary = summarize_report(report, replay_id=binding.replay_id,
                                       job_sha256=binding.job_sha256, source_commit=request["source_commit"])
            receipt = {"schema_version": "blueprint_agent_stage_replay_result.v1",
                       "request_digest": digest(request), "source_commit": request["source_commit"],
                       "summary": summary, "completed_at": time.time()}
            write_json(result_path, receipt)
            service.journal.wake(task.task_id, due_at=time.time())
            return receipt
    return None


def _run_replay(service, task, binding, request, output_root):
    try:
        service.validate_admission(task)
        _job(binding)
        # The exclusive replay lock and absent intent prove this child process
        # has not started. Linearize its start against cancellation/deadline.
        service.journal.mark_executing(request["operation_id"], task=task, reconciled_not_started=True)
    except (AgentExecutionError, OSError, ValueError) as exc:
        return {"status": "refused", "blocker": _code(str(exc)) or "replay_authority_or_input_refused"}
    write_json(output_root / "intent.json", {"request_digest": digest(request)})
    code_root = Path(__file__).resolve().parents[3]
    argv = replay_command(request, output_root=output_root)
    # No inherited credentials, proxy, launch overrides or provider environment.
    environment = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(code_root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        with (output_root / "process.log").open("wb") as log:
            completed = subprocess.run(argv, cwd=code_root, env=environment, stdout=log, stderr=log,
                                       timeout=min(binding.timeout_seconds, max(0.1, request["deadline"] - time.time())),
                                       check=False)
        report_path = output_root / "report.json"
        return (json.loads(report_path.read_text()) if report_path.exists() else
                {"status": "failed", "blocker": "replay_process_did_not_write_report", "exit_code": completed.returncode})
    except subprocess.TimeoutExpired:
        return {"status": "failed", "blocker": "replay_wall_time_exhausted"}


def _task_from_service(service, request):
    # This separate offline worker never trusts queue-selected host paths.
    record = service.record(request["task_id"])
    task = record.task
    matches = [item.model_dump(mode="json") for item in record.stage_replays
               if item.replay_id == request["binding"].get("replay_id")]
    if (matches != [request["binding"]] or task.source_commit != request["source_commit"]
            or task.context_revision != request["context_revision"] or task.run_id != request["run_id"]
            or task.deadline != request["deadline"]):
        raise AgentExecutionError("agent_replay_server_binding_invalid")
    operation = service.journal.operation(request["operation_id"])
    call = operation["request"]
    if (operation["state"] != "executing" or call["task_id"] != task.task_id
            or call["context_revision"] != task.context_revision or call["tool_id"] != "replay_retained_stage"
            or call["arguments"] != {"replay_id": request["binding"]["replay_id"]}):
        raise AgentExecutionError("agent_replay_operation_not_owned")
    return task


def main(argv=None):
    from .production import configured_service

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    service = configured_service()
    while True:
        receipt = process_one(service)
        if receipt is None:
            return 0
        print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
