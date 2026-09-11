"""Production composition: trusted task records, scoped runtimes and worker.

HTTP callers select a server-admitted task by id. They cannot supply a prompt,
credential, filesystem path, executable, disclosure grant or spend authority.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import stat
import threading
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..common import write_json
from ..task_evaluation_supervisor.capabilities import SupervisorContext
from ..task_evaluation_supervisor.contracts import AuthorityEnvelope
from ..task_evaluation_supervisor.tools import ToolRegistry
from .contracts import AgentExecutionError, AgentTask, DIGEST, IDENTIFIER, RUNTIME_API, digest
from .journal import AgentJournal
from .openai_agents_api import OpenAIAgentsRuntime
from .openai_transport import OpenAIAgentsHTTP
from .operations import AgentOperations
from .sdk_runtime import OpenAIAgentsSDKRuntime, SDKCredential
from .service import AgentTaskService
from .supervisor_bridge import SupervisorCapabilityBridge, context_revision
from .stage_recovery import StageReplayBinding, StageReplayTools


CONFIG_ENV = "BLUEPRINT_AGENT_EXECUTION_CONFIG"


class OperationalDiagnosis(BaseModel):
    """Advice references observations; controllers retain execution acceptance."""
    model_config = ConfigDict(extra="forbid")
    disposition: Literal["no_action", "investigate", "recover", "awaiting_input", "abstain"]
    summary: str
    evidence_references: list[str]
    next_actions: list[str]
    uncertainty: list[str]


class ProductionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["blueprint_agent_production_config.v1"]
    state_root: str
    task_store_root: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    project_id: str = Field(pattern=IDENTIFIER)
    credential_id: str = Field(pattern=IDENTIFIER)
    credential_file: str
    allowed_models: tuple[str, ...] = Field(min_length=1)
    max_task_budget_usd: float = Field(gt=0, le=100, allow_inf_nan=False)
    managed_api_enabled: bool = False
    project_guard_receipt_digest: str | None = Field(default=None, pattern=DIGEST)
    webhook_secret_file: str | None = None
    poll_seconds: float = Field(default=5, ge=0.1, le=60)


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["blueprint_agent_admitted_task.v1"]
    enabled: bool
    autostart: bool = False
    owner_client_ids: tuple[str, ...] = Field(min_length=1)
    task: AgentTask
    context: dict
    stage_replays: tuple[StageReplayBinding, ...] = ()


def _read_private(path: Path, *, limit: int = 4_000_000, secret: bool = False) -> bytes:
    """Read a local configuration file once, refusing links and public writes."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o022
                    or metadata.st_uid not in {0, os.getuid()} or metadata.st_size > limit):
                raise AgentExecutionError("agent_configuration_file_unsafe")
            mode = stat.S_IMODE(metadata.st_mode)
            if secret and not (mode in {0o400, 0o600} or (
                mode in {0o440, 0o640} and metadata.st_gid in {os.getgid(), *os.getgroups()}
            )):
                raise AgentExecutionError("agent_secret_file_permissions_unsafe")
            value = handle.read(limit + 1)
        if len(value) > limit:
            raise AgentExecutionError("agent_configuration_file_too_large")
        return value
    except OSError:
        raise AgentExecutionError("agent_configuration_file_unavailable") from None


class ProductionAgentService:
    def __init__(self, config_path: str | Path, *, source_commit: str):
        self.config_path = Path(config_path)
        self.config = ProductionConfig.model_validate_json(_read_private(self.config_path))
        if self.config.source_commit != source_commit:
            raise AgentExecutionError("agent_production_release_mismatch")
        for name in ("state_root", "task_store_root", "credential_file"):
            if not Path(getattr(self.config, name)).is_absolute():
                raise AgentExecutionError("agent_configuration_requires_absolute_path")
        self.config_digest = digest(self.config.model_dump(mode="json"))
        self.journal = AgentJournal(self.config.state_root)
        self.registry = ToolRegistry.default()
        self.service = AgentTaskService(
            journal=self.journal, runtime_for_task=self.runtime_for_task,
            validate_admission=self.validate_admission, poll_seconds=self.config.poll_seconds,
        )

    def record(self, task_id: str) -> TaskRecord:
        if re.fullmatch(IDENTIFIER, task_id) is None:
            raise AgentExecutionError("agent_task_identifier_invalid")
        path = Path(self.config.task_store_root) / (task_id + ".json")
        record = TaskRecord.model_validate_json(_read_private(path))
        if record.task.task_id != task_id:
            raise AgentExecutionError("agent_task_record_identity_mismatch")
        return record

    def _context(self, run_id: str, task_id: str) -> SupervisorContext:
        record = self.record(task_id)
        current = SupervisorContext(**record.context)
        if current.run_id != run_id:
            raise AgentExecutionError("agent_task_context_run_mismatch")
        return current

    def validate_admission(self, task: AgentTask) -> None:
        current_config = ProductionConfig.model_validate_json(_read_private(self.config_path))
        if digest(current_config.model_dump(mode="json")) != self.config_digest:
            raise AgentExecutionError("agent_production_configuration_changed")
        record = self.record(task.task_id)
        if not record.enabled or record.task.task_digest != task.task_digest:
            raise AgentExecutionError("agent_server_admission_revoked")
        if (task.source_commit != self.config.source_commit
                or task.admission.project_id != self.config.project_id
                or task.model not in self.config.allowed_models
                or task.admission.inference_budget_usd > self.config.max_task_budget_usd
                or time.time() >= min(task.deadline, task.admission.expires_at)):
            raise AgentExecutionError("agent_server_admission_scope_invalid")
        if task.admission.runtime == RUNTIME_API and (
            not self.config.managed_api_enabled
            or not self.config.project_guard_receipt_digest
            or task.admission.project_guard_receipt_digest != self.config.project_guard_receipt_digest
        ):
            raise AgentExecutionError("agent_managed_project_policy_not_admitted")
        current = self._context(task.run_id, task.task_id)
        authority = AuthorityEnvelope.from_mapping(current.authority_envelope).to_mapping()
        if (context_revision(current) != task.context_revision
                or authority["authority_digest"] != task.admission.authority_digest
                or not set(authority["immutable_input_digests"]) <= set(task.admission.allowed_input_digests)
                or not set(task.tool_ids) <= set(authority["allowed_tool_ids"])
                or not authority.get("agent_inference_allowed")
                or task.admission.inference_budget_usd > authority["agent_inference_budget_usd"]):
            raise AgentExecutionError("agent_server_context_or_authority_invalid")

    def _credential(self, task: AgentTask) -> SDKCredential:
        if task.admission.project_id != self.config.project_id:
            raise AgentExecutionError("agent_credential_project_mismatch")
        key = _read_private(Path(self.config.credential_file), limit=16_000, secret=True).decode().strip()
        return SDKCredential(self.config.project_id, self.config.credential_id, key)

    def runtime_for_task(self, task: AgentTask):
        bridge = SupervisorCapabilityBridge(
            capability=task.capability, registry=self.registry, journal=self.journal,
            load_context=lambda run_id: self._context(run_id, task.task_id),
        )
        record = self.record(task.task_id)
        tools = (*bridge.tools(), *StageReplayTools(
            journal=self.journal, bindings=record.stage_replays, source_commit=task.source_commit,
        ).tools())
        operations = AgentOperations(
            self.journal, tools,
            authorize=lambda current, _tool, _args: self.validate_admission(current),
        )
        if digest(task.output_schema) != digest(OperationalDiagnosis.model_json_schema()):
            raise AgentExecutionError("agent_production_output_contract_invalid")
        if task.admission.runtime == RUNTIME_API:
            credential = self._credential(task)
            return OpenAIAgentsRuntime(
                transport=OpenAIAgentsHTTP(api_key=credential.api_key, project_id=credential.project_id),
                project_id=credential.project_id, journal=self.journal, operations=operations,
                validate_admission=self.validate_admission,
            )
        return OpenAIAgentsSDKRuntime(
            journal=self.journal, operations=operations,
            output_models={task.capability: OperationalDiagnosis},
            validate_admission=self.validate_admission, resolve_credential=self._credential,
        )

    def authorize_client(self, task_id: str, client_id: str) -> TaskRecord:
        record = self.record(task_id)
        if client_id not in record.owner_client_ids:
            raise AgentExecutionError("agent_task_client_not_authorized")
        saved = self.journal.event("server_admission_" + digest(task_id)[7:])
        if saved is not None and saved != self._ownership(record):
            raise AgentExecutionError("agent_server_ownership_record_changed")
        return record

    @staticmethod
    def _ownership(record):
        return {"task_digest": record.task.task_digest, "owner_client_ids": list(record.owner_client_ids)}

    def _enqueue_record(self, record):
        self.validate_admission(record.task)
        self.journal.record_event("server_admission_" + digest(record.task.task_id)[7:], self._ownership(record))
        return self.service.enqueue(record.task)

    def enqueue(self, task_id: str, client_id: str) -> dict:
        record = self.authorize_client(task_id, client_id)
        self._enqueue_record(record)
        return self.status(task_id, client_id)

    def status(self, task_id: str, client_id: str) -> dict:
        record = self.authorize_client(task_id, client_id)
        state = self.journal.task(task_id)
        if record.task.task_digest != state["task_digest"]:
            raise AgentExecutionError("agent_server_and_journal_task_mismatch")
        # No prompts, source evidence, host paths or authority objects leave this
        # status surface. Result text is returned only to this task's owner.
        return {
            "schema_version": "blueprint_agent_task_status.v1", "task_id": task_id,
            "run_id": record.task.run_id, "source_commit": record.task.source_commit,
            "runtime": record.task.admission.runtime, "task_digest": state["task_digest"],
            **{key: state[key] for key in ("state", "error_code", "cancel_requested", "cleanup_state", "updated_at")},
            "result": state["result"], "usage": state["usage"],
            "resource_closeout": "not_established_by_agent_completion", "proof_effect": "none",
        }

    def act(self, task_id: str, client_id: str, action: str) -> dict:
        self.status(task_id, client_id)
        if action == "cancel":
            self.service.cancel(task_id)
        elif action == "cleanup":
            self.service.request_cleanup(task_id)
        else:
            raise AgentExecutionError("agent_task_action_invalid")
        return self.status(task_id, client_id)

    def autostart(self) -> int:
        count = 0
        for path in sorted(Path(self.config.task_store_root).glob("*.json")):
            try:
                record = self.record(path.stem)
                if record.enabled and record.autostart:
                    try:
                        self.journal.task(record.task.task_id)
                    except AgentExecutionError as exc:
                        if str(exc) != "agent_task_missing":
                            raise
                        self._enqueue_record(record)
                        count += 1
            except (AgentExecutionError, ValueError, TypeError):
                # A malformed/expired task cannot prevent reconciliation of
                # unrelated existing sessions. Never log the private record.
                continue
        return count

    def health(self) -> dict:
        return {"schema_version": "blueprint_agent_worker_health.v1",
                "source_commit": self.config.source_commit, "config_digest": self.config_digest,
                "managed_api_enabled": self.config.managed_api_enabled,
                "active_tasks_up_to_1000": len(self.journal.tasks(limit=1000)), "proof_effect": "none"}


def configured_service() -> ProductionAgentService:
    from ..live_pipeline_intake_service import running_source_commit

    path = os.environ.get(CONFIG_ENV, "")
    if not path or not Path(path).exists():
        raise AgentExecutionError("agent_production_not_configured")
    return ProductionAgentService(path, source_commit=running_source_commit(__file__))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    service = configured_service()
    service.service.recover()
    while not stopped.is_set():
        service.autostart()
        receipt = service.service.tick()
        heartbeat = {**service.health(), "observed_at": time.time(), "last_step": receipt}
        write_json(service.journal.root / "worker_health.json", heartbeat)
        if receipt:
            print(json.dumps(receipt, sort_keys=True), flush=True)
        if args.once:
            return 0
        stopped.wait(0.1 if receipt else service.config.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
