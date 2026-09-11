"""Persistent, revision-driven supervision of existing Blueprint job receipts.

Only server-admitted templates and exact local receipt bindings may be watched.
New semantic evidence can revisit a capability. Heartbeat-only changes cannot
consume another inference reservation; active owners settle before successors.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..common import write_json
from ..task_evaluation_supervisor.capabilities import SupervisorContext
from .contracts import AgentExecutionError, AgentTask, DIGEST, IDENTIFIER, RUNTIME_API, digest
from .journal import TERMINAL_STATES
from .supervisor_bridge import context_revision

CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
FIELDS = ("status", "state", "phase", "error_code", "blocker", "source_commit", "expected_source_commit",
          "input_digest", "result_digest", "receipt_digest", "job_digest", "request_digest", "blocker_count",
          "provider_zero_proven", "provider_mutation_performed", "completed_prefix_adopted", "delivery_verified")


class ObservationSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(pattern=IDENTIFIER)
    path: str
    schema_version: str
    identity_field: Literal["run_id", "child_id", "parent_preparation_id", "launch_id", "preparation_id", "intent_id"]
    identity_value: str = Field(pattern=IDENTIFIER)
    eligible_statuses: tuple[str, ...] = ()


class SupervisionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    watch_id: str = Field(pattern=IDENTIFIER)
    watch_digest: str = Field(pattern=DIGEST)
    observation_digest: str = Field(pattern=DIGEST)
    sources: tuple[ObservationSource, ...] = Field(min_length=1, max_length=32)


class SupervisionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["blueprint_agent_supervision_plan.v1"]
    watch_id: str = Field(pattern=IDENTIFIER)
    enabled: bool
    run_id: str = Field(pattern=IDENTIFIER)
    template_task_id: str = Field(pattern=IDENTIFIER)
    source_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    sources: tuple[ObservationSource, ...] = Field(min_length=1, max_length=32)
    maximum_revisions: int = Field(ge=1, le=100)
    maximum_reserved_inference_usd: float = Field(gt=0, le=100, allow_inf_nan=False)
    expires_at: float = Field(gt=0, allow_inf_nan=False)
    revision_ttl_seconds: int = Field(default=900, ge=1, le=1800)
    automatic_intent_digest: str | None = Field(default=None, pattern=DIGEST)
    automatic_allowance_digest: str | None = Field(default=None, pattern=DIGEST)

    @property
    def plan_digest(self):
        return digest({key: value for key, value in self.model_dump(mode="json").items()
            if key != "enabled" and not (key in {"automatic_intent_digest", "automatic_allowance_digest"} and value is None)})


def observe(sources: tuple[ObservationSource, ...]):
    from .production import _read_private

    facts, records = [], []
    if len({row.source_id for row in sources}) != len(sources):
        raise AgentExecutionError("agent_supervision_duplicate_source")
    for source in sources:
        path = Path(source.path)
        if not path.is_absolute() or any(parent.is_symlink() for parent in (path, *path.parents)):
            raise AgentExecutionError("agent_supervision_source_path_unsafe")
        if not path.exists():
            facts.append({"source_id": source.source_id, "status": "not_yet_recorded"})
            continue
        raw = _read_private(path)
        value = json.loads(raw)
        if (not isinstance(value, dict) or value.get("schema_version") != source.schema_version
                or value.get(source.identity_field) != source.identity_value):
            raise AgentExecutionError("agent_supervision_source_identity_mismatch")
        projection = {"source_id": source.source_id, "schema_version": source.schema_version}
        for key in FIELDS:
            field = value.get(key)
            if type(field) in {bool, int} or isinstance(field, str) and CODE.fullmatch(field):
                projection[key] = field
        projection["blockers"] = sorted({code for code in value.get("blockers", [])
            if isinstance(code, str) and CODE.fullmatch(code)}) if isinstance(value.get("blockers"), list) else []
        facts.append(projection)
        records.append({"source_id": source.source_id, "source_sha256": "sha256:" + hashlib.sha256(raw).hexdigest()})
    observation = {"schema_version": "blueprint_agent_run_observation.v1", "sources": facts,
                   "observation_digest": digest(facts), "source_records": records, "proof_effect": "none"}
    return observation


def validate_current_observation(service, binding: SupervisionBinding, *, controller_recoveries=()):
    from .production import _read_private
    from .supervision_authority import allowance_is_current
    from .recovery_lineage import recovery_binding_authorized
    plan_path = Path(service.config.supervision_store_root or service.journal.root / "supervision-plans") / (binding.watch_id + ".json")
    plan = SupervisionPlan.model_validate_json(_read_private(plan_path))
    if (not plan.enabled or time.time() >= plan.expires_at or plan.plan_digest != binding.watch_digest
            or plan.sources != binding.sources or not allowance_is_current(service, plan)):
        raise AgentExecutionError("agent_supervision_authority_revoked")
    if plan.automatic_intent_digest and (not service.config.automatic_run_supervision
            or any(not recovery_binding_authorized(service, row) for row in controller_recoveries)):
        raise AgentExecutionError("agent_automatic_supervision_scope_revoked")
    if observe(binding.sources)["observation_digest"] != binding.observation_digest:
        raise AgentExecutionError("agent_supervision_observation_stale")


def _state_path(service, plan):
    return service.journal.root / "supervision" / plan.source_commit / (plan.watch_id + ".json")


def ownership_key(source_commit, run_id, allowance_digest=None):
    value = {"source_commit": source_commit, "run_id": run_id}
    if allowance_digest is not None:
        value["allowance_digest"] = allowance_digest
    return "supervision_owner_" + digest(value)[7:]


def ownership_event(service, run_id):
    allowances = [row for row in service.config.automatic_supervision_allowances if row.intent_id == run_id]
    if len(allowances) > 1:
        raise AgentExecutionError("automatic_supervision_allowance_scope_invalid")
    if allowances:
        event = service.journal.event(ownership_key(service.config.source_commit, run_id, allowances[0].allowance_digest))
        if event is not None:
            return event
    return service.journal.event(ownership_key(service.config.source_commit, run_id))


def _write_state(service, plan, state):
    state["state_digest"] = digest({key: value for key, value in state.items() if key != "state_digest"})
    write_json(_state_path(service, plan), state)


def progress_plan(service, plan: SupervisionPlan):
    from .production import TaskRecord, _read_private
    from .supervision_authority import allowance_is_current

    if plan.source_commit != service.config.source_commit:
        raise AgentExecutionError("agent_supervision_release_mismatch")
    template = service.record(plan.template_task_id)
    template_digest = digest({key: value for key, value in template.model_dump(mode="json").items() if key != "enabled"})
    if (template.task.run_id != plan.run_id or template.autostart or template.supervision is not None
            or template.episode_investigation is not None or template.visual_investigation is not None):
        raise AgentExecutionError("agent_supervision_template_not_admitted")
    # A durable workflow owner is separate from any provider session.
    with service.journal.own_task("supervision-run:" + plan.run_id):
        path = _state_path(service, plan)
        state = json.loads(_read_private(path)) if path.exists() else {
            "schema_version": "blueprint_agent_supervision_state.v1", "watch_digest": plan.plan_digest,
            "run_id": plan.run_id, "template_digest": template_digest,
            "revisions": [], "reserved_inference_usd": 0.0, "active_task_id": None,
            "status": "watching"}
        if path.exists() and (state.get("state_digest") != digest({k: v for k, v in state.items() if k != "state_digest"})
                or state.get("watch_digest") != plan.plan_digest or state.get("run_id") != plan.run_id
                or state.get("template_digest") != template_digest):
            raise AgentExecutionError("agent_supervision_plan_or_state_changed")
        service.journal.record_event(ownership_key(plan.source_commit, plan.run_id, plan.automatic_allowance_digest),
            {"watch_id": plan.watch_id, "watch_digest": plan.plan_digest, "run_id": plan.run_id})
        active_id = state["active_task_id"]
        active = None
        if active_id:
            try:
                active = service.journal.task(active_id)
            except AgentExecutionError as exc:
                if str(exc) != "agent_task_missing":
                    raise
        automatic_revoked = bool(plan.automatic_intent_digest and (
            not service.config.automatic_run_supervision or not allowance_is_current(service, plan)))
        if not plan.enabled or not template.enabled or time.time() >= plan.expires_at or automatic_revoked:
            if active is not None and active["state"] not in TERMINAL_STATES:
                service.service.cancel(active_id)
            elif (active is not None and active["cleanup_state"] == "not_requested"
                    and not service.journal.unsettled_operations(active_id)
                    and service.journal.successor(active_id) is None):
                service.service.request_cleanup(active_id)
            state["status"] = "revoked" if not plan.enabled or not template.enabled or automatic_revoked else "expired"
            _write_state(service, plan, state)
            return state
        observation = observe(plan.sources)
        if active is not None and active["state"] not in TERMINAL_STATES:
            if observation["observation_digest"] != state["revisions"][-1]["observation_digest"]:
                service.service.cancel(active_id)
                state["status"] = "settling_stale_revision"
                _write_state(service, plan, state)
            return state
        if active is not None and service.journal.unsettled_operations(active_id):
            state["status"] = "reconciling_prior_operations"
            _write_state(service, plan, state)
            return state
        # Recover a crash after durable reservation and task write, before queue admission.
        if active_id and active is None:
            service._enqueue_record(service.record(active_id), _run_lock_held=True)
            return state
        if (plan.automatic_intent_digest and active is not None
                and all(row.get("status") in {"completed", "cancelled"} for row in observation["sources"])
                and observation["observation_digest"] == state["revisions"][-1]["observation_digest"]):
            if active["cleanup_state"] != "deleted":
                service.service.request_cleanup(active_id)
            state["status"] = "completed" if active["cleanup_state"] == "deleted" else "final_cleanup_pending"
            _write_state(service, plan, state)
            return state
        if any(row["observation_digest"] == observation["observation_digest"] for row in state["revisions"]):
            state["status"] = "waiting_for_new_evidence"
            _write_state(service, plan, state)
            return state
        facts = {row["source_id"]: row for row in observation["sources"]}
        if any(source.eligible_statuses and facts[source.source_id].get("status", facts[source.source_id].get("state"))
               not in source.eligible_statuses for source in plan.sources):
            return state
        budget = template.task.admission.inference_budget_usd
        if (len(state["revisions"]) >= plan.maximum_revisions
                or state["reserved_inference_usd"] + budget > plan.maximum_reserved_inference_usd):
            state["status"] = "reserved_inference_limit_reached"
            _write_state(service, plan, state)
            return state
        # A different admitted task may already own this run. Never introduce
        # a second supervisor while its work is still pending.
        if any(task["task"]["run_id"] == plan.run_id for task in service.journal.tasks(limit=1000)):
            raise AgentExecutionError("agent_supervision_run_has_active_owner")
        task_id = "watch-" + digest({"watch": plan.plan_digest, "observation": observation["observation_digest"]})[7:]
        context_value = json.loads(json.dumps(template.context))
        context_value["fresh_scene_preparation_status"] = {
            **observation, "status_digest": observation["observation_digest"],
            "status": facts[plan.sources[0].source_id].get("status", facts[plan.sources[0].source_id].get("state", "unclassified")),
            "first_blocker": next((code for row in facts.values() for code in row.get("blockers", [])), None),
            "next_required_stage": None,
        }
        context_value["supervisor_output_dir"] = str(service.journal.root / "supervisor" / task_id)
        context = SupervisorContext(**context_value)
        payload = [*template.task.input, {"role": "user", "content": json.dumps({
            "current_run_observation": observation, "context_revision": context_revision(context),
            "instruction": "Re-read this revision. Revisit the admitted capability if the new evidence requires it. "
                           "Prior outputs are retained evidence, not authority to repeat completed work.",
        }, sort_keys=True)}]
        prior = AgentTask.model_validate(active["task"]) if active else None
        deadline = min(time.time() + plan.revision_ttl_seconds, plan.expires_at, template.task.admission.expires_at)
        values = template.task.model_dump(mode="json")
        values.update(task_id=task_id, context_revision=context_revision(context), input=payload, deadline=deadline)
        # Continuation preserves an existing session only when its owned turn
        # completed normally; cancelled/failed sessions are never revived.
        if (prior is not None and prior.admission.runtime == RUNTIME_API and active["state"] == "completed"
                and not active["cancel_requested"] and active["cleanup_state"] == "not_requested"):
            values["parent_task_id"] = prior.task_id
        allowed = set(template.task.admission.allowed_input_digests)
        if prior:
            allowed.update(prior.admission.allowed_input_digests)
        allowed.update((digest(payload), digest(observation)))
        values["admission"]["allowed_input_digests"] = sorted(allowed)
        values["admission"]["expires_at"] = deadline
        task = AgentTask.model_validate(values)
        record = TaskRecord(**{**template.model_dump(mode="json"), "task": task,
            "autostart": False, "cleanup_when_terminal": False,
            "owner_client_ids": ("blueprint-webapp",) if plan.automatic_intent_digest else template.owner_client_ids,
            "context": asdict(context), "supervision": SupervisionBinding(
                watch_id=plan.watch_id, watch_digest=plan.plan_digest,
                observation_digest=observation["observation_digest"], sources=plan.sources)})
        if plan.automatic_intent_digest:
            from .supervision_producer import automatic_failure_revision, reserve_automatic_revision
            failure_record = automatic_failure_revision(service, plan, task_id, deadline)
            if failure_record is not None:
                record = TaskRecord(**{**failure_record.model_dump(mode="json"),
                    "supervision": record.supervision.model_dump(mode="json"), "autostart": False,
                    "cleanup_when_terminal": False})
                task = record.task
            # Changing the tool scope starts a new session only after the old
            # session is cleaned. It never creates a competing reasoning owner.
            compatible = bool(prior is not None and prior.admission.runtime == RUNTIME_API
                and task.admission.runtime == RUNTIME_API and active["state"] == "completed"
                and not active["cancel_requested"] and active["cleanup_state"] == "not_requested"
                and prior.capability == task.capability
                and prior.instructions == task.instructions and prior.tool_digests == task.tool_digests
                and prior.admission.authority_digest == task.admission.authority_digest)
            if compatible:
                values = task.model_dump(mode="json")
                values["parent_task_id"] = prior.task_id
                values["admission"]["allowed_input_digests"] = sorted(
                    set(task.admission.allowed_input_digests) | set(prior.admission.allowed_input_digests))
                task = AgentTask.model_validate(values)
                record = TaskRecord(**{**record.model_dump(mode="json"), "task": task.model_dump(mode="json")})
            if prior is not None and (prior.admission.runtime != RUNTIME_API or not compatible):
                if active["cleanup_state"] != "deleted":
                    service.service.request_cleanup(active_id)
                    state["status"] = "settling_prior_scope"
                    _write_state(service, plan, state)
                    return state
                task = AgentTask.model_validate({**task.model_dump(mode="json"), "parent_task_id": None})
                record = TaskRecord(**{**record.model_dump(mode="json"), "task": task.model_dump(mode="json")})
            if not reserve_automatic_revision(service, plan, task_id, budget):
                state["status"] = "reserved_inference_limit_reached"
                _write_state(service, plan, state)
                return state
        event_id = "supervision_revision_" + task_id
        intent = service.journal.event(event_id)
        if intent is None:
            service.journal.record_event(event_id, {"watch_digest": plan.plan_digest,
                "record": record.model_dump(mode="json")})
        else:
            if intent["watch_digest"] != plan.plan_digest:
                raise AgentExecutionError("agent_supervision_revision_intent_conflict")
            record = TaskRecord.model_validate(intent["record"])
            task = record.task
        destination = Path(service.config.task_store_root) / (task_id + ".json")
        if destination.exists():
            if digest(json.loads(_read_private(destination))) != digest(record.model_dump(mode="json")):
                raise AgentExecutionError("agent_supervision_task_write_conflict")
        else:
            write_json(destination, record.model_dump(mode="json"))
            destination.chmod(0o640)
        state["reserved_inference_usd"] += budget
        state["active_task_id"] = task_id
        state["revisions"].append({"task_id": task_id, "task_digest": task.task_digest,
            "observation_digest": observation["observation_digest"], "reserved_inference_usd": budget})
        state["status"] = "executing_revision"
        _write_state(service, plan, state)
        service._enqueue_record(record, _run_lock_held=True)
        return state


def progress_supervision(service):
    from .production import _read_private

    if (service.journal.root / "release_drain.json").exists():
        return []

    root = service.config.supervision_store_root or service.journal.root / "supervision-plans"
    results = []
    for path in sorted(Path(root).glob("*.json")):
        try:
            plan = SupervisionPlan.model_validate_json(_read_private(path))
            if path.stem != plan.watch_id:
                raise AgentExecutionError("agent_supervision_plan_identity_mismatch")
            state = progress_plan(service, plan)
            results.append({"watch_id": plan.watch_id, "status": state["status"], "active_task_id": state["active_task_id"]})
        except (ValueError, AgentExecutionError, OSError) as exc:
            code = str(exc) if isinstance(exc, AgentExecutionError) else "agent_supervision_plan_refused"
            results.append({"watch_id": path.stem, "status": "refused", "error_code": code})
    return results
