"""Turn new failures of one admitted preparation into durable investigations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..common import write_json
from .contracts import AgentExecutionError, DIGEST, IDENTIFIER, digest


class FailureSubscription(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["blueprint_agent_failure_subscription.v1"] = "blueprint_agent_failure_subscription.v1"
    subscription_id: str = Field(pattern=IDENTIFIER)
    run_id: str = Field(pattern=IDENTIFIER)
    parent_preparation_id: str = Field(pattern=IDENTIFIER)
    parent_request_digest: str = Field(pattern=DIGEST)
    child_queue_root: str
    parent_queue_root: str
    input_root: str
    approved_roots: tuple[str, ...] = Field(min_length=1)
    owner_client_id: str = "blueprint-webapp"
    maximum_tasks: int = Field(default=3, ge=1, le=10)
    per_task_budget_usd: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    expires_at: float = Field(gt=0, allow_inf_nan=False)


def register_preparation_failure_subscription(*, preparation_link, controller_config, agent_config_path="/etc/blueprint/agent-execution.json"):
    """Called by the existing trusted preparation producer, never by a model."""
    from .production import ProductionConfig, _read_private
    path = Path(agent_config_path)
    if not path.exists():
        return None
    config = ProductionConfig.model_validate_json(_read_private(path))
    if not (config.automatic_failure_investigation or config.automatic_run_supervision):
        return None
    from ..task_evaluation_stage_replay import DEFAULT_INPUT_ROOT, DEFAULT_APPROVED_ROOTS, DEFAULT_QUEUE_ROOT, DEFAULT_PARENT_QUEUE_ROOT
    policy_id = "preparation-" + digest({"id": preparation_link["preparation_id"], "request": preparation_link["request_digest"]})[7:]
    destination = Path(config.state_root) / "failure-subscriptions" / (policy_id + ".json")
    if destination.exists():
        return FailureSubscription.model_validate_json(_read_private(destination))
    policy = FailureSubscription(subscription_id=policy_id, run_id=preparation_link["intent_id"],
        parent_preparation_id=preparation_link["preparation_id"], parent_request_digest=preparation_link["request_digest"],
        child_queue_root=str(controller_config.get("child_queue_root") or DEFAULT_QUEUE_ROOT),
        parent_queue_root=str(controller_config.get("preparation_queue_root") or DEFAULT_PARENT_QUEUE_ROOT),
        input_root=str((controller_config.get("preparation_worker") or {}).get("input_root") or DEFAULT_INPUT_ROOT),
        approved_roots=tuple(str(path) for path in DEFAULT_APPROVED_ROOTS),
        per_task_budget_usd=min(1.0, config.max_task_budget_usd), expires_at=time.time() + 86400)
    write_json(destination, policy.model_dump(mode="json"))
    destination.chmod(0o640)
    return policy


def discover_retained_failures(service):
    from .production import _read_private
    from .prepare import prepare_retained_failure

    if not service.config.automatic_failure_investigation or (service.journal.root / "release_drain.json").exists():
        return []
    results = []
    for policy_path in sorted((service.journal.root / "failure-subscriptions").glob("*.json")):
        try:
            policy = FailureSubscription.model_validate_json(_read_private(policy_path))
            if policy_path.stem != policy.subscription_id or time.time() >= policy.expires_at:
                continue
            with service.journal.own_task("failure-subscription:" + policy.subscription_id):
                state_path = service.journal.root / "failure-subscription-state" / (policy.subscription_id + ".json")
                state = json.loads(_read_private(state_path)) if state_path.exists() else {
                    "subscription_digest": digest(policy.model_dump(mode="json")), "tasks": []}
                if state["subscription_digest"] != digest(policy.model_dump(mode="json")):
                    raise AgentExecutionError("agent_failure_subscription_changed")
                from .supervision import ownership_event
                if ownership_event(service, policy.run_id) is not None:
                    results.append({"subscription_id": policy.subscription_id, "state": "owned_by_persistent_supervisor"})
                    continue
                if service.config.automatic_run_supervision:
                    # Startup must not race the controller's registration and
                    # create a second inference owner with a separate budget.
                    results.append({"subscription_id": policy.subscription_id, "state": "waiting_for_persistent_supervisor"})
                    continue
                for job_path in sorted((Path(policy.child_queue_root) / "failed").glob("*.json")):
                    raw = _read_private(job_path)
                    job = json.loads(raw)
                    if (job.get("parent_preparation_id") != policy.parent_preparation_id
                            or job.get("parent_request_digest") != policy.parent_request_digest):
                        continue
                    from .recovery_lineage import resolve_recovery_binding
                    recovery = resolve_recovery_binding(service, intent_id=policy.run_id,
                        parent_request_digest=policy.parent_request_digest, parent_queue_root=policy.parent_queue_root)
                    job_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
                    identity = {"subscription": state["subscription_digest"], "job": job_sha256,
                                "source_commit": service.config.source_commit}
                    if recovery is not None:
                        identity["controller_recovery_digest"] = digest(recovery.model_dump(mode="json"))
                    task_id = "auto-failure-" + digest(identity)[7:]
                    entry = next((row for row in state["tasks"] if row["task_id"] == task_id), None)
                    if entry is None:
                        if len(state["tasks"]) >= policy.maximum_tasks:
                            continue
                        entry = {"task_id": task_id, "job_sha256": job_sha256, "source_commit": service.config.source_commit,
                                 "reserved_inference_usd": policy.per_task_budget_usd}
                        state["tasks"].append(entry)
                        write_json(state_path, state)  # reserve before a task can become runnable
                    try:
                        existing = service.journal.task(task_id)
                        results.append({"task_id": task_id, "state": existing["state"]})
                        continue
                    except AgentExecutionError as exc:
                        if str(exc) != "agent_task_missing":
                            raise
                    path = Path(service.config.task_store_root) / (task_id + ".json")
                    if path.exists():
                        record = service.record(task_id)
                        if (record.task.run_id != policy.run_id or not record.stage_replays
                                or record.stage_replays[0].job_sha256 != job_sha256
                                or record.task.admission.inference_budget_usd != policy.per_task_budget_usd):
                            raise AgentExecutionError("agent_failure_task_binding_changed")
                    else:
                        record = prepare_retained_failure(service, task_id=task_id, run_id=policy.run_id,
                            child_id=job["child_id"], owner_client_id=policy.owner_client_id,
                            inference_budget_usd=policy.per_task_budget_usd,
                            runtime=service.config.automatic_failure_runtime, controller_recovery=recovery,
                            queue_root=Path(policy.child_queue_root), parent_queue_root=Path(policy.parent_queue_root),
                            input_root=Path(policy.input_root), approved_roots=tuple(Path(path) for path in policy.approved_roots),
                            ttl_seconds=min(600, max(1, int(policy.expires_at - time.time()))))
                    service._enqueue_record(record)
                    results.append({"task_id": task_id, "state": "queued"})
        except (OSError, ValueError, AgentExecutionError):
            results.append({"subscription_id": policy_path.stem, "state": "admission_refused"})
    return results
