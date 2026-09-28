"""Conservatively join legacy failure holds into the same intent lifetime cap.

Migration adds immutable events, never deletes subscription history or refunds
estimates. Stable task identities prevent a hold appearing twice in the total.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from .contracts import AgentExecutionError, IDENTIFIER, digest


def failure_hold_inventory(service, run_id):
    from .failure_events import FailureSubscription
    from .production import _read_private
    rows = {}
    paths = sorted((service.journal.root / "failure-subscriptions").glob("*.json"))
    if len(paths) > 10000:
        raise AgentExecutionError("automatic_supervision_failure_inventory_unbounded")
    for path in paths:
        policy = FailureSubscription.model_validate_json(_read_private(path))
        if policy.run_id != run_id:
            continue
        state_path = service.journal.root / "failure-subscription-state" / path.name
        if not state_path.exists():
            continue
        state = json.loads(_read_private(state_path))
        if (path.stem != policy.subscription_id or state.get("subscription_digest") != digest(policy.model_dump(mode="json"))
                or len(state.get("tasks", [])) > policy.maximum_tasks):
            raise AgentExecutionError("automatic_supervision_failure_reservation_invalid")
        for entry in state["tasks"]:
            task_id, amount = entry.get("task_id"), entry.get("reserved_inference_usd")
            if (not isinstance(task_id, str) or re.fullmatch(IDENTIFIER, task_id) is None
                    or not task_id.startswith("auto-failure-")
                    or not isinstance(amount, (int, float)) or isinstance(amount, bool)
                    or not math.isfinite(amount) or not 0 < amount == policy.per_task_budget_usd):
                raise AgentExecutionError("automatic_supervision_failure_reservation_invalid")
            task_path = Path(service.config.task_store_root) / (task_id + ".json")
            if task_path.exists():
                task = json.loads(_read_private(task_path)).get("task", {})
                if (task.get("task_id") != task_id or task.get("run_id") != run_id
                        or task.get("source_commit") != entry.get("source_commit")
                        or task.get("admission", {}).get("inference_budget_usd") != amount):
                    raise AgentExecutionError("automatic_supervision_failure_task_grant_changed")
            row = {"task_id": task_id, "reserved_inference_usd": amount,
                   "source_subscription_id": policy.subscription_id, "source_entry_digest": digest(entry)}
            if task_id in rows and rows[task_id] != row:
                raise AgentExecutionError("automatic_supervision_failure_reservation_conflict")
            rows[task_id] = row
    return list(rows.values())


def migrate_failure_holds(service, plan):
    """Called while holding the existing automatic-supervision-budget lock."""
    prefix = "automatic_supervision_budget_" + plan.automatic_intent_digest[7:] + "_"
    for row in failure_hold_inventory(service, plan.run_id):
        event_id = prefix + digest(row["task_id"])[7:]
        existing = service.journal.event(event_id)
        if existing is not None:
            if (existing.get("task_id") != row["task_id"]
                    or existing.get("reserved_inference_usd") != row["reserved_inference_usd"]
                    or existing.get("source_entry_digest", row["source_entry_digest"]) != row["source_entry_digest"]):
                raise AgentExecutionError("automatic_supervision_failure_reservation_conflict")
            continue
        service.journal.record_event(event_id, {**row,
            "maximum_revisions": plan.maximum_revisions,
            "maximum_reserved_inference_usd": plan.maximum_reserved_inference_usd,
            "source_kind": "conservative_failure_subscription_hold_adoption",
            **({"allowance_digest": plan.automatic_allowance_digest} if plan.automatic_allowance_digest else {})})
