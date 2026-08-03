"""Cloud Tasks dispatch for job envelopes — explicit-job transport lane.

Google distinguishes Cloud Tasks from Pub/Sub by task-name deduplication,
scheduling, rate control, and precise retry limits — the right fit for "run
this job once, at this endpoint". This module derives the task name from the
envelope id, so re-dispatching the same job content is deduplicated by the
queue itself (note: task-name tombstones expire after roughly one hour;
Blueprint's own idempotent claims remain the durable dedup).

Fail-closed boundaries:
- Only allowlisted lanes may dispatch (``fixture`` until the strangler
  migration proves a lane); widening the set is a deliberate code change.
- Missing configuration or the optional ``google-cloud-tasks`` dependency
  downgrades to an explicit ``unavailable`` result — never a partial send.
- Envelopes are validated (including the no-credentials rule) before any
  network call; the queue carries job identity, never provider credentials.
- Dispatch is a mutation: exactly one attempt, no automatic retry. An
  ambiguous network outcome is reported as such for reconciliation by task
  name, mirroring the paid-lane ``allocation_outcome_ambiguous`` discipline.

The deploy-side precedent is ``functions/storage_trigger.py`` (mode switch
``pubsub | cloud_tasks | direct``); this module is the control-plane
equivalent behind the strangler envelope.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from .job_transport_envelope import validate_job_envelope

CLOUD_TASKS_ALLOWED_LANES = frozenset({"fixture"})

TASKS_QUEUE_ENV = "BLUEPRINT_JOB_TRANSPORT_TASKS_QUEUE"
TASKS_LOCATION_ENV = "BLUEPRINT_JOB_TRANSPORT_TASKS_LOCATION"
TASKS_URL_ENV = "BLUEPRINT_JOB_TRANSPORT_TASKS_URL"
TASKS_SERVICE_ACCOUNT_ENV = "BLUEPRINT_JOB_TRANSPORT_TASKS_SERVICE_ACCOUNT"

DISPATCH_SCHEMA_VERSION = "job_transport_cloud_tasks_dispatch.v1"


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _project(env: Mapping[str, str]) -> str:
    for name in ("PIPELINE_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"):
        value = str(env.get(name) or "").strip()
        if value:
            return value
    return ""


def _result(status: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": DISPATCH_SCHEMA_VERSION, "status": status, **fields}


def dispatch_job_envelope_task(
    *,
    envelope: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Dispatch one envelope as a named Cloud Task (single attempt)."""

    mapping = _env(env)
    envelope_id = str(envelope.get("envelope_id") or "")

    blockers = validate_job_envelope(envelope)
    if blockers:
        return _result("blocked_envelope_invalid", envelope_id=envelope_id, blockers=blockers)

    lane = str(envelope.get("source_lane") or "")
    if lane not in CLOUD_TASKS_ALLOWED_LANES:
        return _result(
            "blocked_lane_not_allowlisted",
            envelope_id=envelope_id,
            blockers=[f"cloud_tasks_lane_not_allowlisted:{lane}"],
        )

    config_blockers: list[str] = []
    queue = str(mapping.get(TASKS_QUEUE_ENV) or "").strip()
    location = str(mapping.get(TASKS_LOCATION_ENV) or "").strip()
    url = str(mapping.get(TASKS_URL_ENV) or "").strip()
    project = _project(mapping)
    if not queue:
        config_blockers.append("cloud_tasks_queue_missing")
    if not location:
        config_blockers.append("cloud_tasks_location_missing")
    if not url:
        config_blockers.append("cloud_tasks_url_missing")
    if not project:
        config_blockers.append("cloud_tasks_project_missing")
    if config_blockers:
        return _result("unavailable", envelope_id=envelope_id, blockers=config_blockers)

    if client is None:
        try:
            from google.cloud import tasks_v2  # optional dependency
        except ImportError:
            return _result(
                "unavailable",
                envelope_id=envelope_id,
                blockers=["google_cloud_tasks_dependency_missing"],
            )
        client = tasks_v2.CloudTasksClient()

    parent = client.queue_path(project, location, queue)
    task: dict[str, Any] = {
        # Task name == envelope id: the queue deduplicates re-dispatch of the
        # same job content while the tombstone lives.
        "name": f"{parent}/tasks/{envelope_id}",
        "http_request": {
            "http_method": "POST",
            "url": url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(dict(envelope), sort_keys=True).encode("utf-8"),
        },
    }
    service_account = str(mapping.get(TASKS_SERVICE_ACCOUNT_ENV) or "").strip()
    if service_account:
        task["http_request"]["oidc_token"] = {"service_account_email": service_account}

    try:
        response = client.create_task(request={"parent": parent, "task": task})
    except Exception as exc:  # noqa: BLE001 - classify, never auto-retry a mutation
        if type(exc).__name__ == "AlreadyExists":
            return _result("deduplicated", envelope_id=envelope_id, task_name=task["name"])
        return _result(
            "dispatch_outcome_ambiguous",
            envelope_id=envelope_id,
            task_name=task["name"],
            error=f"{type(exc).__name__}:{exc}",
            reconcile_by="task_name_lookup",
        )
    return _result(
        "dispatched",
        envelope_id=envelope_id,
        task_name=str(getattr(response, "name", task["name"])),
    )


__all__ = [
    "CLOUD_TASKS_ALLOWED_LANES",
    "DISPATCH_SCHEMA_VERSION",
    "TASKS_QUEUE_ENV",
    "TASKS_LOCATION_ENV",
    "TASKS_URL_ENV",
    "TASKS_SERVICE_ACCOUNT_ENV",
    "dispatch_job_envelope_task",
]
