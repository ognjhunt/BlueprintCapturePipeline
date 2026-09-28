"""Durably accept one signed team-owned G1 policy choice without spend.

Acceptance is intentionally separate from operator approval and paid launch.
The existing owner-scoped task registry supplies the trusted scene packet;
the team profile is validated against that packet before an immutable intent
is written. A later dispatcher must recheck both records and find an exact
operator approval before any site observation or provider allocation.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_campaign_intake import (
    REGISTRY_SCHEMA,
    _read,
    _selected_binding,
)
from .native_g1_team_policy_run_request import validate_g1_team_policy_run_request
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)
from .task_evaluation_packet_planning_setup import make_packet_planning_setup


QUEUE_ENV = "BLUEPRINT_NATIVE_G1_TEAM_POLICY_QUEUE_ROOT"
DEFAULT_QUEUE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/native-g1-team-policy-runs")
INTENT_SCHEMA = "native_g1_team_policy_run_intent.v1"
RECEIPT_SCHEMA = "native_g1_team_policy_run_intake_receipt.v1"


def stage_g1_team_policy_run(
    *,
    value: dict[str, Any],
    registry_path: Path,
    queue_root: Path,
    authenticated_client: str,
    trusted_clients: set[str],
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Record one owner-selected policy while making zero provider mutations."""

    if not authenticated_client or authenticated_client not in trusted_clients:
        raise ValueError("g1_team_policy_issuer_not_authorized")
    if not isinstance(value, dict):
        raise ValueError("g1_team_policy_request_invalid")
    registry = _read(Path(registry_path), field="registry_digest")
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("g1_team_policy_registry_invalid")
    binding = _selected_binding(registry, value)
    setup = make_packet_planning_setup(source_packet_dir=Path(binding["source_packet_dir"]))
    if (
        binding["scene_id"] != setup["scene_id"]
        or binding["task_id"] != setup["task_id"]
        or binding["source_packet_receipt_digest"] != setup["source_packet_receipt_digest"]
    ):
        raise ValueError("g1_team_policy_binding_packet_mismatch")
    moment = time.time() if now_epoch is None else now_epoch
    request = validate_g1_team_policy_run_request(
        value, trusted_setup=setup, authenticated_owner=binding["owner"],
        now_epoch=moment,
    )
    root = Path(queue_root)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("g1_team_policy_queue_root_invalid")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    intent_id = "g1-team-policy-" + digest({
        "owner": request["owner"], "run_id": request["run_id"]
    }).removeprefix("sha256:")
    record = {
        "schema_version": INTENT_SCHEMA,
        "status": "accepted_pending_operator_approval",
        "intent_id": intent_id,
        "request": request,
        "authenticated_issuer": authenticated_client,
        "registry_digest": registry["registry_digest"],
        "binding_digest": digest(binding),
        "binding": binding,
        "accepted_at_epoch": moment,
        "claim_ceiling": "development_only",
        "provider_mutation_performed": False,
    }
    record["intent_digest"] = digest(record, digest_field="intent_digest")
    descriptor = os.open(root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        directory = root / intent_id
        if directory.is_symlink():
            raise ValueError("g1_team_policy_intent_path_unsafe")
        directory.mkdir(mode=0o750, exist_ok=True)
        path = directory / "intent.json"
        if path.exists() or path.is_symlink():
            prior = _read(path, field="intent_digest")
            if (
                prior.get("request") != request
                or prior.get("binding_digest") != record["binding_digest"]
                or prior.get("authenticated_issuer") != authenticated_client
            ):
                raise ValueError("g1_team_policy_idempotency_conflict")
            record = prior
        else:
            write_exclusive(path, record)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "accepted_pending_operator_approval",
        "intent_id": intent_id,
        "intent_digest": record["intent_digest"],
        "request_digest": request["request_digest"],
        "provider_mutation_performed_inside_http_request": False,
    }
    receipt["receipt_digest"] = digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = ["DEFAULT_QUEUE_ROOT", "QUEUE_ENV", "stage_g1_team_policy_run"]
