"""Durably stage a signed G1 team request against operator-owned packet paths.

HTTP acceptance is no-spend. A separate controller must reverify the retained
binding, rights, release, spend admission, watchdog, and provider teardown.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_development_pair import PAIR_ORDER
from .native_g1_team_campaign_request import validate_g1_team_campaign_request
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)
from .task_evaluation_packet_planning_setup import make_packet_planning_setup


REGISTRY_SCHEMA = "native_g1_team_campaign_registry.v1"
INTENT_SCHEMA = "native_g1_team_campaign_intent.v1"
RECEIPT_SCHEMA = "native_g1_team_campaign_intake_receipt.v1"
REGISTRY_ENV = "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_REGISTRY_PATH"
QUEUE_ENV = "BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_QUEUE_ROOT"
BINDING_FIELDS = frozenset({
    "owner", "scene_id", "task_id", "source_packet_receipt_digest",
    "source_packet_dir", "manipulation_packet_dir", "movement_packet_dir",
    "navigation_authority_path", "publisher_source_dir",
    "runtime_source_receipt_path", "rights_review_paths",
})
DIRECTORIES = (
    "source_packet_dir", "manipulation_packet_dir", "movement_packet_dir",
    "publisher_source_dir",
)
FILES = ("navigation_authority_path", "runtime_source_receipt_path")


def _read(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_team_campaign_record_unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get(field) != digest(value, digest_field=field):
        raise ValueError("g1_team_campaign_record_invalid")
    return value


def _binding_path(value: Any, *, directory: bool) -> Path:
    if not isinstance(value, str):
        raise ValueError("g1_team_campaign_binding_path_invalid")
    path = Path(value)
    if (
        not path.is_absolute() or path.is_symlink()
        or not (path.is_dir() if directory else path.is_file())
    ):
        raise ValueError("g1_team_campaign_binding_path_invalid")
    return path


def _selected_binding(registry: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    if (
        set(registry) != {"schema_version", "bindings", "registry_digest"}
        or registry.get("schema_version") != REGISTRY_SCHEMA
        or not isinstance(registry.get("bindings"), list)
        or not 1 <= len(registry["bindings"]) <= 100
    ):
        raise ValueError("g1_team_campaign_registry_invalid")
    matches = [row for row in registry["bindings"] if isinstance(row, dict)
        and row.get("owner") == request.get("owner")
        and row.get("source_packet_receipt_digest")
        == request.get("source_packet_receipt_digest")]
    if len(matches) != 1:
        raise ValueError("g1_team_campaign_binding_unavailable")
    binding = matches[0]
    if set(binding) != BINDING_FIELDS:
        raise ValueError("g1_team_campaign_binding_invalid")
    for field in DIRECTORIES:
        _binding_path(binding[field], directory=True)
    for field in FILES:
        _binding_path(binding[field], directory=False)
    rights = binding["rights_review_paths"]
    if not isinstance(rights, dict) or set(rights) != set(PAIR_ORDER):
        raise ValueError("g1_team_campaign_binding_rights_invalid")
    for value in rights.values():
        _binding_path(value, directory=False)
    return binding


@contextmanager
def _queue_lock(root: Path):
    descriptor = os.open(
        root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def stage_g1_team_campaign(
    *,
    value: dict[str, Any],
    registry_path: Path,
    queue_root: Path,
    authenticated_client: str,
    trusted_clients: set[str],
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Record one immutable authorized choice without launching resources."""

    if not authenticated_client or authenticated_client not in trusted_clients:
        raise ValueError("g1_team_campaign_issuer_not_authorized")
    registry = _read(registry_path, field="registry_digest")
    if not isinstance(value, dict):
        raise ValueError("g1_team_campaign_request_invalid")
    binding = _selected_binding(registry, value)
    setup = make_packet_planning_setup(
        source_packet_dir=Path(binding["source_packet_dir"])
    )
    if (
        binding["scene_id"] != setup["scene_id"]
        or binding["task_id"] != setup["task_id"]
        or binding["source_packet_receipt_digest"]
        != setup["source_packet_receipt_digest"]
    ):
        raise ValueError("g1_team_campaign_binding_packet_mismatch")
    moment = time.time() if now_epoch is None else now_epoch
    request = validate_g1_team_campaign_request(
        value, trusted_setup=setup, authenticated_owner=binding["owner"],
        now_epoch=moment,
    )
    root = Path(queue_root)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("g1_team_campaign_queue_root_invalid")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    intent_id = "g1-" + digest({
        "owner": request["owner"], "run_id": request["run_id"]
    }).removeprefix("sha256:")
    record = {
        "schema_version": INTENT_SCHEMA,
        "status": "accepted_not_dispatched",
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
    with _queue_lock(root):
        directory = root / intent_id
        if directory.is_symlink():
            raise ValueError("g1_team_campaign_queue_record_unsafe")
        directory.mkdir(mode=0o750, exist_ok=True)
        path = directory / "intent.json"
        if path.exists() or path.is_symlink():
            prior = _read(path, field="intent_digest")
            if (
                prior.get("request") != request
                or prior.get("binding_digest") != record["binding_digest"]
                or prior.get("authenticated_issuer") != authenticated_client
            ):
                raise ValueError("g1_team_campaign_idempotency_conflict")
            record = prior
        else:
            write_exclusive(path, record)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "accepted_not_dispatched",
        "intent_id": intent_id,
        "intent_digest": record["intent_digest"],
        "request_digest": request["request_digest"],
        "provider_mutation_performed_inside_http_request": False,
    }
    receipt["receipt_digest"] = digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = ["QUEUE_ENV", "REGISTRY_ENV", "REGISTRY_SCHEMA", "stage_g1_team_campaign"]
