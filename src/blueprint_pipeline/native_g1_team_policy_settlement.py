"""Reconcile one selected G1 episode and deliver it to its existing private page.

Reopens frozen inputs, actual worker evidence and posted provider charges.
This post-run worker never renews launch authority or allocates resources.
"""

from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
import re
from typing import Any
import urllib.parse

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest as digest
from .native_g1_private_review_delivery import materialize_g1_private_review_delivery
from .native_g1_private_review_ingest import _verified_review, ingest_g1_private_review
from .native_g1_team_campaign_intake import _read
from .native_g1_team_campaign_settlement import _instance_and_label
from .native_g1_team_paid_policy import CONSUMPTION_FILENAME, INSTANCE_LABEL_PREFIX
from .native_g1_team_policy_approval import validate_g1_team_policy_approval
from .native_g1_team_policy_dispatcher import FINAL_SCHEMA, START_SCHEMA
from .native_g1_team_policy_preparation import SCHEMA as PREPARATION_SCHEMA
from .native_g1_team_policy_run_intake import INTENT_SCHEMA
from .native_g1_team_policy_run_request import validate_g1_team_policy_run_request
from .native_g1_team_review_evidence import _json, verify_retained_g1_team_review
from .policy_canary_billing_recovery import _candidate_sources
from .task_evaluation_launch_preparation_queue import _write_launch_preparation_record_exclusive_locked as write_exclusive
from .task_evaluation_packet_planning_setup import validate_packet_planning_setup
from .vast_official_billing_extractor import (
    VastOfficialBillingExtractionError, materialize_vast_official_same_goal_reconciliation,
    validate_vast_official_same_goal_reconciliation,
)


SCHEMA = "native_g1_team_policy_settlement.v1"
_ID = re.compile(r"g1-team-policy-[0-9a-f]{64}\Z")


def _directory(path: Path, *, required=False) -> None:
    if (not path.is_absolute() or path.resolve() != path
            or any(part.is_symlink() for part in (path, *path.parents))
            or (required and not path.is_dir()) or (path.exists() and not path.is_dir())):
        raise ValueError("g1_team_policy_settlement_path_invalid")


def _sealed(path: Path, field: str) -> dict[str, Any]:
    _json(path)  # Reject ancestor aliases as well as leaf symlinks.
    return _read(path, field=field)


def _pending(status: str, identity: str, **extra) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "status": status, "intent_id": identity,
            "provider_mutation_performed": False, **extra}


def settle_g1_team_policy(
    *, intent_path: Path, work_root: Path, billing_audit_root: Path,
    result_root: Path, webapp_url: str, sync_token: str, collect_zero=None,
) -> dict[str, Any]:
    """Serialize this intent's settlement; never launch a replacement attempt."""
    intent = _sealed(Path(intent_path), "intent_digest")
    identity, request = intent.get("intent_id"), intent.get("request")
    if (intent.get("schema_version") != INTENT_SCHEMA
            or not isinstance(identity, str) or _ID.fullmatch(identity) is None
            or not isinstance(request, dict) or Path(intent_path).parent.name != identity
            or identity != "g1-team-policy-" + digest({"owner": request.get("owner"), "run_id": request.get("run_id")}).removeprefix("sha256:")
            or intent.get("status") != "accepted_pending_operator_approval"
            or intent.get("claim_ceiling") != "development_only"
            or intent.get("provider_mutation_performed") is not False):
        raise ValueError("g1_team_policy_settlement_intent_invalid")
    for root in (Path(work_root), Path(billing_audit_root), Path(result_root)):
        _directory(root)
    directory = Path(work_root) / identity
    _directory(directory, required=True)
    descriptor = os.open(directory / ".settlement.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return _pending("settlement_owned_by_another_worker", identity)
        return _settle(intent=intent, directory=directory, billing_audit_root=Path(billing_audit_root),
                       result_root=Path(result_root), webapp_url=webapp_url, sync_token=sync_token,
                       collect_zero=collect_zero)
    finally:
        os.close(descriptor)


def _settle(*, intent, directory, billing_audit_root, result_root, webapp_url, sync_token, collect_zero):
    final = _sealed(directory / "dispatch_final.json", "dispatch_digest")
    start = _sealed(directory / "execution_started.json", "start_digest")
    prepared = _sealed(directory / "preparation.json", "preparation_digest")
    identity, request = intent["intent_id"], intent["request"]
    if (final.get("schema_version") != FINAL_SCHEMA
            or final.get("status") != "controller_completed_pending_billing_and_private_delivery"
            or final.get("controller_reported_episode_verified") is not True
            or final.get("run_teardown_confirmed_by_adapter") is not True
            or final.get("paid_allocator_exit_code") != 0 or final.get("adapter_status") != "completed"
            or start.get("schema_version") != START_SCHEMA or start.get("status") != "execution_started_once"
            or start.get("retry_cap") != 0 or prepared.get("schema_version") != PREPARATION_SCHEMA
            or prepared.get("status") != "bundle_prepared_not_executed"
            or any(value.get("intent_id") != identity or value.get("intent_digest") != intent["intent_digest"]
                   or value.get("implementation_commit") != prepared.get("implementation_commit")
                   for value in (prepared, start, final))
            or final.get("start_digest") != start["start_digest"]
            or start.get("preparation_digest") != prepared["preparation_digest"]):
        raise ValueError("g1_team_policy_settlement_controller_incomplete")
    bundle_path = directory / "bundle/native_g1_team_provider_bundle.v1.json"
    adapter_path = Path(str(final.get("adapter_result_path") or ""))
    run = directory / "run"
    if (prepared.get("bundle_receipt_path") != str(bundle_path)
            or adapter_path.parent != run or re.fullmatch(r"adapter_paid_[0-9]+\.json", adapter_path.name) is None):
        raise ValueError("g1_team_policy_settlement_frozen_path_changed")
    adapter = _json(adapter_path)
    if (digest(adapter) != final.get("adapter_result_digest")
            or adapter.get("attempt_root") != str(run / "attempts/attempt_001")):
        raise ValueError("g1_team_policy_settlement_adapter_changed")
    evidence = verify_retained_g1_team_review(adapter_result_path=adapter_path, bundle_receipt_path=bundle_path)
    bundle, packet, review = evidence.bundle, evidence.execution_packet, evidence.review
    if (packet.get("intent_id") != identity or packet.get("intent_digest") != intent["intent_digest"]
            or packet.get("request") != request or packet.get("implementation_commit") != prepared["implementation_commit"]
            or any(prepared.get(key) != bundle[key] for key in ("bundle_sha256", "manifest_digest", "execution_packet_digest"))
            or prepared.get("operator_approval_digest") != packet["operator_approval"]["approval_digest"]
            or prepared.get("policy_profile_digest") != packet["policy_profile_digest"]
            or prepared.get("objective_id") != packet["objective_id"]
            or prepared.get("authorization") != request["authorization"]
            or adapter.get("bundle_sha256") != bundle["bundle_sha256"]):
        raise ValueError("g1_team_policy_settlement_frozen_input_changed")
    consumed = _json(run / CONSUMPTION_FILENAME)
    if (consumed.get("schema_version") != "native_g1_team_paid_attempt_consumption.v1"
            or consumed.get("status") != "consumed" or consumed.get("retry_cap") != 0
            or consumed.get("execution_packet_digest") != packet["packet_digest"]
            or consumed.get("orchestrator_source_commit") != packet["implementation_commit"]
            or consumed.get("allocation_binding_digest") != adapter.get("allocation_binding_digest")
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(consumed.get("allocation_binding_digest"))) is None):
        raise ValueError("g1_team_policy_settlement_consumed_attempt_invalid")
    started = start.get("started_at_epoch")
    if type(started) not in (int, float) or not math.isfinite(started) or started <= 0:
        raise ValueError("g1_team_policy_settlement_start_time_invalid")
    # Historical validation proves approval at the consumed start. It does not
    # renew it or alter the launch loader's mandatory present-time checks.
    setup = validate_packet_planning_setup(packet["trusted_setup"])
    validate_g1_team_policy_run_request(request, trusted_setup=setup, authenticated_owner=request["owner"], now_epoch=started)
    validate_g1_team_policy_approval(packet["operator_approval"], profile=request["policy_profile"],
                                  trusted_setup=setup, authenticated_owner=request["owner"],
                                  objective_id=request["objective_id"], now_epoch=started)
    instance_id, label = _instance_and_label(adapter, expected_prefix=INSTANCE_LABEL_PREFIX)
    terminal_path = run / "adp_arena_vast_result.json"
    terminal = _json(terminal_path)
    if any(adapter.get(key) != value for key, value in terminal.items()):
        raise ValueError("g1_team_policy_settlement_terminal_adapter_mismatch")

    from .adp009d_provider_zero import collect_provider_zero_receipt
    zero = (collect_zero or collect_provider_zero_receipt)()
    if not isinstance(zero, dict) or zero.get("provider_zero_verified") is not True:
        return _pending("awaiting_global_provider_zero", identity,
                        blockers=(zero.get("blockers") if isinstance(zero, dict) else None) or ["global_provider_zero_unproven"])
    if (zero.get("schema_version") != "gpu_spend_guard.v1" or zero.get("live_instance_count") != 0
            or zero.get("receipt_digest") != canonical_digest(zero, digest_field="receipt_digest")):
        raise ValueError("g1_team_policy_settlement_provider_zero_invalid")
    zero_path = directory / "post_teardown_global_provider_zero.json"
    if not zero_path.exists():
        write_exclusive(zero_path, zero)
    retained_zero = _json(zero_path)
    if (retained_zero.get("schema_version") != "gpu_spend_guard.v1" or retained_zero.get("provider_zero_verified") is not True
            or retained_zero.get("live_instance_count") != 0
            or retained_zero.get("receipt_digest") != canonical_digest(retained_zero, digest_field="receipt_digest")):
        raise ValueError("g1_team_policy_settlement_retained_zero_invalid")

    billing_path = directory / "official_billing.json"
    if not billing_path.exists():
        for source in _candidate_sources(billing_audit_root, adapter_path, adapter):
            try:
                materialize_vast_official_same_goal_reconciliation(provider_billing_source_receipt_path=source,
                    expected_instances=[(instance_id, label, terminal_path)], output_path=billing_path)
            except (OSError, VastOfficialBillingExtractionError):
                continue
            break
        if not billing_path.exists():
            return _pending("awaiting_posted_official_billing", identity)
    billing = validate_vast_official_same_goal_reconciliation(billing_path)
    entries = billing.get("entries")
    if (not isinstance(entries, list) or len(entries) != 1
            or entries[0].get("provider_instance_id") != instance_id or entries[0].get("launch_label") != label
            or (entries[0].get("terminal_execution_evidence") or {}).get("terminal_result", {}).get("path") != str(terminal_path)):
        raise ValueError("g1_team_policy_settlement_charge_identity_invalid")

    review_path = run / "native_g1_team_private_review.v1.json"
    if review_path.exists():
        if _json(review_path) != review:
            raise ValueError("g1_team_policy_settlement_review_changed")
    else:
        write_exclusive(review_path, dict(review))
    common = {"adapter_result_path": adapter_path, "bundle_receipt_path": bundle_path,
              "retained_review_path": review_path, "result_root": result_root, "run_id": request["run_id"]}
    result_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    delivery = materialize_g1_private_review_delivery(**common)  # Idempotent byte-verifying registry reopen.
    delivery_path = directory / "private_delivery.json"
    if delivery_path.exists():
        if _json(delivery_path) != delivery:
            raise ValueError("g1_team_policy_settlement_delivery_changed")
    else:
        write_exclusive(delivery_path, delivery)
    if delivery.get("artifact_count") != 3:
        raise ValueError("g1_team_policy_settlement_selected_artifact_count_invalid")
    common["delivery_receipt_path"] = delivery_path
    if _verified_review(**common) != review:
        raise ValueError("g1_team_policy_settlement_registered_review_changed")
    owner = request["owner"]
    ingest_path = directory / "private_ingest.json"
    if ingest_path.exists():
        ingested = _json(ingest_path)
    else:
        ingested = ingest_g1_private_review(**common, owner_user_id=owner["user_id"],
            organization_id=owner["organization_id"], webapp_url=webapp_url, sync_token=sync_token)
        write_exclusive(ingest_path, ingested)
    if (ingested.get("status") not in {"ingested", "already_ingested"}
            or ingested.get("run_id") != request["run_id"] or ingested.get("review_digest") != review["review_digest"]
            or ingested.get("owner_user_id") != owner["user_id"] or ingested.get("organization_id") != owner["organization_id"]
            or ingested.get("access_visibility") != "owner_only" or ingested.get("claim_ceiling") != "development_only"
            or ingested.get("public_redistribution_authorized") is not False):
        raise ValueError("g1_team_policy_settlement_ingest_invalid")
    url, origin = urllib.parse.urlsplit(str(ingested.get("review_url") or "")), urllib.parse.urlsplit(webapp_url)
    if (origin.scheme != "https" or origin.username or origin.password or url.scheme != "https"
            or url.netloc != origin.netloc or url.path != "/app/g1-reviews/" + urllib.parse.quote(request["run_id"], safe="")
            or url.query or url.fragment):
        raise ValueError("g1_team_policy_settlement_url_invalid")
    settled = {"schema_version": SCHEMA, "status": "delivered_owner_only", "intent_id": identity,
               "intent_digest": intent["intent_digest"], "start_digest": start["start_digest"],
               "dispatch_digest": final["dispatch_digest"], "run_id": request["run_id"],
               "execution_packet_digest": packet["packet_digest"], "provider_instance_id": instance_id,
               "review_digest": review["review_digest"], "official_billing_receipt_digest": billing["receipt_digest"],
               "official_total_usd": billing["official_total_usd"], "global_provider_zero_receipt_digest": retained_zero["receipt_digest"],
               "review_url": ingested["review_url"], "access_visibility": "owner_only", "claim_ceiling": "development_only",
               "public_redistribution_authorized": False, "provider_mutation_performed": False}
    settled["settlement_digest"] = digest(settled, digest_field="settlement_digest")
    settled_path = directory / "settlement.json"
    if settled_path.exists():
        if _sealed(settled_path, "settlement_digest") != settled:
            raise ValueError("g1_team_policy_settlement_existing_conflict")
    else:
        write_exclusive(settled_path, settled)
    return settled


def settle_pending_g1_team_policies(*, queue_root: Path, work_root: Path, **common) -> dict[str, Any]:
    queue, work = Path(queue_root), Path(work_root)
    _directory(queue, required=True)
    _directory(work)
    pending = []
    for path in sorted(queue.glob("g1-team-policy-*/intent.json")):
        _json(path)
        directory = work / path.parent.name
        _directory(directory)
        final_path = directory / "dispatch_final.json"
        if not final_path.exists():
            continue
        final = _sealed(final_path, "dispatch_digest")
        if final.get("status") != "controller_completed_pending_billing_and_private_delivery":
            continue
        if (directory / "settlement.json").exists():
            continue
        result = settle_g1_team_policy(intent_path=path, work_root=work, **common)
        if result["status"] == "delivered_owner_only":
            return result
        pending.append({"intent_id": path.parent.name, "status": result["status"]})
    return {"schema_version": SCHEMA, "status": "awaiting_settlement_evidence" if pending else "no_pending_settlement",
            "pending": pending, "provider_mutation_performed": False}
