"""Finish one paid G1 team campaign with billing and owner-only review.

This worker can be retried after a charge posts or another team's GPU closes.
It never starts a provider instance or repeats the paid allocator attempt.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest as digest,
)
from .native_g1_paid_campaign import INSTANCE_LABEL_PREFIX, verify_g1_paid_output
from .native_g1_private_review import project_g1_private_review
from .native_g1_private_review_delivery import materialize_g1_private_review_delivery
from .native_g1_private_review_ingest import ingest_g1_private_review
from .native_g1_provider_bundle import load_verified_g1_provider_bundle
from .native_g1_team_campaign_dispatcher import FINAL_SCHEMA, START_SCHEMA
from .native_g1_team_campaign_intake import INTENT_SCHEMA, _read
from .native_g1_team_campaign_preparation import SCHEMA as PREPARATION_SCHEMA
from .policy_canary_billing_recovery import _candidate_sources
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)
from .vast_official_billing_extractor import (
    VastOfficialBillingExtractionError,
    materialize_vast_official_same_goal_reconciliation,
    validate_vast_official_same_goal_reconciliation,
)


SCHEMA = "native_g1_team_campaign_settlement.v1"
ZeroCollector = Callable[[], dict[str, Any]]
BillingMaterializer = Callable[..., dict[str, Any]]


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("g1_team_settlement_input_missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_team_settlement_input_invalid")
    return value


def _instance_and_label(adapter: dict[str, Any]) -> tuple[int, str]:
    ids = adapter.get("vast_instance_ids")
    watchdog = adapter.get("independent_watchdog")
    if ids is None and isinstance(watchdog, dict):
        ids = watchdog.get("instance_ids")
    if (
        not isinstance(ids, list) or len(ids) != 1
        or type(ids[0]) is not int or ids[0] <= 0
    ):
        raise ValueError("g1_team_settlement_instance_identity_invalid")
    attempt = Path(str(adapter.get("attempt_root") or ""))
    if not attempt.is_absolute() or attempt.is_symlink():
        raise ValueError("g1_team_settlement_attempt_root_invalid")
    startup = _strict_json(
        attempt / "vast_provider_run/vast_startup_probe_manifest.json"
    )
    created = startup.get("create_request_summary")
    observed = startup.get("last_instance_payload")
    instance = observed.get("instances") if isinstance(observed, dict) else None
    label = created.get("label") if isinstance(created, dict) else None
    if (
        startup.get("schema_version") != "vast_startup_probe_manifest.v1"
        or startup.get("status") != "completed"
        or not isinstance(instance, dict)
        or instance.get("id") != ids[0]
        or instance.get("label") != label
        or not isinstance(label, str)
        or not label.startswith(INSTANCE_LABEL_PREFIX)
    ):
        raise ValueError("g1_team_settlement_launch_identity_invalid")
    return ids[0], label


def settle_g1_team_campaign(
    *, intent_path: Path, work_root: Path, billing_audit_root: Path,
    result_root: Path, webapp_url: str, sync_token: str,
    collect_zero: ZeroCollector | None = None,
    materialize_billing: BillingMaterializer = materialize_vast_official_same_goal_reconciliation,
) -> dict[str, Any]:
    """Resume verified post-run steps without creating or retaining a GPU."""

    from .adp009d_provider_zero import collect_provider_zero_receipt

    intent = _read(Path(intent_path), field="intent_digest")
    if intent.get("schema_version") != INTENT_SCHEMA:
        raise ValueError("g1_team_settlement_intent_invalid")
    directory = Path(work_root) / str(intent.get("intent_id"))
    if (
        not Path(work_root).is_absolute() or Path(work_root).is_symlink()
        or directory.is_symlink() or not directory.is_dir()
        or not Path(billing_audit_root).is_absolute()
        or Path(billing_audit_root).is_symlink()
        or not Path(result_root).is_absolute() or Path(result_root).is_symlink()
    ):
        raise ValueError("g1_team_settlement_paths_invalid")
    settled_path = directory / "settlement.json"
    if settled_path.is_file():
        settled = _read(settled_path, field="settlement_digest")
        if (
            settled.get("schema_version") != SCHEMA
            or settled.get("status") != "delivered_owner_only"
            or settled.get("intent_digest") != intent["intent_digest"]
            or settled.get("claim_ceiling") != "development_only"
            or settled.get("access_visibility") != "owner_only"
            or settled.get("public_redistribution_authorized") is not False
        ):
            raise ValueError("g1_team_settlement_existing_conflict")
        return settled
    final = _read(directory / "dispatch_final.json", field="dispatch_digest")
    start = _read(directory / "execution_started.json", field="start_digest")
    prepared = _read(directory / "preparation.json", field="preparation_digest")
    if (
        final.get("schema_version") != FINAL_SCHEMA
        or final.get("status") != "controller_completed_pending_billing_and_private_delivery"
        or final.get("intent_digest") != intent["intent_digest"]
        or final.get("start_digest") != start.get("start_digest")
        or final.get("four_episodes_verified") is not True
        or final.get("run_teardown_confirmed_by_adapter") is not True
        or start.get("schema_version") != START_SCHEMA
        or start.get("intent_digest") != intent["intent_digest"]
        or start.get("preparation_digest") != prepared.get("preparation_digest")
        or prepared.get("schema_version") != PREPARATION_SCHEMA
        or prepared.get("intent_digest") != intent["intent_digest"]
    ):
        raise ValueError("g1_team_settlement_controller_incomplete")
    bundle_path = Path(prepared["bundle_receipt_path"])
    if bundle_path != directory / "bundle/native_g1_provider_bundle.v1.json":
        raise ValueError("g1_team_settlement_bundle_path_changed")
    bundle = load_verified_g1_provider_bundle(
        bundle_path, expected_implementation_commit=prepared["implementation_commit"],
    )
    run_root = directory / "run"
    adapter_path = run_root / "adapter_paid.json"
    adapter = _strict_json(adapter_path)
    verification = verify_g1_paid_output(adapter, bundle)
    if adapter.get("g1_output_verification") != verification:
        raise ValueError("g1_team_settlement_output_verification_changed")
    review = project_g1_private_review(verification=verification, bundle=bundle)
    review_path = run_root / "native_g1_private_review.v1.json"
    if (
        (adapter.get("g1_private_review") or {}).get("path") != str(review_path)
        or _strict_json(review_path) != review
    ):
        raise ValueError("g1_team_settlement_private_review_changed")
    instance_id, launch_label = _instance_and_label(adapter)

    zero_path = directory / "post_teardown_global_provider_zero.json"
    if zero_path.is_file():
        zero = _strict_json(zero_path)
    else:
        zero = (collect_zero or collect_provider_zero_receipt)()
        if zero.get("provider_zero_verified") is not True:
            return {"schema_version": SCHEMA, "status": "awaiting_global_provider_zero",
                    "intent_id": intent["intent_id"], "provider_mutation_performed": False,
                    "blockers": zero.get("blockers") or ["global_provider_zero_unproven"]}
        write_exclusive(zero_path, zero)
    if (
        zero.get("schema_version") != "gpu_spend_guard.v1"
        or zero.get("receipt_digest")
        != canonical_digest(zero, digest_field="receipt_digest")
        or zero.get("provider_zero_verified") is not True
        or zero.get("live_instance_count") != 0
    ):
        raise ValueError("g1_team_settlement_provider_zero_invalid")

    billing_path = directory / "official_billing.json"
    if billing_path.is_file():
        billing = validate_vast_official_same_goal_reconciliation(billing_path)
    else:
        billing = None
        for source in _candidate_sources(Path(billing_audit_root), adapter_path, adapter):
            try:
                billing = materialize_billing(
                    provider_billing_source_receipt_path=source,
                    expected_instances=[(instance_id, launch_label,
                                         run_root / "adp_arena_vast_result.json")],
                    output_path=billing_path,
                )
            except (OSError, VastOfficialBillingExtractionError):
                continue
            break
        if billing is None:
            return {"schema_version": SCHEMA, "status": "awaiting_posted_official_billing",
                    "intent_id": intent["intent_id"], "provider_mutation_performed": False}
        billing = validate_vast_official_same_goal_reconciliation(billing_path)
    entries = billing.get("entries")
    if (
        not isinstance(entries, list) or len(entries) != 1
        or entries[0].get("provider_instance_id") != instance_id
        or entries[0].get("launch_label") != launch_label
    ):
        raise ValueError("g1_team_settlement_official_billing_identity_invalid")

    delivery_path = directory / "private_delivery.json"
    run_id = intent["request"]["run_id"]
    if delivery_path.is_file():
        delivery = _strict_json(delivery_path)
    else:
        delivery = materialize_g1_private_review_delivery(
            adapter_result_path=adapter_path, bundle_receipt_path=bundle_path,
            retained_review_path=review_path, result_root=Path(result_root),
            run_id=run_id,
        )
        write_exclusive(delivery_path, delivery)
    if (
        delivery.get("status") != "registered_private_development_review"
        or delivery.get("run_id") != run_id
        or delivery.get("review_digest") != review["review_digest"]
        or delivery.get("artifact_count") != 12
    ):
        raise ValueError("g1_team_settlement_delivery_invalid")
    ingest_path = directory / "private_ingest.json"
    owner = intent["request"]["owner"]
    if ingest_path.is_file():
        ingested = _strict_json(ingest_path)
    else:
        ingested = ingest_g1_private_review(
            adapter_result_path=adapter_path, bundle_receipt_path=bundle_path,
            retained_review_path=review_path, delivery_receipt_path=delivery_path,
            result_root=Path(result_root), run_id=run_id,
            owner_user_id=owner["user_id"],
            organization_id=owner["organization_id"],
            webapp_url=webapp_url, sync_token=sync_token,
        )
        write_exclusive(ingest_path, ingested)
    if (
        ingested.get("status") not in {"ingested", "already_ingested"}
        or ingested.get("run_id") != run_id
        or ingested.get("review_digest") != review["review_digest"]
        or ingested.get("owner_user_id") != owner["user_id"]
        or ingested.get("organization_id") != owner["organization_id"]
        or ingested.get("access_visibility") != "owner_only"
        or ingested.get("claim_ceiling") != "development_only"
        or ingested.get("public_redistribution_authorized") is not False
    ):
        raise ValueError("g1_team_settlement_ingest_invalid")
    review_url = urllib.parse.urlsplit(str(ingested.get("review_url") or ""))
    webapp = urllib.parse.urlsplit(webapp_url)
    if (
        review_url.scheme != "https"
        or review_url.netloc != webapp.netloc
        or review_url.path != "/app/g1-reviews/" + urllib.parse.quote(run_id, safe="")
        or review_url.query or review_url.fragment
    ):
        raise ValueError("g1_team_settlement_review_url_invalid")
    settled = {
        "schema_version": SCHEMA,
        "status": "delivered_owner_only",
        "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"],
        "run_id": run_id,
        "provider_instance_id": instance_id,
        "review_digest": review["review_digest"],
        "official_billing_receipt_digest": billing["receipt_digest"],
        "official_total_usd": billing["official_total_usd"],
        "global_provider_zero_receipt_digest": zero["receipt_digest"],
        "review_url": ingested["review_url"],
        "access_visibility": "owner_only",
        "claim_ceiling": "development_only",
        "public_redistribution_authorized": False,
        "provider_mutation_performed": False,
    }
    settled["settlement_digest"] = digest(settled, digest_field="settlement_digest")
    write_exclusive(settled_path, settled)
    return settled


def settle_pending_g1_team_campaigns(
    *, queue_root: Path, work_root: Path, billing_audit_root: Path,
    result_root: Path, webapp_url: str, sync_token: str,
) -> dict[str, Any]:
    """Advance one complete review; skip charges and zero checks still pending."""

    queue = Path(queue_root)
    work = Path(work_root)
    if (
        not queue.is_absolute() or queue.is_symlink() or not queue.is_dir()
        or not work.is_absolute() or work.is_symlink()
    ):
        raise ValueError("g1_team_settlement_queue_invalid")
    pending: list[dict[str, str]] = []
    for intent_path in sorted(queue.glob("g1-*/intent.json")):
        if intent_path.parent.is_symlink() or intent_path.is_symlink():
            raise ValueError("g1_team_settlement_intent_path_unsafe")
        directory = work / intent_path.parent.name
        final_path = directory / "dispatch_final.json"
        if not final_path.is_file():
            continue
        final = _read(final_path, field="dispatch_digest")
        if final.get("status") != "controller_completed_pending_billing_and_private_delivery":
            continue
        if (directory / "settlement.json").is_file():
            continue
        result = settle_g1_team_campaign(
            intent_path=intent_path, work_root=work,
            billing_audit_root=billing_audit_root, result_root=result_root,
            webapp_url=webapp_url, sync_token=sync_token,
        )
        if result["status"] == "delivered_owner_only":
            return result
        pending.append({"intent_id": intent_path.parent.name, "status": result["status"]})
    return {
        "schema_version": SCHEMA,
        "status": "awaiting_settlement_evidence" if pending else "no_pending_settlement",
        "pending": pending,
        "provider_mutation_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--intent-path", type=Path)
    source.add_argument("--queue-root", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--billing-audit-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--webapp-url", default=os.getenv("PIPELINE_SYNC_WEBAPP_URL", ""))
    args = parser.parse_args(argv)
    common = {
        "work_root": args.work_root,
        "billing_audit_root": args.billing_audit_root,
        "result_root": args.result_root,
        "webapp_url": args.webapp_url,
        "sync_token": os.getenv("PIPELINE_SYNC_TOKEN", ""),
    }
    if args.queue_root:
        result = settle_pending_g1_team_campaigns(queue_root=args.queue_root, **common)
    else:
        result = settle_g1_team_campaign(intent_path=args.intent_path, **common)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {
        "delivered_owner_only", "no_pending_settlement", "awaiting_settlement_evidence",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
