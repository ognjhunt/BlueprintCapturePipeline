"""Settlement holds private delivery behind exact output, billing and zero proof."""

from __future__ import annotations

import json

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.native_g1_team_campaign_dispatcher import FINAL_SCHEMA, START_SCHEMA
from blueprint_pipeline.native_g1_team_campaign_intake import INTENT_SCHEMA
from blueprint_pipeline.native_g1_team_campaign_preparation import SCHEMA as PREPARATION_SCHEMA
from blueprint_pipeline.native_g1_team_campaign_settlement import (
    settle_g1_team_campaign,
    settle_pending_g1_team_campaigns,
)


def _sealed(path, value, field):
    value[field] = cross_runtime_canonical_digest(value, digest_field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value


def _case(tmp_path, monkeypatch):
    intent_id = "g1-" + "a" * 64
    queue = tmp_path / "queue" / intent_id / "intent.json"
    work = tmp_path / "work"
    directory = work / intent_id
    run = directory / "run"
    run.mkdir(parents=True)
    bundle_path = directory / "bundle/native_g1_provider_bundle.v1.json"
    bundle_path.parent.mkdir()
    bundle_path.write_text("{}")
    intent = _sealed(queue, {
        "schema_version": INTENT_SCHEMA, "intent_id": intent_id,
        "request": {"run_id": "g1-review-one", "owner": {
            "user_id": "owner", "organization_id": "team",
        }},
    }, "intent_digest")
    prepared = _sealed(directory / "preparation.json", {
        "schema_version": PREPARATION_SCHEMA,
        "intent_digest": intent["intent_digest"],
        "bundle_receipt_path": str(bundle_path),
        "implementation_commit": "a" * 40,
    }, "preparation_digest")
    started = _sealed(directory / "execution_started.json", {
        "schema_version": START_SCHEMA,
        "intent_digest": intent["intent_digest"],
        "preparation_digest": prepared["preparation_digest"],
    }, "start_digest")
    _sealed(directory / "dispatch_final.json", {
        "schema_version": FINAL_SCHEMA,
        "status": "controller_completed_pending_billing_and_private_delivery",
        "intent_digest": intent["intent_digest"],
        "start_digest": started["start_digest"],
        "four_episodes_verified": True,
        "run_teardown_confirmed_by_adapter": True,
    }, "dispatch_digest")
    adapter = {"status": "completed", "g1_output_verification": {"episodes": [1, 2, 3, 4]},
               "g1_private_review": {"path": str(run / "native_g1_private_review.v1.json")}}
    (run / "adapter_paid.json").write_text(json.dumps(adapter))
    review = {"review_digest": "sha256:" + "b" * 64}
    (run / "native_g1_private_review.v1.json").write_text(json.dumps(review))
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement.load_verified_g1_provider_bundle",
                        lambda *_args, **_kwargs: {"bundle_sha256": "sha256:" + "c" * 64})
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement.verify_g1_paid_output",
                        lambda *_args: {"episodes": [1, 2, 3, 4]})
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement.project_g1_private_review",
                        lambda **_kwargs: review)
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement._instance_and_label",
                        lambda _adapter: (123, "blueprint-native-task-arena-g1-841757-one"))
    return {
        "intent_path": queue, "work_root": work,
        "billing_audit_root": tmp_path / "audit", "result_root": tmp_path / "results",
        "webapp_url": "https://tryblueprint.io", "sync_token": "secret",
    }, directory


def test_private_delivery_waits_for_fresh_global_provider_zero(tmp_path, monkeypatch):
    kwargs, directory = _case(tmp_path, monkeypatch)
    result = settle_g1_team_campaign(**kwargs, collect_zero=lambda: {
        "provider_zero_verified": False, "blockers": ["another_gpu_active"],
    })
    assert result["status"] == "awaiting_global_provider_zero"
    assert result["blockers"] == ["another_gpu_active"]
    assert not (directory / "post_teardown_global_provider_zero.json").exists()
    assert not (directory / "private_delivery.json").exists()


def test_verified_charge_and_zero_allow_owner_only_ingest_once(tmp_path, monkeypatch):
    kwargs, directory = _case(tmp_path, monkeypatch)
    zero = {"schema_version": "gpu_spend_guard.v1", "provider_zero_verified": True,
            "live_instance_count": 0}
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    source = kwargs["billing_audit_root"] / "provider_billing_source_receipt.json"
    source.parent.mkdir()
    source.write_text("{}")
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement._candidate_sources",
                        lambda *_args: [source])
    billing = {"receipt_digest": "sha256:" + "d" * 64, "official_total_usd": 0.5,
               "entries": [{"provider_instance_id": 123,
                            "launch_label": "blueprint-native-task-arena-g1-841757-one"}]}
    def materialize_billing(**args):
        assert args["expected_instances"] == [
            (123, "blueprint-native-task-arena-g1-841757-one",
             directory / "run/adp_arena_vast_result.json")]
        args["output_path"].write_text(json.dumps(billing))
        return billing
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement.validate_vast_official_same_goal_reconciliation",
                        lambda _path: billing)
    calls = []
    def deliver(**args):
        calls.append("deliver")
        return {"status": "registered_private_development_review",
                "run_id": args["run_id"], "review_digest": "sha256:" + "b" * 64,
                "artifact_count": 12}
    def ingest(**args):
        calls.append("ingest")
        return {"status": "ingested", "run_id": args["run_id"],
                "owner_user_id": args["owner_user_id"],
                "organization_id": args["organization_id"],
                "claim_ceiling": "development_only",
                "review_digest": "sha256:" + "b" * 64,
                "review_url": "https://tryblueprint.io/app/g1-reviews/g1-review-one",
                "access_visibility": "owner_only", "public_redistribution_authorized": False}
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement.materialize_g1_private_review_delivery", deliver)
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement.ingest_g1_private_review", ingest)
    result = settle_g1_team_campaign(**kwargs, collect_zero=lambda: zero,
                                     materialize_billing=materialize_billing)
    assert result["status"] == "delivered_owner_only"
    assert result["official_total_usd"] == 0.5
    assert result["public_redistribution_authorized"] is False
    assert calls == ["deliver", "ingest"]
    assert settle_g1_team_campaign(**kwargs, collect_zero=lambda: None)["settlement_digest"] == result["settlement_digest"]
    assert calls == ["deliver", "ingest"]


def test_queue_can_deliver_later_run_while_earlier_charge_is_pending(tmp_path, monkeypatch):
    kwargs, directory = _case(tmp_path, monkeypatch)
    second_id = "g1-" + "b" * 64
    second_intent = kwargs["intent_path"].parent.parent / second_id / "intent.json"
    second_intent.parent.mkdir()
    second_intent.write_text("{}")
    second_final = kwargs["work_root"] / second_id / "dispatch_final.json"
    _sealed(second_final, {
        "status": "controller_completed_pending_billing_and_private_delivery",
    }, "dispatch_digest")
    seen = []
    def settle(**args):
        seen.append(args["intent_path"].parent.name)
        return {"status": ("awaiting_posted_official_billing" if len(seen) == 1
                           else "delivered_owner_only")}
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_campaign_settlement.settle_g1_team_campaign", settle)
    result = settle_pending_g1_team_campaigns(
        queue_root=kwargs["intent_path"].parent.parent,
        work_root=kwargs["work_root"],
        billing_audit_root=kwargs["billing_audit_root"],
        result_root=kwargs["result_root"],
        webapp_url=kwargs["webapp_url"], sync_token=kwargs["sync_token"],
    )
    assert result["status"] == "delivered_owner_only"
    assert seen == [directory.name, second_id]
