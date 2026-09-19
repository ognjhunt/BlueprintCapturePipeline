"""Current owner task binding; no network, credentials, or model inference."""
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import website_task_context as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def context(confirmed=True):
    value = {"schema_version": "website_site_task_context.v1", "request_id": "req1",
             "scene_id": "site-req1", "capture_id": "walkthrough-req1",
             "description": "Move the carton onto the pallet", "confirmed": confirmed,
             "confirmed_at": "2026-09-19T01:00:00Z" if confirmed else None,
             "operator_answers": {}, "unresolved": []}
    value["context_digest"] = canonical_digest(value, digest_field="context_digest")
    return value


def validate(value):
    return module.validate_website_task_context(value, request_id="req1",
                                                scene_id="site-req1", capture_id="walkthrough-req1")


def test_task_is_bound_to_exact_capture_and_confirmed_content():
    assert validate(context())["confirmed"] is True
    with pytest.raises(ValueError, match="not_confirmed"):
        validate(context(False))
    with pytest.raises(ValueError, match="identity_mismatch"):
        validate({**context(), "capture_id": "walkthrough-other"})
    with pytest.raises(ValueError, match="digest_mismatch"):
        validate({**context(), "description": "Move the unrelated chair"})


def test_refresh_reads_current_confirmation_with_signed_bounded_request(monkeypatch):
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setattr(module, "load_pipeline_sync_token", lambda: "test-secret")
    calls = []

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(body=json.dumps(context()).encode())

    monkeypatch.setattr(module, "safe_request", fetch)
    value = module.load_current_website_task_context(request_id="req1", scene_id="site-req1", capture_id="walkthrough-req1")
    assert value["confirmed"] is True
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url.endswith("/api/internal/pipeline/creator-captures/walkthrough-req1/task-context")
    assert kwargs["headers"]["X-Blueprint-Pipeline-Signature"].startswith("sha256=")
    assert json.loads(kwargs["data"]) == {"request_id": "req1", "scene_id": "site-req1"}
    assert kwargs["timeout_seconds"] == 10


def test_missing_endpoint_cannot_reuse_stale_manifest_confirmation(monkeypatch):
    monkeypatch.delenv("PIPELINE_SYNC_WEBAPP_URL", raising=False)
    with pytest.raises(ValueError, match="webapp_url_missing"):
        module.load_current_website_task_context(request_id="req1", scene_id="site-req1", capture_id="walkthrough-req1")


def sponsorship():
    value = {"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint",
        "request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
        "task_context_digest": context()["context_digest"],
        "preparation_max_total_spend_usd": 25, "upstream_max_spend_usd": 5,
        "max_total_spend_usd": 20, "expires_at_epoch": 2000}
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    return value


def test_sponsor_is_separate_from_model_context_and_rejects_budget_or_capture_changes(monkeypatch):
    value = sponsorship()
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return value

    monkeypatch.setattr(module, "website_webapp_request", fetch)
    task = context()
    assert module.load_website_scene_sponsorship(task_context=task, now=1000) == value
    assert "owner" not in task and "sponsor" not in task
    assert calls[0]["operation"] == "scene-sponsorship"
    for changed in ({"capture_id": "walkthrough-other"}, {"task_context_digest": "sha256:" + "0" * 64},
                    {"upstream_max_spend_usd": 6}, {"expires_at_epoch": 1000},
                    {"max_total_spend_usd": True}, {"preparation_max_total_spend_usd": float("inf")}):
        value = {**sponsorship(), **changed}
        # A correctly hashed but semantically invalid response still fails.
        if changed.get("preparation_max_total_spend_usd") != float("inf"):
            value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
        with pytest.raises(ValueError):
            module.load_website_scene_sponsorship(task_context=task, now=1000)


def test_prepared_scene_enters_webapp_outbox_not_a_forged_local_owner_intent(monkeypatch):
    request = {"submission_id": "walkthrough-req1", "source": {"binding_id": "website-splat-test"}}
    calls = []
    value = {"id": "scene-test", "state": "forward_pending", "request_digest": canonical_digest(request)}

    def fetch(**kwargs):
        calls.append(kwargs)
        return value

    monkeypatch.setattr(module, "website_webapp_request", fetch)
    assert module.enqueue_website_prepared_scene(task_context=context(), request=request) == value
    assert calls[0]["operation"] == "prepared-scene"
    assert calls[0]["payload"]["request"] == request
    value["request_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="outbox_receipt_invalid"):
        module.enqueue_website_prepared_scene(task_context=context(), request=request)


def test_visual_scene_publishes_only_viewer_assets_and_checks_receipt(monkeypatch):
    value = context()
    descriptor = {"capture_id": value["capture_id"], "scene_id": value["scene_id"], "metadata": {
        "capture_entry_source": "browser_self_capture", "site_task_context": value,
        "clean_plate": {"privacy_verified": True}}}
    world = {"world_id": "world-1", "world_marble_url": "https://marble.worldlabs.ai/world/1",
             "assets": {"thumbnail_url": "https://cdn.example/thumb.png", "imagery": {"pano_url": "https://cdn.example/pano.jpg"},
                        "mesh": {"collider_mesh_url": "not-needed-for-first-view"}}}
    calls = []
    def post(**kwargs):
        calls.append(kwargs)
        return {"state": "ready", "world_id": "world-1", "task_context_digest": value["context_digest"]}
    monkeypatch.setattr(module, "website_webapp_request", post)
    assert module.publish_website_visual_scene(descriptor=descriptor, world=world, operation_id="op-1")["state"] == "ready"
    assert calls[0]["operation"] == "visual-scene"
    assert calls[0]["payload"]["thumbnail_url"] == "https://cdn.example/thumb.png"
    assert "collider" not in json.dumps(calls)
    descriptor["metadata"]["clean_plate"]["privacy_verified"] = False
    with pytest.raises(ValueError, match="preparation_missing"):
        module.publish_website_visual_scene(descriptor=descriptor, world=world, operation_id="op-1")
    assert len(calls) == 1


def test_sam_spend_uses_signed_current_capture_reservation(monkeypatch):
    from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission_grant
    calls = []
    def post(**kwargs):
        calls.append(kwargs)
        return {**kwargs["payload"]["spend"], "schema_version": "paid_lane_admission.v1", "status": "admitted",
                "blockers": [], "external_disclosure_allowed": True, "expires_at_epoch": 9_999_999_999}
    monkeypatch.setattr(module, "website_webapp_request", post)
    digest = "sha256:" + "a" * 64
    receipt, grant = module.reserve_website_sam_spend(task_context=context(), binding_digest=digest,
                                                     maximum_cost_usd=0.02, request_count=2)
    require_paid_resource_admission_grant(grant, resource_class="evaluator_api", allocation_binding_digest=digest,
                                          require_allocation_binding=True)
    assert calls[0]["operation"] == "preparation-spend"
    assert receipt["maximum_cost_usd"] == .02
    monkeypatch.setattr(module, "website_webapp_request", lambda **kw: {**receipt, "allocation_binding_digest": "sha256:" + "b" * 64})
    with pytest.raises(ValueError, match="receipt_invalid"):
        module.reserve_website_sam_spend(task_context=context(), binding_digest=digest, maximum_cost_usd=.02, request_count=2)
