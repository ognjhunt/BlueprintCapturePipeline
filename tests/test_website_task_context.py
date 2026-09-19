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
