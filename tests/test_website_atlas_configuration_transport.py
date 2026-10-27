"""Maximum-row accounting fits one finite, noncreating configuration read."""

import json
from io import BytesIO

import pytest

from blueprint_pipeline import safe_outbound_http as http
from blueprint_pipeline import website_atlas_pipeline as pipeline
from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.selected_atlas_preview_recovery import _accounting, _capture_inference_accounting


def _max_rows_configuration():
    """Exercise both row ceilings and practical bounded authority text together.

    Runtime policy prose has no schema maximum; arbitrarily larger prose still
    has to fail the transport's finite cap. This is a maximum-row fixture.
    """
    sha = "sha256:" + "a" * 64
    rows = [{"allocation_binding_digest": "sha256:" + f"{index:064x}",
             "provider": "world_labs_atlas" if index < 64 else "world_labs",
             "maximum_cost_usd": 1000, "settled_cost_usd": 999.123456,
             "accounted_cost_usd": 999.123456, "request_count": 32, "status": "settled",
             "task_context_digest": sha, "source_video_digest": sha,
             "atlas_authorization_digest": sha if index < 64 else None, "settlement_digest": sha}
            for index in range(128)]

    def ledger(native):
        selected = rows[64:] if native else rows[:64]
        return {"schema_version": "website_native_preparation_accounting.v1" if native
                    else "website_atlas_allowance_accounting.v1",
                "source_scope": "request_native_preparation_ledger" if native else "request_atlas_allowance",
                "settled_cost_usd": round(999.123456 * 64, 6), "held_cost_usd": 0,
                "reserved_or_settled_cost_usd": round(999.123456 * 64, 6), "request_count": 32 * 64,
                "reservations": selected, "external_capture_assessment_costs": "not_in_preparation_ledger"}

    disclosure = {"provider_terms_reference": sha, "private_processing_authorized": True,
                  "provider_training_authorized": False, "approval_reference": "a" * 1000}
    authority = {"schema_version": "website_atlas_authorization.v2", "request_id": "req1",
        "capture_id": "walkthrough-req1", "source_video_digest": sha,
        "source_video_identity": {"kind": "website_browser_video", "object_name": "r" * 1024,
            "generation": "1" * 20, "size_bytes": (1 << 53) - 1, "crc32c": "AAAAAA=="},
        "capture_rights_digest": sha, "policy_digest": sha, "provider_terms_reference": sha,
        "planner": {"provider": "openai", **disclosure}, "auxiliary": {"meta": disclosure, "google": disclosure},
        "accepted_by": "o" * 200, "approval_reference": "a" * 1000,
        "grant_basis": "automatic_consented_preview", "private_processing_authorized": True,
        "provider_training_authorized": False, "provider_retention": "account_does_not_retain",
        "maximum_cost_usd": 100, "maximum_requests": 32, "created_at_epoch": 1_792_000_000,
        "expires_at_epoch": 1_793_000_000, "claim_scope": "development_only", "authorization_digest": sha}
    calls = [{"call_digest": "sha256:" + f"{index:064x}", "provider": "gemini", "model": "gemini-3.8-flash",
              "state": "recorded", "source_digest": sha, "context_digest": sha, "run_digest": sha,
              "recorded_cost_usd": 0.123456, "bounded_unknown_cost_usd": None, "usage_digest": sha}
             for index in range(64)]
    inference = {"schema_version": "website_capture_inference_accounting.v1", "request_id": "req1",
        "capture_id": "walkthrough-req1", "budget_digest": sha, "source_video_digest": sha,
        "ledger_status": "complete", "known_usage_cost_usd": round(0.123456 * 64, 6),
        "bounded_unknown_cost_usd": 0, "unbounded_unknown_call_count": 0, "calls": calls,
        "linked_assessment": {"status": "verified", "job_digest": sha, "run_digest": sha,
            "known_usage_cost_usd": round(0.123456 * 64, 6), "ledger_overlap": "matched_call_ids"},
        "privacy": {"status": "unknown", "historical_cost_status": "unknown"},
        "preparation_ledger_overlap": "not_established"}
    return {"schema_version": "website_atlas_configuration.v1", "authority": authority,
        "policy": {"schema_version": "website_atlas_runtime_policy.v1", "model": "gpt-6.1-sol",
            "maximum_review_cost_usd": 5, "pose_input_cost_usd": 0.1, "rgbd_view_cost_usd": 0.1,
            "mesh_export_cost_usd": 1, "rate_card_reference": "https://worldlabs.ai/" + "r" * 1000,
            "reviewed_by": "r" * 200, "reviewed_on": "2026-10-27"},
        "reserved_or_settled_cost_usd": round(999.123456 * 64, 6), "remaining_cost_usd": 0,
        "remaining_requests": 0, "task_context_digest": sha, "source_video_digest": sha,
        "dispatch_authorized": False, "grant_creation_allowed": False,
        "allowance_accounting": ledger(False), "native_preparation_accounting": ledger(True),
        "capture_inference_accounting": inference}


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://tryblueprint.io/api/internal/pipeline/sync")
    monkeypatch.setattr(control, "load_pipeline_sync_token", lambda: "test-secret")
    state = {"body": b"{}", "calls": []}
    real_request = control.safe_request

    def fetch(url, **kwargs):
        state["calls"].append((url, kwargs))
        return real_request(url, **kwargs)

    class Response(BytesIO):
        status = 200

        def geturl(self):
            return None

    monkeypatch.setattr(control, "safe_request", fetch)
    monkeypatch.setattr(http, "_open_with_policy", lambda *args: Response(state["body"]))
    return state


def test_maximum_combined_row_counts_pass_real_capped_noncreating_transport(transport):
    value = _max_rows_configuration()
    assert len(value["allowance_accounting"]["reservations"]) + len(value["native_preparation_accounting"]["reservations"]) == 128
    assert len(value["capture_inference_accounting"]["calls"]) == 64
    _accounting(value["allowance_accounting"])
    _accounting(value["native_preparation_accounting"], native=True)
    _capture_inference_accounting(value["capture_inference_accounting"], request_id="req1",
        capture_id="walkthrough-req1", source_video_digest=value["source_video_digest"])
    transport["body"] = json.dumps(value, separators=(",", ":")).encode()
    assert 100_000 < len(transport["body"]) < 200_000
    context = {"request_id": "req1", "scene_id": "site-req1", "capture_id": "walkthrough-req1",
               "context_digest": value["task_context_digest"]}
    assert pipeline.load_configuration(context, value["source_video_digest"], create=False) == value
    _, kwargs = transport["calls"][-1]
    assert kwargs["max_response_bytes"] == kwargs["policy"].max_response_bytes == 200_000
    assert kwargs["policy"].allowed_hosts == frozenset({"tryblueprint.io"})
    assert kwargs["timeout_seconds"] == 10
    # The ordinary, potentially creating configuration caller keeps its old cap.
    with pytest.raises(http.SafeOutboundHttpError, match="outbound_http_response_too_large"):
        pipeline.load_configuration(context, value["source_video_digest"])


@pytest.mark.parametrize("operation,payload", [
    ("atlas-configuration", {}), ("atlas-configuration", {"create": True}),
    ("atlas-configuration", {"create": 0}), ("atlas-configuration", {"create": "false"}),
    ("atlas-configuration", {"create": None}), ("scene-sponsorship", {"create": False}),
    ("task-context", {"create": False}), ("preparation-spend", {"create": False}),
    ("atlas-preview", {"create": False}),
    ("task-context", {"max_response_bytes": 1_000_000}),
])
def test_other_calls_and_falsy_or_caller_limits_cannot_select_larger_response(transport, operation, payload):
    transport["body"] = b" " * 100_001
    with pytest.raises(http.SafeOutboundHttpError, match="outbound_http_response_too_large"):
        control.website_webapp_request(capture_id="walkthrough-req1", operation=operation, payload=payload)
    _, kwargs = transport["calls"][-1]
    assert kwargs["max_response_bytes"] == kwargs["policy"].max_response_bytes == 100_000


def test_noncreating_configuration_still_refuses_one_byte_over_its_fixed_cap(transport):
    transport["body"] = b" " * 200_001
    with pytest.raises(http.SafeOutboundHttpError, match="outbound_http_response_too_large:200001>200000"):
        control.website_webapp_request(capture_id="walkthrough-req1", operation="atlas-configuration",
            payload={"create": False, "max_response_bytes": 1_000_000})
    _, kwargs = transport["calls"][-1]
    assert kwargs["max_response_bytes"] == kwargs["policy"].max_response_bytes == 200_000
