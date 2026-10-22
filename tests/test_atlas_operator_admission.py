"""Signed IPC receipts cannot substitute another source, provider, or budget."""
import hashlib
import hmac
import json
import time

import pytest

from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


def test_exact_signed_operator_reservation_and_tamper_expiry_provider_guards(tmp_path, monkeypatch):
    key = "test-sync-token-not-secret"
    monkeypatch.setattr(control, "load_pipeline_sync_token", lambda: key)
    context = {"request_id": "r", "capture_id": "c"}
    command = {"provider": "world_labs_atlas", "resource_class": "provider_reconstruction_api",
               "source_video_digest": "sha256:" + "a" * 64, "maximum_cost_usd": .05}
    payload = {"schema_version": "website_atlas_operator_admission.v1", **context,
               "issued_at_epoch": time.time(), "expires_at_epoch": time.time() + 600,
               "admission": {**command, "status": "admitted"}}
    # Use one instant so the signed validity interval is at most 600 seconds.
    payload["expires_at_epoch"] = payload["issued_at_epoch"] + 600
    signature = hmac.new(key.encode(), cross_runtime_canonical_digest(payload).encode(), hashlib.sha256).hexdigest()
    path = tmp_path / "admission.json"
    path.write_text(json.dumps({**payload, "signature": signature}))
    assert control.load_atlas_operator_admission(path, command=command, task_context=context) == payload["admission"]
    with pytest.raises(ValueError, match="admission_invalid"):
        control.load_atlas_operator_admission(path, command={**command, "maximum_cost_usd": 5}, task_context=context)
    with pytest.raises(ValueError, match="provider_invalid"):
        control.load_atlas_operator_admission(path, command={**command, "provider": "vast"}, task_context=context)
    payload["admission"]["maximum_cost_usd"] = 5
    path.write_text(json.dumps({**payload, "signature": signature}))
    with pytest.raises(ValueError, match="admission_invalid"):
        control.load_atlas_operator_admission(path, command=command, task_context=context)
    payload["expires_at_epoch"] = time.time() - 1
    signature = hmac.new(key.encode(), cross_runtime_canonical_digest(payload).encode(), hashlib.sha256).hexdigest()
    path.write_text(json.dumps({**payload, "signature": signature}))
    with pytest.raises(ValueError, match="admission_invalid"):
        control.load_atlas_operator_admission(path, command=command, task_context=context)
