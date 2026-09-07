"""Remaining handoff gaps: no missing-period bypass, exact recovery, terminal idempotence."""

import json
from pathlib import Path

import pytest

from blueprint_pipeline import paid_lane_guard as guard
from blueprint_pipeline import vast_official_billing_extractor as billing
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_offline_lifecycle_fault_matrix import proof
from tests.test_vast_official_billing_extractor import _fixture, INSTANCE_A, LABEL_A


def test_missing_billing_window_cannot_issue_new_charge_binding(tmp_path):
    fixture = _fixture(tmp_path)
    receipt = json.loads(fixture["receipt"].read_text())
    receipt.pop("cohort_start_at")
    receipt.pop("cohort_end_at")
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    fixture["receipt"].write_text(json.dumps(receipt))
    before = fixture["receipt"].read_bytes()
    with pytest.raises(billing.VastOfficialBillingExtractionError, match="period"):
        billing.extract_vast_official_instance_charge(
            provider_billing_source_receipt_path=fixture["receipt"],
            instance_id=INSTANCE_A,
            launch_label=LABEL_A,
        )
    assert fixture["receipt"].read_bytes() == before


def test_terminal_ledger_updates_are_idempotent_and_cannot_reopen(tmp_path):
    record = guard.open_pending_teardown(
        provider="vast", lane="offline", run_id="one", instance_id="owned", registry_dir=tmp_path
    )
    path = Path(record["path"])
    guard.close_pending_teardown(path, proof())
    original = path.read_bytes()
    guard.close_pending_teardown(path, proof())
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="terminal"):
        guard.mark_pending_teardown_ambiguous(path, reason="late_duplicate")
    assert path.read_bytes() == original


def test_binding_missing_record_cannot_invent_a_new_obligation(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="record"):
        guard.bind_pending_teardown_instance(path, "owned")
    assert not path.exists()


@pytest.mark.parametrize(
    "inventory_fault", [None, "duplicate", "wrong_name", "stale", "malformed", "wrong_provider"]
)
def test_lost_create_identity_recovered_only_from_exact_fresh_inventory(
    tmp_path, monkeypatch, inventory_fault
):
    monkeypatch.setattr(guard.time, "time", lambda: 100.0)
    record = guard.open_pending_teardown(
        provider="vast",
        lane="offline",
        run_id="frozen-one",
        resource_name="blueprint-offline-frozen-one",
        registry_dir=tmp_path,
        max_age_seconds=10,
    )
    guard.mark_pending_teardown_ambiguous(record["path"], reason="lost_create_ack")
    calls = []
    rows = [
        {"instance_id": "123", "name": "blueprint-offline-frozen-one"},
        {"instance_id": "456", "name": "other-task"},
    ]
    if inventory_fault == "duplicate":
        rows.append({"instance_id": "789", "name": "blueprint-offline-frozen-one"})
    if inventory_fault == "wrong_name":
        rows[0]["name"] = "blueprint-offline-frozen-one-other"

    class Provider:
        def billable_inventory(self, *, name_prefix):
            assert name_prefix == ""
            return {
                "status": "observed",
                "provider": "runpod" if inventory_fault == "wrong_provider" else "vast",
                "api_confirmed": True,
                "observed_at_epoch": 100.0 if inventory_fault == "stale" else 200.0,
                "live_resource_count": len(rows),
                "resources": None if inventory_fault == "malformed" else rows,
            }

        def inspect(self, instance):
            calls.append(("inspect", instance))
            return (
                {"http": 404}
                if any(c[0] == "terminate" for c in calls)
                else {"http": 200, "status": "observed", "desiredStatus": "running"}
            )

        def terminate(self, instance):
            calls.append(("terminate", instance))
            raise TimeoutError("lost delete acknowledgement after success")

    result = guard.reap_orphans(
        registry_dir=tmp_path, provider_clients={"vast": Provider()}, now_epoch=200.0
    )
    retained = json.loads(Path(record["path"]).read_text())
    if inventory_fault is None:
        assert retained["instance_id"] == "123"
        assert retained["status"] == "closed"
        assert retained["identity_recovery"]["resource_name"] == "blueprint-offline-frozen-one"
        assert retained["identity_recovery"]["inventory_digest"] == canonical_digest(
            retained["identity_recovery"]["inventory"]
        )
        assert calls == [("inspect", "123"), ("terminate", "123"), ("inspect", "123")]
        before = Path(record["path"]).read_bytes()
        guard.reap_orphans(
            registry_dir=tmp_path, provider_clients={"vast": Provider()}, now_epoch=201.0
        )
        assert Path(record["path"]).read_bytes() == before
    else:
        assert retained["status"] == "open" and retained["instance_id"] is None
        assert calls == []
    assert result


def test_failed_reconciliation_does_not_cancel_other_obligations(tmp_path, monkeypatch):
    monkeypatch.setattr(guard.time, "time", lambda: 100.0)
    bad = guard.open_pending_teardown(
        provider="vast",
        lane="offline",
        run_id="bad",
        instance_id="bad",
        registry_dir=tmp_path,
        max_age_seconds=1,
    )
    good = guard.open_pending_teardown(
        provider="vast",
        lane="offline",
        run_id="good",
        instance_id="good",
        registry_dir=tmp_path,
        max_age_seconds=1,
    )
    deleted = []

    class Provider:
        def inspect(self, instance):
            if instance == "bad":
                raise TimeoutError("inventory unavailable")
            return (
                {"http": 404}
                if instance in deleted
                else {"http": 200, "status": "observed", "desiredStatus": "running"}
            )

        def terminate(self, instance):
            deleted.append(instance)
            return {"status": "terminated"}

    result = guard.reap_orphans(
        registry_dir=tmp_path, provider_clients={"vast": Provider()}, now_epoch=200.0
    )
    assert result["open_billing_risk_count"] == 1 and result["reaped_count"] == 1
    assert json.loads(Path(bad["path"]).read_text())["status"] == "open"
    assert json.loads(Path(good["path"]).read_text())["status"] == "closed"
    assert deleted == ["good"]


@pytest.mark.parametrize(
    "payload", [[], {}, [{"id": "vol", "name": "exact", "dataCenterId": "loc"}]]
)
def test_volume_recovery_dry_run_never_changes_ledger(tmp_path, monkeypatch, payload):
    from types import SimpleNamespace

    monkeypatch.setattr(guard.time, "time", lambda: 100.0)
    record = guard.open_pending_teardown(
        provider="runpod",
        lane="offline",
        run_id="one",
        resource_kind="network_volume",
        resource_name="exact",
        provider_location="loc",
        registry_dir=tmp_path,
        max_age_seconds=1,
    )
    path = Path(record["path"])
    before = path.read_bytes()
    calls = []

    def api(method, endpoint, *_a, **_k):
        calls.append((method, endpoint))
        assert method == "GET"
        return (200, payload) if endpoint == "/networkvolumes" else (200, {"id": "vol"})

    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", api)
    result = guard.reap_orphans(
        registry_dir=tmp_path,
        provider_clients={"runpod": SimpleNamespace(_key=lambda: "fake")},
        now_epoch=200.0,
        dry_run=True,
    )
    assert path.read_bytes() == before
    assert result["reaped_count"] == 0
    assert calls


def test_ambiguous_volume_is_not_reported_closed_when_cancellation_refused(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(guard.time, "time", lambda: 100.0)
    record = guard.open_pending_teardown(
        provider="runpod",
        lane="offline",
        run_id="one",
        resource_kind="network_volume",
        resource_name="exact",
        provider_location="loc",
        registry_dir=tmp_path,
        max_age_seconds=1,
    )
    guard.mark_pending_teardown_ambiguous(record["path"], reason="lost acknowledgement")
    monkeypatch.setattr(
        "blueprint_pipeline.gpu_render_providers._runpod_call", lambda *_a, **_k: (200, [])
    )
    result = guard.reap_orphans(
        registry_dir=tmp_path,
        provider_clients={"runpod": SimpleNamespace(_key=lambda: "fake")},
        now_epoch=200.0,
    )
    assert result["reaped_count"] == 0 and result["open_billing_risk_count"] == 1
    assert json.loads(Path(record["path"]).read_text())["status"] == "open"


def test_existing_billing_receipt_cannot_close_another_instance(tmp_path):
    from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatch
    from tests.test_vast_official_billing_extractor import _materialize, _spec, INSTANCE_B, LABEL_B

    fixture = _fixture(tmp_path)
    output = tmp_path / "official.json"
    _materialize(fixture, output, expected=[_spec(fixture, INSTANCE_A, LABEL_A)])
    original = output.read_bytes()
    with pytest.raises(ValueError, match="billing_identity"):
        dispatch._materialize_official_billing_if_posted(
            billing_audit_root=tmp_path,
            adapter_result_path=fixture["terminals"][INSTANCE_B],
            adapter={"vast_instance_ids": [INSTANCE_B]},
            launch_label=LABEL_B,
            output_path=output,
        )
    assert output.read_bytes() == original


def test_invalid_billing_source_leaves_durable_recovery_action(tmp_path):
    from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatch

    fixture = _fixture(tmp_path / "old")
    receipt = json.loads(fixture["receipt"].read_text())
    receipt.pop("cohort_start_at")
    receipt.pop("cohort_end_at")
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    fixture["receipt"].write_text(json.dumps(receipt))
    output = tmp_path / "result" / "official.json"
    assert not dispatch._materialize_official_billing_if_posted(
        billing_audit_root=tmp_path / "old",
        adapter_result_path=fixture["terminals"][INSTANCE_A],
        adapter={"vast_instance_ids": [INSTANCE_A]},
        launch_label=LABEL_A,
        output_path=output,
    )
    pending = json.loads((output.parent / "official_billing_recovery.json").read_text())
    assert pending["status"] == "pending"
    assert pending["recovery_action"] == "refresh_official_billing_with_declared_period"
    assert pending["provider_instance_id"] == INSTANCE_A
    assert pending["official_charge_usd"] is None
    assert not output.exists()
    invalid_source_bytes = fixture["receipt"].read_bytes()
    _fixture(tmp_path / "fresh")
    assert dispatch._materialize_official_billing_if_posted(
        billing_audit_root=tmp_path,
        adapter_result_path=fixture["terminals"][INSTANCE_A],
        adapter={"vast_instance_ids": [INSTANCE_A]},
        launch_label=LABEL_A,
        output_path=output,
    )
    assert (
        json.loads((output.parent / "official_billing_recovery.json").read_text())["status"]
        == "resolved"
    )
    frozen = output.read_bytes()
    assert dispatch._materialize_official_billing_if_posted(
        billing_audit_root=tmp_path,
        adapter_result_path=fixture["terminals"][INSTANCE_A],
        adapter={"vast_instance_ids": [INSTANCE_A]},
        launch_label=LABEL_A,
        output_path=output,
    )
    assert output.read_bytes() == frozen
    assert fixture["receipt"].read_bytes() == invalid_source_bytes
