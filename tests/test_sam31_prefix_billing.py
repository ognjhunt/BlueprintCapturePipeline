"""Official audit discovery is read-only; missing billing never licenses reruns."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_evaluation_sam31_prefix_adoption as adoption
from blueprint_pipeline import task_evaluation_sam31_prefix_billing as billing
from blueprint_pipeline import task_evaluation_sam31_prefix_evidence as evidence
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_billing_reconciler import BILLING_SOURCE_SCHEMA_VERSION, VAST_CHARGES_URL
from tests.test_vast_official_billing_extractor import _charge, _sha256, _write
from tests.test_restart_adoption_topology import topology as topology, A, B
from tests.test_sam31_prefix_adoption import prefix as prefix

REQUEST = "sha256:" + "d" * 64
INSTANCE = 50691996
LABEL = "blueprint-sam31-source-tracks-fixture-" + REQUEST[7:19]


def _audit(tmp_path, monkeypatch, defect=None):
    root = tmp_path / "billing-audit"
    monkeypatch.setenv("BLUEPRINT_PROVIDER_BILLING_AUDIT_ROOT", str(root))
    row = _charge(instance_id=INSTANCE, label=LABEL, total=.105, gpu=.1, disk=.005)
    if defect == "instance":
        row["source"] = "instance-42"
    if defect == "label":
        row["metadata"]["label"] = "unrelated-owner"
    rows = [row, deepcopy(row)] if defect == "duplicate" else [row]
    response = _write(root / "20260817T055421Z/response-001-vast.json",
                      {"success": True, "next_token": None, "results": rows})
    value = {"schema_version": BILLING_SOURCE_SCHEMA_VERSION, "status": "reconciled",
        "generated_at": "2026-08-17T05:54:21+00:00", "cohort_start_at": "2026-07-01T00:00:00+00:00",
        "cohort_end_at": "2026-08-17T05:54:21+00:00", "provider_totals_usd": {"vast": .105},
        "sources": [{"provider": "vast", "endpoint": VAST_CHARGES_URL,
            "request_query_digest": "sha256:" + "1" * 64, "response_digest": _sha256(response),
            "response_size_bytes": response.stat().st_size, "retained_path": str(response)}],
        "provider_mutation_performed": False, "raw_secret_values_recorded": False}
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    source = _write(response.parent / "provider_billing_source_receipt.json", value)
    if defect == "bytes":
        response.write_text("{}")
    if defect == "symlink":
        moved = source.with_name("original.json")
        source.rename(moved)
        source.symlink_to(moved)
    return source


def test_discovers_exact_official_charge_without_writes_or_provider_calls(tmp_path, monkeypatch):
    source = _audit(tmp_path, monkeypatch)
    monkeypatch.delenv("BLUEPRINT_PROVIDER_BILLING_AUDIT_ROOT")
    monkeypatch.setattr(billing, "DEFAULT_AUDIT_ROOT", str(source.parent.parent))
    before = {p: p.read_bytes() for p in source.parent.glob("*.json")}
    def forbidden(*args, **kwargs):
        pytest.fail("billing discovery must not fetch or write")
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    charge = billing.tracking_charge({"instance_id": INSTANCE, "request_digest": REQUEST}, approved_roots=(tmp_path,))
    assert charge["provider_billing_source_receipt"]["path"] == str(source)
    assert charge["provider_instance_id"] == INSTANCE and charge["launch_label"] == LABEL
    assert charge["official_charge_usd"] == .105 and charge["provider_mutation_performed"] is False
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("defect", ["instance", "label", "duplicate", "bytes", "symlink", "outside", "missing"])
def test_unverified_or_unavailable_charge_remains_pending(tmp_path, monkeypatch, defect):
    source = _audit(tmp_path, monkeypatch, defect)
    if defect == "missing":
        source.unlink()
    with pytest.raises(billing.Sam31PrefixBillingPending):
        billing.tracking_charge({"instance_id": INSTANCE, "request_digest": REQUEST},
                                approved_roots=() if defect == "outside" else (tmp_path,))


def test_audit_read_race_cannot_downgrade_to_a_shorter_prefix(tmp_path, monkeypatch):
    source = _audit(tmp_path, monkeypatch)
    stat = Path.stat
    def unavailable(path, *args, **kwargs):
        if path == source:
            raise FileNotFoundError("audit observation unavailable")
        return stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", unavailable)
    with pytest.raises(billing.Sam31PrefixBillingPending, match="audit_read_unavailable"):
        billing.tracking_charge({"instance_id": INSTANCE, "request_digest": REQUEST}, approved_roots=(tmp_path,))


def test_new_adoption_binds_discovered_receipt_and_retry_keeps_same_bytes(topology, tmp_path, monkeypatch):
    source = _audit(tmp_path, monkeypatch)
    build, materialize, _calls = topology
    original = build(A, "sam31_tracking")
    profile_before = Path(original["profile_ref"]["path"]).read_bytes()
    validated = adoption.validate_tracking
    def tracking(*args, **kwargs):
        identity = validated(*args)
        return {**identity, "official_charge": billing.tracking_charge(
            {"instance_id": INSTANCE, "request_digest": REQUEST}, args[-1],
            approved_roots=kwargs.get("billing_audit_roots"))}
    monkeypatch.setattr(adoption, "validate_tracking", tracking)
    real = adoption.materialize_completed_prefix_adoption
    monkeypatch.setattr(adoption, "materialize_completed_prefix_adoption",
                        lambda **kw: real(**{**kw, "sam31_billing_source_path": None}))
    result, ref, args = materialize(original, B, "sam31_tracking", "auto-billing")
    assert result["sam31_billing_source"] == adoption.record(source)
    before = Path(ref["path"]).read_bytes()
    assert real(**{**args, "sam31_billing_source_path": None}) == result
    assert Path(ref["path"]).read_bytes() == before
    assert Path(original["profile_ref"]["path"]).read_bytes() == profile_before


def test_selector_propagates_pending_after_completed_tracking_validation(topology, tmp_path, monkeypatch):
    build, materialize, calls = topology
    original = build(A, "sam31_tracking")
    _value, _ref, kwargs = materialize(original, B, "sam31_tracking", "argument-fixture")
    kwargs.pop("through_phase")
    kwargs.update(output_path=tmp_path / "must-not-publish.json", sam31_billing_source_path=None)
    real = adoption.validate_tracking
    def pending(*args, **kw):
        real(*args)  # Synthetic topology's original producer/model assertions.
        raise billing.Sam31PrefixBillingPending("sam31_adoption_official_billing_pending")
    monkeypatch.setattr(adoption, "validate_tracking", pending)
    with pytest.raises(billing.Sam31PrefixBillingPending):
        adoption.select_completed_prefix_adoption(**kwargs)
    assert calls["tracking_validation"] > 0
    assert not kwargs["output_path"].exists()
    # A changed task must fail its actual science binding before billing lookup.
    current = Path(kwargs["current_host_inputs"]["task_request"]["path"])
    task = json.loads(current.read_text())
    task["subject"]["source_instance_id"] = "different"
    _write(current, task)
    kwargs["current_host_inputs"]["task_request"] = adoption.record(current)
    monkeypatch.setattr(adoption, "validate_tracking", lambda *a, **kw: pytest.fail("unrelated task reached billing"))
    with pytest.raises(ValueError, match="sam31_adoption_task_or_source_changed"):
        adoption.materialize_completed_prefix_adoption(**kwargs, through_phase="sam31_tracking")


@pytest.mark.parametrize("failed_check", ["model", "tracking"])
def test_unverified_tracking_cannot_invent_pending_billing(tmp_path, monkeypatch, failed_check):
    def refused(*a, **kw):
        raise ValueError("unverified_" + failed_check)
    monkeypatch.setattr(evidence, "_load", lambda _: SimpleNamespace(
        validate_retained_paid_stage=refused if failed_check == "tracking" else lambda *a, **kw: None))
    monkeypatch.setattr(evidence, "validate_model_sources", refused)
    monkeypatch.setattr(billing, "tracking_charge", lambda *a, **kw: pytest.fail("unverified tracking reached billing"))
    with pytest.raises(ValueError, match="unverified_" + failed_check):
        evidence.validate_tracking({}, {}, {}, tmp_path, A, None, billing_audit_roots=(tmp_path,))
