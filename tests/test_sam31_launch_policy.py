"""Replay the observed capacity/create mismatch without a provider call."""
from __future__ import annotations

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def test_admission_cost_uses_frozen_rate_ceiling_instead_of_cheaper_offer():
    from tests.test_sam31_gpu_admission import _build, _preflight as admission_preflight
    snapshot = admission_preflight()
    snapshot["capacity_request"]["max_hourly_rate_usd"] = 10
    snapshot["preflight_digest"] = canonical_digest(snapshot, digest_field="preflight_digest")
    admission, bound = _build(preflight=snapshot, execute=True, qualified=True)
    assert "sam31_gpu_budget_below_worst_case_cost" in admission["blockers"]
    assert bound["provider_mutation_authorized"] is False


