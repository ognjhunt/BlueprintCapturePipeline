"""A clear stop is never refused for a budget reason: the GCP monthly budget refuses new spend only."""

from __future__ import annotations

import pytest

from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


@pytest.mark.parametrize("resource_class", ["gpu_render", "vast_provider_adapter"])
def test_a_stop_is_admitted_while_the_gcp_monthly_budget_refuses_new_spend(tmp_path, monkeypatch, resource_class):
    # The two callers that pass spend_effect="stop" (qualification_control_admission, the warm scene closeout).
    monkeypatch.setenv("BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", str(tmp_path / "absent-report.json"))
    admission = build_paid_lane_admission(resource_class=resource_class)
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        require_paid_resource_admission(admission, resource_class=resource_class,
                                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    assert "gcp_monthly_budget_report_missing_or_invalid" in refused.value.blockers
    grant = require_paid_resource_admission({**admission, "spend_effect": "stop"}, resource_class=resource_class,
                                            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                            spend_effect="stop")
    assert grant.resource_class == resource_class


def test_a_stop_declared_once_is_still_checked_against_the_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", str(tmp_path / "absent-report.json"))
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        require_paid_resource_admission(build_paid_lane_admission(resource_class="gpu_render"),
                                        resource_class="gpu_render",
                                        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
                                        spend_effect="stop")
    assert "gcp_monthly_budget_report_missing_or_invalid" in refused.value.blockers
