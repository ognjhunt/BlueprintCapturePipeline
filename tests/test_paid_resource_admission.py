import pytest

from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
    require_paid_resource_admission_grant,
)


@pytest.mark.parametrize(
    "resource_class", ["cpu_build", "gpu_canary", "runpod_serverless_active_worker"]
)
def test_shared_chokepoint_grants_only_exact_admitted_contract(
    resource_class: str,
) -> None:
    grant = require_paid_resource_admission(
        {"schema_version": "example.v1", "status": "admitted", "blockers": []},
        resource_class=resource_class,
        expected_schema_version="example.v1",
    )
    assert grant.resource_class == resource_class


@pytest.mark.parametrize(
    "admission",
    [
        {},
        {"schema_version": "wrong", "status": "admitted", "blockers": []},
        {"schema_version": "example.v1", "status": "blocked", "blockers": ["x"]},
    ],
)
def test_shared_chokepoint_fails_closed(admission: dict) -> None:
    with pytest.raises(PaidResourceAdmissionBlocked):
        require_paid_resource_admission(
            admission,
            resource_class="cpu_build",
            expected_schema_version="example.v1",
        )


def test_paid_lane_admission_normalizes_existing_lane_checks() -> None:
    admission = build_paid_lane_admission(
        resource_class="runpod_provider_adapter",
        blockers=[],
    )
    grant = require_paid_resource_admission(
        admission,
        resource_class="runpod_provider_adapter",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    assert grant.resource_class == "runpod_provider_adapter"


def test_paid_lane_admission_preserves_blockers_and_fails_closed() -> None:
    admission = build_paid_lane_admission(
        resource_class="vast_wam_async",
        blockers=["budget_missing", "budget_missing"],
    )
    assert admission["blockers"] == ["budget_missing"]
    with pytest.raises(PaidResourceAdmissionBlocked):
        require_paid_resource_admission(
            admission,
            resource_class="vast_wam_async",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )


def test_paid_lane_admission_cannot_be_reused_across_resource_classes() -> None:
    admission = build_paid_lane_admission(
        resource_class="runpod_provider_adapter",
        blockers=[],
    )
    with pytest.raises(PaidResourceAdmissionBlocked) as raised:
        require_paid_resource_admission(
            admission,
            resource_class="vast_wam_async",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    assert "paid_resource_admission_class_mismatch" in raised.value.blockers


def test_opaque_grant_rejects_missing_forged_and_cross_class_capabilities() -> None:
    admission = build_paid_lane_admission(
        resource_class="runpod_provider_adapter",
        blockers=[],
    )
    grant = require_paid_resource_admission(
        admission,
        resource_class="runpod_provider_adapter",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    require_paid_resource_admission_grant(
        grant,
        resource_class="runpod_provider_adapter",
    )
    with pytest.raises(PaidResourceAdmissionBlocked) as unbound:
        require_paid_resource_admission_grant(
            grant,
            resource_class="runpod_provider_adapter",
            require_allocation_binding=True,
        )
    assert "paid_resource_admission_grant_binding_missing" in unbound.value.blockers
    forged = PaidResourceAdmissionGrant(
        resource_class="runpod_provider_adapter",
        schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        _issuer=object(),
    )
    for invalid, resource_class in (
        (None, "runpod_provider_adapter"),
        (forged, "runpod_provider_adapter"),
        (grant, "vast_provider_adapter"),
    ):
        with pytest.raises(PaidResourceAdmissionBlocked):
            require_paid_resource_admission_grant(
                invalid,
                resource_class=resource_class,
            )
