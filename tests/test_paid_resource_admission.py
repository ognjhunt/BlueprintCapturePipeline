import pytest

from blueprint_pipeline import paid_resource_admission
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    RUNPOD_ALLOCATION_RESOURCE_CLASSES,
    RUNPOD_RETIRED_BLOCKER,
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
    require_paid_resource_admission_grant,
    require_paid_resource_budget,
)


@pytest.mark.parametrize(
    "resource_class",
    ["cpu_build", "evaluator_api", "gpu_canary", "vast_wam_async"],
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
        resource_class="vast_wam_async",
        blockers=[],
    )
    grant = require_paid_resource_admission(
        admission,
        resource_class="vast_wam_async",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    assert grant.resource_class == "vast_wam_async"


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
        resource_class="vast_provider_adapter",
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
        resource_class="vast_wam_async",
        blockers=[],
    )
    grant = require_paid_resource_admission(
        admission,
        resource_class="vast_wam_async",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    require_paid_resource_admission_grant(
        grant,
        resource_class="vast_wam_async",
    )
    with pytest.raises(PaidResourceAdmissionBlocked) as unbound:
        require_paid_resource_admission_grant(
            grant,
            resource_class="vast_wam_async",
            require_allocation_binding=True,
        )
    assert "paid_resource_admission_grant_binding_missing" in unbound.value.blockers
    forged = PaidResourceAdmissionGrant(
        resource_class="vast_wam_async",
        schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        _issuer=object(),
    )
    for invalid, resource_class in (
        (None, "vast_wam_async"),
        (forged, "vast_wam_async"),
        (grant, "vast_provider_adapter"),
    ):
        with pytest.raises(PaidResourceAdmissionBlocked):
            require_paid_resource_admission_grant(
                invalid,
                resource_class=resource_class,
            )


@pytest.mark.parametrize("resource_class", sorted(RUNPOD_ALLOCATION_RESOURCE_CLASSES))
def test_no_grant_is_issued_for_a_retired_runpod_allocation(resource_class: str) -> None:
    """RunPod is retired (2026-11-05): every class that starts RunPod spend is refused
    at the shared chokepoint, so no RunPod adapter can ever hold a grant."""

    admission = build_paid_lane_admission(resource_class=resource_class, blockers=[])
    assert admission["status"] == "blocked"
    assert admission["blockers"] == [RUNPOD_RETIRED_BLOCKER]
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        require_paid_resource_admission(
            {"schema_version": "example.v1", "status": "admitted", "blockers": [],
             "resource_class": resource_class},
            resource_class=resource_class, expected_schema_version="example.v1",
        )
    assert RUNPOD_RETIRED_BLOCKER in refused.value.blockers
    with pytest.raises(PaidResourceAdmissionBlocked) as budget:
        require_paid_resource_budget(resource_class=resource_class)
    assert budget.value.blockers == [RUNPOD_RETIRED_BLOCKER]


def test_runpod_teardown_and_live_classes_are_not_retired() -> None:
    # A RunPod live-execution grant only stops or terminates an existing pod.
    assert "runpod_live_execution" not in RUNPOD_ALLOCATION_RESOURCE_CLASSES
    for resource_class in ("runpod_live_execution", "vast_provider_adapter", "gpu_render", "gpu_canary"):
        assert paid_resource_admission.runpod_retirement_blockers(resource_class) == []
    assert paid_resource_admission.runpod_retirement_blockers() == [RUNPOD_RETIRED_BLOCKER]
    grant = require_paid_resource_admission(
        build_paid_lane_admission(resource_class="runpod_live_execution"),
        resource_class="runpod_live_execution",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
    require_paid_resource_admission_grant(grant, resource_class="runpod_live_execution")

