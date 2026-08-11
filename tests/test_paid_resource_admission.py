import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
    require_paid_resource_admission_grant,
)


@pytest.mark.parametrize(
    "resource_class",
    ["cpu_build", "evaluator_api", "gpu_canary", "runpod_serverless_active_worker"],
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


def test_vast_grant_binds_exact_allowed_active_instance_inventory() -> None:
    binding = {
        "allowed_active_vast_instance_ids": [47373597],
    }
    admission = build_paid_lane_admission(
        resource_class="vast_provider_adapter",
        blockers=[],
    )
    admission.update(
        {
            "allocation_binding": binding,
            "allocation_binding_digest": canonical_digest(binding),
        }
    )

    grant = require_paid_resource_admission(
        admission,
        resource_class="vast_provider_adapter",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )

    assert grant.allowed_active_instance_ids == (47373597,)
    require_paid_resource_admission_grant(
        grant,
        resource_class="vast_provider_adapter",
        allowed_active_instance_ids=(47373597,),
    )
    with pytest.raises(PaidResourceAdmissionBlocked) as mismatch:
        require_paid_resource_admission_grant(
            grant,
            resource_class="vast_provider_adapter",
            allowed_active_instance_ids=(47373598,),
        )
    assert "paid_resource_admission_grant_active_instances_mismatch" in mismatch.value.blockers


@pytest.mark.parametrize(
    ("allowed_ids", "expected_blocker"),
    [
        ([47373597, 47373597], "paid_resource_allowed_active_instance_ids_not_unique"),
        ([True], "paid_resource_allowed_active_instance_ids_invalid"),
    ],
)
def test_vast_grant_rejects_invalid_allowed_active_instance_inventory(
    allowed_ids: list[object], expected_blocker: str
) -> None:
    admission = build_paid_lane_admission(
        resource_class="vast_provider_adapter",
        blockers=[],
    )
    binding = {
        "allowed_active_vast_instance_ids": allowed_ids,
    }
    admission["allocation_binding"] = binding
    admission["allocation_binding_digest"] = canonical_digest(binding)

    with pytest.raises(PaidResourceAdmissionBlocked) as raised:
        require_paid_resource_admission(
            admission,
            resource_class="vast_provider_adapter",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    assert expected_blocker in raised.value.blockers


def test_vast_grant_recomputes_allocation_binding_digest_before_issue() -> None:
    binding = {
        "allowed_active_vast_instance_ids": [47373597],
        "hard_cap_usd": 2.0,
    }
    admission = build_paid_lane_admission(
        resource_class="vast_provider_adapter",
        blockers=[],
    )
    admission.update(
        {
            "allocation_binding": binding,
            "allocation_binding_digest": canonical_digest(binding),
        }
    )
    admission["allocation_binding"]["hard_cap_usd"] = 20.0

    with pytest.raises(PaidResourceAdmissionBlocked) as raised:
        require_paid_resource_admission(
            admission,
            resource_class="vast_provider_adapter",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )

    assert "paid_resource_allocation_binding_digest_mismatch" in raised.value.blockers
