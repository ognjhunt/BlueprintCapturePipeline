import pytest

from blueprint_pipeline.paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission,
)


@pytest.mark.parametrize("resource_class", ["cpu_build", "gpu_canary"])
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
