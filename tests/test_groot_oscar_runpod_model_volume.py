from blueprint_pipeline.groot_oscar_runpod_model_volume import (
    SCHEMA_VERSION,
    build_model_volume_admission,
)
from blueprint_pipeline.paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission,
)


def _admission(**overrides):
    values = {
        "release_image_ref": "docker.io/example/worker@sha256:" + "a" * 64,
        "data_center_id": "US-CA-2",
        "gpu_type_id": "NVIDIA A40",
        "required_cuda_version": "12.8",
        "volume_size_gib": 50,
        "hard_ttl_seconds": 2700,
        "max_spend_usd": 0.40,
        "hourly_rate_usd": 0.44,
        "inventory_verified_zero": True,
        "paid_mutation_authorized": True,
        "watchdog_armed_before_allocation": True,
    }
    values.update(overrides)
    return build_model_volume_admission(**values)


def test_model_volume_admission_accepts_bounded_exact_tuple() -> None:
    admission = _admission()
    assert admission["status"] == "admitted"
    grant = require_paid_resource_admission(
        admission,
        resource_class="model_volume",
        expected_schema_version=SCHEMA_VERSION,
    )
    assert grant.resource_class == "model_volume"


def test_model_volume_admission_rejects_unbounded_or_duplicate_resources() -> None:
    admission = _admission(
        hard_ttl_seconds=7200,
        inventory_verified_zero=False,
        watchdog_armed_before_allocation=False,
    )
    assert admission["status"] == "blocked"
    assert "model_volume_ttl_outside_guardrail" in admission["blockers"]
    assert "model_volume_preallocation_inventory_not_zero" in admission["blockers"]
    assert "model_volume_watchdog_not_armed_before_allocation" in admission["blockers"]
    try:
        require_paid_resource_admission(
            admission,
            resource_class="model_volume",
            expected_schema_version=SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked:
        pass
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("blocked admission reached paid mutation seam")


def test_model_volume_admission_rejects_cost_above_cap() -> None:
    admission = _admission(max_spend_usd=0.10)
    assert admission["status"] == "blocked"
    assert "model_volume_ttl_cost_exceeds_max_spend" in admission["blockers"]


def test_model_volume_admission_allows_explicitly_bounded_higher_rate() -> None:
    admission = _admission(
        gpu_type_id="NVIDIA L40S",
        hourly_rate_usd=0.99,
        max_spend_usd=0.75,
    )
    assert admission["status"] == "admitted"
