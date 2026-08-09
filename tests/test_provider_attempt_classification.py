from __future__ import annotations

from blueprint_pipeline.provider_attempt_classification import (
    classify_provider_attempt,
)
from blueprint_pipeline.vast_provider_adapter import _default_machine_avoidlist_path


def test_pre_bundle_host_failure_is_not_a_scientific_attempt_or_auto_retry() -> None:
    receipt = classify_provider_attempt(
        provider_command={
            "provider_bundle_started": False,
            "provider_entrypoint_started": False,
            "provider_runtime_output_zip_received": False,
        },
        blockers=["vast_heartbeat_container_missing"],
    )
    assert receipt["classification"] == "pre_execution_provider_null"
    assert receipt["scientific_attempt_consumed"] is False
    assert receipt["pre_execution_requeue_eligible_in_principle"] is True
    assert receipt["automatic_requeue_authorized"] is False
    assert receipt["maximum_automatic_requeues"] == 0


def test_started_entrypoint_consumes_attempt_even_when_output_is_missing() -> None:
    receipt = classify_provider_attempt(
        provider_command={
            "provider_bundle_started": True,
            "provider_entrypoint_started": True,
            "provider_runtime_output_zip_received": False,
        },
        blockers=["provider_output_upload_marker_missing"],
    )
    assert receipt["classification"] == "provider_bundle_attempt_started"
    assert receipt["scientific_attempt_consumed"] is True
    assert receipt["pre_execution_requeue_eligible_in_principle"] is False


def test_machine_avoidlist_is_shared_across_sibling_provider_jobs(tmp_path) -> None:
    first_job = tmp_path / "attempt_001"
    second_job = tmp_path / "attempt_002"
    assert _default_machine_avoidlist_path(first_job) == (
        tmp_path / "vast_machine_avoidlist.json"
    )
    assert _default_machine_avoidlist_path(second_job) == (
        tmp_path / "vast_machine_avoidlist.json"
    )
