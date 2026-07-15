import json
import os
import time
from pathlib import Path

import pytest

from blueprint_pipeline import groot_oscar_runpod_model_volume as model_volume
from blueprint_pipeline.groot_oscar_runpod_model_volume import (
    SCHEMA_VERSION,
    _extract_id,
    _matching_resources,
    _safe_provider_error_summary,
    _single_gpu_capacity_verified,
    _watchdog_process_running,
    build_model_volume_admission,
    launch_detached,
    run_model_volume,
    watchdog,
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
        "volume_hourly_rate_usd": 50 * 0.07 / (30 * 24),
        "capacity_verified": True,
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


def test_model_volume_admission_rejects_gpu_only_data_center() -> None:
    admission = _admission(data_center_id="US-NC-2")
    assert admission["status"] == "blocked"
    assert (
        "model_volume_data_center_not_network_volume_capable"
        in admission["blockers"]
    )


def test_model_volume_admission_rejects_gpu_outside_campaign_contract() -> None:
    admission = _admission(
        gpu_type_id="NVIDIA GeForce RTX 4090",
        hourly_rate_usd=0.77,
        max_spend_usd=0.60,
    )
    assert admission["status"] == "blocked"
    assert (
        "model_volume_gpu_type_outside_authorized_campaign"
        in admission["blockers"]
    )


def test_model_volume_admission_accepts_authorized_rtx_fallbacks() -> None:
    for gpu_type_id in (
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    ):
        admission = _admission(
            data_center_id="US-WA-1",
            gpu_type_id=gpu_type_id,
            hourly_rate_usd=0.77,
            max_spend_usd=0.60,
        )
        assert admission["status"] == "admitted"


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


def test_model_volume_admission_rejects_unverified_single_gpu_capacity() -> None:
    admission = _admission(capacity_verified=False)
    assert admission["status"] == "blocked"
    assert "model_volume_single_gpu_capacity_not_verified" in admission["blockers"]


def test_model_volume_accepts_exact_one_gpu_offer_when_counts_are_nullable() -> None:
    capacity = {"status": "available", "capacity_confidence": "advisory"}
    selected = {
        "capacity_confidence": "advisory",
        "single_gpu_count_known": False,
        "available_gpu_counts": [],
        "single_gpu_offer_requested": True,
        "single_gpu_offer_available": True,
        "capacity_data_center_id": "US-NC-1",
        "capacity_allowed_cuda_versions": ["12.8"],
    }
    assert _single_gpu_capacity_verified(
        capacity=capacity,
        selected=selected,
        data_center_id="US-NC-1",
        required_cuda_version="12.8",
    )
    selected["single_gpu_offer_available"] = False
    assert not _single_gpu_capacity_verified(
        capacity=capacity,
        selected=selected,
        data_center_id="US-NC-1",
        required_cuda_version="12.8",
    )


def test_model_volume_admission_requires_verified_capacity_and_storage_rate() -> None:
    admission = _admission(
        capacity_verified=False,
        volume_hourly_rate_usd=0,
    )
    assert admission["status"] == "blocked"
    assert "model_volume_single_gpu_capacity_not_verified" in admission["blockers"]
    assert "model_volume_storage_hourly_rate_missing" in admission["blockers"]


def test_model_volume_inventory_failure_is_not_treated_as_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        model_volume,
        "_runpod_call",
        lambda method, path, body, **kwargs: (503, {}),
    )
    pods, volumes, verified = _matching_resources(
        key="secret",
        pod_prefix=model_volume.POD_NAME_PREFIX,
        volume_prefix=model_volume.VOLUME_NAME_PREFIX,
    )
    assert pods == []
    assert volumes == []
    assert verified is False


def test_model_volume_global_inventory_counts_unrelated_names(monkeypatch) -> None:
    def fake_call(method, path, body, **kwargs):
        del method, body, kwargs
        if path == "/pods":
            return 200, [{"id": "other-pod", "name": "unrelated-worker"}]
        return 200, [{"id": "other-volume", "name": "unrelated-cache"}]

    monkeypatch.setattr(model_volume, "_runpod_call", fake_call)
    pods, volumes, verified = _matching_resources(
        key="secret",
        pod_prefix=None,
        volume_prefix=None,
    )
    assert verified is True
    assert pods == ["other-pod"]
    assert volumes == ["other-volume"]


def test_model_volume_uses_provider_reported_volume_size() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/groot_oscar_runpod_model_volume.py"
    ).read_text(encoding="utf-8")
    assert "build_runpod_network_volume_evidence(" in source
    assert '"size_bytes": volume_size_gib * 1024**3' not in source
    assert 'volume_evidence["size_bytes"] != volume_size_gib * 1024**3' in source


def test_model_volume_rejects_provider_ids_that_can_escape_urls() -> None:
    assert _extract_id({"id": "safe-pod_123"}) == "safe-pod_123"
    assert _extract_id({"id": "../../pods/other"}) == ""
    assert _extract_id({"id": True}) == ""


def test_model_volume_records_sanitized_provider_error_without_request_body() -> None:
    summary = _safe_provider_error_summary(
        {
            "statusCode": 500,
            "message": "No network volume capacity; token=hf_secretvalue123456",
            "request": {"env": {"HF_TOKEN": "must-not-be-recorded"}},
        }
    )
    assert summary == (
        "statusCode=500; message=No network volume capacity; [REDACTED]"
    )
    assert "request" not in summary
    assert "secretvalue" not in summary
    assert _safe_provider_error_summary({"request": {"unsafe": "body"}}) is None


def test_model_volume_detached_launch_is_single_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        pid = 1234

    monkeypatch.setattr(model_volume.subprocess, "Popen", lambda *args, **kwargs: Process())
    launched = launch_detached(output_dir=tmp_path, run_arguments=["--allow-paid"])
    assert launched["status"] == "supervisor_started"
    with pytest.raises(ValueError, match="already_has_supervisor"):
        launch_detached(output_dir=tmp_path, run_arguments=["--allow-paid"])


def test_model_volume_watchdog_emits_nonce_bound_armed_handoff(tmp_path: Path) -> None:
    state = tmp_path / "watchdog_state.json"
    state.write_text(
        json.dumps(
            {
                "deadline_epoch": time.time() + 120,
                "pod_name_prefix": "blueprint-groot-oscar-canary-model-test",
                "volume_name": "blueprint-groot-oscar-models-test",
                "watchdog_nonce": "nonce-for-test",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "watchdog_handoff.json").write_text(
        json.dumps({"status": "cancelled_before_provider_allocation"}),
        encoding="utf-8",
    )
    assert watchdog(state_path=state) == 0
    armed = json.loads((tmp_path / "watchdog_armed.json").read_text())
    assert armed["status"] == "armed"
    assert armed["pid"] == os.getpid()
    assert armed["watchdog_nonce"] == "nonce-for-test"


def test_model_volume_requires_armed_handoff_before_paid_admission() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/groot_oscar_runpod_model_volume.py"
    ).read_text(encoding="utf-8")
    run_source = source[
        source.index("def run_model_volume(") : source.index("def launch_detached(")
    ]
    assert run_source.index('armed_path = output / "watchdog_armed.json"') < run_source.index(
        "require_paid_resource_admission("
    )
    assert "watchdog_armed_before_allocation=watchdog_armed" in run_source
    assert '"status": "volume_ready_watchdog_retained"' in run_source
    assert '"teardown_owner": "independent_model_volume_watchdog"' in run_source
    assert '"watchdog_pid": watchdog_pid' in run_source
    assert '"watchdog_state_path": str(state_path)' in run_source
    assert 'final_volumes == [volume_id]' in run_source


def test_ready_volume_handoff_does_not_disarm_deadline_watchdog() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/groot_oscar_runpod_model_volume.py"
    ).read_text(encoding="utf-8")
    watchdog_source = source[source.index("def watchdog(") : source.index("def _worker_script(")]
    assert "volume_ready_watchdog_retained" not in watchdog_source
    assert "failure_cleanup_provider_terminal" in watchdog_source
    assert "cancelled_before_provider_allocation" in watchdog_source


def test_ready_volume_handoff_still_deletes_volume_at_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "watchdog_state.json"
    state.write_text(
        json.dumps(
            {
                "deadline_epoch": time.time() + 0.02,
                "pod_name_prefix": "blueprint-groot-oscar-canary-model-test",
                "volume_name": "blueprint-groot-oscar-models-test",
                "watchdog_nonce": "nonce-for-ready-test",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "watchdog_handoff.json").write_text(
        json.dumps({"status": "volume_ready_watchdog_retained"}), encoding="utf-8"
    )

    class Provider:
        @staticmethod
        def _key() -> str:
            return "runpod-test-key"

    inventories = iter(
        [([], ["volume-1"], True), ([], [], True)]
    )
    deleted: list[str] = []
    monkeypatch.setattr(model_volume, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        model_volume, "_matching_resources", lambda **_kwargs: next(inventories)
    )
    monkeypatch.setattr(
        model_volume,
        "_delete_volume",
        lambda **kwargs: deleted.append(kwargs["volume_id"])
        or {"provider_absence_confirmed": True},
    )
    monkeypatch.setattr(model_volume.time, "sleep", lambda _seconds: None)

    assert watchdog(state_path=state) == 0
    assert deleted == ["volume-1"]
    result = json.loads((tmp_path / "watchdog_result.json").read_text())
    assert result["status"] == "provider_terminal"


def test_model_volume_reads_hf_token_before_provider_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provider:
        @staticmethod
        def _key() -> str:
            return "runpod-test-key"

    monkeypatch.setattr(model_volume, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        model_volume,
        "_matching_resources",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("inventory queried")),
    )
    result = run_model_volume(
        output_dir=tmp_path / "out",
        release_image_ref="docker.io/example/worker@sha256:" + "a" * 64,
        data_center_id="US-WA-1",
        gpu_type_id="NVIDIA L40S",
        required_cuda_version="12.8",
        volume_size_gib=50,
        hard_ttl_seconds=2700,
        max_spend_usd=0.40,
        volume_hourly_rate_usd=0.01,
        hf_token_file=tmp_path / "missing-hf-token",
        allow_paid=True,
    )
    assert result["status"] == "blocked_before_allocation"
    assert result["blockers"] == ["model_volume_hf_token_unavailable"]
    assert result["provider_mutation_attempted"] is False


def test_model_volume_persists_blocked_result_when_runpod_key_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Provider:
        @staticmethod
        def _key() -> str:
            return ""

    monkeypatch.setattr(model_volume, "get_render_provider", lambda _name: Provider())
    monkeypatch.setattr(
        model_volume,
        "_matching_resources",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("inventory queried")),
    )
    output = tmp_path / "out"
    result = run_model_volume(
        output_dir=output,
        release_image_ref="docker.io/example/worker@sha256:" + "a" * 64,
        data_center_id="US-WA-1",
        gpu_type_id="NVIDIA L40S",
        required_cuda_version="12.8",
        volume_size_gib=50,
        hard_ttl_seconds=2700,
        max_spend_usd=0.40,
        volume_hourly_rate_usd=0.01,
        hf_token_file=tmp_path / "hf-token",
        allow_paid=True,
    )

    assert result["status"] == "blocked_before_allocation"
    assert result["blockers"] == ["model_volume_runpod_api_key_unavailable"]
    assert result["provider_mutation_attempted"] is False
    assert json.loads((output / "model_volume_result.json").read_text()) == result


def test_model_volume_dead_watchdog_forces_failure_cleanup_before_handoff() -> None:
    class Process:
        @staticmethod
        def poll() -> int:
            return 1

    assert _watchdog_process_running(Process()) is False
    source = (
        Path(__file__).resolve().parents[1]
        / "src/blueprint_pipeline/groot_oscar_runpod_model_volume.py"
    ).read_text(encoding="utf-8")
    run_source = source[
        source.index("def run_model_volume(") : source.index("def launch_detached(")
    ]
    dead_guard = run_source.index("if success and not watchdog_retained:")
    failure_cleanup = run_source.index("retained_volume_ids =", dead_guard)
    ready_handoff = run_source.index('"status": "volume_ready_watchdog_retained"')
    assert dead_guard < failure_cleanup < ready_handoff
    assert "and watchdog_pid is not None" in run_source
