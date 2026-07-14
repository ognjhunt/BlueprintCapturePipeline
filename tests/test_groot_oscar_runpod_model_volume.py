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
    build_model_volume_admission,
    launch_detached,
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


def test_model_volume_admission_rejects_low_stock_without_single_gpu_count() -> None:
    admission = _admission(capacity_verified=False)
    assert admission["status"] == "blocked"
    assert "model_volume_single_gpu_capacity_not_verified" in admission["blockers"]


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
    (tmp_path / "watchdog_handoff.json").write_text("{}", encoding="utf-8")
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
