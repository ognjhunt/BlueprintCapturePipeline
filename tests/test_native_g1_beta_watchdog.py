"""Actual name, dry-run and fleet-owner contracts; no provider operations."""
from __future__ import annotations

import shlex
import time

import pytest

from blueprint_pipeline import groot_oscar_runpod_watchdog as watchdog
from blueprint_pipeline import vast_independent_watchdog_control as control
from blueprint_pipeline import vast_provider_adapter as adapter
from blueprint_pipeline.vast_probe_guards import cold_pull_aware_heartbeat_no_progress_seconds
from scripts import gpu_spend_guard as guard

PREFIX = "blueprint-native-task-policy-canary-"


def exercise_native_g1_watchdog(tmp_path, monkeypatch, resource_name):
    """Also called by the session test with its real generated authority name."""
    name = resource_name
    assert control.validate_independent_vast_watchdog_names(
        pod_name_prefix=PREFIX, resource_name_exact=name) == (PREFIX, name)
    root = tmp_path / "allocator/attempts/attempt_001/independent_vast_watchdog"
    deadline = time.time() + 1800
    armed = watchdog.arm_watchdog(out_dir=root, pod_name_prefix=name,
        resource_name_exact=name, deadline_epoch=deadline, provider_name="vast")
    assert armed["resource_name_exact"] == name
    monkeypatch.setattr(adapter, "_api_json", lambda **kwargs: pytest.fail("provider API call"))
    result = adapter.run_vast_provider_adapter(job_dir=tmp_path / "dry-run", mode="dry-run",
        instance_label_prefix=PREFIX, instance_label_exact=name)
    assert result["status"] == "dry_run_ready" and result["api_call_performed"] is False
    (root / "started_vast_instance_id.txt").write_text("101")
    command = ["python", "-m", "blueprint_pipeline.groot_oscar_runpod_watchdog", "--provider", "vast",
        "--out-dir", str(root), "--pod-name-prefix", name, "--resource-name-exact", name,
        "--deadline-epoch", str(deadline)]
    loading = guard.GpuInstance(provider="vast", id="101", name=name, state="loading", booted=False,
        live=True, age_seconds=588)
    for process, now, expected in (
        (shlex.join(command), time.time(), {"101"}),
        (shlex.join(command), deadline + 1, set()),
        ("unrelated", time.time(), set()),
        (shlex.join([item.replace(name, PREFIX + "f" * 32) for item in command]), time.time(), set()),
    ):
        protected = guard.find_protected_pod_ids([tmp_path], process_cmdlines=[process], now=now)
        assert protected == expected
        assert guard.is_reapable(loading, max_boot_seconds=480, protected_ids=protected) is (not bool(expected))
    # The native launcher explicitly grants the admitted 30-minute image startup
    # window; the shared helper must preserve it for the dedicated G1 kind.
    assert cold_pull_aware_heartbeat_no_progress_seconds(configured_seconds=1800,
        provider_bundle_kind="native_g1_beta_screening", allow_cold_image_pull=True,
        min_cold_image_pull_live_minutes=30, startup_timeout_seconds=1800, max_live_minutes=30) == 1800


def test_g1_name_survives_actual_parent_child_adapter_and_fleet_contracts(tmp_path, monkeypatch):
    exercise_native_g1_watchdog(tmp_path, monkeypatch, PREFIX + "a" * 32)


@pytest.mark.parametrize("suffix", ["", "a" * 24, "a" * 31, "a" * 33, "A" * 32, "z" * 32])
def test_g1_invalid_exact_identity_cannot_arm_parent_or_child(tmp_path, suffix):
    name = PREFIX + suffix
    with pytest.raises(ValueError):
        control.validate_independent_vast_watchdog_names(pod_name_prefix=PREFIX, resource_name_exact=name)
    with pytest.raises(ValueError):
        watchdog.arm_watchdog(out_dir=tmp_path, pod_name_prefix=name, resource_name_exact=name,
            deadline_epoch=time.time() + 1800, provider_name="vast")
