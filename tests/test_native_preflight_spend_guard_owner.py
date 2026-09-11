"""Canonical native preflight names must enter the existing exact-watchdog owner guard."""

import json

import pytest

from blueprint_pipeline import native_task_arena_vast as native
from scripts import gpu_spend_guard as guard


def prefix_from_producer(tmp_path, monkeypatch):
    observed = {}
    monkeypatch.setattr(
        native, "run_arena_native_control_vast", lambda **kwargs: observed.update(kwargs) or {}
    )
    native.run_native_task_arena_runtime_preflight_vast(
        job_dir=tmp_path,
        execute=False,
        prepared_bundle={
            "schema_version": "native_task_arena_provider_bundle.v1",
            "execution_mode": "runtime_preflight",
            "policy_candidate_id": None,
            "candidate_policy_queried": False,
            "expected_output_filename": "native_task_arena_runtime_preflight.v1.json",
            "container_image": "test",
        },
        paid_resource_admission_grant=None,
        hard_cap_usd=0.75,
        hard_ttl_seconds=1800,
        max_hourly_rate_usd=0.8,
    )
    assert observed["hard_cap_usd"] == 0.75 and observed["hard_ttl_seconds"] == 1800
    return observed["instance_label_prefix"]


@pytest.mark.parametrize(
    "fault",
    [None, "old_prefix", "dead_owner", "expired", "cancelled", "wrong_deadline", "wrong_directory"],
)
def test_retained_v28b_owner_lookup_preserves_orphan_rule(tmp_path, monkeypatch, fault):
    # Retained V28b ID/deadline/prefix suffix and observed age at the reaper event.
    # Its live armed state is reconstructed offline; terminal records stay untouched.
    prefix = prefix_from_producer(tmp_path, monkeypatch) + "20260910t233211093709000-"
    if fault == "old_prefix":
        prefix = "blueprint-native-task-arena-preflight-20260910t233211093709000-"
    root = tmp_path / "allocator/attempts/attempt_001/independent_vast_watchdog"
    root.mkdir(parents=True)
    deadline = 1789084952.574677
    now = 1789083742.0
    record = {
        "schema_version": "groot_oscar_runpod_canary_watchdog.v1",
        "status": "armed",
        "provider": "vast",
        "independent_process": True,
        "pre_deadline_provider_mutation_allowed": False,
        "provider_mutation_trigger": "hard_deadline_only",
        "watchdog_out_dir": str(root.resolve()),
        "deadline_epoch": deadline,
        "pod_name_prefix": prefix,
        "name_prefix": prefix,
    }
    (root / "groot_oscar_runpod_canary_watchdog.json").write_text(json.dumps(record))
    (root / "started_vast_instance_id.txt").write_text("50534312\n")
    command = f"python -m blueprint_pipeline.groot_oscar_runpod_watchdog --provider vast --out-dir {root} --pod-name-prefix {prefix} --deadline-epoch {deadline}"
    if fault == "dead_owner":
        command = "unrelated-process"
    elif fault == "expired":
        now = deadline + 1
    elif fault == "cancelled":
        (root / "groot_oscar_runpod_canary_watchdog_cancel.json").write_text("{}")
    elif fault == "wrong_deadline":
        command = command.replace(str(deadline), str(deadline + 1))
    elif fault == "wrong_directory":
        command = command.replace(str(root), str(root.parent / "unrelated"))
    protected = guard.find_protected_pod_ids([tmp_path], process_cmdlines=[command], now=now)
    assert protected == ({"50534312"} if fault is None else set())
    instance = guard.GpuInstance(
        provider="vast",
        id="50534312",
        name=prefix,
        state="loading",
        booted=False,
        live=True,
        age_seconds=554.0,
        cost_per_hr=0.8,
    )
    reason = guard.reap_candidate_reason(instance, max_boot_seconds=480, protected_ids=protected)
    assert reason == (None if fault is None else "unbooted_dud_past_boot_ttl")


def test_real_watchdog_subprocess_arms_is_owned_and_cancels_without_allocation(
    tmp_path, monkeypatch
):
    import shlex
    import time
    from pathlib import Path
    from blueprint_pipeline import vast_independent_watchdog_control as control

    prefix = prefix_from_producer(tmp_path, monkeypatch)
    monkeypatch.setenv("PYTHONPATH", str(Path(native.__file__).resolve().parents[1]))
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path / "real-process",
        max_live_minutes=30,
        generated_at="2026-09-11T00:00:00+00:00",
        pod_name_prefix=prefix,
        startup_wait_seconds=15,
    )
    assert handle is not None, handoff
    try:
        assert handoff["status"] == "armed" and handle.process.poll() is None
        armed = json.loads((handle.out_dir / control.EVIDENCE_NAME).read_text())
        assert armed["pid"] == handle.process.pid
        assert not handle.started_instance_id_path.exists()
        # This is a local owner-lookup fixture, not a provider allocation.
        handle.started_instance_id_path.write_text("50534312\n")
        protected = guard.find_protected_pod_ids(
            [tmp_path], process_cmdlines=[shlex.join(handle.process.args)], now=time.time()
        )
        assert protected == {"50534312"}
    finally:
        handle.started_instance_id_path.unlink(missing_ok=True)
        closed = control.close_independent_vast_watchdog(
            job_dir=tmp_path / "real-process",
            handle=handle,
            instance_ids=[],
            provider_teardown_completed=True,
            provider_allocation_impossible=True,
        )
    assert closed["status"] == "cancelled_no_allocation"
    assert closed["provider_mutations_performed"] == 0
    assert handle.process.poll() is not None


def test_unregistered_namespace_refuses_before_creating_attempt_watchdog_directory(tmp_path):
    from blueprint_pipeline import vast_independent_watchdog_control as control

    root = tmp_path / "never-created"
    with pytest.raises(ValueError, match="watchdog_pod_name_prefix_not_canary_scoped"):
        control.arm_independent_vast_watchdog(
            job_dir=root,
            max_live_minutes=30,
            generated_at="2026-09-11T00:00:00+00:00",
            pod_name_prefix="blueprint-task-evaluation-unregistered-probe-",
        )
    assert not root.exists()
