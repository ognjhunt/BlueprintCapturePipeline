"""Development-only request/allocator/watchdog/cleanup integration; no provider calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_preparation_contract as contract
from blueprint_pipeline import deployment_embodiment_preparation_lane as lane
from blueprint_pipeline import groot_oscar_runpod_watchdog as watchdog
from blueprint_pipeline import vast_independent_watchdog_control as control
from blueprint_pipeline import vast_provider_adapter as adapter
from scripts import gpu_spend_guard as guard
from tests.test_embodiment_preparation_lane import COMMIT, _dispatch_fixture

PREFIX = contract.INSTANCE_PREFIX


@pytest.fixture
def child_arm(monkeypatch):
    """Replace only process creation; run the actual child arming/receipt validator."""
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv(control.CALLER_EXIT_SURVIVAL_ENV, raising=False)
    calls = []
    def popen(command, **kwargs):
        assert kwargs["start_new_session"] is True
        def value(flag):
            return command[command.index(flag) + 1]
        armed = watchdog.arm_watchdog(out_dir=value("--out-dir"),
            pod_name_prefix=value("--pod-name-prefix"), resource_name_exact=value("--resource-name-exact"),
            deadline_epoch=float(value("--deadline-epoch")), provider_name=value("--provider"), pid=4321)
        calls.append((command, armed))
        return SimpleNamespace(pid=4321, poll=lambda: None)
    monkeypatch.setattr(control.subprocess, "Popen", popen)
    return calls


def test_production_authorization_name_uses_standard_32_hex_and_real_watchdog_validators(tmp_path):
    authorization = "public-robot-imports-20261030-01"
    name = contract.resource_name({"authorization_id": authorization})
    assert name == PREFIX + hashlib.sha256(authorization.encode()).hexdigest()[:32]
    assert len(PREFIX) == 33 and len(name) == 65
    assert control.validate_independent_vast_watchdog_names(
        pod_name_prefix=PREFIX, resource_name_exact=name) == (PREFIX, name)
    assert watchdog.arm_watchdog(out_dir=tmp_path, pod_name_prefix=name, resource_name_exact=name,
        deadline_epoch=time.time() + 180, provider_name="vast")["resource_name_exact"] == name
    # The previous failed request had no allocation; it is not a legacy name exception.
    with pytest.raises(ValueError, match="exact_resource_name_invalid"):
        control.validate_independent_vast_watchdog_names(pod_name_prefix=PREFIX, resource_name_exact=name[:-8])


def test_real_adapter_dry_run_accepts_canonical_name_without_api_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "_api_json", lambda **kwargs: pytest.fail("provider API call"))
    name = contract.resource_name({"authorization_id": "development-only"})
    result = adapter.run_vast_provider_adapter(job_dir=tmp_path, mode="dry-run",
        instance_label_prefix=PREFIX, instance_label_exact=name)
    assert result["status"] == "dry_run_ready" and result["api_call_performed"] is False


def test_request_to_allocator_to_independent_child_and_exact_cleanup(tmp_path, monkeypatch, child_arm):
    grant, _, args, trusted = _dispatch_fixture(tmp_path, monkeypatch)
    name = contract.resource_name(grant)
    monkeypatch.setattr(lane, "require_paid_resource_admission", lambda *a, **k: "development-only")
    def transport(**kwargs):
        assert kwargs["instance_label_exact"] == name and kwargs["instance_label_prefix"] == PREFIX
        assert kwargs["require_independent_watchdog"] is True
        handoff, handle = control.arm_independent_vast_watchdog(job_dir=kwargs["job_dir"],
            max_live_minutes=kwargs["hard_ttl_seconds"] // 60, generated_at="development-only",
            pod_name_prefix=kwargs["instance_label_prefix"], resource_name_exact=kwargs["instance_label_exact"],
            deadline_cap_epoch=kwargs["hard_deadline_epoch"])
        assert handoff["status"] == "armed" and handle.resource_name_exact == name
        command, armed = child_arm[0]
        assert armed["resource_name_exact"] == name == armed["pod_name_prefix"]
        assert armed["deadline_epoch"] <= min(grant["expires_at_epoch"], time.time() + grant["ttl_seconds"])
        (handle.out_dir / "started_vast_instance_id.txt").write_text("101")
        assert guard.find_protected_pod_ids([kwargs["job_dir"]], process_cmdlines=[shlex.join(command)]) == {"101"}
        # Inventory includes a same-family sibling and a prefix lookalike. Neither is owned.
        rows = [{"id": identity, "label": label, "actual_status": "running"} for identity, label in
                ((101, name), (102, name + "-extra"), (103, PREFIX + "b" * 32), (104, "unrelated"))]
        def api_json(**request):
            assert request["method"] == "GET" and request["path"] == "/instances/"
            return 200, {"instances": list(rows)}
        monkeypatch.setattr(adapter, "_api_json", api_json)
        class Provider:
            def _key(self):
                return "development-only-fake-key"
            def terminate(self, identity):
                assert identity == "101"
                rows[0]["actual_status"] = "destroyed"
                return {"status": "stopped", "http": 204}
            def inspect(self, identity):
                assert identity == "101"
                return {"status": "absent", "instance_id": identity, "api_confirmed": True,
                        "provider_absence_confirmed": True, "http": 404}
        result = watchdog.terminate_canary_resources(provider=Provider(), provider_name="vast",
            pod_name_prefix=name, resource_name_exact=name, armed=armed)
        assert result["provider_absence_confirmed"] is True
        assert [item["instance_id"] for item in result["terminations"]] == ["101"]
        assert all(row["actual_status"] == "running" for row in rows[1:])
        return {"status": "completed"}
    assert lane.dispatch(args, ([], {"orchestrator_source_commit": COMMIT}),
                         transport=transport, trusted_root=trusted) == 0
    admission = contract.read(Path(args.admission_out))
    assert admission["allocation_binding"]["resource_name"] == name


@pytest.mark.parametrize("name", ["", PREFIX, PREFIX + "a" * 24, PREFIX + "a" * 31,
    PREFIX + "a" * 33, PREFIX + "A" * 32, PREFIX + "z" * 32, PREFIX + "other-" + "a" * 32,
    "blueprint-embodiment-preparations-" + "a" * 32])
def test_invalid_preparation_identity_refused_before_parent_or_child_arms(tmp_path, name):
    with pytest.raises(ValueError):
        control.validate_independent_vast_watchdog_names(pod_name_prefix=PREFIX, resource_name_exact=name)
    with pytest.raises(ValueError):
        watchdog.arm_watchdog(out_dir=tmp_path, pod_name_prefix=name or PREFIX,
                              resource_name_exact=name or None, deadline_epoch=time.time() + 180)


@pytest.mark.parametrize("prefix", [PREFIX + "extra-", "blueprint-native-task-policy-"])
def test_preparation_exact_name_cannot_borrow_another_parent_scope(prefix):
    with pytest.raises(ValueError, match="exact_resource_name_invalid"):
        control.validate_independent_vast_watchdog_names(pod_name_prefix=prefix, resource_name_exact=PREFIX + "a" * 32)


@pytest.mark.parametrize("prefix,exact", [(PREFIX, None),
    ("blueprint-native-task-policy-", PREFIX + "a" * 32)])
def test_preparation_cleanup_refuses_broad_scope_before_provider_access(prefix, exact):
    with pytest.raises(ValueError, match="exact_resource_name_invalid"):
        watchdog.terminate_canary_resources(provider=object(), provider_name="vast", pod_name_prefix=prefix,
                                            resource_name_exact=exact, armed={})


@pytest.mark.parametrize("fault", [None, "expired", "dead_owner", "missing_exact", "wrong_exact",
    "missing_command_exact", "wrong_command_exact", "old_24_hex", "terminal", "wrong_deadline", "cancelled"])
def test_fleet_guard_protects_only_exact_live_preparation_watchdog(tmp_path, fault):
    root = tmp_path / "embodiment-preparation/run/allocator/attempts/attempt_001/independent_vast_watchdog"
    name, deadline = PREFIX + "a" * 32, time.time() + 180
    record = watchdog.arm_watchdog(out_dir=root, pod_name_prefix=name, resource_name_exact=name,
                                   deadline_epoch=deadline, provider_name="vast")
    (root / "started_vast_instance_id.txt").write_text("101")
    command = ["python", "-m", "blueprint_pipeline.groot_oscar_runpod_watchdog", "--provider", "vast",
               "--out-dir", str(root), "--pod-name-prefix", name, "--resource-name-exact", name,
               "--deadline-epoch", str(deadline)]
    now = time.time()
    if fault == "expired":
        now = deadline + 1
    elif fault == "dead_owner":
        command = ["unrelated"]
    elif fault == "missing_exact":
        record.pop("resource_name_exact")
    elif fault == "wrong_exact":
        record["resource_name_exact"] = PREFIX + "b" * 32
    elif fault == "missing_command_exact":
        index = command.index("--resource-name-exact")
        del command[index:index + 2]
    elif fault == "wrong_command_exact":
        command[command.index("--resource-name-exact") + 1] = PREFIX + "b" * 32
    elif fault == "old_24_hex":
        record.update(resource_name_exact=name[:-8], pod_name_prefix=name[:-8], name_prefix=name[:-8])
        command = [item.replace(name, name[:-8]) for item in command]
    elif fault == "terminal":
        record["status"] = "provider_terminal"
    elif fault == "wrong_deadline":
        command[-1] = str(deadline + 1)
    elif fault == "cancelled":
        (root / "groot_oscar_runpod_canary_watchdog_cancel.json").write_text("{}")
    (root / watchdog.EVIDENCE_NAME).write_text(json.dumps(record))
    protected = guard.find_protected_pod_ids([tmp_path], process_cmdlines=[shlex.join(command)], now=now)
    assert protected == ({"101"} if fault is None else set())
    loading = guard.GpuInstance(provider="vast", id="101", name=name, state="loading", booted=False,
                                live=True, age_seconds=588)
    assert guard.is_reapable(loading, max_boot_seconds=480, protected_ids=protected) is (fault is not None)


def test_deployed_fleet_guard_scans_only_fixed_preparation_workspace_addition():
    unit = (Path(__file__).parents[1] / "deploy/systemd/blueprint-gpu-spend-guard.service").read_text()
    assert "--embodiment-preparation-watchdogs " in unit
    assert "--output-root /mnt/blueprint-work/embodiment-preparation" not in unit
    assert "ReadOnlyPaths=-/mnt/blueprint-work/embodiment-preparation\n" in unit
    assert "--output-root /mnt " not in unit


def test_source_cache_record_cannot_borrow_live_allocator_watchdog_ownership(tmp_path, monkeypatch):
    workspace = tmp_path / "embodiment-preparation"
    owner = workspace / "development-only/allocator/attempts/attempt_001/independent_vast_watchdog"
    cache = workspace / "development-only/sources/cache/independent_vast_watchdog"
    name, deadline = PREFIX + "a" * 32, time.time() + 180
    for root, identity in ((owner, "101"), (cache, "202")):
        watchdog.arm_watchdog(out_dir=root, pod_name_prefix=name, resource_name_exact=name,
                             deadline_epoch=deadline, provider_name="vast")
        (root / "started_vast_instance_id.txt").write_text(identity)
    command = ["python", "-m", "blueprint_pipeline.groot_oscar_runpod_watchdog", "--provider", "vast",
               "--out-dir", str(owner), "--pod-name-prefix", name, "--resource-name-exact", name,
               "--deadline-epoch", str(deadline)]
    monkeypatch.setattr(guard, "EMBODIMENT_PREPARATION_WORK_ROOT", workspace)
    (cache / "warm_serve_pod.json").write_text(json.dumps({"pod_id": "303", "status": "serving",
                                                        "lease_expires_at": "2099-01-01T00:00:00Z"}))
    # Even a process that names a cached fake watchdog is outside the admitted layout.
    fake_command = [item.replace(str(owner), str(cache)) for item in command]
    assert guard.find_embodiment_preparation_watchdog_ids(
        process_cmdlines=[shlex.join(command), shlex.join(fake_command)], now=time.time()) == {"101"}
    assert guard.find_embodiment_preparation_watchdog_ids(process_cmdlines=[], now=time.time()) == set()
    assert guard.find_expected_serve_pod_ids([workspace]) == {"303"}  # why this tree must not enter the legacy reader



def test_guard_main_keeps_preparation_cache_out_of_legacy_warm_scan(tmp_path, monkeypatch):
    workspace, legacy = tmp_path / "preparation", tmp_path / "legacy"
    legacy.mkdir()
    owner = workspace / "development-only/allocator/attempts/attempt_001/independent_vast_watchdog"
    name, deadline = PREFIX + "a" * 32, time.time() + 180
    watchdog.arm_watchdog(out_dir=owner, pod_name_prefix=name, resource_name_exact=name,
                         deadline_epoch=deadline, provider_name="vast")
    (owner / "started_vast_instance_id.txt").write_text("101")
    cache = workspace / "development-only/source-cache"
    cache.mkdir()
    (cache / "warm_serve_pod.json").write_text(json.dumps({"pod_id": "202", "status": "serving",
                                                        "lease_expires_at": "2099-01-01T00:00:00Z"}))
    command = ["python", "-m", "blueprint_pipeline.groot_oscar_runpod_watchdog", "--provider", "vast",
               "--out-dir", str(owner), "--pod-name-prefix", name, "--resource-name-exact", name,
               "--deadline-epoch", str(deadline)]
    monkeypatch.setattr(guard, "EMBODIMENT_PREPARATION_WORK_ROOT", workspace)
    monkeypatch.setattr(guard, "list_process_cmdlines", lambda: [shlex.join(command)])
    monkeypatch.setattr(guard, "_refresh_gcp_monthly_budget", lambda: None)
    monkeypatch.setattr(guard, "_read_secret", lambda *a, **k: None)
    monkeypatch.setattr(guard, "cloud_provider_configured", lambda *a: False)
    monkeypatch.setattr(guard, "_http_request", lambda *a, **k: pytest.fail("provider request"))
    observed = []
    report = guard.build_report
    def capture(*args, **kwargs):
        observed.append(kwargs)
        return report(*args, **kwargs)
    monkeypatch.setattr(guard, "build_report", capture)
    # No credentials means no provider query/admission; ownership evaluation is still real.
    assert guard.main(["--output-root", str(legacy), "--embodiment-preparation-watchdogs"]) == 2
    assert observed[-1]["protected_ids"] == {"101"} and observed[-1]["serve_pod_ids"] == set()
    assert guard.main(["--output-root", str(legacy)]) == 2
    assert observed[-1]["protected_ids"] == set()
