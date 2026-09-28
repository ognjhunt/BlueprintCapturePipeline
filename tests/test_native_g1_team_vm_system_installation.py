"""Offline installation proof stops before any unsupported runtime claim."""

import importlib
import json
import subprocess

import pytest


def load():
    return importlib.import_module("blueprint_pipeline.native_g1_team_vm_system_installation")


@pytest.mark.parametrize("fault", [None, "install", "package", "module", "help", "namespace", "configure", "restart", "runtimes"])
def test_installation_observes_actual_boundaries_and_stops_on_failure(tmp_path, monkeypatch, fault):
    installation = load()
    monkeypatch.setattr(installation, "require_cpu_guest", lambda: None)
    packages = {"fixture": {"package": "fixture", "version": "1"}}
    calls = []
    outputs = {
        "offline_install": "installed",
        "installed_packages": "install ok installed\tfixture\t1\n",
        "kernel_module": "580.65.06\n",
        "bwrap_help": " ".join(installation.BWRAP_OPTIONS),
        "archive_namespace": json.dumps({"uid": 65534, "gid": 65534, "cap_eff": "0000000000000000", "interfaces": ["lo"]}),
        "toolkit_configure": "configured",
        "docker_restart": "",
        "docker_runtimes": '{"runc":{},"nvidia":{}}',
    }
    stages = list(outputs)
    if fault in ("package", "module", "help", "namespace", "runtimes"):
        stage = {"package": "installed_packages", "module": "kernel_module", "help": "bwrap_help",
                 "namespace": "archive_namespace", "runtimes": "docker_runtimes"}[fault]
        outputs[stage] = "foreign"

    def execute(argv, *, stdout, stderr, timeout, check, env, **kwargs):
        stage = stages[len(calls)]
        calls.append((stage, argv, kwargs))
        stdout.write(outputs[stage].encode())
        failed = {"install": "offline_install", "configure": "toolkit_configure", "restart": "docker_restart"}.get(fault)
        return subprocess.CompletedProcess(argv, 1 if stage == failed else 0)

    monkeypatch.setattr(installation.subprocess, "run", execute)
    verified = []
    helpers = {
        "SYSTEM_PACKAGES": packages,
        "KERNEL": "5.15.0-1067-kvm",
        "offline_apt_simulation_command": lambda root, **kwargs: verified.append((root, kwargs)) or
        ["apt-get", "--simulate", "--no-download", "--no-install-recommends", "install", str(root / "exact.deb")],
    }
    result = installation.rehearse_offline_installation(tmp_path, implementation_commit="a" * 40, system_helpers=helpers)
    assert verified == [(tmp_path, {"expected_implementation_commit": "a" * 40})]
    assert "--simulate" not in calls[0][1] and "--no-download" in calls[0][1] and "--yes" in calls[0][1]
    assert result["gpu_runtime_qualified"] is False and result["provider_mutation_performed"] is False
    assert result["policy_inference_performed"] is False
    assert result["runtime_installation_attempted"] is True
    if fault:
        failed_stage = {"install": "offline_install", "package": "installed_packages", "module": "kernel_module",
                        "help": "bwrap_help", "namespace": "archive_namespace", "configure": "toolkit_configure",
                        "restart": "docker_restart", "runtimes": "docker_runtimes"}[fault]
        assert result["status"] == "blocked" and result["stage"] == failed_stage
        assert len(calls) == stages.index(failed_stage) + 1
        assert result["probes"][failed_stage]["stdout"] == outputs[failed_stage]
    else:
        assert result["status"] == "cpu_installation_observed" and len(calls) == len(stages)
        namespace = calls[4]
        assert "--ro-bind-fd" in namespace[1] and namespace[2]["pass_fds"]
        assert "--unshare-all" in namespace[1] and "--cap-drop" in namespace[1]
        assert calls[-3][1] == ["nvidia-ctk", "runtime", "configure", "--runtime=docker"]
        assert all("pull" not in call[1] for call in calls)


@pytest.mark.parametrize("fault", ["context", "packages"])
def test_context_and_package_binding_refuse_before_any_mutation(tmp_path, monkeypatch, fault):
    installation = load()
    calls = []

    def refuse():
        if fault == "context":
            raise ValueError("g1_vm_system_install_guest_context_invalid")

    def verify(*args, **kwargs):
        raise ValueError("g1_vm_system_package_hash_invalid")

    monkeypatch.setattr(installation, "require_cpu_guest", refuse)
    monkeypatch.setattr(installation.subprocess, "run", lambda *a, **k: calls.append(a))
    result = installation.rehearse_offline_installation(tmp_path, implementation_commit="a" * 40,
                                                       system_helpers={"offline_apt_simulation_command": verify})
    assert result["status"] == "blocked" and not calls
    assert result["runtime_installation_attempted"] is False
    assert result["blocker_code"].startswith("g1_vm_system_")


def test_actual_host_cannot_be_installed_by_accidental_import_or_invocation(tmp_path):
    installation = load()
    # This check performs metadata reads only; the test host has no matching VM seed/network/kernel.
    with pytest.raises(ValueError, match="guest_context"):
        installation.require_cpu_guest()


@pytest.mark.parametrize("fault", ["uid", "gid", "caps", "network", "extra"])
def test_namespace_claim_is_independently_rejected(fault):
    installation = load()
    observed = {"uid": 65534, "gid": 65534, "cap_eff": "0000000000000000", "interfaces": ["lo"]}
    if fault == "uid":
        observed["uid"] = 0
    elif fault == "gid":
        observed["gid"] = 0
    elif fault == "caps":
        observed["cap_eff"] = "0000000000000001"
    elif fault == "network":
        observed["interfaces"] = ["lo", "eth0"]
    else:
        observed["extra"] = True
    with pytest.raises(ValueError, match="namespace"):
        installation.validate_namespace_observation(json.dumps(observed))
