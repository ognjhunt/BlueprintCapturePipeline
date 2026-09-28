"""Reject an installed sandbox missing the archive launcher's actual features."""

import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline import native_g1_team_vm_host as host


OPTIONS = ("--cap-drop", "--chdir", "--clearenv", "--dev", "--dev-bind", "--die-with-parent",
           "--dir", "--gid", "--new-session", "--proc", "--ro-bind", "--ro-bind-fd", "--setenv",
           "--tmpfs", "--uid", "--unshare-all", "--unsetenv")


@pytest.mark.parametrize("fault", [None, "ro_bind_fd", "clearenv", "substring", "nonzero"])
def test_archive_preflight_requires_exact_installed_flags_before_device_observation(monkeypatch, fault):
    monkeypatch.setattr(host.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host.os, "geteuid", lambda: 0)
    monkeypatch.setattr(host.sys, "version_info", (3, 12))
    monkeypatch.setattr(host, "version", lambda name: {"numpy": "2.3.1", "rfc8785": "0.1.4"}[name])
    monkeypatch.setattr(host, "_verified_local_image", lambda reference: "sha256:" + "c" * 64)
    observed = []
    monkeypatch.setattr(host, "observe_archive_gpu_binding", lambda **kwargs: observed.append(kwargs) or {"fixture": True})
    commands = []

    def invoke(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "info"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"nvidia": {}}))
        if command[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="0, 580.65.06")
        if command == ["bwrap", "--version"]:
            return SimpleNamespace(returncode=0, stdout="bubblewrap 0.9.0")
        assert command == ["bwrap", "--help"]
        options = list(OPTIONS)
        if fault in {"ro_bind_fd", "substring"}:
            options.remove("--ro-bind-fd")
        if fault == "substring":
            options.append("--fake--ro-bind-fd")
        if fault == "clearenv":
            options.remove("--clearenv")
        return SimpleNamespace(returncode=1 if fault == "nonzero" else 0,
                               stdout="\n".join("    " + value + " ARG  explanation" for value in options))

    monkeypatch.setattr(host.subprocess, "run", invoke)
    packet = {"packet_digest": "sha256:" + "a" * 64,
              "request": {"policy_profile": {"profile_digest": "sha256:" + "b" * 64,
                                            "delivery": {"mode": "noncontainer_artifact"}}}}
    if fault:
        with pytest.raises(ValueError, match="archive_sandbox_features"):
            host.preflight_g1_vm_host(packet)
        assert observed == []
    else:
        result = host.preflight_g1_vm_host(packet)
        assert result["archive_sandbox_required_features"] == list(OPTIONS)
        assert len(observed) == 1
    assert ["bwrap", "--help"] in commands
