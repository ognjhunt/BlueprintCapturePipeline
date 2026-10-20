"""No network or package mutations: exercise the narrow installer contract."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from blueprint_pipeline import flora_dependency_bootstrap as bootstrap


def _versions():
    return {"vllm": "0.24.0", "torch": "2.10.0", "torchvision": "0.25.0", "torchaudio": None,
            **{name: spec["version"] for name, spec in bootstrap.ADDONS.items()}}


def _fake_process(monkeypatch, *, before=None, after=None, exit_code=0, timeout=False):
    calls = []
    initial = _versions() if before is None else before
    final = _versions() if after is None else after
    probes = 0
    def run(argv, **kwargs):
        nonlocal probes
        calls.append(argv)
        if "install" not in argv:
            values = initial if probes == 0 else final
            probes += 1
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"versions": values,
                "python_executable": argv[0], "python_version": "3.12.11", "platform": "linux", "machine": "x86_64"}))
        kwargs["stdout"].write("Hermetic installer fixture; no network or environment change.\n")
        if timeout:
            raise subprocess.TimeoutExpired(argv, 600)
        return SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    return calls


def test_satisfied_environment_never_invokes_pip_or_network(tmp_path, monkeypatch):
    calls = _fake_process(monkeypatch)
    report = bootstrap.ensure_flora_runtime_dependencies(output_root=tmp_path, python_executable="/sealed/python")
    assert len(calls) == 1 and calls[0][1:4] == ["-I", "-B", "-c"]
    assert report["status"] == "ready"
    assert report["install_attempted"] is False
    assert report["network_install_attempted"] is False
    assert report["installed_packages"] == []
    assert report["before"] == report["after"]
    assert json.loads(Path(report["receipt_path"]).read_text()) == report
    assert report["model_inference_performed"] is False


@pytest.mark.parametrize("base_version", [None, "0.23.0", "0.24.0+cu130"])
def test_wrong_base_vllm_blocks_before_install(tmp_path, monkeypatch, base_version):
    before = _versions()
    before.update(vllm=base_version, transformers=None)
    calls = _fake_process(monkeypatch, before=before)
    with pytest.raises(bootstrap.FloraDependencyBootstrapError, match="base_vllm_version_mismatch") as caught:
        bootstrap.ensure_flora_runtime_dependencies(output_root=tmp_path)
    assert len(calls) == 1 and "install" not in calls[0]
    assert caught.value.receipt["install_attempted"] is False
    assert Path(caught.value.receipt["receipt_path"]).is_file()


@pytest.mark.parametrize("package", list(bootstrap.ADDONS))
def test_installs_only_required_addon_with_exact_hash_and_no_dependency_resolution(tmp_path, monkeypatch, package):
    before = _versions()
    before[package] = "outdated-version"
    calls = _fake_process(monkeypatch, before=before)
    report = bootstrap.ensure_flora_runtime_dependencies(output_root=tmp_path, python_executable="/sealed/python")
    install = next(command for command in calls if "install" in command)
    for flag in ("--isolated", "--no-deps", "--only-binary=:all:", "--require-hashes", "--no-input"):
        assert flag in install
    assert install[:6] == ["/sealed/python", "-I", "-B", "-m", "pip", "--isolated"]
    requirements = Path(install[install.index("--requirement") + 1]).read_text()
    spec = bootstrap.ADDONS[package]
    assert requirements == f"{package}=={spec['version']} --hash=sha256:{spec['sha256']}\n"
    assert "torch" not in requirements and "vllm" not in requirements
    assert report["installed_packages"] == [package]
    assert all(report["before"]["versions"][name] == report["after"]["versions"][name]
               for name in bootstrap.PROTECTED_PACKAGES)


def test_wheel_hashes_are_exact_approved_artifacts():
    assert bootstrap.ADDONS == {
        "transformers": {"version": "5.8.1", "sha256": "5340fb95962162cdfdae5cc91d7f8fedd92ed75216c1154c5e1f590fcf56dd0e"},
        "qwen-vl-utils": {"version": "0.0.14", "sha256": "5e28657bfd031e56bd447c5901b58ddfc3835285ed100f4c56580e0ade054e96"},
        "decord": {"version": "0.6.0", "sha256": "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"},
    }


def test_wrong_local_version_reinstalls_only_its_exact_approved_wheel(tmp_path, monkeypatch):
    before = _versions()
    before["transformers"] = "5.8.1+local"
    calls = _fake_process(monkeypatch, before=before)
    report = bootstrap.ensure_flora_runtime_dependencies(output_root=tmp_path)
    command = next(row for row in calls if "install" in row)
    assert "--force-reinstall" in command and "--no-deps" in command
    requirements = Path(command[command.index("--requirement") + 1]).read_text().splitlines()
    assert requirements == ["transformers==5.8.1 --hash=sha256:" + bootstrap.ADDONS["transformers"]["sha256"]]
    assert report["installed_packages"] == ["transformers"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "wrong_addon", "torch_changed", "vllm_changed"])
def test_failure_propagates_with_logs_and_post_attempt_version_receipt(tmp_path, monkeypatch, failure):
    before = _versions()
    before["transformers"] = None
    after = _versions()
    if failure == "wrong_addon":
        after["transformers"] = "5.8.0"
    if failure == "torch_changed":
        after["torch"] = "2.11.0"
    if failure == "vllm_changed":
        after["vllm"] = "0.25.0"
    calls = _fake_process(monkeypatch, before=before, after=after,
                          exit_code=1 if failure == "exit" else 0, timeout=failure == "timeout")
    with pytest.raises(bootstrap.FloraDependencyBootstrapError) as caught:
        bootstrap.ensure_flora_runtime_dependencies(output_root=tmp_path)
    report = caught.value.receipt
    assert report["status"] == "blocked" and report["blockers"]
    assert report["install_attempted"] is True
    assert report["after"]["versions"] == after
    assert len([command for command in calls if "install" in command]) == 1
    artifact_names = {Path(record["path"]).name for record in report["artifacts"]}
    assert {"install.log", "requirements.txt", "versions_before.log", "versions_after.log"} <= artifact_names
    assert json.loads(Path(report["receipt_path"]).read_text()) == report
