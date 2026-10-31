"""Capture transfer must have a working scanner before a release activates."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import capture_upload_runtime as runtime
from blueprint_pipeline.capture_upload_runtime import (
    CaptureUploadRuntimeError,
    verify_capture_upload_runtime,
)
from blueprint_pipeline.control_plane_storage_roots import classify_path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("capture_runtime_deploy", ROOT / "scripts/deploy_control_plane_commit.py")
deploy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(deploy)


def _scanner(tmp_path, mode="eicar"):
    import sys
    program = tmp_path / "scanner"
    program.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, pathlib, sys\n"
        + ("sys.exit(1 if hashlib.md5(pathlib.Path(sys.argv[-1]).read_bytes()).hexdigest()"
           " == '44d88612fea8a8f36de82e1278abb02f' else 0)\n" if mode == "eicar"
           else f"sys.exit({mode})\n")
    )
    program.chmod(0o700)
    return program


@pytest.mark.slow
def test_scanner_controls_run_and_are_cleaned_up(tmp_path):
    program = _scanner(tmp_path)
    result = verify_capture_upload_runtime(store_root=str(tmp_path), scanner_argv=[str(program)])
    assert result["controls"]["clean"]["exit_code"] == 0
    assert result["controls"]["known_test_signature"]["exit_code"] == 1
    assert result["scanner_program_sha256"] == hashlib.sha256(program.read_bytes()).hexdigest()
    assert result["service_uid"] == os.geteuid()
    assert not list(tmp_path.glob(".scanner-preflight-*"))
    assert result["source_footage_processed"] is False
    assert result["robot_episode_executed"] is False
    for control in result["controls"].values():
        evidence = tmp_path / control["evidence_relative_path"]
        assert evidence.stat().st_mode & 0o777 == 0o600
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == control["evidence_sha256"]


@pytest.mark.parametrize("mode,control", [(0, "known_test_signature"), (1, "clean"), (2, "clean")])
@pytest.mark.slow
def test_non_discriminating_or_failed_scanner_refuses(tmp_path, mode, control):
    program = _scanner(tmp_path, mode)
    with pytest.raises(CaptureUploadRuntimeError, match="scanner_control_failed:" + control) as failure:
        verify_capture_upload_runtime(store_root=str(tmp_path), scanner_argv=[str(program)])
    evidence = tmp_path / failure.value.diagnostic["evidence_relative_path"]
    assert json.loads(evidence.read_text())["exit_code"] == mode
    assert not list(tmp_path.glob(".scanner-preflight-*"))


def test_a_loaded_host_gets_time_to_scan_and_a_hung_scanner_still_refuses(tmp_path, monkeypatch):
    """A 60 s limit blocked a deploy on 2026-10-31: one scan outlived it at load average 8."""
    limits = []

    def hung(argv, **kwargs):
        limits.append(kwargs["timeout"])
        raise runtime.subprocess.TimeoutExpired(argv, kwargs["timeout"])

    program = _scanner(tmp_path)
    monkeypatch.setattr(runtime.subprocess, "run", hung)
    with pytest.raises(CaptureUploadRuntimeError, match="^capture_transfer_scanner_timed_out$"):
        verify_capture_upload_runtime(store_root=str(tmp_path), scanner_argv=[str(program)])
    assert limits == [runtime.SCANNER_CONTROL_TIMEOUT_SECONDS]
    assert not list(tmp_path.glob(".scanner-preflight-*"))


def test_callers_allow_both_controls_their_whole_limit():
    assert runtime.SCANNER_CONTROL_TIMEOUT_SECONDS >= 240
    assert runtime.SCANNER_CONTROLS_TIMEOUT_SECONDS > 2 * runtime.SCANNER_CONTROL_TIMEOUT_SECONDS


def test_missing_scanner_and_store_alias_are_not_admitted(tmp_path):
    with pytest.raises(CaptureUploadRuntimeError, match="scanner_executable_missing"):
        verify_capture_upload_runtime(store_root=str(tmp_path), scanner_argv=[str(tmp_path / "absent")])
    link = tmp_path / "alias"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(CaptureUploadRuntimeError, match="store_unavailable"):
        verify_capture_upload_runtime(store_root=str(link), scanner_argv=[str(_scanner(tmp_path))])


def test_world_writable_store_is_refused_before_running_a_scanner(tmp_path):
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(CaptureUploadRuntimeError, match="store_permissions_unsafe"):
            verify_capture_upload_runtime(store_root=str(tmp_path), scanner_argv=["/usr/bin/absent"])
    finally:
        tmp_path.chmod(0o700)


def test_unit_defaults_and_later_environment_overrides_are_preserved():
    unit = (ROOT / "deploy/systemd/blueprint-pipeline-intake.service").read_text()
    files = {}

    def read(path):
        if str(path) not in files:
            raise FileNotFoundError()
        return files[str(path)]

    defaults = deploy._capture_transfer_configuration(unit, read_file=read)
    assert defaults == {"store_root": "/var/lib/blueprint/capture-intakes",
                        "scanner_argv": ["/usr/bin/clamscan", "--no-summary"]}
    files["/etc/blueprint/pipeline-control-plane.env"] = (
        "UNRELATED_SECRET='unbalanced\n"
        "PIPELINE_CAPTURE_INTAKE_STORE_ROOT=/var/lib/blueprint/custom-intakes\n"
        "PIPELINE_CAPTURE_MALWARE_SCANNER_ARGV_JSON='[\"/usr/bin/custom-scan\",\"--strict\"]'\n"
    )
    files["/etc/blueprint/task-evaluation-scene-progression.env"] = (
        'PIPELINE_CAPTURE_INTAKE_STORE_ROOT="/var/lib/blueprint/later-intakes"\n'
    )
    actual = deploy._capture_transfer_configuration(unit, read_file=read)
    assert actual == {"store_root": "/var/lib/blueprint/later-intakes",
                      "scanner_argv": ["/usr/bin/custom-scan", "--strict"]}
    assert "UNRELATED_SECRET" not in json.dumps(actual)
    assert deploy._capture_transfer_configuration("[Service]\nExecStart=/bin/true\n") is None


def _ready():
    return {"status": "ready", "provider_mutation_performed": False,
            "source_footage_processed": False, "robot_episode_executed": False,
            "controls": {"clean": {"exit_code": 0}, "known_test_signature": {"exit_code": 1}}}


def test_deploy_probes_as_service_user_with_no_provider_environment(tmp_path, monkeypatch):
    configuration = {"store_root": str(tmp_path), "scanner_argv": ["/usr/bin/clamscan"]}
    monkeypatch.setattr(deploy, "_capture_transfer_configuration", lambda _: configuration)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[:4] == ["/usr/sbin/runuser", "--user", "blueprint", "--"]
        assert json.loads(kwargs["input"]) == configuration
        assert kwargs["env"]["PYTHONPATH"] == str(tmp_path / "src")
        assert not any("KEY" in name or "TOKEN" in name for name in kwargs["env"])
        assert kwargs["timeout"] == runtime.SCANNER_CONTROLS_TIMEOUT_SECONDS
        return SimpleNamespace(returncode=0, stdout=json.dumps(_ready()))

    monkeypatch.setattr(deploy.subprocess, "run", run)
    assert deploy._verify_capture_transfer_runtime(release=tmp_path, unit_text="", account="blueprint")["status"] == "ready"
    assert len(calls) == 1


@pytest.mark.parametrize("value,code", [([], 0), ({"status": "ready"}, 0),
                                        ({"reason": "do not echo a token"}, 2),
                                        ({"reason": "capture_transfer_scanner_executable_missing"}, 2)])
def test_deploy_refuses_incomplete_or_failed_probe_without_raw_output(tmp_path, monkeypatch, value, code):
    monkeypatch.setattr(deploy, "_capture_transfer_configuration", lambda _: {})
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=code, stdout=json.dumps(value)))
    with pytest.raises(deploy.ControlPlaneDeployError) as error:
        deploy._verify_capture_transfer_runtime(release=tmp_path, unit_text="", account="blueprint")
    assert "do not echo a token" not in str(error.value)
    if code == 2 and "capture_transfer_scanner_executable_missing" == value.get("reason"):
        assert str(error.value) == "deploy_capture_transfer_scanner_executable_missing"


def test_source_capture_store_is_protected_evidence():
    root = classify_path("/var/lib/blueprint/capture-intakes/session/raw/source.mp4")
    assert root.storage_class == "evidence_hot"


@pytest.mark.parametrize("refuse", [False, True])
def test_unit_provisioning_requires_scanner_preflight_before_returning_ready(tmp_path, monkeypatch, refuse):
    release = tmp_path / "release"
    unit = release / "deploy/systemd/blueprint-pipeline-intake.service"
    unit.parent.mkdir(parents=True)
    store = tmp_path / "store"
    store.mkdir()
    unit.write_text("[Service]\nReadWritePaths=" + str(store) + "\n")
    calls = []

    def preflight(**kwargs):
        calls.append(kwargs)
        assert store.is_dir()
        if refuse:
            raise deploy.ControlPlaneDeployError("deploy_capture_transfer_scanner_executable_missing")
        return _ready()

    monkeypatch.setattr(deploy, "_verify_capture_transfer_runtime", preflight)
    arguments = {"release_path": release, "units": (unit.name,),
                 "owner_ids": (os.getuid(), os.getgid())}
    if refuse:
        with pytest.raises(deploy.ControlPlaneDeployError, match="scanner_executable_missing"):
            deploy._install_unit_sandbox_paths(**arguments)
    else:
        receipt = deploy._install_unit_sandbox_paths(**arguments)
        assert receipt["capture_upload_runtime"] == _ready()
    assert calls == [{"release": release, "unit_text": unit.read_text(), "account": "blueprint"}]
