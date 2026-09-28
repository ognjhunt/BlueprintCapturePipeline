"""VM transport plumbing, with explicit fake VM/Docker/network observations."""

import hashlib
import json
import shutil
import stat
import subprocess
import zipfile

import pytest

from blueprint_pipeline import native_g1_team_vm_transport as transport
from blueprint_pipeline import vast_provider_adapter as adapter
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


def _zip(path, rows):
    with zipfile.ZipFile(path, "w") as archive:
        for name, value, mode in rows:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, value)


@pytest.mark.parametrize("fault", ["traversal", "link", "fifo", "duplicate"])
def test_transport_refuses_unsafe_zip_before_writing_members(tmp_path, fault):
    path = tmp_path / "input.zip"
    name = "../outside" if fault == "traversal" else "provider_runtime/source.py"
    mode = stat.S_IFLNK if fault == "link" else stat.S_IFIFO if fault == "fifo" else stat.S_IFREG | 0o600
    rows = [(name, b"source", mode)]
    if fault == "duplicate":
        rows.append(rows[0])
        with pytest.warns(UserWarning):
            _zip(path, rows)
    else:
        _zip(path, rows)
    destination = tmp_path / "unpacked"
    with pytest.raises(ValueError, match="zip"):
        transport.extract_bundle(path, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside").exists()


def test_transport_environment_values_are_data_and_only_allowlisted(tmp_path):
    marker = tmp_path / "must-not-exist"
    env = tmp_path / "environment"
    env.write_text('BLUEPRINT_EVAL_MANIFEST_URI="https://files.example.org/b?x=$(touch ' + str(marker) + ')"\n'
                   'UNRELATED_SECRET=fixture-private-value\n')
    parsed = transport.read_environment({}, env)
    assert "$(touch " in parsed["BLUEPRINT_EVAL_MANIFEST_URI"]
    assert "UNRELATED_SECRET" not in parsed
    assert not marker.exists()
    parsed = transport.read_environment({"BLUEPRINT_EVAL_MANIFEST_URI": "https://current.example.org/a"}, env)
    assert parsed["BLUEPRINT_EVAL_MANIFEST_URI"] == "https://current.example.org/a"


def test_evidence_zip_keeps_large_files_and_excludes_only_archive_execution_input(tmp_path):
    root = tmp_path / "runtime_output"
    (root / "policy-host/runtime/artifact").mkdir(parents=True)
    (root / "policy-host/runtime/artifact/weights.bin").write_bytes(b"execution input")
    (root / "policy-host/runtime/child.private.log").write_bytes(b"typed failure detail")
    large = root / "lossless-camera.bin"
    with large.open("wb") as stream:
        stream.truncate(100_000_001)
    archive_path = tmp_path / "output.zip"
    transport.archive_output(root, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.getinfo("lossless-camera.bin").file_size == 100_000_001
        assert archive.read("policy-host/runtime/child.private.log") == b"typed failure detail"
        assert "policy-host/runtime/artifact/weights.bin" not in archive.namelist()


def test_output_symlink_refuses_retention(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    (root / "foreign").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="output"):
        transport.archive_output(root, tmp_path / "output.zip")


@pytest.mark.parametrize("fault", ["container", "pid1", "kind", "uid", "machine"])
def test_transport_observes_vm_identity_before_execution(tmp_path, monkeypatch, fault):
    pid1 = tmp_path / "comm"
    pid1.write_text("bash" if fault == "pid1" else "systemd")
    monkeypatch.setattr(transport.platform, "system", lambda: "Linux")
    monkeypatch.setattr(transport.platform, "machine", lambda: "aarch64" if fault == "machine" else "x86_64")
    monkeypatch.setattr(transport.os, "geteuid", lambda: 1000 if fault == "uid" else 0)
    def probe(argv):
        if "--container" in argv:
            return (0, "docker") if fault == "container" else (1, "none")
        return (0, "xen" if fault == "kind" else "kvm")
    monkeypatch.setattr(transport, "read_probe", probe)
    with pytest.raises(ValueError, match="vm_transport"):
        transport.observe_vm(pid1_path=pid1)


@pytest.mark.parametrize("fault", [None, "docker", "driver", "features"])
def test_transport_checks_system_components_before_image_pull(monkeypatch, fault):
    def probe(argv):
        if argv[0] == "docker":
            return 0, json.dumps({} if fault == "docker" else {"nvidia": {}})
        if argv[0] == "nvidia-smi":
            return 0, "570.00.00" if fault == "driver" else "580.95.05"
        if "--version" in argv:
            return 0, "bubblewrap 0.9.0"
        flags = transport.BWRAP_OPTIONS[:-1] if fault == "features" else transport.BWRAP_OPTIONS
        return 0, "\n".join("--" + flag for flag in flags)
    monkeypatch.setattr(transport, "read_probe", probe)
    if fault:
        with pytest.raises(ValueError, match="vm_transport"):
            transport.observe_system("noncontainer_artifact")
    else:
        value = transport.observe_system("noncontainer_artifact")
        assert value["driver_version"] == "580.95.05"
        assert value["archive_sandbox_required_features"] == list(transport.BWRAP_OPTIONS)


@pytest.mark.parametrize("fault", [None, "bundle_hash", "dependency_hash", "host_exit", "system", "image", "receipt", "timeout"])
def test_transport_preserves_terminal_failure_without_claiming_gpu_or_billing(tmp_path, monkeypatch, fault, capsys):
    dependency = tmp_path / "dependency.zip"
    dependency.write_bytes(b"fixture approved external runtime")
    runtime_sha = "sha256:" + hashlib.sha256(dependency.read_bytes()).hexdigest()
    source = tmp_path / "sealed.zip"
    manifest = {"delivery_mode": "container", "policy_runtime_required": True,
                "runtime_entrypoint": "provider_runtime/run_g1_team_vm_host.sh",
                "container_image": NATIVE_TASK_ARENA_IMAGE,
                "execution_packet_digest": "sha256:" + "a" * 64}
    packet = {"packet_digest": manifest["execution_packet_digest"], "delivery_mode": "container",
              "request": {"policy_profile": {"delivery": {
                  "mode": "container", "image_ref": "registry.example.org/policy@sha256:" + "b" * 64}}}}
    receipt = {"packet_sha256": runtime_sha, "packet_size_bytes": dependency.stat().st_size}
    rows = [("provider_runtime/native_g1_team_provider_manifest.json", json.dumps(manifest).encode(), stat.S_IFREG | 0o600),
            ("provider_runtime/inputs/execution_packet.json", json.dumps(packet).encode(), stat.S_IFREG | 0o600),
            ("provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json",
             json.dumps(receipt).encode(), stat.S_IFREG | 0o600),
            ("provider_runtime/run_g1_team_vm_host.sh", b"fixture-host", stat.S_IFREG | 0o700)]
    _zip(source, rows)
    expected = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []
    monkeypatch.setattr(transport, "observe_vm", lambda: {"vm_kind": "kvm", "pid1": "systemd"})
    def observe(mode):
        if fault == "system":
            raise ValueError("vm_transport_missing_fixture_component")
        return {"fixture": True}
    monkeypatch.setattr(transport, "observe_system", observe)
    def download(url, destination, *, sha256, size_bytes=None, log_path):
        shutil.copyfile(dependency if "dependency" in url else source, destination)
        if fault == ("dependency_hash" if "dependency" in url else "bundle_hash"):
            destination.write_bytes(b"changed bytes")
        transport.verify_file(destination, sha256, size_bytes=size_bytes)
    monkeypatch.setattr(transport, "download", download)
    def command(argv, *, log_path, timeout):
        calls.append(argv)
        if argv[0] == "docker" and fault == "image":
            return 1
        if argv[0] == "bash":
            output = log_path.parent
            if fault == "timeout":
                raise ValueError("vm_transport_child_timeout")
            if fault != "receipt":
                (output / "native_g1_team_vm_host_result.v1.json").write_text('{"fixture": true}')
            return 7 if fault == "host_exit" else 0
        return 0
    monkeypatch.setattr(transport, "run_private", command)
    env = {"BLUEPRINT_EVAL_MANIFEST_URI": "https://files.example.org/bundle",
           "BLUEPRINT_RUNTIME_DEPENDENCY_URI": "https://files.example.org/dependency",
           "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": "https://files.example.org/output"}
    result = transport.run_transport(root=tmp_path / "staging", expected_bundle_sha256=expected,
                                     simulator_image=NATIVE_TASK_ARENA_IMAGE, environment=env)
    assert result["status"] == ("blocked" if fault else "host_exited")
    assert result["gpu_runtime_qualified"] is False
    assert result["provider_teardown_verified"] is False
    assert result["official_billing_reconciled"] is False
    assert result["claim_ceiling"] == "development_only"
    assert "https://" not in json.dumps(result)
    root = tmp_path / "staging"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    with zipfile.ZipFile(root / "adp_arena_provider_runtime_output.zip") as archive:
        retained = json.loads(archive.read(transport.RECEIPT_FILENAME))
        assert retained == result
    markers = capsys.readouterr().out
    assert "BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN:" + str((root / "adp_arena_provider_runtime_output.zip").stat().st_size) in markers
    assert "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK" in markers
    if fault:
        assert "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:" in markers
    if fault in {"bundle_hash", "dependency_hash", "system"}:
        assert not any(row[0] in {"bash", "docker"} for row in calls)
    assert any(row[:3] == ["curl", "--fail", "--silent"] for row in calls)


def test_recovery_publication_preserves_exact_zip_and_refuses_existing_files(tmp_path):
    archive = tmp_path / "retained.zip"
    archive.write_bytes(b"retained exact archive bytes")
    workspace = tmp_path / "workspace"
    transport.publish_recovery_archive(archive, workspace=workspace)
    retained = workspace / "adp_arena_provider_runtime_output.zip"
    assert retained.read_bytes() == archive.read_bytes()
    with pytest.raises(FileExistsError):
        transport.publish_recovery_archive(archive, workspace=workspace)
    assert retained.read_bytes() == archive.read_bytes()


def test_transport_sandbox_features_match_worker_contract():
    from blueprint_pipeline.native_g1_team_artifact_runtime import BWRAP_REQUIRED_OPTIONS
    assert tuple("--" + flag for flag in transport.BWRAP_OPTIONS) == BWRAP_REQUIRED_OPTIONS


@pytest.mark.parametrize("fault", ["unpinned_image", "mode", "entrypoint", "digest"])
def test_selected_inputs_refuse_cross_binding_before_images(tmp_path, fault):
    runtime = tmp_path / "provider_runtime"
    (runtime / "inputs").mkdir(parents=True)
    manifest = {"delivery_mode": "container", "policy_runtime_required": True,
                "runtime_entrypoint": transport.HOST_ENTRYPOINT, "container_image": NATIVE_TASK_ARENA_IMAGE,
                "execution_packet_digest": "sha256:" + "a" * 64}
    delivery = {"mode": "container", "image_ref": "registry.example.org/policy@sha256:" + "b" * 64}
    packet = {"packet_digest": manifest["execution_packet_digest"], "delivery_mode": "container",
              "request": {"policy_profile": {"delivery": delivery}}}
    if fault == "unpinned_image":
        delivery["image_ref"] = "registry.example.org/policy:latest"
    elif fault == "mode":
        delivery["mode"] = "authenticated_endpoint"
    elif fault == "entrypoint":
        manifest["runtime_entrypoint"] = "provider_runtime/foreign.sh"
    else:
        packet["packet_digest"] = "sha256:" + "c" * 64
    (runtime / "native_g1_team_provider_manifest.json").write_text(json.dumps(manifest))
    (runtime / "inputs/execution_packet.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="vm_transport"):
        transport._inputs(tmp_path, NATIVE_TASK_ARENA_IMAGE)


@pytest.mark.parametrize("fault", ["kind", "smoke", "hash"])
def test_probe_constructor_rejects_invalid_vm_contract(fault):
    arguments = {"enable_blueprint_bundle": True, "provider_bundle_kind": "native_g1_team_policy",
                 "expected_provider_bundle_sha256": "sha256:" + "a" * 64, "virtual_machine": True}
    if fault == "kind":
        arguments["provider_bundle_kind"] = "native_task_arena"
    elif fault == "smoke":
        arguments["enable_isaac_smoke"] = True
    else:
        arguments["expected_provider_bundle_sha256"] = None
    with pytest.raises(ValueError, match="virtual_machine|vm_transport"):
        adapter._probe_shell_script("https://heartbeat.example.org", **arguments)


def test_vm_probe_is_standalone_isolated_and_uses_existing_markers(tmp_path):
    script = adapter._probe_shell_script(
        "https://heartbeat.example.org", enable_blueprint_bundle=True,
        provider_bundle_kind="native_g1_team_policy", expected_provider_bundle_sha256="sha256:" + "a" * 64,
        virtual_machine=True)
    assert "python3 -I -B -S" in script
    assert "/isaac-sim/python.sh" not in script
    assert "BLUEPRINT_VAST_PROVIDER_ENTRYPOINT_EXIT_CODE" in script
    assert "BLUEPRINT_VAST_PROVIDER_OUTPUT_UPLOAD_OK" in script
    assert NATIVE_TASK_ARENA_IMAGE in script
    assert script.startswith("#!/usr/bin/env bash\n")
    path = tmp_path / "script.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    # Execute the actual embedded source with stdlib-only isolated Python.
    help_script = script.replace(" <<'BLUEPRINT_G1_VM_TRANSPORT_PY'", " --help <<'BLUEPRINT_G1_VM_TRANSPORT_PY'")
    result = subprocess.run(["bash", "-c", help_script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--expected-bundle-sha256" in result.stdout
