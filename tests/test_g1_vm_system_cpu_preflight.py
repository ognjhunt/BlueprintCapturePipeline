"""Local CPU image inspection is never GPU, policy or provider proof."""

import hashlib
import importlib.util
from pathlib import Path
import tarfile
import io

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rehearse_g1_vm_system_cpu.py"
spec = importlib.util.spec_from_file_location("g1_vm_cpu", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("fault", [None, "hash", "size", "link", "extra", "duplicate", "traversal"])
def test_guest_disk_is_extracted_only_after_complete_asset_verification(tmp_path, fault):
    path = tmp_path / "layer.tar.gz"
    data = b"QFI\xfbfixture CPU image"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("../foreign" if fault == "traversal" else "root/images/ubuntu.img")
        info.size = len(data)
        if fault == "link":
            info.type = tarfile.SYMTYPE
            info.linkname = "/foreign"
            info.size = 0
        archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
        if fault == "extra":
            archive.addfile(tarfile.TarInfo("foreign"), io.BytesIO())
        if fault == "duplicate":
            archive.addfile(info, io.BytesIO(data))
    sha = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    target = tmp_path / "base.qcow2"
    if fault:
        with pytest.raises(ValueError, match="g1_vm_cpu"):
            probe.extract_guest_disk(path, target,
                                     expected_sha256="sha256:" + "a" * 64 if fault == "hash" else sha,
                                     expected_layer_bytes=path.stat().st_size,
                                     expected_disk_bytes=len(data) + (1 if fault == "size" else 0))
        assert not target.exists()
        assert not (tmp_path / "foreign").exists()
    else:
        result = probe.extract_guest_disk(path, target, expected_sha256=sha,
                                         expected_layer_bytes=path.stat().st_size, expected_disk_bytes=len(data))
        assert target.read_bytes() == data
        assert result == "sha256:" + hashlib.sha256(data).hexdigest()


def test_capacity_is_reserved_before_public_asset_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 10})())
    with pytest.raises(ValueError, match="capacity"):
        probe.require_capacity(tmp_path, additional_bytes=11)


def test_cpu_command_has_no_network_host_mounts_or_hardware_acceleration(tmp_path):
    argv = probe.qemu_command("qemu-system-x86_64", tmp_path / "overlay.qcow2", tmp_path / "seed.iso")
    assert argv[argv.index("-machine") + 1] == "q35,accel=tcg"
    assert argv[argv.index("-m") + 1] == "2048"
    assert argv[argv.index("-smp") + 1] == "2"
    assert argv[argv.index("-nic") + 1] == "none"
    assert not any(option in argv for option in ("-virtfs", "-fsdev", "-enable-kvm", "-device", "-usbdevice"))
    assert len([option for option in argv if option == "-drive"]) == 2
    seed_drive = argv[argv.index("-drive", argv.index("-drive") + 1) + 1]
    assert ",if=virtio,readonly=on" in seed_drive
    assert "media=cdrom" not in seed_drive


def test_seed_contains_only_fixed_cpu_diagnostics_and_terminal_receipt():
    script = probe.guest_probe_source()
    for forbidden in ("pip install", "apt install", "docker run", "docker pull", "policy_client", "BLUEPRINT_EVAL_MANIFEST_URI"):
        assert forbidden not in script
    assert "gpu_runtime_qualified" in script and "False" in script
    assert "BLUEPRINT_G1_VM_CPU_RESULT:" in script
    compile(script, "guest_cpu_probe.py", "exec")


def test_cpu_result_cannot_be_accepted_without_one_terminal_guest_receipt():
    with pytest.raises(ValueError, match="terminal"):
        probe.read_guest_result("boot output only")
    with pytest.raises(ValueError, match="terminal"):
        probe.read_guest_result('BLUEPRINT_G1_VM_CPU_RESULT:{}\nBLUEPRINT_G1_VM_CPU_RESULT:{}')


def test_optional_packages_only_replay_readonly_offline_apt_simulation():
    source = probe.guest_probe_source(system_packages={
        'implementation_commit': 'a' * 40, 'source_sha256': 'sha256:' + 'b' * 64,
        'manifest_digest': 'sha256:' + 'c' * 64})
    assert 'ro,nodev,nosuid,noexec' in source and '/dev/vdb' in source
    assert 'offline_apt_simulation_command' in source
    assert 'verify_system_packages' in source
    assert 'runtime_installation_performed' in source
    assert 'apt-get install' not in source and 'docker pull' not in source
    compile(source, 'guest_cpu_probe.py', 'exec')


def test_system_replay_rejects_unbound_package_source():
    with pytest.raises(ValueError, match='g1_vm_cpu'):
        probe.guest_probe_source(system_packages={
            'implementation_commit': 'foreign', 'source_sha256': 'sha256:' + 'b' * 64,
            'manifest_digest': 'sha256:' + 'c' * 64})


@pytest.mark.parametrize("fault", [None, "layer", "base"])
def test_replay_reverifies_existing_guest_bytes_without_copy_or_download(tmp_path, fault):
    layer, base = tmp_path / "layer.tar.gz", tmp_path / "base.qcow2"
    data = b"QFI\xfbimmutable guest bytes"
    with tarfile.open(layer, "w:gz") as archive:
        info = tarfile.TarInfo("root/images/ubuntu.img")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    base.write_bytes(data)
    sha = "sha256:" + hashlib.sha256(layer.read_bytes()).hexdigest()
    size = layer.stat().st_size
    if fault:
        (base if fault == "base" else layer).write_bytes(b"changed retained asset")
        with pytest.raises(ValueError, match="g1_vm_cpu"):
            probe.verify_retained_guest_disk(layer, base, expected_sha256=sha,
                                              expected_layer_bytes=size, expected_disk_bytes=len(data))
    else:
        digest = probe.verify_retained_guest_disk(layer, base, expected_sha256=sha,
                                                 expected_layer_bytes=size, expected_disk_bytes=len(data))
        assert digest == "sha256:" + hashlib.sha256(data).hexdigest()
        assert base.read_bytes() == data
        assert sorted(path.name for path in tmp_path.iterdir()) == ["base.qcow2", "layer.tar.gz"]
