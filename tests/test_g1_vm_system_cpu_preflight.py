"""Local CPU image inspection is never GPU, policy or provider proof."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import io
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rehearse_g1_vm_system_cpu.py"
spec = importlib.util.spec_from_file_location("g1_vm_cpu", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def fixture_capacity(monkeypatch):
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 10**12})())


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


def test_inspection_retains_every_installation_context_predicate_without_mutation():
    source = probe.guest_probe_source()
    assert "'network_interfaces':['ip','-j','link']" in source
    assert "'network_routes':['ip','-j','route']" in source
    assert "'mounts':['cat','/proc/mounts']" in source
    assert "rehearse_offline_installation" not in source


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
    assert "'blocker_code'" in source and "'stage'" in source
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


def _terminal_observation(binding=None):
    value = {"schema_version": "g1_vm_guest_system_cpu_observation.v1",
             "scope": "local_tcg_cpu_only", "gpu_runtime_qualified": False,
             "policy_inference_performed": False, "provider_mutation_performed": False,
             "probes": {"offline_apt_simulation": {"exit_code": 0}}}
    if binding:
        value["system_packages"] = {**binding, "runtime_installation_performed": False}
    value["receipt_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return probe.TERMINAL + json.dumps(value) + "\n"


def test_capacity_refusal_retains_completed_guest_without_promoting_parent(tmp_path, monkeypatch):
    retained = tmp_path / "retained"
    retained.mkdir()
    for name in ("image-manifest.json", "guest-layer.tar.gz", "base.qcow2"):
        (retained / name).write_bytes(b"unchanged CPU fixture")
    binary = tmp_path / "binary"
    binary.write_bytes(b"fixture")
    monkeypatch.setattr(probe.subprocess, "check_output", lambda argv, **kwargs: "a" * 40 if "rev-parse" in argv else "")
    monkeypatch.setattr(probe.shutil, "which", lambda name: str(binary))
    monkeypatch.setattr(probe, "file_sha", lambda path: probe.IMAGE_SHA)
    monkeypatch.setattr(probe, "verify_retained_guest_disk", lambda *args, **kwargs: probe.IMAGE_SHA)
    calls = []

    def capacity(path, **kwargs):
        calls.append(path)
        if len(calls) == 3:
            raise ValueError("g1_vm_cpu_capacity_insufficient")

    monkeypatch.setattr(probe, "require_capacity", capacity)

    def command(argv, log):
        Path(argv[argv.index("-o") + 1] if "makehybrid" in argv else argv[-1]).write_bytes(b"fixture")

    monkeypatch.setattr(probe, "_command", command)

    class Child:
        pid = 1234
        stopped = False

        def poll(self):
            return -15 if self.stopped else None

    child = Child()

    def start(argv, *, stdout, **kwargs):
        stdout.write(_terminal_observation().encode())
        stdout.flush()
        return child

    monkeypatch.setattr(probe.subprocess, "Popen", start)
    monkeypatch.setattr(probe, "_stop", lambda own_child: setattr(own_child, "stopped", True))
    result = probe.run(root=tmp_path / "run", implementation_commit="a" * 40, retained_image_root=retained)
    assert child.stopped and result["child_terminal"] is True
    assert result["status"] == "blocked" and result["blocker_code"] == "g1_vm_cpu_capacity_insufficient"
    assert result["guest_observation"]["probes"]["offline_apt_simulation"]["exit_code"] == 0
    assert result["guest_observation_retention"] == "retained_after_parent_refusal"
    assert result["gpu_runtime_qualified"] is False and result["policy_inference_performed"] is False


@pytest.mark.parametrize("fault", [None, "foreign", "digest", "duplicate", "missing", "installation"])
def test_retained_observation_rejects_foreign_or_invalid_receipt(tmp_path, fault):
    binding = {"implementation_commit": "a" * 40, "source_sha256": "sha256:" + "b" * 64,
               "manifest_digest": "sha256:" + "c" * 64}
    native_binding = {**binding, "implementation_commit": "d" * 40} if fault == "foreign" else binding
    text = _terminal_observation(native_binding)
    if fault == "digest":
        text = text.replace('"exit_code": 0', '"exit_code": 1')
    elif fault == "duplicate":
        text += text
    elif fault == "missing":
        text = "boot only"
    elif fault == "installation":
        value = json.loads(text.split(probe.TERMINAL, 1)[1])
        value.pop("receipt_digest")
        value["system_packages"]["runtime_installation_performed"] = True
        value["receipt_digest"] = "sha256:" + hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        text = probe.TERMINAL + json.dumps(value)
    log = tmp_path / "serial.log"
    log.write_text(text)
    result = {"status": "blocked", "blocker_code": "g1_vm_cpu_capacity_insufficient", "system_packages": binding}
    probe.retain_guest_observation(result, log)
    assert result["status"] == "blocked" and result["blocker_code"] == "g1_vm_cpu_capacity_insufficient"
    if fault:
        assert "guest_observation" not in result
        assert result["guest_observation_retention_blocker"].startswith("g1_vm_cpu_")
    else:
        assert result["guest_observation"]["system_packages"]["manifest_digest"] == binding["manifest_digest"]


def test_install_mode_binds_its_source_and_rejects_missing_inputs(tmp_path):
    with pytest.raises(ValueError, match="install_inputs"):
        probe.run(root=tmp_path / "run", implementation_commit="a" * 40, install_system_packages=True)
    assert not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="install_inputs"):
        probe.guest_probe_source(install_system_packages=True)
    binding = {"implementation_commit": "a" * 40, "source_sha256": "sha256:" + "b" * 64,
               "manifest_digest": "sha256:" + "c" * 64,
               "installation_source_sha256": "sha256:" + "d" * 64}
    source = probe.guest_probe_source(system_packages=binding, install_system_packages=True)
    assert "rehearse_offline_installation" in source and "system-package-installation.py" in source
    assert "installation_source_sha256" in source and "runtime_installation_attempted" in source
    compile(source, "guest_cpu_installation.py", "exec")
    with pytest.raises(ValueError, match="binding"):
        probe.guest_probe_source(system_packages=binding)


def test_install_resource_budget_is_explicit_and_preserves_inspection_budget():
    assert probe.resource_bounds(False) == (64 * 1024**2, 900)
    assert probe.resource_bounds(True) == (3 * 1024**3, 2700)
    with pytest.raises(ValueError, match="install_inputs"):
        probe.resource_bounds("yes")


def test_installation_reserves_seed_and_full_overlay_before_creating_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.subprocess, "check_output", lambda argv, **kwargs: "a" * 40 if "rev-parse" in argv else "")
    module = SimpleNamespace(SYSTEM_PACKAGES={"fixture": {"size_bytes": 10}},
                             verify_system_packages=lambda *args, **kwargs: {"manifest_digest": "sha256:" + "c" * 64})
    monkeypatch.setattr(probe.importlib.util, "spec_from_file_location", lambda *args: SimpleNamespace(
        loader=SimpleNamespace(exec_module=lambda value: None)))
    monkeypatch.setattr(probe.importlib.util, "module_from_spec", lambda spec: module)
    monkeypatch.setattr(probe, "file_sha", lambda path: "sha256:" + "b" * 64)
    reservations = []

    def capacity(path, *, additional_bytes):
        reservations.append(additional_bytes)
        raise ValueError("g1_vm_cpu_capacity_insufficient")

    monkeypatch.setattr(probe, "require_capacity", capacity)
    with pytest.raises(ValueError, match="capacity"):
        probe.run(root=tmp_path / "run", implementation_commit="a" * 40,
                  retained_image_root=tmp_path / "retained", system_package_root=tmp_path / "packages",
                  install_system_packages=True)
    assert reservations == [10 + 262144 + 3 * 1024**3 + probe.LOG_LIMIT]
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("fault", [None, "source", "digest", "gpu", "attempt"])
def test_installation_observation_retention_requires_bound_cpu_only_receipt(tmp_path, fault):
    binding = {"implementation_commit": "a" * 40, "source_sha256": "sha256:" + "b" * 64,
               "manifest_digest": "sha256:" + "c" * 64, "installation_source_sha256": "sha256:" + "d" * 64}
    guest = json.loads(_terminal_observation(binding).split(probe.TERMINAL, 1)[1])
    installation = {"schema_version": "g1_vm_system_cpu_installation.v1", "scope": "local_tcg_cpu_only",
                    "status": "blocked", "runtime_installation_attempted": True,
                    "gpu_runtime_qualified": False, "provider_mutation_performed": False,
                    "policy_inference_performed": False, "claim_ceiling": "development_only"}
    if fault == "gpu":
        installation["gpu_runtime_qualified"] = True
    installation["receipt_digest"] = "sha256:" + hashlib.sha256(json.dumps(
        installation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if fault == "digest":
        installation["receipt_digest"] = "sha256:" + "0" * 64
    guest["system_installation"] = installation
    guest["system_packages"]["runtime_installation_performed"] = fault != "attempt"
    if fault == "source":
        guest["system_packages"]["installation_source_sha256"] = "sha256:" + "e" * 64
    guest.pop("receipt_digest")
    guest["receipt_digest"] = "sha256:" + hashlib.sha256(json.dumps(
        guest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    log = tmp_path / "serial.log"
    log.write_text(probe.TERMINAL + json.dumps(guest))
    result = {"status": "blocked", "blocker_code": "g1_vm_cpu_capacity_insufficient", "system_packages": binding}
    probe.retain_guest_observation(result, log)
    assert result["status"] == "blocked"
    if fault:
        assert "guest_observation" not in result
    else:
        assert result["guest_observation"]["system_installation"]["runtime_installation_attempted"] is True
